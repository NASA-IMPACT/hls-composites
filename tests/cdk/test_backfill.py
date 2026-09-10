import pytest
from aws_cdk import assertions

from tests.cdk.test_stack import build_settings, synth


@pytest.fixture(scope="module")
def template() -> assertions.Template:
    return synth(build_settings())


def feeder_environments(template: assertions.Template) -> dict[str, dict]:
    """Each feeder's environment, keyed by the plan it feeds."""
    return {
        variables["BACKFILL_PLAN_KEY"]: variables
        for function in template.find_resources("AWS::Lambda::Function").values()
        if (variables := function["Properties"].get("Environment", {}).get("Variables"))
        and "BACKFILL_PLAN_KEY" in variables
    }


def schedule_rules(template: assertions.Template) -> list[dict]:
    """The feeders' schedule rules, excluding the monitor's event rules."""
    return [
        rule["Properties"]
        for rule in template.find_resources("AWS::Events::Rule").values()
        if "ScheduleExpression" in rule["Properties"]
    ]


def schedules_by_plan(template: assertions.Template) -> dict[str, dict]:
    """Each schedule rule, keyed by the plan its target feeder works on.

    Both rules carry the same schedule expression, so they are told apart by
    following each target back to the function it invokes.
    """
    plan_of_function = {
        logical_id: function["Properties"]["Environment"]["Variables"][
            "BACKFILL_PLAN_KEY"
        ]
        for logical_id, function in template.find_resources(
            "AWS::Lambda::Function"
        ).items()
        if "BACKFILL_PLAN_KEY"
        in function["Properties"].get("Environment", {}).get("Variables", {})
    }

    schedules = {}
    for rule in schedule_rules(template):
        for target in rule["Targets"]:
            target_id = target["Arn"]["Fn::GetAtt"][0]
            if target_id in plan_of_function:
                schedules[plan_of_function[target_id]] = rule
    return schedules


def test_both_feeders_are_deployed(template):
    assert set(feeder_environments(template)) == {
        "plans/backfill.json",
        "plans/forward.json",
    }


def test_feeders_share_one_tile_list(template):
    """One list, two plans, both pinned to its digest."""
    keys = {
        env["BACKFILL_TILE_LIST_KEY"] for env in feeder_environments(template).values()
    }
    assert keys == {"tiles.txt"}


def test_forward_outranks_backfill_on_queue_depth(template):
    """Ordered ceilings are what stop a saturated backfill starving forward work."""
    envs = feeder_environments(template)
    backfill = int(envs["plans/backfill.json"]["BACKFILL_MAX_ACTIVE_JOBS"])
    forward = int(envs["plans/forward.json"]["BACKFILL_MAX_ACTIVE_JOBS"])

    assert forward > backfill


def test_backfill_is_disabled_but_forward_is_live_by_default(template):
    """A backfill enabled at deploy time would start spending unattended.

    Forward processing is the steady state, so it ships enabled; an idle feeder
    costs nothing until a month is opened.
    """
    schedules = schedules_by_plan(template)

    assert schedules["plans/backfill.json"]["State"] == "DISABLED"
    assert schedules["plans/forward.json"]["State"] == "ENABLED"


def test_feeder_lambdas_are_single_concurrency(template):
    """Concurrent ticks would both read one cursor and one would lose the CAS."""
    functions = [
        function["Properties"]
        for function in template.find_resources("AWS::Lambda::Function").values()
        if "BACKFILL_PLAN_KEY"
        in function["Properties"].get("Environment", {}).get("Variables", {})
    ]

    assert len(functions) == 2
    for function in functions:
        assert function["ReservedConcurrentExecutions"] == 1
        assert function["Timeout"] == 900


def test_each_schedule_passes_its_own_submit_count(template):
    inputs = [
        target.get("Input")
        for rule in template.find_resources("AWS::Events::Rule").values()
        if "ScheduleExpression" in rule["Properties"]
        for target in rule["Properties"]["Targets"]
    ]

    assert len(inputs) == 2
    assert all(payload and "submit_count" in payload for payload in inputs)


def test_feeders_may_list_jobs(template):
    """ListJobs takes no resource-level permissions, so it is granted on `*`."""
    lists = [
        statement
        for policy in template.find_resources("AWS::IAM::Policy").values()
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]
        if "batch:ListJobs" in statement["Action"]
    ]

    assert len(lists) == 2
    assert all(statement["Resource"] == "*" for statement in lists)


def test_schedules_can_be_flipped_independently(template):
    """An operator pausing one feeder must not pause the other."""
    settings = build_settings(SCHEDULE_BACKFILL=True, SCHEDULE_FORWARD=False)
    schedules = schedules_by_plan(synth(settings))

    assert schedules["plans/backfill.json"]["State"] == "ENABLED"
    assert schedules["plans/forward.json"]["State"] == "DISABLED"
