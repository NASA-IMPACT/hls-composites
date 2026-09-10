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


def test_each_feeder_schedule_passes_its_own_submit_count(template):
    """The opener's schedule is excluded: it takes no input, the month is the clock."""
    schedules = schedules_by_plan(template)

    assert set(schedules) == {"plans/backfill.json", "plans/forward.json"}
    for rule in schedules.values():
        (target,) = rule["Targets"]
        assert "submit_count" in target["Input"]


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


def opener_function(template: assertions.Template) -> dict:
    """The month opener: it knows the forward plan but not the Batch queue."""
    (function,) = [
        function["Properties"]
        for function in template.find_resources("AWS::Lambda::Function").values()
        if "FORWARD_PLAN_KEY"
        in function["Properties"].get("Environment", {}).get("Variables", {})
    ]
    return function


def test_month_opener_runs_on_a_monthly_cron(template):
    expressions = [rule["ScheduleExpression"] for rule in schedule_rules(template)]
    crons = [e for e in expressions if e.startswith("cron(")]

    assert crons == ["cron(0 6 10 * ? *)"]


def test_month_opener_day_is_configurable(template):
    """The schedule day is the forward-processing lag knob."""
    rules = schedule_rules(synth(build_settings(MONTH_OPENER_DAY=20)))
    crons = [
        r["ScheduleExpression"]
        for r in rules
        if r["ScheduleExpression"].startswith("cron(")
    ]

    assert crons == ["cron(0 6 20 * ? *)"]


def test_month_opener_cannot_submit_jobs(template):
    """It only edits a plan; the feeders submit. No Batch access at all."""
    opener = opener_function(template)
    variables = opener["Environment"]["Variables"]

    assert "BATCH_QUEUE_NAME" not in variables
    assert "BATCH_JOB_DEFINITION_NAME" not in variables


def test_month_opener_writes_the_forward_plan(template):
    variables = opener_function(template)["Environment"]["Variables"]

    assert variables["FORWARD_PLAN_KEY"] == "plans/forward.json"
    assert variables["BACKFILL_TILE_LIST_KEY"] == "tiles.txt"


def test_month_opener_is_single_concurrency(template):
    """A retried invoke must not race itself into two segments for one month."""
    assert opener_function(template)["ReservedConcurrentExecutions"] == 1


def test_month_opener_can_be_disabled_for_an_outage(template):
    """Disabling the opener leaves the month unopened, rather than partial."""
    rules = schedule_rules(synth(build_settings(SCHEDULE_MONTH_OPENER=False)))
    crons = [r for r in rules if r["ScheduleExpression"].startswith("cron(")]

    assert [r["State"] for r in crons] == ["DISABLED"]
