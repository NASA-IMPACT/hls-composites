import json

import pytest
from aws_cdk import assertions

from hls_constructs import JOB_TYPE, partition_keys
from tests.cdk.test_stack import build_settings, synth


@pytest.fixture(scope="module")
def template() -> assertions.Template:
    return synth(build_settings())


def resources_of(template: assertions.Template, type_: str) -> list[dict]:
    return list(template.find_resources(type_).values())


def join_suffix(value: dict) -> str:
    """The trailing literal of an `Fn::Join`, e.g. `":*"` in `<arn>:*`."""
    _, parts = value["Fn::Join"]
    return parts[-1]


def join_literals(value: dict) -> str:
    """An `Fn::Join`'s literal text, with its embedded tokens left out.

    The job-type config is JSON built around CloudFormation tokens for the
    queue and job-definition ARNs, so it reaches the template as a join rather
    than a plain string.
    """
    _, parts = value["Fn::Join"]
    return "".join(part for part in parts if isinstance(part, str))


def test_processing_bucket_inventories_cover_state_and_outputs(template):
    (bucket,) = [
        bucket
        for bucket in resources_of(template, "AWS::S3::Bucket")
        if bucket["Properties"].get("BucketName") == "hls-composites-dev"
    ]

    inventories = bucket["Properties"]["InventoryConfigurations"]
    assert {(i["Id"], i["Prefix"]) for i in inventories} == {
        ("state", "state/"),
        ("outputs", "outputs/"),
    }
    assert all(i["ScheduleFrequency"] == "Daily" for i in inventories)
    assert all(i["Destination"]["Format"] == "Parquet" for i in inventories)
    assert all(i["Destination"]["Prefix"] == "inventories" for i in inventories)
    # Reports land in the same bucket they inventory.
    assert all(
        i["Destination"]["BucketArn"] == "arn:aws:s3:::hls-composites-dev"
        for i in inventories
    )


def retry_queue(template: assertions.Template) -> dict:
    """The retry queue, identified by being the only one with redrive."""
    (queue,) = [
        queue["Properties"]
        for queue in resources_of(template, "AWS::SQS::Queue")
        if "RedrivePolicy" in queue["Properties"]
    ]
    return queue


def test_every_failure_path_has_a_queue(template):
    """Retry, retry DLQ, failure DLQ, untracked, and the EventBridge DLQ."""
    queues = resources_of(template, "AWS::SQS::Queue")

    assert len(queues) == 5
    # Encrypted at rest with SQS-managed keys. CDK omits this property unless
    # asked, so it has to be set explicitly rather than left to the default.
    assert all(queue["Properties"]["SqsManagedSseEnabled"] for queue in queues)
    # These queues exist to preserve evidence, so nothing expires early.
    assert all(
        queue["Properties"]["MessageRetentionPeriod"] == 1209600 for queue in queues
    )
    # enforce_ssl renders as a deny-non-TLS queue policy, one per queue.
    assert len(resources_of(template, "AWS::SQS::QueuePolicy")) == 5


def test_retry_queue_redrives_to_its_own_dlq(template):
    assert "RedrivePolicy" in retry_queue(template)


def test_retry_queue_visibility_clears_the_resubmit_lambda_timeout(template):
    """SQS must not redeliver a message the resubmit Lambda is still handling."""
    resubmit_timeouts = [
        function["Properties"]["Timeout"]
        for function in resources_of(template, "AWS::Lambda::Function")
        if "job_resubmit_handler" in function["Properties"].get("Handler", "")
    ]

    assert resubmit_timeouts
    assert retry_queue(template)["VisibilityTimeout"] > max(resubmit_timeouts)


def test_monitor_lambda_knows_the_bucket_and_both_queues(template):
    (monitor,) = [
        function["Properties"]
        for function in resources_of(template, "AWS::Lambda::Function")
        if "job_monitor_handler" in function["Properties"].get("Handler", "")
    ]

    environment = monitor["Environment"]["Variables"]
    # The bucket is created in this stack, so its name arrives as a Ref.
    assert "ProcessingBucket" in environment["PROCESSING_BUCKET_NAME"]["Ref"]
    assert "JOB_RETRY_QUEUE_URL" in environment
    assert "JOB_FAILURE_DLQ_URL" in environment

    configs = join_literals(environment["PROCESSING_JOB_TYPE_CONFIGS"])
    assert f'"{JOB_TYPE}"' in configs
    assert '"max_attempts": 3' in configs
    # The ARNs themselves are tokens, so only their keys survive as literals.
    assert '"job_queue_arn"' in configs
    assert '"job_definition_arn"' in configs


def tracked_rule(template: assertions.Template) -> dict:
    """The rule for jobs carrying the bejm_* parameters."""
    (rule,) = [
        rule["Properties"]
        for rule in resources_of(template, "AWS::Events::Rule")
        if "jobDefinition" in rule["Properties"]["EventPattern"]["detail"]
    ]
    return rule


def test_event_rule_is_scoped_to_our_queue_and_job_definition(template):
    pattern = tracked_rule(template)["EventPattern"]
    assert pattern["source"] == ["aws.batch"]
    assert pattern["detail-type"] == ["Batch Job State Change"]
    # The full lifecycle, not just the terminal states.
    assert pattern["detail"]["status"] == [
        "SUBMITTED",
        "PENDING",
        "RUNNABLE",
        "STARTING",
        "RUNNING",
        "SUCCEEDED",
        "FAILED",
    ]

    (queue,) = pattern["detail"]["jobQueue"]
    assert "JobQueue" in json.dumps(queue)

    # Matching the family ARN by prefix keeps the rule working when a new job
    # definition revision is published.
    (job_definition,) = pattern["detail"]["jobDefinition"]
    assert join_suffix(job_definition["prefix"]) == ":"


def test_untracked_jobs_are_caught_by_their_own_rule(template):
    """A job on our queue without bejm_* params is routed, not dropped."""
    (untracked,) = [
        rule["Properties"]
        for rule in resources_of(template, "AWS::Events::Rule")
        if "jobDefinition" not in rule["Properties"]["EventPattern"]["detail"]
    ]

    parameters = untracked["EventPattern"]["detail"]["parameters"]
    assert parameters == {"bejm_job_type": [{"exists": False}]}

    # The queue is a target in its own right, so the raw event is preserved
    # even if the Lambda never runs.
    queue_targets = [
        target
        for target in untracked["Targets"]
        if "UntrackedQueue" in json.dumps(target["Arn"])
    ]
    assert queue_targets


def test_both_rules_dead_letter_undeliverable_events(template):
    """An event the monitor Lambda cannot be handed is preserved, not lost."""
    rules = resources_of(template, "AWS::Events::Rule")

    assert len(rules) == 2
    for rule in rules:
        lambda_targets = [
            target
            for target in rule["Properties"]["Targets"]
            if "Function" in json.dumps(target["Arn"])
        ]
        assert lambda_targets
        assert all("DeadLetterConfig" in target for target in lambda_targets)


def test_resubmit_lambda_may_submit_only_our_queue_and_job_definition(template):
    submits = [
        statement
        for policy in resources_of(template, "AWS::IAM::Policy")
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]
        if "batch:SubmitJob" in statement["Action"]
    ]

    (submit,) = submits
    queue, job_definition = submit["Resource"]
    assert "JobQueueArn" in json.dumps(queue)
    # Any revision of our job definition family, and nothing else.
    assert join_suffix(job_definition) == ":*"


def test_glue_database_and_tables_exist(template):
    (database,) = resources_of(template, "AWS::Glue::Database")
    assert database["Properties"]["DatabaseInput"]["Name"] == "hls_composites_dev"

    tables = {
        table["Properties"]["TableInput"]["Name"]: table["Properties"]["TableInput"]
        for table in resources_of(template, "AWS::Glue::Table")
    }
    assert set(tables) == {
        "records",
        "state-inventory",
        "state",
        "outputs-inventory",
        "outputs",
    }
    assert tables["state"]["TableType"] == "VIRTUAL_VIEW"
    assert tables["outputs"]["TableType"] == "VIRTUAL_VIEW"


def test_records_table_projects_job_type_and_year_month(template):
    (records,) = [
        table["Properties"]["TableInput"]
        for table in resources_of(template, "AWS::Glue::Table")
        if table["Properties"]["TableInput"]["Name"] == "records"
    ]

    parameters = records["Parameters"]
    assert parameters["projection.enabled"] == "true"
    assert parameters["projection.job_type.values"] == JOB_TYPE
    assert parameters["projection.year_month.type"] == "date"
    assert parameters["projection.year_month.format"] == "yyyy-MM"
    assert parameters["projection.year_month.range"] == "2013-01,NOW"
    assert "projection.tile_id.type" not in parameters
    assert parameters["storage.location.template"] == (
        "s3://hls-composites-dev/records/job_type=${job_type}/year_month=${year_month}/"
    )


def test_inventory_tables_anchor_on_the_configured_datetime(template):
    inventory_tables = [
        table["Properties"]["TableInput"]
        for table in resources_of(template, "AWS::Glue::Table")
        if table["Properties"]["TableInput"]["Name"].endswith("-inventory")
    ]

    assert len(inventory_tables) == 2
    for table in inventory_tables:
        assert table["Parameters"]["projection.dt.range"] == "2026-09-01-01-00,NOW"


def test_partition_key_order_matches_the_key_path():
    """Submitted partition_fields must line up with this order."""
    keys = partition_keys("2013-01")

    assert [key.name for key in keys] == ["job_type", "year_month"]


def test_no_injected_partition_key():
    """An injected key would force an equality predicate onto every query."""
    assert not [
        key for key in partition_keys("2013-01") if key.projection == "injected"
    ]


def test_dev_processing_bucket_is_emptied_and_deleted(template):
    (bucket,) = [
        bucket
        for bucket in resources_of(template, "AWS::S3::Bucket")
        if bucket["Properties"].get("BucketName") == "hls-composites-dev"
    ]

    assert bucket["DeletionPolicy"] == "Delete"
    assert resources_of(template, "Custom::S3AutoDeleteObjects")


def test_prod_processing_bucket_is_retained_and_not_auto_deleted():
    template = synth(
        build_settings(
            STAGE="prod",
            STACK_NAME="hls-composites-prod",
            PROCESSING_BUCKET_NAME="hls-composites-prod",
        )
    )

    (bucket,) = [
        bucket
        for bucket in resources_of(template, "AWS::S3::Bucket")
        if bucket["Properties"].get("BucketName") == "hls-composites-prod"
    ]

    assert bucket["DeletionPolicy"] == "RetainExceptOnCreate"
    assert not resources_of(template, "Custom::S3AutoDeleteObjects")
