import json

import pytest
from aws_cdk import App, assertions

from hls_composites.exit_codes import NO_INPUTS
from hls_constructs.job_monitoring import NO_INPUTS_STATE
from settings import StackSettings
from stack import HlsCompositesStack

ACCOUNT_ID = "123456789012"
REGION = "us-west-2"
ECR_URI = f"{ACCOUNT_ID}.dkr.ecr.{REGION}.amazonaws.com/hls-composites:v0.1.0"
LPDAAC_ROLE_ARN = (
    f"arn:aws:iam::{ACCOUNT_ID}:role/hls-vi-historical-processing-role-dev"
)


def build_settings(**overrides) -> StackSettings:
    """Settings with every required field filled in, before `overrides`."""
    values = {
        "STACK_NAME": "hls-composites-dev",
        "STAGE": "dev",
        "MCP_ACCOUNT_ID": ACCOUNT_ID,
        "MCP_ACCOUNT_REGION": REGION,
        "MCP_IAM_PERMISSION_BOUNDARY_ARN": (
            f"arn:aws:iam::{ACCOUNT_ID}:policy/mcp-tenantOperator"
        ),
        "VPC_ID": "vpc-12345",
        "INPUT_BUCKET_NAME": "hls-input-bucket",
        "BACKFILL_OUTPUT_BUCKET_NAME": "hls-output-historical",
        "FORWARD_OUTPUT_BUCKET_NAME": "hls-output-forward",
        "LPDAAC_READER_ROLE_ARN": LPDAAC_ROLE_ARN,
        "PROCESSING_CONTAINER_ECR_URI": ECR_URI,
        "PROCESSING_LOG_GROUP_NAME": "hls-composites-processing-dev",
        "PROCESSING_BUCKET_NAME_PREFIX": "hls-composites-dev",
        "ATHENA_DATABASE_NAME": "hls_composites_dev",
        "ATHENA_INVENTORY_START_DATETIME": "2026-09-01T01:00:00",
    }
    values.update(overrides)
    # Ignore any real environment; these settings are the whole input to the stack.
    return StackSettings(_env_file=None, **values)


def synth(settings: StackSettings) -> assertions.Template:
    # Skip PythonFunction's Docker bundling: these tests assert on the template,
    # not on bundle contents, and bundling every synth would need Docker running.
    app = App(context={"aws:cdk:bundling-stacks": []})
    stack = HlsCompositesStack(
        app,
        settings.STACK_NAME,
        settings=settings,
        env={"account": settings.MCP_ACCOUNT_ID, "region": settings.MCP_ACCOUNT_REGION},
    )
    return assertions.Template.from_stack(stack)


def resources_of(template: assertions.Template, type_: str) -> list[dict]:
    return list(template.find_resources(type_).values())


def render(value) -> str:
    """Render a CloudFormation value as readable text, tokens as their names."""
    if isinstance(value, str):
        return value
    if "Fn::Join" in value:
        separator, parts = value["Fn::Join"]
        return separator.join(render(part) for part in parts)
    if "Ref" in value:
        return value["Ref"]
    return json.dumps(value)


def processing_bucket(template: assertions.Template, prefix: str) -> dict:
    (bucket,) = [
        bucket
        for bucket in resources_of(template, "AWS::S3::Bucket")
        if bucket["Properties"].get("BucketNamePrefix") == prefix
    ]
    return bucket


def monitor_environment(template: assertions.Template) -> dict:
    (monitor,) = [
        function["Properties"]
        for function in resources_of(template, "AWS::Lambda::Function")
        if "job_monitor_handler" in function["Properties"].get("Handler", "")
    ]
    return monitor["Environment"]["Variables"]


def feeder_environment(template: assertions.Template, plan_key: str) -> dict:
    (feeder,) = [
        function["Properties"]["Environment"]["Variables"]
        for function in resources_of(template, "AWS::Lambda::Function")
        if function["Properties"]
        .get("Environment", {})
        .get("Variables", {})
        .get("BACKFILL_PLAN_KEY")
        == plan_key
    ]
    return feeder


@pytest.fixture(scope="module")
def template() -> assertions.Template:
    return synth(build_settings())


def test_every_submit_job_grant_is_scoped_to_our_queue_and_job_definition(template):
    """Held by the resubmit Lambda and by each feeder; all must be scoped."""
    submits = [
        statement
        for policy in resources_of(template, "AWS::IAM::Policy")
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]
        if "batch:SubmitJob" in statement["Action"]
    ]

    assert submits
    for submit in submits:
        queue, *job_definitions = submit["Resource"]
        assert "JobQueueArn" in json.dumps(queue)
        assert job_definitions
        for job_definition in job_definitions:
            # Our job definition family, bare or at any revision, and nothing else.
            text = render(job_definition)
            assert "job-definition/" in text
            assert "ProcessingJobDef" in text
            assert "*" not in text.removesuffix(":*")


def test_dev_processing_bucket_is_emptied_and_deleted(template):
    bucket = processing_bucket(template, "hls-composites-dev")

    assert bucket["DeletionPolicy"] == "Delete"
    assert resources_of(template, "Custom::S3AutoDeleteObjects")


def test_prod_processing_bucket_is_retained_and_not_auto_deleted():
    template = synth(
        build_settings(
            STAGE="prod",
            STACK_NAME="hls-composites-prod",
            PROCESSING_BUCKET_NAME_PREFIX="hls-composites-prod",
        )
    )
    bucket = processing_bucket(template, "hls-composites-prod")

    assert bucket["DeletionPolicy"] == "RetainExceptOnCreate"
    assert not resources_of(template, "Custom::S3AutoDeleteObjects")


def test_no_input_exit_code_is_neither_retried_nor_dead_lettered(template):
    """The container exits NO_INPUTS for an empty tile-month; that is not a fault."""
    _, parts = monitor_environment(template)["PROCESSING_JOB_TYPE_CONFIGS"]["Fn::Join"]
    configs = "".join(part for part in parts if isinstance(part, str))

    assert f'"{NO_INPUTS}"' in configs
    assert NO_INPUTS_STATE in configs
    assert '"dlq": false' in configs
    assert '"retryable": false' in configs


def test_job_monitor_keys_are_written_where_they_are_inventoried(template):
    """Writers and scanners must agree, or the Athena tables go silently empty."""
    bucket = processing_bucket(template, "hls-composites-dev")
    inventories = bucket["Properties"]["InventoryConfigurations"]
    assert {i["Prefix"] for i in inventories} == {
        "logging/state/",
        "logging/outputs/",
        "logging/records/",
    }
    assert {i["Destination"]["Prefix"] for i in inventories} == {"logging/inventories"}

    assert monitor_environment(template)["PROCESSING_KEY_PREFIX"] == "logging/"

    (rollup_rule,) = [
        rule["Properties"]["EventPattern"]
        for rule in resources_of(template, "AWS::Events::Rule")
        if rule["Properties"].get("EventPattern", {}).get("source") == ["aws.s3"]
    ]
    assert rollup_rule["detail"]["object"]["key"] == [{"prefix": "logging/records/"}]


@pytest.mark.parametrize(
    ("plan_key", "bucket"),
    [
        ("plans/backfill.json", "hls-output-historical"),
        ("plans/forward.json", "hls-output-forward"),
    ],
)
def test_each_feeder_targets_its_own_output_bucket(template, plan_key, bucket):
    assert feeder_environment(template, plan_key)["OUTPUT_BUCKET_NAME"] == bucket


def test_job_definition_leaves_the_output_bucket_to_the_submitter(template):
    """A job submitted without a bucket should fail, not land in the wrong one."""
    (job_def,) = resources_of(template, "AWS::Batch::JobDefinition")
    environment = job_def["Properties"]["ContainerProperties"]["Environment"]
    assert "OUTPUT_BUCKET" not in {variable["Name"] for variable in environment}


def test_job_role_can_write_both_output_buckets(template):
    writes = {
        render(resource)
        for policy in resources_of(template, "AWS::IAM::Policy")
        if "ProcessingJobRole" in json.dumps(policy["Properties"]["Roles"])
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]
        if "s3:PutObject" in statement["Action"]
        for resource in statement["Resource"]
    }
    for bucket in ("hls-output-historical", "hls-output-forward"):
        assert any(f":s3:::{bucket}/*" in resource for resource in writes)


def test_athena_workgroup_is_named_per_stage(template):
    """Workgroup names are account-wide, so dev and prod must not share one."""
    prod = synth(
        build_settings(
            STAGE="prod",
            STACK_NAME="hls-composites-prod",
            PROCESSING_BUCKET_NAME_PREFIX="hls-composites-prod",
        )
    )
    (dev_workgroup,) = resources_of(template, "AWS::Athena::WorkGroup")
    (prod_workgroup,) = resources_of(prod, "AWS::Athena::WorkGroup")

    assert dev_workgroup["Properties"]["Name"] != prod_workgroup["Properties"]["Name"]
