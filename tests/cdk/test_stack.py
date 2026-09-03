import pytest
from aws_cdk import App, assertions

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
        "OUTPUT_BUCKET_NAME": "hls-output-bucket",
        "LPDAAC_READER_ROLE_ARN": LPDAAC_ROLE_ARN,
        "PROCESSING_CONTAINER_ECR_URI": ECR_URI,
        "PROCESSING_LOG_GROUP_NAME": "hls-composites-processing-dev",
        "PROCESSING_BUCKET_NAME": "hls-composites-dev",
        "ATHENA_DATABASE_NAME": "hls_composites_dev",
        "ATHENA_INVENTORY_START_DATETIME": "2026-09-01T01:00:00",
        "BATCH_MAX_VCPU": 32,
        "PROCESSING_JOB_VCPU": 4,
        "PROCESSING_JOB_MEMORY_MB": 16_000,
    }
    values.update(overrides)
    # Ignore any real environment; these settings are the whole input to the stack.
    return StackSettings(_env_file=None, **values)


def synth(settings: StackSettings) -> assertions.Template:
    app = App()
    stack = HlsCompositesStack(
        app,
        settings.STACK_NAME,
        settings=settings,
        env={"account": settings.MCP_ACCOUNT_ID, "region": settings.MCP_ACCOUNT_REGION},
    )
    return assertions.Template.from_stack(stack)


def policy_statements(template: assertions.Template) -> list[dict]:
    """Every statement across every IAM policy in the template."""
    return [
        statement
        for policy in template.find_resources("AWS::IAM::Policy").values()
        for statement in policy["Properties"]["PolicyDocument"]["Statement"]
    ]


def resolve(template: assertions.Template, value) -> str:
    """Render a CloudFormation value as the literal text it stands for.

    Grants render as intrinsics rather than strings: imported buckets as an
    `Fn::Join` over a partition `Ref`, and buckets this stack owns as an
    `Fn::GetAtt` on the bucket's ARN. Both resolve back to plain ARNs so tests
    can assert on something readable.
    """
    if isinstance(value, str):
        return value
    if "Ref" in value:
        return "aws" if value["Ref"] == "AWS::Partition" else value["Ref"]
    if "Fn::Join" in value:
        separator, parts = value["Fn::Join"]
        return separator.join(resolve(template, part) for part in parts)
    if "Fn::GetAtt" in value:
        logical_id, attribute = value["Fn::GetAtt"]
        resource = template.to_json()["Resources"][logical_id]
        if resource["Type"] == "AWS::S3::Bucket" and attribute == "Arn":
            return f"arn:aws:s3:::{resource['Properties']['BucketName']}"
    raise AssertionError(f"cannot resolve {value!r}")


def resource_arns(template: assertions.Template, statement: dict) -> list[str]:
    """The literal ARNs a statement applies to."""
    resources = statement["Resource"]
    if not isinstance(resources, list):
        resources = [resources]
    return [resolve(template, resource) for resource in resources]


@pytest.fixture(scope="module")
def template() -> assertions.Template:
    return synth(build_settings())


def test_compute_environment_is_capped_spot(template):
    template.has_resource_properties(
        "AWS::Batch::ComputeEnvironment",
        {
            "Type": "managed",
            "ComputeResources": assertions.Match.object_like(
                {
                    "Type": "SPOT",
                    "MaxvCpus": 32,
                    "MinvCpus": 0,
                    "AllocationStrategy": "SPOT_CAPACITY_OPTIMIZED",
                }
            ),
        },
    )


def test_job_queue_is_named_for_the_stage(template):
    template.has_resource_properties(
        "AWS::Batch::JobQueue",
        assertions.Match.object_like({"JobQueueName": "hls-composites-dev-job-queue"}),
    )


def test_job_definition_uses_configured_container(template):
    template.has_resource_properties(
        "AWS::Batch::JobDefinition",
        assertions.Match.object_like(
            {
                "Type": "container",
                "Timeout": {"AttemptDurationSeconds": 7200},
                "RetryStrategy": assertions.Match.object_like({"Attempts": 3}),
                "ContainerProperties": assertions.Match.object_like(
                    {
                        "Image": ECR_URI,
                        "ResourceRequirements": assertions.Match.array_with(
                            [
                                {"Type": "MEMORY", "Value": "16000"},
                                {"Type": "VCPU", "Value": "4"},
                            ]
                        ),
                        "Environment": assertions.Match.array_with(
                            [
                                {"Name": "HLS_BUCKET", "Value": "hls-input-bucket"},
                                {"Name": "OUTPUT_BUCKET", "Value": "hls-output-bucket"},
                                {
                                    "Name": "LPDAAC_READER_ROLE_ARN",
                                    "Value": LPDAAC_ROLE_ARN,
                                },
                            ]
                        ),
                    }
                ),
            }
        ),
    )


def test_log_group_is_explicit(template):
    template.has_resource_properties(
        "AWS::Logs::LogGroup",
        {
            "LogGroupName": "hls-composites-processing-dev",
            "RetentionInDays": 30,
        },
    )


def test_job_role_has_a_stable_name(template):
    """Pinned: other accounts name this role in their trust policies."""
    template.has_resource_properties(
        "AWS::IAM::Role",
        assertions.Match.object_like(
            {"RoleName": "hls-composites-processing-role-dev"}
        ),
    )


def test_execution_role_pull_is_scoped_to_the_repository(template):
    repo_arn = f"arn:aws:ecr:{REGION}:{ACCOUNT_ID}:repository/hls-composites"
    pulls = [
        statement
        for statement in policy_statements(template)
        if "ecr:BatchGetImage" in statement["Action"]
    ]

    assert [resource_arns(template, statement) for statement in pulls] == [[repo_arn]]


def test_job_role_may_assume_the_lpdaac_reader_role(template):
    assumes = [
        statement
        for statement in policy_statements(template)
        if statement["Action"] == "sts:AssumeRole"
    ]

    assert [resource_arns(template, statement) for statement in assumes] == [
        [LPDAAC_ROLE_ARN]
    ]


def test_job_role_can_write_the_output_bucket(template):
    writes = [
        statement
        for statement in policy_statements(template)
        if "s3:PutObject" in statement["Action"]
        and any(
            "hls-output-bucket" in arn for arn in resource_arns(template, statement)
        )
    ]

    assert [resource_arns(template, statement) for statement in writes] == [
        ["arn:aws:s3:::hls-output-bucket", "arn:aws:s3:::hls-output-bucket/*"]
    ]


def test_job_role_can_only_read_the_input_bucket(template):
    reads = [
        statement
        for statement in policy_statements(template)
        if statement["Action"] == ["s3:GetObject*", "s3:GetBucket*", "s3:List*"]
    ]

    assert [resource_arns(template, statement) for statement in reads] == [
        ["arn:aws:s3:::hls-input-bucket", "arn:aws:s3:::hls-input-bucket/*"]
    ]


def test_no_assume_role_statement_without_an_lpdaac_role():
    template = synth(build_settings(LPDAAC_READER_ROLE_ARN=None))

    statements = policy_statements(template)
    assert not [s for s in statements if s.get("Action") == "sts:AssumeRole"]

    job_definitions = template.find_resources("AWS::Batch::JobDefinition")
    environments = [
        variable
        for job_definition in job_definitions.values()
        for variable in job_definition["Properties"]["ContainerProperties"][
            "Environment"
        ]
    ]
    assert "LPDAAC_READER_ROLE_ARN" not in {v["Name"] for v in environments}
