import pytest
from pydantic import ValidationError

from settings import StackSettings

REQUIRED = {
    "STACK_NAME": "hls-composites-dev",
    "STAGE": "dev",
    "MCP_ACCOUNT_ID": "123456789012",
    "MCP_IAM_PERMISSION_BOUNDARY_ARN": (
        "arn:aws:iam::123456789012:policy/mcp-tenantOperator"
    ),
    "VPC_ID": "vpc-12345",
    "INPUT_BUCKET_NAME": "hls-input-bucket",
    "OUTPUT_BUCKET_NAME": "hls-output-bucket",
    "PROCESSING_CONTAINER_ECR_URI": (
        "123456789012.dkr.ecr.us-west-2.amazonaws.com/hls-composites:v0.1.0"
    ),
    "PROCESSING_LOG_GROUP_NAME": "hls-composites-processing-dev",
}


def build(env, monkeypatch):
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    return StackSettings(_env_file=None)


def test_settings_come_from_the_environment(monkeypatch):
    settings = build(REQUIRED, monkeypatch)

    assert settings.STACK_NAME == "hls-composites-dev"
    assert settings.MCP_ACCOUNT_REGION == "us-west-2"
    assert settings.LPDAAC_READER_ROLE_ARN is None


def test_instance_classes_parse_from_a_comma_separated_string(monkeypatch):
    settings = build(
        {**REQUIRED, "BATCH_INSTANCE_CLASSES": "C5, C6I ,M6A"}, monkeypatch
    )

    assert settings.BATCH_INSTANCE_CLASSES == ["C5", "C6I", "M6A"]


def test_instance_classes_may_be_emptied_to_get_optimal(monkeypatch):
    settings = build({**REQUIRED, "BATCH_INSTANCE_CLASSES": ""}, monkeypatch)

    assert settings.BATCH_INSTANCE_CLASSES == []


def test_missing_required_setting_is_an_error(monkeypatch):
    incomplete = {k: v for k, v in REQUIRED.items() if k != "OUTPUT_BUCKET_NAME"}

    with pytest.raises(ValidationError, match="OUTPUT_BUCKET_NAME"):
        build(incomplete, monkeypatch)
