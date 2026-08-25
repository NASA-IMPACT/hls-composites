import pytest
from aws_cdk import aws_logs as logs

from hls_constructs import ecr_uri_to_repo_arn, log_retention


def test_ecr_uri_to_repo_arn_private():
    uri = "012345678901.dkr.ecr.us-west-2.amazonaws.com/hls-composites:v1.2.3"
    assert (
        ecr_uri_to_repo_arn(uri)
        == "arn:aws:ecr:us-west-2:012345678901:repository/hls-composites"
    )


def test_ecr_uri_to_repo_arn_namespaced_repo():
    uri = "012345678901.dkr.ecr.us-west-2.amazonaws.com/team/hls-composites:latest"
    assert (
        ecr_uri_to_repo_arn(uri)
        == "arn:aws:ecr:us-west-2:012345678901:repository/team/hls-composites"
    )


def test_ecr_uri_to_repo_arn_public_is_none():
    assert ecr_uri_to_repo_arn("public.ecr.aws/amazonlinux/amazonlinux:latest") is None


def test_log_retention_maps_days():
    assert log_retention(30) is logs.RetentionDays.ONE_MONTH
    assert log_retention(0) is logs.RetentionDays.INFINITE


def test_log_retention_rejects_unsupported():
    with pytest.raises(ValueError, match="unsupported log retention of 31 days"):
        log_retention(31)
