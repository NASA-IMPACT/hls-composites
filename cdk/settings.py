import datetime as dt
from typing import Annotated, Any, Literal

from aws_cdk import aws_logs as logs
from pydantic import BeforeValidator
from pydantic_settings import BaseSettings, NoDecode


def split_comma_separated(value: Any) -> Any:
    """Split a comma-separated string into a list, leaving other values alone."""
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return value


CommaSeparated = Annotated[list[str], NoDecode, BeforeValidator(split_comma_separated)]
"""A list settable from an environment variable as `a,b,c`."""


class StackSettings(BaseSettings):
    """Deployment settings for HLS monthly composite processing.

    Every field is read from an environment variable so that GitHub Actions can
    override any of them per deployment environment.
    """

    STACK_NAME: str
    STAGE: Literal["dev", "prod"]

    MCP_ACCOUNT_ID: str
    MCP_ACCOUNT_REGION: str = "us-west-2"
    MCP_IAM_PERMISSION_BOUNDARY_ARN: str

    VPC_ID: str

    # ----- Buckets
    # Bucket of HLS granules the composites are built from
    INPUT_BUCKET_NAME: str
    # Bucket the monthly composites are written to
    OUTPUT_BUCKET_NAME: str

    # Bucket the job monitor writes records, state pointers, and output index
    # entries to. Created by the batch-event-job-monitor ProcessingBucket
    # construct, which also configures its S3 Inventories.
    PROCESSING_BUCKET_NAME: str

    # ----- LPDAAC access
    # Role from `hls-vi-historical-orchestration` that LPDAAC bucket policies grant
    # read access to. Our job role is allowed to assume it; that only takes effect
    # once the role's own trust policy names our job role.
    LPDAAC_READER_ROLE_ARN: str | None = None

    # ----- Composite processing
    PROCESSING_CONTAINER_ECR_URI: str
    PROCESSING_JOB_VCPU: int = 2
    PROCESSING_JOB_MEMORY_MB: int = 8_000
    PROCESSING_JOB_RETRY_ATTEMPTS: int = 3
    PROCESSING_JOB_TIMEOUT_HOURS: int = 2
    # Custom log group (otherwise logs land in the catch-all AWS Batch log group)
    PROCESSING_LOG_GROUP_NAME: str
    PROCESSING_LOG_RETENTION: logs.RetentionDays = logs.RetentionDays.ONE_MONTH

    # ----- Job monitoring
    # The monitoring queues are created and named by the MonitoringQueues
    # construct; their URLs come out as stack outputs.
    # Attempts a job gets before a retryable failure becomes terminal
    JOB_RETRY_MAX_ATTEMPTS: int = 3

    # ----- Athena / Glue
    ATHENA_DATABASE_NAME: str
    # Anchor for the inventory tables' `dt` partition projection. Its time of day
    # must match the hour S3 delivers inventory reports.
    ATHENA_INVENTORY_START_DATETIME: dt.datetime
    # Start of the `year_month` partition projection range
    YEAR_MONTH_PARTITION_START: str = "2013-01"

    # ----- AWS Batch cluster
    # Reference to the SSM parameter describing the AMI _or_ the AMI ID itself.
    # If using SSM to resolve the AMI ID, prefix with `resolve:ssm:`.
    MCP_AMI_ID: str = "resolve:ssm:/mcp/amis/aml2023-ecs"

    # Cluster instance classes. An empty list lets AWS Batch pick "optimal" classes.
    BATCH_INSTANCE_CLASSES: CommaSeparated = [
        "C5",
        "C5A",
        "C6A",
        "C6I",
        "M5",
        "M5A",
        "M6A",
        "M6I",
    ]

    # Cluster scaling max
    BATCH_MAX_VCPU: int = 16
