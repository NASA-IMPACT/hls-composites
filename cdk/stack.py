from typing import Any

from aws_cdk import (
    CfnOutput,
    Stack,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_s3 as s3,
)
from constructs import Construct

from hls_constructs import BatchInfra, BatchJob, JobMonitoring
from settings import StackSettings


class HlsCompositesStack(Stack):
    """AWS Batch infrastructure for HLS monthly composite processing."""

    def __init__(
        self, scope: Construct, stack_id: str, *, settings: StackSettings, **kwargs: Any
    ) -> None:
        super().__init__(scope, stack_id, **kwargs)

        # Apply the IAM permission boundary to the entire stack
        boundary = iam.ManagedPolicy.from_managed_policy_arn(
            self,
            "PermissionBoundary",
            settings.MCP_IAM_PERMISSION_BOUNDARY_ARN,
        )
        iam.PermissionsBoundary.of(self).apply(boundary)

        # ----------------------------------------------------------------------
        # Networking
        # ----------------------------------------------------------------------
        self.vpc = ec2.Vpc.from_lookup(self, "VPC", vpc_id=settings.VPC_ID)

        # ----------------------------------------------------------------------
        # Buckets
        # ----------------------------------------------------------------------
        self.input_bucket = s3.Bucket.from_bucket_name(
            self,
            "InputBucket",
            bucket_name=settings.INPUT_BUCKET_NAME,
        )
        self.output_bucket = s3.Bucket.from_bucket_name(
            self,
            "OutputBucket",
            bucket_name=settings.OUTPUT_BUCKET_NAME,
        )
        # ----------------------------------------------------------------------
        # AWS Batch infrastructure
        # ----------------------------------------------------------------------
        self.batch_infra = BatchInfra(
            self,
            "HLSCompositeCompute",
            vpc=self.vpc,
            instance_classes=settings.BATCH_INSTANCE_CLASSES,
            max_vcpu=settings.BATCH_MAX_VCPU,
            ami_id=settings.MCP_AMI_ID,
            stage=settings.STAGE,
            job_queue_name=f"hls-composites-{settings.STAGE}-job-queue",
        )

        # ----------------------------------------------------------------------
        # Composite processing job
        # ----------------------------------------------------------------------
        environment = {
            "PYTHONUNBUFFERED": "TRUE",
            "HLS_BUCKET": settings.INPUT_BUCKET_NAME,
            "OUTPUT_BUCKET": settings.OUTPUT_BUCKET_NAME,
        }
        if settings.LPDAAC_READER_ROLE_ARN:
            environment["LPDAAC_READER_ROLE_ARN"] = settings.LPDAAC_READER_ROLE_ARN

        self.processing_job = BatchJob(
            self,
            "HLSCompositeProcessing",
            container_ecr_uri=settings.PROCESSING_CONTAINER_ECR_URI,
            vcpu=settings.PROCESSING_JOB_VCPU,
            memory_mb=settings.PROCESSING_JOB_MEMORY_MB,
            retry_attempts=settings.PROCESSING_JOB_RETRY_ATTEMPTS,
            timeout_hours=settings.PROCESSING_JOB_TIMEOUT_HOURS,
            log_group_name=settings.PROCESSING_LOG_GROUP_NAME,
            log_retention=settings.PROCESSING_LOG_RETENTION,
            job_role_name=f"hls-composites-processing-role-{settings.STAGE}",
            job_definition_name=f"hls-composites-{settings.STAGE}-job-definition",
            environment=environment,
        )

        self.input_bucket.grant_read(self.processing_job.role)
        self.output_bucket.grant_read_write(self.processing_job.role)

        if settings.LPDAAC_READER_ROLE_ARN:
            self.processing_job.role.add_to_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    resources=[settings.LPDAAC_READER_ROLE_ARN],
                    actions=["sts:AssumeRole"],
                )
            )

        # ----------------------------------------------------------------------
        # Job monitoring
        # ----------------------------------------------------------------------
        self.monitoring = JobMonitoring(
            self,
            "Monitoring",
            job_queue=self.batch_infra.queue,
            job_definition=self.processing_job.job_def,
            processing_bucket_name=settings.PROCESSING_BUCKET_NAME,
            retry_max_attempts=settings.JOB_RETRY_MAX_ATTEMPTS,
            stage=settings.STAGE,
            database_name=settings.ATHENA_DATABASE_NAME,
            inventory_start_datetime=settings.ATHENA_INVENTORY_START_DATETIME,
            year_month_start=settings.YEAR_MONTH_PARTITION_START,
        )

        CfnOutput(
            self,
            "JobDefinitionArn",
            value=self.processing_job.job_def_arn_without_revision,
        )
        CfnOutput(
            self,
            "JobRoleArn",
            value=self.processing_job.role.role_arn,
        )
