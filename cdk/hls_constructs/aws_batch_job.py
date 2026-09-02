from typing import Any

from aws_cdk import (
    Aws,
    Duration,
    Size,
    aws_batch as batch,
    aws_ecs as ecs,
    aws_iam as iam,
    aws_logs as logs,
)
from constructs import Construct


def ecr_uri_to_repo_arn(uri: str) -> str | None:
    """Convert an ECR container URI into the ARN of its repository.

    Returns `None` when the container URI is not a private ECR URI (e.g. a public
    ECR or Docker Hub image), since there is no repository ARN to scope to.

    Examples
    --------
    >>> ecr_uri_to_repo_arn("012345678901.dkr.ecr.us-west-2.amazonaws.com/my-repo:latest")
    'arn:aws:ecr:us-west-2:012345678901:repository/my-repo'
    >>> ecr_uri_to_repo_arn("public.ecr.aws/amazonlinux/amazonlinux:latest") is None
    True
    """
    if "dkr" not in uri:
        return None

    tagless = uri.split(":")[0]
    dkr, repo = tagless.split("/", 1)
    account_id, _, _, region, _, _ = dkr.split(".")
    return f"arn:aws:ecr:{region}:{account_id}:repository/{repo}"


class BatchJob(Construct):
    """An AWS Batch job running a Docker container."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        container_ecr_uri: str,
        vcpu: int,
        memory_mb: int,
        retry_attempts: int,
        timeout_hours: int,
        log_group_name: str,
        log_retention: logs.RetentionDays,
        job_role_name: str,
        environment: dict[str, str] | None = None,
        secrets: dict[str, batch.Secret] | None = None,
        stage: str,
        **kwargs: Any,
    ) -> None:
        """Set up a CloudWatch log group, IAM roles, and an AWS Batch job definition.

        Parameters
        ----------
        container_ecr_uri:
            Full container image URI, including the tag.
        vcpu:
            vCPUs reserved for each job.
        memory_mb:
            Memory, in MiB, reserved for each job.
        retry_attempts:
            Number of AWS Batch attempts per job.
        timeout_hours:
            Wall-clock limit after which AWS Batch terminates an attempt.
        log_group_name:
            Name of the log group job logs are written to.
        log_retention:
            How long to keep job logs.
        job_role_name:
            Name of the IAM role the container itself runs as. Named explicitly so
            other accounts' trust policies can refer to it.
        environment:
            Environment variables set on the container.
        secrets:
            Secrets exposed to the container as environment variables.
        stage:
            Deployment stage, carried into the JobDefinition's construct id.
        """
        super().__init__(scope, construct_id, **kwargs)

        self.log_group = logs.LogGroup(
            self,
            "JobLogGroup",
            log_group_name=log_group_name,
            retention=log_retention,
        )

        # The execution role is used by the ECS agent, not by our code: it pulls the
        # image and ships the logs.
        # https://docs.aws.amazon.com/AmazonECS/latest/developerguide/task-iam-roles.html#ecr-required-iam-permissions
        self.execution_role = iam.Role(
            self,
            "ExecutionRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonECSTaskExecutionRolePolicy"
                )
            ],
        )
        self.execution_role.add_to_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                resources=["*"],
                actions=[
                    "ecr:GetAuthorizationToken",
                ],
            )
        )
        if ecr_repo_arn := ecr_uri_to_repo_arn(container_ecr_uri):
            self.execution_role.add_to_policy(
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    resources=[
                        ecr_repo_arn,
                    ],
                    actions=[
                        "ecr:BatchGetImage",
                        "ecr:GetDownloadUrlForLayer",
                    ],
                )
            )

        # The job role is what the container's own AWS calls are made with.
        self.role = iam.Role(
            self,
            "JobRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
            role_name=job_role_name,
        )

        self.job_def = batch.EcsJobDefinition(
            self,
            f"JobDef{stage.capitalize()}",
            container=batch.EcsEc2ContainerDefinition(
                self,
                "BatchContainerDef",
                image=ecs.ContainerImage.from_registry(container_ecr_uri),
                execution_role=self.execution_role,
                job_role=self.role,
                cpu=vcpu,
                memory=Size.mebibytes(memory_mb),
                logging=ecs.LogDriver.aws_logs(
                    stream_prefix="job",
                    log_group=self.log_group,
                ),
                secrets=secrets,
                environment=environment or {},
            ),
            timeout=Duration.hours(timeout_hours),
            retry_attempts=retry_attempts,
            retry_strategies=[
                batch.RetryStrategy.of(
                    batch.Action.RETRY, batch.Reason.CANNOT_PULL_CONTAINER
                ),
                batch.RetryStrategy.of(
                    batch.Action.RETRY, batch.Reason.SPOT_INSTANCE_RECLAIMED
                ),
                batch.RetryStrategy.of(
                    batch.Action.EXIT,
                    batch.Reason.custom(on_reason="*"),
                ),
            ],
            propagate_tags=True,
        )

        # Submitting against the revision-less ARN always runs the latest active
        # revision of the job definition.
        self.job_def_arn_without_revision = ":".join(
            [
                "arn",
                "aws",
                "batch",
                Aws.REGION,
                Aws.ACCOUNT_ID,
                f"job-definition/{self.job_def.job_definition_name}",
            ]
        )
