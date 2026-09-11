"""Scheduled Lambdas that feed composite jobs onto the Batch queue.

`FeederFunction` is instantiated once per plan: the historical backfill and
ongoing forward processing run the same code against separate plan objects.
"""

from typing import Any

from aws_cdk import (
    CfnOutput,
    Duration,
    aws_batch as batch,
    aws_events as events,
    aws_events_targets as events_targets,
    aws_iam as iam,
    aws_lambda as lambda_,
    aws_lambda_python_alpha as lambda_python,
    aws_s3 as s3,
)
from constructs import Construct

from hls_constructs.lambda_bundling import LAMBDA_EXCLUDE, export_requirements

BACKFILL_GROUP = "backfill"
"""Dependency group the Lambdas bundle: boto3 and bejm, nothing heavier."""

LAMBDA_ENTRY = "src/"
"""Asset root. Both handlers live under it and share one exported manifest."""


class FeederFunction(Construct):
    """Submits the next slice of a plan's work on a schedule."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        processing_bucket: s3.IBucket,
        job_queue: batch.IJobQueue,
        job_definition: batch.IJobDefinition,
        job_definition_arn: str,
        plan_key: str,
        tile_list_key: str,
        max_active_jobs: int,
        submit_count: int,
        schedule_rate_minutes: int,
        enabled: bool,
        **kwargs: Any,
    ) -> None:
        """Wire a feeder to a plan, a queue, a job definition, and a schedule.

        Parameters
        ----------
        processing_bucket:
            Bucket holding the plan and the tile lists it indexes.
        job_definition_arn:
            Revision-less job definition ARN. The SubmitJob grant covers any
            revision of it.
        max_active_jobs:
            Queue-depth ceiling. Both feeders share one queue and read the same
            depth, so ordering their ceilings is what keeps a saturated backfill
            from starving forward processing.
        enabled:
            Whether the schedule starts enabled. False leaves the feeder
            deployed but idle until someone turns it on.
        """
        super().__init__(scope, construct_id, **kwargs)

        export_requirements(LAMBDA_ENTRY, BACKFILL_GROUP)

        self.function = lambda_python.PythonFunction(
            self,
            "Feeder",
            entry=LAMBDA_ENTRY,
            index="backfill_feeder/handler.py",
            handler="handler",
            runtime=lambda_.Runtime.PYTHON_3_12,
            memory_size=512,
            timeout=Duration.minutes(15),
            reserved_concurrent_executions=1,
            environment={
                "PYTHONUNBUFFERED": "TRUE",
                "PROCESSING_BUCKET_NAME": processing_bucket.bucket_name,
                "BACKFILL_PLAN_KEY": plan_key,
                "BACKFILL_TILE_LIST_KEY": tile_list_key,
                "BACKFILL_MAX_ACTIVE_JOBS": str(max_active_jobs),
                "BATCH_QUEUE_NAME": job_queue.job_queue_name,
                "BATCH_JOB_DEFINITION_NAME": job_definition.job_definition_name,
            },
            bundling=lambda_python.BundlingOptions(
                asset_excludes=LAMBDA_EXCLUDE,
            ),
        )

        processing_bucket.grant_read_write(self.function)

        # SubmitJob authorizes against the resolved revision ARN, so the grant
        # needs the ":*" suffix even though submissions name the revision-less
        # ARN. Without it every submission is denied.
        self.function.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                resources=[job_queue.job_queue_arn, f"{job_definition_arn}:*"],
                actions=["batch:SubmitJob"],
            )
        )
        # ListJobs takes no resource-level permissions.
        self.function.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                resources=["*"],
                actions=["batch:ListJobs"],
            )
        )

        self.schedule = events.Rule(
            self,
            "Schedule",
            schedule=events.Schedule.rate(Duration.minutes(schedule_rate_minutes)),
            enabled=enabled,
        )
        self.schedule.add_target(
            events_targets.LambdaFunction(
                self.function,
                event=events.RuleTargetInput.from_object(
                    {"submit_count": submit_count}
                ),
            )
        )

        CfnOutput(self, "FeederFunctionName", value=self.function.function_name)
        CfnOutput(self, "FeederScheduleName", value=self.schedule.rule_name)


class MonthOpenerFunction(Construct):
    """Queues the month that just ended onto the forward plan, once a month."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        processing_bucket: s3.IBucket,
        forward_plan_key: str,
        tile_list_key: str,
        day_of_month: int,
        enabled: bool,
        **kwargs: Any,
    ) -> None:
        """Wire the opener to the forward plan and a monthly schedule.

        Parameters
        ----------
        day_of_month:
            Day the opener fires, composing the month that just ended. This is
            a lag, not a completeness check: HLS withholds tiles above its
            cloud threshold, so nothing can assert a tile-month is finished.
        enabled:
            Whether the schedule starts enabled. Disable it during a known
            upstream outage, so the affected month is never opened at all
            rather than composited against partial data.
        """
        super().__init__(scope, construct_id, **kwargs)

        export_requirements(LAMBDA_ENTRY, BACKFILL_GROUP)

        self.function = lambda_python.PythonFunction(
            self,
            "Opener",
            entry=LAMBDA_ENTRY,
            index="month_opener/handler.py",
            handler="handler",
            runtime=lambda_.Runtime.PYTHON_3_12,
            memory_size=256,
            timeout=Duration.minutes(1),
            reserved_concurrent_executions=1,
            environment={
                "PYTHONUNBUFFERED": "TRUE",
                "PROCESSING_BUCKET_NAME": processing_bucket.bucket_name,
                "FORWARD_PLAN_KEY": forward_plan_key,
                "FORWARD_TILE_LIST_KEY": tile_list_key,
            },
            bundling=lambda_python.BundlingOptions(
                asset_excludes=LAMBDA_EXCLUDE,
            ),
        )

        # No Batch permissions: the opener only edits a plan, the feeder submits.
        processing_bucket.grant_read_write(self.function)

        self.schedule = events.Rule(
            self,
            "Schedule",
            schedule=events.Schedule.cron(
                minute="0", hour="6", day=str(day_of_month), month="*", year="*"
            ),
            enabled=enabled,
        )
        self.schedule.add_target(events_targets.LambdaFunction(self.function))

        CfnOutput(self, "OpenerFunctionName", value=self.function.function_name)
        CfnOutput(self, "OpenerScheduleName", value=self.schedule.rule_name)
