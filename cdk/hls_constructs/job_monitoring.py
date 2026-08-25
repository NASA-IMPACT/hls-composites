"""Job monitoring built on the `batch-event-job-monitor` construct library.

The library owns the moving parts -- the processing bucket and its S3
Inventories, the monitor and resubmit Lambdas, and the Athena/Glue databases.
This construct is the wiring: it names the resources, declares this project's
one job type, and picks the partition keys the Athena tables project.
"""

import datetime as dt
from typing import Any

from aws_cdk import (
    CfnOutput,
    RemovalPolicy,
    aws_batch as batch,
    aws_glue as glue,
)
from batch_event_job_monitor.models import RetryPolicy
from batch_event_job_monitor_cdk import (
    AthenaOutputsTable,
    AthenaRecordsTable,
    AthenaStateTable,
    JobMonitorFunction,
    JobResubmitFunction,
    MonitoringQueues,
    PartitionKeySpec,
    ProcessingBucket,
    job_type_config,
)
from constructs import Construct

JOB_TYPE = "monthly-composite"
"""The one job type this project submits."""

INVENTORY_PREFIX = "inventories/"
"""Shared root the bucket's S3 Inventory reports are delivered under."""

STATE_INVENTORY_ID = "state"
OUTPUTS_INVENTORY_ID = "outputs"


def partition_keys(year_month_start: str) -> list[PartitionKeySpec]:
    """The ordered partition keys shared by the records/state/outputs tables.

    MGRS tile IDs are deliberately not a partition key. Injected projection
    could carry them, but Athena then rejects any query lacking an equality
    predicate on the injected key -- which would make reconciliation queries
    spanning tiles ("everything that failed last month") impossible to express.
    Tiles travel in the entity IDs and are extracted at query time instead.

    The order here is the order of the key path, so it must match the order of
    the `partition_fields` mapping that submitted jobs carry.
    """
    return [
        PartitionKeySpec(
            name="job_type",
            glue_type="string",
            projection="enum",
            enum_values=(JOB_TYPE,),
        ),
        PartitionKeySpec(
            name="year_month",
            glue_type="string",
            projection="date",
            date_range=(year_month_start, "NOW"),
            date_format="yyyy-MM",
            date_interval_unit="MONTHS",
        ),
    ]


class JobMonitoring(Construct):
    """Tracking, retry, and reconciliation for this project's Batch jobs."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        job_queue: batch.IJobQueue,
        job_definition: batch.IJobDefinition,
        processing_bucket_name: str,
        retry_max_attempts: int,
        stage: str,
        database_name: str,
        inventory_start_datetime: dt.datetime,
        year_month_start: str,
        **kwargs: Any,
    ) -> None:
        """Wire up job monitoring for a single Batch queue and job definition.

        Parameters
        ----------
        job_queue:
            Queue whose job state changes are monitored.
        job_definition:
            Job definition whose job state changes are monitored. Resubmissions
            go back to this same queue and definition.
        processing_bucket_name:
            Name of the bucket holding records, state pointers, and the output
            index. Created here.
        retry_max_attempts:
            Attempts a job gets before a retryable failure becomes terminal.
        stage:
            Deployment stage. A `dev` processing bucket is emptied and deleted
            with its stack; anything else keeps its processing history.
        database_name:
            Glue database holding the records, state, and outputs tables.
        inventory_start_datetime:
            Anchor for the inventory tables' `dt` partition projection.
        year_month_start:
            Start of the `year_month` partition projection range, as `YYYY-MM`.
        """
        super().__init__(scope, construct_id, **kwargs)

        is_dev = stage == "dev"
        self.processing_bucket = ProcessingBucket(
            self,
            "ProcessingBucket",
            bucket_name=processing_bucket_name,
            inventory_prefix=INVENTORY_PREFIX,
            inventories=[
                (STATE_INVENTORY_ID, "state/"),
                (OUTPUTS_INVENTORY_ID, "outputs/"),
            ],
            removal_policy=(
                RemovalPolicy.DESTROY
                if is_dev
                else RemovalPolicy.RETAIN_ON_UPDATE_OR_DELETE
            ),
            auto_delete_objects=is_dev,
        )

        # Retry, both dead-letter queues, and the untracked-job queue, with
        # redrive wired between the retry queue and its own DLQ.
        self.queues = MonitoringQueues(self, "Queues")

        self.job_type_configs = {
            JOB_TYPE: job_type_config(
                job_queue=job_queue,
                job_definition=job_definition,
                retry_policy=RetryPolicy(max_attempts=retry_max_attempts),
            )
        }

        self.monitor = JobMonitorFunction(
            self,
            "JobMonitor",
            processing_bucket=self.processing_bucket.bucket,
            job_type_configs=self.job_type_configs,
            queues=self.queues,
        )
        self.resubmit = JobResubmitFunction(
            self,
            "JobResubmit",
            job_type_configs=self.job_type_configs,
            retry_queue=self.queues.retry_queue,
        )

        self.database = glue.CfnDatabase(
            self,
            "Database",
            catalog_id=self.processing_bucket.bucket.stack.account,
            database_input=glue.CfnDatabase.DatabaseInputProperty(name=database_name),
        )
        self.database.apply_removal_policy(RemovalPolicy.DESTROY)

        keys = partition_keys(year_month_start)

        self.records_table = AthenaRecordsTable(
            self,
            "RecordsTable",
            database=self.database,
            database_name=database_name,
            records_bucket_name=processing_bucket_name,
            partition_keys=keys,
            table_name="records",
        )
        self.state_table = AthenaStateTable(
            self,
            "StateTable",
            database=self.database,
            database_name=database_name,
            inventory_location_s3path=self.processing_bucket.inventory_location(
                STATE_INVENTORY_ID
            ),
            table_datetime_start=inventory_start_datetime,
            table_name="state-inventory",
            view_name="state",
            partition_keys=keys,
        )
        self.outputs_table = AthenaOutputsTable(
            self,
            "OutputsTable",
            database=self.database,
            database_name=database_name,
            inventory_location_s3path=self.processing_bucket.inventory_location(
                OUTPUTS_INVENTORY_ID
            ),
            table_datetime_start=inventory_start_datetime,
            table_name="outputs-inventory",
            view_name="outputs",
            partition_keys=keys,
        )

        CfnOutput(
            self,
            "ProcessingBucketName",
            value=self.processing_bucket.bucket.bucket_name,
        )
        CfnOutput(self, "RetryQueueUrl", value=self.queues.retry_queue.queue_url)
        CfnOutput(self, "FailureDlqUrl", value=self.queues.failure_dlq.queue_url)
        CfnOutput(
            self, "UntrackedQueueUrl", value=self.queues.untracked_queue.queue_url
        )
        CfnOutput(self, "EventDlqUrl", value=self.queues.event_dlq.queue_url)
        if self.queues.retry_dlq is not None:
            CfnOutput(self, "RetryDlqUrl", value=self.queues.retry_dlq.queue_url)
