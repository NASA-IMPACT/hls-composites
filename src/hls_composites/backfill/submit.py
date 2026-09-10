"""Submitting backfill units to AWS Batch, under queue-depth backpressure."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import boto3
from batch_event_job_monitor import JobGroup, submit_job
from botocore.config import Config

from hls_composites.models import JOB_TYPE, YearMonth, composite_id

logger = logging.getLogger(__name__)

ACTIVE_JOB_STATUSES = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")
"""Every Batch job status before a job reaches a terminal state."""


def batch_client(region_name: str | None = None) -> Any:
    """A Batch client that backs off rather than failing when throttled."""
    return boto3.client(
        "batch",
        region_name=region_name,
        config=Config(retries={"mode": "adaptive", "max_attempts": 10}),
    )


@dataclass
class BackfillSubmitter:
    """Submits one composite job per unit of work to a single queue."""

    client: Any
    job_queue: str
    job_definition: str

    def active_jobs_below_threshold(self, threshold: int) -> bool:
        """Whether fewer than `threshold` jobs are in a pre-terminal state.

        Stops counting as soon as the threshold is crossed, so a deep queue
        costs no more API calls than a shallow one.
        """
        paginator = self.client.get_paginator("list_jobs")
        count = 0
        for status in ACTIVE_JOB_STATUSES:
            for page in paginator.paginate(jobQueue=self.job_queue, jobStatus=status):
                count += len(page.get("jobSummaryList", []))
                if count >= threshold:
                    return False
        return count < threshold

    def submit_unit(self, tile_id: str, year_month: YearMonth) -> str:
        """Submit one tile-month, tagged for the job monitor. Returns the job id."""
        period = str(year_month)
        job_group = JobGroup.new(
            job_type=JOB_TYPE,
            partition_fields={"year_month": period},
            input_entity_ids=[f"{tile_id}_{period}"],
            output_entity_id=composite_id(tile_id, year_month.date_range()),
        )
        return submit_job(
            batch_client=self.client,
            build_submit_job_params=lambda group: {
                "jobName": group.batch_job_name(),
                "jobQueue": self.job_queue,
                "jobDefinition": self.job_definition,
                "containerOverrides": {
                    "command": [
                        "--tile-id",
                        tile_id,
                        "--year-month",
                        period,
                    ],
                },
            },
            job_group=job_group,
        )

    def submit_units(self, units: Iterable[tuple[str, YearMonth]]) -> int:
        """Submit `(tile_id, year_month)` pairs in order, returning how many landed.

        Stops at the first failure. The caller advances its cursor by the
        returned count, so stopping leaves no unsubmitted hole below it.
        """
        submitted = 0
        for tile_id, year_month in units:
            try:
                self.submit_unit(tile_id, year_month)
            except Exception:
                logger.exception(
                    "SubmitJob failed for %s %s; stopping after %d",
                    tile_id,
                    year_month,
                    submitted,
                )
                break
            submitted += 1
            if submitted % 100 == 0:
                logger.info("Submitted %d units", submitted)
        return submitted
