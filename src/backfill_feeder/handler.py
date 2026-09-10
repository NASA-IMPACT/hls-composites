"""Scheduled Lambda that submits the next slice of backfill work.

One tick: check queue depth, read the plan, submit the next N units of the
first unfinished segment, and advance that segment's cursor by the number that
actually landed.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Any, Literal

from hls_composites.backfill.state import (
    DEFAULT_PLAN_KEY,
    DEFAULT_TILE_LIST_KEY,
    PlanStore,
)
from hls_composites.backfill.submit import BackfillSubmitter, batch_client
from hls_composites.models import YearMonth

logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)


FeedStatus = Literal["throttled", "complete", "submitted"]
"""Why a tick ended: the queue was full, the plan is finished, or work went out."""


@dataclass(frozen=True)
class FeedResult:
    """The outcome of one feeder tick.

    `year_month` and `submitted_count` describe the segment the tick worked
    on, and are None for the statuses that never reach one.
    """

    status: FeedStatus
    submitted: int
    year_month: YearMonth | None = None
    submitted_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """The JSON-safe form Lambda returns and CloudWatch records.

        Keys are always present so log queries do not have to special-case a
        throttled tick.
        """
        return {
            "status": self.status,
            "submitted": self.submitted,
            "year_month": None if self.year_month is None else str(self.year_month),
            "submitted_count": self.submitted_count,
        }


class TileListMismatchError(RuntimeError):
    """Raised when the tile list no longer matches the plan's `plan_version`.

    Every cursor in the plan is an index into the tile list it was built
    against, so a changed list silently reindexes unfinished segments.
    """


def backfill_feeder(
    *,
    store: PlanStore,
    submitter: BackfillSubmitter,
    tile_list_key: str,
    max_active_jobs: int,
    submit_count: int,
) -> FeedResult:
    """Run one feeder tick."""
    if not submitter.active_jobs_below_threshold(max_active_jobs):
        logger.info("Queue at or above %d active jobs, skipping tick", max_active_jobs)
        return FeedResult(status="throttled", submitted=0)

    stored = store.get()
    plan = stored.plan

    tiles, digest = store.read_tile_list(tile_list_key)
    if digest != plan.plan_version:
        raise TileListMismatchError(
            f"tile list {tile_list_key} is {digest}, plan was built against "
            f"{plan.plan_version}"
        )

    segment = plan.next_segment()
    if segment is None:
        logger.info("Backfill complete, every segment is done")
        return FeedResult(status="complete", submitted=0)

    if segment.tiles_key is not None:
        segment_tiles, _ = store.read_tile_list(segment.tiles_key)
    else:
        segment_tiles = tiles

    if len(segment_tiles) != segment.total_count:
        raise ValueError(
            f"segment {segment.year_month} expects {segment.total_count} tiles, "
            f"its tile list holds {len(segment_tiles)}"
        )

    start = segment.submitted_count
    end = min(start + submit_count, segment.total_count)
    units = [(segment_tiles[index], segment.year_month) for index in range(start, end)]

    submitted = submitter.submit_units(units)
    if submitted == 0:
        logger.warning(
            "Nothing submitted for %s, leaving the plan alone", segment.year_month
        )
        return FeedResult(
            status="submitted",
            submitted=0,
            year_month=segment.year_month,
            submitted_count=segment.submitted_count,
        )

    plan.advance(segment, submitted)
    store.update(replace(stored, plan=plan))

    logger.info(
        "Submitted %d units for %s, cursor now at %d of %d",
        submitted,
        segment.year_month,
        segment.submitted_count,
        segment.total_count,
    )
    return FeedResult(
        status="submitted",
        submitted=submitted,
        year_month=segment.year_month,
        submitted_count=segment.submitted_count,
    )


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entrypoint.

    The EventBridge rule supplies the per-tick batch size:

    ```json
    {"submit_count": 2000}
    ```
    """
    processing_bucket = os.environ["PROCESSING_BUCKET_NAME"]
    store = PlanStore(
        bucket=processing_bucket,
        plan_key=os.environ.get("BACKFILL_PLAN_KEY", DEFAULT_PLAN_KEY),
    )
    submitter = BackfillSubmitter(
        client=batch_client(),
        job_queue=os.environ["BATCH_QUEUE_NAME"],
        job_definition=os.environ["BATCH_JOB_DEFINITION_NAME"],
    )
    result = backfill_feeder(
        store=store,
        submitter=submitter,
        tile_list_key=os.environ.get("BACKFILL_TILE_LIST_KEY", DEFAULT_TILE_LIST_KEY),
        max_active_jobs=int(os.environ["BACKFILL_MAX_ACTIVE_JOBS"]),
        submit_count=int(event["submit_count"]),
    )
    return result.to_dict()
