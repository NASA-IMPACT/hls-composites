"""Monthly Lambda that queues the month that just ended for forward processing.

Appends one segment to the forward plan; the feeder drains it under the same
backpressure as historical work. A month is not submitted in one shot because
18,952 serial SubmitJob calls take about 26 minutes, past Lambda's ceiling.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from dataclasses import dataclass, replace
from typing import Any, Literal

from hls_composites.backfill.plan import BackfillPlan, Segment
from hls_composites.backfill.state import (
    DEFAULT_TILE_LIST_KEY,
    PlanNotFoundError,
    PlanStore,
    TileListMismatchError,
)
from hls_composites.models import YearMonth

logger = logging.getLogger(__name__)
if logger.hasHandlers():
    logger.setLevel(logging.INFO)
else:
    logging.basicConfig(level=logging.INFO)

DEFAULT_FORWARD_PLAN_KEY = "plans/forward.json"

OpenStatus = Literal["opened", "already_open"]
"""Whether this run added the month, or found it already queued."""


@dataclass(frozen=True)
class OpenResult:
    """The outcome of one opener run."""

    status: OpenStatus
    year_month: YearMonth
    total_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """The JSON-safe form Lambda returns and CloudWatch records."""
        return {
            "status": self.status,
            "year_month": str(self.year_month),
            "total_count": self.total_count,
        }


def target_month(today: dt.date) -> YearMonth:
    """The month before `today`.

    Derived from the date rather than configured, so the opener names the right
    month whichever day of the month its schedule fires on. The schedule day is
    a lag knob only.
    """
    return YearMonth.from_date(today).previous()


def open_month(
    *,
    store: PlanStore,
    tile_list_key: str,
    year_month: YearMonth,
) -> OpenResult:
    """Append `year_month` to the forward plan, creating the plan if needed."""
    tiles, digest = store.read_tile_list(tile_list_key)

    try:
        stored = store.get()
    except PlanNotFoundError:
        logger.info("No forward plan yet, creating one at %s", store.plan_key)
        stored = store.create(
            BackfillPlan(
                plan_version=digest,
                tile_list_source=f"s3://{store.bucket}/{tile_list_key}",
                segments=[],
            )
        )

    plan = stored.plan
    if digest != plan.plan_version:
        raise TileListMismatchError(
            f"tile list {tile_list_key} is {digest}, forward plan was built "
            f"against {plan.plan_version}"
        )

    if any(segment.year_month == year_month for segment in plan.segments):
        logger.info("%s is already on the forward plan, nothing to do", year_month)
        return OpenResult(status="already_open", year_month=year_month)

    plan.segments.append(
        Segment(year_month=year_month, submitted_count=0, total_count=len(tiles))
    )
    store.update(replace(stored, plan=plan))

    logger.info("Opened %s with %d units", year_month, len(tiles))
    return OpenResult(status="opened", year_month=year_month, total_count=len(tiles))


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entrypoint. Takes no input; the month comes from the clock."""
    store = PlanStore(
        bucket=os.environ["PROCESSING_BUCKET_NAME"],
        plan_key=os.environ.get("FORWARD_PLAN_KEY", DEFAULT_FORWARD_PLAN_KEY),
    )
    result = open_month(
        store=store,
        tile_list_key=os.environ.get("BACKFILL_TILE_LIST_KEY", DEFAULT_TILE_LIST_KEY),
        year_month=target_month(dt.datetime.now(dt.UTC).date()),
    )
    return result.to_dict()
