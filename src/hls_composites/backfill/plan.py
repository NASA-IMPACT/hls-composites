"""The backfill's units of work and the cursor through them.

A unit of work is one (tile, year-month) pair. Units are a pure function of a
pinned tile list and a month range, so only the cursor is stored: unit `i` of a
segment is `tiles[i]`. Nothing in this module touches AWS.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from hls_composites.models import YearMonth

DEFAULT_MONTH_START = YearMonth(2013, 4)
"""First month with Landsat 8 OLI data, and so the first month worth compositing."""


def months(start: YearMonth, end: YearMonth) -> list[YearMonth]:
    """Every month from `start` through `end`, both inclusive.

    Raises
    ------
    ValueError
        If `start` is after `end`.
    """
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    return [YearMonth.from_ordinal(i) for i in range(start.ordinal, end.ordinal + 1)]


def tile_list_digest(data: bytes) -> str:
    """Content hash of a tile list, as stored in a plan's `plan_version`."""
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def parse_tile_list(data: bytes) -> list[str]:
    """Parse a newline-delimited tile list, ignoring blank lines."""
    return [line.strip() for line in data.decode().splitlines() if line.strip()]


@dataclass
class Segment:
    """One month's worth of units, and how far into them we have submitted.

    `tiles_key` names an S3 object holding a tile subset. When set, the
    segment's units come from that object rather than the plan's tile list;
    this is how gap-fill work re-enters the queue.
    """

    year_month: YearMonth
    submitted_count: int
    total_count: int
    tiles_key: str | None = None

    @property
    def is_complete(self) -> bool:
        return self.submitted_count >= self.total_count

    @property
    def remaining(self) -> int:
        return max(self.total_count - self.submitted_count, 0)


@dataclass
class BackfillPlan:
    """Ordered segments plus the tile list they are indexed against."""

    plan_version: str
    tile_list_source: str
    segments: list[Segment]

    @classmethod
    def new(
        cls,
        *,
        plan_version: str,
        tile_list_source: str,
        year_months: list[YearMonth],
        tile_count: int,
    ) -> BackfillPlan:
        """A fresh plan with one unstarted dense segment per month."""
        return cls(
            plan_version=plan_version,
            tile_list_source=tile_list_source,
            segments=[
                Segment(
                    year_month=year_month, submitted_count=0, total_count=tile_count
                )
                for year_month in year_months
            ],
        )

    def to_json(self) -> str:
        """Serialize, indented so the plan stays hand-editable for rewinds."""
        segments: list[dict[str, object]] = []
        for segment in self.segments:
            entry: dict[str, object] = {
                "year_month": str(segment.year_month),
                "submitted_count": segment.submitted_count,
                "total_count": segment.total_count,
            }
            if segment.tiles_key is not None:
                entry["tiles_key"] = segment.tiles_key
            segments.append(entry)
        return json.dumps(
            {
                "plan_version": self.plan_version,
                "tile_list_source": self.tile_list_source,
                "segments": segments,
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> BackfillPlan:
        data = json.loads(text)
        return cls(
            plan_version=data["plan_version"],
            tile_list_source=data["tile_list_source"],
            segments=[
                Segment(
                    year_month=YearMonth.parse(entry["year_month"]),
                    submitted_count=entry["submitted_count"],
                    total_count=entry["total_count"],
                    tiles_key=entry.get("tiles_key"),
                )
                for entry in data["segments"]
            ],
        )

    @property
    def is_complete(self) -> bool:
        return all(segment.is_complete for segment in self.segments)

    def next_segment(self) -> Segment | None:
        """The first segment with work left, or None when the plan is done."""
        for segment in self.segments:
            if not segment.is_complete:
                return segment
        return None

    def advance(self, segment: Segment, count: int) -> int:
        """Move `segment`'s cursor forward by `count`, returning the new count."""
        if count < 0:
            raise ValueError(f"cannot advance by a negative count: {count}")
        segment.submitted_count = min(
            segment.submitted_count + count, segment.total_count
        )
        return segment.submitted_count
