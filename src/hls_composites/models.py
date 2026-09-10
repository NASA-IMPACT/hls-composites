"""Shared data types for granule discovery and composite creation."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date
from typing import Literal

Satellite = Literal["L30", "S30"]


@dataclass(frozen=True)
class Granule:
    """One HLS granule's S3 location, satellite, and observation date.

    Parameters
    ----------
    path : str
        S3 URI up to and including the granule ID, WITHOUT band suffix, e.g.
        ``s3://bucket/HLSL30.020/HLS.L30.T55HDT.2026151T235621.v2.0/HLS.L30.T55HDT.2026151T235621.v2.0``.
    satellite : {"L30", "S30"}
        Which HLS product the granule belongs to.
    date : datetime.date
        Observation date parsed from the granule ID.
    """

    path: str
    satellite: Satellite
    date: date


@dataclass(frozen=True)
class DateRange:
    """An inclusive start/end date range with S3 key-prefix generation.

    Parameters
    ----------
    start : datetime.date
        First date in the range, inclusive.
    end : datetime.date
        Last date in the range, inclusive.

    Raises
    ------
    ValueError
        If `start` is after `end`.
    """

    start: date  # inclusive
    end: date  # inclusive

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")

    @classmethod
    def for_month(cls, year_month: str) -> DateRange:
        """Build the range covering one calendar month.

        Parameters
        ----------
        year_month : str
            Month as ``YYYY-MM``, e.g. ``"2015-07"``.

        Returns
        -------
        DateRange
            First through last day of that month, inclusive.

        Raises
        ------
        ValueError
            If `year_month` is not in ``YYYY-MM`` form.
        """
        return YearMonth.parse(year_month).date_range()

    def __contains__(self, d: date) -> bool:
        """Check whether a date falls within this range, inclusive.

        Parameters
        ----------
        d : datetime.date
            Date to check.

        Returns
        -------
        bool
            True if `start <= d <= end`.
        """
        return self.start <= d <= self.end

    def key_prefixes(self) -> list[str]:
        """Compute S3 year prefixes covering this range.

        One ``YYYY`` prefix per calendar year touched by the range.
        HLS has at most one observation per day per satellite, so even
        a full year (<=366 keys) fits well within a single
        ``list_objects_v2`` page (max 1000 keys); the boto3 paginator
        transparently fetches further pages if that assumption is ever
        exceeded. Splitting more finely than "one call per year" would
        add round-trips without reducing page count, so this doesn't
        bother.

        Returns
        -------
        list of str
            One ``YYYY`` prefix string per year in `[start.year,
            end.year]`. Overcovers (matches dates outside the range
            within the same year); callers must filter results against
            the range themselves.
        """
        return [f"{year:04d}" for year in range(self.start.year, self.end.year + 1)]


@dataclass(frozen=True, order=True)
class YearMonth:
    """One calendar month, ordered chronologically.

    The canonical in-memory form of a composite's period. `YYYY-MM` strings
    exist only at boundaries -- JSON, S3 keys, Batch parameters, CLI
    arguments -- and are parsed back into this type on the way in.

    Parameters
    ----------
    year : int
        Four-digit year.
    month : int
        Month number, 1-12.

    Raises
    ------
    ValueError
        If `month` is outside 1-12.
    """

    year: int
    month: int

    def __post_init__(self) -> None:
        if not 1 <= self.month <= 12:
            raise ValueError(f"month out of range: {self.month}")

    @classmethod
    def parse(cls, text: str) -> YearMonth:
        """Parse the canonical `YYYY-MM` form.

        Deliberately stricter than `strptime("%Y-%m")`, which accepts an
        unpadded `2020-7`. This value is an Athena partition key and part of
        every entity ID, so a non-canonical spelling would produce a partition
        the date projection cannot match.

        Raises
        ------
        ValueError
            If `text` is not exactly `YYYY-MM`.
        """
        parts = text.split("-")
        if len(parts) != 2 or len(parts[0]) != 4 or len(parts[1]) != 2:
            raise ValueError(f"expected YYYY-MM, got {text!r}")
        try:
            year, month = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(f"expected YYYY-MM, got {text!r}") from None
        return cls(year, month)

    @classmethod
    def from_date(cls, day: date) -> YearMonth:
        """The month containing `day`."""
        return cls(day.year, day.month)

    @classmethod
    def from_ordinal(cls, ordinal: int) -> YearMonth:
        """Invert `ordinal`."""
        return cls(ordinal // 12, ordinal % 12 + 1)

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def ordinal(self) -> int:
        """Months elapsed since year 0, so months can be counted and stepped."""
        return self.year * 12 + self.month - 1

    def next(self) -> YearMonth:
        """The following month."""
        return YearMonth.from_ordinal(self.ordinal + 1)

    def previous(self) -> YearMonth:
        """The preceding month."""
        return YearMonth.from_ordinal(self.ordinal - 1)

    def date_range(self) -> DateRange:
        """The inclusive first-to-last-day range this month covers."""
        first = date(self.year, self.month, 1)
        last_day = calendar.monthrange(self.year, self.month)[1]
        return DateRange(first, first.replace(day=last_day))


JOB_TYPE = "monthly-composite"
"""The one job type this project submits, as recorded by the job monitor."""


def composite_id(tile: str, date_range: DateRange) -> str:
    """Build the monthly composite granule ID for `tile` over `date_range`.

    Parameters
    ----------
    tile : str
        MGRS tile ID, without the leading "T", e.g. `"14TPN"`.
    date_range : DateRange
        The composite's date range; encoded as `%Y%j` day-of-year bounds.

    Returns
    -------
    str
        e.g. `"HLS.M30.T14TPN.2020183.2020213.v2.0"`.
    """
    start = date_range.start.strftime("%Y%j")
    end = date_range.end.strftime("%Y%j")
    return f"HLS.M30.T{tile}.{start}.{end}.v2.0"
