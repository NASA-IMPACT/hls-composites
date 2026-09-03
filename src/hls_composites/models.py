"""Shared data types for granule discovery and composite creation."""

import calendar
from dataclasses import dataclass
from datetime import date, datetime
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
    def for_month(cls, year_month: str) -> "DateRange":
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
        try:
            first = datetime.strptime(year_month, "%Y-%m").date()
        except ValueError as error:
            raise ValueError(f"expected YYYY-MM, got {year_month!r}") from error
        last_day = calendar.monthrange(first.year, first.month)[1]
        return cls(first, first.replace(day=last_day))

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
