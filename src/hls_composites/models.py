"""Shared data types for granule discovery and composite creation."""

import os
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
    end: date    # inclusive

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")

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
        """Compute S3 prefixes covering this range's YYYYDDD keys.

        Returns
        -------
        list of str
            A small set of prefix strings guaranteed to cover every date in
            this range when used as an S3 ``list_objects_v2`` prefix. May
            overcover (match some dates outside the range); callers must
            filter results against the range themselves.
        """
        is_full_year = (
            self.start.month == 1
            and self.start.day == 1
            and self.end.month == 12
            and self.end.day == 31
            and self.start.year == self.end.year
        )
        if is_full_year:
            return [f"{self.start.year:04d}"]
        return _range_prefixes(_date_key(self.start), _date_key(self.end))


def _date_key(d: date) -> str:
    return f"{d.year:04d}{d.timetuple().tm_yday:03d}"


def _range_prefixes(lo: str, hi: str) -> list[str]:
    if lo == hi:
        return [lo]
    common = os.path.commonprefix([lo, hi])
    if len(common) == len(lo) - 1:
        return [common]  # only the last digit differs -- already tight
    pos = len(common)
    width = len(lo)
    block = 10 ** (width - pos - 1)
    boundary = (int(lo[: pos + 1]) + 1) * block
    left_hi = str(boundary - 1).zfill(width)
    right_lo = str(boundary).zfill(width)
    return _range_prefixes(lo, left_hi) + _range_prefixes(right_lo, hi)
