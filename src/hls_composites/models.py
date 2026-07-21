"""Shared data types for granule discovery and composite creation."""

from dataclasses import dataclass
from datetime import date
from typing import Literal

Satellite = Literal["L30", "S30"]


@dataclass(frozen=True)
class Granule:
    # s3:// URI up to and including the granule ID, WITHOUT band suffix, e.g.
    # s3://bucket/HLSL30.020/HLS.L30.T55HDT.2026151T235621.v2.0/HLS.L30.T55HDT.2026151T235621.v2.0
    path: str
    satellite: Satellite
    date: date


@dataclass(frozen=True)
class DateRange:
    start: date  # inclusive
    end: date    # inclusive

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError(f"start {self.start} is after end {self.end}")

    def __contains__(self, d: date) -> bool:
        return self.start <= d <= self.end

    def key_prefixes(self) -> list[str]:
        """S3 prefix strings covering this range's YYYYDDD keys. Implemented in Task 3."""
        raise NotImplementedError
