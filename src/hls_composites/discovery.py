"""Bottom-up S3 bucket scanning for HLS granule discovery."""

import re
from datetime import datetime
from typing import TYPE_CHECKING

from hls_composites.models import DateRange, Granule

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

COLLECTION_DIR: dict[str, str] = {"L30": "HLSL30.020", "S30": "HLSS30.020"}

_GRANULE_ID_PATTERN = re.compile(
    r"HLS\.(?P<sat>L30|S30)\.T(?P<tile>[A-Z0-9]+)\.(?P<datetime>\d{7}T\d{6})\.v2\.0"
)


def parse_granule_common_prefix(common_prefix: str, bucket: str) -> Granule | None:
    """Parse an S3 CommonPrefix string into a Granule.

    Parameters
    ----------
    common_prefix : str
        A CommonPrefix from a ``list_objects_v2`` response, e.g.
        ``"HLSL30.020/HLS.L30.T55HDT.2026151T235621.v2.0/"``.
    bucket : str
        Name of the S3 bucket the prefix was listed from.

    Returns
    -------
    Granule or None
        The parsed granule, or None if `common_prefix` doesn't match the
        expected granule-ID pattern.
    """
    trimmed = common_prefix.rstrip("/")
    granule_id = trimmed.rsplit("/", 1)[-1]
    match = _GRANULE_ID_PATTERN.fullmatch(granule_id)
    if match is None:
        return None
    granule_date = datetime.strptime(match.group("datetime")[:7], "%Y%j").date()
    satellite = match.group("sat")
    assert satellite in ("L30", "S30")
    return Granule(
        path=f"s3://{bucket}/{trimmed}/{granule_id}",
        satellite=satellite,  # type: ignore[arg-type]
        date=granule_date,
    )


def list_common_prefixes(s3_client: "S3Client", bucket: str, prefix: str) -> list[str]:
    """List S3 CommonPrefixes under a prefix.

    Paginates ``list_objects_v2`` with ``Delimiter="/"``. Retries on
    throttling are expected to be handled by `s3_client`'s own retry
    configuration (e.g. boto3's `Config(retries={"mode": "adaptive"})`),
    not by this function.

    Parameters
    ----------
    s3_client : mypy_boto3_s3.client.S3Client
        An S3 client, e.g. from `boto3.client("s3")`, configured with
        the desired retry policy.
    bucket : str
        Name of the S3 bucket to list.
    prefix : str
        Key prefix to list under.

    Returns
    -------
    list of str
        The raw ``CommonPrefixes[].Prefix`` strings returned by S3.
    """
    prefixes: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for entry in page.get("CommonPrefixes", []):
            prefixes.append(entry["Prefix"])
    return prefixes


def scan_bucket_for_granules(
    s3_client: "S3Client",
    bucket: str,
    tile: str,
    date_range: DateRange,
    satellites: tuple[str, ...] = ("L30", "S30"),
) -> list[Granule]:
    """Find HLS granules for a tile within a date range via bucket scan.

    Lists the bucket bottom-up (no STAC/CMR involved), using
    `DateRange.key_prefixes` to avoid scanning the whole bucket.

    Parameters
    ----------
    s3_client : mypy_boto3_s3.client.S3Client
        An S3 client, e.g. from `boto3.client("s3")`, configured with
        the desired retry policy.
    bucket : str
        Name of the S3 bucket to scan.
    tile : str
        MGRS tile ID, without the leading "T", e.g. `"18SUJ"`.
    date_range : DateRange
        Date range to find granules within.
    satellites : tuple of str, optional
        Which satellites to search, by default `("L30", "S30")`.

    Returns
    -------
    list of Granule
        Granules found for `tile` within `date_range`, across all
        requested `satellites`.
    """
    granules: list[Granule] = []
    for sat in satellites:
        collection_dir = COLLECTION_DIR[sat]
        for key_prefix in date_range.key_prefixes():
            list_prefix = f"{collection_dir}/HLS.{sat}.T{tile}.{key_prefix}"
            for common_prefix in list_common_prefixes(s3_client, bucket, list_prefix):
                granule = parse_granule_common_prefix(common_prefix, bucket)
                if granule is None:
                    continue
                if granule.date not in date_range:
                    continue
                granules.append(granule)
    return granules
