"""Bottom-up S3 bucket scanning for HLS granule discovery."""

import random
import re
import time
from datetime import datetime

from botocore.exceptions import ClientError

from hls_composites.models import DateRange, Granule

COLLECTION_DIR: dict[str, str] = {"L30": "HLSL30.020", "S30": "HLSS30.020"}

_GRANULE_ID_PATTERN = re.compile(
    r"HLS\.(?P<sat>L30|S30)\.T(?P<tile>[A-Z0-9]+)\.(?P<datetime>\d{7}T\d{6})\.v2\.0"
)


def parse_granule_common_prefix(common_prefix: str, bucket: str) -> Granule | None:
    """Parse a CommonPrefix string like
    'HLSL30.020/HLS.L30.T55HDT.2026151T235621.v2.0/' into a Granule.
    Returns None if it doesn't match the expected granule-ID pattern.
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


_THROTTLE_ERROR_CODES = {"SlowDown", "RequestLimitExceeded", "InternalError", "ServiceUnavailable"}


def list_common_prefixes_with_retry(
    s3_client,
    bucket: str,
    prefix: str,
    max_retries: int = 5,
    base_delay: float = 1.0,
) -> list[str]:
    """Paginate list_objects_v2 under `prefix` with Delimiter='/', retrying on
    throttling errors with exponential backoff + jitter. Returns the raw
    CommonPrefixes 'Prefix' strings.
    """
    for attempt in range(max_retries):
        try:
            prefixes: list[str] = []
            paginator = s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
                for entry in page.get("CommonPrefixes", []):
                    prefixes.append(entry["Prefix"])
            return prefixes
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code not in _THROTTLE_ERROR_CODES or attempt == max_retries - 1:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, base_delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


def scan_bucket_for_granules(
    s3_client,
    bucket: str,
    tile: str,
    date_range: DateRange,
    satellites: tuple[str, ...] = ("L30", "S30"),
) -> list[Granule]:
    """List the bucket bottom-up to find granules for `tile` within `date_range`."""
    granules: list[Granule] = []
    for sat in satellites:
        collection_dir = COLLECTION_DIR[sat]
        for key_prefix in date_range.key_prefixes():
            list_prefix = f"{collection_dir}/HLS.{sat}.T{tile}.{key_prefix}"
            for common_prefix in list_common_prefixes_with_retry(s3_client, bucket, list_prefix):
                granule = parse_granule_common_prefix(common_prefix, bucket)
                if granule is None:
                    continue
                if granule.date not in date_range:
                    continue
                granules.append(granule)
    return granules
