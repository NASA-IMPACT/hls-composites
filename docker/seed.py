"""Seed a local MinIO bucket with real HLS granules for one tile-month.

Granules are discovered through NASA's CMR (via earthaccess) and each band
asset is streamed over HTTPS with Earthdata Login credentials into a local
MinIO bucket, under the same keys LP DAAC uses, so the CLI can then composite
them offline against MinIO.

LP DAAC's cloud buckets deny `s3:ListBucket` outright and only serve GETs to
callers in us-west-2, so the bottom-up bucket scan the CLI uses cannot be
pointed at them; CMR is the only way to enumerate granules from outside.

Everything is driven by environment variables (see docker-compose.yml /
.env.example): TILE, YEARMONTH, LOCAL_BUCKET, MINIO_ENDPOINT, MINIO_ROOT_USER
and MINIO_ROOT_PASSWORD. Earthdata Login credentials come from a mounted
~/.netrc or the EARTHDATA_USERNAME / EARTHDATA_PASSWORD variables.
"""

import os
import time
from typing import TYPE_CHECKING

import boto3
import earthaccess
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError
from earthaccess.exceptions import LoginStrategyUnavailable

from hls_composites.bands import DEFAULT_BANDS
from hls_composites.cli import _month_range
from hls_composites.composite import asset_url
from hls_composites.discovery import COLLECTION_DIR, parse_granule_common_prefix
from hls_composites.models import DateRange, Granule, Satellite

if TYPE_CHECKING:
    from earthaccess.results import DataGranule
    from mypy_boto3_s3.client import S3Client

SHORT_NAME: dict[Satellite, str] = {"L30": "HLSL30", "S30": "HLSS30"}
"""CMR collection short name for each HLS product."""

VERSION = "2.0"
"""HLS collection version searched in CMR."""


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def _ensure_bucket(
    client: "S3Client", bucket: str, attempts: int = 30, delay: float = 2.0
) -> None:
    """Create the bucket, waiting for MinIO to accept connections."""
    for attempt in range(attempts):
        try:
            client.create_bucket(Bucket=bucket)
            return
        except EndpointConnectionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
        except ClientError as error:
            code = error.response.get("Error", {}).get("Code", "")
            if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
                return
            raise


def _object_exists(client: "S3Client", bucket: str, key: str) -> bool:
    """Whether an object is already present in the local bucket."""
    try:
        client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError:
        return False


def _login() -> None:
    """Authenticate with Earthdata Login without ever prompting.

    earthaccess's default "all" strategy ends in an interactive prompt, which
    a one-shot container run cannot answer, so only the two non-interactive
    strategies are attempted.
    """
    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy)
        except LoginStrategyUnavailable:
            continue
        if auth.authenticated:
            return
    raise SystemExit(
        "Earthdata Login failed: set EARTHDATA_USERNAME and EARTHDATA_PASSWORD, "
        "or provide a .netrc with a urs.earthdata.nasa.gov entry (its path can "
        "be given by the NETRC environment variable)"
    )


def search_granules(
    satellite: Satellite, tile: str, date_range: DateRange
) -> list["DataGranule"]:
    """Find CMR granules for one satellite, tile and date range.

    The tile is matched against the granule ID, which CMR exposes as a
    wildcard-searchable readable granule name.
    """
    return earthaccess.search_data(
        short_name=SHORT_NAME[satellite],
        version=VERSION,
        granule_name=f"HLS.{satellite}.T{tile}.*",
        temporal=(date_range.start, date_range.end),
    )


def local_granule(
    result: "DataGranule", satellite: Satellite, bucket: str
) -> Granule | None:
    """Map a CMR result onto the `Granule` it will become inside `bucket`.

    Returns None if the granule ID doesn't match the expected HLS pattern.
    """
    granule_id = result["umm"]["GranuleUR"]
    prefix = f"{COLLECTION_DIR[satellite]}/{granule_id}/"
    return parse_granule_common_prefix(prefix, bucket)


def asset_downloads(
    result: "DataGranule", granule: Granule, bucket: str
) -> list[tuple[str, str]]:
    """Pair each wanted band asset's HTTPS download URL with its local key.

    CMR lists every asset of a granule; only the `DEFAULT_BANDS` subset is
    needed to build a composite. Assets are matched by file name so the
    source bucket (public vs. protected) stays CMR's business.
    """
    links = {url.rsplit("/", 1)[-1]: url for url in result.data_links()}
    prefix = f"s3://{bucket}/"

    downloads: list[tuple[str, str]] = []
    for band in DEFAULT_BANDS:
        key = asset_url(granule, band).removeprefix(prefix)
        url = links.get(key.rsplit("/", 1)[-1])
        if url is None:
            print(f"warning: no CMR link for {key}", flush=True)
            continue
        downloads.append((url, key))
    return downloads


def main() -> None:
    tile = _require("TILE")
    year_month = _require("YEARMONTH")
    local_bucket = _require("LOCAL_BUCKET")
    minio_endpoint = _require("MINIO_ENDPOINT")
    minio_key = _require("MINIO_ROOT_USER")
    minio_secret = _require("MINIO_ROOT_PASSWORD")

    _login()
    session = earthaccess.get_requests_https_session()

    # Destination: local MinIO, addressed path-style with explicit credentials.
    dest = boto3.client(
        "s3",
        endpoint_url=minio_endpoint,
        aws_access_key_id=minio_key,
        aws_secret_access_key=minio_secret,
        region_name="us-east-1",
        config=Config(s3={"addressing_style": "path"}),
    )
    _ensure_bucket(dest, local_bucket)

    date_range = _month_range(year_month)
    seeded = 0
    granule_count = 0

    for satellite in ("L30", "S30"):
        for result in search_granules(satellite, tile, date_range):
            granule = local_granule(result, satellite, local_bucket)
            if granule is None or granule.date not in date_range:
                continue
            granule_count += 1

            for url, key in asset_downloads(result, granule, local_bucket):
                if _object_exists(dest, local_bucket, key):
                    print(f"present {key}", flush=True)
                    continue
                with session.get(url, stream=True) as response:
                    response.raise_for_status()
                    response.raw.decode_content = True
                    dest.upload_fileobj(response.raw, local_bucket, key)
                seeded += 1
                print(f"seeded {key}", flush=True)

    print(
        f"done: {granule_count} granules, {seeded} assets -> s3://{local_bucket}",
        flush=True,
    )


if __name__ == "__main__":
    main()
