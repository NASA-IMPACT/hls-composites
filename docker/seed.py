"""Seed a local MinIO bucket with real HLS granules for one tile-month.

Reads from a production S3 bucket (real AWS, ambient credentials from the
mounted ~/.aws or AWS_PROFILE) and
copies every band asset of the discovered granules into a local MinIO bucket,
under identical keys, so the CLI can then composite them offline.

Everything is driven by environment variables (see docker-compose.yml /
.env.example): TILE, YEARMONTH, PROD_BUCKET, LOCAL_BUCKET, MINIO_ENDPOINT,
MINIO_ROOT_USER, MINIO_ROOT_PASSWORD, and (optional) PROD_REQUESTER_PAYS.
"""

import os
import time

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError

from hls_composites.bands import DEFAULT_BANDS
from hls_composites.cli import _month_range
from hls_composites.composite import asset_url
from hls_composites.discovery import scan_bucket_for_granules


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"missing required environment variable: {name}")
    return value


def _ensure_bucket(client, bucket: str, attempts: int = 30, delay: float = 2.0) -> None:
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


def main() -> None:
    tile = _require("TILE")
    year_month = _require("YEARMONTH")
    prod_bucket = _require("PROD_BUCKET")
    local_bucket = _require("LOCAL_BUCKET")
    minio_endpoint = _require("MINIO_ENDPOINT")
    minio_key = _require("MINIO_ROOT_USER")
    minio_secret = _require("MINIO_ROOT_PASSWORD")
    requester_pays = os.environ.get("PROD_REQUESTER_PAYS", "false").lower() == "true"

    # Source: production S3 (real AWS) via the mounted ~/.aws / AWS_PROFILE.
    source = boto3.client("s3")
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
    granules = scan_bucket_for_granules(source, prod_bucket, tile, date_range)
    print(f"discovered {len(granules)} granules for {tile} {year_month}", flush=True)

    get_args = {"RequestPayer": "requester"} if requester_pays else {}
    prefix = f"s3://{prod_bucket}/"
    for granule in granules:
        for band in DEFAULT_BANDS:
            key = asset_url(granule, band).removeprefix(prefix)
            body = source.get_object(Bucket=prod_bucket, Key=key, **get_args)["Body"]
            dest.upload_fileobj(body, local_bucket, key)
            print(f"seeded {key}", flush=True)

    print(f"done: {len(granules)} granules -> s3://{local_bucket}", flush=True)


if __name__ == "__main__":
    main()
