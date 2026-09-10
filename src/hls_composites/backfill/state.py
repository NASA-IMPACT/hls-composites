"""Reading and writing the backfill plan on S3.

Every write is conditional: creation uses `IfNoneMatch`, updates use `IfMatch`
against the ETag the plan was read at. A losing writer raises rather than
overwriting a plan that moved underneath it.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import boto3
from botocore.exceptions import ClientError

from hls_composites.backfill.plan import BackfillPlan, parse_tile_list, tile_list_digest

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

DEFAULT_PLAN_KEY = "plans/backfill.json"
DEFAULT_TILE_LIST_KEY = "tiles/current.txt"

_NOT_FOUND_CODES = frozenset({"NoSuchKey", "404"})
_PRECONDITION_FAILED = "PreconditionFailed"


class PlanNotFoundError(FileNotFoundError):
    """Raised when the object being read does not exist."""


class PlanConflictError(RuntimeError):
    """Raised when a conditional write loses to a concurrent writer."""


class TileListMismatchError(RuntimeError):
    """Raised when a tile list no longer matches a plan's `plan_version`.

    Every cursor in a plan is an index into the tile list it was built against,
    so a changed list silently reindexes unfinished segments.
    """


@dataclass(frozen=True)
class StoredPlan:
    """A plan together with the ETag it was read at."""

    plan: BackfillPlan
    etag: str


def _sanitize_etag(etag: str) -> str:
    return etag.replace('"', "")


@dataclass
class PlanStore:
    """Conditional read/write access to the backfill plan and its tile lists."""

    bucket: str
    plan_key: str = DEFAULT_PLAN_KEY
    client: S3Client = field(default_factory=lambda: boto3.client("s3"))

    def create(self, plan: BackfillPlan) -> StoredPlan:
        """Write a plan, failing if one is already there.

        Raises
        ------
        PlanConflictError
            If a plan already exists at `plan_key`.
        """
        try:
            response = self.client.put_object(
                Bucket=self.bucket,
                Key=self.plan_key,
                Body=plan.to_json().encode(),
                IfNoneMatch="*",
            )
        except ClientError as error:
            if error.response["Error"]["Code"] == _PRECONDITION_FAILED:
                raise PlanConflictError(
                    f"a plan already exists at s3://{self.bucket}/{self.plan_key}"
                ) from error
            raise
        return StoredPlan(plan=plan, etag=_sanitize_etag(response["ETag"]))

    def get(self) -> StoredPlan:
        """Read the plan and the ETag to write it back against.

        Raises
        ------
        PlanNotFoundError
            If no plan exists yet. Plans are created by `scripts/backfill-init`.
        """
        body, etag = self._get_object(self.plan_key)
        return StoredPlan(plan=BackfillPlan.from_json(body.decode()), etag=etag)

    def update(self, stored: StoredPlan) -> StoredPlan:
        """Write `stored.plan` back, failing if the object moved since it was read.

        Raises
        ------
        PlanConflictError
            If the stored ETag no longer matches.
        """
        try:
            response = self.client.put_object(
                Bucket=self.bucket,
                Key=self.plan_key,
                Body=stored.plan.to_json().encode(),
                IfMatch=stored.etag,
            )
        except ClientError as error:
            if error.response["Error"]["Code"] == _PRECONDITION_FAILED:
                raise PlanConflictError(
                    f"s3://{self.bucket}/{self.plan_key} changed since it was read"
                ) from error
            raise
        return replace(stored, etag=_sanitize_etag(response["ETag"]))

    def read_tile_list(self, key: str) -> tuple[list[str], str]:
        """Read a tile list, returning its tiles and its content digest.

        Raises
        ------
        PlanNotFoundError
            If no object exists at `key`.
        """
        body, _ = self._get_object(key)
        return parse_tile_list(body), tile_list_digest(body)

    def write_tile_list(self, key: str, tiles: list[str]) -> str:
        """Write a tile list, returning its content digest."""
        body = ("\n".join(tiles) + "\n").encode()
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body)
        return tile_list_digest(body)

    def _get_object(self, key: str) -> tuple[bytes, str]:
        try:
            response = self.client.get_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if error.response["Error"]["Code"] in _NOT_FOUND_CODES:
                raise PlanNotFoundError(f"s3://{self.bucket}/{key}") from error
            raise
        return response["Body"].read(), _sanitize_etag(response["ETag"])
