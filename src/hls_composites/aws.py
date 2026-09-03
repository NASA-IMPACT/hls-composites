"""Credential scoping and S3 upload for running against LP DAAC and the output bucket.

Reading LP DAAC data needs an IAM role this account can assume; writing composites
needs the job's own role. Both identities live in one process, so the assumed
credentials are scoped to a context manager that covers only the read side.

Credentials are exported as environment variables rather than carried on a boto3
session because the band reads happen on dask worker threads inside GDAL, which
neither sees a boto3 session nor inherits a `rasterio.Env` from the main thread.
Environment variables are read by both, from any thread.
"""

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import boto3

if TYPE_CHECKING:
    from mypy_boto3_s3.client import S3Client

CREDENTIAL_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)
"""The variables GDAL and botocore both read credentials from."""

DEFAULT_SESSION_NAME = "hls-composites"


def _sts_client():  # pragma: no cover - trivial, replaced in tests
    return boto3.client("sts")


@contextmanager
def assumed_role_env(
    role_arn: str | None, session_name: str = DEFAULT_SESSION_NAME
) -> Iterator[str | None]:
    """Run the body with `role_arn`'s credentials in the environment.

    A falsy `role_arn` is a no-op, leaving the ambient credential chain in
    place -- the local path, where the role cannot be assumed. The container
    always sets the variable, so an empty string has to mean "no role" too.

    Parameters
    ----------
    role_arn : str or None
        Role to assume, or None/empty to use ambient credentials.
    session_name : str, optional
        STS session name, by default `DEFAULT_SESSION_NAME`.

    Yields
    ------
    str or None
        The ARN that was assumed, or None when running with ambient
        credentials.

    Raises
    ------
    RuntimeError
        If the role could not be assumed.
    """
    if not role_arn:
        yield None
        return

    try:
        response = _sts_client().assume_role(
            RoleArn=role_arn, RoleSessionName=session_name
        )
    except Exception as error:
        raise RuntimeError(f"could not assume {role_arn}: {error}") from error

    credentials = response["Credentials"]
    new = {
        "AWS_ACCESS_KEY_ID": credentials["AccessKeyId"],
        "AWS_SECRET_ACCESS_KEY": credentials["SecretAccessKey"],
        "AWS_SESSION_TOKEN": credentials["SessionToken"],
    }
    previous = {name: os.environ.get(name) for name in CREDENTIAL_ENV_VARS}

    os.environ.update(new)
    try:
        yield role_arn
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def upload_directory(
    s3_client: "S3Client", local_dir: Path, bucket: str, prefix: str
) -> list[str]:
    """Upload every file under `local_dir` to `bucket` beneath `prefix`.

    Parameters
    ----------
    s3_client : mypy_boto3_s3.client.S3Client
        Client to upload with. Built from the caller's own credentials, so
        the upload runs as the job role rather than any assumed role.
    local_dir : pathlib.Path
        Directory whose files are uploaded, recursively.
    bucket : str
        Destination bucket.
    prefix : str
        Key prefix the directory's contents are placed under.

    Returns
    -------
    list of str
        The keys written, in upload order.
    """
    local_dir = Path(local_dir)
    root = prefix.rstrip("/")
    keys = []
    for path in sorted(p for p in local_dir.rglob("*") if p.is_file()):
        key = f"{root}/{path.relative_to(local_dir).as_posix()}"
        s3_client.upload_file(str(path), bucket, key)
        keys.append(key)
    return keys
