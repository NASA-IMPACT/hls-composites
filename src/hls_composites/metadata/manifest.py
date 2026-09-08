"""CNM submission message for a finished granule.

The message tells the DAAC's ingest system what a granule contains: every
file, its size, and a checksum. It is built by `hls_manifest`, the library the
other HLS products use, so a composite's submission looks like theirs.
"""

import json
import os
import uuid
from pathlib import Path

from hls_manifest.hls_manifest import build_manifest

COLLECTION = "HLSM30"
"""Collection the submission names."""

MANIFEST_SUFFIX = ".cnm.json"
"""Suffix chosen so the manifest is not selected into its own file list.

`hls_manifest` picks up `.tif`, `.jpg`, `.xml`, and `_stac.json`; a file cannot
checksum itself.
"""


def job_id() -> str:
    """Identifier for this submission.

    The AWS Batch job ID when running as a Batch job, which ties the
    submission to the job that produced it and to the monitor's records for
    it. A fresh UUID otherwise.
    """
    return os.getenv("AWS_BATCH_JOB_ID") or str(uuid.uuid4())


def write_manifest(
    granule_dir: Path, bucket_uri: str, granule_id: str, identifier: str | None = None
) -> Path:
    """Write the CNM submission message for a granule directory.

    Must run after every other file is in place: the message carries a size
    and a SHA512 checksum for each one.

    Parameters
    ----------
    granule_dir : pathlib.Path
        Directory holding the granule's files.
    bucket_uri : str
        Where those files will live, e.g.
        ``s3://bucket/M30/data/HLS.M30.T14TPN...``. Each file's URI is this
        plus its name.
    granule_id : str
        The granule the submission is for.
    identifier : str, optional
        Submission identifier, by default `job_id()`.

    Returns
    -------
    pathlib.Path
        The written message.
    """
    manifest = build_manifest(
        str(granule_dir),
        bucket_uri,
        COLLECTION,
        granule_id,
        identifier or job_id(),
        False,
    )
    path = granule_dir / f"{granule_id}{MANIFEST_SUFFIX}"
    path.write_text(json.dumps(manifest, indent=2))
    return path
