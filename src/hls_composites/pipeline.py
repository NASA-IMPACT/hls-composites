"""The composite pipeline: discover -> composite -> write -> deliver.

Kept free of `click` so the pipeline can be driven from anywhere -- the CLI,
a test, or a future Lambda. Progress is reported through a callback rather
than printed, and the caller chooses where the product lands by passing a
`Destination`.
"""

import datetime as dt
import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import boto3

from hls_composites.aws import (
    assumed_role_env,
    requester_pays_env,
    upload_directory,
)
from hls_composites.browse import write_browse_image
from hls_composites.composite import (
    CompositeOutput,
    build_composite,
    spatial_coverage,
)
from hls_composites.discovery import scan_bucket_for_granules
from hls_composites.io import composite_id, write_rasters
from hls_composites.metadata.manifest import write_manifest
from hls_composites.metadata.models import COMPOSITING_ALGORITHM
from hls_composites.metadata.writer import write_metadata
from hls_composites.models import DateRange

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class LocalDestination:
    """Write the composite to a directory on this machine."""

    directory: Path


@dataclass(frozen=True)
class S3Destination:
    """Upload the composite to S3 beneath `prefix`.

    The composite is built in a temporary directory first, so a failed run
    leaves no partial objects behind.
    """

    bucket: str
    prefix: str = ""


Destination = LocalDestination | S3Destination


@dataclass(frozen=True)
class CompositeResult:
    """What one composite run produced."""

    granule_id: str
    granule_count: int
    uploaded_keys: list[str] = field(default_factory=list)

    @property
    def found_granules(self) -> bool:
        return self.granule_count > 0


def _noop(message: str) -> None:
    """Discard progress messages."""


def create_composite(
    *,
    tile_id: str,
    date_range: DateRange,
    input_bucket: str,
    destination: Destination,
    output: CompositeOutput = "indexes",
    role_arn: str | None = None,
    on_progress: ProgressCallback = _noop,
) -> CompositeResult:
    """Build one tile-month composite and deliver it to `destination`.

    Parameters
    ----------
    tile_id : str
        MGRS tile ID, without the leading "T".
    date_range : DateRange
        Period to composite over.
    input_bucket : str
        Bucket scanned for input granules.
    destination : LocalDestination or S3Destination
        Where the composite is delivered.
    output : {"indexes", "bands"}, optional
        What to composite, by default the spectral indices.
    role_arn : str or None, optional
        Role assumed for reading inputs. None uses ambient credentials.
    on_progress : callable, optional
        Called with human-readable progress messages.

    Returns
    -------
    CompositeResult
        The granule ID, how many input granules were composited, and the
        keys written when delivering to S3.
    """
    with ExitStack() as stack:
        if isinstance(destination, S3Destination):
            work_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        else:
            work_dir = destination.directory

        with requester_pays_env(), assumed_role_env(role_arn) as session:
            on_progress(
                f"Reading via assumed role {role_arn}"
                if role_arn
                else "Reading with ambient credentials"
            )
            granules = scan_bucket_for_granules(
                session.client("s3"), input_bucket, tile_id, date_range
            )

            granule_id = composite_id(tile_id, date_range)
            if not granules:
                # Nothing to composite, so nothing to write or upload. The
                # caller signals this with exit_codes.NO_INPUTS, which the job
                # monitor records as FAILURE_NO_INPUTS.
                on_progress(f"No granules found for {tile_id} in {date_range}")
                return CompositeResult(granule_id, 0)

            on_progress(
                f"Compositing {output} from {len(granules)} granules "
                f"for {tile_id} in {date_range}"
            )
            composite = build_composite(granules, output=output)
            computed = composite.compute()
            dest = Path(
                write_rasters(
                    computed,
                    work_dir,
                    tile_id,
                    date_range,
                    tags=granule_tags(
                        date_range,
                        spatial_coverage(computed["ValidCount"].to_numpy()),
                    ),
                )
            )
            browse = write_browse_image(computed, dest / f"{dest.name}.jpg")

        documents = write_metadata(
            tile_id,
            date_range,
            dest,
            browse,
            inputs=granules,
        )
        on_progress(f"Wrote {len(documents)} metadata documents")

        if isinstance(destination, S3Destination):
            # Last, so it can checksum everything else. Only for S3: its URIs
            # name where the files land, which a local run never reaches.
            prefix = object_prefix(destination.prefix, dest.name)
            write_manifest(dest, f"s3://{destination.bucket}/{prefix}", dest.name)
            on_progress("Wrote the CNM submission message")

            keys = upload_directory(
                boto3.client("s3"), dest, destination.bucket, prefix
            )
            on_progress(f"Uploaded {len(keys)} files to {destination.bucket}")
            return CompositeResult(granule_id, len(granules), keys)

        on_progress(f"Wrote composite to {dest}")
        return CompositeResult(granule_id, len(granules))


def granule_tags(date_range: DateRange, coverage: float) -> dict[str, str]:
    """GeoTIFF tags describing how and when the composite was produced.

    Named as the daily HLS products name their equivalents, so a consumer
    reading both finds the processing time under the same key. `coverage`
    is rounded to whole percent for the same reason.
    """
    return {
        "COMPOSITING_ALGORITHM": COMPOSITING_ALGORITHM,
        "COMPOSITING_START_DATE": date_range.start.isoformat(),
        "COMPOSITING_END_DATE": date_range.end.isoformat(),
        "HLS_PROCESSING_TIME": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spatial_coverage": str(round(coverage)),
    }


def object_prefix(prefix: str, granule_id: str) -> str:
    """Join a configured prefix and a granule ID into an S3 key prefix.

    A blank `prefix` -- empty, whitespace, or bare slashes, all of which an
    unset environment variable can produce -- puts the granule directory at
    the bucket root rather than under a leading slash.
    """
    return "/".join(part for part in (prefix.strip().strip("/"), granule_id) if part)
