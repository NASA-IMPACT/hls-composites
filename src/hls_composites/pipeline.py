"""The composite pipeline: discover -> composite -> write -> deliver.

Kept free of `click` so the pipeline can be driven from anywhere -- the CLI,
a test, or a future Lambda. Progress is reported through a callback rather
than printed, and the caller chooses where the product lands by passing a
`Destination`.
"""

import tempfile
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path

import boto3

from hls_composites.aws import assumed_role_env, upload_directory
from hls_composites.composite import CompositeOutput, build_composite
from hls_composites.discovery import scan_bucket_for_granules
from hls_composites.io import composite_id, write_composite
from hls_composites.models import DateRange

ProgressCallback = Callable[[str], None]

NO_GRANULES_MARKER = "No granules found"
"""Written into an otherwise empty granule directory when nothing was found."""


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

        # Only the reads run as the assumed role; the upload below uses this
        # process's own credentials.
        with assumed_role_env(role_arn) as assumed:
            on_progress(
                f"Reading via assumed role {assumed}"
                if assumed
                else "Reading with ambient credentials"
            )
            granules = scan_bucket_for_granules(
                boto3.client("s3"), input_bucket, tile_id, date_range
            )

            granule_id = composite_id(tile_id, date_range)
            # FIXME: exit with some specific exit code we can parse in job
            #        monitor (could be useful for leading edge)
            if not granules:
                dest = work_dir / granule_id
                dest.mkdir(parents=True, exist_ok=True)
                (dest / NO_GRANULES_MARKER).touch()
                on_progress(f"No granules found for {tile_id} in {date_range}")
            else:
                on_progress(
                    f"Compositing {output} from {len(granules)} granules "
                    f"for {tile_id} in {date_range}"
                )
                composite = build_composite(granules, output=output)
                dest = Path(write_composite(composite, work_dir, tile_id, date_range))

        # FIXME: metadata
        # FIXME: UMM-G

        if isinstance(destination, S3Destination):
            keys = upload_directory(
                boto3.client("s3"),
                dest,
                destination.bucket,
                object_prefix(destination.prefix, dest.name),
            )
            on_progress(f"Uploaded {len(keys)} files to {destination.bucket}")
            return CompositeResult(granule_id, len(granules), keys)

        on_progress(f"Wrote composite to {dest}")
        return CompositeResult(granule_id, len(granules))


def object_prefix(prefix: str, granule_id: str) -> str:
    """Join a configured prefix and a granule ID into an S3 key prefix.

    A blank `prefix` -- empty, whitespace, or bare slashes, all of which an
    unset environment variable can produce -- puts the granule directory at
    the bucket root rather than under a leading slash.
    """
    return "/".join(part for part in (prefix.strip().strip("/"), granule_id) if part)
