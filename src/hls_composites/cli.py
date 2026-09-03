"""Command-line entrypoint: discover -> composite -> write for one tile-month."""

import calendar
from datetime import datetime
from pathlib import Path

import boto3
import click

from hls_composites.composite import CompositeOutput, build_composite
from hls_composites.discovery import scan_bucket_for_granules
from hls_composites.io import composite_id, write_composite
from hls_composites.models import DateRange


def _month_range(year_month: str) -> DateRange:
    """Parse a ``YYYY-MM`` string into that calendar month's `DateRange`."""
    try:
        first = datetime.strptime(year_month, "%Y-%m").date()
    except ValueError as error:
        raise click.BadParameter(
            f"expected YYYY-MM, got {year_month!r}", param_hint="--year-month"
        ) from error
    last_day = calendar.monthrange(first.year, first.month)[1]
    return DateRange(first, first.replace(day=last_day))


@click.command()
@click.option("--tile-id", "tile_id", required=True, help="MGRS tile ID, e.g. 14TPN.")
@click.option(
    "--year-month",
    "year_month",
    required=True,
    help='Composite month as YYYY-MM, e.g. "2015-07".',
)
@click.option(
    "--bucket",
    envvar="HLS_BUCKET",
    required=True,
    help="S3 bucket to scan for HLS granules (or set HLS_BUCKET).",
)
@click.option(
    "--output-dir",
    "output_dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Local directory to write the composite into.",
)
@click.option(
    "--indexes",
    "indexes",
    is_flag=True,
    help="Composite the spectral indices. The default.",
)
@click.option(
    "--bands",
    "bands",
    is_flag=True,
    help="Composite the raw reflectance bands instead of the spectral indices.",
)
def main(
    tile_id: str,
    year_month: str,
    bucket: str,
    output_dir: Path,
    indexes: bool,
    bands: bool,
) -> None:
    """Build the monthly HLS composite for one tile and write it locally."""
    if indexes and bands:
        raise click.UsageError("--indexes and --bands are mutually exclusive")
    output: CompositeOutput = "bands" if bands else "indexes"

    date_range = _month_range(year_month)
    s3_client = boto3.client("s3")
    granules = scan_bucket_for_granules(s3_client, bucket, tile_id, date_range)

    # FIXME: exit with some specific exit code we can parse in job monitor
    #        (could be useful for leading edge)
    if not granules:
        dest = output_dir / composite_id(tile_id, date_range)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "No granules found").touch()
        click.echo(f"No granules found for {tile_id} {year_month}; wrote {dest}")
        return

    click.echo(
        f"Compositing {output} from {len(granules)} granules for {tile_id} {year_month}"
    )
    composite = build_composite(granules, output=output)
    dest = write_composite(composite, output_dir, tile_id, date_range)
    click.echo(f"Wrote composite to {dest}")
