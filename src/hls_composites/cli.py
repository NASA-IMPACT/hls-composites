"""Command-line entrypoint: argument handling around `pipeline.create_composite`."""

from pathlib import Path

import click

from hls_composites.composite import CompositeOutput
from hls_composites.exit_codes import NO_INPUTS
from hls_composites.models import DateRange
from hls_composites.pipeline import (
    Destination,
    LocalDestination,
    S3Destination,
    create_composite,
)


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
    type=click.Path(file_okay=False, path_type=Path),
    help="Local directory to write the composite into.",
)
@click.option(
    "--output-bucket",
    "output_bucket",
    envvar="OUTPUT_BUCKET",
    help="S3 bucket to upload the composite to (or set OUTPUT_BUCKET).",
)
@click.option(
    "--output-prefix",
    "output_prefix",
    envvar="OUTPUT_PREFIX",
    default="",
    help=(
        "Key prefix within --output-bucket, e.g. M30/data (or set "
        "OUTPUT_PREFIX). Composites land under {prefix}/{granule_id}/."
    ),
)
@click.option(
    "--role-arn",
    "role_arn",
    envvar="LPDAAC_READER_ROLE_ARN",
    help=(
        "IAM role to assume for reading input granules (or set "
        "LPDAAC_READER_ROLE_ARN). Omit to use ambient credentials."
    ),
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
    output_dir: Path | None,
    output_bucket: str | None,
    output_prefix: str,
    role_arn: str | None,
    indexes: bool,
    bands: bool,
) -> None:
    """Build the monthly HLS composite for one tile and write it out."""
    if indexes and bands:
        raise click.UsageError("--indexes and --bands are mutually exclusive")
    if bool(output_dir) == bool(output_bucket):
        raise click.UsageError(
            "exactly one of --output-dir or --output-bucket is required"
        )

    try:
        date_range = DateRange.for_month(year_month)
    except ValueError as error:
        raise click.BadParameter(str(error), param_hint="--year-month") from error

    destination: Destination = (
        S3Destination(output_bucket, output_prefix)
        if output_bucket
        else LocalDestination(output_dir)  # type: ignore[arg-type]
    )
    output: CompositeOutput = "bands" if bands else "indexes"

    result = create_composite(
        tile_id=tile_id,
        date_range=date_range,
        input_bucket=bucket,
        destination=destination,
        output=output,
        role_arn=role_arn,
        on_progress=click.echo,
    )

    if not result.found_granules:
        # A distinct code so the job monitor can record FAILURE_NO_INPUTS
        # rather than treating an empty period as a generic failure.
        raise SystemExit(NO_INPUTS)
