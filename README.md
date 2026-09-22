# hls-composites

Monthly, cloud-free composites of Harmonized Landsat Sentinel-2 (HLS) spectral indexes. Spectral indexes NDVI, EVI, and
NBR are produced as monthly composites per MGRS grid tile. The pixel selection criterion is the observation whose EVI2
is closest to that pixel's median EVI2 over the month.

This repository contains three main components:

- The composite algorithm and implementation as a command line interface script
- Tooling to orchestrate their forward and backward production
- Tooling to monitor and log the granule production statuses

## What a composite contains

A granule directory is named for the tile and the day-of-year bounds of the month, e.g.
`HLS.M30.T14TPN.2020183.2020213.v2.0`, and holds:

| File                           | Contents                                                             |
| ------------------------------ | -------------------------------------------------------------------- |
| `{granule_id}.{INDEX}.tif`     | EVI, NBR and NDVI, int16 scaled by 10,000                            |
| `{granule_id}.{INDEX}_std.tif` | Standard deviation of each index over the month's valid observations |
| `{granule_id}.Fmask.tif`       | QA of the selected observation                                       |
| `{granule_id}.ValidCount.tif`  | Count of valid observations per pixel                                |
| `{granule_id}.DOY.tif`         | Day of year each pixel was taken from                                |
| `{granule_id}.{INDEX}.png`     | Pseudocolor browse image per index, MODIS MYD13A3 styling            |
| `{granule_id}.cmr.xml`         | ECHO-10 granule metadata for CMR                                     |
| `{granule_id}_stac.json`       | STAC item                                                            |
| `{granule_id}.cnm.json`        | CNM submission message (S3 deliveries only)                          |

All imagery data (indexes, standard deviations, & QA bands) are Cloud Optimized GeoTIFFs (COGs) with overviews.

## Developer Setup

### Running it

This project uses the `uv` package and project manager, so you'll first want to make sure that is installed (see
[documentation](https://docs.astral.sh/uv/getting-started/installation/)).

Once `uv` is installed, you can install the project using:

```bash
# Full set of test and developer dependencies
uv sync --all-groups

# Minimal dependencies
uv sync
```

Once installed you can view the help for the main entrypoint into this project, the `hls-composites` command line
interface:

```bash
$ uv run hls-composites --help
Usage: hls-composites [OPTIONS]

  Build the monthly HLS composite for one tile and write it out.

Options:
  --tile-id TEXT          MGRS tile ID, e.g. 14TPN.  [required]
  --year-month TEXT       Composite month as YYYY-MM, e.g. "2015-07".
                          [required]
  --bucket TEXT           S3 bucket to scan for HLS granules (or set
                          HLS_BUCKET).  [required]
  --output-dir DIRECTORY  Local directory to write the composite into.
  --output-bucket TEXT    S3 bucket to upload the composite to (or set
                          OUTPUT_BUCKET).
  --output-prefix TEXT    Key prefix within --output-bucket, e.g. M30/data (or
                          set OUTPUT_PREFIX). Composites land under
                          {prefix}/{granule_id}/.
  --role-arn TEXT         IAM role to assume for reading input granules (or
                          set LPDAAC_READER_ROLE_ARN). Omit to use ambient
                          credentials.
  --indexes               Composite the spectral indices. The default.
  --bands                 Composite the raw reflectance bands instead of the
                          spectral indices.
  --help                  Show this message and exit.
```

### Linting, formatting, and tests

We have a suite of helper scripts that define our commonly used developer workflows in a single place:

```bash
scripts/test        # pytest with coverage
scripts/lint        # ruff check + format --check
scripts/format      # ruff --fix + format
scripts/typecheck   # mypy
```

### Local development

While you _could_ run the CLI locally on your host machine, the pipeline reads the original HLS products from S3 buckets
directly using an AWS IAM role that the LP DAAC have blessed for direct bucket access. To help mitigate this trouble, we
provide a Docker Compose based local developer setup that fakes the LP DAAC protected bucket.

This compose stack runs the CLI against a MinIO fake-S3 seeded with real granules. Seeding pulls from LP DAAC and needs
Earthdata Login credentials, either in `~/.netrc` or as `EARTHDATA_USERNAME` / `EARTHDATA_PASSWORD`.

```bash
docker compose up -d minio
docker compose run --rm seed
docker compose run --rm composite      # MODE=bands for the reflectance bands
```

Outputs land in `./out`; the MinIO console is at <http://localhost:9001>.

## Layout

| Path                   | Contents                                                           |
| ---------------------- | ------------------------------------------------------------------ |
| `src/hls_composites/`  | Discovery, compositing, COG writing, browse imagery, metadata, CLI |
| `src/month_opener/`    | Lambda that opens each month for forward processing                |
| `src/backfill_feeder/` | Lambda that feeds tile-months into the queue                       |
| `cdk/`                 | AWS Batch infrastructure                                           |
| `scripts/`             | Dev tooling and job submission                                     |

## Deployment

This project is deployed with the CDK:

- `dev` environment deployed on every push to `main`
- `prod` environment deployed on every release

See [docs/deployment.md](docs/deployment.md) for the stack's resources, configuration, and how to submit jobs by hand.
