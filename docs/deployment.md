# Deployment

The AWS Batch infrastructure that runs the `hls-composites` container lives in `cdk/` and is deployed with the AWS CDK:

- **dev** -- on every push to `main` (`.github/workflows/pr.yaml`, job `deploy-dev`)
- **prod** -- on publishing a release, after the release image is in ECR
  (`.github/workflows/build_push_release_ecr.yml`, job `deploy-prod`)
- **either, on demand** -- run the `Deploy` workflow manually and pick the environment

## What the stack creates

| Resource            | Notes                                                                       |
| ------------------- | --------------------------------------------------------------------------- |
| Compute environment | Managed EC2, spot, capacity-optimized, MCP AMI, private isolated subnets    |
| Job queue           | `hls-composites-{stage}-job-queue`                                          |
| Job definition      | Runs `PROCESSING_CONTAINER_ECR_URI` on EC2                                  |
| Batch service role  | `AWSBatchServiceRole` + `AmazonSSMReadOnlyAccess`                           |
| Execution role      | Pulls the image from ECR and ships logs                                     |
| Job role            | `hls-composites-processing-role-{stage}` -- the container's own credentials |
| Log group           | `PROCESSING_LOG_GROUP_NAME`, `PROCESSING_LOG_RETENTION` retention           |

## Submitting jobs by hand

Use `scripts/submit-job` for backfills and testing:

```bash
scripts/submit-job --tile-id 14TPN --year-month 2020-07 \
  --job-queue hls-composites-dev-job-queue \
  --job-definition hls-composites-dev-job-definition
```

It submits through the job monitor's `JobGroup`, which carries the `bejm_*` identity fields in the job's Batch
parameters. The monitor only tracks jobs that have them, so a job submitted any other way runs but never appears in
its records or in the Athena tables. The script defaults to `us-west-2`; pass `--region` for anything else.

`aws batch submit-job` still works for a one-off you do not need tracked:

```bash
aws batch submit-job \
  --job-name composite-14TPN-2020-07 \
  --job-queue hls-composites-dev-job-queue \
  --job-definition <JobDefinitionArn from the stack outputs> \
  --container-overrides 'command=["--tile-id","14TPN","--year-month","2020-07"]'
```

The job definition already sets `OUTPUT_BUCKET` and `OUTPUT_PREFIX`, so the composite is uploaded to S3. You _could_
pass your own `--output-bucket` to upload elsewhere, BUT the IAM role would need to be manually updated to include
permissions to write to that bucket.

## Configuration

Every setting in `cdk/settings.py` is read from an environment variable. In CI, the deploy workflow copies every
variable and secret of the target GitHub environment into the process environment before running `cdk deploy`, so
configuration is entirely a matter of what those environments hold. See `.env.example` for the full list and
local-development defaults.

Set these as **variables** on the GitHub environments `dev` and `prod`:

- `AWS_ROLE_TO_ASSUME_ARN` -- the deploy role assumed via OIDC
- `STACK_NAME`, `STAGE`
- `MCP_ACCOUNT_ID`, `MCP_ACCOUNT_REGION`, `MCP_IAM_PERMISSION_BOUNDARY_ARN`, `VPC_ID`
- `INPUT_BUCKET_NAME`, `OUTPUT_BUCKET_NAME`, `OUTPUT_PREFIX`, `PROCESSING_BUCKET_NAME_PREFIX`
- `ATHENA_DATABASE_NAME`, `ATHENA_INVENTORY_START_DATETIME`
- `PROCESSING_CONTAINER_ECR_URI`, `PROCESSING_LOG_GROUP_NAME`
- optionally `LPDAAC_READER_ROLE_ARN` and any of the tuning settings

### Container image

The deploy workflow does not compute an image tag. `PROCESSING_CONTAINER_ECR_URI` is a GitHub environment variable, so
`dev` points at whatever tag you choose and `prod` points at the release tag (or `:latest`, which the release build also
pushes). Note that nothing builds an image on push to `main` today -- only pull requests (`pr-<n>`, `sha-<sha>`) and
releases produce images.

## Prerequisites outside this repository

1. **GitHub environments** `dev` and `prod`, holding the variables above.
2. **Deploy role trust policy.** The role named by `AWS_ROLE_TO_ASSUME_ARN` must accept the OIDC subject claims
   `repo:NASA-IMPACT/hls-composites:environment:dev` and `repo:NASA-IMPACT/hls-composites:environment:prod`. It also
   needs CDK deploy permissions and the EC2 describe permissions that `Vpc.from_lookup` uses (`cdk.context.json` is not
   committed, so the VPC is looked up on every deploy).
3. **LP DAAC reader role.** `LPDAAC_READER_ROLE_ARN` names the role in `hls-vi-historical-orchestration` that LP DAAC
   bucket policies grant read access to. This stack grants our job role `sts:AssumeRole` on it, but that only works once
   _that_ role's trust policy names `hls-composites-processing-role-dev` / `hls-composites-processing-role-prod`.

## Local synth

```bash
uv sync --frozen --group deploy
cp .env.example .env   # then edit
set -a; source .env; set +a
npx aws-cdk@v2 synth
```

## Container environment variables

The job definition sets:

| Variable                 | Meaning                                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------------------- |
| `HLS_BUCKET`             | Input bucket the CLI scans for granules. Already read by `hls_composites`.                            |
| `OUTPUT_BUCKET`          | Destination bucket.                                                                                   |
| `LPDAAC_READER_ROLE_ARN` | Role to assume for LP DAAC reads.                                                                     |
| `PYTHONUNBUFFERED`       | Keeps logs flowing to CloudWatch.                                                                     |
| `DASK_NUM_WORKERS`       | Dask threads per job, from `PROCESSING_JOB_DASK_NUM_WORKERS`. Omitted when unset (one per host core). |

Each composite directory also carries `{granule_id}.cmr.xml` (ECHO-10 granule metadata for CMR) and
`{granule_id}_stac.json` (a STAC item). Both are written from one model, so they cannot disagree. The collection-level
values they carry -- short name, dataset ID, DOI, product URI, and the compositing algorithm description -- are
constants in `src/hls_composites/metadata/models.py`. Those the DAAC has not assigned yet are the literal string
`PLACEHOLDER`; the STAC item omits `sci:doi` entirely until a real DOI exists, since the scientific extension validates
it against a DOI pattern.
