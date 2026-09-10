"""Bundling hooks that export a requirements.txt from uv.lock for PythonFunction.

`aws-lambda-python-alpha` does have native uv support, but it cannot be used
here. It runs a fixed `uv export --frozen --no-emit-workspace --no-dev
--no-editable` with no way to select a dependency group, so it would bundle the
project's own dependencies -- rasterio, GDAL, xarray, dask -- into a Lambda that
imports none of them. It also requires uv.lock beside the handler, whereas the
lockfile lives at the repository root.

These hooks mount the lockfile into the bundling container and export only the
`backfill` group instead: 8 packages rather than 60.
"""

import os

import jsii
from aws_cdk import DockerVolume, aws_lambda_python_alpha as lambda_python

UV_ASSET_REQUIREMENTS = "/asset-requirements"

UV_DOCKER_VOLUMES = [
    DockerVolume(
        host_path=os.path.abspath(file),
        container_path=f"{UV_ASSET_REQUIREMENTS}/{file}",
    )
    for file in ("pyproject.toml", "uv.lock")
]

LAMBDA_EXCLUDE = [
    "**/__pycache__",
    "**/*.egg-info",
]


@jsii.implements(lambda_python.ICommandHooks)
class UvHooks:
    """Export `only_groups` from uv.lock as the bundle's requirements.txt.

    `--only-group` excludes the project's own dependencies, which is what keeps
    rasterio and the rest of the geospatial stack out of the Lambda bundle. The
    handlers' `hls_composites` imports resolve from the copied `src/` tree.
    """

    def __init__(self, only_groups: list[str]):
        super().__init__()
        self.only_groups = only_groups

    def before_bundling(self, input_dir: str, output_dir: str) -> list[str]:
        # input_dir is a bind mount of the host's src/, so everything written
        # there lands in the working tree. uv installs into the container's own
        # filesystem, and the exported manifest is cleaned up afterwards.
        groups = " ".join(f"--only-group {group}" for group in self.only_groups)
        return [
            "pip install --quiet --target /tmp/uv-bin uv",
            "export PATH=/tmp/uv-bin/bin:$PATH",
            "export PYTHONPATH=/tmp/uv-bin",
            "export UV_CACHE_DIR=/tmp/uv-cache",
            f"cd {UV_ASSET_REQUIREMENTS}",
            (
                f"uv export {groups} --frozen --no-emit-project --no-dev "
                f"--no-editable -o {input_dir}/requirements.txt"
            ),
        ]

    def after_bundling(self, input_dir: str, output_dir: str) -> list[str]:
        """Remove the manifest this hook generated in the mounted source tree."""
        return [f"rm -f {input_dir}/requirements.txt"]
