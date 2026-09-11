"""Export a Lambda bundle's requirements.txt from uv.lock, before synth.

`aws-lambda-python-alpha` does have native uv support, but it cannot be used:
it runs a fixed `uv export` with no way to select a dependency group, so it
would bundle the project's own dependencies -- rasterio, GDAL, xarray, dask --
into Lambdas that import none of them. Exporting one group keeps the bundle to
8 packages rather than 60.
"""

import subprocess
from functools import cache
from pathlib import Path

REQUIREMENTS_NAME = "requirements.txt"
"""The filename PythonFunction looks for. Not configurable on its side."""

LAMBDA_EXCLUDE = [
    "**/__pycache__",
    "**/*.egg-info",
]


@cache
def export_requirements(entry: str, group: str) -> str:
    """Write `group`'s locked dependencies to `{entry}/requirements.txt`.

    Cached so that several functions sharing an entry and group export once.

    Returns
    -------
    str
        Path to the written manifest.
    """
    destination = Path(entry) / REQUIREMENTS_NAME
    command = [
        "uv",
        "export",
        "--only-group",
        group,
        "--frozen",
        "--no-emit-project",
        "--no-dev",
        "--no-editable",
        "-o",
        str(destination),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            "uv is not on PATH, so the Lambda dependency manifest cannot be "
            "exported. Synth requires uv."
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"failed to export the '{group}' dependency group:\n{error.stderr}"
        ) from error

    if not destination.exists():
        raise RuntimeError(
            f"{destination} was not written; PythonFunction would bundle source "
            f"with no dependencies installed"
        )
    return str(destination)
