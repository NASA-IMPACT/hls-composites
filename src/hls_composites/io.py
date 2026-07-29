"""Write a composite Dataset to internally-tiled, compressed GeoTIFFs.

One GeoTIFF per data variable, named like the prototype's monthly product
(`HLS.M30.T{tile}.{start_doy}.{end_doy}.v2.0`).

Each band's nodata and scale factor come from the variable's own attrs
(set by `build_composite`), so this module needs no per-index knowledge.

The GDAL creation options (compression, predictor, etc.) are a caller-overridable
arguments.
"""

from pathlib import Path
from typing import TypedDict

import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr

from hls_composites.composite import BLOCK_SIZE
from hls_composites.models import DateRange


class GeoTiffCreationOptions(TypedDict, total=False):
    """Common GDAL GeoTIFF creation options (all optional).
    """

    compress: str
    predictor: int
    zlevel: int
    zstd_level: int
    level: int
    num_threads: int | str
    interleave: str
    bigtiff: str


DEFAULT_CREATION_OPTIONS: GeoTiffCreationOptions = {"compress": "LZW"}
"""GDAL GeoTIFF creation options applied when the caller passes none."""


def composite_id(tile: str, date_range: DateRange) -> str:
    """Build the monthly composite granule ID for `tile` over `date_range`.

    Parameters
    ----------
    tile : str
        MGRS tile ID, without the leading "T", e.g. `"14TPN"`.
    date_range : DateRange
        The composite's date range; encoded as `%Y%j` day-of-year bounds.

    Returns
    -------
    str
        e.g. `"HLS.M30.T14TPN.2020183.2020213.v2.0"`.
    """
    start = date_range.start.strftime("%Y%j")
    end = date_range.end.strftime("%Y%j")
    return f"HLS.M30.T{tile}.{start}.{end}.v2.0"


def _write_geotiff(
    path: Path,
    array: xr.DataArray,
    block_size: int,
    creation_options: GeoTiffCreationOptions,
) -> None:
    values = np.asarray(array.values)
    profile: dict[str, object] = {
        "driver": "GTiff",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": values.dtype,
        "crs": array.rio.crs,  # type: ignore[attr-defined]
        "transform": array.rio.transform(),  # type: ignore[attr-defined]
        "tiled": True,
        "blockxsize": block_size,
        "blockysize": block_size,
        **creation_options,
    }
    nodata = array.attrs.get("nodata")
    if nodata is not None:
        profile["nodata"] = nodata
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(values, 1)
        scale = array.attrs.get("scale_factor")
        if scale is not None:
            dst.scales = (scale,)


def write_composite(
    ds: xr.Dataset,
    out_dir: str | Path,
    tile: str,
    date_range: DateRange,
    block_size: int = BLOCK_SIZE,
    creation_options: GeoTiffCreationOptions | None = None,
) -> Path:
    """Write each variable of a composite Dataset to a tiled GeoTIFF.

    The Dataset is computed once (a single fused pass over its lazy graph),
    then each variable is written to `{out_dir}/{granule_id}/{granule_id}.{var}.tif`
    with internal `block_size` tiling, the given creation options, and the
    variable's own nodata/scale attrs.

    Parameters
    ----------
    ds : xarray.Dataset
        Composite from `build_composite` (lazy or already computed), carrying
        CRS/transform and per-variable `nodata`/`scale_factor` attrs.
    out_dir : str or pathlib.Path
        Directory the `{granule_id}/` output folder is created under.
    tile : str
        MGRS tile ID, without the leading "T" (see `composite_id`).
    date_range : DateRange
        The composite's date range (see `composite_id`).
    block_size : int, optional
        Internal GeoTIFF tile size, by default `BLOCK_SIZE` (512).
    creation_options : GeoTiffCreationOptions or None, optional
        GDAL GeoTIFF creation options (e.g. `compress`, `predictor`), merged
        over the internal-tiling defaults. Defaults to `DEFAULT_CREATION_OPTIONS`
        when None.

    Returns
    -------
    pathlib.Path
        The `{out_dir}/{granule_id}` directory the files were written to.
    """
    if creation_options is None:
        creation_options = DEFAULT_CREATION_OPTIONS
    computed = ds.compute()
    granule_id = composite_id(tile, date_range)
    dest = Path(out_dir) / granule_id
    dest.mkdir(parents=True, exist_ok=True)
    for name, array in computed.data_vars.items():
        _write_geotiff(
            dest / f"{granule_id}.{name}.tif", array, block_size, creation_options
        )
    return dest
