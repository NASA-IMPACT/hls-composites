"""Write a composite Dataset to Cloud Optimized GeoTIFFs.

One COG per data variable, named `HLS.M30.T{tile}.{start_doy}.{end_doy}.v2.0`.

Each band's nodata and scale factor come from the variable's own attrs
(set by `build_composite`), so this module needs no per-index knowledge.

The GDAL creation options (compression, predictor, etc.) are a caller-overridable
argument, defaulting to the daily HLS products' own settings.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers the .rio accessor)
import xarray as xr
from affine import Affine
from rasterio.crs import CRS

from hls_composites.composite import BROWSE_BANDS
from hls_composites.crs import corrected_grid, crs_name
from hls_composites.models import DateRange


class CogCreationOptions(TypedDict, total=False):
    """Common GDAL COG driver creation options (all optional).

    Spelled as the COG driver names them, which differs from the GTiff driver:
    `level` rather than `zlevel`, `blocksize` rather than `blockxsize`.
    """

    compress: str
    predictor: int
    level: int
    num_threads: int | str
    overview_resampling: str
    overviews: str
    bigtiff: str


DEFAULT_CREATION_OPTIONS: CogCreationOptions = {
    "compress": "DEFLATE",
    "level": 9,
    "predictor": 2,
    "overview_resampling": "NEAREST",
}
"""GDAL COG creation options applied when the caller passes none.

Matches the daily HLS products, so a composite decompresses and resamples its
overviews the same way its inputs do.
"""

ADD_OFFSET = 0.0
"""Additive offset of every encoded band. No index or reflectance band has one."""

BLOCK_SIZE = 256
"""Internal COG tile size, matching the daily HLS products.

The COG driver derives the overview levels from this and the raster size: a
3660 px HLS grid in 256 px tiles yields levels 2, 4, 8 and 16, as the daily
products carry.
"""


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


def _write_cog(
    path: Path,
    array: xr.DataArray,
    block_size: int,
    creation_options: CogCreationOptions,
    crs: CRS,
    transform: Affine,
    tags: Mapping[str, str],
) -> None:
    values = np.asarray(array.values)
    profile: dict[str, Any] = {
        "driver": "COG",
        "height": values.shape[0],
        "width": values.shape[1],
        "count": 1,
        "dtype": values.dtype,
        "crs": crs,
        "transform": transform,
        "blocksize": block_size,
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
        long_name = array.attrs.get("long_name")
        if long_name is not None:
            dst.set_band_description(1, long_name)
        dst.update_tags(**_band_tags(array), **_grid_tags(crs, transform, values.shape))
        dst.update_tags(**tags)


def _band_tags(array: xr.DataArray) -> dict[str, str]:
    """The band's own encoding, spelled as the daily HLS products spell it."""
    out = {}
    if "long_name" in array.attrs:
        out["long_name"] = str(array.attrs["long_name"])
    if "scale_factor" in array.attrs:
        out["scale_factor"] = str(array.attrs["scale_factor"])
        out["add_offset"] = str(ADD_OFFSET)
    if "nodata" in array.attrs:
        out["_FillValue"] = str(array.attrs["nodata"])
    return out


def _grid_tags(crs: CRS, transform: Affine, shape: tuple[int, ...]) -> dict[str, str]:
    """The grid, spelled as the daily HLS products spell it."""
    epsg = crs.to_epsg()
    return {
        "NROWS": str(shape[0]),
        "NCOLS": str(shape[1]),
        "ULX": str(transform.c),
        "ULY": str(transform.f),
        "SPATIAL_RESOLUTION": str(transform.a),
        "HORIZONTAL_CS_CODE": f"EPSG:{epsg}" if epsg is not None else crs.to_wkt(),
        "HORIZONTAL_CS_NAME": crs_name(crs),
    }


def write_rasters(
    computed: xr.Dataset,
    out_dir: str | Path,
    tile: str,
    date_range: DateRange,
    block_size: int = BLOCK_SIZE,
    creation_options: CogCreationOptions | None = None,
    tags: Mapping[str, str] | None = None,
) -> Path:
    """Write each product variable of a computed composite to a COG.

    Takes an already-computed Dataset rather than computing one, so the same
    arrays can also feed the browse-image renderer without a second pass over
    the graph.

    Variables named in `BROWSE_BANDS` are skipped: they are composited for the
    browse image and are not products.

    Parameters
    ----------
    computed : xarray.Dataset
        Computed composite, carrying CRS/transform and per-variable
        `nodata`/`scale_factor` attrs.
    out_dir : str or pathlib.Path
        Directory the `{granule_id}/` output folder is created under.
    tile : str
        MGRS tile ID, without the leading "T" (see `composite_id`).
    date_range : DateRange
        The composite's date range (see `composite_id`).
    block_size : int, optional
        Internal COG tile size, by default `BLOCK_SIZE`.
    creation_options : CogCreationOptions or None, optional
        GDAL COG creation options, by default `DEFAULT_CREATION_OPTIONS`.
    tags : mapping of str to str, optional
        Granule-scope GeoTIFF tags written to every file, describing facts
        this module cannot derive from the arrays -- provenance, production
        time, the compositing period. The band's own encoding and the grid
        are always written and need not be passed.

    Returns
    -------
    pathlib.Path
        The `{out_dir}/{granule_id}` directory the files were written to.
    """
    if creation_options is None:
        creation_options = DEFAULT_CREATION_OPTIONS
    granule_id = composite_id(tile, date_range)
    dest = Path(out_dir) / granule_id
    dest.mkdir(parents=True, exist_ok=True)

    sample = next(iter(computed.data_vars.values()))
    crs, transform = corrected_grid(
        sample.rio.crs,
        sample.rio.transform(),
        (sample.shape[0], sample.shape[1]),
        tile,
    )

    for name, array in computed.data_vars.items():
        if name in BROWSE_BANDS:
            continue
        _write_cog(
            dest / f"{granule_id}.{name}.tif",
            array,
            block_size,
            creation_options,
            crs,
            transform,
            {"GRANULE_ID": granule_id, **(tags or {})},
        )
    return dest
