"""Composite creation: masking, median-EVI2 selection, aggregation."""

import time
from collections.abc import Callable
from datetime import date
from typing import Literal

import numpy as np
import rasterio as rio
import rioxarray
import xarray as xr

from hls_composites.bands import (
    DEFAULT_BANDS,
    FMASK,
    QA_FILL,
    SPEC_BY_BAND,
    Band,
    BandSpec,
)
from hls_composites.indices import DEFAULT_INDICES, SELECTION_INDEX, Index
from hls_composites.models import Granule

CompositeOutput = Literal["indexes", "bands"]
"""Which quantity a composite is built over: spectral indices or raw bands."""

DOY_FILL = -1
"""Fill value for `DOY`. Julian days are 1..366, so a negative is unreachable."""

VALID_COUNT_FILL = 255
"""Fill value for `ValidCount`: the uint8 maximum.

A tile-month holds nowhere near 255 granules, so no real count can reach it.
"""

QA_BIT = {
    "cirrus": 0,
    "cloud": 1,
    "adj_cloud": 2,
    "cloud_shadow": 3,
    "snowice": 4,
    "water": 5,
    "aerosol_low": 6,
    "aerosol_high": 7,
}


def asset_url(granule: Granule, band: BandSpec) -> str:
    """Build the S3 URI for one band asset of a granule.

    Parameters
    ----------
    granule : Granule
        Granule to build the asset URL for.
    band : Band
        Band to build the asset URL for.

    Returns
    -------
    str
        Full S3 URI of the band's GeoTIFF asset.
    """
    band_code = band.code[granule.satellite]
    return f"{granule.path}.{band_code}.tif"


def compute_out_of_range_mask(bands: dict[BandSpec, np.ndarray]) -> np.ndarray:
    """Flag observations outside each band's valid range, if any.

    Checked on raw digital numbers, before `Band.scale` is applied --
    `valid_range` is expressed in the same units as the raw data.

    Parameters
    ----------
    bands : dict of Band to numpy.ndarray
        Band stacks, each shaped `(T, Y, X)`, keyed by the `Band` they
        belong to. Bands with `valid_range=(None, None)` (see `Band`,
        the default) are not checked.

    Returns
    -------
    numpy.ndarray
        Boolean mask, shaped `(T, Y, X)`, True where any checked band
        is outside its `valid_range`.
    """
    template = next(iter(bands.values()))
    mask = np.zeros_like(template, dtype=bool)
    for band, arr in bands.items():
        lo, hi = band.valid_range
        if lo is not None:
            mask |= arr < lo
        if hi is not None:
            mask |= arr > hi
    return mask


def compute_basic_mask(
    bands: dict[BandSpec, np.ndarray], fmask: np.ndarray
) -> np.ndarray:
    """Flag cloud, shadow, fill, and out-of-range observations.

    Parameters
    ----------
    bands : dict of Band to numpy.ndarray
        Band stacks, each shaped `(T, Y, X)` (see `compute_out_of_range_mask`).
    fmask : numpy.ndarray
        Fmask QA band stack, shaped `(T, Y, X)`.

    Returns
    -------
    numpy.ndarray
        Boolean mask, shaped `(T, Y, X)`, True where an observation is
        cloud, adjacent-cloud, cloud-shadow, fill, or outside a band's
        valid range.
    """
    cloud = (fmask & (1 << QA_BIT["cloud"])) > 0
    adj_cloud = (fmask & (1 << QA_BIT["adj_cloud"])) > 0
    cloud_shadow = (fmask & (1 << QA_BIT["cloud_shadow"])) > 0
    fill = fmask == QA_FILL
    out_of_range = compute_out_of_range_mask(bands)
    return cloud | adj_cloud | cloud_shadow | fill | out_of_range


def compute_bad_pixel_mask(
    bands: dict[BandSpec, np.ndarray], fmask: np.ndarray
) -> np.ndarray:
    """Flag observations excluded from compositing, aerosol included.

    Extends `compute_basic_mask` with a conditional high-aerosol rule: a
    high-aerosol observation is excluded only if a low/moderate-aerosol
    alternative exists elsewhere in that pixel's temporal stack, to avoid
    manufacturing data holes.

    Parameters
    ----------
    bands : dict of Band to numpy.ndarray
        Band stacks, each shaped `(T, Y, X)` (see `compute_out_of_range_mask`).
    fmask : numpy.ndarray
        Fmask QA band stack, shaped `(T, Y, X)`.

    Returns
    -------
    numpy.ndarray
        Boolean mask, shaped `(T, Y, X)`, True where an observation is
        excluded from compositing.
    """
    basic = compute_basic_mask(bands, fmask)
    is_high_aerosol = ((fmask & (1 << QA_BIT["aerosol_high"])) > 0) & (
        (fmask & (1 << QA_BIT["aerosol_low"])) > 0
    )
    is_low_mod_aerosol = ~is_high_aerosol & ~basic
    any_low_mod_available = np.any(is_low_mod_aerosol, axis=0)
    return basic | (is_high_aerosol & any_low_mod_available)


def compute_all_nan_mask(bad_pixel_mask: np.ndarray) -> np.ndarray:
    """Flag pixels with no valid observation in the temporal stack.

    Parameters
    ----------
    bad_pixel_mask : numpy.ndarray
        Boolean mask, shaped `(T, Y, X)` (see `compute_bad_pixel_mask`).

    Returns
    -------
    numpy.ndarray
        Boolean mask, shaped `(Y, X)`, True where every timestep at
        that pixel is excluded.
    """
    return np.all(bad_pixel_mask, axis=0)


def to_reflectance(
    digital_numbers: dict[BandSpec, np.ndarray], bands: tuple[Band, ...]
) -> dict[Band, np.ndarray]:
    """Scale the requested bands' digital numbers to physical reflectance.

    Parameters
    ----------
    digital_numbers : dict of BandSpec to numpy.ndarray
        Raw band stacks, shaped `(T, Y, X)`, keyed by their `BandSpec`.
    bands : tuple of Band
        Spectral bands to convert, e.g. an `Index`'s `bands`.

    Returns
    -------
    dict of Band to numpy.ndarray
        `float32` reflectance stacks keyed by spectral `Band`, ready to
        pass to an `Index`.
    """
    return {
        band: digital_numbers[SPEC_BY_BAND[band]].astype(np.float32)
        * SPEC_BY_BAND[band].scale
        for band in bands
    }


def select_best_index(
    evi2: np.ndarray, bad_pixel_mask: np.ndarray, all_nan_mask: np.ndarray
) -> np.ndarray:
    """Pick the observation whose EVI2 is closest to the per-pixel median.

    Parameters
    ----------
    evi2 : numpy.ndarray
        EVI2 values, shaped `(T, Y, X)` (see `compute_evi2`).
    bad_pixel_mask : numpy.ndarray
        Boolean mask, shaped `(T, Y, X)` (see `compute_bad_pixel_mask`).
    all_nan_mask : numpy.ndarray
        Boolean mask, shaped `(Y, X)` (see `compute_all_nan_mask`).

    Returns
    -------
    numpy.ndarray
        `int16` array, shaped `(Y, X)`, with the chosen timestep index
        per pixel. Pixels in `all_nan_mask` get index 0.
    """
    evi2_masked = evi2.copy()
    evi2_masked[bad_pixel_mask] = np.nan
    with np.errstate(all="ignore"):
        target = np.nanmedian(evi2_masked, axis=0)
        diff = np.abs(evi2_masked - target)
    diff[np.isnan(diff)] = 1e9
    idx = np.argmin(diff, axis=0).astype(np.int16)
    idx[all_nan_mask] = 0
    return idx


def composite_band(
    values: np.ndarray, best_idx: np.ndarray, all_nan_mask: np.ndarray, nodata: int
) -> np.ndarray:
    """Select each pixel's band value at its chosen observation index.

    Parameters
    ----------
    values : numpy.ndarray
        Band values, shaped `(T, Y, X)`.
    best_idx : numpy.ndarray
        Chosen timestep index per pixel, shaped `(Y, X)` (see
        `select_best_index`).
    all_nan_mask : numpy.ndarray
        Boolean mask, shaped `(Y, X)` (see `compute_all_nan_mask`).
    nodata : int
        Fill value for pixels in `all_nan_mask`.

    Returns
    -------
    numpy.ndarray
        Composite band values, shaped `(Y, X)`.
    """
    chosen = np.take_along_axis(values, best_idx[None, :, :], axis=0)[0]
    return np.where(all_nan_mask, nodata, chosen)


def band_std(
    values: np.ndarray, bad_pixel_mask: np.ndarray, all_nan_mask: np.ndarray
) -> np.ndarray:
    """Compute the per-pixel standard deviation across valid observations.

    Parameters
    ----------
    values : numpy.ndarray
        Band values, shaped `(T, Y, X)`.
    bad_pixel_mask : numpy.ndarray
        Boolean mask, shaped `(T, Y, X)` (see `compute_bad_pixel_mask`).
    all_nan_mask : numpy.ndarray
        Boolean mask, shaped `(Y, X)` (see `compute_all_nan_mask`).

    Returns
    -------
    numpy.ndarray
        `float32` standard deviation per pixel, shaped `(Y, X)`. Pixels
        in `all_nan_mask` are 0.
    """
    values_f = values.astype(np.float32).copy()
    values_f[bad_pixel_mask] = np.nan
    with np.errstate(all="ignore"):
        std = np.nanstd(values_f, axis=0)
    std[all_nan_mask] = 0
    return std


def valid_count(bad_pixel_mask: np.ndarray) -> np.ndarray:
    """Count valid (unmasked) observations per pixel.

    Parameters
    ----------
    bad_pixel_mask : numpy.ndarray
        Boolean mask, shaped `(T, Y, X)` (see `compute_bad_pixel_mask`).

    Returns
    -------
    numpy.ndarray
        `uint8` count of unmasked observations per pixel, shaped `(Y, X)`.
        Pixels with no valid observation get `VALID_COUNT_FILL` rather than 0, so
        they read as absent data rather than as a measured zero.
    """
    counts = np.sum(~bad_pixel_mask, axis=0)
    return np.where(counts == 0, VALID_COUNT_FILL, counts).astype(np.uint8)


def observation_doy(
    dates: list[date], best_idx: np.ndarray, all_nan_mask: np.ndarray
) -> np.ndarray:
    """Compute the Julian day of each pixel's chosen observation.

    Parameters
    ----------
    dates : list of datetime.date
        Observation date per timestep, same order as the `T` axis of
        `best_idx`'s source stacks.
    best_idx : numpy.ndarray
        Chosen timestep index per pixel, shaped `(Y, X)` (see
        `select_best_index`).
    all_nan_mask : numpy.ndarray
        Boolean mask, shaped `(Y, X)` (see `compute_all_nan_mask`).

    Returns
    -------
    numpy.ndarray
        `int16` Julian day-of-year (1..366) per pixel, shaped `(Y, X)`.
        Pixels in `all_nan_mask` get `DOY_FILL`.
    """
    doy_vals = np.array([d.timetuple().tm_yday for d in dates], dtype=np.int16)
    return np.where(all_nan_mask, DOY_FILL, doy_vals[best_idx]).astype(np.int16)


def _encode_index(
    values: np.ndarray, index: Index, all_nan_mask: np.ndarray
) -> np.ndarray:
    """Scale a raw float index (or its std) to its int16 storage encoding.

    Applies `index.scale_factor` (values are divided by it, matching the
    HLS-VI convention), clips to the int16 range, rounds, and writes
    `index.fill_value` where the pixel has no valid observation or the raw
    value is non-finite.

    Parameters
    ----------
    values : numpy.ndarray
        Raw float index values, shaped `(Y, X)`.
    index : Index
        The index these values belong to (supplies `scale_factor` and
        `fill_value`).
    all_nan_mask : numpy.ndarray
        Boolean mask, shaped `(Y, X)` (see `compute_all_nan_mask`).

    Returns
    -------
    numpy.ndarray
        `int16` encoded values, shaped `(Y, X)`.
    """
    scaled = values / index.scale_factor
    invalid = all_nan_mask | ~np.isfinite(scaled)
    info = np.iinfo(np.int16)
    clipped = np.clip(np.where(invalid, 0, scaled), info.min, info.max)
    out = np.round(clipped).astype(np.int16)
    out[invalid] = index.fill_value
    return out


def _encode_band_std(
    values: np.ndarray, band: BandSpec, all_nan_mask: np.ndarray
) -> np.ndarray:
    """Round a raw band's temporal standard deviation to its int16 encoding.

    The standard deviation is computed on raw digital numbers, which are
    already the band's storage encoding, so -- unlike `_encode_index` -- no
    rescaling happens here.

    Parameters
    ----------
    values : numpy.ndarray
        Standard deviation in digital numbers, shaped `(Y, X)` (see `band_std`).
    band : BandSpec
        The band these values belong to (supplies `nodata`).
    all_nan_mask : numpy.ndarray
        Boolean mask, shaped `(Y, X)` (see `compute_all_nan_mask`).

    Returns
    -------
    numpy.ndarray
        `int16` encoded standard deviation, shaped `(Y, X)`.
    """
    invalid = all_nan_mask | ~np.isfinite(values)
    info = np.iinfo(np.int16)
    clipped = np.clip(np.where(invalid, 0, values), info.min, info.max)
    out = np.round(clipped).astype(np.int16)
    out[invalid] = band.nodata
    return out


def _composite_block(
    reflectance: dict[BandSpec, np.ndarray],
    fmask: np.ndarray,
    dates: list[date],
    indices: list[Index] = DEFAULT_INDICES,
    output: CompositeOutput = "indexes",
) -> dict[str, np.ndarray]:
    """Composite one spatial block: the whole per-pixel pipeline, fused.

    Runs entirely on in-memory numpy for a single spatial block holding the
    full temporal stack: masks bad observations, selects the median-EVI2
    timestep per pixel, then takes the composite value at the selected
    timestep and the standard deviation across valid timesteps -- either for
    each spectral index or for each reflectance band, per `output`.

    In `"indexes"` mode only one index's temporal stack is materialized at a
    time, so peak memory stays bounded by the reflectance stack plus a single
    index stack. `"bands"` mode composites the reflectance stack directly and
    materializes nothing extra.

    Pixel selection is identical in both modes: median-EVI2, which needs
    `RED` and `NIR_NARROW` regardless of what is being composited.

    Parameters
    ----------
    reflectance : dict of BandSpec to numpy.ndarray
        Reflectance band stacks (raw digital numbers), each shaped
        `(T, Y, X)`, keyed by `BandSpec`. Must include every band referenced
        by `indices` (via `SPEC_BY_BAND`) plus `RED`/`NIR_NARROW` for EVI2.
    fmask : numpy.ndarray
        Fmask QA band stack, shaped `(T, Y, X)`.
    dates : list of datetime.date
        Observation date per timestep, in `T`-axis order.
    indices : list of Index, optional
        Indices to composite, by default `DEFAULT_INDICES`. Ignored when
        `output` is `"bands"`.
    output : {"indexes", "bands"}, optional
        Whether to composite the spectral indices (the default) or the raw
        reflectance bands.

    Returns
    -------
    dict of str to numpy.ndarray
        One `(Y, X)` array per output variable, plus `ValidCount` (uint8,
        filled with `VALID_COUNT_FILL`) and `DOY` (int16, filled with
        `DOY_FILL`). For `"indexes"`, `{index.name}` and `{index.name}_std`
        (int16) per index; for `"bands"`, `{band.name}` and `{band.name}_std`
        (int16) per reflectance band, in `reflectance` order.
    """
    bad = compute_bad_pixel_mask(reflectance, fmask)
    all_nan = compute_all_nan_mask(bad)
    evi2 = SELECTION_INDEX(to_reflectance(reflectance, SELECTION_INDEX.bands))
    best_idx = select_best_index(evi2, bad, all_nan)

    out: dict[str, np.ndarray] = {}
    if output == "bands":
        for band, values in reflectance.items():
            composited = composite_band(values, best_idx, all_nan, band.nodata)
            out[band.name] = composited.astype(band.dtype)
            out[f"{band.name}_std"] = _encode_band_std(
                band_std(values, bad, all_nan), band, all_nan
            )
    else:
        for index in indices:
            per_timestep = to_reflectance(reflectance, index.bands)
            stack = np.where(bad, np.nan, index(per_timestep))
            value = np.take_along_axis(stack, best_idx[None, :, :], axis=0)[0]
            with np.errstate(all="ignore"):
                std = np.nanstd(stack, axis=0)
            out[index.name] = _encode_index(value, index, all_nan)
            out[f"{index.name}_std"] = _encode_index(std, index, all_nan)

    out["ValidCount"] = valid_count(bad)
    out["DOY"] = observation_doy(dates, best_idx, all_nan)
    return out


def _default_opener(url: str) -> np.ndarray:
    with rio.open(url) as src:
        return src.read(1)


def read_band_with_retry[ReadResult](
    url: str,
    max_retries: int = 3,
    delay: float = 3.0,
    opener: Callable[[str], ReadResult] = _default_opener,  # type: ignore[assignment]
) -> ReadResult:
    """Read one band asset, retrying transient failures.

    Parameters
    ----------
    url : str
        S3 or local URL of the band's GeoTIFF asset (see `asset_url`).
    max_retries : int, optional
        Maximum number of attempts before giving up, by default 3.
    delay : float, optional
        Seconds to wait between attempts, by default 3.0.
    opener : callable, optional
        Function taking a URL and returning a 2D array; defaults to
        reading band 1 via rasterio. Overridable for testing.

    Returns
    -------
    numpy.ndarray
        The band's pixel data.

    Raises
    ------
    Exception
        Whatever `opener` raised on the last attempt, if every attempt
        failed.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            return opener(url)
        except Exception as e:  # noqa: BLE001  any read failure is retryable
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(delay)
    assert last_error is not None
    raise last_error


BLOCK_SIZE = 512  # HLS COG native internal tiling; the spatial chunk we fuse over


def _default_da_opener(url: str) -> xr.DataArray:
    """Open one band asset lazily, chunked to the HLS native block size.

    Parameters
    ----------
    url : str
        S3 or local URL of the band's GeoTIFF asset (see `asset_url`).

    Returns
    -------
    xarray.DataArray
        Lazy, dask-backed `(y, x)` array chunked at `BLOCK_SIZE`, carrying
        the raster's CRS/transform via the `rio` accessor.
    """
    array = rioxarray.open_rasterio(
        url, chunks={"x": BLOCK_SIZE, "y": BLOCK_SIZE}, lock=False
    )
    # open_rasterio's return type spans DataArray/Dataset/list; a single-asset
    # COG always yields a DataArray.
    assert isinstance(array, xr.DataArray)
    if "band" in array.dims:
        array = array.squeeze("band", drop=True)
    return array


def _map_block_kernel(
    ds_block: xr.Dataset,
    *,
    dates: list[date],
    indices: list[Index],
    bands: list[BandSpec],
    output: CompositeOutput,
) -> xr.Dataset:
    """Run `_composite_block` on one spatial block, wrapped for `xr.map_blocks`.

    Receives an in-memory `Dataset` block (one spatial chunk, full time axis),
    delegates the numpy pipeline to `_composite_block`, and re-wraps the
    outputs into a `Dataset` that reuses the block's `(y, x)` coords (and thus
    its CRS/transform).
    """
    reflectance = {b: ds_block[b.name].values for b in bands if b.is_reflectance}
    fmask = ds_block[FMASK.name].values
    out = _composite_block(reflectance, fmask, dates, indices, output)
    # Carry every non-temporal coord through (y, x, and the CRS's spatial_ref)
    # so the returned block matches the template map_blocks validates against.
    coords = {
        name: coord
        for name, coord in ds_block.coords.items()
        if "time" not in coord.dims
    }
    return xr.Dataset(
        {name: (("y", "x"), array) for name, array in out.items()},
        coords=coords,
    )


def build_composite(
    granules: list[Granule],
    bands: list[BandSpec] | None = None,
    indices: list[Index] | None = None,
    output: CompositeOutput = "indexes",
    opener=_default_da_opener,
) -> xr.Dataset:
    """Build a lazy composite Dataset from a list of granules.

    Reads each band across all granules lazily (dask-chunked at the HLS native
    block size), then applies the entire masking / median-EVI2 selection /
    aggregation pipeline as a single fused `xr.map_blocks` kernel per spatial
    block (see `_composite_block`). The returned Dataset is lazy; computation
    streams block-by-block when it is written or `.compute()`-ed, keeping
    memory bounded to roughly one block's temporal stack.

    Parameters
    ----------
    granules : list of Granule
        Granules to composite, typically from `scan_bucket_for_granules`.
        Must be non-empty.
    bands : list of BandSpec or None, optional
        Reflectance + QA bands to read, defaulting to `DEFAULT_BANDS` when None.
        Every band is read whatever `output` is, since the QA band drives
        masking and RED/NIR_NARROW drive selection.
    indices : list of Index or None, optional
        Spectral indices to composite, defaulting to `DEFAULT_INDICES` when
        None. Pass `ALL_INDICES` for every index defined in `indices.py`.
        Ignored when `output` is `"bands"`.
    output : {"indexes", "bands"}, optional
        Whether to composite the spectral indices (the default) or the raw
        reflectance bands.
    opener : callable, optional
        Function taking a URL and returning a lazy `(y, x)` DataArray, passed
        through to `read_band_with_retry`. Overridable for testing.

    Returns
    -------
    xarray.Dataset
        Lazy Dataset carrying the granules' CRS/transform, with `ValidCount`
        (uint8) and `DOY` (int16) plus, per `output`, either `{index.name}`
        and `{index.name}_std` per index or `{band.name}` and
        `{band.name}_std` per reflectance band (int16 either way).

    Raises
    ------
    ValueError
        If `granules` is empty.
    """
    if not granules:
        raise ValueError("build_composite requires at least one granule")
    if bands is None:
        bands = DEFAULT_BANDS
    if indices is None:
        indices = DEFAULT_INDICES

    data_vars: dict[str, xr.DataArray] = {}
    for band in bands:
        arrays = [
            read_band_with_retry(asset_url(g, band), opener=opener) for g in granules
        ]
        data_vars[band.name] = xr.concat(arrays, dim="time")
    stacked = xr.Dataset(data_vars).chunk(
        {"time": -1, "y": BLOCK_SIZE, "x": BLOCK_SIZE}
    )

    template2d = stacked[FMASK.name].isel(time=0, drop=True)
    template_vars: dict[str, xr.DataArray] = {}
    if output == "bands":
        for band in bands:
            if not band.is_reflectance:
                continue
            template_vars[band.name] = xr.zeros_like(template2d, dtype=band.dtype)
            template_vars[f"{band.name}_std"] = xr.zeros_like(
                template2d, dtype=np.int16
            )
    else:
        for index in indices:
            template_vars[index.name] = xr.zeros_like(template2d, dtype=np.int16)
            template_vars[f"{index.name}_std"] = xr.zeros_like(
                template2d, dtype=np.int16
            )
    template_vars["ValidCount"] = xr.zeros_like(template2d, dtype=np.uint8)
    template_vars["DOY"] = xr.zeros_like(template2d, dtype=np.int16)
    template = xr.Dataset(template_vars)

    dates = [g.date for g in granules]
    composite = xr.map_blocks(
        _map_block_kernel,
        stacked,
        kwargs={
            "dates": dates,
            "indices": indices,
            "bands": bands,
            "output": output,
        },
        template=template,
    )

    # Self-describe each var's nodata/scale so the writer stays generic. The
    # aux layers share one fill and need no scale.
    if output == "bands":
        for band in bands:
            if not band.is_reflectance:
                continue
            for name in (band.name, f"{band.name}_std"):
                composite[name].attrs["nodata"] = band.nodata
                composite[name].attrs["scale_factor"] = band.scale
    else:
        for index in indices:
            for name in (index.name, f"{index.name}_std"):
                composite[name].attrs["nodata"] = index.fill_value
                composite[name].attrs["scale_factor"] = index.scale_factor
    composite["ValidCount"].attrs["nodata"] = VALID_COUNT_FILL
    composite["DOY"].attrs["nodata"] = DOY_FILL
    return composite
