"""Composite creation: masking, median-EVI2 selection, aggregation."""

import time
from datetime import date

import numpy as np
import rasterio as rio
import xarray as xr

from hls_composites.bands import (
    DEFAULT_BANDS,
    FMASK,
    NIR_NARROW,
    QA_FILL,
    RED,
    SPEC_BY_BAND,
    BandSpec,
)
from hls_composites.indices import ALL_INDICES, Index
from hls_composites.models import Granule

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


def compute_evi2(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Compute the two-band Enhanced Vegetation Index (EVI2).

    Parameters
    ----------
    red : numpy.ndarray
        Red band stack, unscaled digital numbers.
    nir : numpy.ndarray
        Near-infrared (narrow) band stack, unscaled digital numbers,
        same shape as `red`.

    Returns
    -------
    numpy.ndarray
        EVI2 values, same shape as `red`, as `float32`.
    """
    red_r = red.astype(np.float32) * RED.scale
    nir_r = nir.astype(np.float32) * NIR_NARROW.scale
    return 2.5 * (nir_r - red_r) / (nir_r + 2.4 * red_r + 1)


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
    """
    return np.sum(~bad_pixel_mask, axis=0).astype(np.uint8)


def relative_doy(
    dates: list[date], best_idx: np.ndarray, all_nan_mask: np.ndarray, start_date: date
) -> np.ndarray:
    """Compute each pixel's chosen observation date, relative to start_date.

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
    start_date : datetime.date
        Reference date the day-of-year offset is computed against.

    Returns
    -------
    numpy.ndarray
        `uint8` day-of-year offset per pixel, shaped `(Y, X)`, relative
        to `start_date`. Pixels in `all_nan_mask` are 0.
    """
    doy_vals = np.array([d.timetuple().tm_yday for d in dates], dtype=np.int32)
    start_doy = start_date.timetuple().tm_yday
    chosen_doy = doy_vals[best_idx]
    rel = chosen_doy - start_doy + 1
    return np.where(all_nan_mask, 0, rel).astype(np.uint8)


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


def _composite_block(
    reflectance: dict[BandSpec, np.ndarray],
    fmask: np.ndarray,
    dates: list[date],
    start_date: date,
    indices: list[Index] = ALL_INDICES,
) -> dict[str, np.ndarray]:
    """Composite one spatial block: the whole per-pixel pipeline, fused.

    Runs entirely on in-memory numpy for a single spatial block holding the
    full temporal stack: masks bad observations, selects the median-EVI2
    timestep per pixel, then -- for each index, one at a time -- computes the
    per-timestep index, takes the composite value at the selected timestep,
    and the standard deviation across valid timesteps. Only one index's
    temporal stack is materialized at a time, so peak memory stays bounded by
    the reflectance stack plus a single index stack.

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
    start_date : datetime.date
        Reference date for the `DOY` output (see `relative_doy`).
    indices : list of Index, optional
        Indices to composite, by default `ALL_INDICES`.

    Returns
    -------
    dict of str to numpy.ndarray
        One `(Y, X)` array per output variable: `{index.name}` and
        `{index.name}_std` (int16) for each index, plus `ValidCount` (uint8)
        and `DOY` (uint8).
    """
    bad = compute_bad_pixel_mask(reflectance, fmask)
    all_nan = compute_all_nan_mask(bad)
    evi2 = compute_evi2(reflectance[RED], reflectance[NIR_NARROW])
    best_idx = select_best_index(evi2, bad, all_nan)

    out: dict[str, np.ndarray] = {}
    for index in indices:
        per_timestep = {
            band: reflectance[SPEC_BY_BAND[band]].astype(np.float32)
            * SPEC_BY_BAND[band].scale
            for band in index.bands
        }
        stack = np.where(bad, np.nan, index(per_timestep))
        value = np.take_along_axis(stack, best_idx[None, :, :], axis=0)[0]
        with np.errstate(all="ignore"):
            std = np.nanstd(stack, axis=0)
        out[index.name] = _encode_index(value, index, all_nan)
        out[f"{index.name}_std"] = _encode_index(std, index, all_nan)

    out["ValidCount"] = valid_count(bad)
    out["DOY"] = relative_doy(dates, best_idx, all_nan, start_date)
    return out


def _default_opener(url: str) -> np.ndarray:
    with rio.open(url) as src:
        return src.read(1)


def read_band_with_retry(
    url: str,
    max_retries: int = 3,
    delay: float = 3.0,
    opener=_default_opener,
) -> np.ndarray:
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
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def build_composite(
    granules: list[Granule],
    start_date: date,
    bands: list[BandSpec] | None = None,
    opener=_default_opener,
) -> xr.Dataset:
    """Build a cloud-free composite Dataset from a list of granules.

    Reads each band across all granules, masks bad pixels (cloud,
    shadow, aerosol), picks the per-pixel observation closest to the
    median EVI2, and aggregates the result plus std/ValidCount/DOY into
    a single Dataset.

    Parameters
    ----------
    granules : list of Granule
        Granules to composite, typically from `scan_bucket_for_granules`.
        Must be non-empty.
    start_date : datetime.date
        Reference date for the output `DOY` variable (see
        `relative_doy`).
    bands : list of Band or None, optional
        Bands to include, defaulting to `DEFAULT_BANDS` when None.
    opener : callable, optional
        Function taking a URL and returning a 2D array, passed through
        to `read_band_with_retry`. Overridable for testing.

    Returns
    -------
    xarray.Dataset
        One variable per band (composite value), `{band.name}_std` for
        each reflectance band, plus `ValidCount` and `DOY`.

    Raises
    ------
    ValueError
        If `granules` is empty.
    """
    if not granules:
        raise ValueError("build_composite requires at least one granule")
    if bands is None:
        bands = DEFAULT_BANDS
    reflectance_bands = [b for b in bands if b.is_reflectance]

    raw: dict[str, np.ndarray] = {}
    for band in bands:
        urls = [asset_url(g, band) for g in granules]
        raw[band.name] = np.stack(
            [read_band_with_retry(u, opener=opener) for u in urls], axis=0
        )

    fmask = raw[FMASK.name]
    reflectance_arrays = {b: raw[b.name] for b in reflectance_bands}

    bad_pixel_mask = compute_bad_pixel_mask(reflectance_arrays, fmask)
    all_nan_mask = compute_all_nan_mask(bad_pixel_mask)
    evi2 = compute_evi2(raw[RED.name], raw[NIR_NARROW.name])
    best_idx = select_best_index(evi2, bad_pixel_mask, all_nan_mask)

    data_vars: dict[str, tuple] = {}
    for band in bands:
        composite = composite_band(raw[band.name], best_idx, all_nan_mask, band.nodata)
        data_vars[band.name] = (("y", "x"), composite.astype(band.dtype))
        if band.is_reflectance:
            std = band_std(raw[band.name], bad_pixel_mask, all_nan_mask)
            data_vars[f"{band.name}_std"] = (("y", "x"), np.round(std).astype(np.int16))

    data_vars["ValidCount"] = (("y", "x"), valid_count(bad_pixel_mask))
    dates = [g.date for g in granules]
    data_vars["DOY"] = (
        ("y", "x"),
        relative_doy(dates, best_idx, all_nan_mask, start_date),
    )

    return xr.Dataset(data_vars)
