"""Composite creation: masking, median-EVI2 selection, aggregation."""

import time
from datetime import date

import numpy as np
import rasterio as rio
import xarray as xr

from hls_composites.models import Granule

DEFAULT_BANDS = ["red", "green", "blue", "nir_narrow", "swir_1", "swir_2", "Fmask"]

BAND_CODE: dict[str, dict[str, str]] = {
    "L30": {
        "red": "B04",
        "green": "B03",
        "blue": "B02",
        "nir_narrow": "B05",
        "swir_1": "B06",
        "swir_2": "B07",
        "Fmask": "Fmask",
    },
    "S30": {
        "red": "B04",
        "green": "B03",
        "blue": "B02",
        "nir_narrow": "B8A",
        "swir_1": "B11",
        "swir_2": "B12",
        "Fmask": "Fmask",
    },
}

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

SR_SCALE = 0.0001
SR_FILL = -9999
QA_FILL = 255


def asset_url(granule: Granule, band: str) -> str:
    band_code = BAND_CODE[granule.satellite][band]
    return f"{granule.path}.{band_code}.tif"


_NEGATIVE_CHECK_BANDS = ("red", "nir_narrow", "blue", "green", "swir_1", "swir_2")


def compute_negative_mask(bands: dict[str, np.ndarray]) -> np.ndarray:
    mask = np.zeros_like(bands["red"], dtype=bool)
    for band in _NEGATIVE_CHECK_BANDS:
        mask |= bands[band] < 0
    return mask


def compute_basic_mask(bands: dict[str, np.ndarray], fmask: np.ndarray) -> np.ndarray:
    cloud = (fmask & (1 << QA_BIT["cloud"])) > 0
    adj_cloud = (fmask & (1 << QA_BIT["adj_cloud"])) > 0
    cloud_shadow = (fmask & (1 << QA_BIT["cloud_shadow"])) > 0
    fill = fmask == QA_FILL
    negative = compute_negative_mask(bands)
    return cloud | adj_cloud | cloud_shadow | fill | negative


def compute_bad_pixel_mask(bands: dict[str, np.ndarray], fmask: np.ndarray) -> np.ndarray:
    basic = compute_basic_mask(bands, fmask)
    is_high_aerosol = ((fmask & (1 << QA_BIT["aerosol_high"])) > 0) & (
        (fmask & (1 << QA_BIT["aerosol_low"])) > 0
    )
    is_low_mod_aerosol = ~is_high_aerosol & ~basic
    any_low_mod_available = np.any(is_low_mod_aerosol, axis=0)
    return basic | (is_high_aerosol & any_low_mod_available)


def compute_all_nan_mask(bad_pixel_mask: np.ndarray) -> np.ndarray:
    return np.all(bad_pixel_mask, axis=0)


def compute_evi2(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    red_r = red.astype(np.float32) * SR_SCALE
    nir_r = nir.astype(np.float32) * SR_SCALE
    return 2.5 * (nir_r - red_r) / (nir_r + 2.4 * red_r + 1)


def select_best_index(
    evi2: np.ndarray, bad_pixel_mask: np.ndarray, all_nan_mask: np.ndarray
) -> np.ndarray:
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
    chosen = np.take_along_axis(values, best_idx[None, :, :], axis=0)[0]
    return np.where(all_nan_mask, nodata, chosen)


def band_std(
    values: np.ndarray, bad_pixel_mask: np.ndarray, all_nan_mask: np.ndarray
) -> np.ndarray:
    values_f = values.astype(np.float32).copy()
    values_f[bad_pixel_mask] = np.nan
    with np.errstate(all="ignore"):
        std = np.nanstd(values_f, axis=0)
    std[all_nan_mask] = 0
    return std


def valid_count(bad_pixel_mask: np.ndarray) -> np.ndarray:
    return np.sum(~bad_pixel_mask, axis=0).astype(np.uint8)


def relative_doy(
    dates: list[date], best_idx: np.ndarray, all_nan_mask: np.ndarray, start_date: date
) -> np.ndarray:
    doy_vals = np.array([d.timetuple().tm_yday for d in dates], dtype=np.int32)
    start_doy = start_date.timetuple().tm_yday
    chosen_doy = doy_vals[best_idx]
    rel = chosen_doy - start_doy + 1
    return np.where(all_nan_mask, 0, rel).astype(np.uint8)


def _default_opener(url: str) -> np.ndarray:
    with rio.open(url) as src:
        return src.read(1)


def read_band_with_retry(
    url: str,
    max_retries: int = 3,
    delay: float = 3.0,
    opener=_default_opener,
) -> np.ndarray:
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
    bands: list[str] = DEFAULT_BANDS,
    opener=_default_opener,
) -> xr.Dataset:
    reflectance_bands = [b for b in bands if b != "Fmask"]

    raw: dict[str, np.ndarray] = {}
    for band in bands:
        urls = [asset_url(g, band) for g in granules]
        raw[band] = np.stack(
            [read_band_with_retry(u, opener=opener) for u in urls], axis=0
        )

    fmask = raw["Fmask"]
    reflectance_arrays = {b: raw[b] for b in reflectance_bands}

    bad_pixel_mask = compute_bad_pixel_mask(reflectance_arrays, fmask)
    all_nan_mask = compute_all_nan_mask(bad_pixel_mask)
    evi2 = compute_evi2(raw["red"], raw["nir_narrow"])
    best_idx = select_best_index(evi2, bad_pixel_mask, all_nan_mask)

    data_vars: dict[str, tuple] = {}
    for band in bands:
        nodata = QA_FILL if band == "Fmask" else SR_FILL
        dtype = np.uint8 if band == "Fmask" else np.int16
        composite = composite_band(raw[band], best_idx, all_nan_mask, nodata).astype(dtype)
        data_vars[band] = (("y", "x"), composite)
        if band != "Fmask":
            std = band_std(raw[band], bad_pixel_mask, all_nan_mask)
            data_vars[f"{band}_std"] = (("y", "x"), np.round(std).astype(np.int16))

    data_vars["ValidCount"] = (("y", "x"), valid_count(bad_pixel_mask))
    dates = [g.date for g in granules]
    data_vars["DOY"] = (("y", "x"), relative_doy(dates, best_idx, all_nan_mask, start_date))

    return xr.Dataset(data_vars)
