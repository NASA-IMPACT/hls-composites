from datetime import date
from datetime import date as date_type

import numpy as np
import pytest

from hls_composites.composite import (
    DEFAULT_BANDS,
    QA_BIT,
    asset_url,
    band_std,
    composite_band,
    compute_all_nan_mask,
    compute_bad_pixel_mask,
    compute_basic_mask,
    compute_evi2,
    compute_negative_mask,
    relative_doy,
    select_best_index,
    valid_count,
)
from hls_composites.models import Granule


def _granule(satellite: str) -> Granule:
    return Granule(
        path="s3://lp-prod-protected/HLSL30.020/HLS.L30.T55HDT.2026151T235621.v2.0/HLS.L30.T55HDT.2026151T235621.v2.0",
        satellite=satellite,
        date=date(2026, 5, 31),
    )


def test_default_bands_matches_prototype():
    assert DEFAULT_BANDS == ["red", "green", "blue", "nir_narrow", "swir_1", "swir_2", "Fmask"]


def test_asset_url_l30_red_band():
    url = asset_url(_granule("L30"), "red")
    assert url == (
        "s3://lp-prod-protected/HLSL30.020/HLS.L30.T55HDT.2026151T235621.v2.0/"
        "HLS.L30.T55HDT.2026151T235621.v2.0.B04.tif"
    )


def test_asset_url_s30_nir_narrow_uses_b8a():
    url = asset_url(_granule("S30"), "nir_narrow")
    assert url.endswith(".B8A.tif")


def test_asset_url_l30_nir_narrow_uses_b05():
    url = asset_url(_granule("L30"), "nir_narrow")
    assert url.endswith(".B05.tif")


def test_asset_url_fmask_same_code_both_satellites():
    assert asset_url(_granule("L30"), "Fmask").endswith(".Fmask.tif")
    assert asset_url(_granule("S30"), "Fmask").endswith(".Fmask.tif")


def _clear_bands(t: int, y: int, x: int) -> dict[str, np.ndarray]:
    return {
        band: np.full((t, y, x), 1000, dtype=np.int16)
        for band in ("red", "green", "blue", "nir_narrow", "swir_1", "swir_2")
    }


def test_negative_mask_flags_any_negative_band():
    bands = _clear_bands(2, 1, 1)
    bands["red"][0, 0, 0] = -100
    mask = compute_negative_mask(bands)
    assert mask[0, 0, 0] == True
    assert mask[1, 0, 0] == False


def test_basic_mask_flags_cloud_bit():
    bands = _clear_bands(1, 1, 1)
    fmask = np.array([[[1 << QA_BIT["cloud"]]]], dtype=np.uint8)
    mask = compute_basic_mask(bands, fmask)
    assert mask[0, 0, 0] == True


def test_basic_mask_flags_fill_value():
    bands = _clear_bands(1, 1, 1)
    fmask = np.array([[[255]]], dtype=np.uint8)  # QA_FILL
    mask = compute_basic_mask(bands, fmask)
    assert mask[0, 0, 0] == True


def test_basic_mask_clear_pixel_not_flagged():
    bands = _clear_bands(1, 1, 1)
    fmask = np.array([[[0]]], dtype=np.uint8)
    mask = compute_basic_mask(bands, fmask)
    assert mask[0, 0, 0] == False


def test_bad_pixel_mask_excludes_high_aerosol_when_alternative_exists():
    # 3 timesteps at one pixel: high aerosol, low/mod aerosol, high aerosol.
    bands = _clear_bands(3, 1, 1)
    high_aerosol_bits = (1 << QA_BIT["aerosol_high"]) | (1 << QA_BIT["aerosol_low"])
    low_mod_bits = 1 << QA_BIT["aerosol_low"]
    fmask = np.array(
        [[[high_aerosol_bits]], [[low_mod_bits]], [[high_aerosol_bits]]], dtype=np.uint8
    )
    mask = compute_bad_pixel_mask(bands, fmask)
    assert mask[0, 0, 0] == True   # high aerosol, alternative exists -> excluded
    assert mask[1, 0, 0] == False  # the low/mod alternative itself -> kept
    assert mask[2, 0, 0] == True


def test_bad_pixel_mask_keeps_high_aerosol_when_no_alternative():
    # All 3 timesteps are high aerosol -- none should be excluded, to avoid a data hole.
    bands = _clear_bands(3, 1, 1)
    high_aerosol_bits = (1 << QA_BIT["aerosol_high"]) | (1 << QA_BIT["aerosol_low"])
    fmask = np.full((3, 1, 1), high_aerosol_bits, dtype=np.uint8)
    mask = compute_bad_pixel_mask(bands, fmask)
    assert mask[0, 0, 0] == False
    assert mask[1, 0, 0] == False
    assert mask[2, 0, 0] == False


def test_all_nan_mask_true_only_when_every_timestep_bad():
    bad_pixel_mask = np.array([[[True]], [[True]]])  # shape (2,1,1), both bad
    assert compute_all_nan_mask(bad_pixel_mask)[0, 0] == True

    bad_pixel_mask = np.array([[[True]], [[False]]])  # one good timestep
    assert compute_all_nan_mask(bad_pixel_mask)[0, 0] == False


def test_compute_evi2_known_value():
    # red=0.1, nir=0.4 (scaled) -> EVI2 = 2.5*(0.4-0.1)/(0.4+2.4*0.1+1) = 0.75/1.64
    red = np.array([[[1000]]], dtype=np.int16)
    nir = np.array([[[4000]]], dtype=np.int16)
    evi2 = compute_evi2(red, nir)
    assert evi2[0, 0, 0] == pytest.approx(0.75 / 1.64, rel=1e-5)


def test_select_best_index_picks_closest_to_median():
    # 3 timesteps, EVI2 values 0.1, 0.5, 0.9 at one pixel -> median is 0.5 -> index 1.
    evi2 = np.array([[[0.1]], [[0.5]], [[0.9]]], dtype=np.float32)
    bad_pixel_mask = np.zeros((3, 1, 1), dtype=bool)
    all_nan_mask = np.zeros((1, 1), dtype=bool)
    idx = select_best_index(evi2, bad_pixel_mask, all_nan_mask)
    assert idx[0, 0] == 1


def test_select_best_index_excludes_masked_observations():
    # Timestep 1 (value 0.5, the true median-closest) is masked out;
    # among the remaining {0.0, 1.0}, median is 0.5, both are equally close
    # (an exact tie in float32 -- unlike 0.1/0.9, which round asymmetrically) -> first wins (index 0).
    evi2 = np.array([[[0.0]], [[0.5]], [[1.0]]], dtype=np.float32)
    bad_pixel_mask = np.array([[[False]], [[True]], [[False]]])
    all_nan_mask = np.zeros((1, 1), dtype=bool)
    idx = select_best_index(evi2, bad_pixel_mask, all_nan_mask)
    assert idx[0, 0] == 0


def test_select_best_index_all_masked_pixel_falls_back_to_zero():
    evi2 = np.array([[[0.1]], [[0.5]]], dtype=np.float32)
    bad_pixel_mask = np.ones((2, 1, 1), dtype=bool)
    all_nan_mask = np.ones((1, 1), dtype=bool)
    idx = select_best_index(evi2, bad_pixel_mask, all_nan_mask)
    assert idx[0, 0] == 0


def test_composite_band_picks_value_at_chosen_index():
    values = np.array([[[10]], [[20]], [[30]]], dtype=np.int16)
    best_idx = np.array([[1]], dtype=np.int16)
    all_nan_mask = np.zeros((1, 1), dtype=bool)
    result = composite_band(values, best_idx, all_nan_mask, nodata=-9999)
    assert result[0, 0] == 20


def test_composite_band_all_masked_pixel_gets_nodata():
    values = np.array([[[10]], [[20]]], dtype=np.int16)
    best_idx = np.array([[0]], dtype=np.int16)
    all_nan_mask = np.ones((1, 1), dtype=bool)
    result = composite_band(values, best_idx, all_nan_mask, nodata=-9999)
    assert result[0, 0] == -9999


def test_band_std_computed_over_unmasked_observations_only():
    values = np.array([[[10]], [[20]], [[999]]], dtype=np.float32)
    bad_pixel_mask = np.array([[[False]], [[False]], [[True]]])  # last one excluded
    all_nan_mask = np.zeros((1, 1), dtype=bool)
    result = band_std(values, bad_pixel_mask, all_nan_mask)
    assert result[0, 0] == pytest.approx(np.std([10, 20]), rel=1e-5)


def test_band_std_all_masked_pixel_is_zero():
    values = np.array([[[10]], [[20]]], dtype=np.float32)
    bad_pixel_mask = np.ones((2, 1, 1), dtype=bool)
    all_nan_mask = np.ones((1, 1), dtype=bool)
    result = band_std(values, bad_pixel_mask, all_nan_mask)
    assert result[0, 0] == 0


def test_valid_count_counts_unmasked_observations():
    bad_pixel_mask = np.array([[[True]], [[False]], [[False]]])
    result = valid_count(bad_pixel_mask)
    assert result[0, 0] == 2


def test_relative_doy_uses_chosen_observations_date():
    dates = [date_type(2020, 1, 1), date_type(2020, 1, 10), date_type(2020, 1, 20)]
    best_idx = np.array([[1]], dtype=np.int16)  # Jan 10
    all_nan_mask = np.zeros((1, 1), dtype=bool)
    result = relative_doy(dates, best_idx, all_nan_mask, start_date=date_type(2020, 1, 1))
    assert result[0, 0] == 10  # Jan 10 is DOY 10, start is DOY 1: 10 - 1 + 1 = 10


def test_relative_doy_all_masked_pixel_is_zero():
    dates = [date_type(2020, 1, 1), date_type(2020, 1, 10)]
    best_idx = np.array([[0]], dtype=np.int16)
    all_nan_mask = np.ones((1, 1), dtype=bool)
    result = relative_doy(dates, best_idx, all_nan_mask, start_date=date_type(2020, 1, 1))
    assert result[0, 0] == 0
