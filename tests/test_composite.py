import dataclasses
from datetime import date
from datetime import date as date_type

import numpy as np
import pytest
import xarray as xr

from hls_composites.bands import (
    BLUE,
    DEFAULT_BANDS,
    FMASK,
    GREEN,
    NIR_NARROW,
    QA_FILL,
    RED,
    REFLECTANCE_BANDS,
    SPEC_BY_BAND,
    SR_FILL,
    SWIR_1,
    SWIR_2,
    Band,
)
from hls_composites.composite import (
    DOY_FILL,
    QA_BIT,
    VALID_COUNT_FILL,
    _composite_block,
    asset_url,
    band_std,
    build_composite,
    composite_band,
    compute_all_nan_mask,
    compute_bad_pixel_mask,
    compute_basic_mask,
    compute_out_of_range_mask,
    observation_doy,
    read_band_with_retry,
    select_best_index,
    to_reflectance,
    valid_count,
)
from hls_composites.indices import (
    ALL_INDICES,
    DEFAULT_INDICES,
    NDVI,
    SELECTION_INDEX,
)
from hls_composites.models import Granule


def _granule(satellite: str) -> Granule:
    return Granule(
        path="s3://lp-prod-protected/HLSL30.020/HLS.L30.T55HDT.2026151T235621.v2.0/HLS.L30.T55HDT.2026151T235621.v2.0",
        satellite=satellite,
        date=date(2026, 5, 31),
    )


def test_default_bands_matches_prototype():
    assert DEFAULT_BANDS == [RED, GREEN, BLUE, NIR_NARROW, SWIR_1, SWIR_2, FMASK]


def test_default_bands_names_match_prototype():
    assert [b.name for b in DEFAULT_BANDS] == [
        "red",
        "green",
        "blue",
        "nir_narrow",
        "swir_1",
        "swir_2",
        "Fmask",
    ]


def test_asset_url_l30_red_band():
    url = asset_url(_granule("L30"), RED)
    assert url == (
        "s3://lp-prod-protected/HLSL30.020/HLS.L30.T55HDT.2026151T235621.v2.0/"
        "HLS.L30.T55HDT.2026151T235621.v2.0.B04.tif"
    )


def test_asset_url_s30_nir_narrow_uses_b8a():
    url = asset_url(_granule("S30"), NIR_NARROW)
    assert url.endswith(".B8A.tif")


def test_asset_url_l30_nir_narrow_uses_b05():
    url = asset_url(_granule("L30"), NIR_NARROW)
    assert url.endswith(".B05.tif")


def test_asset_url_fmask_same_code_both_satellites():
    assert asset_url(_granule("L30"), FMASK).endswith(".Fmask.tif")
    assert asset_url(_granule("S30"), FMASK).endswith(".Fmask.tif")


def test_reflectance_bands_have_sr_fill_and_int16():
    for band in (RED, GREEN, BLUE, NIR_NARROW, SWIR_1, SWIR_2):
        assert band.is_reflectance is True
        assert band.nodata == SR_FILL
        assert band.dtype == np.int16


def test_reflectance_bands_map_to_spectral_index_bands():
    assert RED.index_band is Band.R
    assert GREEN.index_band is Band.G
    assert BLUE.index_band is Band.B
    assert NIR_NARROW.index_band is Band.NIR
    assert SWIR_1.index_band is Band.SWIR1
    assert SWIR_2.index_band is Band.SWIR2


def test_reflectance_bands_excludes_fmask():
    assert REFLECTANCE_BANDS == [RED, GREEN, BLUE, NIR_NARROW, SWIR_1, SWIR_2]
    assert FMASK not in REFLECTANCE_BANDS


def test_spec_by_band_reverse_lookup():
    assert SPEC_BY_BAND[Band.R] is RED
    assert SPEC_BY_BAND[Band.NIR] is NIR_NARROW
    assert set(SPEC_BY_BAND) == {
        Band.B,
        Band.G,
        Band.R,
        Band.NIR,
        Band.SWIR1,
        Band.SWIR2,
    }


def test_fmask_band_has_qa_fill_and_uint8():
    assert FMASK.is_reflectance is False
    assert FMASK.index_band is None
    assert FMASK.nodata == QA_FILL
    assert FMASK.dtype == np.uint8


def _clear_bands(t: int, y: int, x: int) -> dict:
    return {
        band: np.full((t, y, x), 1000, dtype=np.int16)
        for band in (RED, GREEN, BLUE, NIR_NARROW, SWIR_1, SWIR_2)
    }


def test_out_of_range_mask_flags_any_negative_band():
    bands = _clear_bands(2, 1, 1)
    bands[RED][0, 0, 0] = -100
    mask = compute_out_of_range_mask(bands)
    assert mask[0, 0, 0] == True
    assert mask[1, 0, 0] == False


def test_out_of_range_mask_skips_bands_with_no_valid_range():
    # FMASK has valid_range=(None, None) -- a negative value there must not be flagged.
    bands = {
        RED: np.full((1, 1, 1), 1000, dtype=np.int16),
        FMASK: np.full((1, 1, 1), -1, dtype=np.int16),
    }
    mask = compute_out_of_range_mask(bands)
    assert mask[0, 0, 0] == False


def test_out_of_range_mask_flags_values_above_max():
    values = np.array([[[500]], [[2000]]], dtype=np.int16)

    # RED's default valid_range is (0, None) -- no upper bound, so 2000 passes.
    assert compute_out_of_range_mask({RED: values})[0, 0, 0] == False
    assert compute_out_of_range_mask({RED: values})[1, 0, 0] == False

    # A band with an explicit upper bound flags values above it.
    capped_band = dataclasses.replace(RED, valid_range=(0, 1000))
    mask = compute_out_of_range_mask({capped_band: values})
    assert mask[0, 0, 0] == False
    assert mask[1, 0, 0] == True


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
    assert mask[0, 0, 0] == True  # high aerosol, alternative exists -> excluded
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


def test_selection_index_known_value():
    # red=0.1, nir=0.4 (scaled) -> EVI2 = 2.5*(0.4-0.1)/(0.4+2.4*0.1+1) = 0.75/1.64
    digital_numbers = {
        RED: np.array([[[1000]]], dtype=np.int16),
        NIR_NARROW: np.array([[[4000]]], dtype=np.int16),
    }
    evi2 = SELECTION_INDEX(to_reflectance(digital_numbers, SELECTION_INDEX.bands))
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


def test_observation_doy_uses_chosen_observations_date():
    dates = [date_type(2020, 1, 1), date_type(2020, 1, 10), date_type(2020, 1, 20)]
    best_idx = np.array([[1]], dtype=np.int16)  # Jan 10
    all_nan_mask = np.zeros((1, 1), dtype=bool)
    result = observation_doy(dates, best_idx, all_nan_mask)
    assert result[0, 0] == 10  # Jan 10 is DOY 10


def test_read_band_with_retry_succeeds_first_try():
    calls = []

    def opener(url):
        calls.append(url)
        return np.array([[1, 2], [3, 4]])

    result = read_band_with_retry("s3://x/y.tif", opener=opener)
    assert result.tolist() == [[1, 2], [3, 4]]
    assert calls == ["s3://x/y.tif"]


def test_read_band_with_retry_succeeds_after_transient_failures(monkeypatch):
    monkeypatch.setattr("hls_composites.composite.time.sleep", lambda _: None)
    attempts = {"count": 0}

    def opener(url):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OSError("transient")
        return np.array([[1]])

    result = read_band_with_retry("s3://x/y.tif", max_retries=3, opener=opener)
    assert result.tolist() == [[1]]
    assert attempts["count"] == 3


def test_read_band_with_retry_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr("hls_composites.composite.time.sleep", lambda _: None)

    def opener(url):
        raise OSError("permanent failure")

    try:
        read_band_with_retry("s3://x/y.tif", max_retries=2, opener=opener)
        assert False, "expected IOError"
    except OSError as e:
        assert "permanent failure" in str(e)


def test_build_composite_lazy_reader_matches_block_kernel():
    # The lazy rioxarray + xr.map_blocks path must produce exactly what the
    # direct numpy kernel produces on the same stack. Reuse the block fixture
    # and feed each (granule, band) as a lazy DataArray via a synthetic opener.
    reflectance, fmask, dates = _block_fixture()
    granules = [
        Granule(path=f"s3://bucket/g{i}", satellite="L30", date=d)
        for i, d in enumerate(dates)
    ]

    band_data: dict[str, np.ndarray] = {}
    for i, granule in enumerate(granules):
        for spec in DEFAULT_BANDS:
            arr = fmask[i] if spec is FMASK else reflectance[spec][i]
            band_data[asset_url(granule, spec)] = arr

    def fake_opener(url: str) -> xr.DataArray:
        return xr.DataArray(band_data[url], dims=("y", "x"))

    result = build_composite(granules, opener=fake_opener).compute()

    expected = _composite_block(reflectance, fmask, dates, DEFAULT_INDICES)
    for name, arr in expected.items():
        np.testing.assert_array_equal(result[name].values, arr)

    # Spot checks matching the kernel test's fixture semantics.
    assert result["NDVI"].values[0, 0] == 6000
    assert result["ValidCount"].values[1, 0] == VALID_COUNT_FILL
    assert "NDVI_std" in result.data_vars
    assert "Fmask" not in result.data_vars  # output is index-based, not reflectance
    assert "SAVI" not in result.data_vars  # not in the default index set


def test_build_composite_raises_on_empty_granule_list():
    with pytest.raises(ValueError, match="at least one granule"):
        build_composite([])


def _block_reflectance(red: np.ndarray, nir: np.ndarray) -> dict:
    """Build a (T, 2, 2) reflectance stack varying only red/nir; others clear."""
    t = red.shape[0]
    stack = {b: np.full((t, 2, 2), 1000, dtype=np.int16) for b in REFLECTANCE_BANDS}
    stack[RED] = red
    stack[NIR_NARROW] = nir
    return stack


def _block_fixture():
    # T=3, 2x2 tile. red fixed (0.10), nir rises 0.20/0.40/0.60 so EVI2 is strictly
    # increasing and the per-pixel median lands unambiguously on the middle timestep.
    # Pixel (0,0)/(1,1) clear at all three; (0,1) valid only at t2; (1,0) cloudy at all.
    cloud = 1 << QA_BIT["cloud"]
    red = np.full((3, 2, 2), 1000, dtype=np.int16)
    nir = np.stack(
        [
            np.full((2, 2), 2000, dtype=np.int16),
            np.full((2, 2), 4000, dtype=np.int16),
            np.full((2, 2), 6000, dtype=np.int16),
        ]
    )
    fmask = np.array(
        [
            [[0, cloud], [cloud, 0]],
            [[0, cloud], [cloud, 0]],
            [[0, 0], [cloud, 0]],
        ],
        dtype=np.uint8,
    )
    reflectance = _block_reflectance(red, nir)
    dates = [date(2020, 1, 5), date(2020, 1, 15), date(2020, 1, 25)]
    return reflectance, fmask, dates


def test_composite_block_ndvi_value_std_and_aux():
    reflectance, fmask, dates = _block_fixture()
    out = _composite_block(reflectance, fmask, dates, indices=[NDVI()])

    # Pixel (0,0): clear at all 3 -> median-EVI2 selects the middle timestep (t1) ->
    # NDVI of t1: (0.40-0.10)/(0.40+0.10) = 0.6 -> encoded 6000.
    assert out["NDVI"][0, 0] == 6000
    assert out["ValidCount"][0, 0] == 3
    # DOY of chosen obs (t1 = Jan 15), relative to Jan 1: 15 - 1 + 1 = 15.
    assert out["DOY"][0, 0] == 15

    # Pixel (0,1): only t2 valid -> NDVI of t2: (0.60-0.10)/(0.60+0.10).
    t2 = (0.60 - 0.10) / (0.60 + 0.10)
    assert out["NDVI"][0, 1] == round(t2 / 1e-4)
    assert out["ValidCount"][0, 1] == 1
    assert out["DOY"][0, 1] == 25

    # Pixel (1,0): cloudy at every timestep -> fill everywhere.
    assert out["NDVI"][1, 0] == NDVI.fill_value
    assert out["NDVI_std"][1, 0] == NDVI.fill_value
    assert out["ValidCount"][1, 0] == VALID_COUNT_FILL
    assert out["DOY"][1, 0] == DOY_FILL

    # NDVI_std at (0,0): std across all 3 timesteps' NDVI.
    ndvi_t = [
        (0.20 - 0.10) / (0.20 + 0.10),
        (0.40 - 0.10) / (0.40 + 0.10),
        (0.60 - 0.10) / (0.60 + 0.10),
    ]
    expected_std = round(float(np.std(ndvi_t)) / 1e-4)
    assert out["NDVI_std"][0, 0] == expected_std


def test_composite_block_defaults_to_the_default_indices_and_aux():
    reflectance, fmask, dates = _block_fixture()
    out = _composite_block(reflectance, fmask, dates)

    assert list(out) == [
        "EVI",
        "EVI_std",
        "NBR",
        "NBR_std",
        "NDVI",
        "NDVI_std",
        "ValidCount",
        "DOY",
    ]
    for index in DEFAULT_INDICES:
        assert out[index.name].shape == (2, 2)
        assert out[index.name].dtype == np.int16
        assert out[f"{index.name}_std"].dtype == np.int16
    assert out["ValidCount"].dtype == np.uint8
    assert out["DOY"].dtype == np.int16


def test_composite_block_emits_every_index_when_asked():
    reflectance, fmask, dates = _block_fixture()
    out = _composite_block(reflectance, fmask, dates, indices=ALL_INDICES)
    for index in ALL_INDICES:
        assert index.name in out
        assert f"{index.name}_std" in out
        assert out[index.name].dtype == np.int16


def test_composite_block_bands_output_emits_reflectance_values_and_std():
    reflectance, fmask, dates = _block_fixture()
    out = _composite_block(reflectance, fmask, dates, output="bands")

    assert list(out) == [
        "red",
        "red_std",
        "green",
        "green_std",
        "blue",
        "blue_std",
        "nir_narrow",
        "nir_narrow_std",
        "swir_1",
        "swir_1_std",
        "swir_2",
        "swir_2_std",
        "ValidCount",
        "DOY",
    ]
    for spec in REFLECTANCE_BANDS:
        assert out[spec.name].dtype == np.int16
        assert out[f"{spec.name}_std"].dtype == np.int16
    assert out["ValidCount"].dtype == np.uint8
    assert out["DOY"].dtype == np.int16


def test_composite_block_bands_output_has_no_indices_or_qa():
    reflectance, fmask, dates = _block_fixture()
    out = _composite_block(reflectance, fmask, dates, output="bands")

    for index in ALL_INDICES:
        assert index.name not in out
    assert FMASK.name not in out


def test_composite_block_bands_values_taken_at_selected_timestep():
    reflectance, fmask, dates = _block_fixture()
    out = _composite_block(reflectance, fmask, dates, output="bands")

    # Pixel (0,0): clear at all 3 -> median-EVI2 selects t1, whose nir is 4000.
    assert out["nir_narrow"][0, 0] == 4000
    # Pixel (0,1): only t2 valid -> its nir, 6000.
    assert out["nir_narrow"][0, 1] == 6000
    # red is constant across the stack, so selection cannot change it.
    assert out["red"][0, 0] == 1000


def test_composite_block_bands_all_masked_pixel_gets_band_fill():
    reflectance, fmask, dates = _block_fixture()
    out = _composite_block(reflectance, fmask, dates, output="bands")

    # Pixel (1,0) is cloudy at every timestep.
    for spec in REFLECTANCE_BANDS:
        assert out[spec.name][1, 0] == SR_FILL
        assert out[f"{spec.name}_std"][1, 0] == SR_FILL
    assert out["ValidCount"][1, 0] == VALID_COUNT_FILL
    assert out["DOY"][1, 0] == DOY_FILL


def test_composite_block_bands_std_is_digital_number_std_rounded():
    reflectance, fmask, dates = _block_fixture()
    out = _composite_block(reflectance, fmask, dates, output="bands")

    # Pixel (0,0): all 3 timesteps valid; nir is 2000/4000/6000 raw DN. The std
    # is stored in the band's own encoding, so it is not rescaled.
    assert out["nir_narrow"][0, 0] == 4000
    assert out["nir_narrow_std"][0, 0] == round(float(np.std([2000, 4000, 6000])))
    # Constant band -> zero spread.
    assert out["red_std"][0, 0] == 0


def test_build_composite_bands_output_matches_block_kernel():
    reflectance, fmask, dates = _block_fixture()
    granules = [
        Granule(path=f"s3://bucket/g{i}", satellite="L30", date=d)
        for i, d in enumerate(dates)
    ]

    band_data: dict[str, np.ndarray] = {}
    for i, granule in enumerate(granules):
        for spec in DEFAULT_BANDS:
            arr = fmask[i] if spec is FMASK else reflectance[spec][i]
            band_data[asset_url(granule, spec)] = arr

    def fake_opener(url: str) -> xr.DataArray:
        return xr.DataArray(band_data[url], dims=("y", "x"))

    result = build_composite(granules, output="bands", opener=fake_opener).compute()

    expected = _composite_block(reflectance, fmask, dates, output="bands")
    for name, arr in expected.items():
        np.testing.assert_array_equal(result[name].values, arr)

    assert "NDVI" not in result.data_vars
    assert FMASK.name not in result.data_vars


def test_build_composite_bands_output_carries_band_encoding_attrs():
    reflectance, fmask, dates = _block_fixture()
    granules = [
        Granule(path=f"s3://bucket/g{i}", satellite="L30", date=d)
        for i, d in enumerate(dates)
    ]
    band_data = {
        asset_url(g, spec): (fmask[i] if spec is FMASK else reflectance[spec][i])
        for i, g in enumerate(granules)
        for spec in DEFAULT_BANDS
    }
    result = build_composite(
        granules,
        output="bands",
        opener=lambda url: xr.DataArray(band_data[url], dims=("y", "x")),
    )

    for spec in REFLECTANCE_BANDS:
        for name in (spec.name, f"{spec.name}_std"):
            assert result[name].attrs["nodata"] == spec.nodata
            assert result[name].attrs["scale_factor"] == spec.scale


def test_build_composite_honours_an_explicit_index_list():
    reflectance, fmask, dates = _block_fixture()
    granules = [
        Granule(path=f"s3://bucket/g{i}", satellite="L30", date=d)
        for i, d in enumerate(dates)
    ]
    band_data = {
        asset_url(g, spec): (fmask[i] if spec is FMASK else reflectance[spec][i])
        for i, g in enumerate(granules)
        for spec in DEFAULT_BANDS
    }
    result = build_composite(
        granules,
        indices=ALL_INDICES,
        opener=lambda url: xr.DataArray(band_data[url], dims=("y", "x")),
    )

    for index in ALL_INDICES:
        assert index.name in result.data_vars
        assert f"{index.name}_std" in result.data_vars


def test_observation_doy_is_absolute_julian_day_in_int16():
    # Late in a leap year the DOY exceeds what a uint8 could hold.
    dates = [date(2020, 7, 1), date(2020, 12, 31)]
    best_idx = np.array([[0, 1]], dtype=np.int16)
    all_nan = np.zeros((1, 2), dtype=bool)

    out = observation_doy(dates, best_idx, all_nan)

    assert out.dtype == np.int16
    assert out[0, 0] == 183
    assert out[0, 1] == 366


def test_observation_doy_fills_pixels_with_no_valid_observation():
    dates = [date(2020, 7, 1), date(2020, 7, 31)]
    best_idx = np.zeros((1, 2), dtype=np.int16)
    all_nan = np.array([[False, True]])

    out = observation_doy(dates, best_idx, all_nan)

    assert out[0, 0] == 183
    assert out[0, 1] == DOY_FILL


def test_observation_doy_fill_cannot_collide_with_a_real_day():
    # Julian days are 1..366, so a negative fill is unreachable by construction.
    assert DOY_FILL < 1


def test_valid_count_fills_pixels_with_no_valid_observation():
    # Column 0 has two usable observations, column 1 has none.
    bad = np.array([[[False, True]], [[False, True]]])

    out = valid_count(bad)

    assert out.dtype == np.uint8
    assert out[0, 0] == 2
    assert out[0, 1] == VALID_COUNT_FILL


def _lazy_composite(output: str):
    reflectance, fmask, dates = _block_fixture()
    granules = [
        Granule(path=f"s3://bucket/g{i}", satellite="L30", date=d)
        for i, d in enumerate(dates)
    ]
    band_data = {
        asset_url(g, spec): (fmask[i] if spec is FMASK else reflectance[spec][i])
        for i, g in enumerate(granules)
        for spec in DEFAULT_BANDS
    }
    return build_composite(
        granules,
        output=output,
        opener=lambda url: xr.DataArray(band_data[url], dims=("y", "x")),
    )


@pytest.mark.parametrize("output", ["indexes", "bands"])
@pytest.mark.parametrize(
    ("name", "fill", "dtype"),
    [("DOY", DOY_FILL, np.int16), ("ValidCount", VALID_COUNT_FILL, np.uint8)],
)
def test_build_composite_aux_layers_declare_their_fill_value(output, name, fill, dtype):
    result = _lazy_composite(output)

    assert result[name].dtype == dtype
    assert result[name].attrs["nodata"] == fill
    # Pixel (1,0) is cloudy at every timestep, so both aux layers are filled.
    assert result[name].compute().values[1, 0] == fill


@pytest.mark.parametrize("output", ["indexes", "bands"])
def test_build_composite_aux_layers_keep_real_values_where_valid(output):
    result = _lazy_composite(output).compute()

    # Pixel (0,0) is clear at all 3 timesteps; (0,1) only at the last.
    assert result["ValidCount"].values[0, 0] == 3
    assert result["ValidCount"].values[0, 1] == 1
    assert result["DOY"].values[0, 0] == 15
    assert result["DOY"].values[0, 1] == 25


def _two_obs_evi2(a: float, b: float) -> np.ndarray:
    """A 2-timestep, 1-pixel EVI2 stack with the given values."""
    return np.array([[[a]], [[b]]], dtype=np.float32)


def test_select_best_index_even_stack_median_is_a_tie():
    # np.nanmedian of an even stack is the mean of the two middle values, which
    # is no observation's EVI2. Both are exactly equidistant from it, so the
    # result is decided entirely by stack position -- not by the data.
    evi2 = _two_obs_evi2(0.30, 0.50)
    target = np.nanmedian(evi2, axis=0)
    assert abs(evi2[0] - target) == abs(evi2[1] - target)


def test_select_best_index_breaks_even_stack_ties_toward_the_earlier_observation():
    # Granules arrive in chronological order (see scan_bucket_for_granules), so
    # the lowest index is the earliest observation. Reversing the stack must
    # therefore flip the choice -- this is what made a reverse-ordered run
    # select a different date from the same pixels.
    bad = np.zeros((2, 1, 1), dtype=bool)
    all_nan = np.zeros((1, 1), dtype=bool)

    forward = select_best_index(_two_obs_evi2(0.30, 0.50), bad, all_nan)
    reversed_ = select_best_index(_two_obs_evi2(0.50, 0.30), bad, all_nan)

    assert forward[0, 0] == 0
    assert reversed_[0, 0] == 0


def test_select_best_index_odd_stack_has_no_tie_and_ignores_order():
    # An odd stack has a real median observation, so the choice is the same
    # whichever order the granules arrive in.
    bad = np.zeros((3, 1, 1), dtype=bool)
    all_nan = np.zeros((1, 1), dtype=bool)
    forward = np.array([[[0.10]], [[0.50]], [[0.90]]], dtype=np.float32)
    backward = forward[::-1].copy()

    assert select_best_index(forward, bad, all_nan)[0, 0] == 1
    assert select_best_index(backward, bad, all_nan)[0, 0] == 1


def test_build_composite_selection_is_stable_under_granule_reordering_for_odd_stacks():
    # Three granules -> odd stack -> no tie, so the composite is identical
    # whatever order build_composite receives them in.
    reflectance, fmask, dates = _block_fixture()
    granules = [
        Granule(path=f"s3://bucket/g{i}", satellite="L30", date=d)
        for i, d in enumerate(dates)
    ]
    band_data = {
        asset_url(g, spec): (fmask[i] if spec is FMASK else reflectance[spec][i])
        for i, g in enumerate(granules)
        for spec in DEFAULT_BANDS
    }
    opener = lambda url: xr.DataArray(band_data[url], dims=("y", "x"))

    forward = build_composite(granules, opener=opener).compute()
    shuffled = build_composite(granules[::-1], opener=opener).compute()

    np.testing.assert_array_equal(forward["DOY"].values, shuffled["DOY"].values)
    np.testing.assert_array_equal(forward["NDVI"].values, shuffled["NDVI"].values)
