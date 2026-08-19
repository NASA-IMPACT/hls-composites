import numpy as np
import pytest

from hls_composites.bands import Band
from hls_composites.indices import (
    ALL_INDICES,
    DEFAULT_INDICES,
    EVI,
    MSAVI,
    NBR,
    NBR2,
    NDMI,
    NDVI,
    NDWI,
    SAVI,
    TVI,
)


def _refl() -> dict[Band, np.ndarray]:
    """Physical-reflectance (0-1) inputs, one pixel, hand-computed downstream."""
    return {
        Band.B: np.array([0.05], dtype=np.float32),
        Band.G: np.array([0.15], dtype=np.float32),
        Band.R: np.array([0.10], dtype=np.float32),
        Band.NIR: np.array([0.40], dtype=np.float32),
        Band.SWIR1: np.array([0.30], dtype=np.float32),
        Band.SWIR2: np.array([0.20], dtype=np.float32),
    }


def test_default_indices_are_evi_nbr_ndvi():
    assert [type(ix).__name__ for ix in DEFAULT_INDICES] == ["EVI", "NBR", "NDVI"]


def test_default_indices_are_a_subset_of_the_registry():
    registry = {type(ix).__name__ for ix in ALL_INDICES}
    assert {type(ix).__name__ for ix in DEFAULT_INDICES} <= registry


def test_all_indices_registry_has_nine():
    assert len(ALL_INDICES) == 9
    names = {type(ix).__name__ for ix in ALL_INDICES}
    assert names == {
        "EVI",
        "MSAVI",
        "NBR",
        "NBR2",
        "NDMI",
        "NDVI",
        "NDWI",
        "SAVI",
        "TVI",
    }


def test_each_index_declares_band_requirements_as_classvar():
    for ix in ALL_INDICES:
        assert isinstance(type(ix).bands, tuple)
        assert len(type(ix).bands) >= 2
        assert all(isinstance(b, Band) for b in type(ix).bands)


def test_default_scale_and_fill():
    assert NDVI.scale_factor == pytest.approx(1e-4)
    assert NDVI.fill_value == -19999
    # TVI overrides the scale factor.
    assert TVI.scale_factor == pytest.approx(1e-2)


def test_ndvi_value():
    assert NDVI()(_refl())[0] == pytest.approx(0.6, rel=1e-5)


def test_evi_value():
    assert EVI()(_refl())[0] == pytest.approx(0.75 / 1.625, rel=1e-5)


def test_msavi_value():
    assert MSAVI()(_refl())[0] == pytest.approx((1.8 - np.sqrt(0.84)) / 2, rel=1e-5)


def test_msavi_negative_sqrt_term_is_nan():
    # sqrt_term = (2*nir+1)^2 - 8*(nir-r) = (2*nir-1)^2 + 8*r, which is >= 0 for all
    # physical reflectance (r >= 0). It only goes negative for out-of-range r < 0.
    # nir=0.5, r=-0.1 -> (0)^2 + 8*(-0.1) = -0.8 < 0 -> nan.
    data = {
        Band.R: np.array([-0.1], dtype=np.float32),
        Band.NIR: np.array([0.5], dtype=np.float32),
    }
    assert np.isnan(MSAVI()(data)[0])


def test_nbr_value():
    assert NBR()(_refl())[0] == pytest.approx(0.2 / 0.6, rel=1e-5)


def test_nbr2_value():
    assert NBR2()(_refl())[0] == pytest.approx(0.2, rel=1e-5)


def test_ndmi_value():
    assert NDMI()(_refl())[0] == pytest.approx(0.1 / 0.7, rel=1e-5)


def test_ndwi_value():
    assert NDWI()(_refl())[0] == pytest.approx(-0.25 / 0.55, rel=1e-5)


def test_savi_value():
    assert SAVI()(_refl())[0] == pytest.approx(0.45, rel=1e-5)


def test_tvi_value():
    assert TVI()(_refl())[0] == pytest.approx(20.0, rel=1e-5)
