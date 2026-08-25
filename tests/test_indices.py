import numpy as np
import pytest

from hls_composites.bands import Band
from hls_composites.indices import (
    ALL_INDICES,
    EVI,
    NBR,
    NDVI,
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


def test_each_index_declares_band_requirements_as_classvar():
    for ix in ALL_INDICES:
        assert isinstance(type(ix).bands, tuple)
        assert len(type(ix).bands) >= 2
        assert all(isinstance(b, Band) for b in type(ix).bands)


def test_default_scale_and_fill():
    assert NDVI.scale_factor == pytest.approx(1e-4)
    assert NDVI.fill_value == -19999


def test_ndvi_value():
    assert NDVI()(_refl())[0] == pytest.approx(0.6, rel=1e-5)


def test_evi_value():
    assert EVI()(_refl())[0] == pytest.approx(0.75 / 1.625, rel=1e-5)


def test_nbr_value():
    assert NBR()(_refl())[0] == pytest.approx(0.2 / 0.6, rel=1e-5)
