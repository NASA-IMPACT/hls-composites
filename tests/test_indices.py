import numpy as np
import pytest

from hls_composites.bands import Band
from hls_composites.indices import EVI, NBR, NDVI


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


def test_ndvi_value():
    assert NDVI()(_refl())[0] == pytest.approx(0.6, rel=1e-5)


def test_evi_value():
    assert EVI()(_refl())[0] == pytest.approx(0.75 / 1.625, rel=1e-5)


def test_nbr_value():
    assert NBR()(_refl())[0] == pytest.approx(0.2 / 0.6, rel=1e-5)
