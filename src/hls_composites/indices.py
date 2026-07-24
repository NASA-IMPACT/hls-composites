"""Spectral index definitions as tiny, band-declaring callables.

Each index is a class declaring its required bands as a ``ClassVar`` and
computing the raw (physical-unit) index in ``__call__`` from a mapping of
``Band`` to array. Inputs are physical reflectance (0-1), not raw digital
numbers -- indices with additive constants (EVI, MSAVI, SAVI, TVI) are only
correct in reflectance units. Scaling/clipping to the int16 storage encoding
is deliberately kept out of ``__call__`` so the same float result can feed both
the composite value and its temporal standard deviation.
"""

from collections.abc import Mapping
from enum import Enum, unique
from typing import ClassVar

import numpy as np

BandData = Mapping["Band", np.ndarray]


@unique
class Band(Enum):
    B = "B"
    G = "G"
    R = "R"
    NIR = "NIR"
    SWIR1 = "SWIR1"
    SWIR2 = "SWIR2"


class Index:
    """Base class: a spectral index with its band requirements and encoding.

    Subclasses set the ``bands`` ClassVar and implement ``__call__``. The
    encoding attributes (`scale_factor`, `fill_value`) describe how the raw
    float index maps onto its int16 on-disk representation; encoding itself is
    applied by the writer, not here.
    """

    bands: ClassVar[tuple[Band, ...]]
    long_name: ClassVar[str]
    scale_factor: ClassVar[float] = 1e-4
    fill_value: ClassVar[int] = -19999

    @property
    def name(self) -> str:
        return type(self).__name__

    def __call__(self, data: BandData) -> np.ndarray:
        raise NotImplementedError


class EVI(Index):
    bands = (Band.B, Band.R, Band.NIR)
    long_name = "Enhanced Vegetation Index"

    def __call__(self, data: BandData) -> np.ndarray:
        b, r, nir = data[Band.B], data[Band.R], data[Band.NIR]
        return 2.5 * (nir - r) / (nir + 6 * r - 7.5 * b + 1)


class MSAVI(Index):
    bands = (Band.R, Band.NIR)
    long_name = "Modified Soil-Adjusted Vegetation Index"

    def __call__(self, data: BandData) -> np.ndarray:
        r, nir = data[Band.R], data[Band.NIR]
        sqrt_term = (2 * nir + 1) ** 2 - 8 * (nir - r)
        return np.where(
            sqrt_term >= 0,
            (2 * nir + 1 - np.sqrt(np.where(sqrt_term >= 0, sqrt_term, np.nan))) / 2,
            np.nan,
        )


class NBR(Index):
    bands = (Band.NIR, Band.SWIR2)
    long_name = "Normalized Burn Ratio"

    def __call__(self, data: BandData) -> np.ndarray:
        nir, swir2 = data[Band.NIR], data[Band.SWIR2]
        return (nir - swir2) / (nir + swir2)


class NBR2(Index):
    bands = (Band.SWIR1, Band.SWIR2)
    long_name = "Normalized Burn Ratio 2"

    def __call__(self, data: BandData) -> np.ndarray:
        swir1, swir2 = data[Band.SWIR1], data[Band.SWIR2]
        return (swir1 - swir2) / (swir1 + swir2)


class NDMI(Index):
    bands = (Band.NIR, Band.SWIR1)
    long_name = "Normalized Difference Moisture Index"

    def __call__(self, data: BandData) -> np.ndarray:
        nir, swir1 = data[Band.NIR], data[Band.SWIR1]
        return (nir - swir1) / (nir + swir1)


class NDVI(Index):
    bands = (Band.R, Band.NIR)
    long_name = "Normalized Difference Vegetation Index"

    def __call__(self, data: BandData) -> np.ndarray:
        r, nir = data[Band.R], data[Band.NIR]
        return (nir - r) / (nir + r)


class NDWI(Index):
    bands = (Band.G, Band.NIR)
    long_name = "Normalized Difference Water Index"

    def __call__(self, data: BandData) -> np.ndarray:
        g, nir = data[Band.G], data[Band.NIR]
        return (g - nir) / (g + nir)


class SAVI(Index):
    bands = (Band.R, Band.NIR)
    long_name = "Soil-Adjusted Vegetation Index"

    def __call__(self, data: BandData) -> np.ndarray:
        r, nir = data[Band.R], data[Band.NIR]
        return 1.5 * (nir - r) / (nir + r + 0.5)


class TVI(Index):
    bands = (Band.G, Band.R, Band.NIR)
    long_name = "Triangular Vegetation Index"
    scale_factor = 1e-2

    def __call__(self, data: BandData) -> np.ndarray:
        g, r, nir = data[Band.G], data[Band.R], data[Band.NIR]
        return (120 * (nir - g) - 200 * (r - g)) / 2


ALL_INDICES: list[Index] = [
    EVI(),
    MSAVI(),
    NBR(),
    NBR2(),
    NDMI(),
    NDVI(),
    NDWI(),
    SAVI(),
    TVI(),
]
