"""Spectral index definitions as tiny, band-declaring callables.

Each index is a class declaring its required bands as a ``ClassVar`` and
computing the raw (physical-unit) index in ``__call__`` from a mapping of
``Band`` to array.

Inputs are physical reflectance (0-1), not raw digital numbers, since indices with
additive constants (e.g., EVI) are only correct in reflectance units.
Scaling/clipping to the int16 storage encoding is deliberately kept out of
``__call__`` so the same float result can feed both the composite value
and its temporal standard deviation. Each index declares the physical range
that encoding clips to as ``valid_min``/``valid_max``.
"""

from collections.abc import Mapping
from typing import ClassVar

import numpy as np

from hls_composites.bands import Band

BandData = Mapping[Band, np.ndarray]


class Index:
    """Base class: a spectral index with its band requirements and encoding.

    `valid_min`/`valid_max` bound the index in physical units. Encoding clips to
    them, so a raw value from an ill-conditioned denominator cannot overflow the
    int16 storage encoding. They are declared per index rather than shared,
    since a range that happens to coincide today need not tomorrow.
    """

    bands: ClassVar[tuple[Band, ...]]
    long_name: ClassVar[str]
    scale_factor: ClassVar[float] = 1e-4
    fill_value: ClassVar[int] = -19999
    valid_min: ClassVar[float]
    valid_max: ClassVar[float]

    @property
    def name(self) -> str:
        return type(self).__name__

    def __call__(self, data: BandData) -> np.ndarray:
        raise NotImplementedError


class EVI(Index):
    bands = (Band.B, Band.R, Band.NIR)
    long_name = "Enhanced Vegetation Index"
    # Unbounded in principle: the denominator approaches zero for some
    # reflectance combinations. Bounded here to the physically meaningful range.
    valid_min = -1.0
    valid_max = 1.0

    def __call__(self, data: BandData) -> np.ndarray:
        b, r, nir = data[Band.B], data[Band.R], data[Band.NIR]
        return 2.5 * (nir - r) / (nir + 6 * r - 7.5 * b + 1)


class EVI2(Index):
    bands = (Band.R, Band.NIR)
    long_name = "Two-band Enhanced Vegetation Index"
    valid_min = -1.0
    valid_max = 1.0

    def __call__(self, data: BandData) -> np.ndarray:
        r, nir = data[Band.R], data[Band.NIR]
        return 2.5 * (nir - r) / (nir + 2.4 * r + 1)


class NBR(Index):
    bands = (Band.NIR, Band.SWIR2)
    long_name = "Normalized Burn Ratio"
    valid_min = -1.0
    valid_max = 1.0

    def __call__(self, data: BandData) -> np.ndarray:
        nir, swir2 = data[Band.NIR], data[Band.SWIR2]
        return (nir - swir2) / (nir + swir2)


class NDVI(Index):
    bands = (Band.R, Band.NIR)
    long_name = "Normalized Difference Vegetation Index"
    valid_min = -1.0
    valid_max = 1.0

    def __call__(self, data: BandData) -> np.ndarray:
        r, nir = data[Band.R], data[Band.NIR]
        return (nir - r) / (nir + r)


ALL_INDICES: list[Index] = [
    EVI(),
    EVI2(),
    NBR(),
    NDVI(),
]
"""Every index defined here, whether or not the composite algorithm emits it."""

SELECTION_INDEX: Index = EVI2()
"""The index whose per-pixel median drives observation selection."""

DEFAULT_INDICES: list[Index] = [EVI(), NBR(), NDVI()]
"""The indices the composite algorithm emits unless told otherwise."""
