"""Spectral bands and HLS asset band specifications.

- `Band` is the instrument-agnostic spectral band vocabulary (what a wavelength *is*)
- `BandSpec` describes one HLS asset (how to read it, encode it, and which spectral `Band` it supplies)

Both the index definitions (`indices.py`) and the composite algorithm (`composite.py`) build on these.
"""

from dataclasses import dataclass, field
from enum import Enum, unique

import numpy as np

from hls_composites.models import Satellite

SR_SCALE = 0.0001
SR_FILL = -9999
QA_FILL = 255


@unique
class Band(Enum):
    B = "B"
    G = "G"
    R = "R"
    NIR = "NIR"
    SWIR1 = "SWIR1"
    SWIR2 = "SWIR2"


@dataclass(frozen=True)
class BandSpec:
    """One HLS asset band: how to read it, encode it, and what it means.

    Parameters
    ----------
    name : str
        Logical band name, e.g. `"red"` or `"Fmask"`. Used as the
        output Dataset variable name.
    index_band : Band or None
        The spectral `Band` this asset supplies, e.g. `Band.NIR`. A
        reflectance band maps to exactly one spectral band; a QA band
        like Fmask maps to None. This is also what distinguishes
        reflectance from QA bands (see `is_reflectance`).
    code : dict of Satellite to str
        Per-satellite asset band code, e.g. `{"L30": "B05", "S30": "B8A"}`.
    nodata : int
        Fill value for pixels with no valid observation.
    dtype : type
        numpy dtype the composite output is cast to.
    valid_range : tuple of (int or None, int or None), optional
        `(min, max)` valid values for this band's raw digital numbers,
        checked *before* `scale` is applied (see `scale` below).
        Observations outside this range are masked out. Either bound may
        be None to skip that side of the check; the default `(None, None)`
        skips both -- e.g. Fmask, whose values are QA bit flags, not a
        physical quantity with a valid range.
    scale : float, optional
        Factor converting this band's raw digital numbers to physical
        units, by default 1.0 (no scaling) -- e.g. Fmask, whose values
        are QA bit flags, not a scaled physical quantity. Applied
        *after* `valid_range` is checked, never before.
    """

    name: str
    index_band: Band | None
    code: dict[Satellite, str] = field(compare=False)
    nodata: int
    dtype: type
    valid_range: tuple[int | None, int | None] = (None, None)
    scale: float = 1.0

    @property
    def is_reflectance(self) -> bool:
        """Whether this is a reflectance band (vs. a QA band like Fmask).

        A reflectance band feeds spectral-index computation and gets a
        `{name}_std` output; a QA band does not. Equivalent to
        `index_band is not None`.
        """
        return self.index_band is not None


RED = BandSpec(
    "red",
    Band.R,
    {"L30": "B04", "S30": "B04"},
    nodata=SR_FILL,
    dtype=np.int16,
    valid_range=(0, None),
    scale=SR_SCALE,
)
GREEN = BandSpec(
    "green",
    Band.G,
    {"L30": "B03", "S30": "B03"},
    nodata=SR_FILL,
    dtype=np.int16,
    valid_range=(0, None),
    scale=SR_SCALE,
)
BLUE = BandSpec(
    "blue",
    Band.B,
    {"L30": "B02", "S30": "B02"},
    nodata=SR_FILL,
    dtype=np.int16,
    valid_range=(0, None),
    scale=SR_SCALE,
)
NIR_NARROW = BandSpec(
    "nir_narrow",
    Band.NIR,
    {"L30": "B05", "S30": "B8A"},
    nodata=SR_FILL,
    dtype=np.int16,
    valid_range=(0, None),
    scale=SR_SCALE,
)
SWIR_1 = BandSpec(
    "swir_1",
    Band.SWIR1,
    {"L30": "B06", "S30": "B11"},
    nodata=SR_FILL,
    dtype=np.int16,
    valid_range=(0, None),
    scale=SR_SCALE,
)
SWIR_2 = BandSpec(
    "swir_2",
    Band.SWIR2,
    {"L30": "B07", "S30": "B12"},
    nodata=SR_FILL,
    dtype=np.int16,
    valid_range=(0, None),
    scale=SR_SCALE,
)
FMASK = BandSpec(
    "Fmask", None, {"L30": "Fmask", "S30": "Fmask"}, nodata=QA_FILL, dtype=np.uint8
)

DEFAULT_BANDS: list[BandSpec] = [RED, GREEN, BLUE, NIR_NARROW, SWIR_1, SWIR_2, FMASK]

REFLECTANCE_BANDS: list[BandSpec] = [b for b in DEFAULT_BANDS if b.is_reflectance]

SPEC_BY_BAND: dict[Band, BandSpec] = {
    b.index_band: b for b in DEFAULT_BANDS if b.index_band is not None
}
"""Reverse lookup from a spectral `Band` to the reflectance `BandSpec` supplying it."""
