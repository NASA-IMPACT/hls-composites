"""What a composite writes: one declaration per output layer.

Every variable the compositor emits -- a spectral index, its temporal standard
deviation, or a layer derived from the selection itself -- is described here
once: how it is stored, what marks a pixel as absent, and which ECHO-10
attribute names that fill. The compositor builds its dask template from these,
the writer stamps them onto each COG, and the metadata reads its fill
attributes from them, so the three cannot disagree.

Reflectance and QA bands read from the input granules keep their own
`BandSpec`, which this module wraps rather than restates.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from hls_composites.bands import FMASK, REFLECTANCE_BANDS, BandSpec
from hls_composites.indices import ALL_INDICES, Index


@runtime_checkable
class OutputBand(Protocol):
    """What every consumer of a written layer needs to know about it."""

    @property
    def name(self) -> str:
        """Variable name, which is also the file suffix and the asset key."""

    @property
    def long_name(self) -> str:
        """Band description written into the COG."""

    @property
    def dtype(self) -> type:
        """numpy dtype the layer is stored as."""

    @property
    def nodata(self) -> int:
        """Value marking a pixel with no data."""

    @property
    def scale(self) -> float:
        """Factor converting stored values to physical units."""

    @property
    def fill_attribute(self) -> str:
        """ECHO-10 attribute naming this layer's fill value."""


@dataclass(frozen=True)
class DerivedBand:
    """A layer computed from the per-pixel selection rather than read.

    Parameters
    ----------
    name : str
        Variable name, e.g. ``ValidCount``.
    long_name : str
        Band description written into the COG.
    dtype : type
        numpy dtype the layer is stored as.
    nodata : int
        Value marking a pixel with no data.
    fill_attribute : str
        ECHO-10 attribute naming that fill. These layers exist only in the
        composite, so each declares an attribute of its own rather than
        sharing one of the daily products'.
    scale : float, optional
        Factor converting stored values to physical units, by default 1.0 --
        a count and a day of year are already in their own units.
    """

    name: str
    long_name: str
    dtype: type
    nodata: int
    fill_attribute: str
    scale: float = 1.0


@dataclass(frozen=True)
class IndexBand:
    """A spectral index, or its temporal standard deviation, as written.

    Wraps an `Index`, which owns the encoding, so the index definition stays
    the single source of the scale and fill it computes with.

    Parameters
    ----------
    index : Index
        The index this layer carries.
    deviation : bool, optional
        Whether this is the `{index}_std` layer rather than the index itself,
        by default False.
    """

    index: Index
    deviation: bool = False

    @property
    def name(self) -> str:
        return f"{self.index.name}_std" if self.deviation else self.index.name

    @property
    def long_name(self) -> str:
        if self.deviation:
            return f"{self.index.long_name} standard deviation"
        return self.index.long_name

    @property
    def dtype(self) -> type:
        return np.int16

    @property
    def nodata(self) -> int:
        return self.index.fill_value

    @property
    def scale(self) -> float:
        return self.index.scale_factor

    @property
    def fill_attribute(self) -> str:
        # As the daily HLS and HLS-VI products name their value bands' fill.
        return "FILLVALUE"


@dataclass(frozen=True)
class ReflectanceBand:
    """One reflectance band, or its temporal standard deviation, as written.

    The `--bands` counterpart of `IndexBand`, wrapping the `BandSpec` that
    already describes how the band is read and stored.

    Parameters
    ----------
    band : BandSpec
        The reflectance band this layer carries.
    deviation : bool, optional
        Whether this is the `{band}_std` layer, by default False.
    """

    band: BandSpec
    deviation: bool = False

    @property
    def name(self) -> str:
        return f"{self.band.name}_std" if self.deviation else self.band.name

    @property
    def long_name(self) -> str:
        if self.deviation:
            return f"{self.band.long_name} standard deviation"
        return self.band.long_name

    @property
    def dtype(self) -> type:
        return self.band.dtype

    @property
    def nodata(self) -> int:
        return self.band.nodata

    @property
    def scale(self) -> float:
        return self.band.scale

    @property
    def fill_attribute(self) -> str:
        # As the daily HLS and HLS-VI products name their value bands' fill.
        return "FILLVALUE"


@dataclass(frozen=True)
class QaBand:
    """Fmask, carried through from the selected observation.

    Wraps the `BandSpec` it is read with, and declares the fill attribute the
    daily products use for their own QA band.
    """

    band: BandSpec = FMASK

    @property
    def name(self) -> str:
        return self.band.name

    @property
    def long_name(self) -> str:
        return self.band.long_name

    @property
    def dtype(self) -> type:
        return self.band.dtype

    @property
    def nodata(self) -> int:
        return self.band.nodata

    @property
    def scale(self) -> float:
        return self.band.scale

    @property
    def fill_attribute(self) -> str:
        # As the daily HLS and HLS-VI products name their QA band's fill.
        return "QA_FILLVALUE"


VALID_COUNT = DerivedBand(
    name="ValidCount",
    long_name="Count of valid observations",
    dtype=np.int16,
    nodata=-999,
    fill_attribute="VALIDCOUNT_FILLVALUE",
)
"""How many observations a pixel was composited from.

Negative fill, so it cannot be mistaken for a count however the band is read.
That costs `int16` storage for a quantity that is never negative.
"""

DOY = DerivedBand(
    name="DOY",
    long_name="Day of year of the selected observation",
    dtype=np.int16,
    nodata=-999,
    fill_attribute="DOY_FILLVALUE",
)
"""Which observation each pixel was taken from. Julian days are 1..366."""

FMASK_OUTPUT = QaBand()
"""The selected observation's QA, carried through to the composite."""

SELECTION_BANDS: tuple[OutputBand, ...] = (FMASK_OUTPUT, VALID_COUNT, DOY)
"""Layers describing the selection itself, written whatever is composited."""


def value_bands(values: list[Index] | list[BandSpec]) -> list[OutputBand]:
    """The layers `values` themselves produce: each one and its deviation.

    Parameters
    ----------
    values : list of Index or list of BandSpec
        What the composite is built over: the spectral indices, or the
        reflectance bands of a `--bands` run.

    Returns
    -------
    list of OutputBand
        Each value band followed by its standard deviation, in `values` order.
    """
    layers: list[OutputBand] = []
    for value in values:
        if isinstance(value, Index):
            layers += [IndexBand(value), IndexBand(value, deviation=True)]
        else:
            layers += [ReflectanceBand(value), ReflectanceBand(value, deviation=True)]
    return layers


def output_bands(values: list[Index] | list[BandSpec]) -> list[OutputBand]:
    """Every layer a composite of `values` writes, in the order it writes them."""
    return value_bands(values) + list(SELECTION_BANDS)


BAND_BY_NAME: dict[str, OutputBand] = {
    band.name: band
    for band in (
        *value_bands(ALL_INDICES),
        *value_bands(REFLECTANCE_BANDS),
        *SELECTION_BANDS,
    )
}
"""Every layer this product can write, keyed by name.

How a written file is matched back to what declared it, whichever quantity the
composite was built over. Ordered, since the fill attributes it groups are
written in this order.
"""


def fill_attributes(fills: Mapping[str, float | None]) -> dict[str, int]:
    """Group written fill values under the ECHO-10 attributes naming them.

    Parameters
    ----------
    fills : mapping of str to float or None
        Fill value per written variable name, as read back from the files.
        A variable with no fill, or one nothing declares, is skipped.

    Returns
    -------
    dict of str to int
        Attribute name to fill value, in `BAND_BY_NAME` order.

    Raises
    ------
    ValueError
        If two variables sharing an attribute were written with different
        fills, which one attribute cannot describe.
    """
    attributes: dict[str, int] = {}
    for band in BAND_BY_NAME.values():
        fill = fills.get(band.name)
        if fill is None:
            continue
        previous = attributes.setdefault(band.fill_attribute, int(fill))
        if previous != int(fill):
            raise ValueError(
                f"{band.name} was written with fill {int(fill)}, but "
                f"{band.fill_attribute} already declares {previous}"
            )
    return attributes
