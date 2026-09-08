"""Reconcile a raster's declared CRS with the hemisphere its tile ID implies.

HLS labels some southern-hemisphere granules with a northern UTM code while
writing coordinates that carry the southern false northing. The pixels are
where they belong, but anything that reprojects via the declared CRS -- this
project's metadata, among others -- places the granule in the wrong hemisphere.

The two UTM flavours of a zone differ only by that 10,000,000 m false northing,
so switching between them is a relabel: the coordinates already mean what the
southern definition says they mean, and no pixel moves.
"""

from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import array_bounds

from hls_composites.models import is_southern

SOUTHERN_FALSE_NORTHING = 10_000_000.0
"""Metres the southern UTM definition adds, so its northings stay positive."""

_NORTHERN_UTM = range(32601, 32661)
_NORTHERN_TO_SOUTHERN = 100
"""EPSG offset between a zone's northern and southern definitions."""


def _declares_southern(crs: CRS) -> bool:
    """Whether `crs` is the southern flavour of a UTM zone."""
    return bool(crs.to_dict().get("south", False))


def corrected_crs(
    crs: CRS, transform: Affine, shape: tuple[int, int], tile_id: str
) -> CRS:
    """The CRS matching where a southern tile's coordinates actually are.

    Relabels only when all of these hold, so it corrects the upstream
    mislabelling without depending on it:

    - the tile's MGRS band is south of the equator,
    - the CRS is a northern UTM zone, and
    - the northings sit between zero and the southern false northing, which is
      what carrying that offset looks like.

    A granule already declaring the southern flavour fails the second test; one
    using a northern zone with negative northings -- equally valid -- fails the
    third. Both are returned untouched.

    Parameters
    ----------
    crs : rasterio.crs.CRS
        CRS as declared by the source data.
    transform : affine.Affine
        The raster's affine transform.
    shape : tuple of int
        `(height, width)` in pixels.
    tile_id : str
        MGRS tile ID, without the leading "T". Its latitude band is the
        independent statement of which hemisphere the data belongs to.

    Returns
    -------
    rasterio.crs.CRS
        `crs`, or the same zone's southern definition when that is the one the
        coordinates were written against.
    """
    epsg = crs.to_epsg()
    if (
        not is_southern(tile_id)
        or _declares_southern(crs)
        or epsg is None
        or epsg not in _NORTHERN_UTM
    ):
        return crs

    _, bottom, _, top = array_bounds(shape[0], shape[1], transform)
    if bottom < 0.0:
        # A northern zone with negative northings is the other valid way to
        # write southern data. Not ours to rewrite.
        return crs
    if top > SOUTHERN_FALSE_NORTHING:
        raise ValueError(
            f"tile {tile_id} is southern, but its northings exceed "
            f"{SOUTHERN_FALSE_NORTHING:,.0f}: north of the equator with the "
            f"offset removed, and beyond the valid range of {crs} without it"
        )

    return CRS.from_epsg(epsg + _NORTHERN_TO_SOUTHERN)
