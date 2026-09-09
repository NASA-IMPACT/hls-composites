"""MGRS tile vocabulary, and putting southern granules in a southern UTM zone.

HLS writes southern-hemisphere granules with a northern UTM code and negative
northings. Read with that code the coordinates resolve correctly, so nothing is
misplaced, but the northern definition is the wrong one for southern data and
consumers expect the southern zone.

Tile IDs live here because their latitude band is what says which hemisphere a
granule belongs to, independently of whatever its CRS claims.
"""

import re

from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import array_bounds

# Zone 1-60, latitude band excluding I and O, two-letter grid square.
_MGRS_TILE = re.compile(r"^([0-9]{1,2})([C-HJ-NP-X])([A-Z]{2})$")

SOUTHERN_BANDS = frozenset("CDEFGHJKLM")
"""MGRS latitude bands south of the equator. N through X are northern."""


def crs_name(crs: CRS) -> str:
    """The CRS's declared name, e.g. ``WGS 84 / UTM zone 14N``.

    It is the first quoted string in the WKT, so no pyproj lookup is needed.
    """
    parts = crs.to_wkt().split('"')
    return parts[1] if len(parts) > 1 else str(crs)


def mgrs_fields(tile_id: str) -> tuple[int, str, str]:
    """Split an MGRS tile ID into its UTM zone, latitude band, and grid square.

    Parameters
    ----------
    tile_id : str
        Tile ID without the leading "T", e.g. ``14TPN``.

    Returns
    -------
    tuple
        ``(utm_zone, latitude_band, grid_square)``, e.g. ``(14, "T", "PN")``.

    Raises
    ------
    ValueError
        If `tile_id` is not a well-formed MGRS tile.
    """
    match = _MGRS_TILE.match(tile_id)
    if match is None:
        raise ValueError(f"not an MGRS tile: {tile_id!r}")
    zone, band, square = match.groups()
    return int(zone), band, square


def is_southern(tile_id: str) -> bool:
    """Whether an MGRS tile lies south of the equator."""
    _, band, _ = mgrs_fields(tile_id)
    return band in SOUTHERN_BANDS


SOUTHERN_FALSE_NORTHING = 10_000_000.0
"""Metres the southern UTM definition measures northings from."""

_NORTHERN_UTM = range(32601, 32661)
_NORTHERN_TO_SOUTHERN = 100
"""EPSG offset between a zone's northern and southern definitions."""


def _declares_southern(crs: CRS) -> bool:
    """Whether `crs` is the southern definition of a UTM zone."""
    return bool(crs.to_dict().get("south", False))


def corrected_grid(
    crs: CRS, transform: Affine, shape: tuple[int, int], tile_id: str
) -> tuple[CRS, Affine]:
    """Georeferencing for a southern tile, expressed in its southern UTM zone.

    A zone's two definitions differ only in where northings are measured from:
    zero in the north, 10,000,000 m in the south. Converting therefore moves the
    origin as well as changing the code. The array is never touched and no pixel
    is resampled.

    Inputs that already agree are returned untouched -- a granule declaring the
    southern zone, or a northern tile -- so this corrects the upstream labelling
    without depending on it.

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
    tuple of (rasterio.crs.CRS, affine.Affine)
        The CRS and transform to write.

    Raises
    ------
    ValueError
        If the northings sit above the southern false northing, which neither
        input above can produce for a southern tile.
    """
    epsg = crs.to_epsg()
    if (
        not is_southern(tile_id)
        or _declares_southern(crs)
        or epsg is None
        or epsg not in _NORTHERN_UTM
    ):
        return crs, transform

    _, bottom, _, top = array_bounds(shape[0], shape[1], transform)
    if top > SOUTHERN_FALSE_NORTHING:
        raise ValueError(
            f"tile {tile_id} is southern, but its northings exceed "
            f"{SOUTHERN_FALSE_NORTHING:,.0f}, which no southern tile reaches"
        )

    southern = CRS.from_epsg(epsg + _NORTHERN_TO_SOUTHERN)
    if bottom >= 0.0:
        # The offset is already carried, so only the label is wrong.
        return southern, transform

    return southern, Affine(
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f + SOUTHERN_FALSE_NORTHING,
    )
