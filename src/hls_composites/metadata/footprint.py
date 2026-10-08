"""The lon/lat outline of where a composite has data.

The outline is the convex hull of the valid pixels, taken in the rasters' own
projected CRS: a UTM grid is planar and never crosses the antimeridian, so the
hull is well defined there. The hull's edges are densified before its vertices
are reprojected, and `antimeridian` splits the result wherever it crosses 180
degrees.
"""

import antimeridian
import numpy as np
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform as transform_points
from shapely import MultiPoint, MultiPolygon, Polygon, segmentize
from shapely.geometry.polygon import orient

Ring = list[tuple[float, float]]

# A straight projected edge bows away from the straight lon/lat line between
# its reprojected ends, by up to ~2 km along a 110 km tile edge at 84N. The bow
# grows with the square of edge length, so 10 km edges keep it under a 30 m
# pixel.
MAX_EDGE_LENGTH = 10_000.0


def _pixel_corners(valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column and row of every corner of each row's outermost valid pixels.

    The hull of these is the hull of every valid pixel, without listing the
    pixels in between.
    """
    rows = np.flatnonzero(valid.any(axis=1))
    if rows.size == 0:
        nrows, ncols = valid.shape
        return np.array([0, ncols, ncols, 0]), np.array([0, 0, nrows, nrows])

    first = valid[rows].argmax(axis=1)
    last = valid.shape[1] - valid[rows, ::-1].argmax(axis=1)
    cols = np.concatenate([first, last, first, last])
    corner_rows = np.concatenate([rows, rows, rows + 1, rows + 1])
    return cols, corner_rows


def _rings(geometry: Polygon | MultiPolygon) -> list[Ring]:
    polygons = geometry.geoms if isinstance(geometry, MultiPolygon) else [geometry]
    return [
        [(float(x), float(y)) for x, y in polygon.exterior.coords[:-1]]
        for polygon in polygons
    ]


def footprint(valid: np.ndarray, transform: Affine, crs: CRS) -> list[Ring]:
    """Outline the valid pixels of a grid in longitude and latitude.

    Parameters
    ----------
    valid : numpy.ndarray
        Boolean mask, True where the composite has data. An empty mask
        outlines the whole grid.
    transform : affine.Affine
        The grid's geotransform.
    crs : rasterio.crs.CRS
        The grid's projected CRS.

    Returns
    -------
    list of list of tuple of float
        One ring of ``(longitude, latitude)`` vertices per polygon:
        counter-clockwise as GeoJSON orders an exterior ring, and not closed.
        Two rings when the outline crosses the antimeridian, one otherwise.
    """
    cols, rows = _pixel_corners(valid)
    xs = transform.c + transform.a * cols + transform.b * rows
    ys = transform.f + transform.d * cols + transform.e * rows

    hull = MultiPoint(np.column_stack([xs, ys])).convex_hull
    # Even a single pixel's four corners enclose an area.
    assert isinstance(hull, Polygon)
    hull = segmentize(orient(hull, sign=1.0), MAX_EDGE_LENGTH)

    hull_xs, hull_ys = hull.exterior.coords.xy
    lons, lats = transform_points(crs, "EPSG:4326", list(hull_xs), list(hull_ys))[:2]
    outline = Polygon(zip(lons, lats, strict=True))
    return _rings(antimeridian.fix_polygon(outline, fix_winding=True))


def footprint_bbox(rings: list[Ring]) -> tuple[float, float, float, float]:
    """Bounding box of `footprint` rings, as GeoJSON and STAC spell it.

    Parameters
    ----------
    rings : list of list of tuple of float
        Rings as `footprint` returns them.

    Returns
    -------
    tuple of float
        ``(west, south, east, north)`` in degrees. West exceeds east when the
        rings cross the antimeridian, as RFC 7946 section 5.2 specifies.
    """
    geometry = {
        "type": "MultiPolygon",
        "coordinates": [[[*ring, ring[0]]] for ring in rings],
    }
    west, south, east, north = antimeridian.bbox(geometry)
    return west, south, east, north
