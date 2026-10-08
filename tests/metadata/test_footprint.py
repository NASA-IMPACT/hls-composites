from itertools import pairwise

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import transform as transform_points
from shapely import Point, Polygon

from hls_composites.metadata.footprint import footprint, footprint_bbox

SIZE = 366
PIXEL = 300.0

# (EPSG, upper-left x, upper-left y) of SIZE x SIZE grids.
MID_LATITUDE = (32614, 600000.0, 4800000.0)  # 14TPN, South Dakota
HIGH_LATITUDE = (32633, 400000.0, 8900000.0)  # Svalbard, ~80N
# Zone 1 at ~60N: 180 degrees falls near x=332705, inside the grid.
ANTIMERIDIAN = (32601, 300000.0, 6700000.0)


def grid(location):
    epsg, ulx, uly = location
    return from_origin(ulx, uly, PIXEL, PIXEL), CRS.from_epsg(epsg)


def corners_lonlat(transform, crs, cols, rows):
    xs = [transform.c + transform.a * col for col in cols]
    ys = [transform.f + transform.e * row for row in rows]
    return transform_points(crs, "EPSG:4326", xs, ys)[:2]


def signed_area(ring):
    """Shoelace sum: positive for a counter-clockwise ring."""
    closed = [*ring, ring[0]]
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in pairwise(closed))


@pytest.mark.parametrize("location", [MID_LATITUDE, HIGH_LATITUDE])
def test_a_full_grid_is_outlined_by_its_corners(location):
    transform, crs = grid(location)
    valid = np.ones((SIZE, SIZE), dtype=bool)

    [ring] = footprint(valid, transform, crs)

    lons, lats = corners_lonlat(transform, crs, [0, SIZE, SIZE, 0], [0, 0, SIZE, SIZE])
    np.testing.assert_allclose(sorted(ring), sorted(zip(lons, lats, strict=True)))


def test_an_empty_grid_is_outlined_whole():
    transform, crs = grid(MID_LATITUDE)
    empty = np.zeros((SIZE, SIZE), dtype=bool)
    full = np.ones((SIZE, SIZE), dtype=bool)

    assert footprint(empty, transform, crs) == footprint(full, transform, crs)


def test_rings_are_counter_clockwise_and_open():
    transform, crs = grid(MID_LATITUDE)

    [ring] = footprint(np.ones((SIZE, SIZE), dtype=bool), transform, crs)

    assert signed_area(ring) > 0
    assert ring[0] != ring[-1]


def test_a_partial_grid_is_outlined_tightly():
    """Half the grid is valid, so the outline covers about half its area."""
    transform, crs = grid(MID_LATITUDE)
    triangle = np.tril(np.ones((SIZE, SIZE), dtype=bool))

    [ring] = footprint(triangle, transform, crs)
    [whole] = footprint(np.ones((SIZE, SIZE), dtype=bool), transform, crs)

    assert len(ring) <= 5
    assert Polygon(ring).area / Polygon(whole).area == pytest.approx(0.5, abs=0.01)


def test_the_outline_covers_every_valid_pixel():
    transform, crs = grid(HIGH_LATITUDE)
    rows, cols = np.mgrid[:SIZE, :SIZE]
    disk = (rows - 100) ** 2 + (cols - 250) ** 2 < 80**2

    [ring] = footprint(disk, transform, crs)

    lons, lats = corners_lonlat(transform, crs, cols[disk] + 0.5, rows[disk] + 0.5)
    outline = Polygon(ring)
    assert all(outline.contains(Point(lon, lat)) for lon, lat in zip(lons, lats))


def test_a_sliver_is_outlined_around_just_the_sliver():
    transform, crs = grid(MID_LATITUDE)
    sliver = np.zeros((SIZE, SIZE), dtype=bool)
    sliver[SIZE - 1, SIZE - 1] = True

    [ring] = footprint(sliver, transform, crs)

    lons, lats = corners_lonlat(
        transform,
        crs,
        [SIZE - 1, SIZE, SIZE, SIZE - 1],
        [SIZE - 1, SIZE - 1, SIZE, SIZE],
    )
    np.testing.assert_allclose(sorted(ring), sorted(zip(lons, lats, strict=True)))


def test_an_outline_crossing_the_antimeridian_is_split_there():
    transform, crs = grid(ANTIMERIDIAN)

    rings = footprint(np.ones((SIZE, SIZE), dtype=bool), transform, crs)

    assert len(rings) == 2
    east, west = sorted(rings, key=lambda ring: -max(lon for lon, _ in ring))
    assert max(lon for lon, _ in east) == 180
    assert min(lon for lon, _ in west) == -180
    assert all(signed_area(ring) > 0 for ring in rings)


def test_bbox_of_a_split_outline_wraps_the_antimeridian():
    """RFC 7946 spells a crossing bbox with west greater than east."""
    transform, crs = grid(ANTIMERIDIAN)
    rings = footprint(np.ones((SIZE, SIZE), dtype=bool), transform, crs)

    west, south, east, north = footprint_bbox(rings)

    assert 170 < west < 180
    assert -180 < east < -170
    assert south < north


def test_bbox_of_an_unsplit_outline_is_its_extent():
    transform, crs = grid(MID_LATITUDE)
    [ring] = footprint(np.ones((SIZE, SIZE), dtype=bool), transform, crs)
    lons = [lon for lon, _ in ring]
    lats = [lat for _, lat in ring]

    assert footprint_bbox([ring]) == pytest.approx(
        (min(lons), min(lats), max(lons), max(lats))
    )
