"""Reconciling a raster's georeferencing with the hemisphere its tile ID implies."""

import pytest
from rasterio.crs import CRS
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import transform_bounds

from hls_composites.crs import SOUTHERN_FALSE_NORTHING, corrected_grid

SHAPE = (3660, 3660)
UTM_19N = CRS.from_epsg(32619)
UTM_19S = CRS.from_epsg(32719)

# A real southern granule as HLS writes it: northern zone, negative northings.
NEGATIVE = from_origin(799980.0, -999960.0, 30.0, 30.0)
# The same ground position with the southern false northing carried instead.
OFFSET = from_origin(799980.0, -999960.0 + SOUTHERN_FALSE_NORTHING, 30.0, 30.0)
NORTHERN = from_origin(699960.0, 4600020.0, 30.0, 30.0)


def latlon(crs, transform):
    """Where a grid actually sits, which no correction may change."""
    bounds = array_bounds(SHAPE[0], SHAPE[1], transform)
    return [round(v, 6) for v in transform_bounds(crs, "EPSG:4326", *bounds)]


class TestNegativeNorthings:
    """What HLS writes today: EPSG:326xx with northings below zero."""

    def test_relabels_to_the_southern_flavour(self):
        crs, _ = corrected_grid(UTM_19N, NEGATIVE, SHAPE, "19LHK")

        assert crs.to_epsg() == 32719

    def test_shifts_the_origin_by_the_false_northing(self):
        """The southern definition measures from 10,000,000 m, not zero."""
        _, transform = corrected_grid(UTM_19N, NEGATIVE, SHAPE, "19LHK")

        assert transform.f == NEGATIVE.f + SOUTHERN_FALSE_NORTHING

    def test_the_ground_position_is_unchanged(self):
        """The property that matters: relabelling alone would move it to Antarctica."""
        before = latlon(UTM_19N, NEGATIVE)

        crs, transform = corrected_grid(UTM_19N, NEGATIVE, SHAPE, "19LHK")

        assert latlon(crs, transform) == before

    def test_only_the_origin_moves(self):
        _, transform = corrected_grid(UTM_19N, NEGATIVE, SHAPE, "19LHK")

        assert (transform.a, transform.b, transform.c, transform.d, transform.e) == (
            NEGATIVE.a,
            NEGATIVE.b,
            NEGATIVE.c,
            NEGATIVE.d,
            NEGATIVE.e,
        )


class TestOffsetNorthings:
    """The other spelling: the false northing already carried in the values."""

    def test_relabels_without_moving_the_origin(self):
        crs, transform = corrected_grid(UTM_19N, OFFSET, SHAPE, "19LHK")

        assert crs.to_epsg() == 32719
        assert transform == OFFSET

    def test_the_ground_position_is_unchanged(self):
        crs, transform = corrected_grid(UTM_19N, OFFSET, SHAPE, "19LHK")

        assert latlon(crs, transform) == latlon(UTM_19S, OFFSET)


class TestLeftAlone:
    """The correction must switch itself off, not merely be idempotent."""

    def test_a_southern_crs_is_untouched(self):
        crs, transform = corrected_grid(UTM_19S, OFFSET, SHAPE, "19LHK")

        assert (crs, transform) == (UTM_19S, OFFSET)

    def test_a_northern_tile_is_untouched(self):
        crs, transform = corrected_grid(UTM_19N, NORTHERN, SHAPE, "19TCH")

        assert (crs, transform) == (UTM_19N, NORTHERN)

    def test_a_non_utm_crs_is_untouched(self):
        geographic = CRS.from_epsg(4326)
        grid = from_origin(-66.0, -9.0, 0.0003, 0.0003)

        assert corrected_grid(geographic, grid, SHAPE, "19LHK") == (geographic, grid)

    def test_correcting_twice_changes_nothing(self):
        once = corrected_grid(UTM_19N, NEGATIVE, SHAPE, "19LHK")

        assert corrected_grid(*once, SHAPE, "19LHK") == once


class TestUnresolvable:
    def test_northings_past_the_offset_raise(self):
        """Impossible under either spelling, so any correction would invent a place."""
        impossible = from_origin(799980.0, 11_000_000.0, 30.0, 30.0)

        with pytest.raises(ValueError, match="northings"):
            corrected_grid(UTM_19N, impossible, SHAPE, "19LHK")
