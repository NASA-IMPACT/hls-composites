"""Reconciling a raster's declared CRS with the hemisphere its tile ID implies."""

import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from hls_composites.crs import corrected_crs

SHAPE = (3660, 3660)
# A southern tile's grid, written with the 10,000,000 m false northing baked in.
SOUTHERN_TRANSFORM = from_origin(300000.0, 9100020.0, 30.0, 30.0)
# The same tile expressed the other valid way: northern CRS, negative northings.
NEGATIVE_TRANSFORM = from_origin(300000.0, -899980.0, 30.0, 30.0)
NORTHERN_TRANSFORM = from_origin(300000.0, 4600020.0, 30.0, 30.0)

UTM_18N = CRS.from_epsg(32618)
UTM_18S = CRS.from_epsg(32718)


class TestUpstreamIsWrong:
    """326xx declared, southern coordinates: today's HLS."""

    def test_relabels_to_the_southern_flavour(self):
        result = corrected_crs(UTM_18N, SOUTHERN_TRANSFORM, SHAPE, "18LWP")

        assert result.to_epsg() == 32718

    def test_coordinates_are_untouched(self):
        """A relabel, not a reprojection: 327xx already carries the offset."""
        result = corrected_crs(UTM_18N, SOUTHERN_TRANSFORM, SHAPE, "18LWP")

        assert result.to_dict()["zone"] == 18
        assert result.to_dict().get("south") is True


class TestUpstreamIsRight:
    """The workaround must switch itself off."""

    def test_southern_crs_is_left_alone(self):
        """If upstream starts declaring 327xx, change nothing."""
        result = corrected_crs(UTM_18S, SOUTHERN_TRANSFORM, SHAPE, "18LWP")

        assert result.to_epsg() == 32718

    def test_northern_crs_with_negative_northings_is_left_alone(self):
        """Also a valid way to express southern data; not ours to rewrite."""
        result = corrected_crs(UTM_18N, NEGATIVE_TRANSFORM, SHAPE, "18LWP")

        assert result.to_epsg() == 32618

    def test_northern_tiles_are_left_alone(self):
        result = corrected_crs(UTM_18N, NORTHERN_TRANSFORM, SHAPE, "18TWL")

        assert result.to_epsg() == 32618


class TestUnresolvable:
    def test_northings_past_the_offset_raise(self):
        """Impossible under either reading, so relabelling would invent a place."""
        impossible = from_origin(300000.0, 11_000_000.0, 30.0, 30.0)

        with pytest.raises(ValueError, match="northings exceed"):
            corrected_crs(UTM_18N, impossible, SHAPE, "18LWP")

    def test_a_non_utm_crs_is_out_of_scope(self):
        """There is no other flavour to relabel to."""
        geographic = CRS.from_epsg(4326)
        grid = from_origin(-76.0, -8.5, 0.0003, 0.0003)

        assert corrected_crs(geographic, grid, SHAPE, "18LWP") == geographic
