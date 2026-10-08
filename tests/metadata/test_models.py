from datetime import UTC, datetime

import numpy as np
import pytest
from rasterio.warp import transform as transform_points

from hls_composites.indices import NDVI
from hls_composites.metadata.models import granule_metadata
from hls_composites.outputs import VALID_COUNT
from tests.metadata.conftest import (
    EPSG,
    FEBRUARY,
    GRANULE_ID,
    PIXEL,
    PLATFORMS,
    ULX,
    ULY,
)

PRODUCED_AT = datetime(2026, 9, 3, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def meta(rasters, browse_images):
    return granule_metadata(
        "14TPN",
        FEBRUARY,
        rasters,
        browse_images,
        platforms=PLATFORMS,
        produced_at=PRODUCED_AT,
    )


def test_identity_comes_from_the_tile_and_period(meta):
    assert meta.granule_id == GRANULE_ID
    assert meta.tile_id == "14TPN"
    assert meta.date_range == FEBRUARY


def test_grid_is_read_from_the_written_raster(meta):
    assert meta.epsg == EPSG
    assert meta.ulx == ULX
    assert meta.uly == ULY
    assert meta.ncols == 4
    assert meta.nrows == 4


def test_crs_name_is_human_readable(meta):
    assert "UTM" in meta.crs_name


def test_spatial_coverage_is_the_percentage_of_valid_pixels(meta):
    """The fixture has 12 of 16 pixels valid."""
    assert meta.spatial_coverage == 75


def test_footprint_outlines_only_the_valid_pixels(meta):
    """The fixture's top row is fill, so the outline stops a pixel short of it."""
    [ring] = meta.footprint
    lons, lats = transform_points(
        f"EPSG:{EPSG}",
        "EPSG:4326",
        [ULX, ULX + 4 * PIXEL, ULX + 4 * PIXEL, ULX],
        [ULY - PIXEL, ULY - PIXEL, ULY - 4 * PIXEL, ULY - 4 * PIXEL],
    )[:2]

    np.testing.assert_allclose(sorted(ring), sorted(zip(lons, lats, strict=True)))


def test_bbox_is_the_extent_of_the_footprint(meta):
    [ring] = meta.footprint
    lons = [lon for lon, _ in ring]
    lats = [lat for _, lat in ring]

    assert meta.bbox == pytest.approx((min(lons), min(lats), max(lons), max(lats)))


def test_encoding_constants_match_the_index_definitions(meta):
    assert meta.scale_factor == 1e-4
    assert meta.add_offset == 0
    assert isinstance(meta.add_offset, int)


def test_each_written_fill_is_declared_under_its_own_attribute(meta):
    """The fixture writes NDVI and ValidCount, which name different attributes."""
    assert meta.fill_values == {
        "FILLVALUE": NDVI.fill_value,
        "VALIDCOUNT_FILLVALUE": VALID_COUNT.nodata,
    }


def test_assets_are_the_written_geotiffs_sorted(meta):
    assert [path.name for path in meta.assets] == [
        f"{GRANULE_ID}.NDVI.tif",
        f"{GRANULE_ID}.ValidCount.tif",
    ]


def test_size_is_the_total_of_those_files(meta, granule_dir):
    expected = sum(p.stat().st_size for p in granule_dir.glob("*.tif"))

    assert meta.size_bytes == expected


def test_platforms_are_the_ones_given(meta):
    assert meta.platforms == PLATFORMS


def test_no_platforms_is_refused(rasters, browse_images):
    """Nothing stands in for them: a stand-in would name the wrong fleet."""
    with pytest.raises(ValueError, match="platform"):
        granule_metadata("14TPN", FEBRUARY, rasters, browse_images, platforms=[])


def test_a_composite_without_valid_count_is_refused(rasters, browse_images):
    """Coverage and the footprint both come from it."""
    del rasters["ValidCount"]

    with pytest.raises(ValueError, match="ValidCount"):
        granule_metadata("14TPN", FEBRUARY, rasters, browse_images, platforms=PLATFORMS)


def test_produced_at_defaults_to_now(rasters, browse_images):
    meta = granule_metadata(
        "14TPN", FEBRUARY, rasters, browse_images, platforms=PLATFORMS
    )

    assert meta.produced_at.tzinfo is UTC
