from datetime import UTC, datetime

import pytest

from hls_composites.metadata.models import (
    COMPOSITING_ALGORITHM,
    DATASET_ID,
    DOI,
    PLACEHOLDER,
    PRODUCT_URI_BASE,
    SHORT_NAME,
    SPATIAL_RESOLUTION,
    granule_metadata,
)
from tests.metadata.conftest import EPSG, FEBRUARY, GRANULE_ID, ULX, ULY

PRODUCED_AT = datetime(2026, 9, 3, 12, 0, 0, tzinfo=UTC)


@pytest.fixture
def meta(granule_dir):
    return granule_metadata("14TPN", FEBRUARY, granule_dir, produced_at=PRODUCED_AT)


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


def test_boundary_is_lon_lat_and_encloses_the_grid(meta):
    lons = [lon for lon, _ in meta.boundary]
    lats = [lat for _, lat in meta.boundary]

    assert len(meta.boundary) == 4
    # Tile 14TPN sits in the northern hemisphere, west of Greenwich.
    assert all(-180 <= lon <= 0 for lon in lons)
    assert all(0 < lat < 90 for lat in lats)
    assert meta.bbox == (min(lons), min(lats), max(lons), max(lats))


def test_encoding_constants_match_the_index_definitions(meta):
    assert meta.scale_factor == 1e-4
    assert meta.add_offset == 0.0
    assert meta.fill_value == -19999
    assert meta.qa_fill_value == 255


def test_assets_are_the_written_geotiffs_sorted(meta):
    assert [path.name for path in meta.assets] == [
        f"{GRANULE_ID}.NDVI.tif",
        f"{GRANULE_ID}.ValidCount.tif",
    ]


def test_size_is_the_total_of_those_files(meta, granule_dir):
    expected = sum(p.stat().st_size for p in granule_dir.glob("*.tif"))

    assert meta.size_bytes == expected


def test_produced_at_defaults_to_now(granule_dir):
    meta = granule_metadata("14TPN", FEBRUARY, granule_dir)

    assert meta.produced_at.tzinfo is UTC


def test_constants_that_are_known_are_set():
    assert SHORT_NAME == "HLSM30"
    assert SPATIAL_RESOLUTION == 30.0
    assert COMPOSITING_ALGORITHM != PLACEHOLDER


def test_placeholders():
    """Records the values still awaiting the DAAC.

    Update this list as they are assigned; it is the inventory of what is
    not yet real, so a reader never has to guess which values are invented.
    """
    assert [DATASET_ID, DOI, PRODUCT_URI_BASE] == [PLACEHOLDER] * 3
