import datetime as dt

import pystac
import pytest

from hls_composites.metadata.models import granule_metadata
from hls_composites.metadata.stac import (
    PROJECTION_SCHEMA_URI,
    RASTER_SCHEMA_URI,
    SCIENTIFIC_SCHEMA_URI,
    to_stac_item,
)
from tests.metadata.conftest import (
    EPSG,
    FEBRUARY,
    GRANULE_ID,
    NDVI_DESCRIPTION,
    PIXEL,
    ULX,
    ULY,
)

PRODUCED_AT = dt.datetime(2026, 9, 3, 12, 0, 0, tzinfo=dt.UTC)


@pytest.fixture
def item(granule_dir, browse_image):
    meta = granule_metadata(
        "14TPN", FEBRUARY, granule_dir, browse_image, produced_at=PRODUCED_AT
    )
    return to_stac_item(meta)


def test_item_is_identified_by_the_granule_id(item):
    assert item["id"] == GRANULE_ID


def test_item_spans_the_compositing_period(item):
    """A composite has no single instant, so datetime is null and the range set."""
    assert item["properties"]["datetime"] is None
    assert item["properties"]["start_datetime"].startswith("2020-02-01")
    assert item["properties"]["end_datetime"].startswith("2020-02-29")


def test_projection_uses_the_v1_2_schema(item):
    """v1.1.0 rejects proj:code; v2.0.0 drops proj:epsg. v1.2.0 has both."""
    assert PROJECTION_SCHEMA_URI in item["stac_extensions"]
    assert "v1.2.0" in PROJECTION_SCHEMA_URI


def test_both_projection_spellings_are_written(item):
    """proj:epsg keeps consumers of the daily HLS products working."""
    assert item["properties"]["proj:epsg"] == EPSG
    assert item["properties"]["proj:code"] == f"EPSG:{EPSG}"


def test_projection_carries_shape_and_transform(item):
    assert item["properties"]["proj:shape"] == [4, 4]
    assert len(item["properties"]["proj:transform"]) == 6


def test_no_doi_is_claimed_while_it_is_a_placeholder(item):
    """The scientific extension requires a real DOI pattern; do not fake one."""
    assert "sci:doi" not in item["properties"]
    assert SCIENTIFIC_SCHEMA_URI not in item["stac_extensions"]


def test_the_doi_appears_once_assigned(granule_dir, browse_image, monkeypatch):
    monkeypatch.setattr("hls_composites.metadata.stac.DOI", "10.5067/HLS/HLSM30.001")
    meta = granule_metadata("14TPN", FEBRUARY, granule_dir, browse_image)

    assigned = to_stac_item(meta)

    assert assigned["properties"]["sci:doi"] == "10.5067/HLS/HLSM30.001"
    assert SCIENTIFIC_SCHEMA_URI in assigned["stac_extensions"]
    pystac.Item.from_dict(assigned).validate()


def test_every_geotiff_becomes_a_cog_asset(item):
    data = {
        key: asset
        for key, asset in item["assets"].items()
        if asset["roles"] == ["data"]
    }

    assert set(data) == {"NDVI", "ValidCount"}
    for asset in data.values():
        assert asset["type"] == pystac.MediaType.COG


def test_asset_hrefs_are_the_file_names(item):
    """Relative hrefs, so the item resolves beside its data in any bucket."""
    assert item["assets"]["NDVI"]["href"] == f"{GRANULE_ID}.NDVI.tif"


def test_geometry_matches_the_boundary(item):
    ring = item["geometry"]["coordinates"][0]

    assert item["geometry"]["type"] == "Polygon"
    # Five points: four corners, with the first repeated to close the ring.
    assert len(ring) == 5
    assert ring[0] == ring[-1]


def test_each_data_asset_declares_its_band(item):
    band = item["assets"]["NDVI"]["bands"][0]

    assert band["name"] == "NDVI"
    assert band["description"] == NDVI_DESCRIPTION
    assert band["data_type"] == "int16"
    assert band["nodata"] == -19999
    assert band["raster:scale"] == 1e-4
    assert band["raster:offset"] == 0.0


def test_an_unscaled_band_declares_no_scale(item):
    """ValidCount is a count, not an encoded physical quantity."""
    band = item["assets"]["ValidCount"]["bands"][0]

    assert band["data_type"] == "uint8"
    assert band["nodata"] == 255
    assert "raster:scale" not in band


def test_the_raster_extension_is_declared_for_the_scaled_bands(item):
    assert RASTER_SCHEMA_URI in item["stac_extensions"]


def test_item_declares_its_spatial_coverage(item):
    """12 of 16 pixels carry data."""
    assert item["properties"]["hls:spatial_coverage"] == 75.0


def test_each_band_reports_the_valid_percentage(item):
    """Every layer shares one mask, so each band reports the granule's coverage."""
    for key in ("NDVI", "ValidCount"):
        stats = item["assets"][key]["bands"][0]["statistics"]
        assert stats["valid_percent"] == 75.0


def test_item_records_when_it_was_produced(item):
    assert item["properties"]["created"] == "2026-09-03T12:00:00Z"


def test_item_carries_the_bbox_in_its_own_projection(item):
    """A 4x4 grid at 30 m, so 120 m on a side from the upper-left corner."""
    assert item["properties"]["proj:bbox"] == [
        ULX,
        ULY - 4 * PIXEL,
        ULX + 4 * PIXEL,
        ULY,
    ]


def test_item_validates_against_the_real_schemas(item):
    """Network-dependent, and therefore a test-time check only."""
    pystac.Item.from_dict(item).validate()
