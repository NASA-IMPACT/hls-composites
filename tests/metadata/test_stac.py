import datetime as dt

import pystac
import pytest

from hls_composites.metadata.models import granule_metadata
from hls_composites.metadata.stac import (
    PROJECTION_SCHEMA_URI,
    SCIENTIFIC_SCHEMA_URI,
    to_stac_item,
)
from tests.metadata.conftest import EPSG, FEBRUARY, GRANULE_ID

PRODUCED_AT = dt.datetime(2026, 9, 3, 12, 0, 0, tzinfo=dt.UTC)


@pytest.fixture
def item(granule_dir):
    meta = granule_metadata("14TPN", FEBRUARY, granule_dir, produced_at=PRODUCED_AT)
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


def test_the_doi_appears_once_assigned(granule_dir, monkeypatch):
    monkeypatch.setattr("hls_composites.metadata.stac.DOI", "10.5067/HLS/HLSM30.001")
    meta = granule_metadata("14TPN", FEBRUARY, granule_dir)

    assigned = to_stac_item(meta)

    assert assigned["properties"]["sci:doi"] == "10.5067/HLS/HLSM30.001"
    assert SCIENTIFIC_SCHEMA_URI in assigned["stac_extensions"]
    pystac.Item.from_dict(assigned).validate()


def test_every_geotiff_becomes_a_cog_asset(item):
    assets = item["assets"]

    assert set(assets) == {"NDVI", "ValidCount"}
    for asset in assets.values():
        assert asset["type"] == pystac.MediaType.COG
        assert asset["roles"] == ["data"]


def test_asset_hrefs_are_the_file_names(item):
    """Relative hrefs, so the item resolves beside its data in any bucket."""
    assert item["assets"]["NDVI"]["href"] == f"{GRANULE_ID}.NDVI.tif"


def test_geometry_matches_the_boundary(item):
    ring = item["geometry"]["coordinates"][0]

    assert item["geometry"]["type"] == "Polygon"
    # Five points: four corners, with the first repeated to close the ring.
    assert len(ring) == 5
    assert ring[0] == ring[-1]


def test_item_validates_against_the_real_schemas(item):
    """Network-dependent, and therefore a test-time check only."""
    pystac.Item.from_dict(item).validate()
