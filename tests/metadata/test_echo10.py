import datetime as dt
from xml.etree import ElementTree

import pytest

from hls_composites.metadata.echo10 import to_echo10
from hls_composites.metadata.models import (
    COMPOSITING_ALGORITHM,
    DATASET_ID,
    DOI,
    granule_metadata,
)
from tests.metadata.conftest import FEBRUARY, GRANULE_ID

PRODUCED_AT = dt.datetime(2026, 9, 3, 12, 0, 0, tzinfo=dt.UTC)


@pytest.fixture
def root(granule_dir):
    meta = granule_metadata("14TPN", FEBRUARY, granule_dir, produced_at=PRODUCED_AT)
    return ElementTree.fromstring(to_echo10(meta))


def attribute(root, name):
    """The values of one AdditionalAttribute, by name."""
    for element in root.iter("AdditionalAttribute"):
        if element.findtext("Name") == name:
            return [value.text for value in element.iter("Value")]
    raise AssertionError(f"no AdditionalAttribute named {name}")


def test_document_is_a_granule(root):
    assert root.tag == "Granule"
    assert root.findtext("GranuleUR") == GRANULE_ID


def test_collection_is_identified_by_dataset_id(root):
    assert root.findtext("Collection/DataSetId") == DATASET_ID


def test_data_granule_describes_the_product(root):
    assert root.findtext("DataGranule/ProducerGranuleId") == GRANULE_ID
    assert root.findtext("DataGranule/DayNightFlag") == "DAY"
    assert int(root.findtext("DataGranule/DataGranuleSizeInBytes")) > 0


def test_temporal_range_spans_the_compositing_period(root):
    assert root.findtext("Temporal/RangeDateTime/BeginningDateTime").startswith(
        "2020-02-01"
    )
    assert root.findtext("Temporal/RangeDateTime/EndingDateTime").startswith(
        "2020-02-29"
    )


def test_spatial_boundary_has_four_points(root):
    points = root.findall(
        "Spatial/HorizontalSpatialDomain/Geometry/GPolygon/Boundary/Point"
    )

    assert len(points) == 4
    for point in points:
        assert -180 <= float(point.findtext("PointLongitude")) <= 180
        assert -90 <= float(point.findtext("PointLatitude")) <= 90


def test_platforms_cover_landsat_and_sentinel(root):
    """A composite draws from both, unlike a daily granule."""
    names = [element.text for element in root.iter("ShortName")]

    assert "LANDSAT-8" in names
    assert "Sentinel-2A" in names


def test_required_additional_attributes_are_present(root):
    for name in [
        "MGRS_TILE_ID",
        "SPATIAL_COVERAGE",
        "SPATIAL_RESOLUTION",
        "PROCESSING_TIME",
        "HORIZONTAL_CS_CODE",
        "HORIZONTAL_CS_NAME",
        "ULX",
        "ULY",
        "REF_SCALE_FACTOR",
        "ADD_OFFSET",
        "FILLVALUE",
        "QA_FILL_VALUE",
        "NCOLS",
        "NROWS",
        "PRODUCT_URI",
        "IDENTIFIER_PRODUCT_DOI",
        "IDENTIFIER_PRODUCT_DOI_AUTHORITY",
        "COMPOSITING_ALGORITHM",
        "COMPOSITING_START_DATE",
        "COMPOSITING_END_DATE",
    ]:
        assert attribute(root, name), f"{name} has no value"


def test_attribute_values_come_from_the_model(root):
    assert attribute(root, "MGRS_TILE_ID") == ["14TPN"]
    assert attribute(root, "SPATIAL_COVERAGE") == ["75"]
    assert attribute(root, "SPATIAL_RESOLUTION") == ["30.0"]
    assert attribute(root, "HORIZONTAL_CS_CODE") == ["EPSG:32614"]
    assert attribute(root, "NCOLS") == ["4"]
    assert attribute(root, "NROWS") == ["4"]
    assert attribute(root, "IDENTIFIER_PRODUCT_DOI") == [DOI]
    assert attribute(root, "COMPOSITING_ALGORITHM") == [COMPOSITING_ALGORITHM]


def test_compositing_dates_are_plain_calendar_dates(root):
    """The requirement specifies YYYY-MM-DD, not a timestamp."""
    assert attribute(root, "COMPOSITING_START_DATE") == ["2020-02-01"]
    assert attribute(root, "COMPOSITING_END_DATE") == ["2020-02-29"]


def test_element_order_matches_the_schema_sequence(root):
    """ECHO-10 uses xs:sequence, so a reordered document is invalid.

    Pinning the order here is what a golden-file comparison would have
    caught; a full golden file is not reproducible, because
    DataGranuleSizeInBytes depends on how the fixture happens to compress.
    """
    assert [child.tag for child in root] == [
        "GranuleUR",
        "InsertTime",
        "LastUpdate",
        "Collection",
        "DataGranule",
        "Temporal",
        "Spatial",
        "Platforms",
        "AdditionalAttributes",
        "OnlineAccessURLs",
        "OnlineResources",
        "DataFormat",
        "AssociatedBrowseImageUrls",
    ]


def test_data_format_is_declared(root):
    assert root.findtext("DataFormat") == "Cloud Optimized GeoTIFF (COG)"


def test_document_has_an_xml_declaration(granule_dir):
    meta = granule_metadata("14TPN", FEBRUARY, granule_dir)

    assert to_echo10(meta).startswith("<?xml")
