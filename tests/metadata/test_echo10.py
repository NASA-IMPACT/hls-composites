import datetime as dt
from dataclasses import replace
from itertools import pairwise
from xml.etree import ElementTree

import numpy as np
import pytest
from lxml import etree

from hls_composites.metadata.echo10 import parse_platforms, to_echo10, validate_echo10
from hls_composites.metadata.models import (
    COMPOSITING_ALGORITHM,
    DATASET_ID,
    DOI,
    InputGranule,
    granule_metadata,
)
from tests.metadata.conftest import FEBRUARY, GRANULE_ID, PLATFORMS, SPLIT_FOOTPRINT

PRODUCED_AT = dt.datetime(2026, 9, 3, 12, 0, 0, tzinfo=dt.UTC)


@pytest.fixture
def meta(granule_dir, browse_images):
    return granule_metadata(
        "14TPN",
        FEBRUARY,
        granule_dir,
        browse_images,
        platforms=PLATFORMS,
        produced_at=PRODUCED_AT,
    )


@pytest.fixture
def root(meta):
    return ElementTree.fromstring(to_echo10(meta))


def signed_area(ring):
    """Shoelace sum: positive for a counter-clockwise ring."""
    closed = [*ring, ring[0]]
    return sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in pairwise(closed))


def attribute(root, name):
    """The values of one AdditionalAttribute, by name."""
    for element in root.iter("AdditionalAttribute"):
        if element.findtext("Name") == name:
            return [value.text for value in element.iter("Value")]
    raise AssertionError(f"no AdditionalAttribute named {name}")


@pytest.mark.parametrize(
    "changes",
    [
        {},
        {"browse_images": []},
        {
            "inputs": [
                InputGranule(f"HLS.L30.T14TPN.2020{day:03d}T171803.v2.0", "href")
                for day in (33, 49)
            ]
        },
        {"footprint": SPLIT_FOOTPRINT},
    ],
    ids=["default", "no-browse-images", "with-inputs", "antimeridian"],
)
def test_document_conforms_to_the_granule_schema(meta, changes):
    validate_echo10(to_echo10(replace(meta, **changes)))


def test_a_nonconforming_document_is_rejected(root):
    root.remove(root.find("GranuleUR"))

    with pytest.raises(etree.DocumentInvalid, match="GranuleUR"):
        validate_echo10(ElementTree.tostring(root, encoding="unicode"))


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


def polygons(root):
    """Each GPolygon's boundary as ``(longitude, latitude)`` points."""
    return [
        [
            (
                float(point.findtext("PointLongitude")),
                float(point.findtext("PointLatitude")),
            )
            for point in polygon.iterfind("Boundary/Point")
        ]
        for polygon in root.iterfind(
            "Spatial/HorizontalSpatialDomain/Geometry/GPolygon"
        )
    ]


def test_spatial_boundary_is_the_footprint(meta, root):
    [boundary] = polygons(root)

    assert len(boundary) == 4
    np.testing.assert_allclose(sorted(boundary), sorted(meta.footprint[0]), atol=1e-8)


def test_spatial_boundary_is_clockwise(root):
    """CMR rejects an ECHO-10 boundary listed counter-clockwise."""
    [boundary] = polygons(root)

    assert signed_area(boundary) < 0


def test_a_split_footprint_is_one_polygon_per_side(meta):
    root = ElementTree.fromstring(to_echo10(replace(meta, footprint=SPLIT_FOOTPRINT)))

    boundaries = polygons(root)

    assert [sorted(ring) for ring in boundaries] == [
        sorted(ring) for ring in SPLIT_FOOTPRINT
    ]
    assert all(signed_area(ring) < 0 for ring in boundaries)


def test_add_offset_is_an_integer(root):
    """The collection declares ADD_OFFSET as INT, which rejects "0.0"."""
    assert attribute(root, "ADD_OFFSET") == ["0"]


def test_platforms_are_the_ones_given(root):
    """A composite can draw from several, unlike a daily granule."""
    platforms = [
        (
            platform.findtext("ShortName"),
            platform.findtext("Instruments/Instrument/ShortName"),
        )
        for platform in root.iterfind("Platforms/Platform")
    ]

    assert platforms == PLATFORMS


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
        "VALIDCOUNT_FILLVALUE",
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


def test_spatial_coverage_is_rounded_to_whole_percent(granule_dir, browse_images):
    """The daily products declare an integer percent, so a composite does too."""
    meta = granule_metadata(
        "14TPN",
        FEBRUARY,
        granule_dir,
        browse_images,
        platforms=PLATFORMS,
        produced_at=PRODUCED_AT,
    )
    sparse = replace(meta, spatial_coverage=4.746)

    root = ElementTree.fromstring(to_echo10(sparse))

    assert attribute(root, "SPATIAL_COVERAGE") == ["5"]


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


def test_document_has_an_xml_declaration(granule_dir, browse_images):
    meta = granule_metadata(
        "14TPN", FEBRUARY, granule_dir, browse_images, platforms=PLATFORMS
    )

    assert to_echo10(meta).startswith("<?xml")


def _input_document(*platforms: tuple[str, str]) -> bytes:
    """An input granule's ECHO-10, trimmed to the parts `parse_platforms` reads."""
    blocks = "".join(
        f"<Platform><ShortName>{platform}</ShortName>"
        f"<Instruments><Instrument><ShortName>{instrument}</ShortName>"
        "</Instrument></Instruments></Platform>"
        for platform, instrument in platforms
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        "<Granule><GranuleUR>HLS.L30.T52UDG.2026242T023640.v2.0</GranuleUR>"
        f"<Platforms>{blocks}</Platforms>"
        "<AdditionalAttributes><AdditionalAttribute><Name>SENSOR</Name>"
        "<Values><Value>OLI_TIRS</Value></Values></AdditionalAttribute>"
        "</AdditionalAttributes></Granule>"
    ).encode()


def test_parse_platforms_reads_every_platform():
    document = _input_document(("Sentinel-2B", "Sentinel-2 MSI"), ("LANDSAT-8", "OLI"))

    assert parse_platforms(document) == [
        ("Sentinel-2B", "Sentinel-2 MSI"),
        ("LANDSAT-8", "OLI"),
    ]


def test_parse_platforms_round_trips_what_to_echo10_writes(granule_dir, browse_images):
    """The inputs and the composite share one schema, so the reader reads both."""
    platforms = [("LANDSAT-9", "OLI"), ("Sentinel-2C", "Sentinel-2 MSI")]
    meta = granule_metadata(
        "14TPN", FEBRUARY, granule_dir, browse_images, platforms=platforms
    )

    assert parse_platforms(to_echo10(meta).encode()) == platforms


def test_parse_platforms_refuses_a_document_naming_no_platform():
    with pytest.raises(ValueError, match="no platform"):
        parse_platforms(_input_document())
