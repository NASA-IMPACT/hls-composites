import json
from xml.etree import ElementTree

from hls_composites.metadata.writer import write_metadata
from tests.metadata.conftest import FEBRUARY, GRANULE_ID, PLATFORMS


def test_writes_both_documents_into_the_granule_directory(granule_dir, browse_image):
    written = write_metadata(
        "14TPN", FEBRUARY, granule_dir, browse_image, platforms=PLATFORMS
    )

    assert [path.name for path in written] == [
        f"{GRANULE_ID}.cmr.xml",
        f"{GRANULE_ID}_stac.json",
    ]
    assert all(path.parent == granule_dir for path in written)


def test_the_xml_parses_and_names_the_granule(granule_dir, browse_image):
    xml_path, _ = write_metadata(
        "14TPN", FEBRUARY, granule_dir, browse_image, platforms=PLATFORMS
    )

    root = ElementTree.fromstring(xml_path.read_text())

    assert root.findtext("GranuleUR") == GRANULE_ID


def test_the_json_parses_and_names_the_granule(granule_dir, browse_image):
    _, json_path = write_metadata(
        "14TPN", FEBRUARY, granule_dir, browse_image, platforms=PLATFORMS
    )

    assert json.loads(json_path.read_text())["id"] == GRANULE_ID


def test_both_documents_agree_on_the_granule(granule_dir, browse_image):
    """The property the shared model buys: the two cannot drift apart."""
    xml_path, json_path = write_metadata(
        "14TPN", FEBRUARY, granule_dir, browse_image, platforms=PLATFORMS
    )

    root = ElementTree.fromstring(xml_path.read_text())
    item = json.loads(json_path.read_text())

    assert root.findtext("GranuleUR") == item["id"]
    epsg = next(
        element.findtext("Values/Value")
        for element in root.iter("AdditionalAttribute")
        if element.findtext("Name") == "HORIZONTAL_CS_CODE"
    )
    assert epsg == item["properties"]["proj:code"]


def test_metadata_files_are_not_described_as_assets(granule_dir, browse_image):
    """Only the GeoTIFFs are data; the documents describe them."""
    write_metadata("14TPN", FEBRUARY, granule_dir, browse_image, platforms=PLATFORMS)
    _, json_path = write_metadata(
        "14TPN", FEBRUARY, granule_dir, browse_image, platforms=PLATFORMS
    )

    item = json.loads(json_path.read_text())

    data = {k for k, v in item["assets"].items() if v["roles"] == ["data"]}
    assert data == {"NDVI", "ValidCount"}
