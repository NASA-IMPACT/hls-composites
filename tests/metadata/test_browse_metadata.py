"""How the browse images are referenced from both metadata documents."""

from xml.etree import ElementTree

import pytest

from hls_composites.metadata.echo10 import to_echo10
from hls_composites.metadata.models import browse_description, granule_metadata
from hls_composites.metadata.stac import to_stac_item
from tests.metadata.conftest import FEBRUARY, GRANULE_ID, PLATFORMS


@pytest.fixture
def several_browse_images(granule_dir):
    paths = [granule_dir / f"{GRANULE_ID}.{name}.png" for name in ("EVI", "NDVI")]
    for path in paths:
        path.write_bytes(b"png")
    return paths


@pytest.fixture
def with_browse(rasters, several_browse_images):
    return granule_metadata(
        "14TPN", FEBRUARY, rasters, several_browse_images, platforms=PLATFORMS
    )


@pytest.fixture
def without_browse(rasters):
    return granule_metadata("14TPN", FEBRUARY, rasters, [], platforms=PLATFORMS)


def test_description_names_the_index(several_browse_images):
    assert browse_description(several_browse_images[0]) == "EVI browse image"


class TestEcho10Browse:
    def test_browse_urls_are_left_empty(self, with_browse):
        """As the daily products leave them."""
        root = ElementTree.fromstring(to_echo10(with_browse))

        container = root.find("AssociatedBrowseImageUrls")
        assert container is not None
        assert list(container) == []


class TestStacBrowse:
    def test_one_thumbnail_asset_per_image(self, with_browse):
        assets = to_stac_item(with_browse)["assets"]

        evi = assets["EVI_browse"]
        assert evi["href"] == f"{GRANULE_ID}.EVI.png"
        assert evi["type"] == "image/png"
        assert evi["roles"] == ["thumbnail"]
        assert evi["description"] == "EVI browse image"
        assert assets["NDVI_browse"]["href"] == f"{GRANULE_ID}.NDVI.png"

    def test_the_images_are_not_also_data_assets(self, with_browse):
        """Data assets are the written GeoTIFFs, so each image appears once."""
        assets = to_stac_item(with_browse)["assets"]

        data = {key for key, value in assets.items() if value["roles"] == ["data"]}
        assert data == {"NDVI", "ValidCount"}

    def test_no_images_adds_no_thumbnails(self, without_browse):
        assets = to_stac_item(without_browse)["assets"]

        assert not [a for a in assets.values() if "thumbnail" in a["roles"]]
