"""How the browse image is referenced from both metadata documents."""

from xml.etree import ElementTree

import pystac
import pytest

from hls_composites.metadata.echo10 import to_echo10
from hls_composites.metadata.models import BROWSE_DESCRIPTION, granule_metadata
from hls_composites.metadata.stac import to_stac_item
from tests.metadata.conftest import FEBRUARY, GRANULE_ID


@pytest.fixture
def browse(granule_dir):
    path = granule_dir / f"{GRANULE_ID}.jpg"
    path.write_bytes(b"jpeg")
    return path


@pytest.fixture
def with_browse(granule_dir, browse):
    return granule_metadata("14TPN", FEBRUARY, granule_dir, browse_image=browse)


@pytest.fixture
def without_browse(granule_dir):
    return granule_metadata("14TPN", FEBRUARY, granule_dir)


class TestEcho10Browse:
    def test_browse_url_is_declared(self, with_browse):
        root = ElementTree.fromstring(to_echo10(with_browse))

        url = root.find("AssociatedBrowseImageUrls/ProviderBrowseUrl")
        assert url is not None
        assert url.findtext("URL").endswith(f"{GRANULE_ID}.jpg")
        assert url.findtext("Description") == BROWSE_DESCRIPTION

    def test_element_stays_empty_without_a_browse_image(self, without_browse):
        root = ElementTree.fromstring(to_echo10(without_browse))

        assert root.find("AssociatedBrowseImageUrls") is not None
        assert root.find("AssociatedBrowseImageUrls/ProviderBrowseUrl") is None


class TestStacBrowse:
    def test_thumbnail_asset_is_added(self, with_browse):
        thumbnail = to_stac_item(with_browse)["assets"]["thumbnail"]

        assert thumbnail["href"] == f"{GRANULE_ID}.jpg"
        assert thumbnail["type"] == "image/jpeg"
        assert thumbnail["roles"] == ["thumbnail"]
        assert thumbnail["description"] == BROWSE_DESCRIPTION

    def test_no_thumbnail_without_a_browse_image(self, without_browse):
        assert "thumbnail" not in to_stac_item(without_browse)["assets"]

    def test_the_jpeg_is_not_also_a_data_asset(self, with_browse):
        """Data assets are globbed from *.tif, so the jpg appears once."""
        assets = to_stac_item(with_browse)["assets"]

        data = {key for key, value in assets.items() if value["roles"] == ["data"]}
        assert data == {"NDVI", "ValidCount"}

    def test_item_still_validates(self, with_browse):
        pystac.Item.from_dict(to_stac_item(with_browse)).validate()
