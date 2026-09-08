"""Input provenance and MGRS fields: which granules went in, and where."""

from datetime import date
from xml.etree import ElementTree

import pystac
import pytest

from hls_composites.metadata.echo10 import to_echo10
from hls_composites.metadata.models import (
    CMR_STAC_BASE,
    granule_metadata,
    mgrs_fields,
)
from hls_composites.metadata.stac import to_stac_item
from hls_composites.models import Granule
from tests.metadata.conftest import FEBRUARY

INPUTS = [
    Granule(
        "s3://lp-prod-protected/HLSS30.020/HLS.S30.T14TPN.2020032T171219.v2.0",
        "S30",
        date(2020, 2, 1),
    ),
    Granule(
        "s3://lp-prod-protected/HLSL30.020/HLS.L30.T14TPN.2020040T171219.v2.0",
        "L30",
        date(2020, 2, 9),
    ),
]


@pytest.fixture
def meta(granule_dir):
    return granule_metadata("14TPN", FEBRUARY, granule_dir, inputs=INPUTS)


class TestMgrsFields:
    def test_splits_a_tile_id(self):
        assert mgrs_fields("14TPN") == (14, "T", "PN")

    def test_single_digit_zone(self):
        assert mgrs_fields("1CAB") == (1, "C", "AB")

    @pytest.mark.parametrize("tile", ["", "14T", "14TPNX", "TPN", "14IPN"])
    def test_rejects_malformed_tiles(self, tile):
        """I and O are not MGRS latitude bands."""
        with pytest.raises(ValueError, match="tile"):
            mgrs_fields(tile)


class TestInputProvenance:
    def test_inputs_are_recorded_by_granule_id(self, meta):
        assert [item.granule_id for item in meta.inputs] == [
            "HLS.S30.T14TPN.2020032T171219.v2.0",
            "HLS.L30.T14TPN.2020040T171219.v2.0",
        ]

    def test_hrefs_point_at_the_cmr_stac_item(self, meta):
        """Collection IDs verified against the live LPCLOUD catalog."""
        assert meta.inputs[0].stac_href == (
            f"{CMR_STAC_BASE}/HLSS30_2.0/items/HLS.S30.T14TPN.2020032T171219.v2.0"
        )
        assert meta.inputs[1].stac_href == (
            f"{CMR_STAC_BASE}/HLSL30_2.0/items/HLS.L30.T14TPN.2020040T171219.v2.0"
        )

    def test_no_inputs_is_an_empty_list(self, granule_dir):
        assert granule_metadata("14TPN", FEBRUARY, granule_dir).inputs == []


class TestEcho10Provenance:
    def test_input_granules_is_a_multi_valued_attribute(self, meta):
        root = ElementTree.fromstring(to_echo10(meta))

        values = next(
            [v.text for v in element.iter("Value")]
            for element in root.iter("AdditionalAttribute")
            if element.findtext("Name") == "INPUT_GRANULES"
        )
        assert values == [item.granule_id for item in meta.inputs]

    def test_attribute_is_omitted_without_inputs(self, granule_dir):
        meta = granule_metadata("14TPN", FEBRUARY, granule_dir)
        root = ElementTree.fromstring(to_echo10(meta))

        names = [
            element.findtext("Name") for element in root.iter("AdditionalAttribute")
        ]
        assert "INPUT_GRANULES" not in names


class TestStacProvenance:
    def test_one_derived_from_link_per_input(self, meta):
        item = to_stac_item(meta)

        derived = [link for link in item["links"] if link["rel"] == "derived_from"]
        assert [link["title"] for link in derived] == [
            item_.granule_id for item_ in meta.inputs
        ]
        assert [link["href"] for link in derived] == [
            item_.stac_href for item_ in meta.inputs
        ]

    def test_mgrs_fields_are_queryable(self, meta):
        item = to_stac_item(meta)

        assert item["properties"]["mgrs:utm_zone"] == 14
        assert item["properties"]["mgrs:latitude_band"] == "T"
        assert item["properties"]["mgrs:grid_square"] == "PN"

    def test_item_still_validates(self, meta):
        pystac.Item.from_dict(to_stac_item(meta)).validate()
