from datetime import date

from hls_composites.composite import DEFAULT_BANDS, asset_url
from hls_composites.models import Granule


def _granule(satellite: str) -> Granule:
    return Granule(
        path="s3://lp-prod-protected/HLSL30.020/HLS.L30.T55HDT.2026151T235621.v2.0/HLS.L30.T55HDT.2026151T235621.v2.0",
        satellite=satellite,
        date=date(2026, 5, 31),
    )


def test_default_bands_matches_prototype():
    assert DEFAULT_BANDS == ["red", "green", "blue", "nir_narrow", "swir_1", "swir_2", "Fmask"]


def test_asset_url_l30_red_band():
    url = asset_url(_granule("L30"), "red")
    assert url == (
        "s3://lp-prod-protected/HLSL30.020/HLS.L30.T55HDT.2026151T235621.v2.0/"
        "HLS.L30.T55HDT.2026151T235621.v2.0.B04.tif"
    )


def test_asset_url_s30_nir_narrow_uses_b8a():
    url = asset_url(_granule("S30"), "nir_narrow")
    assert url.endswith(".B8A.tif")


def test_asset_url_l30_nir_narrow_uses_b05():
    url = asset_url(_granule("L30"), "nir_narrow")
    assert url.endswith(".B05.tif")


def test_asset_url_fmask_same_code_both_satellites():
    assert asset_url(_granule("L30"), "Fmask").endswith(".Fmask.tif")
    assert asset_url(_granule("S30"), "Fmask").endswith(".Fmask.tif")
