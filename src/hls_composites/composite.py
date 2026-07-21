"""Composite creation: masking, median-EVI2 selection, aggregation."""

from hls_composites.models import Granule

DEFAULT_BANDS = ["red", "green", "blue", "nir_narrow", "swir_1", "swir_2", "Fmask"]

BAND_CODE: dict[str, dict[str, str]] = {
    "L30": {
        "red": "B04",
        "green": "B03",
        "blue": "B02",
        "nir_narrow": "B05",
        "swir_1": "B06",
        "swir_2": "B07",
        "Fmask": "Fmask",
    },
    "S30": {
        "red": "B04",
        "green": "B03",
        "blue": "B02",
        "nir_narrow": "B8A",
        "swir_1": "B11",
        "swir_2": "B12",
        "Fmask": "Fmask",
    },
}

QA_BIT = {
    "cirrus": 0,
    "cloud": 1,
    "adj_cloud": 2,
    "cloud_shadow": 3,
    "snowice": 4,
    "water": 5,
    "aerosol_low": 6,
    "aerosol_high": 7,
}

SR_SCALE = 0.0001
SR_FILL = -9999
QA_FILL = 255


def asset_url(granule: Granule, band: str) -> str:
    band_code = BAND_CODE[granule.satellite][band]
    return f"{granule.path}.{band_code}.tif"
