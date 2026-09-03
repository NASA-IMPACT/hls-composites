"""A miniature written composite, standing in for a real granule directory."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from hls_composites.composite import VALID_COUNT_FILL
from hls_composites.models import DateRange

GRANULE_ID = "HLS.M30.T14TPN.2020032.2020060.v2.0"
FEBRUARY = DateRange(date(2020, 2, 1), date(2020, 2, 29))
EPSG = 32614
# A 4x4 grid at 30 m, upper-left at a round UTM coordinate.
ULX = 300000.0
ULY = 4600000.0
PIXEL = 30.0


def _write(path: Path, array: np.ndarray, nodata: float) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=f"EPSG:{EPSG}",
        transform=from_origin(ULX, ULY, PIXEL, PIXEL),
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)


@pytest.fixture
def granule_dir(tmp_path: Path) -> Path:
    """A granule directory holding NDVI and ValidCount rasters.

    ValidCount is 12 of 16 pixels valid, so SPATIAL_COVERAGE is 75.
    """
    dest = tmp_path / GRANULE_ID
    dest.mkdir()

    valid = np.full((4, 4), 3, dtype=np.uint8)
    valid[0, :] = VALID_COUNT_FILL  # one row of 4 is fill
    _write(dest / f"{GRANULE_ID}.ValidCount.tif", valid, VALID_COUNT_FILL)

    ndvi = np.full((4, 4), 5000, dtype=np.int16)
    _write(dest / f"{GRANULE_ID}.NDVI.tif", ndvi, -19999)

    return dest
