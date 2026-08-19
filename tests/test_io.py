from datetime import date

import numpy as np
import rasterio
import xarray as xr
from rasterio.transform import from_origin

from hls_composites.io import composite_id, write_composite
from hls_composites.models import DateRange

CRS = "EPSG:32614"
TRANSFORM = from_origin(300000, 4500000, 30, 30)
SIZE = 1024  # > BLOCK_SIZE(512) so 512 internal tiling is genuine (multiple tiles)


def _georef_dataset() -> xr.Dataset:
    x = 300000 + (np.arange(SIZE) + 0.5) * 30
    y = 4500000 - (np.arange(SIZE) + 0.5) * 30
    values = (np.arange(SIZE * SIZE, dtype=np.int16) % 2000 - 1000).reshape(SIZE, SIZE)

    ndvi = xr.DataArray(values, dims=("y", "x"), coords={"y": y, "x": x})
    ndvi.attrs["nodata"] = -19999
    ndvi.attrs["scale_factor"] = 1e-4
    ndvi_std = ndvi.copy()
    valid_count = xr.DataArray(
        (values % 4).astype(np.uint8), dims=("y", "x"), coords={"y": y, "x": x}
    )
    valid_count.attrs["nodata"] = 255
    doy = valid_count.copy()

    ds = xr.Dataset(
        {"NDVI": ndvi, "NDVI_std": ndvi_std, "ValidCount": valid_count, "DOY": doy}
    )
    return ds.rio.write_crs(CRS)


def test_composite_id_follows_prototype_naming():
    date_range = DateRange(start=date(2020, 7, 1), end=date(2020, 7, 31))
    assert composite_id("14TPN", date_range) == "HLS.M30.T14TPN.2020183.2020213.v2.0"


def test_write_composite_creates_named_dir_and_files(tmp_path):
    date_range = DateRange(start=date(2020, 7, 1), end=date(2020, 7, 31))
    dest = write_composite(_georef_dataset(), tmp_path, "14TPN", date_range)

    granule_id = "HLS.M30.T14TPN.2020183.2020213.v2.0"
    assert dest == tmp_path / granule_id
    for var in ("NDVI", "NDVI_std", "ValidCount", "DOY"):
        assert (dest / f"{granule_id}.{var}.tif").exists()


def test_written_geotiff_is_internally_tiled_at_512(tmp_path):
    date_range = DateRange(start=date(2020, 7, 1), end=date(2020, 7, 31))
    dest = write_composite(_georef_dataset(), tmp_path, "14TPN", date_range)

    with rasterio.open(dest / "HLS.M30.T14TPN.2020183.2020213.v2.0.NDVI.tif") as src:
        assert src.profile["tiled"] is True
        assert src.profile["blockxsize"] == 512
        assert src.profile["blockysize"] == 512
        # Default creation options compress with LZW.
        assert src.profile["compress"] == "lzw"


def test_written_geotiff_round_trips_dtype_nodata_crs_and_scale(tmp_path):
    date_range = DateRange(start=date(2020, 7, 1), end=date(2020, 7, 31))
    ds = _georef_dataset()
    dest = write_composite(ds, tmp_path, "14TPN", date_range)
    prefix = dest / "HLS.M30.T14TPN.2020183.2020213.v2.0"

    with rasterio.open(f"{prefix}.NDVI.tif") as src:
        assert src.dtypes[0] == "int16"
        assert src.nodata == -19999
        assert src.crs == rasterio.crs.CRS.from_string(CRS)
        assert src.transform == TRANSFORM
        assert src.scales[0] == 1e-4
        np.testing.assert_array_equal(src.read(1), ds["NDVI"].values)

    # Aux layers reserve the uint8 max as their fill, and the writer must
    # stamp it on the image.
    for var in ("ValidCount", "DOY"):
        with rasterio.open(f"{prefix}.{var}.tif") as src:
            assert src.dtypes[0] == "uint8"
            assert src.nodata == 255
            assert src.scales[0] == 1.0


def test_written_geotiff_omits_nodata_when_the_variable_declares_none(tmp_path):
    date_range = DateRange(start=date(2020, 7, 1), end=date(2020, 7, 31))
    ds = _georef_dataset()
    del ds["DOY"].attrs["nodata"]
    dest = write_composite(ds, tmp_path, "14TPN", date_range)

    with rasterio.open(dest / "HLS.M30.T14TPN.2020183.2020213.v2.0.DOY.tif") as src:
        assert src.nodata is None
