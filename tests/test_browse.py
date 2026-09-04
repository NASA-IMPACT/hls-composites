import numpy as np
import pytest
import xarray as xr
from PIL import Image

from hls_composites.browse import (
    HIGH_THRESHOLD,
    IMAGE_SIZE,
    LOW_THRESHOLD,
    stretch,
    write_browse_image,
)


def _rgb_dataset(size: int, red: int = 500, green: int = 500, blue: int = 500):
    """A minimal computed composite carrying only the browse bands."""
    dims = ("y", "x")
    return xr.Dataset(
        {
            "R": (dims, np.full((size, size), red, np.int16)),
            "G": (dims, np.full((size, size), green, np.int16)),
            "B": (dims, np.full((size, size), blue, np.int16)),
        }
    )


class TestStretch:
    def test_low_threshold_maps_to_black(self):
        assert stretch(np.full((1, 1, 1), LOW_THRESHOLD, np.int16))[0, 0, 0] == 0

    def test_fill_is_black(self):
        """Fill (-9999) is far below the threshold and must not break the log."""
        assert stretch(np.full((1, 1, 1), -9999, np.int16))[0, 0, 0] == 0

    def test_high_threshold_saturates(self):
        assert stretch(np.full((1, 1, 1), HIGH_THRESHOLD, np.int16))[0, 0, 0] == 255

    def test_above_high_threshold_saturates(self):
        assert stretch(np.full((1, 1, 1), 30000, np.int16))[0, 0, 0] == 255

    def test_midpoint_is_between(self):
        value = stretch(np.full((1, 1, 1), 1000, np.int16))[0, 0, 0]
        assert 0 < value < 255

    def test_is_monotonic(self):
        values = stretch(np.array([[[200, 800, 3000, 7000]]], dtype=np.int16))

        assert list(values[0, 0]) == sorted(values[0, 0])

    def test_output_is_uint8(self):
        assert stretch(np.full((3, 2, 2), 500, np.int16)).dtype == np.uint8


class TestWriteBrowseImage:
    def test_writes_a_jpeg_of_the_configured_size(self, tmp_path):
        path = tmp_path / "granule.jpg"

        write_browse_image(_rgb_dataset(4), path)

        with Image.open(path) as img:
            assert img.format == "JPEG"
            assert img.size == (IMAGE_SIZE, IMAGE_SIZE)
            assert img.mode == "RGB"

    def test_channel_order_is_red_green_blue(self, tmp_path):
        """A red-dominant scene must render red, not blue."""
        path = tmp_path / "granule.jpg"

        write_browse_image(_rgb_dataset(4, red=7500, green=100, blue=100), path)

        with Image.open(path) as img:
            red, green, blue = img.convert("RGB").getpixel((500, 500))
        assert red > 200
        assert green < 60
        assert blue < 60

    def test_returns_the_written_path(self, tmp_path):
        path = tmp_path / "granule.jpg"

        assert write_browse_image(_rgb_dataset(4), path) == path

    def test_missing_browse_band_is_an_error(self, tmp_path):
        incomplete = _rgb_dataset(4).drop_vars("G")

        with pytest.raises(KeyError):
            write_browse_image(incomplete, tmp_path / "granule.jpg")
