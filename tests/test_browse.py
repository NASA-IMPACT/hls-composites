import numpy as np
import pytest
import xarray as xr
from PIL import Image

from hls_composites.browse import (
    COLOR_STOPS,
    FILL_ENTRY,
    IMAGE_SIZE,
    RAMP_STEPS,
    browse_path,
    color_table,
    downsample,
    palette_entries,
    ramp_palette,
    write_browse_image,
    write_browse_images,
)
from hls_composites.indices import EVI, NBR, NDVI

INDEX = NDVI()


def _encoded(*values: int) -> np.ndarray:
    return np.array([values], dtype=np.int16)


def _index_dataset(size: int, value: int = 5000) -> xr.Dataset:
    """A minimal computed composite carrying the default indices and a QA layer."""
    dims = ("y", "x")
    data = np.full((size, size), value, np.int16)
    return xr.Dataset(
        {
            "EVI": (dims, data),
            "EVI_std": (dims, data),
            "NBR": (dims, data),
            "NDVI": (dims, data),
            "ValidCount": (dims, np.full((size, size), 3, np.uint8)),
        }
    )


class TestColorTable:
    def test_ends_of_the_ramp_are_the_first_and_last_stops(self):
        rgb = color_table(np.array([0.0, 1.0]))

        assert tuple(rgb[0]) == COLOR_STOPS[0][1]
        assert tuple(rgb[1]) == COLOR_STOPS[-1][1]

    def test_values_outside_the_range_clamp(self):
        rgb = color_table(np.array([-0.5, 0.0, 1.0, 3.0]))

        assert tuple(rgb[0]) == tuple(rgb[1])
        assert tuple(rgb[2]) == tuple(rgb[3])

    def test_break_at_two_tenths_is_hard(self):
        """Just below 0.2 is brown; 0.2 itself starts the green ramp."""
        below, at = color_table(np.array([0.1999, 0.2]))

        red, green, blue = (int(c) for c in below)
        assert red > green > blue
        assert tuple(at) == (190, 222, 125)

    def test_vegetation_darkens_toward_one(self):
        greens = color_table(np.array([0.3, 0.5, 0.7, 0.9]))[:, 1]

        assert list(greens) == sorted(greens, reverse=True)

    def test_output_is_uint8_rgb(self):
        rgb = color_table(np.zeros((2, 3)))

        assert rgb.dtype == np.uint8
        assert rgb.shape == (2, 3, 3)


class TestRampPalette:
    def test_one_entry_per_step_plus_fill(self):
        palette = ramp_palette()

        assert palette.shape == (RAMP_STEPS + 1, 3)
        assert palette.dtype == np.uint8

    def test_steps_follow_the_ramp(self):
        palette = ramp_palette()

        assert tuple(palette[0]) == tuple(color_table(np.array([0.5 / RAMP_STEPS]))[0])
        assert palette[RAMP_STEPS - 1, 1] < palette[RAMP_STEPS // 2, 1]

    def test_is_the_same_every_time(self):
        assert np.array_equal(ramp_palette(), ramp_palette())


class TestDownsample:
    def test_averages_in_physical_units(self):
        encoded = np.array([[2000, 4000], [6000, 8000]], np.int16)

        values, coverage = downsample(encoded, INDEX, size=1)

        assert values[0, 0] == pytest.approx(0.5)
        assert coverage[0, 0] == 1.0

    def test_fill_is_excluded_from_the_mean(self):
        encoded = np.array([[INDEX.fill_value, 4000], [6000, 8000]], np.int16)

        values, coverage = downsample(encoded, INDEX, size=1)

        assert values[0, 0] == pytest.approx(0.6)
        assert coverage[0, 0] == pytest.approx(0.75)

    def test_all_fill_has_no_coverage(self):
        encoded = np.full((2, 2), INDEX.fill_value, np.int16)

        _, coverage = downsample(encoded, INDEX, size=1)

        assert coverage[0, 0] == 0.0


class TestPaletteEntries:
    def test_ends_of_the_range_are_the_first_and_last_steps(self):
        entries = palette_entries(np.array([0.0, 1.0]), np.ones(2))

        assert list(entries) == [0, RAMP_STEPS - 1]

    def test_values_outside_the_range_clamp(self):
        entries = palette_entries(np.array([-0.4, 0.0, 1.0, 1.3]), np.ones(4))

        assert entries[0] == entries[1]
        assert entries[2] == entries[3]

    def test_low_coverage_is_fill(self):
        entries = palette_entries(
            np.array([0.5, 0.5, np.nan]), np.array([0.49, 0.5, 0.0])
        )

        assert list(entries) == [FILL_ENTRY, entries[1], FILL_ENTRY]
        assert entries[1] != FILL_ENTRY

    def test_output_is_uint8(self):
        assert palette_entries(np.zeros((2, 2)), np.ones((2, 2))).dtype == np.uint8


class TestWriteBrowseImage:
    def test_writes_a_png_of_the_configured_size(self, tmp_path):
        path = tmp_path / "granule.NDVI.png"

        write_browse_image(np.full((4, 4), 5000, np.int16), INDEX, path)

        with Image.open(path) as img:
            assert img.format == "PNG"
            assert img.size == (IMAGE_SIZE, IMAGE_SIZE)

    def test_fill_stays_transparent_in_the_file(self, tmp_path):
        """Left half fill, right half data."""
        path = tmp_path / "granule.NDVI.png"
        encoded = np.full((4, 4), 5000, np.int16)
        encoded[:, :2] = INDEX.fill_value

        write_browse_image(encoded, INDEX, path)

        with Image.open(path) as img:
            rgba = img.convert("RGBA")
            assert rgba.getpixel((100, 500))[3] == 0
            assert rgba.getpixel((900, 500))[3] == 255

    def test_uses_the_fixed_palette(self, tmp_path):
        """Two scenes with different value distributions share one palette."""
        sparse, dense = tmp_path / "a.NDVI.png", tmp_path / "b.NDVI.png"

        write_browse_image(np.full((4, 4), 1000, np.int16), INDEX, sparse)
        write_browse_image(np.full((4, 4), 9000, np.int16), INDEX, dense)

        with Image.open(sparse) as a, Image.open(dense) as b:
            assert a.mode == b.mode == "P"
            assert a.getpalette() == b.getpalette() == ramp_palette().ravel().tolist()

    def test_dense_vegetation_renders_green(self, tmp_path):
        path = tmp_path / "granule.NDVI.png"

        write_browse_image(np.full((4, 4), 6000, np.int16), INDEX, path)

        with Image.open(path) as img:
            red, green, blue, _ = img.convert("RGBA").getpixel((500, 500))
        assert green > red + 60
        assert green > blue + 60

    def test_returns_the_written_path(self, tmp_path):
        path = tmp_path / "granule.NDVI.png"

        assert write_browse_image(np.zeros((4, 4), np.int16), INDEX, path) == path


class TestWriteBrowseImages:
    def test_one_image_per_index(self, tmp_path):
        granule_dir = tmp_path / "HLS.M30.T14TPN.2020032.2020060.v2.0"
        granule_dir.mkdir()

        paths = write_browse_images(_index_dataset(4), granule_dir)

        assert paths == [
            browse_path(granule_dir, EVI()),
            browse_path(granule_dir, NBR()),
            browse_path(granule_dir, NDVI()),
        ]
        assert all(path.exists() for path in paths)

    def test_images_are_named_for_the_granule_and_index(self, tmp_path):
        granule_dir = tmp_path / "HLS.M30.T14TPN.2020032.2020060.v2.0"

        path = browse_path(granule_dir, EVI())

        assert path == granule_dir / "HLS.M30.T14TPN.2020032.2020060.v2.0.EVI.png"

    def test_a_composite_without_indices_has_none(self, tmp_path):
        bands = _index_dataset(4).drop_vars(["EVI", "EVI_std", "NBR", "NDVI"])

        assert write_browse_images(bands, tmp_path) == []
