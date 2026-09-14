"""Pseudocolor browse images, one per spectral index.

Styled after the MODIS vegetation index products' browse images (e.g.
MYD13A3), so a composite's preview reads like one a MODIS VI user already
knows: a single index mapped from 0 to 1 onto a tan-to-brown ramp below 0.2
and a light-to-dark green ramp above it.

Fill is transparent: in a composite it marks cloud or missing observations,
not a surface, and should not read as one. JPEG has no alpha channel, so the
images are PNGs.

Every image shares one fixed palette sampled from the ramp, so a given index
value is the same color in every granule, and neighboring tiles viewed
together do not seam.
"""

from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image

from hls_composites.indices import ALL_INDICES, Index

VALUE_RANGE = (0.0, 1.0)
"""Physical index values the color ramp spans; values outside it clamp."""

COLOR_STOPS: tuple[tuple[float, tuple[int, int, int]], ...] = (
    (0.00, (240, 238, 234)),
    (0.03, (225, 218, 212)),
    (0.10, (192, 176, 153)),
    (0.20, (150, 118, 85)),
    (0.20, (190, 222, 125)),
    (0.31, (152, 187, 33)),
    (0.42, (117, 172, 0)),
    (0.62, (57, 135, 0)),
    (0.81, (1, 92, 0)),
    (1.00, (0, 26, 0)),
)
"""Index value and RGB color at each ramp control point, interpolated between.

Sampled from the MODIS VI browse legend. The repeated 0.2 is a hard break
between the bare and vegetated ramps rather than a blend.
"""

IMAGE_SIZE = 1000
"""Width and height of the written PNG, in pixels."""

RAMP_STEPS = 255
"""Palette entries spanning `VALUE_RANGE`, evenly spaced in index value.

HLS pixels are small enough to be entirely vegetated, so much of a tile sits
in a narrow band near the top of the ramp; too few steps would posterize it.
The one remaining entry of an 8-bit palette is reserved for fill.
"""

FILL_ENTRY = RAMP_STEPS
"""Palette entry fill pixels are written as, marked transparent."""

MIN_COVERAGE = 0.5
"""Fraction of an output pixel's source pixels that must carry data.

Below it the output pixel is fill; at or above it, its value averages only
the source pixels that do.
"""


def color_table(values: np.ndarray) -> np.ndarray:
    """Look up the ramp color of each physical index value.

    Parameters
    ----------
    values : numpy.ndarray
        Index values in physical units, of any shape.

    Returns
    -------
    numpy.ndarray
        `uint8` RGB, shaped `(*values.shape, 3)`.
    """
    positions = np.array([position for position, _ in COLOR_STOPS])
    colors = np.array([color for _, color in COLOR_STOPS], dtype=np.float64)

    clamped = np.clip(values, *VALUE_RANGE)
    # side="right" places a value equal to a repeated stop past both copies,
    # so the zero-width segment between them is never interpolated across.
    upper = np.clip(
        np.searchsorted(positions, clamped, side="right"), 1, len(positions) - 1
    )
    lower = upper - 1
    weight = (clamped - positions[lower]) / (positions[upper] - positions[lower])
    rgb = colors[lower] + weight[..., None] * (colors[upper] - colors[lower])
    return np.round(rgb).astype(np.uint8)


def ramp_palette() -> np.ndarray:
    """The palette every browse image is written with.

    Returns
    -------
    numpy.ndarray
        `uint8` RGB, shaped `(RAMP_STEPS + 1, 3)`: the ramp color at the center
        of each step, then black for `FILL_ENTRY`.
    """
    low, high = VALUE_RANGE
    centers = low + (np.arange(RAMP_STEPS) + 0.5) * (high - low) / RAMP_STEPS
    return np.vstack([color_table(centers), np.zeros((1, 3), np.uint8)])


def downsample(
    encoded: np.ndarray, index: Index, size: int = IMAGE_SIZE
) -> tuple[np.ndarray, np.ndarray]:
    """Average an index onto a `size` x `size` grid, ignoring fill.

    Averages index values rather than rendered colors, so an output pixel
    shows the mean index of the area it covers.

    Parameters
    ----------
    encoded : numpy.ndarray
        The index as stored: integers scaled by `index.scale_factor`, with
        `index.fill_value` where there is no data. Shaped `(y, x)`.
    index : Index
        The index the values encode.
    size : int, optional
        Output width and height, by default `IMAGE_SIZE`.

    Returns
    -------
    values : numpy.ndarray
        `float32` mean index in physical units over the valid source pixels,
        shaped `(size, size)`. Meaningless where `coverage` is 0.
    coverage : numpy.ndarray
        `float32` fraction of source pixels carrying data, 0 to 1.
    """
    valid = encoded != index.fill_value
    physical = np.where(valid, encoded * np.float32(index.scale_factor), 0)

    def box(array: np.ndarray) -> np.ndarray:
        image = Image.fromarray(array.astype(np.float32), mode="F")
        return np.asarray(image.resize((size, size), Image.Resampling.BOX))

    coverage = box(valid)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = box(physical) / coverage
    return values, coverage


def palette_entries(values: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Palette entry of each pixel: its ramp step, or `FILL_ENTRY`.

    Parameters
    ----------
    values : numpy.ndarray
        Index values in physical units (see `downsample`).
    coverage : numpy.ndarray
        Fraction of each pixel carrying data (see `downsample`).

    Returns
    -------
    numpy.ndarray
        `uint8` entries into `ramp_palette()`, of the same shape.
    """
    low, high = VALUE_RANGE
    clamped = np.clip(np.nan_to_num(values, nan=low), low, high)
    steps = np.minimum(
        ((clamped - low) / (high - low) * RAMP_STEPS).astype(np.uint8),
        RAMP_STEPS - 1,
    )
    steps[coverage < MIN_COVERAGE] = FILL_ENTRY
    return steps


def browse_path(granule_dir: Path, index: Index) -> Path:
    """Where the browse image for `index` is written in a granule directory."""
    return granule_dir / f"{granule_dir.name}.{index.name}.png"


def write_browse_image(encoded: np.ndarray, index: Index, path: Path) -> Path:
    """Render and save the pseudocolor preview of one index.

    Parameters
    ----------
    encoded : numpy.ndarray
        The index as stored, shaped `(y, x)` (see `downsample`).
    index : Index
        The index the values encode.
    path : pathlib.Path
        Destination PNG.

    Returns
    -------
    pathlib.Path
        `path`, as written.
    """
    image = Image.fromarray(palette_entries(*downsample(encoded, index)), mode="P")
    image.putpalette(ramp_palette().tobytes())
    image.save(path, format="PNG", optimize=True, transparency=FILL_ENTRY)
    return path


def write_browse_images(computed: xr.Dataset, granule_dir: Path) -> list[Path]:
    """Write a browse image for every spectral index in a computed composite.

    Parameters
    ----------
    computed : xarray.Dataset
        Computed composite. Variables that are not indices are ignored, so a
        composite of raw bands yields no browse images.
    granule_dir : pathlib.Path
        Granule directory the PNGs are written into (see `browse_path`).

    Returns
    -------
    list of pathlib.Path
        The written images, in `ALL_INDICES` order.
    """
    return [
        write_browse_image(
            computed[index.name].to_numpy(), index, browse_path(granule_dir, index)
        )
        for index in ALL_INDICES
        if index.name in computed
    ]
