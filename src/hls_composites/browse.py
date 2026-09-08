"""RGB browse image for a composite.

Follows `hls-thumbnails`, which renders the daily HLS products' browse images,
so a composite's preview is comparable with a daily granule's: clamp to a
digital-number range, log stretch onto 0-255, stack, resize, save as JPEG.
"""

import math
from pathlib import Path

import numpy as np
import xarray as xr
from PIL import Image

from hls_composites.composite import BROWSE_BANDS

LOW_THRESHOLD = 100
"""Digital number at or below which a pixel renders black."""

HIGH_THRESHOLD = 7500
"""Digital number at or above which a pixel saturates."""

IMAGE_SIZE = 1000
"""Width and height of the written JPEG, in pixels."""

_MAX = 255.0


def stretch(bands: np.ndarray) -> np.ndarray:
    """Log-stretch digital numbers onto 0-255.

    Values at or below `LOW_THRESHOLD` are floored to `e` before the logarithm,
    which both keeps it defined and maps them to 0 -- so fill (-9999) renders
    black, as it does in the daily products' browse images. Values at or above
    `HIGH_THRESHOLD` saturate.

    Parameters
    ----------
    bands : numpy.ndarray
        Digital numbers, shaped `(band, y, x)`.

    Returns
    -------
    numpy.ndarray
        `uint8` array of the same shape.
    """
    low = math.log(LOW_THRESHOLD)
    high = math.log(HIGH_THRESHOLD)

    values = bands.astype(np.float64)
    values[values <= LOW_THRESHOLD] = math.e
    with np.errstate(all="ignore"):
        values = np.log(values)

    scaled = _MAX * (values - low) / (high - low)
    return np.clip(scaled, 0.0, _MAX).astype(np.uint8)


def write_browse_image(computed: xr.Dataset, path: Path) -> Path:
    """Render and save the true-colour preview of a computed composite.

    Parameters
    ----------
    computed : xarray.Dataset
        Computed composite carrying the `BROWSE_BANDS` variables.
    path : pathlib.Path
        Destination JPEG.

    Returns
    -------
    pathlib.Path
        `path`, as written.

    Raises
    ------
    KeyError
        If the Dataset is missing any browse band.
    """
    bands = np.stack([computed[name].to_numpy() for name in BROWSE_BANDS])
    image = Image.fromarray(np.moveaxis(stretch(bands), 0, 2), mode="RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image.save(path)
    return path
