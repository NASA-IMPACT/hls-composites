"""The facts a composite's metadata documents, read from what was written.

Both serializers build from this one model, so the ECHO-10 document and the
STAC item cannot disagree about what the granule contains.

The grid, extent, and coverage are read back from the written GeoTIFFs rather
than taken from the in-memory Dataset: `write_composite` computes and discards
it, and reading the files describes what was actually produced.
"""

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform_bounds

from hls_composites.composite import VALID_COUNT_FILL
from hls_composites.indices import NDVI
from hls_composites.io import composite_id
from hls_composites.models import DateRange, Granule

PLACEHOLDER = "PLACEHOLDER"
"""Stands in for a value the DAAC has not assigned yet.

Deliberately not a plausible-looking value: a fabricated DOI or product URI
that reads as real could be published and believed. `test_placeholders`
records which constants still carry it.
"""

# Derived from the granule ID, which already encodes M30 and v2.0, and
# follows the HLSL30/HLSS30 naming of the daily products.
SHORT_NAME = "HLSM30"
VERSION_ID = "2.0"

# Not yet assigned.
DATASET_ID = PLACEHOLDER
DOI = PLACEHOLDER
PRODUCT_URI_BASE = PLACEHOLDER

# Universal, and matching the daily products.
DOI_AUTHORITY = "https://doi.org"

# Describes what the code actually does. The DAAC may want a specific token
# rather than prose.
COMPOSITING_ALGORITHM = (
    "Per-pixel selection of the observation closest to the median EVI2"
)
DATA_FORMAT = "Cloud Optimized GeoTIFF (COG)"
BROWSE_DESCRIPTION = "Browse image"
"""Description the DAAC shows for the browse image."""
SPATIAL_RESOLUTION = 30.0
DAY_NIGHT_FLAG = "DAY"

PLATFORMS: list[tuple[str, str]] = [
    ("LANDSAT-8", "OLI"),
    ("LANDSAT-9", "OLI"),
    ("Sentinel-2A", "Sentinel-2 MSI"),
    ("Sentinel-2B", "Sentinel-2 MSI"),
]
"""(platform, instrument) pairs a composite may draw observations from.

A composite mixes L30 and S30 sources, so unlike a daily granule it cannot
name a single platform.
"""

CMR_STAC_BASE = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/collections"
"""Root of the CMR-STAC catalog the input granules are published in."""

CMR_STAC_COLLECTIONS = {"L30": "HLSL30_2.0", "S30": "HLSS30_2.0"}
"""CMR-STAC collection ID per HLS product, as spelled in the live catalog."""

# Zone 1-60, latitude band excluding I and O, two-letter grid square.
_MGRS_TILE = re.compile(r"^([0-9]{1,2})([C-HJ-NP-X])([A-Z]{2})$")

# Densifying the edges before reprojecting keeps the lat/lon bounds tight:
# a UTM rectangle's edges curve on the ellipsoid.
_DENSIFY_POINTS = 21


@dataclass(frozen=True)
class InputGranule:
    """One HLS granule a composite was built from.

    Parameters
    ----------
    granule_id : str
        The granule ID, which is also its STAC item ID.
    stac_href : str
        URL of that item in the CMR-STAC catalog.
    """

    granule_id: str
    stac_href: str


def mgrs_fields(tile_id: str) -> tuple[int, str, str]:
    """Split an MGRS tile ID into its UTM zone, latitude band, and grid square.

    Parameters
    ----------
    tile_id : str
        Tile ID without the leading "T", e.g. ``14TPN``.

    Returns
    -------
    tuple
        ``(utm_zone, latitude_band, grid_square)``, e.g. ``(14, "T", "PN")``.

    Raises
    ------
    ValueError
        If `tile_id` is not a well-formed MGRS tile.
    """
    match = _MGRS_TILE.match(tile_id)
    if match is None:
        raise ValueError(f"not an MGRS tile: {tile_id!r}")
    zone, band, square = match.groups()
    return int(zone), band, square


def _provenance(granules: list[Granule]) -> list[InputGranule]:
    """Where each input granule's STAC item lives."""
    inputs = []
    for granule in granules:
        granule_id = granule.path.rsplit("/", 1)[-1]
        collection = CMR_STAC_COLLECTIONS[granule.satellite]
        inputs.append(
            InputGranule(
                granule_id=granule_id,
                stac_href=f"{CMR_STAC_BASE}/{collection}/items/{granule_id}",
            )
        )
    return inputs


@dataclass(frozen=True)
class GranuleMetadata:
    """Everything the ECHO-10 and STAC serializers need.

    Parameters
    ----------
    granule_id : str
        Granule identifier, e.g. ``HLS.M30.T14TPN.2020032.2020060.v2.0``.
    tile_id : str
        MGRS tile ID, without the leading "T".
    date_range : DateRange
        Period composited over.
    produced_at : datetime.datetime
        When the composite was produced, in UTC.
    boundary : list of tuple of float
        Granule outline as ``(longitude, latitude)`` corners.
    bbox : tuple of float
        ``(west, south, east, north)`` in degrees.
    epsg : int
        Projected CRS code of the written rasters.
    crs_name : str
        Human-readable name of that CRS.
    ulx, uly : float
        Upper-left corner in projected coordinates.
    ncols, nrows : int
        Raster width and height in pixels.
    spatial_coverage : int
        Percentage of pixels carrying data, 0 to 100.
    scale_factor, add_offset : float
        Encoding of the index rasters.
    fill_value, qa_fill_value : int
        Fill values of the index rasters and of ``ValidCount``.
    assets : list of pathlib.Path
        The written GeoTIFFs, sorted by name.
    size_bytes : int
        Total size of those files.
    inputs : list of InputGranule
        The granules composited, in discovery order. Empty when unknown.
    browse_image : pathlib.Path or None
        The rendered browse image, or None when none was produced.
    """

    granule_id: str
    tile_id: str
    date_range: DateRange
    produced_at: dt.datetime
    boundary: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    epsg: int
    crs_name: str
    ulx: float
    uly: float
    ncols: int
    nrows: int
    spatial_coverage: int
    scale_factor: float
    add_offset: float
    fill_value: int
    qa_fill_value: int
    assets: list[Path]
    size_bytes: int
    inputs: list[InputGranule] = field(default_factory=list)
    browse_image: Path | None = None


def _crs_name(crs: rasterio.crs.CRS) -> str:
    """The CRS's declared name, e.g. ``WGS 84 / UTM zone 14N``.

    It is the first quoted string in the WKT, so no pyproj lookup is needed.
    """
    parts = crs.to_wkt().split('"')
    return parts[1] if len(parts) > 1 else str(crs)


def _spatial_coverage(valid_count_path: Path) -> int:
    """Percentage of pixels with at least one contributing observation."""
    with rasterio.open(valid_count_path) as src:
        data = src.read(1)
    covered = int(np.count_nonzero(data != VALID_COUNT_FILL))
    return round(100 * covered / data.size)


def granule_metadata(
    tile_id: str,
    date_range: DateRange,
    granule_dir: Path,
    inputs: list[Granule] | None = None,
    browse_image: Path | None = None,
    produced_at: dt.datetime | None = None,
) -> GranuleMetadata:
    """Describe a written composite directory.

    Parameters
    ----------
    tile_id : str
        MGRS tile ID, without the leading "T".
    date_range : DateRange
        Period composited over.
    granule_dir : pathlib.Path
        Directory holding the written GeoTIFFs.
    inputs : list of Granule, optional
        The granules composited. Recorded as provenance; omitted from both
        documents when not given.
    browse_image : pathlib.Path, optional
        The rendered browse image. Referenced from both documents when given.
    produced_at : datetime.datetime, optional
        Production time, by default the current UTC time.

    Returns
    -------
    GranuleMetadata
        The facts both serializers render.

    Raises
    ------
    FileNotFoundError
        If the directory holds no GeoTIFFs.
    """
    assets = sorted(granule_dir.glob("*.tif"))
    if not assets:
        raise FileNotFoundError(f"no GeoTIFFs in {granule_dir}")

    with rasterio.open(assets[0]) as src:
        epsg = src.crs.to_epsg()
        crs_name = _crs_name(src.crs)
        ulx, uly = src.transform.c, src.transform.f
        ncols, nrows = src.width, src.height
        west, south, east, north = transform_bounds(
            src.crs, "EPSG:4326", *src.bounds, densify_pts=_DENSIFY_POINTS
        )

    valid_count = granule_dir / f"{granule_dir.name}.ValidCount.tif"
    coverage = _spatial_coverage(valid_count) if valid_count.exists() else 0

    index = NDVI()
    return GranuleMetadata(
        granule_id=composite_id(tile_id, date_range),
        tile_id=tile_id,
        date_range=date_range,
        produced_at=produced_at or dt.datetime.now(dt.UTC),
        boundary=[(west, north), (west, south), (east, south), (east, north)],
        bbox=(west, south, east, north),
        epsg=int(epsg) if epsg is not None else 0,
        crs_name=crs_name,
        ulx=ulx,
        uly=uly,
        ncols=ncols,
        nrows=nrows,
        spatial_coverage=coverage,
        scale_factor=index.scale_factor,
        add_offset=0.0,
        fill_value=index.fill_value,
        qa_fill_value=VALID_COUNT_FILL,
        assets=assets,
        size_bytes=sum(path.stat().st_size for path in assets),
        inputs=_provenance(inputs or []),
        browse_image=browse_image,
    )
