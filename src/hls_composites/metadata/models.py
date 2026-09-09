"""The facts a composite's metadata documents, read from what was written.

Both serializers build from this one model, so the ECHO-10 document and the
STAC item cannot disagree about what the granule contains.

The grid, extent, and coverage are read back from the written GeoTIFFs rather
than taken from the in-memory Dataset: `write_composite` computes and discards
it, and reading the files describes what was actually produced.
"""

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

from hls_composites.composite import VALID_COUNT_FILL, spatial_coverage
from hls_composites.crs import crs_name
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


"""CMR-STAC collection ID per HLS product, as spelled in the live catalog."""


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
class AssetBand:
    """One written COG's band, as read back from the file.

    Parameters
    ----------
    name : str
        Variable name, e.g. ``NDVI``, which is also the asset key.
    description : str
        The band's long name.
    data_type : str
        Storage type, spelled as STAC spells it, e.g. ``int16``.
    nodata : float or None
        Fill value, or None for a band that declares none.
    scale : float or None
        Factor converting stored values to physical units, or None for a
        band stored in its own units.
    """

    name: str
    description: str
    data_type: str
    nodata: float | None
    scale: float | None


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
    proj_bbox : tuple of float
        ``(west, south, east, north)`` in the rasters' own projected CRS.
    epsg : int
        Projected CRS code of the written rasters.
    crs_name : str
        Human-readable name of that CRS.
    ulx, uly : float
        Upper-left corner in projected coordinates.
    ncols, nrows : int
        Raster width and height in pixels.
    spatial_coverage : float
        Percentage of pixels carrying data, 0 to 100.
    scale_factor, add_offset : float
        Encoding of the index rasters.
    fill_value, qa_fill_value : int
        Fill values of the index rasters and of ``ValidCount``.
    asset_bands : list of AssetBand
        How each written COG describes its own band.
    assets : list of pathlib.Path
        The written GeoTIFFs, sorted by name.
    size_bytes : int
        Total size of those files.
    inputs : list of InputGranule
        The granules composited, in discovery order. Empty when unknown.
    browse_image : pathlib.Path
        The rendered browse image. Every granule has one.
    """

    granule_id: str
    tile_id: str
    date_range: DateRange
    produced_at: dt.datetime
    boundary: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    proj_bbox: tuple[float, float, float, float]
    epsg: int
    crs_name: str
    ulx: float
    uly: float
    ncols: int
    nrows: int
    spatial_coverage: float
    scale_factor: float
    add_offset: float
    fill_value: int
    qa_fill_value: int
    assets: list[Path]
    asset_bands: list[AssetBand]
    size_bytes: int
    browse_image: Path
    inputs: list[InputGranule] = field(default_factory=list)


def _spatial_coverage(valid_count_path: Path) -> float:
    """Read back the written `ValidCount` and measure what it covers."""
    with rasterio.open(valid_count_path) as src:
        return spatial_coverage(src.read(1))


def _asset_bands(assets: list[Path]) -> list[AssetBand]:
    """Read back how each written COG describes its own band."""
    bands = []
    for path in assets:
        with rasterio.open(path) as src:
            scale = src.scales[0]
            bands.append(
                AssetBand(
                    name=path.stem.rsplit(".", 1)[-1],
                    description=src.descriptions[0] or "",
                    data_type=src.dtypes[0],
                    nodata=src.nodata,
                    scale=scale if scale != 1.0 else None,
                )
            )
    return bands


def granule_metadata(
    tile_id: str,
    date_range: DateRange,
    granule_dir: Path,
    browse_image: Path,
    inputs: list[Granule] | None = None,
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
    browse_image : pathlib.Path
        The rendered browse image, referenced from both documents.
    inputs : list of Granule, optional
        The granules composited. Recorded as provenance; omitted from both
        documents when not given.
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
        name = crs_name(src.crs)
        ulx, uly = src.transform.c, src.transform.f
        ncols, nrows = src.width, src.height
        left, bottom, right, top = src.bounds
        west, south, east, north = transform_bounds(
            src.crs, "EPSG:4326", *src.bounds, densify_pts=_DENSIFY_POINTS
        )

    valid_count = granule_dir / f"{granule_dir.name}.ValidCount.tif"
    coverage = _spatial_coverage(valid_count) if valid_count.exists() else 0.0

    index = NDVI()
    return GranuleMetadata(
        granule_id=composite_id(tile_id, date_range),
        tile_id=tile_id,
        date_range=date_range,
        produced_at=produced_at or dt.datetime.now(dt.UTC),
        boundary=[(west, north), (west, south), (east, south), (east, north)],
        bbox=(west, south, east, north),
        proj_bbox=(left, bottom, right, top),
        epsg=int(epsg) if epsg is not None else 0,
        crs_name=name,
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
        asset_bands=_asset_bands(assets),
        size_bytes=sum(path.stat().st_size for path in assets),
        browse_image=browse_image,
        inputs=_provenance(inputs or []),
    )
