"""The facts a composite's metadata documents, read from what was written.

Both serializers build from this one model, so the ECHO-10 document and the
STAC item cannot disagree about what the granule contains.

The grid, extent, and coverage are read back from the written GeoTIFFs rather
than taken from the in-memory Dataset: `write_composite` computes and discards
it, and reading the files describes what was actually produced.
"""

import datetime as dt
import mimetypes
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import rasterio

from hls_composites.composite import spatial_coverage
from hls_composites.crs import crs_name
from hls_composites.indices import NDVI
from hls_composites.io import ADD_OFFSET
from hls_composites.metadata.footprint import Ring, footprint, footprint_bbox
from hls_composites.models import DateRange, Granule, composite_id
from hls_composites.outputs import VALID_COUNT, fill_attributes

PLACEHOLDER = "PLACEHOLDER"
"""Stands in for a value the DAAC has not assigned yet.

Deliberately not a plausible-looking value: a fabricated DOI or product URI
that reads as real could be published and believed.
"""

# Derived from the granule ID, which already encodes M30 and v2.0, and
# follows the HLSL30/HLSS30 naming of the daily products.
VERSION_ID = "2.0"

DATASET_ID = "HLS Merged Vegetation Indices Monthly Global 30m v2.0"
DOI = "10.5067/HLS/HLSM30_VI.002"

# Not yet assigned.
PRODUCT_URI_BASE = PLACEHOLDER

# Universal, and matching the daily products.
DOI_AUTHORITY = "https://doi.org"

# Describes what the code actually does. The DAAC may want a specific token
# rather than prose.
COMPOSITING_ALGORITHM = (
    "Per-pixel selection of the observation closest to the median EVI2"
)
DATA_FORMAT = "Cloud Optimized GeoTIFF (COG)"
SPATIAL_RESOLUTION = 30.0
DAY_NIGHT_FLAG = "DAY"

CMR_STAC_BASE = "https://cmr.earthdata.nasa.gov/stac/LPCLOUD/collections"
"""Root of the CMR-STAC catalog the input granules are published in."""

CMR_STAC_COLLECTIONS = {"L30": "HLSL30_2.0", "S30": "HLSS30_2.0"}
"""CMR-STAC collection ID per HLS product, as spelled in the live catalog."""


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
    footprint : list of list of tuple of float
        Outline of where the granule has data, as ``(longitude, latitude)``
        rings: counter-clockwise as GeoJSON orders an exterior ring, and not
        closed. Two rings when it crosses the antimeridian, one otherwise.
    bbox : tuple of float
        ``(west, south, east, north)`` of the footprint in degrees; west
        exceeds east when it crosses the antimeridian.
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
    platforms : list of tuple of str
        `(platform, instrument)` pairs that contributed observations.
    scale_factor : float
        Scale of the index rasters.
    add_offset : int
        Offset of the index rasters. An integer because the collection
        declares the ECHO-10 attribute as INT, which rejects "0.0".
    fill_values : dict of str to int
        Fill value of every written band, keyed by the ECHO-10 attribute
        naming it (see `outputs.fill_attributes`).
    asset_bands : list of AssetBand
        How each written COG describes its own band.
    assets : list of pathlib.Path
        The written GeoTIFFs, sorted by name.
    size_bytes : int
        Total size of those files.
    inputs : list of InputGranule
        The granules composited, in discovery order. Empty when unknown.
    browse_images : list of pathlib.Path
        The rendered browse images, one per spectral index.
    """

    granule_id: str
    tile_id: str
    date_range: DateRange
    produced_at: dt.datetime
    footprint: list[Ring]
    bbox: tuple[float, float, float, float]
    proj_bbox: tuple[float, float, float, float]
    epsg: int
    crs_name: str
    ulx: float
    uly: float
    ncols: int
    nrows: int
    spatial_coverage: float
    platforms: list[tuple[str, str]]
    scale_factor: float
    add_offset: int
    fill_values: dict[str, int]
    assets: list[Path]
    asset_bands: list[AssetBand]
    size_bytes: int
    browse_images: list[Path]
    inputs: list[InputGranule] = field(default_factory=list)


def browse_index(image: Path) -> str:
    """Index a browse image renders, from its ``{granule_id}.{index}.png`` name."""
    return image.stem.rsplit(".", 1)[-1]


def browse_media_type(image: Path) -> str:
    """MIME type of a browse image, from its extension."""
    media_type, _ = mimetypes.guess_type(image.name)
    if media_type is None:
        raise ValueError(f"unknown media type for browse image {image.name}")
    return media_type


def browse_description(image: Path) -> str:
    """Description the DAAC shows for a browse image."""
    return f"{browse_index(image)} browse image"


def _asset_bands(rasters: Mapping[str, Path]) -> list[AssetBand]:
    """Read back how each written COG describes its own band, sorted by file."""
    bands = []
    for name, path in sorted(rasters.items(), key=lambda item: item[1]):
        with rasterio.open(path) as src:
            scale = src.scales[0]
            bands.append(
                AssetBand(
                    name=name,
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
    rasters: Mapping[str, Path],
    browse_images: list[Path],
    platforms: list[tuple[str, str]],
    inputs: list[Granule] | None = None,
    produced_at: dt.datetime | None = None,
) -> GranuleMetadata:
    """Describe a written composite.

    Parameters
    ----------
    tile_id : str
        MGRS tile ID, without the leading "T".
    date_range : DateRange
        Period composited over.
    rasters : mapping of str to pathlib.Path
        Each written GeoTIFF by its variable name, as `io.write_rasters`
        returns them.
    browse_images : list of pathlib.Path
        The rendered browse images, referenced from both documents.
    platforms : list of tuple of str
        `(platform, instrument)` pairs that contributed observations, from
        `discovery.read_platforms`.
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
    ValueError
        If `platforms` is empty, or `rasters` holds no `ValidCount`.
    """
    if not platforms:
        raise ValueError("a composite must name the platforms it drew from")
    if VALID_COUNT.name not in rasters:
        raise ValueError(f"a composite must include its {VALID_COUNT.name} raster")

    assets = sorted(rasters.values())
    with rasterio.open(rasters[VALID_COUNT.name]) as src:
        valid_count = src.read(1)
        epsg = src.crs.to_epsg()
        name = crs_name(src.crs)
        ulx, uly = src.transform.c, src.transform.f
        ncols, nrows = src.width, src.height
        left, bottom, right, top = src.bounds
        crs, transform = src.crs, src.transform
    outline = footprint(valid_count != VALID_COUNT.nodata, transform, crs)

    index = NDVI()
    asset_bands = _asset_bands(rasters)
    return GranuleMetadata(
        granule_id=composite_id(tile_id, date_range),
        tile_id=tile_id,
        date_range=date_range,
        produced_at=produced_at or dt.datetime.now(dt.UTC),
        footprint=outline,
        bbox=footprint_bbox(outline),
        proj_bbox=(left, bottom, right, top),
        epsg=int(epsg) if epsg is not None else 0,
        crs_name=name,
        ulx=ulx,
        uly=uly,
        ncols=ncols,
        nrows=nrows,
        spatial_coverage=spatial_coverage(valid_count),
        platforms=platforms,
        scale_factor=index.scale_factor,
        add_offset=ADD_OFFSET,
        fill_values=fill_attributes({band.name: band.nodata for band in asset_bands}),
        assets=assets,
        asset_bands=asset_bands,
        size_bytes=sum(path.stat().st_size for path in assets),
        browse_images=browse_images,
        inputs=_provenance(inputs or []),
    )
