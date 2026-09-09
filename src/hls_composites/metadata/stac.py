"""STAC item for a composite granule.

The projection extension is applied by setting the schema URI and properties
directly rather than through `ProjectionExtension`, whose defaults follow the
installed pystac version. Writing them here keeps the output stable across
pystac releases and lets the item carry both `proj:epsg` (what consumers of
the daily HLS products read) and `proj:code` (its replacement).
"""

import datetime as dt
from typing import Any

import pystac
import rasterio
from pystac.extensions.mgrs import MgrsExtension
from pystac.utils import datetime_to_str

from hls_composites.crs import mgrs_fields
from hls_composites.metadata.models import (
    BROWSE_DESCRIPTION,
    DOI,
    PLACEHOLDER,
    AssetBand,
    GranuleMetadata,
)

PROJECTION_SCHEMA_URI = (
    "https://stac-extensions.github.io/projection/v1.2.0/schema.json"
)
"""The one projection extension version declaring both proj:epsg and proj:code."""

RASTER_SCHEMA_URI = "https://stac-extensions.github.io/raster/v2.0.0/schema.json"
"""Where `raster:scale` and `raster:offset` live, now that STAC 1.1 owns the rest."""

SCIENTIFIC_SCHEMA_URI = (
    "https://stac-extensions.github.io/scientific/v1.0.0/schema.json"
)


def _asset_key(path_name: str, granule_id: str) -> str:
    """Variable name from a file name, e.g. ``...v2.0.NDVI.tif`` -> ``NDVI``."""
    return path_name.removeprefix(f"{granule_id}.").removesuffix(".tif")


def _band_object(band: AssetBand, valid_percent: float) -> dict[str, Any]:
    """One STAC 1.1 band object: what the value means and how to decode it.

    `nodata`, `data_type` and `statistics` are common metadata in STAC 1.1;
    `raster:scale` stayed behind in the raster extension, which the item
    declares when any band carries one.
    """
    obj: dict[str, Any] = {"name": band.name, "data_type": band.data_type}
    if band.description:
        obj["description"] = band.description
    if band.nodata is not None:
        obj["nodata"] = band.nodata
    if band.scale is not None:
        obj["raster:scale"] = band.scale
        obj["raster:offset"] = 0.0
    obj["statistics"] = {"valid_percent": valid_percent}
    return obj


def to_stac_item(meta: GranuleMetadata) -> dict[str, Any]:
    """Render `meta` as a STAC item.

    Parameters
    ----------
    meta : GranuleMetadata
        The granule to describe.

    Returns
    -------
    dict
        The item as a dictionary, ready to serialize as JSON.
    """
    ring = [*meta.boundary, meta.boundary[0]]
    start = dt.datetime.combine(meta.date_range.start, dt.time.min, tzinfo=dt.UTC)
    end = dt.datetime.combine(meta.date_range.end, dt.time.max, tzinfo=dt.UTC)

    item = pystac.Item(
        id=meta.granule_id,
        geometry={"type": "Polygon", "coordinates": [[list(point) for point in ring]]},
        bbox=list(meta.bbox),
        datetime=None,
        start_datetime=start,
        end_datetime=end,
        properties={},
    )

    with rasterio.open(meta.assets[0]) as src:
        transform = list(src.transform)[:6]

    item.stac_extensions.append(PROJECTION_SCHEMA_URI)
    item.properties["proj:epsg"] = meta.epsg
    item.properties["proj:code"] = f"EPSG:{meta.epsg}"
    item.properties["proj:shape"] = [meta.nrows, meta.ncols]
    item.properties["proj:transform"] = transform
    item.properties["proj:bbox"] = list(meta.proj_bbox)

    item.properties["created"] = datetime_to_str(meta.produced_at)

    # The scientific extension constrains sci:doi to a real DOI pattern, so
    # claiming one we do not have would make the item invalid. Declare the
    # extension only once a DOI is assigned.
    if DOI != PLACEHOLDER:
        item.stac_extensions.append(SCIENTIFIC_SCHEMA_URI)
        item.properties["sci:doi"] = DOI

    # Named as the daily HLS products name their own granule-level coverage.
    item.properties["hls:spatial_coverage"] = meta.spatial_coverage

    zone, band, square = mgrs_fields(meta.tile_id)
    mgrs = MgrsExtension.ext(item, add_if_missing=True)
    mgrs.utm_zone = zone
    mgrs.latitude_band = band
    mgrs.grid_square = square

    # Provenance: which HLS granules this composite was built from. A re-run
    # over a period whose inputs have since changed produces a different list.
    for source in meta.inputs:
        item.add_link(
            pystac.Link(
                rel=pystac.RelType.DERIVED_FROM,
                target=source.stac_href,
                media_type=pystac.MediaType.JSON,
                title=source.granule_id,
            )
        )

    bands = {band.name: band for band in meta.asset_bands}
    if any(band.scale is not None for band in meta.asset_bands):
        item.stac_extensions.append(RASTER_SCHEMA_URI)
    for path in meta.assets:
        key = _asset_key(path.name, meta.granule_id)
        item.add_asset(
            key,
            pystac.Asset(
                href=path.name,
                media_type=pystac.MediaType.COG,
                roles=["data"],
                extra_fields={
                    "bands": [_band_object(bands[key], meta.spatial_coverage)]
                },
            ),
        )

    item.add_asset(
        "thumbnail",
        pystac.Asset(
            href=meta.browse_image.name,
            media_type=pystac.MediaType.JPEG,
            roles=["thumbnail"],
            description=BROWSE_DESCRIPTION,
        ),
    )

    return item.to_dict(include_self_link=False)
