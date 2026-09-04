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

from hls_composites.metadata.models import (
    BROWSE_DESCRIPTION,
    DOI,
    PLACEHOLDER,
    GranuleMetadata,
    mgrs_fields,
)

PROJECTION_SCHEMA_URI = (
    "https://stac-extensions.github.io/projection/v1.2.0/schema.json"
)
"""The one projection extension version declaring both proj:epsg and proj:code."""

SCIENTIFIC_SCHEMA_URI = (
    "https://stac-extensions.github.io/scientific/v1.0.0/schema.json"
)


def _asset_key(path_name: str, granule_id: str) -> str:
    """Variable name from a file name, e.g. ``...v2.0.NDVI.tif`` -> ``NDVI``."""
    return path_name.removeprefix(f"{granule_id}.").removesuffix(".tif")


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

    # The scientific extension constrains sci:doi to a real DOI pattern, so
    # claiming one we do not have would make the item invalid. Declare the
    # extension only once a DOI is assigned.
    if DOI != PLACEHOLDER:
        item.stac_extensions.append(SCIENTIFIC_SCHEMA_URI)
        item.properties["sci:doi"] = DOI

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

    for path in meta.assets:
        item.add_asset(
            _asset_key(path.name, meta.granule_id),
            pystac.Asset(
                href=path.name,
                media_type=pystac.MediaType.COG,
                roles=["data"],
            ),
        )

    if meta.browse_image is not None:
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
