"""ECHO-10 granule XML, the form CMR ingests.

Element order follows the daily HLS products' output, which the schema
requires: ECHO-10 uses sequences, so a reordered document is invalid.
"""

import datetime as dt
from xml.etree import ElementTree

from hls_composites.metadata.models import (
    COMPOSITING_ALGORITHM,
    DATA_FORMAT,
    DATASET_ID,
    DAY_NIGHT_FLAG,
    DOI,
    DOI_AUTHORITY,
    PLATFORMS,
    PRODUCT_URI_BASE,
    SPATIAL_RESOLUTION,
    VERSION_ID,
    GranuleMetadata,
)

_TIMESTAMP = "%Y-%m-%dT%H:%M:%S.%fZ"


def _timestamp(moment: dt.datetime) -> str:
    return moment.strftime(_TIMESTAMP)


def _sub(
    parent: ElementTree.Element, tag: str, text: str | None = None
) -> ElementTree.Element:
    element = ElementTree.SubElement(parent, tag)
    if text is not None:
        element.text = text
    return element


def _additional_attributes(meta: GranuleMetadata) -> list[tuple[str, str]]:
    """Every AdditionalAttribute, in the order they are written."""
    return [
        ("PRODUCT_URI", f"{PRODUCT_URI_BASE}/{meta.granule_id}"),
        ("MGRS_TILE_ID", meta.tile_id),
        ("SPATIAL_COVERAGE", str(meta.spatial_coverage)),
        ("SPATIAL_RESOLUTION", str(SPATIAL_RESOLUTION)),
        ("PROCESSING_TIME", _timestamp(meta.produced_at)),
        ("HORIZONTAL_CS_CODE", f"EPSG:{meta.epsg}"),
        ("HORIZONTAL_CS_NAME", meta.crs_name),
        ("ULX", str(meta.ulx)),
        ("ULY", str(meta.uly)),
        ("REF_SCALE_FACTOR", str(meta.scale_factor)),
        ("ADD_OFFSET", str(meta.add_offset)),
        ("FILLVALUE", str(meta.fill_value)),
        ("QA_FILL_VALUE", str(meta.qa_fill_value)),
        ("NCOLS", str(meta.ncols)),
        ("NROWS", str(meta.nrows)),
        ("COMPOSITING_ALGORITHM", COMPOSITING_ALGORITHM),
        ("COMPOSITING_START_DATE", meta.date_range.start.isoformat()),
        ("COMPOSITING_END_DATE", meta.date_range.end.isoformat()),
        ("IDENTIFIER_PRODUCT_DOI", DOI),
        ("IDENTIFIER_PRODUCT_DOI_AUTHORITY", DOI_AUTHORITY),
    ]


def to_echo10(meta: GranuleMetadata) -> str:
    """Render `meta` as an ECHO-10 granule document.

    Parameters
    ----------
    meta : GranuleMetadata
        The granule to describe.

    Returns
    -------
    str
        A complete XML document, including the declaration.
    """
    granule = ElementTree.Element("Granule")
    _sub(granule, "GranuleUR", meta.granule_id)
    _sub(granule, "InsertTime", _timestamp(meta.produced_at))
    _sub(granule, "LastUpdate", _timestamp(meta.produced_at))

    collection = _sub(granule, "Collection")
    _sub(collection, "DataSetId", DATASET_ID)

    data_granule = _sub(granule, "DataGranule")
    _sub(data_granule, "DataGranuleSizeInBytes", str(meta.size_bytes))
    _sub(data_granule, "ProducerGranuleId", meta.granule_id)
    _sub(data_granule, "DayNightFlag", DAY_NIGHT_FLAG)
    _sub(data_granule, "ProductionDateTime", _timestamp(meta.produced_at))
    _sub(data_granule, "LocalVersionId", VERSION_ID)

    temporal = _sub(granule, "Temporal")
    range_date_time = _sub(temporal, "RangeDateTime")
    start = dt.datetime.combine(meta.date_range.start, dt.time.min, tzinfo=dt.UTC)
    end = dt.datetime.combine(meta.date_range.end, dt.time.max, tzinfo=dt.UTC)
    _sub(range_date_time, "BeginningDateTime", _timestamp(start))
    _sub(range_date_time, "EndingDateTime", _timestamp(end))

    spatial = _sub(granule, "Spatial")
    domain = _sub(spatial, "HorizontalSpatialDomain")
    geometry = _sub(domain, "Geometry")
    polygon = _sub(geometry, "GPolygon")
    boundary = _sub(polygon, "Boundary")
    for longitude, latitude in meta.boundary:
        point = _sub(boundary, "Point")
        _sub(point, "PointLongitude", f"{longitude:.8f}")
        _sub(point, "PointLatitude", f"{latitude:.8f}")

    platforms = _sub(granule, "Platforms")
    for platform_name, instrument_name in PLATFORMS:
        platform = _sub(platforms, "Platform")
        _sub(platform, "ShortName", platform_name)
        instruments = _sub(platform, "Instruments")
        instrument = _sub(instruments, "Instrument")
        _sub(instrument, "ShortName", instrument_name)

    attributes = _sub(granule, "AdditionalAttributes")
    for name, value in _additional_attributes(meta):
        attribute = _sub(attributes, "AdditionalAttribute")
        _sub(attribute, "Name", name)
        values = _sub(attribute, "Values")
        _sub(values, "Value", value)

    _sub(granule, "OnlineAccessURLs")
    _sub(granule, "OnlineResources")
    _sub(granule, "DataFormat", DATA_FORMAT)
    _sub(granule, "AssociatedBrowseImageUrls")

    ElementTree.indent(granule, space="  ")
    body = ElementTree.tostring(granule, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{body}'
