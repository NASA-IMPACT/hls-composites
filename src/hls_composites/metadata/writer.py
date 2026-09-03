"""Write both metadata documents beside a composite's GeoTIFFs."""

import json
from pathlib import Path

from hls_composites.metadata.echo10 import to_echo10
from hls_composites.metadata.models import granule_metadata
from hls_composites.metadata.stac import to_stac_item
from hls_composites.models import DateRange, Granule

CMR_SUFFIX = ".cmr.xml"
STAC_SUFFIX = "_stac.json"


def write_metadata(
    tile_id: str,
    date_range: DateRange,
    granule_dir: Path,
    inputs: list[Granule] | None = None,
) -> list[Path]:
    """Describe the composite in `granule_dir` and write both documents there.

    Both are built from one `GranuleMetadata`, so they cannot disagree.

    Parameters
    ----------
    tile_id : str
        MGRS tile ID, without the leading "T".
    date_range : DateRange
        Period composited over.
    granule_dir : pathlib.Path
        Directory holding the written GeoTIFFs; the documents are written
        alongside them.
    inputs : list of Granule, optional
        The granules composited, recorded as provenance in both documents.

    Returns
    -------
    list of pathlib.Path
        The ECHO-10 document and the STAC item, in that order.
    """
    meta = granule_metadata(tile_id, date_range, granule_dir, inputs=inputs)

    xml_path = granule_dir / f"{meta.granule_id}{CMR_SUFFIX}"
    xml_path.write_text(to_echo10(meta))

    json_path = granule_dir / f"{meta.granule_id}{STAC_SUFFIX}"
    json_path.write_text(json.dumps(to_stac_item(meta), indent=2))

    return [xml_path, json_path]
