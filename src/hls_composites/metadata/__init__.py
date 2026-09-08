"""Granule metadata: one model, serialized to ECHO-10 XML and to STAC."""

from hls_composites.metadata.echo10 import to_echo10
from hls_composites.metadata.models import GranuleMetadata, granule_metadata
from hls_composites.metadata.stac import to_stac_item

__all__ = ["GranuleMetadata", "granule_metadata", "to_echo10", "to_stac_item"]
