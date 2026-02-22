"""Public façade for the metadata pipeline — re-exports every symbol.

Consumers should ``import metadata_handler`` (or ``import noai_watermark``)
rather than reaching into the internal modules directly.  This file
gathers all public names so that the API surface stays stable even as
the implementation is reorganised.

Internal modules:

- ``constants``  — configuration values and detection lists
- ``utils``      — format helpers
- ``c2pa``       — C2PA JUMBF chunk detection/extraction/injection
- ``extractor``  — read metadata from images
- ``injector``   — write metadata into images
- ``cleaner``    — AI metadata removal
- ``cloner``     — high-level extract → inject pipeline
"""

from cleaner import has_ai_content, remove_ai_metadata
from cloner import clone_ai_metadata, clone_metadata
from constants import (
    AI_KEYWORDS,
    AI_METADATA_KEYS,
    C2PA_ACTIONS,
    C2PA_AI_TOOLS,
    C2PA_CHUNK_TYPE,
    C2PA_ISSUERS,
    C2PA_SIGNATURES,
    PNG_METADATA_KEYS,
    PNG_SIGNATURE,
    SUPPORTED_FORMATS,
)
from c2pa import (
    extract_c2pa_chunk,
    extract_c2pa_info,
    has_c2pa_metadata,
    inject_c2pa_chunk,
)
from extractor import (
    extract_ai_metadata,
    extract_metadata,
    get_ai_metadata_summary,
    has_ai_metadata,
)
from injector import inject_metadata
from utils import get_image_format, is_supported_format

__all__ = [
    # Constants
    "SUPPORTED_FORMATS",
    "AI_METADATA_KEYS",
    "PNG_METADATA_KEYS",
    "AI_KEYWORDS",
    "C2PA_CHUNK_TYPE",
    "C2PA_SIGNATURES",
    "C2PA_ISSUERS",
    "C2PA_AI_TOOLS",
    "C2PA_ACTIONS",
    "PNG_SIGNATURE",
    # Utils
    "is_supported_format",
    "get_image_format",
    # C2PA
    "has_c2pa_metadata",
    "extract_c2pa_info",
    "extract_c2pa_chunk",
    "inject_c2pa_chunk",
    # Extractor
    "extract_metadata",
    "extract_ai_metadata",
    "has_ai_metadata",
    "get_ai_metadata_summary",
    # Injector
    "inject_metadata",
    # Cleaner
    "remove_ai_metadata",
    "has_ai_content",
    # Cloner
    "clone_metadata",
    "clone_ai_metadata",
]
