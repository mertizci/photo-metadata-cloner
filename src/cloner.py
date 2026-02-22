"""High-level metadata cloning between images.

Orchestrates the extract → inject pipeline by combining the
``extractor`` and ``injector`` modules into convenient one-call
functions.
"""

from __future__ import annotations

from pathlib import Path

from cleaner import has_ai_content, remove_ai_metadata
from extractor import extract_ai_metadata, extract_metadata
from injector import inject_metadata


def clone_metadata(
    source_path: Path,
    target_path: Path,
    output_path: Path | None = None,
    ai_only: bool = False,
) -> Path:
    """
    Clone metadata from source image to target image.

    Supports PNG and JPG formats.

    Args:
        source_path: Path to the source image file (metadata donor).
        target_path: Path to the target image file (image donor).
        output_path: Optional output path. If not provided, modifies target in place.
        ai_only: If True, only clone AI-generated metadata.

    Returns:
        Path to the output file with cloned metadata.
    """
    if ai_only:
        metadata = extract_ai_metadata(source_path)
    else:
        metadata = extract_metadata(source_path)

    if output_path is None:
        output_path = target_path

    inject_metadata(target_path, output_path, metadata)

    return output_path


def clone_ai_metadata(
    source_path: Path,
    target_path: Path,
    output_path: Path | None = None,
) -> Path:
    """
    Clone only AI-generated metadata from source image to target image.

    Supports PNG and JPG formats.

    Args:
        source_path: Path to the source image file (AI metadata donor).
        target_path: Path to the target image file (image donor).
        output_path: Optional output path. If not provided, modifies target in place.

    Returns:
        Path to the output file with cloned AI metadata.
    """
    return clone_metadata(source_path, target_path, output_path, ai_only=True)


__all__ = [
    "clone_metadata",
    "clone_ai_metadata",
    "remove_ai_metadata",
    "has_ai_content",
]
