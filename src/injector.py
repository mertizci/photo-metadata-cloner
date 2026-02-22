"""Write metadata into PNG and JPEG images.

Handles format-specific differences:
- PNG: text chunks via ``PngInfo`` + optional C2PA JUMBF injection.
- JPEG: EXIF data via ``piexif`` with text metadata packed into the
  UserComment field.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import piexif
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from c2pa import inject_c2pa_chunk
from utils import get_image_format


def inject_metadata(target_path: Path, output_path: Path, metadata: dict[str, Any]) -> None:
    """
    Inject metadata into a PNG or JPG file and save to output path.

    Args:
        target_path: Path to the target image file.
        output_path: Path where the output file will be saved.
        metadata: Dictionary containing metadata to inject.
    """
    with Image.open(target_path) as img:
        img = img.copy()

        output_format = get_image_format(output_path)
        save_kwargs: dict[str, Any] = {"format": output_format}

        # Handle EXIF data
        if "exif" in metadata:
            try:
                exif_bytes = piexif.dump(metadata["exif"])
                save_kwargs["exif"] = exif_bytes
            except Exception:
                if "exif_raw" in metadata:
                    save_kwargs["exif"] = metadata["exif_raw"]

        if output_format == "PNG":
            save_kwargs = _prepare_png_save_kwargs(save_kwargs, metadata)
        elif output_format == "JPEG":
            save_kwargs = _prepare_jpeg_save_kwargs(save_kwargs, metadata)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "JPEG" and img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        img.save(output_path, **save_kwargs)

    # Handle C2PA chunk injection for PNG files
    if output_format == "PNG" and "c2pa_chunk" in metadata:
        _inject_c2pa_if_present(output_path, metadata)


def _prepare_png_save_kwargs(
    save_kwargs: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Prepare save kwargs for PNG format."""
    pnginfo = {}
    exclude_keys = ["exif", "exif_raw", "dpi", "gamma", "c2pa", "c2pa_chunk"]

    for key, value in metadata.items():
        if key not in exclude_keys:
            pnginfo[key] = value

    if pnginfo:
        pnginfo_obj = PngInfo()
        for key, value in pnginfo.items():
            if isinstance(value, str):
                pnginfo_obj.add_text(key, value)
            elif isinstance(value, bytes):
                pnginfo_obj.add_text(key, value.decode("utf-8", errors="replace"))
        save_kwargs["pnginfo"] = pnginfo_obj

    if "dpi" in metadata:
        save_kwargs["dpi"] = metadata["dpi"]

    return save_kwargs


def _prepare_jpeg_save_kwargs(
    save_kwargs: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    """Prepare save kwargs for JPEG format."""
    exif_dict = metadata.get("exif", {"0th": {}, "Exif": {}, "1st": {}, "GPS": {}, "Interop": {}})

    text_metadata = []
    exclude_keys = ["exif", "exif_raw", "dpi", "gamma", "c2pa", "c2pa_chunk"]

    for key, value in metadata.items():
        if key not in exclude_keys:
            if isinstance(value, str):
                text_metadata.append(f"{key}={value}")
            elif isinstance(value, bytes):
                text_metadata.append(f"{key}={value.decode('utf-8', errors='replace')}")

    if text_metadata:
        user_comment = "ASCII\x00\x00\x00" + "\n".join(text_metadata)
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = user_comment.encode("utf-8")

    try:
        exif_bytes = piexif.dump(exif_dict)
        save_kwargs["exif"] = exif_bytes
    except Exception:
        pass

    if "dpi" in metadata:
        save_kwargs["dpi"] = metadata["dpi"]

    return save_kwargs


def _inject_c2pa_if_present(output_path: Path, metadata: dict[str, Any]) -> None:
    """Inject C2PA chunk if present in metadata."""
    c2pa_chunk = metadata.get("c2pa_chunk")
    if not isinstance(c2pa_chunk, bytes):
        return

    temp_output = output_path.with_suffix(".tmp.png")
    try:
        inject_c2pa_chunk(output_path, temp_output, c2pa_chunk)
        temp_output.replace(output_path)
    except Exception:
        if temp_output.exists():
            temp_output.unlink()
