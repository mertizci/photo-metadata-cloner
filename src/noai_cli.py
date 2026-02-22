"""Command-line interface for noai-watermark.

Provides the ``noai-watermark`` entry point that dispatches to one of
two modes:

- *(default)*          — diffusion-based invisible watermark removal
- ``--metadata``       — metadata operations (clone, check, remove)

The heavy progress-animation and library-silencing logic lives in the
separate ``progress`` module to keep this file focused on argument
parsing and command routing.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from __init__ import __version__
from metadata_handler import (
    clone_metadata,
    extract_metadata,
    extract_ai_metadata,
    has_ai_metadata,
    get_ai_metadata_summary,
    is_supported_format,
    remove_ai_metadata,
    SUPPORTED_FORMATS,
)


# ── Branding ────────────────────────────────────────────────────────

_ASCII_ART = """\
 ███╗   ██╗ ██████╗  █████╗ ██╗
 ████╗  ██║██╔═══██╗██╔══██╗██║
 ██╔██╗ ██║██║   ██║███████║██║
 ██║╚██╗██║██║   ██║██╔══██║██║
 ██║ ╚████║╚██████╔╝██║  ██║██║
 ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝"""

_VERSION_TAG = f"─── noai-watermark v.{__version__} ───"
_ASCII_LOGO = f"\n{_ASCII_ART}\n {_VERSION_TAG}\n"


def _print_ascii_logo() -> None:
    """Print the startup banner, colored unless ``NO_COLOR`` is set."""
    if os.environ.get("NO_COLOR"):
        print(_ASCII_LOGO, file=sys.stdout)
        return
    yellow = "\033[93m"
    bold = "\033[1m"
    reset = "\033[0m"
    print(f"{bold}{yellow}{_ASCII_LOGO}{reset}", file=sys.stdout)


# ── Verbose metadata formatting (shared by multiple commands) ───────

def _format_metadata_value(key: str, value: Any) -> str:
    """Return a single-line human-readable representation of a metadata entry."""
    if key == "c2pa_chunk":
        return f"  {key}: <binary data>"
    if key == "c2pa" and isinstance(value, dict):
        return f"  {key}: C2PA metadata present"
    if key == "exif":
        return f"  {key}: <EXIF data present>"
    if isinstance(value, str) and len(value) > 100:
        return f"  {key}: {value[:100]}..."
    return f"  {key}: {value}"


# ── Argument parser construction ────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="noai-watermark",
        description=(
            "Remove invisible AI watermarks from images. "
            "Use --metadata for metadata operations (clone, check, remove)."
        ),
        epilog=(
            "Examples:\n"
            "  noai-watermark source.png -o cleaned.png\n"
            "  noai-watermark source.png --metadata --check-ai\n"
            "  noai-watermark source.png target.png --metadata -o output.png"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "source", type=Path,
        help=f"Source image file (formats: {', '.join(SUPPORTED_FORMATS)})",
    )
    parser.add_argument(
        "target", type=Path, nargs="?",
        help="Target image file (only for metadata cloning with --metadata)",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output file path (default: overwrite source file)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print detailed information during processing",
    )

    # ── Metadata mode ───────────────────────────────────────────
    meta_group = parser.add_argument_group("metadata operations (require --metadata)")
    meta_group.add_argument(
        "--metadata", action="store_true",
        help="Switch to metadata mode (clone, check, or remove metadata)",
    )
    meta_group.add_argument(
        "-a", "--ai-only", action="store_true",
        help="Clone only AI-generated metadata",
    )
    meta_group.add_argument(
        "--check-ai", action="store_true",
        help="Check if source contains AI metadata",
    )
    meta_group.add_argument(
        "--remove-ai", action="store_true",
        help="Remove all AI-generated metadata from the source image",
    )
    meta_group.add_argument(
        "--remove-all-metadata", action="store_true",
        help="Remove all metadata including standard EXIF/XMP (use with --remove-ai)",
    )

    # ── Watermark removal options (default mode) ────────────────
    wm_group = parser.add_argument_group("watermark removal options (default mode)")
    wm_group.add_argument(
        "--strength", type=float, default=0.04,
        help="Regeneration intensity (0.0-1.0). Default: 0.04",
    )
    wm_group.add_argument(
        "--steps", type=int, default=50,
        help="Number of denoising steps. Default: 50",
    )
    wm_group.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model ID. Default: Lykon/dreamshaper-8",
    )
    wm_group.add_argument(
        "--model-profile", type=str, default="default",
        choices=["default", "ctrlregen"],
        help=(
            "Pipeline profile. "
            "'default': img2img regen (Lykon/dreamshaper-8). "
            "'ctrlregen': ControlNet + DINOv2 IP-Adapter. "
            "Default: default"
        ),
    )
    wm_group.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device. Default: auto (CUDA > MPS > CPU)",
    )
    wm_group.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace API token. Falls back to HF_TOKEN env var.",
    )
    wm_group.add_argument(
        "-y", "--yes", action="store_true",
        help="Skip confirmation prompt before downloading models.",
    )

    return parser


# ── Command handlers ────────────────────────────────────────────────

def _handle_remove_ai(args: argparse.Namespace) -> int:
    """Strip AI-related metadata from the source image."""
    output_path = args.output if args.output else args.source
    keep_standard = not args.remove_all_metadata

    try:
        if args.verbose:
            print(f"Source: {args.source}")
            print(f"Output: {output_path}")
            print(f"Keep standard metadata: {keep_standard}")

            if has_ai_metadata(args.source):
                print("\n=== AI METADATA TO REMOVE ===")
                for key, value in extract_ai_metadata(args.source).items():
                    print(_format_metadata_value(key, value))
            else:
                print("\nNo AI metadata found in source file.")
            print()

        result_path = remove_ai_metadata(
            source_path=args.source,
            output_path=output_path,
            keep_standard=keep_standard,
        )

        if has_ai_metadata(result_path):
            print(f"Warning: Some AI metadata may still be present in: {result_path}")
            return 1
        print(f"Successfully removed AI metadata from: {result_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def _handle_clone(args: argparse.Namespace) -> int:
    """Clone metadata from source to target image."""
    try:
        if args.verbose:
            print(f"Source: {args.source}")
            print(f"Target: {args.target}")

            if args.ai_only:
                print("\n=== AI METADATA TO CLONE ===")
                metadata = extract_ai_metadata(args.source)
                if not metadata:
                    print("No AI metadata found in source file!")
                    return 1
            else:
                print("\n=== ALL METADATA TO CLONE ===")
                metadata = extract_metadata(args.source)

            for key, value in metadata.items():
                print(_format_metadata_value(key, value))
            print()

        output_path = clone_metadata(
            source_path=args.source,
            target_path=args.target,
            output_path=args.output,
            ai_only=args.ai_only,
        )

        mode = "AI metadata" if args.ai_only else "metadata"
        print(f"Successfully cloned {mode} to: {output_path}")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


# ── Entry point ─────────────────────────────────────────────────────

def main() -> int:
    """Parse arguments and dispatch to the appropriate command handler."""
    _print_ascii_logo()

    parser = _build_parser()

    if len(sys.argv) == 1:
        parser.print_help()
        return 0

    args = parser.parse_args()

    if not args.source.exists():
        print(f"Error: Source file '{args.source}' does not exist.", file=sys.stderr)
        return 1
    if not is_supported_format(args.source):
        print(
            f"Warning: Source file '{args.source}' may not be a supported format "
            f"({', '.join(SUPPORTED_FORMATS)}).",
            file=sys.stderr,
        )

    # ── Metadata mode ───────────────────────────────────────────
    if args.metadata or args.check_ai or args.remove_ai:
        if args.check_ai:
            if has_ai_metadata(args.source):
                print(f"'{args.source}' contains AI-generated image metadata:")
                print(get_ai_metadata_summary(args.source))
                return 0
            print(f"'{args.source}' does not contain AI-generated image metadata.")
            return 1

        if args.remove_ai:
            return _handle_remove_ai(args)

        if args.target is None:
            print("Error: Target file is required for metadata cloning.", file=sys.stderr)
            print("Usage: noai-watermark source.png target.png --metadata [-o output.png]", file=sys.stderr)
            return 1
        if not args.target.exists():
            print(f"Error: Target file '{args.target}' does not exist.", file=sys.stderr)
            return 1
        if not is_supported_format(args.target):
            print(
                f"Warning: Target file '{args.target}' may not be a supported format "
                f"({', '.join(SUPPORTED_FORMATS)}).",
                file=sys.stderr,
            )

        return _handle_clone(args)

    # ── Default: watermark removal ──────────────────────────────
    from noai_cli_watermark import handle_remove_watermark
    return handle_remove_watermark(args)


if __name__ == "__main__":
    sys.exit(main())
