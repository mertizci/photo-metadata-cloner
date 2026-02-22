"""Command-line interface for noai-watermark.

Provides the ``noai-watermark`` entry point that dispatches to one of
four workflows:

- ``--check-ai``        — inspect AI metadata
- ``--remove-ai``       — strip AI metadata fields
- ``--remove-watermark`` — diffusion-based invisible watermark removal
- *(default)*           — clone metadata between images

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

_ASCII_LOGO = """
 ███╗   ██╗ ██████╗  █████╗ ██╗
 ████╗  ██║██╔═══██╗██╔══██╗██║
 ██╔██╗ ██║██║   ██║███████║██║
 ██║╚██╗██║██║   ██║██╔══██║██║
 ██║ ╚████║╚██████╔╝██║  ██║██║
 ╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝
    ─── noai-watermark ───
"""


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
        description="Clone, check, and remove AI-generated metadata between PNG and JPG files.",
        epilog="Example: noai-watermark source.png target.png -o output.png",
    )

    parser.add_argument(
        "source", type=Path,
        help=f"Source image file (formats: {', '.join(SUPPORTED_FORMATS)})",
    )
    parser.add_argument(
        "target", type=Path, nargs="?",
        help="Target image file to apply metadata to (not needed with --check-ai or --remove-ai)",
    )
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output file path (default: overwrite target/source file)",
    )
    parser.add_argument(
        "-a", "--ai-only", action="store_true",
        help="Clone only AI-generated metadata (Stable Diffusion, ComfyUI, Midjourney, etc.)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print extracted metadata information",
    )
    parser.add_argument(
        "--check-ai", action="store_true",
        help="Check if source contains AI metadata, don't clone",
    )
    parser.add_argument(
        "--remove-ai", action="store_true",
        help="Remove all AI-generated metadata from the source image",
    )
    parser.add_argument(
        "--remove-all-metadata", action="store_true",
        help="Remove all AI metadata and standard metadata (use with --remove-ai)",
    )
    parser.add_argument(
        "--remove-watermark", action="store_true",
        help="Remove invisible watermarks using diffusion model regeneration",
    )
    parser.add_argument(
        "--strength", type=float, default=0.04,
        help="Watermark removal strength (0.0-1.0). Default: 0.04",
    )
    parser.add_argument(
        "--steps", type=int, default=50,
        help="Number of denoising steps for watermark removal. Default: 50",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="HuggingFace model ID for watermark removal. Default: Lykon/dreamshaper-8",
    )
    parser.add_argument(
        "--model-profile", type=str, default="default",
        choices=["default", "ctrlregen"],
        help=(
            "Model profile shortcut. "
            "'default' uses Lykon/dreamshaper-8 (simple img2img regen). "
            "'ctrlregen' uses CtrlRegen (ControlNet + DINOv2 IP-Adapter); "
            "requires: pip install noai-watermark[ctrlregen]."
        ),
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Inference device for watermark removal. Default: auto",
    )
    parser.add_argument(
        "--hf-token", type=str, default=None,
        help="HuggingFace API token for authenticated model downloads. Falls back to HF_TOKEN env var.",
    )
    parser.add_argument(
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

    if args.check_ai:
        if has_ai_metadata(args.source):
            print(f"'{args.source}' contains AI-generated image metadata:")
            print(get_ai_metadata_summary(args.source))
            return 0
        print(f"'{args.source}' does not contain AI-generated image metadata.")
        return 1

    if args.remove_ai:
        return _handle_remove_ai(args)

    if args.remove_watermark:
        from cli_watermark import handle_remove_watermark
        return handle_remove_watermark(args)

    if args.target is None:
        print("Error: Target file is required for cloning.", file=sys.stderr)
        print("Usage: noai-watermark source.png target.png [-o output.png]", file=sys.stderr)
        print("       noai-watermark source.png --remove-ai [-o output.png]", file=sys.stderr)
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


if __name__ == "__main__":
    sys.exit(main())
