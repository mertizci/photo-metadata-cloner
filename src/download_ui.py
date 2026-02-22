"""Download progress UI for model weight downloads.

Extracted from ``cli.py`` — handles styled progress bars, download size
estimation, preload orchestration, and the interactive download prompt.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import re
import sys
import warnings
from typing import Any

# ── Model download size estimation ──────────────────────────────────

CTRLREGEN_REPOS: list[str] = [
    "SG161222/Realistic_Vision_V4.0_noVAE",
    "yepengliu/ctrlregen",
    "facebook/dinov2-giant",
    "stabilityai/sd-vae-ft-mse",
]


_SKIP_EXTENSIONS = (".onnx", ".onnx_data", ".ot", ".msgpack", ".ckpt")
_WEIGHT_EXTENSIONS = (".safetensors", ".bin")
_DIFFUSERS_WEIGHT_NAMES = (
    "model", "pytorch_model", "diffusion_pytorch_model",
)


def _is_root_single_file_checkpoint(name: str) -> bool:
    """True for root-level single-file SD checkpoints that from_pretrained skips."""
    if "/" in name:
        return False
    stem = name.rsplit(".", 1)[0]
    base = stem.replace(".fp16", "").replace("_fp16-no-ema", "")
    return base not in _DIFFUSERS_WEIGHT_NAMES


def _estimate_download_bytes(siblings: list) -> int:
    """Estimate the bytes ``from_pretrained`` would actually download.

    HuggingFace repos often contain multiple variants of the same
    weights (fp32/fp16, safetensors/bin, inpainting, ckpt, onnx).
    ``from_pretrained`` only downloads one variant per component, so
    summing the whole repo vastly overestimates.  This function mimics
    the selection logic: safetensors > bin, fp32 default (fp16 requires
    explicit ``variant="fp16"`` which our pipeline doesn't use), and
    skips onnx/ckpt/inpainting variants, safety_checker, and root-level
    single-file checkpoints when diffusers subdirectories are present.
    """
    has_model_index = any(
        s.rfilename == "model_index.json" for s in siblings
    )

    other_bytes = 0
    groups: dict[str, list[tuple[str, int]]] = {}

    for s in siblings:
        name: str = s.rfilename
        size: int = getattr(s, "size", 0) or 0
        if not size:
            continue
        if name.endswith(_SKIP_EXTENSIONS):
            continue
        if "inpainting" in name.lower():
            continue
        if name.startswith("safety_checker/"):
            continue

        if name.endswith(_WEIGHT_EXTENSIONS):
            if has_model_index and _is_root_single_file_checkpoint(name):
                continue
            stem = name.rsplit(".", 1)[0]
            stem = stem.replace(".fp16", "").replace("_fp16-no-ema", "")
            stem = stem.replace(".non_ema", "").replace(".ema", "")
            stem = stem.replace(".pruned", "")
            stem = stem.replace("pytorch_model", "model")
            stem = stem.replace("diffusion_pytorch_model", "model")
            groups.setdefault(stem, []).append((name, size))
        else:
            other_bytes += size

    weight_bytes = 0
    for candidates in groups.values():
        best: tuple[str, int] | None = None
        for cname, csize in candidates:
            is_fp16 = "fp16" in cname
            is_st = cname.endswith(".safetensors")
            b_fp16 = best and "fp16" in best[0]
            b_st = best and best[0].endswith(".safetensors")
            if best is None:
                best = (cname, csize)
            elif not is_fp16 and b_fp16:
                best = (cname, csize)
            elif is_fp16 == b_fp16 and is_st and not b_st:
                best = (cname, csize)
        if best:
            weight_bytes += best[1]

    return weight_bytes + other_bytes


def _fetch_repo_size_gb(repo_id: str) -> float | None:
    """Query HuggingFace Hub API for estimated download size in GB."""
    try:
        from huggingface_hub import model_info

        info = model_info(repo_id, files_metadata=True)
        total_bytes = _estimate_download_bytes(info.siblings or [])
        if total_bytes > 0:
            return round(total_bytes / (1024**3), 1) or 0.1
    except Exception:
        pass
    return None


def get_models_to_download(
    model_id: str, profile: str,
) -> list[tuple[str, float | None]]:
    """Return list of (model_id, size_gb) tuples that need downloading.

    Sizes are fetched live from HuggingFace Hub.  Returns ``None`` for
    size when the API call fails.
    """
    repo_ids = CTRLREGEN_REPOS if profile == "ctrlregen" else [model_id]

    try:
        from huggingface_hub import scan_cache_dir
        cached_repos = {repo.repo_id for repo in scan_cache_dir().repos}
    except Exception:
        cached_repos = set()

    pending = [rid for rid in repo_ids if rid not in cached_repos]
    return [(rid, _fetch_repo_size_gb(rid)) for rid in pending]


# ── Styled stderr download progress filter ──────────────────────────

class DownloadProgressFilter:
    """Stderr wrapper that renders download progress as a styled color bar.

    Intercepts raw tqdm output from HuggingFace Hub, parses it, and
    re-renders with ANSI colors and a clean ``━`` bar style.  Small file
    downloads (config, tokenizer, …) and ``Fetching N files`` wrappers
    are silently discarded.
    """

    _HAS_SIZE = re.compile(r"\d+(\.\d+)?\s*[MG]i?B?/")
    _PCT = re.compile(r"(\d+)%")
    _SIZES = re.compile(r"([\d.]+\s*[kMGT]?i?B?)\s*/\s*([\d.]+\s*[kMGT]?i?B?)")
    _SPEED = re.compile(r",\s*([\d.]+\s*[kMGT]?i?B/s)")
    _ETA = re.compile(r"<\s*([\d:]+)")
    _W = 30

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        _nc = bool(os.environ.get("NO_COLOR"))
        self._cy = "\033[36m" if not _nc else ""
        self._gn = "\033[32m" if not _nc else ""
        self._gb = "\033[32;1m" if not _nc else ""
        self._yl = "\033[33;1m" if not _nc else ""
        self._dm = "\033[2m" if not _nc else ""
        self._rs = "\033[0m" if not _nc else ""
        self.rendered_complete = False
        self._last_tot = ""
        self._last_speed = ""

    def write(self, s: str) -> int:
        if "Fetching" in s:
            return len(s)
        if not self._HAS_SIZE.search(s):
            return len(s)
        m = self._PCT.search(s)
        if not m:
            return len(s)
        pct = int(m.group(1))

        sm = self._SIZES.search(s)
        cur, tot = (sm.group(1).strip(), sm.group(2).strip()) if sm else ("", "")
        if tot:
            self._last_tot = tot

        sp = self._SPEED.search(s)
        speed = sp.group(1) if sp else ""
        if speed:
            self._last_speed = speed

        em = self._ETA.search(s)
        eta = em.group(1) if em else ""

        filled = int(self._W * pct / 100)
        on = "━" * filled
        off = "━" * (self._W - filled)

        if pct >= 100:
            self.rendered_complete = True
            line = (
                f"\r  {self._gn}{on}{self._rs}"
                f"  {self._gb}✓ Complete{self._rs}"
                f"  {self._dm}{tot or self._last_tot}"
            )
            if speed or self._last_speed:
                line += f"  •  {speed or self._last_speed}"
            line += f"{self._rs}\033[K\n"
        else:
            line = (
                f"\r  {self._cy}{on}{self._dm}{off}{self._rs}"
                f"  {self._yl}{pct:3d}%{self._rs}"
                f"  {cur} / {tot}"
            )
            if speed:
                line += f"  {self._dm}•  {speed}{self._rs}"
            if eta and "?" not in eta:
                line += f"  {self._dm}ETA {eta}{self._rs}"
            line += "\033[K"

        return self._stream.write(line)

    def render_complete(self) -> None:
        """Force-render a 100% completion bar (call after downloads finish)."""
        if self.rendered_complete:
            return
        on = "━" * self._W
        line = (
            f"\r  {self._gn}{on}{self._rs}"
            f"  {self._gb}✓ Complete{self._rs}"
        )
        if self._last_tot:
            line += f"  {self._dm}{self._last_tot}"
        if self._last_speed:
            line += f"  •  {self._last_speed}"
        line += f"{self._rs}\033[K\n"
        self._stream.write(line)
        self._stream.flush()

    def flush(self) -> None:
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


# ── Preload with styled progress ────────────────────────────────────

def preload_silently(remover: Any) -> None:
    """Load models showing only styled download progress bars."""
    for name in ("transformers", "diffusers", "huggingface_hub", "torch"):
        logging.getLogger(name).setLevel(logging.CRITICAL)

    os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)

    real_stderr = sys.stderr
    filt = DownloadProgressFilter(real_stderr)
    sys.stderr = filt  # type: ignore[assignment]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(io.StringIO()):
                remover.preload()
    finally:
        sys.stderr = real_stderr

    filt.render_complete()


def _format_size(size_gb: float | None) -> str:
    if size_gb is None:
        return ""
    return f"  ({size_gb:.1f} GB)"


def prompt_for_download(
    pending: list[tuple[str, float | None]],
    *,
    skip_prompt: bool = False,
) -> bool:
    """Show download summary and ask for confirmation.

    Returns True if user confirmed or skip_prompt is True.
    """
    if not pending:
        return True

    if skip_prompt:
        return True

    known_sizes = [sz for _, sz in pending if sz is not None]
    if known_sizes:
        total_gb = sum(known_sizes)
        header = f"~{total_gb:.1f} GB total"
    else:
        header = "size unknown"

    print(f"\nThe following models will be downloaded ({header}):\n")
    for mid, sz in pending:
        print(f"  • {mid}{_format_size(sz)}")
    print(f"\nModels are cached after download — subsequent runs won't re-download.\n")

    try:
        answer = input("Continue? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        return False

    if answer and answer not in ("y", "yes"):
        print("Aborted.")
        return False

    return True
