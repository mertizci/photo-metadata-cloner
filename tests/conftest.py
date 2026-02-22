"""Test configuration and fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from PIL import Image


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_png(temp_dir: Path) -> Path:
    """Create a sample PNG image for testing."""
    img_path = temp_dir / "sample.png"
    img = Image.new("RGB", (100, 100), color="red")
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def sample_jpg(temp_dir: Path) -> Path:
    """Create a sample JPG image for testing."""
    img_path = temp_dir / "sample.jpg"
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(img_path, "JPEG")
    return img_path


@pytest.fixture
def sample_png_with_ai_metadata(temp_dir: Path) -> Path:
    """Create a PNG image with AI metadata (Stable Diffusion style)."""
    from PIL.PngImagePlugin import PngInfo

    img_path = temp_dir / "ai_sample.png"
    img = Image.new("RGB", (512, 512), color="green")

    metadata = PngInfo()
    metadata.add_text(
        "parameters",
        "A beautiful landscape, Steps: 30, Sampler: Euler a, CFG scale: 7.5, Seed: 12345, Size: 512x512",
    )
    metadata.add_text("Model", "v1-5-pruned-emaonly")
    metadata.add_text("Software", "Stable Diffusion WebUI")

    img.save(img_path, "PNG", pnginfo=metadata)
    return img_path


@pytest.fixture
def sample_png_with_standard_metadata(temp_dir: Path) -> Path:
    """Create a PNG image with standard metadata."""
    from PIL.PngImagePlugin import PngInfo

    img_path = temp_dir / "standard_sample.png"
    img = Image.new("RGB", (100, 100), color="yellow")

    metadata = PngInfo()
    metadata.add_text("Author", "Test Author")
    metadata.add_text("Title", "Test Image")
    metadata.add_text("Description", "A test image for unit tests")
    metadata.add_text("Copyright", "Test Copyright")

    img.save(img_path, "PNG", pnginfo=metadata)
    return img_path


@pytest.fixture
def sample_png_rgba(temp_dir: Path) -> Path:
    """Create a PNG image with alpha channel."""
    img_path = temp_dir / "sample_rgba.png"
    img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
    img.save(img_path, "PNG")
    return img_path


@pytest.fixture
def c2pa_png(temp_dir: Path) -> Path:
    """Copy the real C2PA PNG file to temp directory for testing."""
    import shutil

    source = Path("/Users/mertizci/Projects/photo-metadata-cloner/1.png")
    if not source.exists():
        pytest.skip("C2PA test file (1.png) not found")

    dest = temp_dir / "c2pa_test.png"
    shutil.copy(source, dest)
    return dest


@pytest.fixture
def openai_png(temp_dir: Path) -> Path:
    """Copy the real OpenAI PNG file to temp directory for testing."""
    import shutil

    source = Path("/Users/mertizci/Projects/photo-metadata-cloner/2.png")
    if not source.exists():
        pytest.skip("OpenAI test file (2.png) not found")

    dest = temp_dir / "openai_test.png"
    shutil.copy(source, dest)
    return dest
