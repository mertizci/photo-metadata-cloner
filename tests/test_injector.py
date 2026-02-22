"""Tests for injector module."""

from pathlib import Path

from PIL import Image

from extractor import extract_metadata
from injector import inject_metadata


class TestInjectMetadata:
    """Tests for inject_metadata function."""

    def test_injects_into_png(self, sample_png: Path, temp_dir: Path) -> None:
        metadata = {"Author": "Test Author", "Title": "Test Title"}
        output_path = temp_dir / "output.png"

        inject_metadata(sample_png, output_path, metadata)

        assert output_path.exists()
        extracted = extract_metadata(output_path)
        assert extracted.get("Author") == "Test Author"
        assert extracted.get("Title") == "Test Title"

    def test_injects_into_jpg(self, sample_jpg: Path, temp_dir: Path) -> None:
        metadata = {"Author": "Test Author"}
        output_path = temp_dir / "output.jpg"

        inject_metadata(sample_jpg, output_path, metadata)

        assert output_path.exists()

    def test_creates_output_directory(self, sample_png: Path, temp_dir: Path) -> None:
        metadata = {"Author": "Test Author"}
        output_path = temp_dir / "nested" / "dir" / "output.png"

        inject_metadata(sample_png, output_path, metadata)

        assert output_path.exists()

    def test_converts_rgba_to_rgb_for_jpeg(self, sample_png_rgba: Path, temp_dir: Path) -> None:
        metadata = {}
        output_path = temp_dir / "output.jpg"

        inject_metadata(sample_png_rgba, output_path, metadata)

        assert output_path.exists()
        with Image.open(output_path) as img:
            assert img.mode == "RGB"

    def test_preserves_png_alpha(self, sample_png_rgba: Path, temp_dir: Path) -> None:
        metadata = {}
        output_path = temp_dir / "output.png"

        inject_metadata(sample_png_rgba, output_path, metadata)

        assert output_path.exists()
        with Image.open(output_path) as img:
            assert img.mode == "RGBA"

    def test_injects_ai_metadata(self, sample_png: Path, temp_dir: Path) -> None:
        ai_metadata = {
            "parameters": "Test prompt, Steps: 20, Sampler: Euler",
            "Model": "test-model",
        }
        output_path = temp_dir / "output.png"

        inject_metadata(sample_png, output_path, ai_metadata)

        extracted = extract_metadata(output_path)
        assert "parameters" in extracted
        assert "Model" in extracted

    def test_injects_dpi(self, sample_png: Path, temp_dir: Path) -> None:
        metadata = {"dpi": (300, 300), "Author": "Test"}
        output_path = temp_dir / "output.png"

        inject_metadata(sample_png, output_path, metadata)

        assert output_path.exists()

    def test_overwrites_existing_file(self, sample_png: Path) -> None:
        metadata1 = {"Author": "Author 1"}
        metadata2 = {"Author": "Author 2"}

        inject_metadata(sample_png, sample_png, metadata1)
        inject_metadata(sample_png, sample_png, metadata2)

        extracted = extract_metadata(sample_png)
        assert extracted.get("Author") == "Author 2"

    def test_handles_bytes_metadata(self, sample_png: Path, temp_dir: Path) -> None:
        metadata = {"Custom": b"bytes value"}
        output_path = temp_dir / "output.png"

        inject_metadata(sample_png, output_path, metadata)

        assert output_path.exists()


class TestInjectC2PA:
    """Tests for C2PA injection via inject_metadata."""

    def test_injects_c2pa_chunk(self, sample_png: Path, temp_dir: Path, c2pa_png: Path) -> None:
        from c2pa import extract_c2pa_chunk, has_c2pa_metadata

        c2pa_chunk = extract_c2pa_chunk(c2pa_png)
        assert c2pa_chunk is not None

        metadata = {"c2pa_chunk": c2pa_chunk, "Author": "Test"}
        output_path = temp_dir / "output_with_c2pa.png"

        inject_metadata(sample_png, output_path, metadata)

        assert output_path.exists()
        assert has_c2pa_metadata(output_path) is True

    def test_ignores_c2pa_chunk_for_jpeg(
        self, sample_jpg: Path, temp_dir: Path, c2pa_png: Path
    ) -> None:
        from c2pa import extract_c2pa_chunk

        c2pa_chunk = extract_c2pa_chunk(c2pa_png)
        assert c2pa_chunk is not None

        metadata = {"c2pa_chunk": c2pa_chunk, "Author": "Test"}
        output_path = temp_dir / "output.jpg"

        inject_metadata(sample_jpg, output_path, metadata)

        assert output_path.exists()


class TestCrossFormatInjection:
    """Tests for cross-format metadata injection."""

    def test_png_to_jpg(self, sample_png_with_ai_metadata: Path, temp_dir: Path) -> None:
        metadata = extract_metadata(sample_png_with_ai_metadata)
        output_path = temp_dir / "output.jpg"

        inject_metadata(sample_png_with_ai_metadata, output_path, metadata)

        assert output_path.exists()
        assert output_path.suffix == ".jpg"

    def test_jpg_to_png(self, sample_jpg: Path, temp_dir: Path, sample_png: Path) -> None:
        ai_metadata = {
            "parameters": "Test prompt",
            "Model": "test-model",
        }
        output_path = temp_dir / "output.png"

        inject_metadata(sample_png, output_path, ai_metadata)

        assert output_path.exists()
        extracted = extract_metadata(output_path)
        assert extracted.get("parameters") == "Test prompt"
