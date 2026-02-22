"""Tests for C2PA module."""

from pathlib import Path

import pytest

from c2pa import (
    extract_c2pa_chunk,
    extract_c2pa_info,
    has_c2pa_metadata,
    inject_c2pa_chunk,
)


class TestHasC2PAMetadata:
    """Tests for has_c2pa_metadata function."""

    def test_returns_false_for_regular_png(self, sample_png: Path) -> None:
        assert has_c2pa_metadata(sample_png) is False

    def test_returns_false_for_jpg(self, sample_jpg: Path) -> None:
        assert has_c2pa_metadata(sample_jpg) is False

    def test_returns_false_for_nonexistent_file(self, temp_dir: Path) -> None:
        nonexistent = temp_dir / "nonexistent.png"
        assert has_c2pa_metadata(nonexistent) is False

    def test_detects_c2pa_in_google_image(self, c2pa_png: Path) -> None:
        assert has_c2pa_metadata(c2pa_png) is True

    def test_detects_c2pa_in_openai_image(self, openai_png: Path) -> None:
        assert has_c2pa_metadata(openai_png) is True


class TestExtractC2PAInfo:
    """Tests for extract_c2pa_info function."""

    def test_returns_empty_dict_for_regular_png(self, sample_png: Path) -> None:
        info = extract_c2pa_info(sample_png)
        assert info == {}

    def test_returns_empty_dict_for_jpg(self, sample_jpg: Path) -> None:
        info = extract_c2pa_info(sample_jpg)
        assert info == {}

    def test_extracts_google_c2pa_info(self, c2pa_png: Path) -> None:
        info = extract_c2pa_info(c2pa_png)

        assert info.get("has_c2pa") is True
        assert "C2PA" in info.get("type", "")
        assert info.get("issuer") == "Google LLC"

    def test_extracts_openai_c2pa_info(self, openai_png: Path) -> None:
        info = extract_c2pa_info(openai_png)

        assert info.get("has_c2pa") is True
        assert "C2PA" in info.get("type", "")
        assert "OpenAI" in info.get("issuer", "")

    def test_detects_openai_tools(self, openai_png: Path) -> None:
        info = extract_c2pa_info(openai_png)

        ai_tool = info.get("ai_tool", "")
        assert "GPT-4o" in ai_tool or "ChatGPT" in ai_tool

    def test_contains_timestamp(self, c2pa_png: Path) -> None:
        info = extract_c2pa_info(c2pa_png)

        assert "timestamp" in info
        assert info["timestamp"].endswith("Z")

    def test_detects_actions(self, openai_png: Path) -> None:
        info = extract_c2pa_info(openai_png)

        actions = info.get("actions", "")
        assert "created" in actions or "converted" in actions


class TestExtractC2PAChunk:
    """Tests for extract_c2pa_chunk function."""

    def test_returns_none_for_regular_png(self, sample_png: Path) -> None:
        chunk = extract_c2pa_chunk(sample_png)
        assert chunk is None

    def test_returns_none_for_jpg(self, sample_jpg: Path) -> None:
        chunk = extract_c2pa_chunk(sample_jpg)
        assert chunk is None

    def test_extracts_chunk_from_google_image(self, c2pa_png: Path) -> None:
        chunk = extract_c2pa_chunk(c2pa_png)

        assert chunk is not None
        assert isinstance(chunk, bytes)
        assert len(chunk) > 0

    def test_extracts_chunk_from_openai_image(self, openai_png: Path) -> None:
        chunk = extract_c2pa_chunk(openai_png)

        assert chunk is not None
        assert isinstance(chunk, bytes)
        assert len(chunk) > 0


class TestInjectC2PAChunk:
    """Tests for inject_c2pa_chunk function."""

    def test_injects_chunk_into_png(
        self, sample_png: Path, temp_dir: Path, c2pa_png: Path
    ) -> None:
        chunk = extract_c2pa_chunk(c2pa_png)
        assert chunk is not None

        output_path = temp_dir / "output_with_c2pa.png"
        inject_c2pa_chunk(sample_png, output_path, chunk)

        assert output_path.exists()
        assert has_c2pa_metadata(output_path) is True

    def test_raises_error_for_jpg_target(
        self, sample_jpg: Path, temp_dir: Path, c2pa_png: Path
    ) -> None:
        chunk = extract_c2pa_chunk(c2pa_png)
        assert chunk is not None

        output_path = temp_dir / "output.jpg"

        with pytest.raises(ValueError, match="PNG files"):
            inject_c2pa_chunk(sample_jpg, output_path, chunk)

    def test_raises_error_for_jpg_output(
        self, sample_png: Path, temp_dir: Path, c2pa_png: Path
    ) -> None:
        chunk = extract_c2pa_chunk(c2pa_png)
        assert chunk is not None

        output_path = temp_dir / "output.jpg"

        with pytest.raises(ValueError, match="PNG files"):
            inject_c2pa_chunk(sample_png, output_path, chunk)

    def test_creates_parent_directories(
        self, sample_png: Path, temp_dir: Path, c2pa_png: Path
    ) -> None:
        chunk = extract_c2pa_chunk(c2pa_png)
        assert chunk is not None

        output_path = temp_dir / "nested" / "dir" / "output.png"
        inject_c2pa_chunk(sample_png, output_path, chunk)

        assert output_path.exists()
