"""Tests for extractor module."""

from pathlib import Path

from extractor import (
    extract_ai_metadata,
    extract_metadata,
    get_ai_metadata_summary,
    has_ai_metadata,
)


class TestExtractMetadata:
    """Tests for extract_metadata function."""

    def test_extracts_from_png(self, sample_png: Path) -> None:
        metadata = extract_metadata(sample_png)

        assert isinstance(metadata, dict)

    def test_extracts_from_jpg(self, sample_jpg: Path) -> None:
        metadata = extract_metadata(sample_jpg)

        assert isinstance(metadata, dict)

    def test_extracts_standard_metadata(self, sample_png_with_standard_metadata: Path) -> None:
        metadata = extract_metadata(sample_png_with_standard_metadata)

        assert metadata.get("Author") == "Test Author"
        assert metadata.get("Title") == "Test Image"
        assert metadata.get("Description") == "A test image for unit tests"
        assert metadata.get("Copyright") == "Test Copyright"

    def test_extracts_ai_metadata(self, sample_png_with_ai_metadata: Path) -> None:
        metadata = extract_metadata(sample_png_with_ai_metadata)

        assert "parameters" in metadata
        assert "Steps" in metadata["parameters"]

    def test_extracts_c2pa_metadata(self, c2pa_png: Path) -> None:
        metadata = extract_metadata(c2pa_png)

        assert "c2pa" in metadata
        assert metadata["c2pa"].get("has_c2pa") is True

    def test_extracts_c2pa_chunk(self, c2pa_png: Path) -> None:
        metadata = extract_metadata(c2pa_png)

        assert "c2pa_chunk" in metadata
        assert isinstance(metadata["c2pa_chunk"], bytes)


class TestExtractAIMetadata:
    """Tests for extract_ai_metadata function."""

    def test_returns_dict(self, sample_png: Path) -> None:
        ai_metadata = extract_ai_metadata(sample_png)

        assert isinstance(ai_metadata, dict)

    def test_extracts_ai_parameters(self, sample_png_with_ai_metadata: Path) -> None:
        ai_metadata = extract_ai_metadata(sample_png_with_ai_metadata)

        assert "parameters" in ai_metadata
        assert "Model" in ai_metadata

    def test_does_not_extract_standard_metadata(self, sample_png_with_standard_metadata: Path) -> None:
        ai_metadata = extract_ai_metadata(sample_png_with_standard_metadata)

        assert "Author" not in ai_metadata
        assert "Title" not in ai_metadata

    def test_extracts_c2pa_as_ai_metadata(self, c2pa_png: Path) -> None:
        ai_metadata = extract_ai_metadata(c2pa_png)

        assert "c2pa" in ai_metadata
        assert ai_metadata["c2pa"].get("has_c2pa") is True

    def test_detects_openai_metadata(self, openai_png: Path) -> None:
        ai_metadata = extract_ai_metadata(openai_png)

        assert "c2pa" in ai_metadata
        assert "OpenAI" in ai_metadata["c2pa"].get("issuer", "")


class TestHasAIMetadata:
    """Tests for has_ai_metadata function."""

    def test_returns_false_for_regular_png(self, sample_png: Path) -> None:
        assert has_ai_metadata(sample_png) is False

    def test_returns_false_for_standard_metadata(self, sample_png_with_standard_metadata: Path) -> None:
        assert has_ai_metadata(sample_png_with_standard_metadata) is False

    def test_returns_true_for_ai_metadata(self, sample_png_with_ai_metadata: Path) -> None:
        assert has_ai_metadata(sample_png_with_ai_metadata) is True

    def test_returns_true_for_c2pa_metadata(self, c2pa_png: Path) -> None:
        assert has_ai_metadata(c2pa_png) is True

    def test_returns_true_for_openai_metadata(self, openai_png: Path) -> None:
        assert has_ai_metadata(openai_png) is True


class TestGetAIMetadataSummary:
    """Tests for get_ai_metadata_summary function."""

    def test_returns_no_metadata_message_for_regular_png(self, sample_png: Path) -> None:
        summary = get_ai_metadata_summary(sample_png)

        assert "No AI metadata found" in summary

    def test_returns_summary_for_ai_metadata(self, sample_png_with_ai_metadata: Path) -> None:
        summary = get_ai_metadata_summary(sample_png_with_ai_metadata)

        assert "AI Image Metadata" in summary
        assert "parameters" in summary

    def test_returns_summary_for_c2pa(self, c2pa_png: Path) -> None:
        summary = get_ai_metadata_summary(c2pa_png)

        assert "AI Image Metadata" in summary
        assert "C2PA" in summary

    def test_truncates_long_values(self, sample_png_with_ai_metadata: Path) -> None:
        summary = get_ai_metadata_summary(sample_png_with_ai_metadata)

        assert "..." in summary or len(summary) < 500

    def test_does_not_include_raw_c2pa_chunk(self, c2pa_png: Path) -> None:
        summary = get_ai_metadata_summary(c2pa_png)

        assert "c2pa_chunk" not in summary

    def test_includes_c2pa_nested_info(self, openai_png: Path) -> None:
        summary = get_ai_metadata_summary(openai_png)

        assert "C2PA Metadata:" in summary
