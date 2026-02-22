"""Tests for cleaner module."""

from pathlib import Path

from cleaner import (
    _is_ai_metadata_key,
    has_ai_content,
    remove_ai_metadata,
)
from extractor import extract_metadata, has_ai_metadata


class TestIsAIMetadataKey:
    """Tests for _is_ai_metadata_key function."""

    def test_detects_parameters(self) -> None:
        assert _is_ai_metadata_key("parameters") is True

    def test_detects_workflow(self) -> None:
        assert _is_ai_metadata_key("workflow") is True

    def test_detects_model(self) -> None:
        assert _is_ai_metadata_key("Model") is True

    def test_detects_seed(self) -> None:
        assert _is_ai_metadata_key("Seed") is True

    def test_detects_prompt_in_key(self) -> None:
        assert _is_ai_metadata_key("custom_prompt") is True

    def test_detects_stable_in_key(self) -> None:
        assert _is_ai_metadata_key("stable_diffusion_data") is True

    def test_does_not_detect_author(self) -> None:
        assert _is_ai_metadata_key("Author") is False

    def test_does_not_detect_title(self) -> None:
        assert _is_ai_metadata_key("Title") is False

    def test_does_not_detect_description(self) -> None:
        assert _is_ai_metadata_key("Description") is False

    def test_case_insensitive(self) -> None:
        assert _is_ai_metadata_key("PARAMETERS") is True
        assert _is_ai_metadata_key("MODEL") is True


class TestRemoveAIMetadata:
    """Tests for remove_ai_metadata function."""

    def test_removes_ai_metadata_from_png(
        self, sample_png_with_ai_metadata: Path, temp_dir: Path
    ) -> None:
        output_path = temp_dir / "cleaned.png"

        assert has_ai_metadata(sample_png_with_ai_metadata) is True

        result = remove_ai_metadata(sample_png_with_ai_metadata, output_path)

        assert result == output_path
        assert has_ai_metadata(output_path) is False

    def test_keeps_standard_metadata_by_default(
        self, sample_png_with_ai_metadata: Path, temp_dir: Path
    ) -> None:
        output_path = temp_dir / "cleaned.png"

        remove_ai_metadata(sample_png_with_ai_metadata, output_path)

        metadata = extract_metadata(output_path)
        # AI metadata should be gone
        assert "parameters" not in metadata
        assert "Model" not in metadata

    def test_removes_standard_metadata_when_keep_standard_false(
        self, sample_png_with_standard_metadata: Path, temp_dir: Path
    ) -> None:
        output_path = temp_dir / "cleaned.png"

        # Even though this doesn't have AI metadata, we test the flag
        remove_ai_metadata(
            sample_png_with_standard_metadata, output_path, keep_standard=False
        )

        metadata = extract_metadata(output_path)
        assert "Author" not in metadata
        assert "Title" not in metadata

    def test_modifies_in_place_when_output_not_specified(
        self, sample_png_with_ai_metadata: Path, temp_dir: Path
    ) -> None:
        import shutil

        test_file = temp_dir / "test.png"
        shutil.copy(sample_png_with_ai_metadata, test_file)

        assert has_ai_metadata(test_file) is True

        result = remove_ai_metadata(test_file)

        assert result == test_file
        assert has_ai_metadata(test_file) is False

    def test_creates_output_directory(
        self, sample_png_with_ai_metadata: Path, temp_dir: Path
    ) -> None:
        output_path = temp_dir / "nested" / "dir" / "cleaned.png"

        remove_ai_metadata(sample_png_with_ai_metadata, output_path)

        assert output_path.exists()
        assert has_ai_metadata(output_path) is False

    def test_works_with_jpg(
        self, sample_png_with_ai_metadata: Path, sample_jpg: Path, temp_dir: Path
    ) -> None:
        output_path = temp_dir / "cleaned.jpg"

        # First add AI metadata to JPG via cloning
        from cloner import clone_metadata

        clone_metadata(sample_png_with_ai_metadata, sample_jpg, output_path)

        # Now remove it
        cleaned_path = temp_dir / "final_cleaned.jpg"
        remove_ai_metadata(output_path, cleaned_path)

        assert cleaned_path.exists()

    def test_removes_c2pa_metadata(self, c2pa_png: Path, temp_dir: Path) -> None:
        from c2pa import has_c2pa_metadata

        output_path = temp_dir / "cleaned.png"

        assert has_c2pa_metadata(c2pa_png) is True

        remove_ai_metadata(c2pa_png, output_path)

        assert has_c2pa_metadata(output_path) is False
        assert has_ai_metadata(output_path) is False

    def test_removes_openai_c2pa_metadata(self, openai_png: Path, temp_dir: Path) -> None:
        from c2pa import has_c2pa_metadata

        output_path = temp_dir / "cleaned.png"

        assert has_c2pa_metadata(openai_png) is True

        remove_ai_metadata(openai_png, output_path)

        assert has_c2pa_metadata(output_path) is False
        assert has_ai_metadata(output_path) is False

    def test_preserves_dpi(self, sample_png_with_ai_metadata: Path, temp_dir: Path) -> None:
        output_path = temp_dir / "cleaned.png"

        remove_ai_metadata(sample_png_with_ai_metadata, output_path)

        assert output_path.exists()

    def test_works_on_image_without_ai_metadata(
        self, sample_png: Path, temp_dir: Path
    ) -> None:
        output_path = temp_dir / "cleaned.png"

        result = remove_ai_metadata(sample_png, output_path)

        assert result.exists()
        assert has_ai_metadata(output_path) is False


class TestHasAIContent:
    """Tests for has_ai_content function."""

    def test_returns_true_for_ai_metadata(self, sample_png_with_ai_metadata: Path) -> None:
        assert has_ai_content(sample_png_with_ai_metadata) is True

    def test_returns_true_for_c2pa(self, c2pa_png: Path) -> None:
        assert has_ai_content(c2pa_png) is True

    def test_returns_true_for_openai(self, openai_png: Path) -> None:
        assert has_ai_content(openai_png) is True

    def test_returns_false_for_regular_png(self, sample_png: Path) -> None:
        assert has_ai_content(sample_png) is False

    def test_returns_false_for_standard_metadata(
        self, sample_png_with_standard_metadata: Path
    ) -> None:
        assert has_ai_content(sample_png_with_standard_metadata) is False


class TestIntegration:
    """Integration tests for cleaner module."""

    def test_clean_and_verify(
        self, sample_png_with_ai_metadata: Path, temp_dir: Path
    ) -> None:
        cleaned = temp_dir / "cleaned.png"

        # Verify original has AI metadata
        assert has_ai_metadata(sample_png_with_ai_metadata) is True

        # Clean it
        remove_ai_metadata(sample_png_with_ai_metadata, cleaned)

        # Verify cleaned has no AI metadata
        assert has_ai_metadata(cleaned) is False

        # Verify file is still valid image
        from PIL import Image

        with Image.open(cleaned) as img:
            assert img.size == (512, 512)

    def test_clean_cloned_image(
        self, sample_png_with_ai_metadata: Path, sample_png: Path, temp_dir: Path
    ) -> None:
        from cloner import clone_ai_metadata

        # Clone AI metadata to a clean image
        cloned = temp_dir / "cloned.png"
        clone_ai_metadata(sample_png_with_ai_metadata, sample_png, cloned)
        assert has_ai_metadata(cloned) is True

        # Clean the cloned image
        cleaned = temp_dir / "cleaned.png"
        remove_ai_metadata(cloned, cleaned)
        assert has_ai_metadata(cleaned) is False
