"""Tests for cloner module."""

from pathlib import Path

from cloner import (
    clone_ai_metadata,
    clone_metadata,
    has_ai_content,
    remove_ai_metadata,
)
from extractor import extract_metadata, has_ai_metadata


class TestCloneMetadata:
    """Tests for clone_metadata function."""

    def test_clones_all_metadata(
        self, sample_png_with_ai_metadata: Path, sample_png: Path, temp_dir: Path
    ) -> None:
        output_path = temp_dir / "output.png"

        result = clone_metadata(sample_png_with_ai_metadata, sample_png, output_path)

        assert result == output_path
        assert output_path.exists()

        extracted = extract_metadata(output_path)
        assert "parameters" in extracted
        assert "Model" in extracted

    def test_clones_to_same_file(
        self, sample_png_with_ai_metadata: Path, sample_png: Path, temp_dir: Path
    ) -> None:
        target = temp_dir / "target.png"

        from PIL import Image

        img = Image.new("RGB", (100, 100), color="blue")
        img.save(target, "PNG")

        result = clone_metadata(sample_png_with_ai_metadata, target)

        assert result == target
        assert target.exists()

    def test_clones_ai_only(
        self, sample_png_with_ai_metadata: Path, sample_png_with_standard_metadata: Path, temp_dir: Path
    ) -> None:
        output_path = temp_dir / "output.png"

        result = clone_metadata(
            sample_png_with_ai_metadata,
            sample_png_with_standard_metadata,
            output_path,
            ai_only=True,
        )

        extracted = extract_metadata(output_path)
        assert "parameters" in extracted
        assert extracted.get("Author") != "Test Author"

    def test_cross_format_png_to_jpg(
        self, sample_png_with_ai_metadata: Path, sample_jpg: Path, temp_dir: Path
    ) -> None:
        output_path = temp_dir / "output.jpg"

        result = clone_metadata(sample_png_with_ai_metadata, sample_jpg, output_path)

        assert result.suffix == ".jpg"
        assert output_path.exists()

    def test_cross_format_jpg_to_png(
        self, sample_jpg: Path, sample_png: Path, temp_dir: Path
    ) -> None:
        output_path = temp_dir / "output.png"

        result = clone_metadata(sample_png, sample_jpg, output_path)

        assert result.suffix == ".png"
        assert output_path.exists()

    def test_creates_output_directory(
        self, sample_png_with_ai_metadata: Path, sample_png: Path, temp_dir: Path
    ) -> None:
        output_path = temp_dir / "nested" / "dir" / "output.png"

        result = clone_metadata(sample_png_with_ai_metadata, sample_png, output_path)

        assert output_path.exists()

    def test_clones_c2pa_metadata(
        self, c2pa_png: Path, sample_png: Path, temp_dir: Path
    ) -> None:
        from c2pa import has_c2pa_metadata

        output_path = temp_dir / "output.png"

        result = clone_metadata(c2pa_png, sample_png, output_path)

        assert has_c2pa_metadata(output_path) is True


class TestCloneAIMetadata:
    """Tests for clone_ai_metadata function."""

    def test_clones_only_ai_metadata(
        self, sample_png_with_ai_metadata: Path, sample_png_with_standard_metadata: Path, temp_dir: Path
    ) -> None:
        output_path = temp_dir / "output.png"

        result = clone_ai_metadata(
            sample_png_with_ai_metadata, sample_png_with_standard_metadata, output_path
        )

        assert result == output_path
        extracted = extract_metadata(output_path)
        assert "parameters" in extracted
        assert extracted.get("Author") != "Test Author"

    def test_returns_output_path(self, sample_png_with_ai_metadata: Path, sample_png: Path, temp_dir: Path) -> None:
        output_path = temp_dir / "output.png"

        result = clone_ai_metadata(sample_png_with_ai_metadata, sample_png, output_path)

        assert result == output_path

    def test_target_becomes_output_when_not_specified(
        self, sample_png_with_ai_metadata: Path, sample_png: Path, temp_dir: Path
    ) -> None:
        target = temp_dir / "target.png"

        from PIL import Image

        img = Image.new("RGB", (50, 50), color="red")
        img.save(target, "PNG")

        result = clone_ai_metadata(sample_png_with_ai_metadata, target)

        assert result == target
        assert has_ai_metadata(target) is True

    def test_clones_c2pa_as_ai_metadata(
        self, c2pa_png: Path, sample_png: Path, temp_dir: Path
    ) -> None:
        from c2pa import has_c2pa_metadata

        output_path = temp_dir / "output.png"

        clone_ai_metadata(c2pa_png, sample_png, output_path)

        assert has_c2pa_metadata(output_path) is True

    def test_clones_openai_c2pa(
        self, openai_png: Path, sample_png: Path, temp_dir: Path
    ) -> None:
        from c2pa import has_c2pa_metadata

        output_path = temp_dir / "output.png"

        clone_ai_metadata(openai_png, sample_png, output_path)

        assert has_c2pa_metadata(output_path) is True


class TestIntegration:
    """Integration tests for cloner module."""

    def test_full_workflow_with_c2pa(
        self, c2pa_png: Path, sample_png: Path, temp_dir: Path
    ) -> None:
        from c2pa import extract_c2pa_info, has_c2pa_metadata

        intermediate = temp_dir / "intermediate.png"
        final = temp_dir / "final.jpg"

        clone_metadata(c2pa_png, sample_png, intermediate)

        assert has_c2pa_metadata(intermediate) is True
        c2pa_info = extract_c2pa_info(intermediate)
        assert c2pa_info.get("issuer") == "Google LLC"

        clone_metadata(c2pa_png, sample_png, final)

        assert final.exists()

    def test_chain_cloning(
        self, sample_png_with_ai_metadata: Path, sample_png: Path, temp_dir: Path
    ) -> None:
        output1 = temp_dir / "output1.png"
        output2 = temp_dir / "output2.png"
        output3 = temp_dir / "output3.png"

        clone_metadata(sample_png_with_ai_metadata, sample_png, output1)
        clone_metadata(output1, sample_png, output2)
        clone_metadata(output2, sample_png, output3)

        for output in [output1, output2, output3]:
            extracted = extract_metadata(output)
            assert "parameters" in extracted
