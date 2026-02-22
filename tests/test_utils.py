"""Tests for utils module."""

from pathlib import Path

from utils import get_image_format, is_supported_format


class TestIsSupportedFormat:
    """Tests for is_supported_format function."""

    def test_png_lowercase(self) -> None:
        assert is_supported_format(Path("image.png")) is True

    def test_png_uppercase(self) -> None:
        assert is_supported_format(Path("image.PNG")) is True

    def test_jpg_lowercase(self) -> None:
        assert is_supported_format(Path("image.jpg")) is True

    def test_jpg_uppercase(self) -> None:
        assert is_supported_format(Path("image.JPG")) is True

    def test_jpeg_lowercase(self) -> None:
        assert is_supported_format(Path("image.jpeg")) is True

    def test_jpeg_uppercase(self) -> None:
        assert is_supported_format(Path("image.JPEG")) is True

    def test_unsupported_gif(self) -> None:
        assert is_supported_format(Path("image.gif")) is False

    def test_unsupported_bmp(self) -> None:
        assert is_supported_format(Path("image.bmp")) is False

    def test_unsupported_webp(self) -> None:
        assert is_supported_format(Path("image.webp")) is False

    def test_unsupported_tiff(self) -> None:
        assert is_supported_format(Path("image.tiff")) is False

    def test_no_extension(self) -> None:
        assert is_supported_format(Path("image")) is False

    def test_with_path_object(self) -> None:
        assert is_supported_format(Path("/some/path/image.png")) is True


class TestGetImageFormat:
    """Tests for get_image_format function."""

    def test_png_returns_png(self) -> None:
        assert get_image_format(Path("image.png")) == "PNG"

    def test_png_uppercase(self) -> None:
        assert get_image_format(Path("image.PNG")) == "PNG"

    def test_jpg_returns_jpeg(self) -> None:
        assert get_image_format(Path("image.jpg")) == "JPEG"

    def test_jpg_uppercase(self) -> None:
        assert get_image_format(Path("image.JPG")) == "JPEG"

    def test_jpeg_returns_jpeg(self) -> None:
        assert get_image_format(Path("image.jpeg")) == "JPEG"

    def test_jpeg_uppercase(self) -> None:
        assert get_image_format(Path("image.JPEG")) == "JPEG"

    def test_unknown_format_defaults_to_png(self) -> None:
        assert get_image_format(Path("image.gif")) == "PNG"

    def test_no_extension_defaults_to_png(self) -> None:
        assert get_image_format(Path("image")) == "PNG"
