"""Tests for metadata_handler module (re-exports)."""

import metadata_handler


class TestMetadataHandlerReExports:
    """Tests to verify all public APIs are properly re-exported."""

    def test_constants_are_exported(self) -> None:
        assert hasattr(metadata_handler, "SUPPORTED_FORMATS")
        assert hasattr(metadata_handler, "AI_METADATA_KEYS")
        assert hasattr(metadata_handler, "PNG_METADATA_KEYS")
        assert hasattr(metadata_handler, "C2PA_CHUNK_TYPE")

    def test_utils_are_exported(self) -> None:
        assert hasattr(metadata_handler, "is_supported_format")
        assert hasattr(metadata_handler, "get_image_format")

    def test_c2pa_functions_are_exported(self) -> None:
        assert hasattr(metadata_handler, "has_c2pa_metadata")
        assert hasattr(metadata_handler, "extract_c2pa_info")
        assert hasattr(metadata_handler, "extract_c2pa_chunk")
        assert hasattr(metadata_handler, "inject_c2pa_chunk")

    def test_extractor_functions_are_exported(self) -> None:
        assert hasattr(metadata_handler, "extract_metadata")
        assert hasattr(metadata_handler, "extract_ai_metadata")
        assert hasattr(metadata_handler, "has_ai_metadata")
        assert hasattr(metadata_handler, "get_ai_metadata_summary")

    def test_injector_functions_are_exported(self) -> None:
        assert hasattr(metadata_handler, "inject_metadata")

    def test_cloner_functions_are_exported(self) -> None:
        assert hasattr(metadata_handler, "clone_metadata")
        assert hasattr(metadata_handler, "clone_ai_metadata")

    def test_cleaner_functions_are_exported(self) -> None:
        assert hasattr(metadata_handler, "remove_ai_metadata")
        assert hasattr(metadata_handler, "has_ai_content")

    def test_all_list_is_complete(self) -> None:
        assert "clone_metadata" in metadata_handler.__all__
        assert "clone_ai_metadata" in metadata_handler.__all__
        assert "extract_metadata" in metadata_handler.__all__
        assert "inject_metadata" in metadata_handler.__all__
        assert "has_c2pa_metadata" in metadata_handler.__all__
        assert "remove_ai_metadata" in metadata_handler.__all__
        assert "has_ai_content" in metadata_handler.__all__

    def test_functions_are_callable(self) -> None:
        assert callable(metadata_handler.clone_metadata)
        assert callable(metadata_handler.clone_ai_metadata)
        assert callable(metadata_handler.extract_metadata)
        assert callable(metadata_handler.inject_metadata)
        assert callable(metadata_handler.has_ai_metadata)
        assert callable(metadata_handler.has_c2pa_metadata)
        assert callable(metadata_handler.remove_ai_metadata)
        assert callable(metadata_handler.has_ai_content)
