"""Tests for constants module."""

from constants import (
    AI_KEYWORDS,
    AI_METADATA_KEYS,
    C2PA_ACTIONS,
    C2PA_AI_TOOLS,
    C2PA_CHUNK_TYPE,
    C2PA_ISSUERS,
    C2PA_SIGNATURES,
    PNG_METADATA_KEYS,
    PNG_SIGNATURE,
    SUPPORTED_FORMATS,
)


class TestSupportedFormats:
    """Tests for SUPPORTED_FORMATS constant."""

    def test_contains_png(self) -> None:
        assert ".png" in SUPPORTED_FORMATS

    def test_contains_jpg(self) -> None:
        assert ".jpg" in SUPPORTED_FORMATS

    def test_contains_jpeg(self) -> None:
        assert ".jpeg" in SUPPORTED_FORMATS

    def test_lowercase_only(self) -> None:
        for fmt in SUPPORTED_FORMATS:
            assert fmt == fmt.lower()

    def test_has_three_formats(self) -> None:
        assert len(SUPPORTED_FORMATS) == 3


class TestAIMetadataKeys:
    """Tests for AI_METADATA_KEYS constant."""

    def test_contains_stable_diffusion_parameters(self) -> None:
        assert "parameters" in AI_METADATA_KEYS

    def test_contains_comfyui_workflow(self) -> None:
        assert "workflow" in AI_METADATA_KEYS

    def test_contains_dreamstudio_dream(self) -> None:
        assert "Dream" in AI_METADATA_KEYS

    def test_is_list(self) -> None:
        assert isinstance(AI_METADATA_KEYS, list)


class TestPNGMetadataKeys:
    """Tests for PNG_METADATA_KEYS constant."""

    def test_contains_author(self) -> None:
        assert "Author" in PNG_METADATA_KEYS

    def test_contains_title(self) -> None:
        assert "Title" in PNG_METADATA_KEYS

    def test_contains_description(self) -> None:
        assert "Description" in PNG_METADATA_KEYS

    def test_contains_copyright(self) -> None:
        assert "Copyright" in PNG_METADATA_KEYS


class TestAIKeywords:
    """Tests for AI_KEYWORDS constant."""

    def test_contains_common_ai_terms(self) -> None:
        assert "prompt" in AI_KEYWORDS
        assert "seed" in AI_KEYWORDS
        assert "model" in AI_KEYWORDS

    def test_contains_openai_terms(self) -> None:
        assert "chatgpt" in AI_KEYWORDS
        assert "openai" in AI_KEYWORDS
        assert "sora" in AI_KEYWORDS

    def test_is_list(self) -> None:
        assert isinstance(AI_KEYWORDS, list)


class TestC2PAConstants:
    """Tests for C2PA-related constants."""

    def test_chunk_type_is_bytes(self) -> None:
        assert isinstance(C2PA_CHUNK_TYPE, bytes)
        assert C2PA_CHUNK_TYPE == b"caBX"

    def test_signatures_are_bytes(self) -> None:
        for sig in C2PA_SIGNATURES:
            assert isinstance(sig, bytes)

    def test_signatures_contain_c2pa(self) -> None:
        assert b"c2pa" in C2PA_SIGNATURES

    def test_issuers_dict(self) -> None:
        assert isinstance(C2PA_ISSUERS, dict)
        assert C2PA_ISSUERS[b"Google"] == "Google LLC"
        assert C2PA_ISSUERS[b"OpenAI"] == "OpenAI"

    def test_ai_tools_dict(self) -> None:
        assert isinstance(C2PA_AI_TOOLS, dict)
        assert C2PA_AI_TOOLS[b"GPT-4o"] == "GPT-4o"
        assert C2PA_AI_TOOLS[b"ChatGPT"] == "ChatGPT"

    def test_actions_dict(self) -> None:
        assert isinstance(C2PA_ACTIONS, dict)
        assert C2PA_ACTIONS[b"c2pa.created"] == "created"
        assert C2PA_ACTIONS[b"c2pa.converted"] == "converted"


class TestPNGSignature:
    """Tests for PNG_SIGNATURE constant."""

    def test_is_bytes(self) -> None:
        assert isinstance(PNG_SIGNATURE, bytes)

    def test_correct_value(self) -> None:
        assert PNG_SIGNATURE == b"\x89PNG\r\n\x1a\n"

    def test_length(self) -> None:
        assert len(PNG_SIGNATURE) == 8
