"""Tests for watermark_profiles module."""

from __future__ import annotations

import pytest


class TestGetModelIdForProfile:
    """Test profile → model ID mapping."""

    def test_default(self) -> None:
        from watermark_profiles import get_model_id_for_profile

        assert get_model_id_for_profile("default") == "Lykon/dreamshaper-8"

    def test_ctrlregen(self) -> None:
        from watermark_profiles import get_model_id_for_profile

        assert get_model_id_for_profile("ctrlregen") == "yepengliu/ctrlregen"

    def test_unknown_raises(self) -> None:
        from watermark_profiles import get_model_id_for_profile

        with pytest.raises(ValueError, match="Unknown model profile"):
            get_model_id_for_profile("nonexistent")

    def test_case_insensitive(self) -> None:
        from watermark_profiles import get_model_id_for_profile

        assert get_model_id_for_profile("Default") == "Lykon/dreamshaper-8"
        assert get_model_id_for_profile("CTRLREGEN") == "yepengliu/ctrlregen"

    def test_whitespace_stripped(self) -> None:
        from watermark_profiles import get_model_id_for_profile

        assert get_model_id_for_profile("  default  ") == "Lykon/dreamshaper-8"


class TestDetectModelProfile:
    """Test model ID → profile detection."""

    def test_ctrlregen_id(self) -> None:
        from watermark_profiles import detect_model_profile

        assert detect_model_profile("yepengliu/ctrlregen") == "ctrlregen"
        assert detect_model_profile("some/CtrlRegen-fork") == "ctrlregen"

    def test_default_id(self) -> None:
        from watermark_profiles import detect_model_profile

        assert detect_model_profile("Lykon/dreamshaper-8") == "default"


class TestGetRecommendedStrength:
    """Test watermark type → strength recommendations."""

    def test_high_perturbation(self) -> None:
        from watermark_profiles import get_recommended_strength

        assert get_recommended_strength("stegasamp") == 0.7
        assert get_recommended_strength("treering") == 0.7
        assert get_recommended_strength("STEgasTAMP") == 0.7

    def test_low_perturbation(self) -> None:
        from watermark_profiles import get_recommended_strength

        assert get_recommended_strength("stablesignature") == 0.04
        assert get_recommended_strength("rivagan") == 0.04
        assert get_recommended_strength("ssl") == 0.04

    def test_default(self) -> None:
        from watermark_profiles import get_recommended_strength

        assert get_recommended_strength("unknown") == 0.35
        assert get_recommended_strength("medium") == 0.35
        assert get_recommended_strength("") == 0.35


class TestConstants:
    """Test exported constants."""

    def test_model_ids(self) -> None:
        from watermark_profiles import DEFAULT_MODEL_ID, CTRLREGEN_MODEL_ID

        assert DEFAULT_MODEL_ID == "Lykon/dreamshaper-8"
        assert CTRLREGEN_MODEL_ID == "yepengliu/ctrlregen"

    def test_strength_values(self) -> None:
        from watermark_profiles import LOW_STRENGTH, MEDIUM_STRENGTH, HIGH_STRENGTH

        assert 0.0 <= LOW_STRENGTH <= 1.0
        assert 0.0 <= MEDIUM_STRENGTH <= 1.0
        assert 0.0 <= HIGH_STRENGTH <= 1.0
        assert LOW_STRENGTH < MEDIUM_STRENGTH < HIGH_STRENGTH
