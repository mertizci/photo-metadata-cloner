"""Tests for the progress animation and library silencing module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


class TestRunWithProgress:
    """Test run_with_progress animation wrapper."""

    def test_returns_task_result(self) -> None:
        """Successful task result is returned."""
        from progress import run_with_progress

        result = run_with_progress(lambda: 42)
        assert result == 42

    def test_propagates_task_exception(self) -> None:
        """Exceptions from the task are re-raised."""
        from progress import run_with_progress

        def failing_task() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            run_with_progress(failing_task)

    def test_reads_progress_state(self) -> None:
        """Animation reads current message from progress_state dict."""
        from progress import run_with_progress

        state: dict[str, str] = {"message": "initial"}

        def task() -> str:
            state["message"] = "updated"
            return "done"

        result = run_with_progress(task, progress_state=state)
        assert result == "done"
        assert state["message"] == "updated"

    def test_handles_none_progress_state(self) -> None:
        """Works when progress_state is None."""
        from progress import run_with_progress

        assert run_with_progress(lambda: "ok", progress_state=None) == "ok"


class TestSilenceLibraryOutput:
    """Test silence_library_output wrapper."""

    def test_returns_callable(self) -> None:
        """Wrapper returns a zero-arg callable."""
        from progress import silence_library_output

        wrapped = silence_library_output(lambda: 123)
        assert callable(wrapped)

    def test_executes_inner_function(self) -> None:
        """Inner function is actually called and result returned."""
        from progress import silence_library_output

        wrapped = silence_library_output(lambda: "hello")
        assert wrapped() == "hello"

    def test_calls_set_progress(self) -> None:
        """set_progress callback is invoked during execution."""
        from progress import silence_library_output

        messages: list[str] = []
        wrapped = silence_library_output(lambda: None, set_progress=messages.append)
        wrapped()
        assert any("Configuring" in m for m in messages)

    def test_sets_hf_env_var(self) -> None:
        """HF_HUB_DISABLE_PROGRESS_BARS env var is set."""
        from progress import silence_library_output

        env_before = os.environ.pop("HF_HUB_DISABLE_PROGRESS_BARS", None)
        try:
            wrapped = silence_library_output(lambda: None)
            wrapped()
            assert os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS") == "1"
        finally:
            if env_before is not None:
                os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = env_before


class TestBuildBar:
    """Test the bouncing bar builder."""

    def test_bar_has_correct_width(self) -> None:
        """Bar always has 32 visible characters (ignoring ANSI codes)."""
        from progress import _build_bar, _BAR_WIDTH

        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            for step in range(100):
                bar = _build_bar(step)
                assert len(bar) == _BAR_WIDTH

    def test_no_color_mode(self) -> None:
        """NO_COLOR env var produces plain-text bar."""
        from progress import _build_bar

        with patch.dict(os.environ, {"NO_COLOR": "1"}):
            bar = _build_bar(0)
            assert "\033[" not in bar


class TestTruncate:
    """Test the string truncation helper."""

    def test_short_string_unchanged(self) -> None:
        from progress import _truncate

        assert _truncate("hello", max_len=10) == "hello"

    def test_long_string_truncated(self) -> None:
        from progress import _truncate

        result = _truncate("a" * 100, max_len=10)
        assert len(result) == 10
        assert result.endswith("…")
