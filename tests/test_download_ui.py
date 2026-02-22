"""Tests for download_ui module."""

from __future__ import annotations

from io import StringIO
from unittest.mock import patch, MagicMock

import pytest


class TestGetModelsToDownload:
    """Test model download detection."""

    @patch("download_ui._fetch_repo_size_gb", return_value=2.0)
    def test_default_profile_uncached(self, _mock_fetch: MagicMock) -> None:
        from download_ui import get_models_to_download

        with patch("huggingface_hub.scan_cache_dir") as mock_scan:
            mock_scan.return_value = MagicMock(repos=[])
            result = get_models_to_download("Lykon/dreamshaper-8", "default")

        assert len(result) == 1
        assert result[0][0] == "Lykon/dreamshaper-8"
        assert result[0][1] == 2.0

    @patch("download_ui._fetch_repo_size_gb", return_value=2.0)
    def test_default_profile_cached(self, _mock_fetch: MagicMock) -> None:
        from download_ui import get_models_to_download

        mock_repo = MagicMock()
        mock_repo.repo_id = "Lykon/dreamshaper-8"

        with patch("huggingface_hub.scan_cache_dir") as mock_scan:
            mock_scan.return_value = MagicMock(repos=[mock_repo])
            result = get_models_to_download("Lykon/dreamshaper-8", "default")

        assert result == []

    @patch("download_ui._fetch_repo_size_gb", return_value=1.5)
    def test_ctrlregen_profile(self, _mock_fetch: MagicMock) -> None:
        from download_ui import get_models_to_download

        with patch("huggingface_hub.scan_cache_dir") as mock_scan:
            mock_scan.return_value = MagicMock(repos=[])
            result = get_models_to_download("yepengliu/ctrlregen", "ctrlregen")

        assert len(result) == 4

    @patch("download_ui._fetch_repo_size_gb", return_value=None)
    def test_scan_cache_failure_returns_all(self, _mock_fetch: MagicMock) -> None:
        from download_ui import get_models_to_download

        with patch("huggingface_hub.scan_cache_dir", side_effect=Exception("fail")):
            result = get_models_to_download("Lykon/dreamshaper-8", "default")

        assert len(result) == 1
        assert result[0][1] is None


class TestFetchRepoSizeGb:
    """Test HuggingFace Hub size lookup."""

    def test_returns_size_from_siblings(self) -> None:
        from download_ui import _fetch_repo_size_gb

        weight = MagicMock()
        weight.rfilename = "model.safetensors"
        weight.size = 2_147_483_648  # 2 GB
        config = MagicMock()
        config.rfilename = "config.json"
        config.size = 1024
        info = MagicMock(siblings=[weight, config])

        with patch("huggingface_hub.model_info", return_value=info):
            result = _fetch_repo_size_gb("some/model")

        assert result == 2.0

    def test_returns_none_on_api_error(self) -> None:
        from download_ui import _fetch_repo_size_gb

        with patch("huggingface_hub.model_info", side_effect=Exception("network")):
            result = _fetch_repo_size_gb("some/model")

        assert result is None

    def test_returns_none_for_empty_siblings(self) -> None:
        from download_ui import _fetch_repo_size_gb

        info = MagicMock(siblings=[])
        with patch("huggingface_hub.model_info", return_value=info):
            result = _fetch_repo_size_gb("some/model")

        assert result is None


class TestDownloadProgressFilter:
    """Test styled progress bar filter."""

    def test_suppresses_fetching_lines(self) -> None:
        from download_ui import DownloadProgressFilter

        stream = StringIO()
        filt = DownloadProgressFilter(stream)
        result = filt.write("Fetching 13 files: 31%")
        assert result == len("Fetching 13 files: 31%")
        assert stream.getvalue() == ""

    def test_suppresses_small_files(self) -> None:
        from download_ui import DownloadProgressFilter

        stream = StringIO()
        filt = DownloadProgressFilter(stream)
        result = filt.write("model_index.json: 100%|...| 582/582 [00:00<00:00, 546kB/s]")
        assert stream.getvalue() == ""

    def test_renders_large_file_progress(self) -> None:
        from download_ui import DownloadProgressFilter

        stream = StringIO()
        with patch.dict("os.environ", {"NO_COLOR": "1"}):
            filt = DownloadProgressFilter(stream)
        filt.write("50%|...| 2.13G/4.27G [00:25<00:25, 82.8MB/s]")
        assert "50%" in stream.getvalue()

    def test_renders_complete(self) -> None:
        from download_ui import DownloadProgressFilter

        stream = StringIO()
        with patch.dict("os.environ", {"NO_COLOR": "1"}):
            filt = DownloadProgressFilter(stream)
        filt._last_tot = "4.27G"
        filt.render_complete()
        assert "Complete" in stream.getvalue()

    def test_render_complete_noop_if_already_done(self) -> None:
        from download_ui import DownloadProgressFilter

        stream = StringIO()
        filt = DownloadProgressFilter(stream)
        filt.rendered_complete = True
        filt.render_complete()
        assert stream.getvalue() == ""


class TestPromptForDownload:
    """Test download confirmation prompt."""

    def test_empty_pending_returns_true(self) -> None:
        from download_ui import prompt_for_download

        assert prompt_for_download([]) is True

    def test_skip_prompt_returns_true(self) -> None:
        from download_ui import prompt_for_download

        assert prompt_for_download(
            [("Lykon/dreamshaper-8", 2.0)], skip_prompt=True
        ) is True

    def test_user_accepts(self) -> None:
        from download_ui import prompt_for_download

        with patch("builtins.input", return_value="y"):
            assert prompt_for_download([("Lykon/dreamshaper-8", 2.0)]) is True

    def test_user_declines(self) -> None:
        from download_ui import prompt_for_download

        with patch("builtins.input", return_value="n"):
            assert prompt_for_download([("Lykon/dreamshaper-8", 2.0)]) is False

    def test_eof_aborts(self) -> None:
        from download_ui import prompt_for_download

        with patch("builtins.input", side_effect=EOFError):
            assert prompt_for_download([("Lykon/dreamshaper-8", 2.0)]) is False

    def test_none_size_shows_unknown(self, capsys: pytest.CaptureFixture[str]) -> None:
        from download_ui import prompt_for_download

        with patch("builtins.input", return_value="y"):
            prompt_for_download([("some/model", None)])

        captured = capsys.readouterr()
        assert "size unknown" in captured.out
        assert "some/model" in captured.out
