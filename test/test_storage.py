"""Unit tests for artifact storage utilities."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from nubison_model.Storage import (
    DEFAULT_SIZE_THRESHOLD,
    DVCError,
    DVCPullError,
    DVCPushError,
    deserialize_dvc_info,
    find_weight_files,
    get_dvc_cache_key,
    get_size_threshold,
    is_dvc_enabled,
    is_weight_file,
    parse_dvc_file,
    pull_from_dvc,
    push_to_dvc,
    serialize_dvc_info,
    should_use_dvc,
)
from test.utils import temporary_env


class TestIsDvcEnabled:
    """Tests for is_dvc_enabled function."""

    def test_enabled_with_true(self):
        with temporary_env({"DVC_ENABLED": "true"}):
            assert is_dvc_enabled() is True

    def test_enabled_with_1(self):
        with temporary_env({"DVC_ENABLED": "1"}):
            assert is_dvc_enabled() is True

    def test_disabled_with_false(self):
        with temporary_env({"DVC_ENABLED": "false"}):
            assert is_dvc_enabled() is False

    def test_disabled_when_not_set(self):
        env = os.environ.copy()
        env.pop("DVC_ENABLED", None)
        with patch.dict(os.environ, env, clear=True):
            assert is_dvc_enabled() is False


class TestGetSizeThreshold:
    """Tests for get_size_threshold function."""

    def test_default_threshold(self):
        with temporary_env({"DVC_SIZE_THRESHOLD": ""}):
            assert get_size_threshold() == DEFAULT_SIZE_THRESHOLD

    def test_custom_threshold(self):
        with temporary_env({"DVC_SIZE_THRESHOLD": "52428800"}):  # 50MB
            assert get_size_threshold() == 52428800

    def test_invalid_threshold_returns_default(self):
        with temporary_env({"DVC_SIZE_THRESHOLD": "invalid"}):
            assert get_size_threshold() == DEFAULT_SIZE_THRESHOLD


class TestIsWeightFile:
    """Tests for is_weight_file function."""

    @pytest.mark.parametrize(
        "filepath,expected",
        [
            ("model.pt", True),
            ("model.bin", True),
            ("model.safetensors", True),
            ("model.txt", False),
            ("model.py", False),
            ("MODEL.PT", True),  # case insensitive
            ("/path/to/model.pt", True),  # with path
        ],
    )
    def test_weight_file_detection(self, filepath, expected):
        assert is_weight_file(filepath) == expected


class TestShouldUseDvc:
    """Tests for should_use_dvc function."""

    def test_small_weight_file_returns_false(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"x" * 1024)  # 1KB
            f.flush()
            try:
                assert should_use_dvc(f.name, size_threshold=1024 * 1024) is False
            finally:
                os.unlink(f.name)

    def test_large_weight_file_returns_true(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            f.write(b"x" * 2 * 1024 * 1024)  # 2MB
            f.flush()
            try:
                assert should_use_dvc(f.name, size_threshold=1024 * 1024) is True
            finally:
                os.unlink(f.name)

    def test_nonexistent_file_returns_false(self):
        assert should_use_dvc("/nonexistent/path/model.pt") is False


class TestFindWeightFiles:
    """Tests for find_weight_files function."""

    def test_find_large_files_only(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            large_pt = os.path.join(tmpdir, "model.pt")
            small_pt = os.path.join(tmpdir, "small.pt")

            with open(large_pt, "wb") as f:
                f.write(b"x" * 2 * 1024 * 1024)  # 2MB

            with open(small_pt, "wb") as f:
                f.write(b"x" * 1024)  # 1KB

            result = find_weight_files(tmpdir, size_threshold=1024 * 1024)

            assert large_pt in result
            assert small_pt not in result

    def test_empty_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = find_weight_files(tmpdir)
            assert result == []


class TestParseDvcFile:
    """Tests for parse_dvc_file function."""

    def test_parse_valid_dvc_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".dvc", delete=False) as f:
            f.write("outs:\n  - md5: a304afb96060aad90176268345e10355\n    path: model.pt\n")
            f.flush()
            try:
                result = parse_dvc_file(f.name)
                assert result == "a304afb96060aad90176268345e10355"
            finally:
                os.unlink(f.name)

    def test_parse_nonexistent_file(self):
        result = parse_dvc_file("/nonexistent/file.dvc")
        assert result is None


class TestSerializeDeserializeDvcInfo:
    """Tests for serialize/deserialize DVC info functions."""

    def test_round_trip(self):
        original = {"models/model.pt": "abc123", "weights/layer.bin": "def456"}
        serialized = serialize_dvc_info(original)
        deserialized = deserialize_dvc_info(serialized)
        assert deserialized == original


class TestGetDvcCacheKey:
    """Tests for get_dvc_cache_key function."""

    def test_same_inputs_same_key(self):
        uri = "models:/MyModel/1"
        info = {"model.pt": "abc123"}
        key1 = get_dvc_cache_key(uri, info)
        key2 = get_dvc_cache_key(uri, info)
        assert key1 == key2

    def test_different_uri_different_key(self):
        info = {"model.pt": "abc123"}
        key1 = get_dvc_cache_key("models:/MyModel/1", info)
        key2 = get_dvc_cache_key("models:/MyModel/2", info)
        assert key1 != key2


class TestDvcExceptions:
    """Tests for DVC exception classes."""

    def test_dvc_error_hierarchy(self):
        assert issubclass(DVCPushError, DVCError)
        assert issubclass(DVCPullError, DVCError)


class TestPushToDvc:
    """Tests for push_to_dvc function with mocked subprocess."""

    def test_push_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "model.pt")
            dvc_file = test_file + ".dvc"

            with open(test_file, "wb") as f:
                f.write(b"model content")

            # Create mock .dvc file
            with open(dvc_file, "w") as f:
                f.write("outs:\n  - md5: abc123\n    path: model.pt\n")

            with patch("nubison_model.Storage.ensure_dvc_ready"):
                with patch("nubison_model.Storage.Repo") as mock_repo_cls:
                    mock_repo = MagicMock()
                    mock_repo_cls.return_value = mock_repo

                    result = push_to_dvc([test_file])

                    assert test_file in result
                    assert result[test_file] == "abc123"
                    mock_repo.add.assert_called_once_with([test_file])
                    mock_repo.push.assert_called_once()

    def test_push_dvc_add_failure(self):
        with patch("nubison_model.Storage.ensure_dvc_ready"):
            with patch("nubison_model.Storage.Repo") as mock_repo_cls:
                mock_repo = MagicMock()
                mock_repo_cls.return_value = mock_repo
                mock_repo.add.side_effect = Exception("dvc add error")

                with pytest.raises(DVCPushError, match="dvc add failed"):
                    push_to_dvc(["/path/to/model.pt"])

    def test_push_dvc_push_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "model.pt")
            dvc_file = test_file + ".dvc"

            with open(test_file, "wb") as f:
                f.write(b"model content")

            with open(dvc_file, "w") as f:
                f.write("outs:\n  - md5: abc123\n    path: model.pt\n")

            with patch("nubison_model.Storage.ensure_dvc_ready"):
                with patch("nubison_model.Storage.Repo") as mock_repo_cls:
                    mock_repo = MagicMock()
                    mock_repo_cls.return_value = mock_repo
                    mock_repo.push.side_effect = Exception("push error")

                    with pytest.raises(DVCPushError, match="dvc push failed"):
                        push_to_dvc([test_file])


class TestPullFromDvc:
    """Tests for pull_from_dvc function using dvc pull command."""

    def test_pull_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dvc_files = {"model.pt": "abc123def456"}

            with temporary_env({"DVC_REMOTE_URL": "s3://bucket/dvc"}):
                with patch("nubison_model.Storage.ensure_dvc_ready"):
                    with patch("nubison_model.Storage.subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(returncode=0, stderr="")

                        pull_from_dvc(dvc_files, tmpdir)

                        # Verify dvc pull was called
                        mock_run.assert_called_once()
                        call_args = mock_run.call_args[0][0]
                        assert call_args[0:2] == ["dvc", "pull"]
                        assert "-j" in call_args

    def test_pull_dvc_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dvc_files = {"model.pt": "abc123"}

            with temporary_env({"DVC_REMOTE_URL": "s3://bucket/dvc"}):
                with patch("nubison_model.Storage.ensure_dvc_ready"):
                    with patch("nubison_model.Storage.subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(
                            returncode=1, stderr="pull failed"
                        )

                        with pytest.raises(DVCPullError, match="dvc pull failed"):
                            pull_from_dvc(dvc_files, tmpdir)

    def test_pull_missing_remote_url(self):
        with temporary_env({"DVC_REMOTE_URL": ""}):
            with pytest.raises(DVCPullError, match="DVC_REMOTE_URL"):
                pull_from_dvc({"model.pt": "hash"}, ".")
