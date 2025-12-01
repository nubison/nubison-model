"""Unit tests for artifact storage utilities."""

import hashlib
import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from nubison_model.Storage import (
    DEFAULT_SIZE_THRESHOLD,
    ChecksumMismatchError,
    DVCError,
    DVCPullError,
    DVCPushError,
    calculate_file_md5,
    deserialize_dvc_info,
    find_weight_files,
    get_dvc_cache_key,
    get_dvc_remote_path,
    get_size_threshold,
    is_dvc_enabled,
    is_weight_file,
    parse_dvc_file,
    pull_from_dvc,
    push_to_dvc,
    serialize_dvc_info,
    should_use_dvc,
    verify_file_checksum,
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


class TestChecksumFunctions:
    """Tests for checksum related functions."""

    def test_calculate_and_verify_md5(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            content = b"test content for md5"
            f.write(content)
            f.flush()
            try:
                result = calculate_file_md5(f.name)
                expected = hashlib.md5(content).hexdigest()
                assert result == expected
                assert verify_file_checksum(f.name, expected) is True
                assert verify_file_checksum(f.name, "wrong_hash") is False
            finally:
                os.unlink(f.name)


class TestGetDvcRemotePath:
    """Tests for get_dvc_remote_path function."""

    def test_s3_path(self):
        result = get_dvc_remote_path(
            "s3://bucket/dvc-storage", "a304afb96060aad90176268345e10355"
        )
        assert result == "s3://bucket/dvc-storage/files/md5/a3/04afb96060aad90176268345e10355"


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
        assert issubclass(ChecksumMismatchError, DVCError)


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

            with patch("nubison_model.Storage.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stderr="")

                result = push_to_dvc([test_file])

                assert test_file in result
                assert result[test_file] == "abc123"
                assert mock_run.call_count == 2  # dvc add + dvc push

    def test_push_dvc_add_failure(self):
        with patch("nubison_model.Storage.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="dvc add error")

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

            with patch("nubison_model.Storage.subprocess.run") as mock_run:
                # dvc add succeeds, dvc push fails
                mock_run.side_effect = [
                    MagicMock(returncode=0, stderr=""),  # dvc add
                    MagicMock(returncode=1, stderr="push error"),  # dvc push
                ]

                with pytest.raises(DVCPushError, match="dvc push failed"):
                    push_to_dvc([test_file])


class TestPullFromDvc:
    """Tests for pull_from_dvc function with mocked downloads."""

    def test_pull_success(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dvc_files = {"models/model.pt": "abc123def456"}

            with temporary_env({"DVC_REMOTE_URL": "s3://bucket/dvc"}):
                with patch("nubison_model.Storage._download_from_remote") as mock_download:
                    # Mock successful download
                    def create_file(remote, local, show_progress=True):
                        os.makedirs(os.path.dirname(local), exist_ok=True)
                        with open(local, "wb") as f:
                            f.write(b"model content")

                    mock_download.side_effect = create_file

                    with patch("nubison_model.Storage.verify_file_checksum", return_value=True):
                        pull_from_dvc(dvc_files, tmpdir)

                    # Verify file was "downloaded"
                    assert os.path.exists(os.path.join(tmpdir, "models/model.pt"))

    def test_pull_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dvc_files = {"model.pt": "expected_hash"}

            with temporary_env({"DVC_REMOTE_URL": "s3://bucket/dvc"}):
                with patch("nubison_model.Storage._download_from_remote") as mock_download:
                    def create_file(remote, local, show_progress=True):
                        with open(local, "wb") as f:
                            f.write(b"corrupted content")

                    mock_download.side_effect = create_file

                    with patch("nubison_model.Storage.verify_file_checksum", return_value=False):
                        with pytest.raises(ChecksumMismatchError):
                            pull_from_dvc(dvc_files, tmpdir)

    def test_pull_missing_remote_url(self):
        with temporary_env({"DVC_REMOTE_URL": ""}):
            with pytest.raises(ValueError, match="DVC_REMOTE_URL"):
                pull_from_dvc({"model.pt": "hash"}, ".")


class TestDownloadFromS3:
    """Tests for _download_from_s3 with mocked boto3."""

    def test_download_with_boto3(self):
        import sys
        from nubison_model.Storage import _download_from_s3

        # Create mock boto3 module
        mock_boto3 = MagicMock()
        mock_s3 = MagicMock()
        mock_boto3.client.return_value = mock_s3
        mock_s3.head_object.return_value = {"ContentLength": 1024}

        with patch.dict(sys.modules, {"boto3": mock_boto3, "botocore": MagicMock(), "botocore.exceptions": MagicMock()}):
            with tempfile.NamedTemporaryFile(delete=False) as f:
                try:
                    _download_from_s3("s3://bucket/key/file.pt", f.name, show_progress=True)
                    mock_s3.download_file.assert_called_once_with("bucket", "key/file.pt", f.name)
                finally:
                    os.unlink(f.name)

    def test_download_fallback_to_aws_cli(self):
        import sys
        from nubison_model.Storage import _download_from_s3

        # Remove boto3 from modules to trigger ImportError
        with patch.dict(sys.modules, {"boto3": None}):
            with patch("nubison_model.Storage.subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0)

                with tempfile.NamedTemporaryFile(delete=False) as f:
                    try:
                        _download_from_s3("s3://bucket/file.pt", f.name)
                        mock_run.assert_called()
                    finally:
                        os.unlink(f.name)
