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

            # Create .dvc file (simulating MLflow artifact download)
            dvc_file_path = os.path.join(tmpdir, "model.pt.dvc")
            with open(dvc_file_path, "w") as f:
                f.write("outs:\n  - md5: abc123def456\n    path: model.pt\n")

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

            # Create .dvc file
            dvc_file_path = os.path.join(tmpdir, "model.pt.dvc")
            with open(dvc_file_path, "w") as f:
                f.write("outs:\n  - md5: abc123\n    path: model.pt\n")

            with temporary_env({"DVC_REMOTE_URL": "s3://bucket/dvc"}):
                with patch("nubison_model.Storage.ensure_dvc_ready"):
                    with patch("nubison_model.Storage.subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(
                            returncode=1, stderr="pull failed"
                        )

                        with pytest.raises(DVCPullError, match="dvc pull failed"):
                            pull_from_dvc(dvc_files, tmpdir)

    def test_pull_missing_dvc_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dvc_files = {"model.pt": "abc123"}

            with temporary_env({"DVC_REMOTE_URL": "s3://bucket/dvc"}):
                with patch("nubison_model.Storage.ensure_dvc_ready"):
                    with pytest.raises(DVCPullError, match="DVC file not found"):
                        pull_from_dvc(dvc_files, tmpdir)

    def test_pull_missing_remote_url(self):
        with temporary_env({"DVC_REMOTE_URL": ""}):
            with pytest.raises(DVCPullError, match="DVC_REMOTE_URL"):
                pull_from_dvc({"model.pt": "hash"}, ".")

    def test_pull_with_subdirectory_path(self):
        """Test pull_from_dvc with subdirectory paths like 'src/model.pt'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory structure: tmpdir/src/model.pt.dvc
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)

            dvc_files = {"src/model.pt": "abc123def456"}

            # Create .dvc file in subdirectory
            dvc_file_path = os.path.join(src_dir, "model.pt.dvc")
            with open(dvc_file_path, "w") as f:
                f.write("outs:\n  - md5: abc123def456\n    path: model.pt\n")

            with temporary_env({"DVC_REMOTE_URL": "s3://bucket/dvc"}):
                with patch("nubison_model.Storage.ensure_dvc_ready"):
                    with patch("nubison_model.Storage.subprocess.run") as mock_run:
                        mock_run.return_value = MagicMock(returncode=0, stderr="")

                        pull_from_dvc(dvc_files, tmpdir)

                        # Verify dvc pull was called with correct path
                        mock_run.assert_called_once()
                        call_args = mock_run.call_args[0][0]
                        assert dvc_file_path in call_args


class TestCustomWeightExtensions:
    """Tests for custom weight file extensions via DVC_FILE_EXTENSIONS."""

    def test_default_extensions(self):
        from nubison_model.Storage import get_weight_extensions, WEIGHT_FILE_EXTENSIONS

        with temporary_env({"DVC_FILE_EXTENSIONS": ""}):
            extensions = get_weight_extensions()
            assert extensions == WEIGHT_FILE_EXTENSIONS

    def test_custom_extensions_added(self):
        from nubison_model.Storage import get_weight_extensions

        with temporary_env({"DVC_FILE_EXTENSIONS": ".mymodel,.custom"}):
            extensions = get_weight_extensions()
            assert ".mymodel" in extensions
            assert ".custom" in extensions
            # Default extensions should still be present
            assert ".pt" in extensions

    def test_custom_extension_without_dot(self):
        from nubison_model.Storage import get_weight_extensions

        with temporary_env({"DVC_FILE_EXTENSIONS": "npy,npz"}):
            extensions = get_weight_extensions()
            assert ".npy" in extensions
            assert ".npz" in extensions

    def test_is_weight_file_with_custom_extension(self):
        with temporary_env({"DVC_FILE_EXTENSIONS": ".mymodel"}):
            assert is_weight_file("model.mymodel") is True
            assert is_weight_file("model.unknown") is False


class TestPathTraversalPrevention:
    """Tests for path traversal attack prevention."""

    def test_validate_safe_path_normal(self):
        from nubison_model.Storage import validate_safe_path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file
            test_file = os.path.join(tmpdir, "model.pt")
            with open(test_file, "w") as f:
                f.write("test")

            # Normal path should work
            result = validate_safe_path(tmpdir, "model.pt")
            assert result == test_file

    def test_validate_safe_path_subdirectory(self):
        from nubison_model.Storage import validate_safe_path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create subdirectory
            subdir = os.path.join(tmpdir, "src")
            os.makedirs(subdir)
            test_file = os.path.join(subdir, "model.pt")
            with open(test_file, "w") as f:
                f.write("test")

            # Subdirectory path should work
            result = validate_safe_path(tmpdir, "src/model.pt")
            assert result == test_file

    def test_validate_safe_path_traversal_blocked(self):
        from nubison_model.Storage import validate_safe_path, PathTraversalError

        with tempfile.TemporaryDirectory() as tmpdir:
            # Path traversal should be blocked
            with pytest.raises(PathTraversalError):
                validate_safe_path(tmpdir, "../etc/passwd")

    def test_validate_safe_path_absolute_blocked(self):
        from nubison_model.Storage import validate_safe_path, PathTraversalError

        with tempfile.TemporaryDirectory() as tmpdir:
            # Absolute path outside base should be blocked
            with pytest.raises(PathTraversalError):
                validate_safe_path(tmpdir, "/etc/passwd")


class TestIterArtifactFiles:
    """Tests for iter_artifact_files function."""

    def test_iterate_single_directory(self):
        from nubison_model.Storage import iter_artifact_files

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files
            file1 = os.path.join(tmpdir, "model.pt")
            file2 = os.path.join(tmpdir, "config.json")
            with open(file1, "w") as f:
                f.write("model")
            with open(file2, "w") as f:
                f.write("config")

            results = list(iter_artifact_files(tmpdir))
            paths = [r[0] for r in results]

            assert file1 in paths
            assert file2 in paths

    def test_iterate_nested_directory(self):
        from nubison_model.Storage import iter_artifact_files

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            subdir = os.path.join(tmpdir, "src", "models")
            os.makedirs(subdir)
            nested_file = os.path.join(subdir, "model.pt")
            with open(nested_file, "w") as f:
                f.write("model")

            results = list(iter_artifact_files(tmpdir))
            paths = [r[0] for r in results]

            assert nested_file in paths

    def test_iterate_multiple_directories(self):
        from nubison_model.Storage import iter_artifact_files

        with tempfile.TemporaryDirectory() as tmpdir:
            dir1 = os.path.join(tmpdir, "dir1")
            dir2 = os.path.join(tmpdir, "dir2")
            os.makedirs(dir1)
            os.makedirs(dir2)

            file1 = os.path.join(dir1, "a.pt")
            file2 = os.path.join(dir2, "b.pt")
            with open(file1, "w") as f:
                f.write("a")
            with open(file2, "w") as f:
                f.write("b")

            # Comma-separated directories
            results = list(iter_artifact_files(f"{dir1},{dir2}"))
            paths = [r[0] for r in results]

            assert file1 in paths
            assert file2 in paths


class TestServiceDvcFunctions:
    """Tests for DVC-related functions in Service.py."""

    def test_get_dvc_info_from_model_uri_with_tags(self):
        """Test extracting DVC info from MLflow model version tags."""
        from nubison_model.Service import _get_dvc_info_from_model_uri

        with patch("nubison_model.Service.set_tracking_uri"):
            with patch("nubison_model.Service.mlflow.tracking.MlflowClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client

                mock_version = MagicMock()
                mock_version.tags = {"dvc_files": '{"model.pt": "abc123"}'}
                mock_client.get_model_version.return_value = mock_version

                result = _get_dvc_info_from_model_uri("http://test", "models:/MyModel/1")
                assert result == {"model.pt": "abc123"}

    def test_get_dvc_info_from_model_uri_without_tags(self):
        """Test when model version has no DVC tags."""
        from nubison_model.Service import _get_dvc_info_from_model_uri

        with patch("nubison_model.Service.set_tracking_uri"):
            with patch("nubison_model.Service.mlflow.tracking.MlflowClient") as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client

                mock_version = MagicMock()
                mock_version.tags = {}
                mock_client.get_model_version.return_value = mock_version

                result = _get_dvc_info_from_model_uri("http://test", "models:/MyModel/2")
                assert result == {}

    def test_restore_dvc_files_calls_pull(self):
        """Test that _restore_dvc_files calls pull_from_dvc correctly."""
        from nubison_model.Service import _restore_dvc_files

        with tempfile.TemporaryDirectory() as model_root:
            with patch("nubison_model.Service.pull_from_dvc") as mock_pull:
                dvc_info = {"src/model.pt": "abc123"}
                _restore_dvc_files(dvc_info, model_root)

                mock_pull.assert_called_once_with(dvc_info, ".", show_progress=True)

    def test_restore_dvc_files_empty_info(self):
        """Test that _restore_dvc_files does nothing with empty dvc_info."""
        from nubison_model.Service import _restore_dvc_files

        with tempfile.TemporaryDirectory() as model_root:
            with patch("nubison_model.Service.pull_from_dvc") as mock_pull:
                _restore_dvc_files({}, model_root)
                mock_pull.assert_not_called()

    def test_restore_dvc_files_failure_raises(self):
        """Test that _restore_dvc_files raises DVCPullError on failure."""
        from nubison_model.Service import _restore_dvc_files

        with tempfile.TemporaryDirectory() as model_root:
            with patch("nubison_model.Service.pull_from_dvc") as mock_pull:
                mock_pull.side_effect = DVCPullError("Download failed")

                with pytest.raises(DVCPullError):
                    _restore_dvc_files({"model.pt": "abc123"}, model_root)

    def test_create_dvc_symlinks(self):
        """Test that _create_dvc_symlinks creates correct symlinks."""
        from nubison_model.Service import _create_dvc_symlinks

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source file
            source_file = os.path.join(tmpdir, "model.pt")
            with open(source_file, "w") as f:
                f.write("model content")

            # Change to a temp directory for symlink creation
            with tempfile.TemporaryDirectory() as workdir:
                original_cwd = os.getcwd()
                try:
                    os.chdir(workdir)
                    dvc_info = {"model.pt": "abc123"}
                    _create_dvc_symlinks(dvc_info, tmpdir)

                    # Check symlink was created
                    symlink_path = os.path.join(workdir, "model.pt")
                    assert os.path.islink(symlink_path)
                    assert os.readlink(symlink_path) == source_file
                finally:
                    os.chdir(original_cwd)

    def test_create_dvc_symlinks_with_subdirectory(self):
        """Test symlink creation with subdirectory paths."""
        from nubison_model.Service import _create_dvc_symlinks

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create source file in subdirectory
            src_dir = os.path.join(tmpdir, "src")
            os.makedirs(src_dir)
            source_file = os.path.join(src_dir, "model.pt")
            with open(source_file, "w") as f:
                f.write("model content")

            with tempfile.TemporaryDirectory() as workdir:
                original_cwd = os.getcwd()
                try:
                    os.chdir(workdir)
                    dvc_info = {"src/model.pt": "abc123"}
                    _create_dvc_symlinks(dvc_info, tmpdir)

                    # Check symlink was created with subdirectory
                    symlink_path = os.path.join(workdir, "src", "model.pt")
                    assert os.path.islink(symlink_path)
                finally:
                    os.chdir(original_cwd)


class TestDvcVersionCache:
    """Tests for DVC version-based caching."""

    def test_different_model_versions_different_cache_keys(self):
        """Test that different model versions produce different cache keys."""
        dvc_info = {"model.pt": "abc123"}
        key1 = get_dvc_cache_key("models:/Model/1", dvc_info)
        key2 = get_dvc_cache_key("models:/Model/2", dvc_info)
        assert key1 != key2

    def test_different_dvc_info_different_cache_keys(self):
        """Test that different dvc_info produces different cache keys."""
        uri = "models:/Model/1"
        key1 = get_dvc_cache_key(uri, {"model.pt": "abc123"})
        key2 = get_dvc_cache_key(uri, {"model.pt": "def456"})
        assert key1 != key2

    def test_same_inputs_produce_same_key(self):
        """Test cache key consistency."""
        uri = "models:/Model/1"
        dvc_info = {"model.pt": "abc123", "weights.bin": "def456"}

        key1 = get_dvc_cache_key(uri, dvc_info)
        key2 = get_dvc_cache_key(uri, dvc_info)
        assert key1 == key2
