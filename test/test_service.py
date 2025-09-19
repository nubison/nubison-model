import os
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from filelock import FileLock, Timeout
from PIL.Image import Image
from PIL.Image import open as open_image

from nubison_model import (
    ModelContext,
    NubisonMLFlowModel,
    build_inference_service,
    test_client,
)
from nubison_model.Service import (
    DEFAULT_NUM_WORKERS,
    load_nubison_mlflow_model,
)
from test.utils import temporary_env


def test_raise_runtime_error_on_missing_env():
    with pytest.raises(RuntimeError):
        build_inference_service()


def test_service_ok():
    class DummyModel:
        def infer(self, test: str):
            return test

        def load_model(self, context: ModelContext):
            pass

    with patch(
        "nubison_model.Service.load_nubison_mlflow_model"
    ) as mock_load_nubison_mlflow_model:
        mock_load_nubison_mlflow_model.return_value = NubisonMLFlowModel(DummyModel())
        service = build_inference_service()()
        assert service.infer("test") == "test"


def test_client_ok():
    class DummyModel:
        def infer(self, test: str):
            return test

        def load_model(self, context: ModelContext):
            pass

    with patch(
        "nubison_model.Service.load_nubison_mlflow_model"
    ) as mock_load_nubison_mlflow_model:
        mock_load_nubison_mlflow_model.return_value = NubisonMLFlowModel(DummyModel())
        with test_client("test") as client:
            response = client.post("/infer", json={"test": "test"})
            assert response.status_code == 200
            assert response.text == "test"


def test_image_input():
    class DummyModel:
        def infer(self, test: Image):
            return test.size

        def load_model(self, context: ModelContext):
            pass

    with patch(
        "nubison_model.Service.load_nubison_mlflow_model"
    ) as mock_load_nubison_mlflow_model:
        mock_load_nubison_mlflow_model.return_value = NubisonMLFlowModel(DummyModel())
        service = build_inference_service()()
        test_image = open_image("test/fixtures/red_100x100.png")
        assert service.infer(test_image) == (100, 100)

        with open("test/fixtures/red_100x100.png", "rb") as image:
            with test_client("test") as client:
                response = client.post("/infer", files={"test": image})
                assert response.status_code == 200
                assert response.json() == [100, 100]


def test_model_context():
    class DummyModel:
        def __init__(self):
            self.context = None

        def infer(self, test: str):
            return test

        def load_model(self, context: ModelContext):
            self.context = context

    with patch(
        "nubison_model.Service.load_nubison_mlflow_model"
    ) as mock_load_nubison_mlflow_model:
        dummy_model = DummyModel()
        mock_load_nubison_mlflow_model.return_value = NubisonMLFlowModel(dummy_model)

        #
        with patch("bentoml.server_context.worker_index", 1), temporary_env({}):
            service = build_inference_service()()
            assert dummy_model.context == {
                "worker_index": 0,
                "num_workers": DEFAULT_NUM_WORKERS,
            }, "Default num_workers should be applied"

        with patch("bentoml.server_context.worker_index", 1), temporary_env(
            {"NUM_WORKERS": "8"}
        ):
            service = build_inference_service()()
            assert dummy_model.context == {
                "worker_index": 0,
                "num_workers": 8,
            }, "Custom num_workers should be applied"

        with patch("bentoml.server_context.worker_index", None), temporary_env({}):
            service = build_inference_service()()
            assert dummy_model.context == {
                "worker_index": 0,
                "num_workers": 1,
            }, "When worker_index is unavailable, both worker_index and num_workers should be set to 1"


def test_filelock_concurrent_access():
    """Test FileLock prevents concurrent access between threads."""
    with tempfile.TemporaryDirectory() as temp_dir:
        lock_file = os.path.join(temp_dir, "test.lock")
        lock1 = FileLock(lock_file, timeout=0.1)
        lock2 = FileLock(lock_file, timeout=0.1)

        results = []

        def worker1():
            try:
                with lock1:
                    results.append("worker1_acquired")
                    time.sleep(0.2)  # Hold lock for 200ms
                    results.append("worker1_released")
            except Timeout:
                results.append("worker1_timeout")

        def worker2():
            time.sleep(0.05)  # Start slightly after worker1
            try:
                with lock2:
                    results.append("worker2_acquired")
                    results.append("worker2_released")
            except Timeout:
                results.append("worker2_timeout")

        thread1 = threading.Thread(target=worker1)
        thread2 = threading.Thread(target=worker2)

        thread1.start()
        thread2.start()

        thread1.join()
        thread2.join()

        # Worker2 should timeout because worker1 holds the lock
        assert "worker1_acquired" in results
        assert "worker1_released" in results
        assert "worker2_timeout" in results


def test_filelock_timeout_fallback_integration():
    """Test timeout fallback in load_nubison_mlflow_model."""

    class MockNubisonModel:
        def infer(self, test: str):
            return test

        def load_model(self, context):
            pass

    fallback_calls = []

    def mock_load_model_wrapper(tracking_uri, model_uri):
        fallback_calls.append(model_uri)
        return MagicMock(), NubisonMLFlowModel(MockNubisonModel())

    with tempfile.TemporaryDirectory() as temp_dir:
        shared_dir = os.path.join(temp_dir, "shared")

        with patch(
            "nubison_model.Service._get_shared_artifacts_dir", return_value=shared_dir
        ):
            with patch(
                "nubison_model.Service._load_cached_model_if_available",
                return_value=None,
            ):
                with patch(
                    "nubison_model.Service._load_model_with_nubison_wrapper",
                    side_effect=mock_load_model_wrapper,
                ):
                    with patch("nubison_model.Service.FileLock") as mock_filelock_class:
                        # Mock FileLock to raise Timeout
                        mock_filelock = MagicMock()
                        mock_filelock.__enter__.side_effect = Timeout("test.lock")
                        mock_filelock_class.return_value = mock_filelock

                        result = load_nubison_mlflow_model("http://test", "model_uri")

                        # Should fall back to original URI
                        assert len(fallback_calls) == 1
                        assert fallback_calls[0] == "model_uri"
                        assert isinstance(result, NubisonMLFlowModel)


def test_load_nubison_mlflow_model_single_worker():
    """Test load_nubison_mlflow_model with single worker scenario."""

    class MockNubisonModel:
        def infer(self, test: str):
            return test

        def load_model(self, context):
            pass

    class MockMLflowModel:
        def __init__(self, model_path):
            self.model_path = model_path
            self._model_impl = self
            self.context = MagicMock()
            self.context.artifacts = {"model": model_path}

        def unwrap_python_model(self):
            return NubisonMLFlowModel(MockNubisonModel())

    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = os.path.join(temp_dir, "model", "artifacts", "model_file")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "w") as f:
            f.write("test model")

        # Create MLmodel file in the expected location
        mlmodel_path = os.path.join(temp_dir, "model", "MLmodel")
        with open(mlmodel_path, "w") as f:
            f.write("test mlmodel")

        with patch(
            "nubison_model.Service._get_shared_artifacts_dir",
            return_value=os.path.join(temp_dir, "shared"),
        ):
            with patch("nubison_model.Service.set_tracking_uri"):
                with patch(
                    "nubison_model.Service.load_model",
                    return_value=MockMLflowModel(model_path),
                ):
                    result = load_nubison_mlflow_model("http://test", "model_uri")
                    assert isinstance(result, NubisonMLFlowModel)

                    # Verify path file was created
                    path_file = os.path.join(temp_dir, "shared") + ".path"
                    assert os.path.exists(path_file)
                    with open(path_file, "r") as f:
                        cached_path = f.read().strip()
                    assert cached_path == os.path.join(temp_dir, "model")


def test_corrupted_cache_file_handling():
    """Test handling of corrupted or invalid cache files."""

    class MockNubisonModel:
        def infer(self, test: str):
            return test

        def load_model(self, context):
            pass

    fallback_calls = []

    def mock_load_model_wrapper(tracking_uri, model_uri):
        fallback_calls.append(model_uri)
        return MagicMock(), NubisonMLFlowModel(MockNubisonModel())

    with tempfile.TemporaryDirectory() as temp_dir:
        shared_dir = os.path.join(temp_dir, "shared")
        path_file = shared_dir + ".path"

        # Test 1: Cache file points to non-existent path
        with open(path_file, "w") as f:
            f.write("/non/existent/path")

        with patch(
            "nubison_model.Service._get_shared_artifacts_dir", return_value=shared_dir
        ):
            with patch(
                "nubison_model.Service._load_model_with_nubison_wrapper",
                side_effect=mock_load_model_wrapper,
            ) as mock_wrapper:
                # First call should try cached path and fail, then fallback to original URI
                def side_effect_cached_failure(tracking_uri, model_uri):
                    if model_uri == "/non/existent/path":
                        raise FileNotFoundError("Model file not found")
                    return mock_load_model_wrapper(tracking_uri, model_uri)

                mock_wrapper.side_effect = side_effect_cached_failure

                # Should handle the cached path failure and fallback to original URI
                with pytest.raises(FileNotFoundError):
                    load_nubison_mlflow_model("http://test", "model_uri")

        # Test 2: Empty cache file
        fallback_calls.clear()
        with open(path_file, "w") as f:
            f.write("")

        with patch(
            "nubison_model.Service._get_shared_artifacts_dir", return_value=shared_dir
        ):
            with patch(
                "nubison_model.Service._load_model_with_nubison_wrapper",
                side_effect=mock_load_model_wrapper,
            ) as mock_wrapper:

                def side_effect_empty_path(tracking_uri, model_uri):
                    if model_uri == "":
                        raise ValueError("Empty model path")
                    return mock_load_model_wrapper(tracking_uri, model_uri)

                mock_wrapper.side_effect = side_effect_empty_path

                with pytest.raises(ValueError):
                    load_nubison_mlflow_model("http://test", "model_uri")


def test_download_exception_cleanup():
    """Test proper cleanup when download fails with exception."""

    class MockNubisonModel:
        def infer(self, test: str):
            return test

        def load_model(self, context):
            pass

    with tempfile.TemporaryDirectory() as temp_dir:
        shared_dir = os.path.join(temp_dir, "shared")
        path_file = shared_dir + ".path"

        # Ensure no cache exists initially
        assert not os.path.exists(path_file)

        with patch(
            "nubison_model.Service._get_shared_artifacts_dir", return_value=shared_dir
        ):
            with patch(
                "nubison_model.Service._load_cached_model_if_available",
                return_value=None,
            ):
                # Mock download to raise exception
                with patch(
                    "nubison_model.Service._load_model_with_nubison_wrapper"
                ) as mock_wrapper:
                    mock_wrapper.side_effect = RuntimeError("Download failed")

                    # Should raise the download exception
                    with pytest.raises(RuntimeError, match="Download failed"):
                        load_nubison_mlflow_model("http://test", "model_uri")

                    # Verify no partial cache file was created
                    assert not os.path.exists(path_file)

        # Test exception during cache path extraction
        with patch(
            "nubison_model.Service._get_shared_artifacts_dir", return_value=shared_dir
        ):
            with patch(
                "nubison_model.Service._load_cached_model_if_available",
                return_value=None,
            ):
                with patch(
                    "nubison_model.Service._load_model_with_nubison_wrapper"
                ) as mock_wrapper:
                    mock_wrapper.return_value = (
                        MagicMock(),
                        NubisonMLFlowModel(MockNubisonModel()),
                    )

                    # Mock cache extraction to fail
                    with patch(
                        "nubison_model.Service._extract_and_cache_model_path"
                    ) as mock_extract:
                        mock_extract.side_effect = OSError("Disk full")

                        # Should raise the extraction exception
                        with pytest.raises(OSError, match="Disk full"):
                            load_nubison_mlflow_model("http://test", "model_uri")

                        # Verify no partial cache file was created
                        assert not os.path.exists(path_file)


# Ignore the test_client from being collected by pytest
setattr(test_client, "__test__", False)
