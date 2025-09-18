import os
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import pytest
from PIL.Image import Image
from PIL.Image import open as open_image

from nubison_model import (
    ModelContext,
    NubisonMLFlowModel,
    build_inference_service,
    test_client,
)
from nubison_model.Service import DEFAULT_NUM_WORKERS, load_nubison_mlflow_model
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


def test_artifact_sharing_basic():
    """Test basic artifact sharing functionality."""
    from nubison_model.Service import get_shared_artifacts_dir

    with tempfile.TemporaryDirectory() as temp_dir:
        with patch(
            "nubison_model.Service.get_shared_artifacts_dir", return_value=temp_dir
        ):
            # Test path file operations
            path_file = temp_dir + ".path"
            test_path = "/test/model/path"

            # Simulate first worker saving path
            with open(path_file, "w") as f:
                f.write(test_path)

            # Simulate second worker reading path
            with open(path_file, "r") as f:
                cached_path = f.read().strip()

            assert cached_path == test_path


def test_lock_directory_creation():
    """Test lock directory creation and cleanup."""
    with tempfile.TemporaryDirectory() as temp_dir:
        lock_dir = os.path.join(temp_dir, "test.lock")

        # Test lock creation
        try:
            os.makedirs(lock_dir, exist_ok=False)
            assert os.path.exists(lock_dir)

            # Test lock cleanup
            os.rmdir(lock_dir)
            assert not os.path.exists(lock_dir)
        except FileExistsError:
            pytest.fail("Lock directory should not exist initially")


def test_concurrent_lock_behavior():
    """Test concurrent lock acquisition behavior."""
    with tempfile.TemporaryDirectory() as temp_dir:
        lock_dir = os.path.join(temp_dir, "test.lock")

        # First thread should succeed
        try:
            os.makedirs(lock_dir, exist_ok=False)
            lock_acquired = True
        except FileExistsError:
            lock_acquired = False

        assert lock_acquired, "First lock attempt should succeed"

        # Second attempt should fail
        try:
            os.makedirs(lock_dir, exist_ok=False)
            second_lock_acquired = True
        except FileExistsError:
            second_lock_acquired = False

        assert not second_lock_acquired, "Second lock attempt should fail"

        # Cleanup
        os.rmdir(lock_dir)


def test_fallback_on_missing_uris():
    """Test that proper errors are raised when URIs are missing."""
    with pytest.raises(
        RuntimeError, match="MLflow tracking URI and model URI must be set"
    ):
        load_nubison_mlflow_model("", "test_model")

    with pytest.raises(
        RuntimeError, match="MLflow tracking URI and model URI must be set"
    ):
        load_nubison_mlflow_model("http://test", "")


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
            "nubison_model.Service.get_shared_artifacts_dir",
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


def test_load_nubison_mlflow_model_lock_mechanism():
    """Test that load_nubison_mlflow_model implements proper lock mechanism."""

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

    called_uris = []

    with tempfile.TemporaryDirectory() as temp_dir:

        def mock_load_model(model_uri):
            called_uris.append(model_uri)
            # Return the actual artifact path that exists on disk
            return MockMLflowModel(
                os.path.join(temp_dir, "model", "artifacts", "model_file")
            )

        # Create proper directory structure that mimics MLflow artifacts
        model_root = os.path.join(temp_dir, "model")
        artifacts_dir = os.path.join(model_root, "artifacts")
        model_path = os.path.join(artifacts_dir, "model_file")

        os.makedirs(artifacts_dir, exist_ok=True)
        with open(model_path, "w") as f:
            f.write("test model")

        # Create MLmodel file in model root
        mlmodel_path = os.path.join(model_root, "MLmodel")
        with open(mlmodel_path, "w") as f:
            f.write("test mlmodel")

        shared_dir = os.path.join(temp_dir, "shared")

        with patch(
            "nubison_model.Service.get_shared_artifacts_dir", return_value=shared_dir
        ):
            with patch("nubison_model.Service.set_tracking_uri"):
                with patch(
                    "nubison_model.Service.load_model", side_effect=mock_load_model
                ):
                    # First call should download and create cache
                    result1 = load_nubison_mlflow_model("http://test", "model_uri")
                    assert isinstance(result1, NubisonMLFlowModel)
                    assert len(called_uris) == 1
                    assert called_uris[0] == "model_uri"

                    # Path file should exist with the model root path
                    path_file = shared_dir + ".path"
                    assert os.path.exists(path_file)

                    with open(path_file, "r") as f:
                        cached_path = f.read().strip()
                    assert cached_path == model_root

                    # Second call should use cached path
                    result2 = load_nubison_mlflow_model("http://test", "model_uri")
                    assert isinstance(result2, NubisonMLFlowModel)

                    # Second call should load from cached path, not original URI
                    assert len(called_uris) == 2
                    assert (
                        called_uris[1] == model_root
                    )  # This should be the cached path


def test_load_nubison_mlflow_model_uses_cached_path():
    """Test that subsequent calls use cached path without downloading."""

    class MockNubisonModel:
        def infer(self, test: str):
            return test

        def load_model(self, context):
            pass

    download_count = 0

    def mock_load_model(model_uri):
        nonlocal download_count
        download_count += 1
        return MagicMock()

    def mock_load_model_wrapper(tracking_uri, model_uri):
        return NubisonMLFlowModel(MockNubisonModel())

    with tempfile.TemporaryDirectory() as temp_dir:
        shared_dir = os.path.join(temp_dir, "shared")
        path_file = shared_dir + ".path"
        cached_path = "/cached/model/path"

        # Pre-create cached path file
        with open(path_file, "w") as f:
            f.write(cached_path)

        with patch(
            "nubison_model.Service.get_shared_artifacts_dir", return_value=shared_dir
        ):
            with patch("nubison_model.Service.load_model", side_effect=mock_load_model):
                with patch(
                    "nubison_model.Service._load_model_with_nubison_wrapper",
                    side_effect=mock_load_model_wrapper,
                ) as mock_wrapper:
                    result = load_nubison_mlflow_model("http://test", "original_uri")

                    # Should use cached path, not original URI
                    mock_wrapper.assert_called_once_with("http://test", cached_path)
                    assert isinstance(result, NubisonMLFlowModel)

                    # No downloads should have occurred
                    assert download_count == 0


# Ignore the test_client from being collected by pytest
setattr(test_client, "__test__", False)
