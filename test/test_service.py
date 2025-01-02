from unittest.mock import patch

import pytest
from PIL.Image import Image
from PIL.Image import open as open_image

from nubison_model import (
    ModelContext,
    NubisonMLFlowModel,
    build_inference_service,
    test_client,
)
from nubison_model.Service import DEFAULT_NUM_WORKERS
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


# Ignore the test_client from being collected by pytest
setattr(test_client, "__test__", False)
