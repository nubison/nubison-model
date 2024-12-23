from unittest.mock import patch

import pytest
from PIL.Image import Image
from PIL.Image import open as open_image

from nubison_model import NubisonMLFlowModel, build_inference_service, test_client


def test_raise_runtime_error_on_missing_env():
    with pytest.raises(RuntimeError):
        build_inference_service()


def test_service_ok():
    class DummyModel:
        def infer(self, test: str):
            return test

        def load_model(self):
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

        def load_model(self):
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

        def load_model(self):
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


# Ignore the test_client from being collected by pytest
setattr(test_client, "__test__", False)
