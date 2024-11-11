from unittest.mock import patch

import pytest

from nubison_model import build_inference_service


def test_raise_runtime_error_on_missing_env():
    with pytest.raises(RuntimeError):
        build_inference_service()


def test_service_ok():
    class DummyModel:
        def infer(self, test: str):
            return test

    with patch("nubison_model.Service.load_nubison_model") as mock_load_nubison_model:
        mock_load_nubison_model.return_value = DummyModel()
        service = build_inference_service()()
        assert service.infer("test") == "test"
