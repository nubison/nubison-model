"""nubison-model package."""

__version__ = "0.0.0"

from .Model import (
    ENV_VAR_MLFLOW_MODEL_URI,
    ENV_VAR_MLFLOW_TRACKING_URI,
    NubisonMLFlowModel,
    NubisonModel,
    register,
)
from .Service import build_inference_service, test_client

__all__ = [
    "ENV_VAR_MLFLOW_MODEL_URI",
    "ENV_VAR_MLFLOW_TRACKING_URI",
    "NubisonModel",
    "NubisonMLFlowModel",
    "register",
    "build_inference_service",
    "test_client",
]
