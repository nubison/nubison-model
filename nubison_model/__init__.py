"""nubison-model package."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("nubison-model")
except PackageNotFoundError:
    # Editable install or source tree without metadata — fall back to
    # the pyproject placeholder. The real version is filled by
    # poetry-dynamic-versioning at build time.
    __version__ = "0.0.0"

from . import Data as data
from .Data import SOURCE_URI_ATTR
from .Model import (
    ENV_VAR_MLFLOW_MODEL_URI,
    ENV_VAR_MLFLOW_TRACKING_URI,
    ModelContext,
    NubisonMLFlowModel,
    NubisonModel,
    register,
)
from .Service import build_inference_service, test_client
from .Storage import (
    ENV_VAR_AWS_ENDPOINT_URL,
    ENV_VAR_DVC_ENABLED,
    ENV_VAR_DVC_REMOTE_URL,
    ENV_VAR_DVC_SIZE_THRESHOLD,
    DVCError,
    DVCPullError,
    DVCPushError,
    ensure_dvc_ready,
    is_dvc_enabled,
)
from .Train import TrainContext, train

__all__ = [
    "ENV_VAR_MLFLOW_MODEL_URI",
    "ENV_VAR_MLFLOW_TRACKING_URI",
    "ENV_VAR_DVC_ENABLED",
    "ENV_VAR_DVC_REMOTE_URL",
    "ENV_VAR_DVC_SIZE_THRESHOLD",
    "ENV_VAR_AWS_ENDPOINT_URL",
    "SOURCE_URI_ATTR",
    "ModelContext",
    "NubisonModel",
    "NubisonMLFlowModel",
    "register",
    "train",
    "TrainContext",
    "data",
    "build_inference_service",
    "test_client",
    "is_dvc_enabled",
    "ensure_dvc_ready",
    "DVCError",
    "DVCPushError",
    "DVCPullError",
]
