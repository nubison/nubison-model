from contextlib import contextmanager
from functools import wraps
from os import environ, getenv
import os
from tempfile import TemporaryDirectory
from typing import Optional, cast
import tempfile
import time

import bentoml
from filelock import FileLock, Timeout
from mlflow import set_tracking_uri
from mlflow.pyfunc import load_model
from starlette.testclient import TestClient

from nubison_model.Model import (
    DEAFULT_MLFLOW_URI,
    ENV_VAR_MLFLOW_MODEL_URI,
    ENV_VAR_MLFLOW_TRACKING_URI,
    NubisonMLFlowModel,
)
from nubison_model.utils import temporary_cwd

ENV_VAR_NUM_WORKERS = "NUM_WORKERS"
DEFAULT_NUM_WORKERS = 1


def _get_shared_artifacts_dir():
    """Get the shared artifacts directory path (OS-compatible)."""
    return os.path.join(tempfile.gettempdir(), "nubison_shared_artifacts")


def _load_model_with_nubison_wrapper(mlflow_tracking_uri, model_uri):
    """Load MLflow model and wrap with NubisonMLFlowModel.

    Returns:
        tuple: (mlflow_model, nubison_model)
    """
    set_tracking_uri(mlflow_tracking_uri)
    mlflow_model = load_model(model_uri=model_uri)
    nubison_model = cast(NubisonMLFlowModel, mlflow_model.unwrap_python_model())
    return mlflow_model, nubison_model


def _load_cached_model_if_available(mlflow_tracking_uri, path_file):
    """Load model from cached path if available."""
    if not os.path.exists(path_file):
        return None

    with open(path_file, "r") as f:
        cached_path = f.read().strip()
    _, nubison_model = _load_model_with_nubison_wrapper(
        mlflow_tracking_uri, cached_path
    )
    return nubison_model


def _extract_and_cache_model_path(mlflow_model, path_file):
    """Extract model root path from artifacts and cache it."""
    try:
        context = mlflow_model._model_impl.context
        valid_paths = (
            str(path)
            for path in context.artifacts.values()
            if path and os.path.exists(str(path))
        )

        for artifact_path in valid_paths:
            model_root = os.path.dirname(os.path.dirname(artifact_path))
            if os.path.exists(os.path.join(model_root, "MLmodel")):
                with open(path_file, "w") as f:
                    f.write(model_root)
                break

    except (AttributeError, TypeError):
        pass


def load_nubison_mlflow_model(mlflow_tracking_uri, mlflow_model_uri):
    """Load a Nubison MLflow model with robust caching and multi-worker support.

    This function implements a sophisticated model loading strategy that uses FileLock
    for inter-process synchronization, ensuring only one worker downloads the model
    while others wait and reuse the cached result. Includes automatic timeout handling
    and fallback mechanisms for production reliability.

    Args:
        mlflow_tracking_uri (str): MLflow tracking server URI for model registry access
        mlflow_model_uri (str): Model URI in MLflow format (e.g., 'models:/model_name/version')

    Returns:
        NubisonMLFlowModel: Loaded and wrapped model ready for inference

    Raises:
        RuntimeError: If required URIs are not provided
        Timeout: If lock acquisition times out (handled with fallback)

    Note:
        Uses 5-minute timeout and double-check pattern to prevent race conditions.
        Automatically extracts and caches local model paths for faster subsequent loads.
    """
    if not mlflow_tracking_uri or not mlflow_model_uri:
        raise RuntimeError("MLflow tracking URI and model URI must be set")

    shared_info_dir = _get_shared_artifacts_dir()
    lock_file = shared_info_dir + ".lock"
    path_file = shared_info_dir + ".path"

    # Try loading from cache first
    cached_model = _load_cached_model_if_available(mlflow_tracking_uri, path_file)
    if cached_model:
        return cached_model

    # Use FileLock for robust locking with timeout
    file_lock = FileLock(lock_file, timeout=300)

    try:
        with file_lock:
            # Double-check pattern: verify cache doesn't exist after acquiring lock
            cached_model = _load_cached_model_if_available(
                mlflow_tracking_uri, path_file
            )
            if cached_model:
                return cached_model

            # Load model and extract path for caching
            mlflow_model, nubison_model = _load_model_with_nubison_wrapper(
                mlflow_tracking_uri, mlflow_model_uri
            )

            # Cache model path for other workers
            _extract_and_cache_model_path(mlflow_model, path_file)

            return nubison_model

    except Timeout:
        # Fallback to original URI if lock timeout occurs
        _, nubison_model = _load_model_with_nubison_wrapper(
            mlflow_tracking_uri, mlflow_model_uri
        )
        return nubison_model


@contextmanager
def test_client(model_uri):

    # Create a temporary directory and set it as the current working directory to run tests
    # To avoid model initialization conflicts with the current directory
    test_dir = TemporaryDirectory()
    with temporary_cwd(test_dir.name):
        app = build_inference_service(mlflow_model_uri=model_uri)
        # Disable metrics for testing. Avoids Prometheus client duplicated registration error
        app.config["metrics"] = {"enabled": False}

        with TestClient(app.to_asgi()) as client:
            yield client

    test_dir.cleanup()


def build_inference_service(
    mlflow_tracking_uri: Optional[str] = None, mlflow_model_uri: Optional[str] = None
):
    mlflow_tracking_uri = (
        mlflow_tracking_uri or getenv(ENV_VAR_MLFLOW_TRACKING_URI) or DEAFULT_MLFLOW_URI
    )
    mlflow_model_uri = mlflow_model_uri or getenv(ENV_VAR_MLFLOW_MODEL_URI) or ""

    num_workers = int(getenv(ENV_VAR_NUM_WORKERS) or DEFAULT_NUM_WORKERS)

    nubison_mlflow_model = load_nubison_mlflow_model(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_model_uri=mlflow_model_uri,
    )

    @bentoml.service(workers=num_workers)
    class BentoMLService:
        """BentoML Service for serving machine learning models."""

        def __init__(self):
            """Initializes the BentoML Service for serving machine learning models.

            This function retrieves a Nubison Model wrapped as an MLflow model
            The Nubison Model contains user-defined methods for performing inference.

            Raises:
                RuntimeError: Error loading model from the model registry
            """

            # Set default worker index to 1 in case of no bentoml server context is available
            # For example, when running with test client
            context = {
                "worker_index": 0,
                "num_workers": 1,
            }
            if bentoml.server_context.worker_index is not None:
                context = {
                    "worker_index": bentoml.server_context.worker_index - 1,
                    "num_workers": num_workers,
                }

            nubison_mlflow_model.load_model(context)

        @bentoml.api
        @wraps(nubison_mlflow_model.get_nubison_model_infer_method())
        def infer(self, *args, **kwargs):
            """Proxy method to the NubisonModel.infer method

            Raises:
                RuntimeError: Error requested inference with no Model loaded

            Returns:
                _type_: The return type of the NubisonModel.infer method
            """
            return nubison_mlflow_model.infer(*args, **kwargs)

    return BentoMLService


# Make BentoService if the script is loaded by BentoML
# This requires the running mlflow server and the model registered to the model registry
# The model registry URI and model URI should be set as environment variables
loaded_by_bentoml = any(var.startswith("BENTOML_") for var in environ)
if loaded_by_bentoml:
    InferenceService = build_inference_service()
