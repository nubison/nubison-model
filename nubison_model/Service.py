import glob
import logging
import os
import tempfile
from contextlib import contextmanager
from functools import wraps
from os import environ, getenv
from tempfile import TemporaryDirectory
from typing import Optional, cast

import bentoml
import mlflow
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
from nubison_model.Storage import (
    DVC_FILES_TAG_KEY,
    ChecksumMismatchError,
    DVCPullError,
    deserialize_dvc_info,
    get_dvc_cache_key,
    is_dvc_enabled,
    pull_from_dvc,
)
from nubison_model.utils import temporary_cwd

# Setup logging
logger = logging.getLogger(__name__)

ENV_VAR_NUM_WORKERS = "NUM_WORKERS"
DEFAULT_NUM_WORKERS = 1

ENV_VAR_REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
DEFAULT_REQUEST_TIMEOUT = 60


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


def _parse_model_uri(model_uri: str) -> Optional[tuple]:
    """
    Parse MLflow model URI into components.

    Args:
        model_uri: Model URI (e.g., 'models:/model_name/version' or 'runs:/run_id/path')

    Returns:
        Tuple of (uri_type, name, version_or_path) or None if invalid format
        - uri_type: 'models' or 'runs'
        - For 'models': (model_name, version_or_stage)
        - For 'runs': (run_id, artifact_path or None)
    """
    uri_prefixes = [("models:/", "models"), ("runs:/", "runs")]

    for prefix, uri_type in uri_prefixes:
        if model_uri.startswith(prefix):
            parts = model_uri[len(prefix) :].split("/", 1)
            if parts and parts[0]:
                second_part = parts[1] if len(parts) > 1 else None
                return (uri_type, parts[0], second_part)
    return None


def _get_model_version(client, model_name: str, version_or_stage: str):
    """
    Get MLflow model version, handling both numeric versions and stage names.

    Args:
        client: MLflow client
        model_name: Name of the registered model
        version_or_stage: Version number (as string) or stage name

    Returns:
        ModelVersion object or None if not found
    """
    try:
        version = int(version_or_stage)
        return client.get_model_version(model_name, str(version))
    except ValueError:
        # It's a stage name, get latest version in that stage
        versions = client.get_latest_versions(model_name, stages=[version_or_stage])
        return versions[0] if versions else None


def _get_dvc_info_from_model_uri(mlflow_tracking_uri: str, model_uri: str) -> dict:
    """
    Extract DVC file info from MLflow model version tags.

    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        model_uri: Model URI (e.g., 'models:/model_name/version')

    Returns:
        Dictionary mapping file paths to md5 hashes, or empty dict if not found
    """
    set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        parsed = _parse_model_uri(model_uri)
        if not parsed:
            return {}

        uri_type, identifier, extra = parsed

        if uri_type == "models" and extra:
            mv = _get_model_version(client, identifier, extra)
            if mv:
                dvc_files_json = mv.tags.get(DVC_FILES_TAG_KEY)
                if dvc_files_json:
                    return deserialize_dvc_info(dvc_files_json)

        elif uri_type == "runs":
            run = client.get_run(identifier)
            dvc_files_json = run.data.tags.get(DVC_FILES_TAG_KEY)
            if dvc_files_json:
                return deserialize_dvc_info(dvc_files_json)

    except Exception as e:
        logger.warning(f"Could not retrieve DVC info from MLflow: {e}")

    return {}


def _cleanup_old_dvc_done_files(shared_info_dir: str, current_dvc_done_file: str) -> None:
    """
    Clean up old DVC done files from previous model versions.

    Args:
        shared_info_dir: Base directory for shared artifacts
        current_dvc_done_file: Path to the current DVC done file (to preserve)
    """
    pattern = shared_info_dir + ".dvc_done_*"
    for old_file in glob.glob(pattern):
        if old_file != current_dvc_done_file:
            try:
                os.remove(old_file)
                logger.debug(f"Cleaned up old DVC done file: {old_file}")
            except OSError as e:
                logger.warning(f"Failed to remove old DVC done file {old_file}: {e}")


def _needs_dvc_restoration(dvc_enabled: bool, dvc_info: dict, dvc_done_file: str) -> bool:
    """
    Check if DVC files need to be restored for this model version.

    Args:
        dvc_enabled: Whether DVC is enabled
        dvc_info: Dictionary of DVC file info (empty if no DVC files)
        dvc_done_file: Path to the DVC done marker file

    Returns:
        True if DVC restoration is needed, False otherwise
    """
    if not dvc_enabled or not dvc_info or not dvc_done_file:
        return False
    return not os.path.exists(dvc_done_file)


def _restore_dvc_files(dvc_info: dict, model_root: str) -> None:
    """
    Restore DVC-tracked files to the model directory.

    Args:
        dvc_info: Dictionary mapping file paths to md5 hashes
        model_root: Root directory of the model artifacts

    Raises:
        DVCPullError: If download fails
        ChecksumMismatchError: If checksum verification fails
    """
    if not dvc_info:
        return

    logger.info(f"DVC: Restoring {len(dvc_info)} file(s) from remote storage...")

    try:
        pull_from_dvc(dvc_info, model_root, verify_checksum=True, show_progress=True)
        logger.info("DVC: Files restored successfully")
    except (DVCPullError, ChecksumMismatchError) as e:
        logger.error(f"Failed to restore DVC files: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error restoring DVC files: {e}")
        raise DVCPullError(f"Failed to restore DVC files: {e}") from e


def load_nubison_mlflow_model(mlflow_tracking_uri, mlflow_model_uri):
    """Load a Nubison MLflow model with robust caching and multi-worker support.

    This function implements a sophisticated model loading strategy that uses FileLock
    for inter-process synchronization, ensuring only one worker downloads the model
    while others wait and reuse the cached result. Includes automatic timeout handling
    and fallback mechanisms for production reliability.

    When DVC is enabled, this function also restores large weight files from DVC
    remote storage based on the file hashes stored in MLflow tags. The restoration
    is version-aware: each model version has its own cache key, ensuring that
    different versions don't incorrectly share DVC files.

    Args:
        mlflow_tracking_uri (str): MLflow tracking server URI for model registry access
        mlflow_model_uri (str): Model URI in MLflow format (e.g., 'models:/model_name/version')

    Returns:
        NubisonMLFlowModel: Loaded and wrapped model ready for inference

    Raises:
        RuntimeError: If required URIs are not provided
        DVCPullError: If DVC file restoration fails
        ChecksumMismatchError: If downloaded file checksum doesn't match

    Note:
        Uses 5-minute timeout and double-check pattern to prevent race conditions.
        Automatically extracts and caches local model paths for faster subsequent loads.

    Environment Variables:
        DVC_ENABLED: Set to 'true' to enable DVC file restoration
        DVC_REMOTE_URL: URL of DVC remote storage (required when DVC is enabled)
    """
    if not mlflow_tracking_uri or not mlflow_model_uri:
        raise RuntimeError("MLflow tracking URI and model URI must be set")

    shared_info_dir = _get_shared_artifacts_dir()
    lock_file = shared_info_dir + ".lock"
    path_file = shared_info_dir + ".path"

    # Check if DVC is enabled and get DVC info early
    dvc_enabled = is_dvc_enabled()
    dvc_info = {}
    if dvc_enabled:
        dvc_info = _get_dvc_info_from_model_uri(mlflow_tracking_uri, mlflow_model_uri)

    # Generate version-specific cache key for DVC restoration
    dvc_cache_key = get_dvc_cache_key(mlflow_model_uri, dvc_info) if dvc_info else ""
    dvc_done_file = (
        f"{shared_info_dir}.dvc_done_{dvc_cache_key}" if dvc_cache_key else ""
    )

    # Try loading from cache first
    cached_model = _load_cached_model_if_available(mlflow_tracking_uri, path_file)
    if cached_model:
        # Even if model is cached, check if DVC files need restoration for this version
        if _needs_dvc_restoration(dvc_enabled, dvc_info, dvc_done_file):
            logger.info("Model cached but DVC files need restoration for this version")
            # Will proceed to restore DVC files below
        else:
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
                # Check DVC restoration status again under lock
                if not _needs_dvc_restoration(dvc_enabled, dvc_info, dvc_done_file):
                    return cached_model

            # Load model and extract path for caching
            mlflow_model, nubison_model = _load_model_with_nubison_wrapper(
                mlflow_tracking_uri, mlflow_model_uri
            )

            # Cache model path for other workers
            _extract_and_cache_model_path(mlflow_model, path_file)

            # Restore DVC files if enabled and not already done for this version
            if _needs_dvc_restoration(dvc_enabled, dvc_info, dvc_done_file):
                # Clean up old DVC done files from previous versions
                _cleanup_old_dvc_done_files(shared_info_dir, dvc_done_file)

                # Get model root directory for DVC file restoration
                model_root = "."
                if os.path.exists(path_file):
                    with open(path_file, "r") as f:
                        model_root = f.read().strip()

                _restore_dvc_files(dvc_info, model_root)

                # Mark DVC restoration as done for this version
                with open(dvc_done_file, "w") as f:
                    f.write(f"version:{mlflow_model_uri}")

            return nubison_model

    except Timeout:
        logger.warning("Lock acquisition timed out, falling back to direct load")
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
        mlflow_tracking_uri
        or getenv(ENV_VAR_MLFLOW_TRACKING_URI)
        or DEAFULT_MLFLOW_URI
    )
    mlflow_model_uri = mlflow_model_uri or getenv(ENV_VAR_MLFLOW_MODEL_URI) or ""

    num_workers = int(getenv(ENV_VAR_NUM_WORKERS) or DEFAULT_NUM_WORKERS)
    request_timeout = int(getenv(ENV_VAR_REQUEST_TIMEOUT) or DEFAULT_REQUEST_TIMEOUT)

    nubison_mlflow_model = load_nubison_mlflow_model(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_model_uri=mlflow_model_uri,
    )

    @bentoml.service(workers=num_workers, traffic={"timeout": request_timeout})
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
