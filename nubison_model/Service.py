import glob
import hashlib
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
    DEFAULT_MLFLOW_URI,
    ENV_VAR_MLFLOW_MODEL_URI,
    ENV_VAR_MLFLOW_TRACKING_URI,
    NubisonMLFlowModel,
)
from nubison_model.Storage import (
    DVC_FILES_TAG_KEY,
    DVCPullError,
    deserialize_dvc_info,
    get_dvc_cache_key,
    is_dvc_enabled,
    pull_from_dvc,
)
from nubison_model.utils import temporary_cwd

logger = logging.getLogger(__name__)

ENV_VAR_NUM_WORKERS = "NUM_WORKERS"
DEFAULT_NUM_WORKERS = 1

ENV_VAR_REQUEST_TIMEOUT = "REQUEST_TIMEOUT"
DEFAULT_REQUEST_TIMEOUT = 60


def _get_shared_artifacts_dir():
    """Get the shared artifacts directory path (OS-compatible)."""
    return os.path.join(tempfile.gettempdir(), "nubison_shared_artifacts")


def _get_model_cache_key(model_uri: str) -> str:
    """Generate a cache key from model URI."""
    return hashlib.md5(model_uri.encode()).hexdigest()[:12]


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
        versions = client.get_latest_versions(model_name, stages=[version_or_stage])
        return versions[0] if versions else None


def _get_tags_for_uri(client, model_uri: str) -> Optional[dict]:
    """Extract tags from MLflow based on URI type."""
    parsed = _parse_model_uri(model_uri)
    if not parsed:
        return None

    uri_type, identifier, extra = parsed

    if uri_type == "models" and extra:
        mv = _get_model_version(client, identifier, extra)
        return mv.tags if mv else None

    if uri_type == "runs":
        return client.get_run(identifier).data.tags

    return None


def _get_dvc_info_from_model_uri(mlflow_tracking_uri: str, model_uri: str) -> dict:
    """Extract DVC file info from MLflow model version tags."""
    set_tracking_uri(mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        tags = _get_tags_for_uri(client, model_uri)
        dvc_json = tags.get(DVC_FILES_TAG_KEY) if tags else None
        return deserialize_dvc_info(dvc_json) if dvc_json else {}
    except Exception as e:
        logger.warning(f"Could not retrieve DVC info from MLflow: {e}")
        return {}


def _cleanup_old_dvc_done_files(
    shared_info_dir: str, current_dvc_done_file: str
) -> None:
    """Clean up old DVC done files from previous model versions."""
    pattern = shared_info_dir + ".dvc_done_*"
    for old_file in glob.glob(pattern):
        if old_file != current_dvc_done_file:
            try:
                os.remove(old_file)
                logger.debug(f"Cleaned up old DVC done file: {old_file}")
            except OSError as e:
                logger.warning(f"Failed to remove old DVC done file {old_file}: {e}")


def _create_dvc_symlinks(dvc_info: dict, model_root: str) -> None:
    """Create symlinks for DVC-restored files in current working directory."""
    for file_path in dvc_info.keys():
        target_path = os.path.join(model_root, file_path)
        if not os.path.exists(target_path):
            continue
        try:
            parent_dir = os.path.dirname(file_path)
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)
            is_dir = os.path.isdir(target_path)
            os.symlink(target_path, file_path, target_is_directory=is_dir)
            logger.debug(f"DVC: Created symlink {file_path} -> {target_path}")
        except (FileNotFoundError, FileExistsError):
            pass
        except OSError as e:
            logger.error(f"Error creating symlink for {file_path}: {e}")


def _restore_dvc_files(dvc_info: dict, model_root: str, model_uri: str = "") -> None:
    """Restore DVC-tracked files to the model directory and create symlinks."""
    if not dvc_info:
        return

    logger.info(f"DVC: Restoring {len(dvc_info)} file(s) from remote storage...")

    # MLflow stores artifacts in model_root/artifacts/ directory
    artifacts_dir = os.path.join(model_root, "artifacts")
    if not os.path.isdir(artifacts_dir):
        artifacts_dir = model_root  # Fallback if artifacts/ doesn't exist

    try:
        # Run DVC commands from artifacts_dir where .dvc files are located
        with temporary_cwd(artifacts_dir):
            pull_from_dvc(dvc_info, ".", show_progress=True)
        _create_dvc_symlinks(dvc_info, artifacts_dir)
        logger.info("DVC: Files restored successfully")
    except DVCPullError as e:
        logger.error(f"Failed to restore DVC files for {model_uri}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error restoring DVC files for {model_uri}: {e}")
        raise DVCPullError(f"Failed to restore DVC files for {model_uri}: {e}") from e


def _get_model_root(path_file: str) -> str:
    """Get model root directory from cached path file."""
    if not os.path.exists(path_file):
        return "."
    with open(path_file, "r") as f:
        return f.read().strip() or "."


def _mark_dvc_done(dvc_done_file: str, model_uri: str) -> None:
    """Mark DVC restoration as complete for this version."""
    with open(dvc_done_file, "w") as f:
        f.write(f"version:{model_uri}")


def _handle_dvc_restoration(
    dvc_info: dict,
    dvc_done_file: str,
    shared_info_dir: str,
    path_file: str,
    model_uri: str,
) -> None:
    """Handle DVC file restoration if needed."""
    if not dvc_info or not dvc_done_file or os.path.exists(dvc_done_file):
        return

    _cleanup_old_dvc_done_files(shared_info_dir, dvc_done_file)
    _restore_dvc_files(dvc_info, _get_model_root(path_file), model_uri)
    _mark_dvc_done(dvc_done_file, model_uri)


def load_nubison_mlflow_model(mlflow_tracking_uri, mlflow_model_uri):
    """Load a Nubison MLflow model with robust caching and multi-worker support.

    Uses FileLock for inter-process synchronization. When DVC is enabled,
    restores large weight files from DVC remote storage.

    Args:
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_model_uri: Model URI (e.g., 'models:/model_name/version')

    Returns:
        NubisonMLFlowModel: Loaded model ready for inference

    Raises:
        RuntimeError: If required URIs are not provided
        DVCPullError: If DVC file restoration fails
    """
    if not mlflow_tracking_uri or not mlflow_model_uri:
        raise RuntimeError("MLflow tracking URI and model URI must be set")

    shared_info_dir = _get_shared_artifacts_dir()
    model_cache_key = _get_model_cache_key(mlflow_model_uri)
    lock_file = f"{shared_info_dir}.lock_{model_cache_key}"
    path_file = f"{shared_info_dir}.path_{model_cache_key}"

    dvc_info = (
        _get_dvc_info_from_model_uri(mlflow_tracking_uri, mlflow_model_uri)
        if is_dvc_enabled()
        else {}
    )
    dvc_cache_key = get_dvc_cache_key(mlflow_model_uri, dvc_info) if dvc_info else ""
    dvc_done_file = (
        f"{shared_info_dir}.dvc_done_{dvc_cache_key}" if dvc_cache_key else ""
    )

    needs_dvc = bool(dvc_info and dvc_done_file and not os.path.exists(dvc_done_file))

    cached_model = _load_cached_model_if_available(mlflow_tracking_uri, path_file)
    if cached_model and not needs_dvc:
        return cached_model

    try:
        with FileLock(lock_file, timeout=300):
            needs_dvc = bool(
                dvc_info and dvc_done_file and not os.path.exists(dvc_done_file)
            )
            cached_model = _load_cached_model_if_available(
                mlflow_tracking_uri, path_file
            )
            if cached_model and not needs_dvc:
                return cached_model

            mlflow_model, nubison_model = _load_model_with_nubison_wrapper(
                mlflow_tracking_uri, mlflow_model_uri
            )
            _extract_and_cache_model_path(mlflow_model, path_file)
            _handle_dvc_restoration(
                dvc_info, dvc_done_file, shared_info_dir, path_file, mlflow_model_uri
            )

            return nubison_model

    except Timeout:
        logger.warning("Lock timeout, falling back to direct load")
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
        mlflow_tracking_uri or getenv(ENV_VAR_MLFLOW_TRACKING_URI) or DEFAULT_MLFLOW_URI
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
