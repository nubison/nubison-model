import logging
from importlib.metadata import distributions
from os import getenv, makedirs, path, symlink
from sys import version_info as py_version_info
from typing import Any, List, Optional, Protocol, TypedDict, runtime_checkable

import mlflow
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PythonModel

from nubison_model.Storage import (
    DVC_FILES_TAG_KEY,
    find_weight_files,
    get_size_threshold,
    is_dvc_enabled,
    iter_artifact_files,
    push_to_dvc,
    serialize_dvc_info,
    should_use_dvc,
)

logger = logging.getLogger(__name__)

ENV_VAR_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_VAR_MLFLOW_MODEL_URI = "MLFLOW_MODEL_URI"
DEFAULT_MODEL_NAME = "Default"
DEFAULT_MLFLOW_URI = "http://127.0.0.1:5000"
DEFAULT_ARTIFACT_DIRS = ""  # Default code paths comma-separated


class ModelContext(TypedDict):
    """Context information passed to model during loading.

    Attributes:
        worker_index: Index of the worker process running the model. Used to identify
            which worker is running the model in a parallel server setup. Starts from 0.
            Even in a single server process setup, this will be 0. This is particularly
            useful for GPU initialization as you can map specific workers to specific
            GPU devices.
        num_workers: Number of workers running the model.
    """

    worker_index: int

    num_workers: int


@runtime_checkable
class NubisonModel(Protocol):
    """Protocol defining the interface for user-defined models.

    Your model class must implement this protocol by providing:
    1. load_model method - Called once at startup to initialize the model
    2. infer method - Called for each inference request
    """

    def load_model(self, context: ModelContext) -> None:
        """Initialize and load the model.

        This method is called once when the model server starts up.
        Use it to load model weights and initialize any resources needed for inference.

        Args:
            context: A dictionary containing worker information:
                - worker_index: Index of the worker process (0-based)
                - num_workers: Total number of workers running the model
                This information is particularly useful for GPU initialization
                in parallel setups, where you can map specific workers to
                specific GPU devices.
        """
        ...

    def infer(self, input: Any) -> Any:
        """Perform inference on the input.

        This method is called for each inference request.

        Args:
            input: The input data to perform inference on.
                Can be of any type that your model accepts.

        Returns:
            The inference result. Can be of any type that your model produces.
        """
        ...


class NubisonMLFlowModel(PythonModel):
    def __init__(self, nubison_model: NubisonModel):
        self._nubison_model = nubison_model

    def _check_artifacts_prepared(self, artifacts: dict) -> bool:
        """Check if all symlinks for the artifacts are created successfully."""
        for name, target_path in artifacts.items():
            if not path.exists(name):
                return False
        return True

    def prepare_artifacts(self, artifacts: dict) -> None:
        """Create symbolic links for the artifacts provided as a parameter."""
        if self._check_artifacts_prepared(artifacts):
            logger.debug("Skipping artifact preparation as it was already done.")
            return

        for name, target_path in artifacts.items():
            try:
                parent_dir = path.dirname(name)
                if parent_dir:
                    makedirs(parent_dir, exist_ok=True)
                # Check if target is a directory; default to False if path doesn't exist (e.g., broken symlink)
                is_dir = path.isdir(target_path) if path.exists(target_path) else False
                symlink(target_path, name, target_is_directory=is_dir)
                logger.debug(f"Prepared artifact: {name} -> {target_path}")
            except FileExistsError:
                pass
            except OSError as e:
                logger.error(f"Error creating symlink for {name}: {e}")

    def load_context(self, context: Any) -> None:
        """Make the MLFlow artifact accessible to the model in the same way as in the local environment

        Args:
            context (PythonModelContext): A collection of artifacts that a PythonModel can use when performing inference.
        """
        self.prepare_artifacts(context.artifacts)

    def predict(self, context, model_input):
        input = model_input["input"]
        return self._nubison_model.infer(**input)

    def get_nubison_model(self):
        return self._nubison_model

    def load_model(self, context: ModelContext):
        self._nubison_model.load_model(context)

    def infer(self, *args, **kwargs) -> Any:
        return self._nubison_model.infer(*args, **kwargs)

    def get_nubison_model_infer_method(self):
        return self._nubison_model.__class__.infer


def _is_shareable(package: str) -> bool:
    # Nested requirements, constraints files, local packages, and comments are not supported
    if package.startswith(("-r", "-c", "-e .", "-e /", "/", ".", "#")):
        return False
    # Check if the package is a local package
    # eg. git+file:///path/to/repo.git, file:///path/to/repo, -e file:///
    if "file:" in package:
        return False

    return True


def _package_list_from_file() -> Optional[List]:
    # Check if the requirements file exists in order of priority
    candidates = ["requirements-prod.txt", "requirements.txt"]
    filename = next((file for file in candidates if path.exists(file)), None)

    if filename is None:
        return None

    with open(filename, "r") as file:
        packages = file.readlines()
    packages = [package.strip() for package in packages if package.strip()]
    # Remove not sharable dependencies
    packages = [package for package in packages if _is_shareable(package)]

    return packages


def _package_list_from_env() -> List:
    # Get the list of installed packages
    return [
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in distributions()
        if dist.metadata["Name"]
        is not None  # editable installs have a None metadata name
    ]


def _make_conda_env() -> dict:
    # Get the Python version
    python_version = (
        f"{py_version_info.major}.{py_version_info.minor}.{py_version_info.micro}"
    )
    # Get the list of installed packages from the requirements file or environment
    packages_list = _package_list_from_file() or _package_list_from_env()

    return {
        "dependencies": [
            f"python={python_version}",
            "pip",
            {"pip": packages_list},
        ],
    }


def _make_artifact_dir_dict(
    artifact_dirs: Optional[str],
    exclude_weight_files: bool = False,
    size_threshold: Optional[int] = None,
) -> dict:
    """
    Get the dict of artifact directories.

    Args:
        artifact_dirs: Comma-separated list of directories/files to include
        exclude_weight_files: If True, exclude weight files (.pt, .bin, etc.) that exceed size threshold
        size_threshold: Minimum file size in bytes for DVC (default: from env or 100MB)

    Returns:
        Dictionary mapping artifact names to their paths
    """
    artifact_dirs_str = (
        artifact_dirs
        if artifact_dirs is not None
        else getenv("ARTIFACT_DIRS", DEFAULT_ARTIFACT_DIRS)
    )

    if not artifact_dirs_str:
        return {}

    if size_threshold is None:
        size_threshold = get_size_threshold()

    # Fast path: no exclusion needed, return directories as-is
    if not exclude_weight_files:
        return {
            entry.strip(): entry.strip()
            for entry in artifact_dirs_str.split(",")
            if entry.strip()
        }

    # With exclusion: iterate files and filter, preserving directory structure
    # Use path.dirname(base_dir) to keep the base_dir name in the artifact path
    # e.g., "src/model.py" instead of just "model.py"
    return {
        path.join(path.basename(base_dir), path.relpath(full_path, base_dir)): full_path
        for full_path, base_dir in iter_artifact_files(artifact_dirs_str)
        if not should_use_dvc(full_path, size_threshold)
    }


def register(
    model: NubisonModel,
    model_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    artifact_dirs: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    metrics: Optional[dict[str, float]] = None,
    tags: Optional[dict[str, str]] = None,
):
    """Register a model with MLflow.

    When DVC is enabled (DVC_ENABLED=true), large weight files (.pt, .bin, .pkl, etc.)
    that exceed the size threshold are automatically uploaded to DVC remote storage
    instead of MLflow. The DVC file hashes are stored in MLflow tags for retrieval
    during serving.

    Args:
        model: The model to register, must implement NubisonModel protocol
        model_name: Name to register the model under. Defaults to env var MODEL_NAME or 'Default'
        mlflow_uri: MLflow tracking URI. Defaults to env var MLFLOW_TRACKING_URI or local URI
        artifact_dirs: Comma-separated list of directories to include as artifacts
        params: Optional dictionary of parameters to log
        metrics: Optional dictionary of metrics to log
        tags: Optional dictionary of tags to log with the model registration

    Returns:
        str: The URI of the registered model

    Raises:
        TypeError: If the model doesn't implement the NubisonModel protocol
        DVCPushError: If DVC push fails for any weight file

    Environment Variables:
        DVC_ENABLED: Set to 'true' to enable DVC for large files
        DVC_REMOTE_URL: URL of DVC remote storage (required when DVC is enabled)
        DVC_SIZE_THRESHOLD: Minimum file size in bytes for DVC (default: 100MB)
    """
    # Check if the model implements the Model protocol
    if not isinstance(model, NubisonModel):
        raise TypeError("The model must implement the NubisonModel protocol")

    # Get the model name and MLflow URI from environment variables if not provided
    if model_name is None:
        model_name = getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    if mlflow_uri is None:
        mlflow_uri = getenv(ENV_VAR_MLFLOW_TRACKING_URI, DEFAULT_MLFLOW_URI)

    tags = dict(tags) if tags else {}

    dvc_enabled = is_dvc_enabled()
    dvc_info = {}
    size_threshold = get_size_threshold()

    if dvc_enabled and artifact_dirs:
        weight_files = find_weight_files(artifact_dirs, size_threshold)

        if weight_files:
            logger.info(
                f"DVC enabled: Found {len(weight_files)} weight file(s) to upload via DVC "
                f"(threshold: {size_threshold / (1024*1024):.0f} MB)"
            )
            for wf in weight_files:
                file_size = path.getsize(wf) / (1024 * 1024)
                logger.info(f"  - {wf} ({file_size:.1f} MB)")

            dvc_info = push_to_dvc(weight_files, fail_fast=True)
            tags[DVC_FILES_TAG_KEY] = serialize_dvc_info(dvc_info)
            logger.info(f"DVC: Uploaded {len(dvc_info)} file(s) to remote storage")

    # Set the MLflow tracking URI and experiment
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(model_name)

    # Start a new MLflow run
    with mlflow.start_run() as run:
        # Log parameters and metrics
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if tags:
            mlflow.set_tags(tags)

        # Log the model to MLflow
        # Always use folder structure to maintain consistent artifact paths
        model_info: ModelInfo = mlflow.pyfunc.log_model(
            registered_model_name=model_name,
            python_model=NubisonMLFlowModel(model),
            conda_env=_make_conda_env(),
            artifacts=_make_artifact_dir_dict(artifact_dirs),
            artifact_path="",
        )

        # Set tags on the registered model version
        if tags:
            client = mlflow.tracking.MlflowClient()
            for tag_name, tag_value in tags.items():
                client.set_model_version_tag(
                    model_name,
                    str(model_info.registered_model_version),
                    tag_name,
                    tag_value,
                )

        return model_info.model_uri
