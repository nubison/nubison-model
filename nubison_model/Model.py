from importlib.metadata import distributions
from os import getenv, path, symlink
from sys import version_info as py_version_info
from typing import Any, List, Optional, Protocol, TypedDict, runtime_checkable

import mlflow
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PythonModel

ENV_VAR_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_VAR_MLFLOW_MODEL_URI = "MLFLOW_MODEL_URI"
DEFAULT_MODEL_NAME = "Default"
DEAFULT_MLFLOW_URI = "http://127.0.0.1:5000"
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
            print("Skipping artifact preparation as it was already done.")
            return

        for name, target_path in artifacts.items():
            try:
                symlink(target_path, name, target_is_directory=path.isdir(target_path))
                print(f"Prepared artifact: {name} -> {target_path}")
            except OSError as e:
                print(f"Error creating symlink for {name}: {e}")

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


def _make_artifact_dir_dict(artifact_dirs: Optional[str]) -> dict:
    # Get the dict of artifact directories.
    # If not provided, read from environment variables, else use the default
    artifact_dirs_from_param_or_env = (
        artifact_dirs
        if artifact_dirs is not None
        else getenv("ARTIFACT_DIRS", DEFAULT_ARTIFACT_DIRS)
    )

    # Return a dict with the directory as both the key and value
    return {
        dir.strip(): dir.strip()
        for dir in artifact_dirs_from_param_or_env.split(",")
        if dir != ""
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
    """
    # Check if the model implements the Model protocol
    if not isinstance(model, NubisonModel):
        raise TypeError("The model must implement the Model protocol")

    # Get the model name and MLflow URI from environment variables if not provided
    if model_name is None:
        model_name = getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    if mlflow_uri is None:
        mlflow_uri = getenv(ENV_VAR_MLFLOW_TRACKING_URI, DEAFULT_MLFLOW_URI)

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
        model_info: ModelInfo = mlflow.pyfunc.log_model(
            registered_model_name=model_name,
            python_model=NubisonMLFlowModel(model),
            conda_env=_make_conda_env(),
            artifacts=_make_artifact_dir_dict(artifact_dirs),
            artifact_path="",
        )

        # Set tags on the registered model instead of the run
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
