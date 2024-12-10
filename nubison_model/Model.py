from importlib.metadata import distributions
from os import getenv, path, symlink
from sys import version_info as py_version_info
from typing import Any, List, Optional, Protocol, runtime_checkable

import mlflow
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PythonModel

ENV_VAR_MLFLOW_TRACKING_URI = "MLFLOW_TRACKING_URI"
ENV_VAR_MLFLOW_MODEL_URI = "MLFLOW_MODEL_URI"
DEFAULT_MODEL_NAME = "Default"
DEAFULT_MLFLOW_URI = "http://127.0.0.1:5000"
DEFAULT_ARTIFACT_DIRS = ""  # Default code paths comma-separated


@runtime_checkable
class NubisonModel(Protocol):
    def load_model(self) -> None: ...

    def infer(self, input: Any) -> Any: ...


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

    def load_model(self):
        self._nubison_model.load_model()

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
):
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

        # Log the model to MLflow
        model_info: ModelInfo = mlflow.pyfunc.log_model(
            registered_model_name=model_name,
            python_model=NubisonMLFlowModel(model),
            conda_env=_make_conda_env(),
            artifacts=_make_artifact_dir_dict(artifact_dirs),
            artifact_path="",
        )

        return model_info.model_uri
