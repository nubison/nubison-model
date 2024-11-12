from importlib.metadata import distributions
from os import getenv, path
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
DEFAULT_MODEL_CONFIG = {"initialize": True}


@runtime_checkable
class NubisonModel(Protocol):
    def load_model(self) -> None: ...

    def infer(self, input: Any) -> Any: ...


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


def _make_mlflow_model(nubison_model: NubisonModel) -> PythonModel:
    class NubisonMLFlowModel(PythonModel):
        _nubison_model: NubisonModel = nubison_model

        def load_context(self, context):
            """Make the MLFlow artifact is accessible to the model in the same way as in the local environment

            Args:
                context (PythonModelContext): A collection of artifacts that a PythonModel can use when performing inference.
            """
            from os import path, symlink

            load_model = context.model_config.get("initialize", True)
            if not load_model:
                return

            for name, target_path in context.artifacts.items():
                # Create the symbolic link with the key as the symlink name
                try:
                    symlink(
                        target_path, name, target_is_directory=path.isdir(target_path)
                    )
                    print(f"Prepared artifact: {name} -> {target_path}")
                except OSError as e:
                    print(f"Error creating symlink for {name}: {e}")

            self._nubison_model.load_model()

        def predict(self, context, model_input):
            input = model_input["input"]
            return self._nubison_model.infer(**input)

        def get_nubison_model(self):
            return self._nubison_model

    return NubisonMLFlowModel()


def register(
    model: NubisonModel,
    model_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    artifact_dirs: Optional[str] = None,
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
        # Log the model to MLflow
        model_info: ModelInfo = mlflow.pyfunc.log_model(
            registered_model_name=model_name,
            python_model=_make_mlflow_model(model),
            conda_env=_make_conda_env(),
            artifacts=_make_artifact_dir_dict(artifact_dirs),
            model_config=DEFAULT_MODEL_CONFIG,
            artifact_path="",
        )

        return model_info.model_uri
