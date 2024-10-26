import mlflow

from os import getenv
from sys import version_info as py_version_info
from typing import Optional, Protocol, runtime_checkable
from importlib.metadata import distributions
from mlflow.pyfunc import PythonModel

DEFAULT_MODEL_NAME = "nubison_model"
DEAFULT_MLFLOW_URI = "http://127.0.0.1:5000"
DEFAULT_ARTIFACT_DIRS = ""  # Default code paths comma-separated


@runtime_checkable
class Model(Protocol):
    def load_model(self) -> None: ...

    def infer(self, input: any) -> any: ...


def _make_mlflow_model(nubison_model: Model) -> PythonModel:

    class MLflowModel(PythonModel):
        def load_context(self, context):
            from os import path, symlink

            for name, target_path in context.artifacts.items():
                # Create the symbolic link with the key as the symlink name
                try:
                    symlink(
                        target_path, name, target_is_directory=path.isdir(target_path)
                    )
                    print(f"Created symlink: {name} -> {target_path}")
                except OSError as e:
                    print(f"Error creating symlink for {name}: {e}")

            nubison_model.load_model()

        def predict(self, context, model_input):
            return nubison_model.infer(model_input)

    return MLflowModel()


def register(
    model: Model,
    model_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    artifact_dirs: Optional[str] = None,
):
    # Check if the model implements the Model protocol
    if not isinstance(model, Model):
        raise TypeError("The model must implement the Model protocol")

    # Get the model name and MLflow URI from environment variables if not provided
    if model_name is None:
        model_name = getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    if mlflow_uri is None:
        mlflow_uri = getenv("MLFLOW_TRACKING_URI", DEAFULT_MLFLOW_URI)
    # Get the list of artifact directories.
    # If not provided, read from environment variables, else use the default
    artifact_dirs = artifact_dirs or getenv("ARTIFACT_DIRS", DEFAULT_ARTIFACT_DIRS)
    artifact_dirs = {
        dir.strip(): dir.strip() for dir in artifact_dirs.split(",") if dir != ""
    }

    # Get the Python version
    python_version = (
        f"{py_version_info.major}.{py_version_info.minor}.{py_version_info.micro}"
    )
    # Get the list of installed packages
    packages_list = sorted(
        [
            f"{dist.metadata['Name']}=={dist.version}"
            for dist in distributions()
            if dist.metadata["Name"]
            is not None  # editable installs have a None metadata name
        ]
    )

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(model_name)

    # Start a new MLflow run
    with mlflow.start_run() as run:
        # Log the model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path="",
            python_model=_make_mlflow_model(model),
            conda_env={
                "dependencies": [
                    f"python={python_version}",
                    "pip",
                    {"pip": packages_list},
                ],
                "name": model_name,
            },
            registered_model_name=model_name,
            artifacts=artifact_dirs,
        )
