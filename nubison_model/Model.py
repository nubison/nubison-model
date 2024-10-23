import mlflow

from os import getenv
from sys import version_info as py_version_info
from typing import Optional, List
from importlib.metadata import distributions
from mlflow.pyfunc import PythonModel

DEFAULT_MODEL_NAME = "nubison_model"
DEAFULT_MLFLOW_URI = "http://127.0.0.1:5000"
DEFAULT_CONDA_CHANNELS = "default"  # Default Conda channels comma-separated
DEFAULT_CODE_PATHS = "src"  # Default code paths comma-separated


class Model(PythonModel):
    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        pass


def register(
    model: Model,
    model_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    code_paths: Optional[List[str]] = None,
):
    # Get the model name and MLflow URI from environment variables if not provided
    if model_name is None:
        model_name = getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    if mlflow_uri is None:
        mlflow_uri = getenv("MLFLOW_TRACKING_URI", DEAFULT_MLFLOW_URI)
    if code_paths is None:
        code_paths = [
            channel.strip()
            for channel in getenv("CODE_PATHS", DEFAULT_CODE_PATHS).split(",")
        ]

    # Get the list of Conda channels
    conda_channels = [
        channel.strip()
        for channel in getenv("CONDA_CHANNELS", DEFAULT_CONDA_CHANNELS).split(",")
    ]
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
            python_model=model,
            conda_env={
                "channels": conda_channels,
                "dependencies": [
                    f"python={python_version}",
                    "pip",
                    {"pip": packages_list},
                ],
                "name": model_name,
            },
            registered_model_name=model_name,
            code_paths=code_paths,
        )
