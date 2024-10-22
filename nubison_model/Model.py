import mlflow

from os import getenv
from mlflow.pyfunc import PythonModel
from typing import Optional

DEFAULT_MODEL_NAME = "nubison_model"
DEAFULT_MLFLOW_URI = "http://127.0.0.1:5000"


class Model(PythonModel):
    def load_context(self, context):
        pass

    def predict(self, context, model_input):
        pass


def register(
    model: Model,
    model_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
):
    # Get the model name and MLflow URI from environment variables if not provided
    if model_name is None:
        model_name = getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    if mlflow_uri is None:
        mlflow_uri = getenv("MLFLOW_TRACKING_URI", DEAFULT_MLFLOW_URI)

    mlflow.set_tracking_uri(mlflow_uri)

    # Start a new MLflow run
    with mlflow.start_run(run_name=model_name) as run:
        # Log the model to MLflow
        mlflow.pyfunc.log_model(
            artifact_path="",
            python_model=model,
            conda_env={
                "channels": ["defaults"],
                "dependencies": ["python=3.8.5", "pip", {"pip": ["mlflow"]}],
                "name": "mlflow-env",
            },
            registered_model_name=model_name,
        )
