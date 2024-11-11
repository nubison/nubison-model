from functools import wraps
from os import environ, getenv
from typing import Optional

import bentoml
from mlflow import set_tracking_uri
from mlflow.pyfunc import load_model

from nubison_model.Model import ENV_VAR_MLFLOW_MODEL_URI, ENV_VAR_MLFLOW_TRACKING_URI


def load_nubison_model(
    mlflow_tracking_uri,
    mlflow_model_uri,
    initialize: bool,
):

    try:
        if not mlflow_tracking_uri:
            raise RuntimeError("MLflow tracking URI is not set")
        if not mlflow_model_uri:
            raise RuntimeError("MLflow model URI is not set")
        set_tracking_uri(mlflow_tracking_uri)
        mlflow_model = load_model(
            model_uri=mlflow_model_uri, model_config={"initialize": initialize}
        )
        nubison_model = mlflow_model.unwrap_python_model().get_nubison_model()
    except Exception as e:
        raise RuntimeError(
            f"Error loading model(uri: {mlflow_model_uri}) from model registry(uri: {mlflow_tracking_uri})"
        ) from e

    return nubison_model


def build_inference_service(
    mlflow_tracking_uri: Optional[str] = None, mlflow_model_uri: Optional[str] = None
):
    mlflow_tracking_uri = (
        mlflow_tracking_uri or getenv(ENV_VAR_MLFLOW_TRACKING_URI) or ""
    )
    mlflow_model_uri = mlflow_model_uri or getenv(ENV_VAR_MLFLOW_MODEL_URI) or ""

    nubison_model_class = load_nubison_model(
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_model_uri=mlflow_model_uri,
        initialize=False,
    ).__class__

    @bentoml.service
    class BentoMLService:
        """BentoML Service for serving machine learning models."""

        _nubison_model = None

        def __init__(self):
            """Initializes the BentoML Service for serving machine learning models.

            This function retrieves a Nubison Model wrapped as an MLflow model
            The Nubison Model contains user-defined methods for performing inference.

            Raises:
                RuntimeError: Error loading model from the model registry
            """
            self._nubison_model = load_nubison_model(
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_model_uri=mlflow_model_uri,
                initialize=True,
            )

        @bentoml.api
        @wraps(nubison_model_class.infer)
        def infer(self, *args, **kwargs):
            """Proxy method to the NubisonModel.infer method

            Raises:
                RuntimeError: Error requested inference with no Model loaded

            Returns:
                _type_: The return type of the NubisonModel.infer method
            """

            if self._nubison_model is None:
                raise RuntimeError("Model is not loaded")

            return self._nubison_model.infer(*args, **kwargs)

    return BentoMLService


# Make BentoService if the script is loaded by BentoML
# This requires the running mlflow server and the model registered to the model registry
# The model registry URI and model URI should be set as environment variables
loaded_by_bentoml = any(var.startswith("BENTOML_") for var in environ)
if loaded_by_bentoml:
    InferenceService = build_inference_service()