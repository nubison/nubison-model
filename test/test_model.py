from os import path

import pytest
from mlflow.pyfunc import load_model
from mlflow.tracking import MlflowClient

from nubison_model import NubisonModel, register
from nubison_model.Model import _make_artifact_dir_dict, _package_list_from_file
from test.utils import temporary_cwd, temporary_dirs, temporary_env


def test_register_model(mlflow_server):
    """
    Test registering a model to MLflow's Model Registry.
    """
    model_name = "TestRegisteredModel"

    # define a simple model (for example purposes, using a dummy model)
    class DummyModel(NubisonModel):
        pass

    # configure the code directories
    artifact_dirs = ["src1", "src2"]
    with temporary_dirs(artifact_dirs), temporary_env(
        {"ARTIFACT_DIRS": ",".join(artifact_dirs)}
    ):
        # Register the model
        register(DummyModel(), model_name=model_name)

    client = MlflowClient()

    # assert that the model is registered
    registered_model = client.get_registered_model(model_name)
    assert registered_model.name == model_name

    # assert that the model has the correct code paths
    model_versions = client.get_latest_versions(model_name)
    artifact_path = client.download_artifacts(model_versions[0].run_id, "")
    for dir in artifact_dirs:
        assert path.exists(path.join(artifact_path, "artifacts", dir))

    # delete the registered model after the test
    client.delete_registered_model(model_name)


def test_throw_on_model_not_implementing_protocol(mlflow_server):
    """
    Test that the model class implements the Model protocol.
    """

    class WrongModel:
        pass

    class RightModel(NubisonModel):
        def load_model(self):
            pass

        def infer(self, input):
            pass

    with pytest.raises(TypeError):
        register(WrongModel())

    register(RightModel())


def test_package_list_from_file():
    """
    Test reading the package list from a requirements.txt file.
    """
    with temporary_cwd("test/fixtures"):
        packages = _package_list_from_file()
        assert packages == [
            "pandas==2.0.3",
            "scikit-learn>=1.3.2",
            "-e git+ssh://git@github.com/nubison/nubison-model.git",
            "package_name @ git+https://git.example.com/MyProject",
        ]


def test_artifact_dirs_from_env():
    """
    Test creating the artifact directories dictionary from the environment or parameter.
    """
    with temporary_env({"ARTIFACT_DIRS": ""}):
        assert _make_artifact_dir_dict(None) == {}
        assert _make_artifact_dir_dict("src, test") == {"src": "src", "test": "test"}

    with temporary_env({"ARTIFACT_DIRS": "src"}):
        assert _make_artifact_dir_dict(None) == {"src": "src"}
        assert _make_artifact_dir_dict("src,test") == {"src": "src", "test": "test"}


def test_log_params_and_metrics(mlflow_server):
    """
    Test logging parameters and metrics to MLflow.
    """
    model_name = "TestLoggedModel"

    class DummyModel(NubisonModel):
        pass

    # Test parameters and metrics
    test_params = {"param1": "value1", "param2": "value2"}
    test_metrics = {"metric1": 1.0, "metric2": 2.0}

    # Register model with params and metrics
    model_uri = register(
        DummyModel(),
        model_name=model_name,
        params=test_params,
        metrics=test_metrics,
    )

    # Extract run_id from model_uri (format: "runs:/run_id/path")
    run_id = model_uri.split("/")[1]

    # Get the run information from MLflow
    client = MlflowClient()
    run = client.get_run(run_id)

    assert set(test_params.items()) <= set(
        run.data.params.items()
    ), "Not all parameters were logged correctly"
    assert set(test_metrics.items()) <= set(
        run.data.metrics.items()
    ), "Not all metrics were logged correctly"
