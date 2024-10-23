from os import environ, path, getcwd, makedirs, rmdir
from contextlib import contextmanager
from typing import List

from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from nubison_model.Model import Model, register


@contextmanager
def temporary_dirs(dirs: List[str]):
    dirs = [path.join(getcwd(), src_dir) for src_dir in dirs]

    try:
        for dir in dirs:
            makedirs(dir, exist_ok=True)

        yield dirs

    finally:
        for dir in dirs:
            if path.exists(dir):
                rmdir(dir)


def test_register_model(mlflow_server):
    """
    Test registering a model to MLflow's Model Registry.
    """
    model_name = "TestRegisteredModel"

    # define a simple model (for example purposes, using a dummy model)
    class DummyModel(Model):
        pass

    # configure the code directories
    code_dirs = ["src1", "src2"]
    environ["CODE_PATHS"] = ",".join(code_dirs)
    with temporary_dirs(code_dirs):
        # Register the model
        register(DummyModel(), model_name=model_name, mlflow_uri=mlflow_server)

    client = MlflowClient()

    # assert that the model is registered
    registered_model = client.get_registered_model(model_name)
    assert registered_model.name == model_name

    # assert that the model has the correct code paths
    model_versions = client.get_latest_versions(model_name)
    artifact_path = client.download_artifacts(model_versions[0].run_id, "")
    for dir in code_dirs:
        assert path.exists(path.join(artifact_path, "code", dir))

    # delete the registered model after the test
    client.delete_registered_model(model_name)
