import pytest

from os import environ, path, getcwd, makedirs, chdir
from contextlib import contextmanager
from typing import List
from shutil import rmtree

from mlflow.tracking import MlflowClient
from mlflow.pyfunc import load_model
from nubison_model.Model import Model, register, _package_list_from_file


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
                rmtree(dir)


@contextmanager
def temporary_cwd(new_dir):
    original_dir = getcwd()
    try:
        chdir(new_dir)
        yield
    finally:
        chdir(original_dir)


@contextmanager
def temporary_artifact_env(artifact_dirs):
    environ["ARTIFACT_DIRS"] = ",".join(artifact_dirs)
    yield
    environ["ARTIFACT_DIRS"] = ""


def test_register_model(mlflow_server):
    """
    Test registering a model to MLflow's Model Registry.
    """
    model_name = "TestRegisteredModel"

    # define a simple model (for example purposes, using a dummy model)
    class DummyModel(Model):
        pass

    # configure the code directories
    artifact_dirs = ["src1", "src2"]
    with temporary_dirs(artifact_dirs), temporary_artifact_env(artifact_dirs):
        # Register the model
        register(DummyModel(), model_name=model_name, mlflow_uri=mlflow_server)

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

    class RightModel(Model):
        def load_model(self):
            pass

        def infer(self, input):
            pass

    with pytest.raises(TypeError):
        register(WrongModel())

    register(RightModel())


def test_model_load_artifact_code(mlflow_server):
    """
    Test loading the artifact code paths.
    """
    model_name = "TestRegisteredModel"

    class DummyModel(Model):
        def load_model(self):
            # Try to read the contents of the artifact file
            with open("./fixtures/bar.txt", "r") as f:
                self.loaded = f.read()

        def infer(self, input):
            # Try to import a function from the artifact code
            from fixtures.poo import echo

            return echo(self.loaded + input)

    # Switch cwd to the current file directory to register the fixture artifact
    with temporary_cwd("test"):
        register(DummyModel(), model_name=model_name, artifact_dirs="fixtures")

    # Create temp dir and switch to it to test the model.
    # So artifact symlink not to coliide with the current directory
    with temporary_dirs(["infer"]), temporary_cwd("infer"):
        model = load_model(f"models:/{model_name}/latest")
        assert model.predict("test") == "bartest"


def test_package_list_from_file():
    """
    Test reading the package list from a requirements.txt file.
    """
    with temporary_cwd("test/fixtures"):
        packages = _package_list_from_file()
        assert packages == ["pandas==2.0.3", "scikit-learn==1.3.2"]
