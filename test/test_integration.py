from nubison_model import (
    ENV_VAR_MLFLOW_MODEL_URI,
    NubisonModel,
    build_inference_service,
    register,
)
from test.utils import temporary_cwd, temporary_dirs, temporary_env


def test_register_and_serve_model(mlflow_server):
    """
    Test registering a model to MLflow's Model Registry and serving it with BentoML.
    """

    class DummyModel(NubisonModel):
        def load_model(self):
            # Try to read the contents of the artifact file
            with open("./fixtures/bar.txt", "r") as f:
                self.loaded = f.read()

        def infer(self, param1: str):
            # Try to import a function from the artifact code
            from .fixtures.poo import echo

            return echo(self.loaded + param1)

    # Switch cwd to the current file directory to register the fixture artifact
    with temporary_cwd("test"):
        model_uri = register(DummyModel(), artifact_dirs="fixtures")

    # Create temp dir and switch to it to test the model.
    # So artifact symlink not to coliide with the current directory
    with temporary_dirs(["infer"]), temporary_cwd("infer"), temporary_env(
        {ENV_VAR_MLFLOW_MODEL_URI: model_uri}
    ):
        bento_service = build_inference_service()()
        assert bento_service.infer("test") == "bartest"
