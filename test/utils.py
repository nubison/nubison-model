from contextlib import contextmanager
from os import environ, getcwd, makedirs, path
from shutil import rmtree
from typing import List

from nubison_model.utils import temporary_cwd


def get_run_id_from_model_uri(model_uri: str) -> str:
    """
    Extracts the run_id from a given model_uri.

    Supports both legacy formats and mlflow 3.x LoggedModel URIs:
      - ``runs:/<run_id>/<path>`` → returns ``<run_id>``
      - ``models:/<name>/<version>`` → resolves via MlflowClient
      - ``models:/m-<logged_model_id>`` (mlflow 3.x LoggedModel URI)
        → resolves to the source run via MlflowClient

    :param model_uri: The URI of the model.
    :return: The extracted run_id.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    parts = model_uri.split("/")
    if model_uri.startswith("runs:/"):
        return parts[1]

    client = MlflowClient()
    if model_uri.startswith("models:/"):
        identifier = parts[1]
        # mlflow 3.x LoggedModel id starts with "m-"
        if identifier.startswith("m-"):
            logged_model = client.get_logged_model(identifier)
            return logged_model.source_run_id
        # Registered model: models:/<name>/<version>
        version = parts[2] if len(parts) > 2 else None
        if version is None:
            raise ValueError(f"Cannot extract run_id from {model_uri!r}")
        mv = client.get_model_version(identifier, version)
        return mv.run_id

    raise ValueError(f"Unsupported model_uri prefix: {model_uri!r}")


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
def temporary_env(env: dict):
    original_env = environ.copy()
    for key, value in env.items():
        environ[key] = value

    yield

    environ.clear()
    environ.update(original_env)


__all__ = [
    "temporary_cwd",
    "temporary_dirs",
    "temporary_env",
    "get_run_id_from_model_uri",
]
