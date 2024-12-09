from contextlib import contextmanager
from os import environ, getcwd, makedirs, path
from shutil import rmtree
from typing import List

from nubison_model.utils import temporary_cwd


def get_run_id_from_model_uri(model_uri: str) -> str:
    """
    Extracts the run_id from a given model_uri.

    :param model_uri: The URI of the model.
    :return: The extracted run_id.
    """
    return model_uri.split("/")[1]


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
