from contextlib import contextmanager
from os import chdir, environ, getcwd, makedirs, path
from shutil import rmtree
from typing import List, Optional


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
def temporary_artifact_env(artifact_dirs: Optional[List[str]] = None):
    if artifact_dirs is not None:
        environ["ARTIFACT_DIRS"] = ",".join(artifact_dirs)
    yield
    environ["ARTIFACT_DIRS"] = ""


@contextmanager
def temporary_env(env: dict):
    for key, value in env.items():
        environ[key] = value
    yield
    for key in env.keys():
        del environ[key]
