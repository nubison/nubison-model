from contextlib import contextmanager
from os import chdir, getcwd


@contextmanager
def temporary_cwd(new_dir):
    original_dir = getcwd()
    try:
        chdir(new_dir)
        yield
    finally:
        chdir(original_dir)
