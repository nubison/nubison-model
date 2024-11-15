import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path

import psutil
import pytest
import requests

from nubison_model import ENV_VAR_MLFLOW_TRACKING_URI


def find_free_port():
    """
    Finds a free port on localhost.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def terminate_process_tree(pid, timeout=10):
    """
    Terminates a process and all its child processes.
    """
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        gone, alive = psutil.wait_procs(children + [parent], timeout=timeout)
        for p in alive:
            p.kill()
    except psutil.NoSuchProcess:
        pass


def _wait_ready(uri, timeout=30):
    start_time = time.time()
    server_started = False
    while time.time() - start_time < timeout:
        try:
            response = requests.get(uri)
            if response.status_code == 200:
                server_started = True
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)

    return server_started


def _mlflow_cmd(port, backend_store, artifact_store):
    return [
        "mlflow",
        "server",
        "--backend-store-uri",
        f"sqlite:///{Path(backend_store.name) / 'mlflow.db'}",
        "--default-artifact-root",
        artifact_store.name,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]


@pytest.fixture(scope="session")
def mlflow_server():
    """
    Fixture to start and stop an MLflow server for testing.
    """
    # Prepare temporary directories and find a free port
    backend_store = tempfile.TemporaryDirectory()
    artifact_store = tempfile.TemporaryDirectory()
    port = find_free_port()
    uri = f"http://127.0.0.1:{port}"
    cmd = _mlflow_cmd(port, backend_store, artifact_store)

    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ) as process:
        try:
            server_started = _wait_ready(uri, timeout=30)
            if not server_started:
                raise RuntimeError(f"Failed to start server {uri}")

            os.environ[ENV_VAR_MLFLOW_TRACKING_URI] = uri

            yield uri  # Provide the tracking URI to the tests

        finally:
            # Teardown: terminate the MLflow server and clean up
            terminate_process_tree(process.pid)
