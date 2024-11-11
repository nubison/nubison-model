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
        s.bind(("", 0))
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


@pytest.fixture(scope="session")
def mlflow_server():
    """
    Fixture to start and stop an MLflow server for testing.
    """
    # Create temporary directories for backend store and artifact store
    backend_store = tempfile.TemporaryDirectory()
    artifact_store = tempfile.TemporaryDirectory()

    # Define MLflow server URI and port
    port = find_free_port()
    mlflow_uri = f"http://127.0.0.1:{port}"
    os.environ[ENV_VAR_MLFLOW_TRACKING_URI] = mlflow_uri

    # Start MLflow server as a subprocess
    cmd = [
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

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait for the server to start
    timeout = 30  # seconds
    start_time = time.time()
    server_started = False
    while time.time() - start_time < timeout:
        try:
            response = requests.get(mlflow_uri)
            if response.status_code == 200:
                server_started = True
                break
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)

    if not server_started:
        # If the server didn't start, terminate the process and raise an error
        process.terminate()
        stdout, stderr = process.communicate()
        raise RuntimeError(
            f"Failed to start MLflow server:\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        )

    yield mlflow_uri  # Provide the tracking URI to the tests

    # Teardown: terminate the MLflow server and clean up
    terminate_process_tree(process.pid)

    # Optionally, capture and log any remaining output
    try:
        stdout, stderr = process.communicate(timeout=10)
        if stdout:
            print(f"MLflow server STDOUT:\n{stdout}")
        if stderr:
            print(f"MLflow server STDERR:\n{stderr}")
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        if stdout:
            print(f"MLflow server STDOUT after kill:\n{stdout}")
        if stderr:
            print(f"MLflow server STDERR after kill:\n{stderr}")
