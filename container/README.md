# Container for nubison-model
It is a container for the nubison-model. It runs the nubison-model server for a given model URI, using `uv` and `mlflow-skinny` to download the model's dependencies and build the serving venv at runtime.

Two variants are published:

| Variant | Dockerfile | Base image | Tags |
|---------|------------|------------|------|
| CPU (default) | `Dockerfile` | `python:3.12-slim-bookworm` | `:<version>`, `:<version>-cpu`, `:latest`, `:cpu` |
| GPU (CUDA) | `Dockerfile.gpu` | `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04` | `:<version>-gpu`, `:latest-gpu`, `:gpu` |

CPU keeps the **unsuffixed** tag (`:<version>`, `:latest`) for backward compatibility with existing consumers. The GPU variant is identical except for the CUDA base — torch/TF are **not** baked into either image; `start_server.sh` installs the model's own `requirements.txt` with `uv` at runtime. The GPU base only supplies the CUDA runtime + cuDNN system libraries and the NVIDIA container-runtime hooks so a CUDA-enabled torch wheel from the model's requirements can reach the GPU (the container must be scheduled with the NVIDIA runtime / a `nvidia.com/gpu` resource).

## Build

```bash
# CPU
docker build -t ghcr.io/nubison/nubison-model:latest -f Dockerfile .
# GPU
docker build -t ghcr.io/nubison/nubison-model:latest-gpu -f Dockerfile.gpu .
```

## Push

```bash
docker push ghcr.io/nubison/nubison-model:latest       # CPU
docker push ghcr.io/nubison/nubison-model:latest-gpu   # GPU
```

Both variants are normally built and pushed by CI (`.github/workflows/container-image.yml`) on a published GitHub release (matrix over `cpu`/`gpu`), or on demand via `workflow_dispatch`.

## Run

```bash
docker run \
  -e MLFLOW_TRACKING_URI=http://mlflow.host.uri \
  -e MLFLOW_MODEL_URI=runs:/0b8ef6de308842e1ba40f0ae5481f9de/model \
  -e MLFLOW_MODEL_NAME=MyModel \
  -e MLFLOW_MODEL_VERSION=1 \
  -e PORT=3000 \
  -e DEBUG=true \
  ghcr.io/nubison/nubison-model:latest
```

## Environment variables

| Name | Required | Description |
|------|----------|-------------|
| `MLFLOW_TRACKING_URI` | yes | MLflow tracking server URI |
| `MLFLOW_MODEL_URI` | yes | Model URI used by BentoML to load the model (e.g. `runs:/<run_id>/model` or `models:/<name>/<version>`) |
| `MLFLOW_MODEL_NAME` | yes | Registered model name — used to fetch `conda.yaml` / `requirements.txt` via `models:/<name>/<version>` (MLflow 3.x stores them under LoggedModel entity) |
| `MLFLOW_MODEL_VERSION` | yes | Registered model version |
| `PORT` | no (default `3000`) | BentoML HTTP server port |
| `NUM_WORKERS` | no (default `2`) | BentoML worker count |
| `DEBUG` | no | If set, passes `--debug` to `bentoml serve` |
