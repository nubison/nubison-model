# Container for nubison-model
It is a container for the nubison-model. It is based on `python:3.12-slim-bookworm` with `uv` and `mlflow-skinny`. It will run the nubison-model server for a given model URI.

## Build

```bash
docker build -t ghcr.io/nubison/nubison-model:latest .
```

## Push

```bash
docker push ghcr.io/nubison/nubison-model:latest
```

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
