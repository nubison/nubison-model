# Container for nubison-model
It is a container for the nubison-model. It is based on Alpine Linux and Miniforge. It will run the nubison-model server for given model URI.

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
docker run host -e MLFLOW_TRACKING_URI=http://mlflow.host.uri -e MLFLOW_MODEL_URI=runs:/0b8ef6de308842e1ba40f0ae5481f9de/ -e PORT=3000 -e DEBUG=true ghcr.io/nubison/nubison-model:latest
```