#!/bin/bash
set -e

# Check if MLFLOW_TRACKING_URI is set
if [ -z "$MLFLOW_TRACKING_URI" ]; then
  echo "Error: MLFLOW_TRACKING_URI environment variable is not set."
  exit 1
fi

# Check if MLFLOW_TRACKING_URI is set
if [ -z "$MLFLOW_MODEL_URI" ]; then
  echo "Error: MLFLOW_MODEL_URI environment variable is not set."
  exit 1
fi

run_uuid=$(echo "$MLFLOW_MODEL_URI" | sed -E 's|runs:/([a-zA-Z0-9]+)/?$|\1|' || echo "")
ENV_NAME="nubison"

echo "MLFLOW_TRACKING_URI: '$MLFLOW_TRACKING_URI'"
echo "MLFLOW_MODEL_URI: '$MLFLOW_MODEL_URI'"
echo "NUM_WORKERS: '$NUM_WORKERS'"

echo "Downloading conda.yaml & requirements.txt for run $run_uuid"
ret_conda=$(curl -L -o conda.yaml -f "${MLFLOW_TRACKING_URI}/get-artifact?path=conda.yaml&run_uuid=${run_uuid}")
ret_req=$(curl -L -o requirements.txt -f "${MLFLOW_TRACKING_URI}/get-artifact?path=requirements.txt&run_uuid=${run_uuid}")

if [ "$ret_conda" -ne 200 ] || [ "$ret_req" -ne 200 ]; then
  echo "Failed to download conda.yaml or requirements.txt"
  exit 1
fi

python_version=$(grep "python=" conda.yaml | awk -F= '{print $2}')

echo "Starting BentoML server..."
exec uv run --python $python_version --with-requirements requirements.txt bentoml serve nubison_model.Service:InferenceService --port ${PORT} ${DEBUG:+--debug}