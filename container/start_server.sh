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

response_code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "${MLFLOW_TRACKING_URI}" || echo "000")
if [ "$response_code" -ne 200 ]; then
  echo "Failed to connect to the MLflow server"
  exit 1
else
  echo "Server is up"
fi

echo "Downloading conda.yaml & requirements.txt for run $run_uuid"
curl -L -o conda.yaml "${MLFLOW_TRACKING_URI}/get-artifact?path=conda.yaml&run_uuid=${run_uuid}"
curl -L -o requirements.txt "${MLFLOW_TRACKING_URI}/get-artifact?path=requirements.txt&run_uuid=${run_uuid}"

python_version=$(grep "python=" conda.yaml | awk -F= '{print $2}')

echo "Starting BentoML server..."
exec uv run --python $python_version --with-requirements requirements.txt bentoml serve nubison_model.Service:InferenceService --port ${PORT} ${DEBUG:+--debug}