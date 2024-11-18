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

echo "Downloading conda.yaml for run $run_uuid"
curl -L -o conda.yaml "${MLFLOW_TRACKING_URI}/get-artifact?path=conda.yaml&run_uuid=${run_uuid}"

# Create the Conda environment
echo "Creating Conda environment from conda.yaml"
conda env create -f conda.yaml -n $ENV_NAME

echo "Activating Conda environment '$ENV_NAME'"
# Initialize Conda for bash
source ${CONDA_DIR}/etc/profile.d/conda.sh

# Activate the environment
conda activate "$ENV_NAME"

# Verify that the environment is activated
echo "Conda environment '$ENV_NAME' is activated"

# Execute the BentoML serve command
echo "Starting BentoML server..."
exec bentoml serve nubison_model.Service:InferenceService --port ${PORT} ${DEBUG:+--debug}