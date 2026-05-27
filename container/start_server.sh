#!/bin/bash
set -e

log() { echo "[nubison-model] $(date '+%H:%M:%S') $*"; }
err() { echo "[nubison-model] $(date '+%H:%M:%S') ERROR: $*" >&2; }

# --- Validate required env vars ---
# Note: MLFLOW_MODEL_URI is constructed from NAME+VERSION below (models:/ form),
# so it is not required as input.
missing=()
[ -z "$MLFLOW_TRACKING_URI" ]  && missing+=("MLFLOW_TRACKING_URI")
[ -z "$MLFLOW_MODEL_NAME" ]    && missing+=("MLFLOW_MODEL_NAME")
[ -z "$MLFLOW_MODEL_VERSION" ] && missing+=("MLFLOW_MODEL_VERSION")

if [ ${#missing[@]} -gt 0 ]; then
  err "Missing required env vars: ${missing[*]}"
  exit 1
fi

# Build models:/ URI used both for artifact download and BentoML model loading.
# Ignore any inbound MLFLOW_MODEL_URI (e.g. legacy runs:/ form) — MLflow 3.x
# stores pyfunc artifacts under a LoggedModel entity and runs:/ is deprecated.
export MLFLOW_MODEL_URI="models:/${MLFLOW_MODEL_NAME}/${MLFLOW_MODEL_VERSION}"

log "=== Configuration ==="
log "  MLFLOW_TRACKING_URI:  $MLFLOW_TRACKING_URI"
log "  MLFLOW_MODEL_URI:     $MLFLOW_MODEL_URI"
log "  MLFLOW_MODEL_NAME:    $MLFLOW_MODEL_NAME"
log "  MLFLOW_MODEL_VERSION: $MLFLOW_MODEL_VERSION"
log "  NUM_WORKERS:          ${NUM_WORKERS:-2}"
log "  PORT:                 ${PORT:-3000}"

# --- Download conda.yaml & requirements.txt via MLflow API ---
log "Downloading dependencies from $MLFLOW_MODEL_URI ..."

python_version=$(python -c "
import os, sys, time, shutil, yaml
from mlflow.artifacts import download_artifacts

artifact_uri = os.environ['MLFLOW_MODEL_URI']
max_retries = 3
files = ['/conda.yaml', '/requirements.txt']

for attempt in range(1, max_retries + 1):
    try:
        paths = {}
        for f in files:
            uri = artifact_uri + f
            print(f'  Downloading {f[1:]} ...', file=sys.stderr, flush=True)
            paths[f] = download_artifacts(artifact_uri=uri)
        break
    except Exception as e:
        if attempt == max_retries:
            print(f'ERROR: Download failed after {max_retries} attempts: {e}', file=sys.stderr)
            sys.exit(1)
        wait = attempt * 5
        print(f'  Attempt {attempt}/{max_retries} failed: {e}', file=sys.stderr)
        print(f'  Retrying in {wait}s ...', file=sys.stderr, flush=True)
        time.sleep(wait)

for f, src in paths.items():
    shutil.copy(src, '/app' + f)
print('  Dependencies downloaded.', file=sys.stderr, flush=True)

with open('/app/conda.yaml') as f:
    env = yaml.safe_load(f)
for dep in env.get('dependencies', []):
    if isinstance(dep, str) and dep.startswith('python='):
        print(dep.split('=')[1])
        break
else:
    print('ERROR: python version not found in conda.yaml dependencies', file=sys.stderr)
    sys.exit(1)
")

if [ -z "$python_version" ]; then
  err "Failed to extract Python version from conda.yaml"
  exit 1
fi

# major.minor only (e.g. 3.11.12 -> 3.11) — uv doesn't ship every patch version.
# Use cut to be safe when conda.yaml has only major.minor (e.g. "3.11").
python_minor=$(echo "$python_version" | cut -d. -f1,2)

log "Detected Python $python_version (using $python_minor)"
log "Installing Python $python_minor ..."
uv python install "$python_minor"

# --- Start BentoML server ---
log "Starting BentoML server on port ${PORT:-3000} ..."
exec uv run --python "$python_minor" --with-requirements requirements.txt bentoml serve nubison_model.Service:InferenceService --port "${PORT}" ${DEBUG:+--debug}
