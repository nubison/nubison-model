# Nubison

This project is a SDK for integrate ML model to Nubison.

## Usage

### Load training data

`nubison_model.data.load(uri)` fetches a DataFrame from `s3://` or
`file://` URIs (a bare local path is also accepted). The source URI is
recorded in `df.attrs["source_uri"]` so a downstream `train()` call
picks it up as MLflow dataset lineage — you do not name the URI twice.

```python
from nubison_model import data

# S3 (boto3; honors AWS_ENDPOINT_URL env for STS WebIdentity)
train_df = data.load("s3://my-bucket/datasets/train.parquet")
eval_df = data.load("s3://my-bucket/datasets/eval.parquet")

# Local file — bare path or file:// URI (csv / parquet by extension)
train_df = data.load("/home/jovyan/data/train.csv")
train_df = data.load("file:///home/jovyan/data/train.csv")
```

> Loading from a database by raw URI is intentionally not exposed — it
> would embed credentials into notebook cells. Use
> `data.connection(name)` (below) to reuse a JupyterLab SQL Explorer
> connection instead.

### Load from a saved DB connection (JupyterLab SQL Explorer)

`data.connection(name)` binds to a DB connection that's already saved in
the notebook's JupyterLab SQL Explorer side panel — no host/credential
retyping. Lookup order: env `DB_<NAME>` (Pod-injected) →
`~/.local/share/jupyterlab-sql-explorer/db_conf.json`.

```python
db = data.connection("MYDB")

train_df = db.load("SELECT * FROM features WHERE date >= '2026-01-01'")
eval_df  = db.load("SELECT * FROM features WHERE date >= '2026-05-01'")
# df.attrs["source_uri"] == "dbref://MYDB#<query_hash>"
```

Each `db.load()` call gets a distinct lineage identifier (the query
hash), so MLflow tracks every training-data query as a separate dataset
input. The password never appears in the recorded `source_uri`.

### Auto-split into training subsets

`nubison_model.data.split(df, ratios)` shuffles (optional) and slices a
single DataFrame into named subsets ready to hand to `train()`. Each
output's `attrs["source_uri"]` is derived from `source_prefix` (if
given), the input's own `source_uri`, or `memory://<key>`.

```python
full = data.load("s3://my-bucket/datasets/all.parquet")

datasets = data.split(
    full,
    {"training": 0.8, "evaluation": 0.2},
    shuffle=True,
    random_state=42,
)
# datasets["training"].attrs["source_uri"]
#   == "s3://my-bucket/datasets/all.parquet#training"
```

### Train a model (no mlflow boilerplate)

`train()` is a one-line sklearn-like API that handles tracking URI,
experiment selection, `autolog`, `start_run` with system metrics,
per-dataset `log_input` lineage, notebook source / git tags
(best-effort), and pickling the fitted estimator to `src/weights.pkl`.
The user does not touch the mlflow API.

```python
from nubison_model import data, train
from sklearn.linear_model import LogisticRegression

full = data.load("s3://my-bucket/all.parquet")
datasets = data.split(full, {"training": 0.8, "evaluation": 0.2}, random_state=42)

run_id = train(
    estimator=LogisticRegression(max_iter=500),
    datasets=datasets,
    target=["target"],
    model_type="classifier",
    artifact_dirs="src",
    extra_params={"feature_version": "v3"},
    extra_tags={"team": "ds"},
)
```

For frameworks with extra `fit` arguments (xgboost early stopping, Keras
epochs), use `fit_kwargs`. The magic string `"evaluation"` is resolved
to `(X_eval, y_eval)` from the matching `datasets` key:

```python
from xgboost import XGBClassifier

train(
    estimator=XGBClassifier(n_estimators=500),
    datasets=datasets,
    target=["target"],
    fit_kwargs={"eval_set": "evaluation", "early_stopping_rounds": 20},
)
```

PyTorch follows the same pattern via a sklearn-compatible wrapper
(`skorch`, `pytorch-lightning`, `fastai`, etc):

```python
from skorch import NeuralNetClassifier

train(
    estimator=NeuralNetClassifier(MyTorchModule, max_epochs=10, lr=0.01),
    datasets=datasets,
    target=["target"],
)
```

`train()` logs the run but **does not** package the inference code as a
`pyfunc` model — that is `register()`'s job (see below). This keeps the
training notebook and the inference packaging step independent.

### Register a model (default)

Logs the experiment and creates a Model Registry entry in one call.

```python
from nubison_model import register, NubisonModel, ModelContext


class MyModel(NubisonModel):
    def load_model(self, context: ModelContext):
        ...

    def infer(self, input):
        ...


model_uri = register(
    MyModel(),
    model_name="MyModel",
    artifact_dirs="src,weights",
    params={"lr": 0.01, "epochs": 100},
    metrics={"accuracy": 0.95},
)
# model_uri == "models:/MyModel/<version>"
```

### Log experiment only (skip Model Registry)

Pass `skip_model_registration=True` to log the experiment and package the
model as a run artifact, without creating a Model Registry entry. The
returned URI has a `runs:/` prefix and can later be registered via
`mlflow.register_model()`.

```python
run_uri = register(
    MyModel(),
    model_name="MyModel",
    artifact_dirs="src,weights",
    params={"lr": 0.01},
    metrics={"accuracy": 0.95},
    skip_model_registration=True,
)
# run_uri == "runs:/<run_id>/model"

# Later, review results and register when ready:
import mlflow

mlflow.register_model(run_uri, "MyModel")
```

The default is `skip_model_registration=False`, which preserves the original
behavior.

## Migration notes — v0.0.10 (mlflow 3.x bump)

The `mlflow` client pin is bumped from `^2.17` to `^3.0` so it matches a
3.x mlflow server (mlplatform K8s runs `mlflow/mlflow:v3.12`).

User-visible behavior is unchanged — `register()` keeps its signature
and still returns the legacy URI shape (`models:/<name>/<version>` or
`runs:/<run_id>/model`). Two side effects of mlflow 3.x to be aware of:

- **Artifacts are uploaded twice per `register()` call.** mlflow 3.x stores
  `pyfunc.log_model(artifacts=...)` under a per-run "LoggedModel"
  entity, leaving the run's `artifacts/` folder empty. To preserve BC
  for consumers that call `MlflowClient.list_artifacts(run_id)` /
  `download_artifacts(run_id, "")`, `register()` additionally calls
  `mlflow.log_artifacts` at the run level. Network/storage cost roughly
  doubles for large artifact dirs.
- **`test_client()` now cleans up its own cache on exit.** Long-running
  notebooks that call `test_client()` repeatedly after each `register()`
  no longer accumulate `/tmp/nubison_shared_artifacts.*` meta files or
  mlflow's downloaded model folder. Production BentoML serving
  (`build_inference_service`) is unchanged — its caching is still keyed
  to one Pod / one model.
