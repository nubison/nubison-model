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

`train()` is a context manager that opens an MLflow run with autolog,
per-dataset lineage, system-friendly tags, and yields a `TrainContext`
exposing ready-to-fit data plus mlflow-wrapped helpers — you never
`import mlflow`. The same context manager covers every framework.

#### sklearn / xgboost / lightgbm / Keras / skorch

`t.fit(estimator, **fit_kwargs)` runs `estimator.fit(t.X_train, t.y_train, ...)`,
auto-pickles the fitted model to `weights_path`, and (when `model_type`
is set) logs `val_accuracy` or `val_r2` on the `"val"` split.

```python
from nubison_model import data, train
from sklearn.linear_model import LogisticRegression

full = data.load("s3://my-bucket/all.parquet")
datasets = data.split(full, {"train": 0.8, "val": 0.2}, random_state=42)

with train(datasets=datasets, target=["target"], model_type="classifier") as t:
    t.fit(LogisticRegression(max_iter=500))

print("run_id:", t.run_id)
```

For frameworks with extra `fit` arguments (xgboost early stopping, Keras
epochs / validation_data), pass them as kwargs:

```python
from xgboost import XGBClassifier

with train(datasets=datasets, target=["target"], model_type="classifier") as t:
    t.fit(
        XGBClassifier(n_estimators=500),
        eval_set=[(t.X_val, t.y_val)],
        early_stopping_rounds=20,
    )
```

#### PyTorch / PyTorch Lightning / transformers / fastai

Write your training loop directly inside the `with` block. The context
exposes `t.X_train / t.y_train / t.X_val / t.y_val / t.X_test / t.y_test`
and `t.log_metric / t.save` helpers — still no `import mlflow`.

```python
import torch
from nubison_model import train

with train(datasets=datasets, target=["target"], model_type="classifier") as t:
    X = torch.tensor(t.X_train.values, dtype=torch.float32)
    y = torch.tensor(t.y_train.values.ravel(), dtype=torch.long)

    model = MyTorchModule()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(30):
        loss = my_step(model, X, y, optimizer)
        t.log_metric("loss", loss.item(), step=epoch)
    t.save(model)

print("run_id:", t.run_id)
```

`TrainContext` API (no `import mlflow` in user code):

| Member | Role |
|--------|------|
| `t.X_train` / `t.y_train` | features / target from `datasets["train"]` |
| `t.X_val` / `t.y_val` | same for `datasets["val"]` (None if absent) |
| `t.X_test` / `t.y_test` | same for `datasets["test"]` (None if absent) |
| `t.datasets` | original dict — use for non-standard split keys |
| `t.fit(estimator, **fit_kwargs)` | sklearn-fluent shortcut: fit + save + evaluation score |
| `t.log_metric / log_metrics / log_param / log_params / set_tag` | mlflow wrappers |
| `t.save(model, weights_path=None)` | pickle + log as run artifact |
| `t.run_id` | mlflow run id (read after the `with` block) |

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
