# Nubison

This project is a SDK for integrate ML model to Nubison.

Requires Python 3.10 or newer (mlflow 3.x requires Python >=3.10).

## Usage

### Load training data

`data.load(uri)` returns a DataFrame from `s3://`, `file://`, or a bare
local path. The source URI is recorded in `df.attrs["source_uri"]` so a
later `train()` call picks it up as MLflow dataset lineage.

```python
from nubison_model import data

df = data.load("s3://my-bucket/datasets/train.parquet")
df = data.load("/home/jovyan/data/train.csv")
```

To load from a saved JupyterLab SQL Explorer connection (no credentials
in the notebook), use `data.connection(name)`:

```python
db = data.connection("MYDB")
df = db.load("SELECT * FROM features WHERE date >= '2026-01-01'")
# df.attrs["source_uri"] == "dbref://MYDB#<query_hash>"
```

### Split into train / val / test

`data.split(df, ratios)` shuffles and slices a single DataFrame into
named subsets ready to hand to `train()`.

```python
full = data.load("s3://my-bucket/all.parquet")
datasets = data.split(
    full,
    {"train": 0.6, "val": 0.2, "test": 0.2},
    random_state=42,
)
```

### Train a model

`train()` is a context manager that opens an MLflow run with autolog,
per-dataset lineage, and yields a `TrainContext` exposing ready-to-fit
data plus mlflow-wrapped helpers — you never `import mlflow`. The same
context manager covers every framework.

```python
from nubison_model import data, train
from sklearn.linear_model import LogisticRegression

with train(datasets=datasets, target=["target"], model_type="classifier") as t:
    t.fit(LogisticRegression(max_iter=500))

print("run_id:", t.run_id)
```

For custom training loops (PyTorch / Lightning / transformers), use
`t.X_train / t.y_train / t.X_val / t.y_val / t.X_test / t.y_test` and
log via `t.log_metric` / `t.save`:

```python
import torch
from nubison_model import train

with train(datasets=datasets, target=["target"], model_type="classifier") as t:
    X = torch.tensor(t.X_train.values, dtype=torch.float32)
    y = torch.tensor(t.y_train.values.ravel(), dtype=torch.long)

    model = MyTorchModule()
    for epoch in range(30):
        loss = step(model, X, y)
        t.log_metric("loss", loss.item(), step=epoch)
    t.save(model)
```

`train()` only logs the run. Packaging the inference code as a `pyfunc`
model is `register()`'s job (see below).

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

The `mlflow` client pin is bumped from `^2.17` to `>=3.12.0,<3.14.0` so
it matches a 3.x mlflow server (mlplatform K8s runs `mlflow/mlflow:v3.12`).
The upper bound stops short of 3.14 because each mlflow 3.x minor has
shipped breaking changes — when 3.14 lands we widen the pin after
re-running the test matrix against it.

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
