# nubison-model example

End-to-end example showing the **train → register → serve** workflow with
`nubison-model`. The training notebooks produce a fitted estimator at
`src/weights.pkl`, and `model.ipynb` packages that file as an inference
service via `register()`.

## Prerequisites

- An MLflow server reachable at `http://127.0.0.1:5000` (or set
  `MLFLOW_TRACKING_URI`).
- Dependencies in `requirements.txt` installed.

## Structure

```
example/
├── train_sklearn.ipynb       # sklearn / xgboost / lightgbm / Keras / skorch
├── train_pytorch.ipynb       # vanilla PyTorch — `t.log_metric` + `t.save(model)`
├── train_lightning.ipynb     # PyTorch Lightning — Trainer.fit + autolog hook
├── train_transformers.ipynb  # HuggingFace transformers Trainer (small text demo)
├── model.ipynb               # NubisonModel wrapping weights.pkl → register → test_client
├── requirements.txt
└── src/
    ├── iris_demo.py          # demo SQL Explorer connection setup
    └── weights.pkl           # produced by a train_*.ipynb, packaged by model.ipynb
```

Pick the `train_*.ipynb` that matches your framework — all four produce
the same `src/weights.pkl` interface so `model.ipynb` packages any of them.

## Order of execution

1. **Run one of `train_*.ipynb`** — loads the iris dataset, trains a
   model, pickles it to `src/weights.pkl` (and logs the run + dataset
   lineage to MLflow). Pick the notebook that matches your framework.
2. **`model.ipynb`** — defines a `NubisonModel` that loads
   `src/weights.pkl` in `load_model` and calls `predict` in `infer`.
   `register()` packages `src/` as a Model Registry entry, and
   `test_client` runs an HTTP smoke test.

The `model.ipynb` template uses a sklearn-style `predict` for inference;
for PyTorch models, replace the `infer` body with `model(x)` and an
`argmax`.

## API summary

### `train(datasets, target, *, weights_path, ...)` — context manager

Single entry point for every framework. Yields a `TrainContext` with:

- `t.X`, `t.y`, `t.X_eval`, `t.y_eval` — DataFrames split out of `datasets`.
- `t.fit(estimator, **fit_kwargs)` — sklearn-fluent shortcut: runs
  `estimator.fit(t.X, t.y, ...)`, pickles the model to `weights_path`,
  and logs `evaluation_score`. Use for sklearn / xgboost / lightgbm /
  Keras / skorch.
- `t.log_metric(name, value, step=None)` / `log_params` / `set_tag` —
  mlflow wrappers. Use inside custom training loops (PyTorch / fastai /
  transformers / etc.) — no `import mlflow` needed.
- `t.save(model, weights_path=None)` — pickle + log as run artifact.
- `t.run_id` — mlflow run id (read after the `with` block).

Side effects of `train()`:
- sets tracking URI / experiment, enables `mlflow.autolog(log_datasets=False)`
- starts a run + logs notebook source (with `notebook.hash` tag)
- logs each `datasets` entry as a separate `log_input` for lineage
- forwards `extra_params` / `extra_tags`
- on exit, mirrors any `artifact_dirs` at the run level

### `register(NubisonModel(), artifact_dirs="src", ...)`

Packages the inference code as an MLflow `pyfunc` model. The
`NubisonModel` class implements:
- `load_model(self, context: ModelContext)` — runs once at server start
- `infer(self, ...)` — runs per request, returns JSON-serializable result

`artifact_dirs="src"` ships `src/weights.pkl` so `load_model` can
pickle-load it.

### `test_client(model_id)`

Spins up the registered model as a local HTTP service for smoke testing.

## requirements.txt

Specifies the dependencies installed in the inference server's environment.
If omitted, the current Python environment is captured automatically.
