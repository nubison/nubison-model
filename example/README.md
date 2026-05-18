# nubison-model example

End-to-end example showing the **train → infer** workflow with
`nubison-model`. Each `train_*.ipynb` pickles a fitted model to
`src/weights.pkl`; the matching `infer_*.ipynb` wraps it as a
`NubisonModel`, registers it, and runs an HTTP smoke test.

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
├── infer_sklearn.ipynb       # NubisonModel.predict(DataFrame)
├── infer_pytorch.ipynb       # NubisonModel.forward(tensor) + argmax
├── infer_lightning.ipynb     # same shape against IrisLightning
├── infer_transformers.ipynb  # rebuild tokenizer + model(**tokenized)
├── requirements.txt
└── src/
    ├── demo.py               # iris SQL Explorer connection scaffolding
    ├── iris_net.py           # IrisNet — imported by train/infer_pytorch
    ├── iris_lightning.py     # IrisLightning — imported by train/infer_lightning
    └── weights.pkl           # produced by train_*.ipynb, packaged by infer_*.ipynb
```

`infer_*.ipynb` pairs one-to-one with `train_*.ipynb` because the
fitted-model contract differs by framework: sklearn exposes
`.predict(DataFrame)`, PyTorch needs a tensor + `argmax`, transformers
needs a tokenizer. Run only the pair matching your framework.

## Order of execution

1. **Run one `train_*.ipynb`** — loads the iris dataset (or an inline
   sentiment set for transformers), trains a model, pickles it to
   `src/weights.pkl`, and logs the run + dataset lineage to MLflow.
2. **Run the matching `infer_*.ipynb`** — defines a `NubisonModel`
   that loads `src/weights.pkl` in `load_model` and serves it from
   `infer`. `register()` packages `src/` as a Model Registry entry,
   and `test_client` runs an HTTP smoke test.

Re-running a different `train_*.ipynb` overwrites `src/weights.pkl`,
so always re-run its matching `infer_*.ipynb` afterwards.

## API summary

### `train(datasets, target, *, weights_path, ...)` — context manager

Single entry point for every framework. Yields a `TrainContext` with:

- `t.X_train` / `t.y_train` (+ `_val` / `_test` when present) —
  DataFrames split out of `datasets`.
- `t.fit(estimator, **fit_kwargs)` — sklearn-fluent shortcut: runs
  `estimator.fit(t.X_train, t.y_train, ...)`, pickles the model to
  `weights_path`, and logs the matching validation metric. Use for
  sklearn / xgboost / lightgbm / Keras / skorch.
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

`artifact_dirs="src"` ships `src/weights.pkl` (and the `iris_net.py` /
`iris_lightning.py` modules for PyTorch / Lightning) so `load_model`
can pickle-load them.

### `test_client(model_id)`

Spins up the registered model as a local HTTP service for smoke testing.

## requirements.txt

Specifies the dependencies installed in the inference server's environment.
If omitted, the current Python environment is captured automatically.
