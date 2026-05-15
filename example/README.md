# nubison-model example

End-to-end example showing the **train → register → serve** workflow with
`nubison-model`. The two notebooks chain together: `train.ipynb` produces
a fitted estimator at `src/weights.pkl`, and `model.ipynb` packages that
file as an inference service via `register()`.

## Prerequisites

- An MLflow server reachable at `http://127.0.0.1:5000` (or set
  `MLFLOW_TRACKING_URI`).
- Dependencies in `requirements.txt` installed.

## Structure

```
example/
├── train.ipynb              # data.load → data.split → train → src/weights.pkl
├── model.ipynb              # NubisonModel wrapping weights.pkl → register → test_client
├── requirements.txt
└── src/
    ├── iris_demo.py         # demo helper used by train.ipynb (not shipped)
    └── weights.pkl          # produced by train.ipynb, packaged by model.ipynb
```

## Order of execution

1. **`train.ipynb`** — loads the iris dataset (file / S3 / SQL Explorer
   connection), splits it, calls `train()`. The fitted estimator is
   pickled to `src/weights.pkl` automatically.
2. **`model.ipynb`** — defines a `NubisonModel` that loads `src/weights.pkl`
   in `load_model` and calls `predict` in `infer`. `register()` packages
   `src/` (including `weights.pkl`) as an MLflow Model Registry entry, and
   `test_client` runs an HTTP smoke test.

You can run `model.ipynb` standalone too — its `UserModel` only depends on
`src/weights.pkl`. If you haven't run `train.ipynb` yet you'll need to
provide that file by other means.

## API summary

### `train(estimator, datasets, target, ...)`

One-call training entry point. Works for any framework following the
sklearn `fit(X, y, **kwargs)` interface — sklearn / xgboost / lightgbm /
keras / fastai, and PyTorch via `skorch` or `pytorch-lightning`.

Returns the MLflow `run_id`. Side effects:
- sets tracking URI / experiment, runs `autolog()`
- starts a run with system metrics
- logs each `datasets` entry as a separate `log_input` for lineage
- scores the estimator on non-`"training"` splits (`accuracy` / `r2`)
- pickles the fitted estimator to `src/weights.pkl`
- logs any directories listed in `artifact_dirs`

### `register(NubisonModel(), artifact_dirs="src", ...)`

Packages the inference code as an MLflow `pyfunc` model. The `NubisonModel`
class must implement:
- `load_model(self, context: ModelContext)` — runs once at server start
- `infer(self, ...)` — runs per request, returns JSON-serializable result

`artifact_dirs="src"` ships `src/weights.pkl` so `load_model` can pickle-load it.

### `test_client(model_id)`

Spins up the registered model as a local HTTP service for smoke testing.

## requirements.txt

Specifies the dependencies installed in the inference server's environment.
If omitted, the current Python environment is captured automatically.
