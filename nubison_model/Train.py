"""Single training context manager — framework-agnostic.

``train()`` is a context manager that opens an MLflow run with autolog
+ dataset lineage + system-friendly tags, and yields a ``TrainContext``
that exposes ready-to-fit data and ``log_*`` helpers. The user never
imports ``mlflow``.

Two usage paths from the same ``with`` block:

1. **sklearn-fluent** (sklearn / xgboost / lightgbm / Keras / skorch)::

       with train(datasets=datasets, target="target") as t:
           t.fit(LogisticRegression(max_iter=100))
       print(t.run_id)

2. **Custom training loop** (vanilla PyTorch / PyTorch Lightning /
   transformers / fastai / TensorFlow GradientTape, etc.)::

       with train(datasets=datasets, target="target") as t:
           for epoch in range(10):
               loss = my_step(t.X_train, t.y_train)
               t.log_metric("loss", loss, step=epoch)
           t.save(model)
       print(t.run_id)

``pyfunc.log_model`` is intentionally not called here — packaging is
``register()``'s job. ``t.save(model)`` pickles to ``src/weights.pkl``
so ``register(artifact_dirs="src")`` ships it.
"""

import hashlib
import logging
import pickle
import subprocess
from contextlib import contextmanager
from os import getenv, makedirs, path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import mlflow
import pandas as pd

from nubison_model.Data import SOURCE_URI_ATTR
from nubison_model.Model import (
    DEFAULT_MLFLOW_URI,
    ENV_VAR_MLFLOW_TRACKING_URI,
)

logger = logging.getLogger(__name__)

ENV_VAR_MLFLOW_EXPERIMENT_NAME = "MLFLOW_EXPERIMENT_NAME"
ENV_VAR_JPY_SESSION_NAME = "JPY_SESSION_NAME"
ENV_VAR_MLFLOW_SYSTEM_METRICS = "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"
DEFAULT_EXPERIMENT_NAME = "Default"
DEFAULT_WEIGHTS_PATH = "src/weights.pkl"

TRAIN_KEY = "train"
VAL_KEY = "val"
TEST_KEY = "test"

TargetT = Union[str, List[str]]


def _split_features_target(
    df: pd.DataFrame, target: TargetT
) -> Tuple[pd.DataFrame, Any]:
    cols = [target] if isinstance(target, str) else list(target)
    X = df.drop(columns=cols)
    y = df[target] if isinstance(target, str) else df[cols]
    return X, y


def _best_effort_log_notebook_source() -> None:
    """If running inside Jupyter (JPY_SESSION_NAME set), log the notebook
    as an artifact under ``source/`` and tag the run with ``notebook.hash``
    (sha256[:12] of the file).

    The ``notebook.hash`` tag is mlplatform's Source-column primary key:
    its frontend matches runs to notebooks by this tag (sliced to 12
    chars) and uses ``<hash>.ipynb`` as the download filename.
    """
    session = getenv(ENV_VAR_JPY_SESSION_NAME)
    if not session or not path.exists(session):
        return
    try:
        mlflow.log_artifact(session, artifact_path="source")
        with open(session, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()[:12]
        mlflow.set_tag("notebook.hash", sha)
    except Exception as e:
        logger.debug(f"Could not log notebook source: {e}")


def _best_effort_git_tags() -> Dict[str, str]:
    """Capture git commit + dirty state if we're inside a git repo."""
    tags: Dict[str, str] = {}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        if commit:
            tags["mlflow.source.git.commit"] = commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return tags
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True
        )
        is_dirty = "true" if status.strip() else "false"
        tags["mlflow.source.git.dirty"] = is_dirty
        # mlplatform frontend reads the un-prefixed key; keep both for
        # compatibility until the frontend migrates to the mlflow.* form.
        tags["git.dirty"] = is_dirty
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return tags


_RESOLVABLE_SOURCE_SCHEMES = ("s3://", "file://", "http://", "https://")


def _log_dataset_inputs(
    datasets: Dict[str, pd.DataFrame], target: TargetT
) -> None:
    """Log each DataFrame entry as an mlflow input for lineage.

    mlflow 3.x only accepts ``source`` strings whose scheme has a
    registered DatasetSource resolver — we drop the kwarg for our
    in-memory / ``dbref://`` URIs and let mlflow use its default.
    """
    for ctx, df in datasets.items():
        if not isinstance(df, pd.DataFrame):
            continue
        source_uri = df.attrs.get(SOURCE_URI_ATTR, "")
        targets_label = target if isinstance(target, str) else ",".join(target)
        kwargs: Dict[str, Any] = {"name": ctx, "targets": targets_label}
        if source_uri and source_uri.startswith(_RESOLVABLE_SOURCE_SCHEMES):
            kwargs["source"] = source_uri
        try:
            dataset = mlflow.data.from_pandas(df, **kwargs)
            mlflow.log_input(dataset, context=ctx)
        except Exception as e:
            logger.debug(f"log_input failed for {ctx!r}: {e}")
        # Mirror the original source URI as a tag so consumers (e.g. the
        # nubison UI) can render the logical lineage for schemes mlflow
        # doesn't recognise (dbref://, memory://, …). mlflow itself stores
        # only the resolvable URIs above as the Dataset's `source`.
        if source_uri:
            try:
                mlflow.set_tag(f"nubison.dataset.{ctx}.source", source_uri)
            except Exception as e:
                logger.debug(f"source tag for {ctx!r} failed: {e}")


def _log_extra_artifact_dirs(artifact_dirs: Optional[str]) -> None:
    if not artifact_dirs:
        return
    for entry in artifact_dirs.split(","):
        entry = entry.strip()
        if not entry:
            continue
        abs_entry = path.abspath(entry)
        if path.exists(abs_entry):
            try:
                mlflow.log_artifacts(abs_entry)
            except Exception as e:
                logger.debug(f"log_artifacts({abs_entry!r}) failed: {e}")


class TrainContext:
    """Yielded by :func:`train`.

    Attributes:
        X_train, y_train: features / target from ``datasets["train"]``.
        X_val, y_val: from ``datasets["val"]`` (None if absent).
        X_test, y_test: from ``datasets["test"]`` (None if absent).
        datasets: the original dict.
        run_id: mlflow run id (populated once the ``with`` block opens).

    Methods (mlflow wrappers — no ``import mlflow`` needed in user code):
        fit(estimator, **fit_kwargs): sklearn-fluent shortcut. Calls
            ``estimator.fit(X_train, y_train, **fit_kwargs)``,
            auto-saves the fitted estimator to ``weights_path``, and
            (when ``model_type`` is set) logs ``val_accuracy`` /
            ``val_r2`` on the ``val`` split.
        log_metric / log_metrics / log_param / log_params / set_tag:
            thin wrappers over the matching ``mlflow.*`` calls.
        save(model, weights_path=None): pickle ``model`` and log it as a
            run artifact. Default destination is ``train()``'s
            ``weights_path`` so ``register(artifact_dirs="src")`` ships it.
    """

    def __init__(
        self,
        datasets: Dict[str, pd.DataFrame],
        target: TargetT,
        weights_path: str,
        model_type: Optional[str],
    ):
        if TRAIN_KEY not in datasets:
            raise KeyError(
                f"datasets must contain a {TRAIN_KEY!r} key "
                f"(got: {list(datasets)})"
            )
        self.datasets = datasets
        self._target = target
        self._weights_path = weights_path
        self._model_type = model_type
        self.X_train, self.y_train = _split_features_target(
            datasets[TRAIN_KEY], target
        )
        val_df = datasets.get(VAL_KEY)
        if isinstance(val_df, pd.DataFrame):
            self.X_val, self.y_val = _split_features_target(val_df, target)
        else:
            self.X_val = None
            self.y_val = None
        test_df = datasets.get(TEST_KEY)
        if isinstance(test_df, pd.DataFrame):
            self.X_test, self.y_test = _split_features_target(test_df, target)
        else:
            self.X_test = None
            self.y_test = None
        self.run_id: Optional[str] = None

    def fit(self, estimator: Any, **fit_kwargs: Any) -> Any:
        """Train ``estimator`` on ``self.X_train / self.y_train``.

        Auto-saves the fitted model to ``weights_path``. When
        ``model_type`` is set ("classifier" / "regressor") and a
        ``"val"`` split is present, also logs the matching metric
        (``val_accuracy`` / ``val_r2``).
        """
        estimator.fit(self.X_train, self.y_train, **fit_kwargs)
        self.save(estimator)
        if (
            self._model_type
            and self.X_val is not None
            and self.y_val is not None
        ):
            try:
                score = estimator.score(self.X_val, self.y_val)
                metric_key = (
                    "val_accuracy"
                    if self._model_type == "classifier"
                    else "val_r2"
                )
                mlflow.log_metric(metric_key, float(score))
            except Exception as e:
                logger.debug(f"val score skipped: {e}")
        return estimator

    def log_metric(
        self, key: str, value: float, step: Optional[int] = None
    ) -> None:
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_param(self, key: str, value: Any) -> None:
        mlflow.log_param(key, value)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def set_tag(self, key: str, value: Any) -> None:
        mlflow.set_tag(key, value)

    def save(self, model: Any, weights_path: Optional[str] = None) -> None:
        """Pickle ``model`` and log it as a run artifact.

        Default destination is the ``weights_path`` passed to
        ``train()``. ``register(artifact_dirs="src")`` then ships the
        file as an inference artifact.
        """
        dst = weights_path or self._weights_path
        parent = path.dirname(dst)
        if parent:
            makedirs(parent, exist_ok=True)
        with open(dst, "wb") as f:
            pickle.dump(model, f)
        try:
            mlflow.log_artifact(dst)
        except Exception as e:
            logger.debug(f"log_artifact for weights failed: {e}")


@contextmanager
def train(
    datasets: Dict[str, pd.DataFrame],
    target: TargetT,
    *,
    model_type: Optional[str] = None,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
    artifact_dirs: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    extra_tags: Optional[Dict[str, str]] = None,
    experiment_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
) -> Iterator[TrainContext]:
    """Open an autolog'd MLflow run and yield a :class:`TrainContext`.

    Args:
        datasets: ``{name: DataFrame}`` from ``data.split``. Must include
            ``"train"``; ``"val"`` is recognized for the
            ``X_val / y_val`` convenience and auto-scoring;
            ``"test"`` is recognized for ``X_test / y_test``.
        target: label column name (or list of names for multi-target).
        model_type: free-form string tagged on the run as ``model_type``
            (surfaced in the nubison UI). Two values get special
            treatment: ``"classifier"`` and ``"regressor"`` make
            ``t.fit()`` log ``val_accuracy`` / ``val_r2``
            on the ``"val"`` split. Other values (e.g. ``"clustering"``,
            ``"anomaly_detection"``) just tag the run — the user logs
            their own metrics via ``t.log_metric``.
        weights_path: where ``t.save(model)`` writes the pickle. Default
            ``src/weights.pkl`` matches ``register(artifact_dirs="src")``.
        artifact_dirs: comma-separated extra directories logged at the
            run level via ``mlflow.log_artifacts`` on exit.
        extra_params / extra_tags: forwarded to ``mlflow.log_params`` /
            ``mlflow.set_tags`` once the run starts.
        experiment_name: defaults to env ``MLFLOW_EXPERIMENT_NAME`` or
            ``"Default"``.
        mlflow_uri: defaults to env ``MLFLOW_TRACKING_URI``.

    Yields:
        :class:`TrainContext` — see its docstring for the full API.

    Raises:
        KeyError: ``datasets`` does not contain a ``"train"`` key.
    """
    mlflow.set_tracking_uri(
        mlflow_uri or getenv(ENV_VAR_MLFLOW_TRACKING_URI) or DEFAULT_MLFLOW_URI
    )
    mlflow.set_experiment(
        experiment_name
        or getenv(ENV_VAR_MLFLOW_EXPERIMENT_NAME)
        or DEFAULT_EXPERIMENT_NAME
    )
    try:
        # log_datasets=False — keep our explicit per-split lineage from
        # `_log_dataset_inputs`; autolog's generic "dataset" entry would
        # otherwise override the training / evaluation / ... splits.
        mlflow.autolog(log_datasets=False)
    except Exception as e:
        logger.debug(f"mlflow.autolog() failed: {e}")

    ctx = TrainContext(datasets, target, weights_path, model_type)

    # Default ON per issue #22 spec; honor MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING
    # so users (and the test suite against sqlite mlflow) can opt out.
    env_val = getenv(ENV_VAR_MLFLOW_SYSTEM_METRICS, "").strip().lower()
    log_system_metrics = env_val not in ("false", "0", "no")

    with mlflow.start_run(log_system_metrics=log_system_metrics) as run:
        ctx.run_id = run.info.run_id
        _best_effort_log_notebook_source()
        git_tags = _best_effort_git_tags()
        if git_tags:
            mlflow.set_tags(git_tags)
        if model_type:
            mlflow.set_tag("model_type", model_type)
        if extra_params:
            mlflow.log_params(extra_params)
        if extra_tags:
            mlflow.set_tags(extra_tags)
        try:
            yield ctx
        finally:
            # Log per-split lineage *after* the user's training so
            # autolog's hooks don't override ours.
            _log_dataset_inputs(datasets, target)
            _log_extra_artifact_dirs(artifact_dirs)
