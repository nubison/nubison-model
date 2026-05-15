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
               loss = my_step(t.X, t.y)
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
DEFAULT_EXPERIMENT_NAME = "Default"
DEFAULT_WEIGHTS_PATH = "src/weights.pkl"

TRAINING_KEY = "training"
EVALUATION_KEY = "evaluation"

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


def _log_extra_artifact_dirs(artifact_dirs: Optional[str]) -> None:
    if not artifact_dirs:
        return
    for entry in artifact_dirs.split(","):
        entry = entry.strip()
        if entry and path.exists(entry):
            try:
                mlflow.log_artifacts(entry)
            except Exception as e:
                logger.debug(f"log_artifacts({entry!r}) failed: {e}")


class TrainContext:
    """Yielded by :func:`train`.

    Attributes:
        X, y: feature matrix and target from ``datasets["training"]``.
        X_eval, y_eval: from ``datasets["evaluation"]`` (None if absent).
        datasets: the original dict.
        run_id: mlflow run id (populated once the ``with`` block opens).

    Methods (mlflow wrappers — no ``import mlflow`` needed in user code):
        fit(estimator, **fit_kwargs): sklearn-fluent shortcut. Calls
            ``estimator.fit(X, y, **fit_kwargs)``, auto-saves the fitted
            estimator to ``weights_path``, and logs ``evaluation_score``
            if ``X_eval / y_eval`` exist and the estimator has ``score()``.
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
        if TRAINING_KEY not in datasets:
            raise KeyError(
                f"datasets must contain a {TRAINING_KEY!r} key "
                f"(got: {list(datasets)})"
            )
        self.datasets = datasets
        self._target = target
        self._weights_path = weights_path
        self._model_type = model_type
        self.X, self.y = _split_features_target(datasets[TRAINING_KEY], target)
        eval_df = datasets.get(EVALUATION_KEY)
        if isinstance(eval_df, pd.DataFrame):
            self.X_eval, self.y_eval = _split_features_target(eval_df, target)
        else:
            self.X_eval = None
            self.y_eval = None
        self.run_id: Optional[str] = None

    def fit(self, estimator: Any, **fit_kwargs: Any) -> Any:
        """Train ``estimator`` on ``self.X / self.y`` (sklearn-style).

        Auto-saves the fitted model to ``weights_path``. When
        ``model_type`` is set ("classifier" / "regressor") and an
        ``"evaluation"`` split is present, also logs the matching
        metric (``evaluation_accuracy`` / ``evaluation_r2``).
        """
        estimator.fit(self.X, self.y, **fit_kwargs)
        self.save(estimator)
        if (
            self._model_type
            and self.X_eval is not None
            and self.y_eval is not None
        ):
            try:
                score = estimator.score(self.X_eval, self.y_eval)
                metric_key = (
                    "evaluation_accuracy"
                    if self._model_type == "classifier"
                    else "evaluation_r2"
                )
                mlflow.log_metric(metric_key, float(score))
            except Exception as e:
                logger.debug(f"evaluation score skipped: {e}")
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
            ``"training"``; ``"evaluation"`` is recognized for the
            ``X_eval / y_eval`` convenience and auto-scoring.
        target: label column name (or list of names for multi-target).
        model_type: free-form string tagged on the run as ``model_type``
            (surfaced in the nubison UI). Two values get special
            treatment: ``"classifier"`` and ``"regressor"`` make
            ``t.fit()`` log ``evaluation_accuracy`` / ``evaluation_r2``
            on the evaluation split. Other values (e.g. ``"clustering"``,
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
        KeyError: ``datasets`` does not contain a ``"training"`` key.
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

    with mlflow.start_run() as run:
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
