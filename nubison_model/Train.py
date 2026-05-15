"""Boilerplate-free training entry point.

Single ``train(estimator, datasets, target, ...)`` function — sklearn-like
one-liner that handles tracking URI, experiment, autolog, ``start_run``
with system metrics, per-dataset lineage (``log_input``), notebook source
/ git tags (best-effort), eval-metric scoring on non-training splits,
fitted-estimator pickle, and any extra artifact directories. The user
never imports ``mlflow``.

Framework coverage assumes the estimator follows the sklearn
``fit(X, y, **kwargs)`` interface. This covers sklearn / xgboost /
lightgbm / fastai / keras directly and PyTorch via a wrapper such as
``skorch`` or ``pytorch-lightning``.

Example::

    run_id = train(
        estimator=LogisticRegression(max_iter=500),
        datasets=datasets,
        target="target",
    )

``pyfunc.log_model`` is intentionally not called here — packaging the
inference code is ``register()``'s job. The trained estimator is
pickled to ``src/weights.pkl`` so ``register(artifact_dirs="src")``
ships it as an inference artifact.
"""

import logging
import pickle
import subprocess
from os import getenv, makedirs, path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

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

TargetT = Union[str, List[str]]


def _resolve_experiment_name(experiment_name: Optional[str]) -> str:
    return (
        experiment_name
        or getenv(ENV_VAR_MLFLOW_EXPERIMENT_NAME)
        or DEFAULT_EXPERIMENT_NAME
    )


def _resolve_mlflow_uri(mlflow_uri: Optional[str]) -> str:
    return mlflow_uri or getenv(ENV_VAR_MLFLOW_TRACKING_URI) or DEFAULT_MLFLOW_URI


def _target_columns(target: TargetT) -> List[str]:
    if isinstance(target, str):
        return [target]
    return list(target)


def _split_features_target(
    df: pd.DataFrame, target: TargetT
) -> Tuple[pd.DataFrame, Any]:
    cols = _target_columns(target)
    X = df.drop(columns=cols)
    if isinstance(target, str):
        y = df[target]
    else:
        y = df[cols]
    return X, y


def _best_effort_log_notebook_source() -> None:
    """If running inside Jupyter (JPY_SESSION_NAME set), log the notebook
    file as an artifact under ``source/``. Best-effort."""
    session = getenv(ENV_VAR_JPY_SESSION_NAME)
    if not session or not path.exists(session):
        return
    try:
        mlflow.log_artifact(session, artifact_path="source")
    except Exception as e:
        logger.debug(f"Could not log notebook source: {e}")


def _best_effort_git_tags() -> Dict[str, str]:
    """Capture git commit + dirty state if we're inside a git repo."""
    tags: Dict[str, str] = {}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if commit:
            tags["mlflow.source.git.commit"] = commit
    except (subprocess.CalledProcessError, FileNotFoundError):
        return tags
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        tags["mlflow.source.git.dirty"] = "true" if status.strip() else "false"
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return tags


_RESOLVABLE_SOURCE_SCHEMES = ("s3://", "file://", "http://", "https://")


def _log_dataset_inputs(
    datasets: Dict[str, pd.DataFrame], target: TargetT
) -> None:
    """Log each DataFrame entry as an mlflow input for lineage.

    mlflow 3.x only accepts ``source`` strings whose scheme has a
    registered ``DatasetSource`` resolver (``s3://``, ``file://``,
    ``http://``...). Our in-memory marker ``memory://`` and custom
    ``dbref://`` (when no SQL Explorer connection is registered) would
    raise; for those we drop the ``source`` kwarg and let mlflow use
    its default ``CodeDatasetSource``.
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


def _resolve_fit_kwargs(
    fit_kwargs: Optional[Dict[str, Any]],
    datasets: Dict[str, pd.DataFrame],
    target: TargetT,
) -> Dict[str, Any]:
    """Resolve magic keys in fit_kwargs.

    ``eval_set`` accepts a key from ``datasets`` (e.g. ``"evaluation"``)
    and is unpacked to ``[(X_eval, y_eval)]`` — the xgboost / lightgbm
    convention. Lists of keys are also accepted.
    """
    if not fit_kwargs:
        return {}
    resolved = dict(fit_kwargs)
    eval_set = resolved.get("eval_set")
    if isinstance(eval_set, str) and eval_set in datasets:
        X_e, y_e = _split_features_target(datasets[eval_set], target)
        resolved["eval_set"] = [(X_e, y_e)]
    elif isinstance(eval_set, list) and all(
        isinstance(item, str) and item in datasets for item in eval_set
    ):
        resolved["eval_set"] = [
            _split_features_target(datasets[k], target) for k in eval_set
        ]
    return resolved


def _log_eval_metrics(
    estimator: Any,
    datasets: Dict[str, pd.DataFrame],
    target: TargetT,
    model_type: str,
    exclude: str = TRAINING_KEY,
) -> None:
    """Predict on every non-training dataset and log a single summary metric.

    Metric: accuracy for classifier, R^2 for regressor. Best-effort —
    autolog already captures training metrics, so this is just for the
    other splits.
    """
    for ctx, df in datasets.items():
        if ctx == exclude or not isinstance(df, pd.DataFrame):
            continue
        try:
            X_, y_ = _split_features_target(df, target)
            score = estimator.score(X_, y_)
            metric_key = "accuracy" if model_type == "classifier" else "r2"
            mlflow.log_metric(f"{ctx}_{metric_key}", float(score))
        except Exception as e:
            logger.debug(f"eval-metric logging failed for {ctx!r}: {e}")


def _dump_weights(estimator: Any, weights_path: str) -> None:
    """Pickle the fitted estimator to disk and log it as an artifact."""
    parent = path.dirname(weights_path)
    if parent:
        makedirs(parent, exist_ok=True)
    with open(weights_path, "wb") as f:
        pickle.dump(estimator, f)
    try:
        mlflow.log_artifact(weights_path)
    except Exception as e:
        logger.debug(f"log_artifact for weights failed: {e}")


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


def train(
    estimator: Any,
    datasets: Dict[str, pd.DataFrame],
    target: TargetT,
    model_type: Literal["classifier", "regressor"] = "classifier",
    artifact_dirs: Optional[str] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    extra_tags: Optional[Dict[str, str]] = None,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    experiment_name: Optional[str] = None,
    mlflow_uri: Optional[str] = None,
    weights_path: str = DEFAULT_WEIGHTS_PATH,
) -> str:
    """Run one MLflow training run end-to-end and return the ``run_id``.

    Args:
        estimator: An estimator implementing ``fit(X, y, **kwargs)``,
            ``predict``, and ``score``. Works for sklearn / xgboost /
            lightgbm / keras directly, and for PyTorch via wrappers like
            ``skorch`` or ``pytorch-lightning``.
        datasets: Mapping of context name → DataFrame. Must contain a
            ``"training"`` key. Additional keys (e.g. ``"evaluation"``,
            ``"test"``) are logged as inputs and scored.
        target: Column name(s) holding the label(s). String for single-
            target, list for multi-target.
        model_type: ``"classifier"`` or ``"regressor"`` — selects the
            eval metric (``accuracy`` / ``r2``) for non-training splits.
        artifact_dirs: Comma-separated directories logged as run
            artifacts via ``mlflow.log_artifacts``.
        extra_params: Forwarded to ``mlflow.log_params``.
        extra_tags: Forwarded to ``mlflow.set_tags``.
        fit_kwargs: Passed through to ``estimator.fit``. The magic key
            ``eval_set`` accepts a datasets key (e.g.
            ``{"eval_set": "evaluation"}``) and is unpacked to
            ``[(X_eval, y_eval)]``.
        experiment_name: Defaults to env ``MLFLOW_EXPERIMENT_NAME`` or
            ``"Default"``.
        mlflow_uri: Defaults to env ``MLFLOW_TRACKING_URI``.
        weights_path: Pickle destination for the fitted estimator.
            Default ``src/weights.pkl`` matches consumer convention so
            ``register(artifact_dirs="src")`` packs it.

    Returns:
        The mlflow ``run_id`` of the created run.

    Raises:
        KeyError: ``datasets`` does not contain a ``"training"`` key.
    """
    if TRAINING_KEY not in datasets:
        raise KeyError(
            f"datasets must contain a {TRAINING_KEY!r} key "
            f"(got: {list(datasets)})"
        )

    mlflow.set_tracking_uri(_resolve_mlflow_uri(mlflow_uri))
    mlflow.set_experiment(_resolve_experiment_name(experiment_name))
    try:
        mlflow.autolog()
    except Exception as e:
        logger.debug(f"mlflow.autolog() failed: {e}")

    with mlflow.start_run() as run:
        _best_effort_log_notebook_source()
        git_tags = _best_effort_git_tags()
        if git_tags:
            mlflow.set_tags(git_tags)

        train_df = datasets[TRAINING_KEY]
        X_train, y_train = _split_features_target(train_df, target)
        fit_kwargs_resolved = _resolve_fit_kwargs(fit_kwargs, datasets, target)
        estimator.fit(X_train, y_train, **fit_kwargs_resolved)

        # Log per-split lineage *after* fit so autolog's own dataset
        # entry does not override our named splits.
        _log_dataset_inputs(datasets, target)

        _log_eval_metrics(estimator, datasets, target, model_type)
        _dump_weights(estimator, weights_path)
        _log_extra_artifact_dirs(artifact_dirs)

        if extra_params:
            mlflow.log_params(extra_params)
        if extra_tags:
            mlflow.set_tags(extra_tags)

        return run.info.run_id
