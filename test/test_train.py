"""Unit tests for ``nubison_model.train``."""

import pickle
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression, LogisticRegression

from nubison_model import DEFAULT_WEIGHTS_PATH, SOURCE_URI_ATTR, train
from nubison_model.Train import (
    TRAINING_KEY,
    _resolve_fit_kwargs,
    _split_features_target,
)


def _df_with_uri(data: dict, uri: str) -> pd.DataFrame:
    df = pd.DataFrame(data)
    df.attrs[SOURCE_URI_ATTR] = uri
    return df


@pytest.fixture
def datasets():
    train_df = _df_with_uri(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8],
            "b": [2, 3, 4, 5, 6, 7, 8, 9],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        },
        "memory://training",
    )
    eval_df = _df_with_uri(
        {"a": [1, 2, 3, 4], "b": [2, 3, 4, 5], "target": [0, 1, 0, 1]},
        "memory://evaluation",
    )
    return {"training": train_df, "evaluation": eval_df}


# ---------------------------------------------------------------------------
# pure helpers (no mlflow_server needed)
# ---------------------------------------------------------------------------


class TestTargetHandling:
    def test_string_target_splits_correctly(self):
        df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        X, y = _split_features_target(df, "target")
        assert list(X.columns) == ["a"]
        assert y.tolist() == [0, 1]

    def test_list_target_splits_correctly(self):
        df = pd.DataFrame({"a": [1, 2], "t1": [0, 1], "t2": [1, 0]})
        X, y = _split_features_target(df, ["t1", "t2"])
        assert list(X.columns) == ["a"]
        assert list(y.columns) == ["t1", "t2"]


class TestResolveFitKwargs:
    def test_empty_returns_empty(self):
        assert _resolve_fit_kwargs(None, {}, "target") == {}
        assert _resolve_fit_kwargs({}, {}, "target") == {}

    def test_eval_set_string_resolved(self):
        eval_df = pd.DataFrame({"a": [1, 2], "target": [0, 1]})
        out = _resolve_fit_kwargs(
            {"eval_set": "evaluation"},
            {"evaluation": eval_df},
            "target",
        )
        assert "eval_set" in out
        assert isinstance(out["eval_set"], list)
        assert len(out["eval_set"]) == 1
        X_e, y_e = out["eval_set"][0]
        assert list(X_e.columns) == ["a"]
        assert y_e.tolist() == [0, 1]

    def test_eval_set_list_resolved(self):
        df1 = pd.DataFrame({"a": [1], "target": [0]})
        df2 = pd.DataFrame({"a": [2], "target": [1]})
        out = _resolve_fit_kwargs(
            {"eval_set": ["evaluation", "test"]},
            {"evaluation": df1, "test": df2},
            "target",
        )
        assert len(out["eval_set"]) == 2

    def test_unknown_eval_set_passthrough(self):
        out = _resolve_fit_kwargs(
            {"eval_set": [(pd.DataFrame({"a": [1]}), pd.Series([0]))]},
            {},
            "target",
        )
        # Not magically resolved — preserved as-is
        assert isinstance(out["eval_set"], list)


# ---------------------------------------------------------------------------
# train() — end-to-end
# ---------------------------------------------------------------------------


class TestTrainBasic:
    def test_train_classifier_returns_run_id(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        run_id = train(
            estimator=LogisticRegression(max_iter=200),
            datasets=datasets,
            target="target",
        )
        assert isinstance(run_id, str) and run_id
        run = MlflowClient().get_run(run_id)
        assert run.info.status == "FINISHED"

    def test_train_writes_weights_pickle(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        train(
            estimator=LogisticRegression(max_iter=200),
            datasets=datasets,
            target="target",
        )
        weights = Path(DEFAULT_WEIGHTS_PATH)
        assert weights.exists()
        with open(weights, "rb") as f:
            loaded = pickle.load(f)
        assert hasattr(loaded, "predict")

    def test_train_target_as_list(
        self, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        df = _df_with_uri(
            {"a": [1, 2, 3, 4], "t1": [0, 1, 0, 1]}, "memory://training"
        )
        run_id = train(
            estimator=LogisticRegression(max_iter=200),
            datasets={"training": df},
            target=["t1"],
        )
        assert run_id

    def test_train_regressor(
        self, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        df = _df_with_uri(
            {"a": list(range(10)), "target": [2 * x for x in range(10)]},
            "memory://training",
        )
        run_id = train(
            estimator=LinearRegression(),
            datasets={"training": df},
            target="target",
            model_type="regressor",
        )
        assert run_id

    def test_train_missing_training_key_raises(self, mlflow_server):
        df = _df_with_uri({"a": [1], "target": [0]}, "memory://x")
        with pytest.raises(KeyError, match=TRAINING_KEY):
            train(
                estimator=LogisticRegression(),
                datasets={"evaluation": df},
                target="target",
            )


class TestTrainLineage:
    def test_dataset_inputs_logged(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        run_id = train(
            estimator=LogisticRegression(max_iter=200),
            datasets=datasets,
            target="target",
        )
        run = MlflowClient().get_run(run_id)
        input_names = {d.dataset.name for d in run.inputs.dataset_inputs}
        assert "training" in input_names
        assert "evaluation" in input_names


class TestTrainExtraParamsTags:
    def test_extra_params_and_tags_logged(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        run_id = train(
            estimator=LogisticRegression(max_iter=200),
            datasets=datasets,
            target="target",
            extra_params={"feature_version": "v1"},
            extra_tags={"team": "ds"},
        )
        run = MlflowClient().get_run(run_id)
        assert run.data.params.get("feature_version") == "v1"
        assert run.data.tags.get("team") == "ds"


class TestTrainEvalMetric:
    def test_eval_metric_logged_for_non_training_splits(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        run_id = train(
            estimator=LogisticRegression(max_iter=200),
            datasets=datasets,
            target="target",
        )
        run = MlflowClient().get_run(run_id)
        # accuracy on evaluation split is logged
        assert "evaluation_accuracy" in run.data.metrics


class TestTrainArtifactDirs:
    def test_artifact_dirs_logged(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        extra_dir = tmp_path / "extra"
        extra_dir.mkdir()
        (extra_dir / "info.txt").write_text("hello")

        run_id = train(
            estimator=LogisticRegression(max_iter=200),
            datasets=datasets,
            target="target",
            artifact_dirs="extra",
        )
        artifacts = {a.path for a in MlflowClient().list_artifacts(run_id)}
        assert "info.txt" in artifacts
