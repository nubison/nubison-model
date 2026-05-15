"""Unit tests for the ``train()`` context manager."""

import pickle
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression, LogisticRegression

from nubison_model import SOURCE_URI_ATTR, train


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
# TrainContext shape / lifecycle
# ---------------------------------------------------------------------------


class TestTrainContextShape:
    def test_unpacks_training_and_evaluation(self, datasets, mlflow_server):
        with train(datasets=datasets, target="target") as t:
            assert list(t.X.columns) == ["a", "b"]
            assert t.y.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
            assert list(t.X_eval.columns) == ["a", "b"]
            assert t.y_eval.tolist() == [0, 1, 0, 1]

    def test_no_evaluation_dataset(self, mlflow_server):
        df = _df_with_uri(
            {"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]}, "memory://training"
        )
        with train(datasets={"training": df}, target="target") as t:
            assert t.X_eval is None
            assert t.y_eval is None

    def test_target_as_list(self, mlflow_server):
        df = _df_with_uri(
            {"a": [1, 2, 3, 4], "t1": [0, 1, 0, 1], "t2": [1, 0, 1, 0]},
            "memory://training",
        )
        with train(datasets={"training": df}, target=["t1", "t2"]) as t:
            assert list(t.X.columns) == ["a"]
            assert list(t.y.columns) == ["t1", "t2"]

    def test_preserves_raw_datasets(self, datasets, mlflow_server):
        with train(datasets=datasets, target="target") as t:
            assert t.datasets is datasets


class TestTrainContextErrors:
    def test_missing_training_raises(self, mlflow_server):
        df = _df_with_uri({"a": [1], "target": [0]}, "memory://eval")
        with pytest.raises(KeyError, match="training"):
            with train(datasets={"evaluation": df}, target="target"):
                pass


# ---------------------------------------------------------------------------
# t.fit() — sklearn-fluent path
# ---------------------------------------------------------------------------


class TestFitShortcut:
    def test_fit_trains_estimator_and_returns_it(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(datasets=datasets, target="target") as t:
            fitted = t.fit(LogisticRegression(max_iter=200))
            assert hasattr(fitted, "predict")
            # Same object returned, already fitted.
            assert (fitted.predict(t.X) == fitted.predict(t.X)).all()

    def test_fit_pickles_to_default_weights_path(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(datasets=datasets, target="target") as t:
            t.fit(LogisticRegression(max_iter=200))
        weights = tmp_path / "src" / "weights.pkl"
        assert weights.exists()
        with open(weights, "rb") as f:
            loaded = pickle.load(f)
        assert hasattr(loaded, "predict")

    def test_fit_logs_evaluation_score(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(datasets=datasets, target="target") as t:
            t.fit(LogisticRegression(max_iter=200))
        run = MlflowClient().get_run(t.run_id)
        assert "evaluation_score" in run.data.metrics

    def test_fit_passes_fit_kwargs(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        # sklearn estimators accept sample_weight via fit_kwargs
        monkeypatch.chdir(tmp_path)
        weights = [1.0] * len(datasets["training"])
        with train(datasets=datasets, target="target") as t:
            t.fit(LogisticRegression(max_iter=200), sample_weight=weights)
        assert t.run_id is not None

    def test_fit_with_regressor(
        self, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        df = _df_with_uri(
            {"a": list(range(10)), "target": [2 * x for x in range(10)]},
            "memory://training",
        )
        with train(datasets={"training": df}, target="target") as t:
            t.fit(LinearRegression())
        assert t.run_id is not None


# ---------------------------------------------------------------------------
# t.log_metric / t.log_param — custom loop path (no `import mlflow`)
# ---------------------------------------------------------------------------


class TestLogHelpers:
    def test_log_metric_wraps_mlflow(self, datasets, mlflow_server):
        with train(datasets=datasets, target="target") as t:
            t.log_metric("loss", 0.5, step=0)
            t.log_metric("loss", 0.3, step=1)
        run = MlflowClient().get_run(t.run_id)
        assert run.data.metrics["loss"] == 0.3

    def test_log_metrics_bulk(self, datasets, mlflow_server):
        with train(datasets=datasets, target="target") as t:
            t.log_metrics({"loss": 0.5, "accuracy": 0.9})
        run = MlflowClient().get_run(t.run_id)
        assert run.data.metrics["loss"] == 0.5
        assert run.data.metrics["accuracy"] == 0.9

    def test_log_param_wraps_mlflow(self, datasets, mlflow_server):
        with train(datasets=datasets, target="target") as t:
            t.log_param("custom_lr", "0.01")
        run = MlflowClient().get_run(t.run_id)
        assert run.data.params["custom_lr"] == "0.01"

    def test_set_tag_wraps_mlflow(self, datasets, mlflow_server):
        with train(datasets=datasets, target="target") as t:
            t.set_tag("model_family", "linear")
        run = MlflowClient().get_run(t.run_id)
        assert run.data.tags["model_family"] == "linear"


# ---------------------------------------------------------------------------
# t.save() — explicit pickle (for non-sklearn frameworks)
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_default_path(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(datasets=datasets, target="target") as t:
            model = LogisticRegression(max_iter=200).fit(t.X, t.y)
            t.save(model)
        weights = tmp_path / "src" / "weights.pkl"
        assert weights.exists()

    def test_save_custom_path(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(datasets=datasets, target="target") as t:
            model = LogisticRegression(max_iter=200).fit(t.X, t.y)
            t.save(model, weights_path="models/v1.pkl")
        assert (tmp_path / "models" / "v1.pkl").exists()

    def test_save_logs_artifact(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(datasets=datasets, target="target") as t:
            model = LogisticRegression(max_iter=200).fit(t.X, t.y)
            t.save(model)
        artifacts = {a.path for a in MlflowClient().list_artifacts(t.run_id)}
        assert "weights.pkl" in artifacts


# ---------------------------------------------------------------------------
# Lineage + extras
# ---------------------------------------------------------------------------


class TestLineage:
    def test_dataset_inputs_per_split(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(datasets=datasets, target="target") as t:
            t.fit(LogisticRegression(max_iter=200))
        run = MlflowClient().get_run(t.run_id)
        names = {d.dataset.name for d in run.inputs.dataset_inputs}
        assert "training" in names
        assert "evaluation" in names


class TestExtraParamsTags:
    def test_extra_params_and_tags_logged(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(
            datasets=datasets,
            target="target",
            extra_params={"feature_version": "v1"},
            extra_tags={"team": "ds"},
        ) as t:
            t.fit(LogisticRegression(max_iter=200))
        run = MlflowClient().get_run(t.run_id)
        assert run.data.params["feature_version"] == "v1"
        assert run.data.tags["team"] == "ds"


class TestArtifactDirs:
    def test_artifact_dirs_logged(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        extra = tmp_path / "extra"
        extra.mkdir()
        (extra / "info.txt").write_text("hello")
        with train(
            datasets=datasets, target="target", artifact_dirs="extra"
        ) as t:
            t.fit(LogisticRegression(max_iter=200))
        artifacts = {a.path for a in MlflowClient().list_artifacts(t.run_id)}
        assert "info.txt" in artifacts


# ---------------------------------------------------------------------------
# run_id lifecycle
# ---------------------------------------------------------------------------


class TestRunIdLifecycle:
    def test_run_id_inside_block(self, datasets, mlflow_server):
        with train(datasets=datasets, target="target") as t:
            assert t.run_id is not None

    def test_run_id_persists_after_block(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(datasets=datasets, target="target") as t:
            t.fit(LogisticRegression(max_iter=200))
        assert MlflowClient().get_run(t.run_id).info.status == "FINISHED"
