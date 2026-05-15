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
        "memory://train",
    )
    val_df = _df_with_uri(
        {"a": [1, 2, 3, 4], "b": [2, 3, 4, 5], "target": [0, 1, 0, 1]},
        "memory://val",
    )
    return {"train": train_df, "val": val_df}


# ---------------------------------------------------------------------------
# TrainContext shape / lifecycle
# ---------------------------------------------------------------------------


class TestTrainContextShape:
    def test_unpacks_train_and_val(self, datasets, mlflow_server):
        with train(datasets=datasets, target="target") as t:
            assert list(t.X_train.columns) == ["a", "b"]
            assert t.y_train.tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
            assert list(t.X_val.columns) == ["a", "b"]
            assert t.y_val.tolist() == [0, 1, 0, 1]

    def test_no_val_dataset(self, mlflow_server):
        df = _df_with_uri(
            {"a": [1, 2, 3, 4], "target": [0, 1, 0, 1]}, "memory://train"
        )
        with train(datasets={"train": df}, target="target") as t:
            assert t.X_val is None
            assert t.y_val is None

    def test_test_split_recognized(self, datasets, mlflow_server):
        # Add a "test" split — should populate X_test / y_test.
        ext = dict(datasets)
        ext["test"] = _df_with_uri(
            {"a": [1, 2], "b": [2, 3], "target": [0, 1]}, "memory://test"
        )
        with train(datasets=ext, target="target") as t:
            assert list(t.X_test.columns) == ["a", "b"]
            assert t.y_test.tolist() == [0, 1]

    def test_no_test_dataset(self, datasets, mlflow_server):
        with train(datasets=datasets, target="target") as t:
            assert t.X_test is None
            assert t.y_test is None

    def test_target_as_list(self, mlflow_server):
        df = _df_with_uri(
            {"a": [1, 2, 3, 4], "t1": [0, 1, 0, 1], "t2": [1, 0, 1, 0]},
            "memory://train",
        )
        with train(datasets={"train": df}, target=["t1", "t2"]) as t:
            assert list(t.X_train.columns) == ["a"]
            assert list(t.y_train.columns) == ["t1", "t2"]

    def test_preserves_raw_datasets(self, datasets, mlflow_server):
        with train(datasets=datasets, target="target") as t:
            assert t.datasets is datasets


class TestTrainContextErrors:
    def test_missing_train_raises(self, mlflow_server):
        df = _df_with_uri({"a": [1], "target": [0]}, "memory://val")
        with pytest.raises(KeyError, match="train"):
            with train(datasets={"val": df}, target="target"):
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
            assert (fitted.predict(t.X_train) == fitted.predict(t.X_train)).all()

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

    def test_fit_logs_val_accuracy_for_classifier(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(
            datasets=datasets, target="target", model_type="classifier"
        ) as t:
            t.fit(LogisticRegression(max_iter=200))
        run = MlflowClient().get_run(t.run_id)
        assert "val_accuracy" in run.data.metrics

    def test_fit_without_model_type_skips_val_metric(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(datasets=datasets, target="target") as t:
            t.fit(LogisticRegression(max_iter=200))
        run = MlflowClient().get_run(t.run_id)
        assert "val_accuracy" not in run.data.metrics
        assert "val_r2" not in run.data.metrics

    def test_fit_logs_val_r2_for_regressor(
        self, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        train_df = _df_with_uri(
            {"a": list(range(20)), "target": [2 * x for x in range(20)]},
            "memory://train",
        )
        val_df = _df_with_uri(
            {"a": list(range(5)), "target": [2 * x for x in range(5)]},
            "memory://val",
        )
        with train(
            datasets={"train": train_df, "val": val_df},
            target="target",
            model_type="regressor",
        ) as t:
            t.fit(LinearRegression())
        run = MlflowClient().get_run(t.run_id)
        assert "val_r2" in run.data.metrics


# ---------------------------------------------------------------------------
# t.log_metric / t.log_param — custom loop path
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
# t.save()
# ---------------------------------------------------------------------------


class TestSave:
    def test_save_default_path(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(datasets=datasets, target="target") as t:
            model = LogisticRegression(max_iter=200).fit(t.X_train, t.y_train)
            t.save(model)
        assert (tmp_path / "src" / "weights.pkl").exists()

    def test_save_custom_path(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(datasets=datasets, target="target") as t:
            model = LogisticRegression(max_iter=200).fit(t.X_train, t.y_train)
            t.save(model, weights_path="models/v1.pkl")
        assert (tmp_path / "models" / "v1.pkl").exists()

    def test_save_logs_artifact(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(datasets=datasets, target="target") as t:
            model = LogisticRegression(max_iter=200).fit(t.X_train, t.y_train)
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
        assert "train" in names
        assert "val" in names

    def test_dataset_source_uri_mirrored_as_tag(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        # mlflow rejects unknown schemes for dataset source — we mirror the
        # original source_uri (dbref://, memory://, ...) as a run tag so
        # the nubison UI can still render the logical lineage.
        monkeypatch.chdir(tmp_path)
        with train(datasets=datasets, target="target") as t:
            t.fit(LogisticRegression(max_iter=200))
        run = MlflowClient().get_run(t.run_id)
        assert run.data.tags["nubison.dataset.train.source"] == "memory://train"
        assert run.data.tags["nubison.dataset.val.source"] == "memory://val"


class TestModelTypeTag:
    def test_model_type_tag_set(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        monkeypatch.chdir(tmp_path)
        with train(
            datasets=datasets, target="target", model_type="classifier"
        ) as t:
            t.fit(LogisticRegression(max_iter=200))
        run = MlflowClient().get_run(t.run_id)
        assert run.data.tags["model_type"] == "classifier"

    def test_free_form_model_type_accepted(
        self, datasets, tmp_path, monkeypatch, mlflow_server
    ):
        # Not "classifier"/"regressor" — still tagged, no auto-metric.
        monkeypatch.chdir(tmp_path)
        with train(
            datasets=datasets, target="target", model_type="clustering"
        ) as t:
            t.fit(LogisticRegression(max_iter=200))
        run = MlflowClient().get_run(t.run_id)
        assert run.data.tags["model_type"] == "clustering"
        assert "val_accuracy" not in run.data.metrics


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
