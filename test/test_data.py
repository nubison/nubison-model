"""Unit tests for nubison_model.data (load + split)."""

import boto3
import pandas as pd
import pytest
from moto import mock_aws
from sqlalchemy import create_engine

from nubison_model import SOURCE_URI_ATTR, data


@pytest.fixture
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})


# ---------------------------------------------------------------------------
# load()
# ---------------------------------------------------------------------------


class TestLoadFile:
    def test_load_csv(self, tmp_path, sample_df):
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path, index=False)
        uri = f"file://{csv_path}"

        df = data.load(uri)

        pd.testing.assert_frame_equal(df, sample_df)
        assert df.attrs[SOURCE_URI_ATTR] == uri

    def test_load_parquet(self, tmp_path, sample_df):
        parquet_path = tmp_path / "data.parquet"
        sample_df.to_parquet(parquet_path, index=False)
        uri = f"file://{parquet_path}"

        df = data.load(uri)

        pd.testing.assert_frame_equal(df, sample_df)
        assert df.attrs[SOURCE_URI_ATTR] == uri

    def test_load_bare_absolute_path(self, tmp_path, sample_df):
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path, index=False)

        df = data.load(str(csv_path))

        pd.testing.assert_frame_equal(df, sample_df)
        assert df.attrs[SOURCE_URI_ATTR] == str(csv_path)

    def test_load_bare_relative_path(self, tmp_path, sample_df, monkeypatch):
        csv_path = tmp_path / "data.csv"
        sample_df.to_csv(csv_path, index=False)
        monkeypatch.chdir(tmp_path)

        df = data.load("data.csv")

        pd.testing.assert_frame_equal(df, sample_df)
        assert df.attrs[SOURCE_URI_ATTR] == "data.csv"


class TestLoadS3:
    @mock_aws
    def test_load_csv_from_s3(self, sample_df):
        s3 = boto3.client("s3", region_name="us-east-1")
        bucket = "test-bucket"
        key = "path/to/train.csv"
        s3.create_bucket(Bucket=bucket)
        s3.put_object(Bucket=bucket, Key=key, Body=sample_df.to_csv(index=False))

        uri = f"s3://{bucket}/{key}"
        df = data.load(uri)

        pd.testing.assert_frame_equal(df, sample_df)
        assert df.attrs[SOURCE_URI_ATTR] == uri

    @mock_aws
    def test_load_parquet_from_s3(self, sample_df, tmp_path):
        parquet_path = tmp_path / "tmp.parquet"
        sample_df.to_parquet(parquet_path, index=False)

        s3 = boto3.client("s3", region_name="us-east-1")
        bucket = "test-bucket-parquet"
        key = "data/train.parquet"
        s3.create_bucket(Bucket=bucket)
        s3.put_object(Bucket=bucket, Key=key, Body=parquet_path.read_bytes())

        uri = f"s3://{bucket}/{key}"
        df = data.load(uri)

        pd.testing.assert_frame_equal(df, sample_df)
        assert df.attrs[SOURCE_URI_ATTR] == uri

    def test_invalid_s3_uri(self):
        with pytest.raises(ValueError, match="Invalid s3:// URI"):
            data.load("s3://bucket-only")


class TestLoadSqlPrivateHelper:
    """SQL loading by raw URI is not part of the public ``data.load``
    surface — it would force credentials into notebook cells. The
    private helper is reused by ``data.connection(name).load(query)``."""

    def test_load_sqlite_via_private_helper(self, sample_df, tmp_path):
        db_path = tmp_path / "test.db"
        uri = f"sqlite:///{db_path}"
        engine = create_engine(uri)
        sample_df.to_sql("data", engine, index=False)
        engine.dispose()

        df = data._load_sql(uri, "SELECT * FROM data")
        pd.testing.assert_frame_equal(df, sample_df)


# ---------------------------------------------------------------------------
# connection()
# ---------------------------------------------------------------------------

import base64
import json


def _encode_conn(info: dict) -> str:
    return base64.b64encode(json.dumps(info).encode()).decode()


class TestResolveDbInfo:
    def test_env_var_takes_precedence(self, monkeypatch, tmp_path):
        env_info = {"db_id": "X", "db_type": "2", "db_host": "from-env"}
        file_info = {"X": {"db_id": "X", "db_type": "2", "db_host": "from-file"}}
        conf = tmp_path / "db_conf.json"
        conf.write_text(json.dumps(file_info))
        monkeypatch.setenv("JUPYTERLAB_SQL_EXPLORER_DB_CONF", str(conf))
        monkeypatch.setenv("DB_X", _encode_conn(env_info))

        assert data._resolve_db_info("X")["db_host"] == "from-env"

    def test_file_fallback(self, monkeypatch, tmp_path):
        """SQL Explorer UI persists entries keyed by numeric id with the
        user-facing name stored in the ``name`` field. Resolution must
        match on that inner field, not on the outer key."""
        info = {
            "name": "MYDB",
            "db_id": "1",
            "db_type": "2",
            "db_host": "pg.example.svc",
            "db_port": "5432",
            "db_name": "appdb",
            "db_user": "u",
            "db_pass": "p",
        }
        conf = tmp_path / "db_conf.json"
        conf.write_text(json.dumps({"1": info}))
        monkeypatch.delenv("DB_MYDB", raising=False)
        monkeypatch.setenv("JUPYTERLAB_SQL_EXPLORER_DB_CONF", str(conf))

        assert data._resolve_db_info("MYDB") == info

    def test_not_found_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DB_MISSING", raising=False)
        monkeypatch.setenv(
            "JUPYTERLAB_SQL_EXPLORER_DB_CONF", str(tmp_path / "nope.json")
        )
        with pytest.raises(KeyError, match="MISSING"):
            data._resolve_db_info("MISSING")


class TestBuildSqlalchemyUri:
    def test_postgresql(self):
        url = data._build_sqlalchemy_uri({
            "db_type": "2", "db_host": "h", "db_port": 5432,
            "db_name": "d", "db_user": "u", "db_pass": "p@ss",
        })
        assert url.render_as_string(hide_password=False) == (
            "postgresql+psycopg2://u:p%40ss@h:5432/d"
        )

    def test_mysql(self):
        url = data._build_sqlalchemy_uri({
            "db_type": "1", "db_host": "h", "db_port": 3306,
            "db_name": "d", "db_user": "u", "db_pass": "p",
        })
        assert url.render_as_string(hide_password=False) == (
            "mysql+pymysql://u:p@h:3306/d"
        )

    def test_oracle(self):
        url = data._build_sqlalchemy_uri({
            "db_type": "4", "db_host": "h", "db_port": 1521,
            "db_name": "d", "db_user": "u", "db_pass": "p",
        })
        assert url.render_as_string(hide_password=False) == (
            "oracle+cx_oracle://u:p@h:1521/d"
        )

    def test_sqlite_uses_db_name_as_path(self, tmp_path):
        url = data._build_sqlalchemy_uri({
            "db_type": "3", "db_name": str(tmp_path / "f.db"),
        })
        assert url.render_as_string(hide_password=False) == (
            f"sqlite:///{tmp_path / 'f.db'}"
        )

    def test_password_masked_in_string_repr(self):
        url = data._build_sqlalchemy_uri({
            "db_type": "2", "db_host": "h", "db_port": 5432,
            "db_name": "d", "db_user": "u", "db_pass": "supersecret",
        })
        assert "supersecret" not in str(url)
        assert "supersecret" not in repr(url)

    def test_unsupported_db_type(self):
        with pytest.raises(ValueError, match="Unsupported db_type"):
            data._build_sqlalchemy_uri({"db_type": "999"})

    def test_invalid_port_raises_value_error(self):
        with pytest.raises(ValueError, match="db_port must be an integer"):
            data._build_sqlalchemy_uri({
                "db_type": "2", "db_host": "h", "db_port": "5abc",
                "db_name": "d", "db_user": "u", "db_pass": "p",
            })


class TestConnection:
    def test_load_returns_dataframe_with_credential_free_source_uri(
        self, sample_df, monkeypatch, tmp_path
    ):
        db_path = tmp_path / "demo.db"
        engine = create_engine(f"sqlite:///{db_path}")
        sample_df.to_sql("iris", engine, index=False)
        engine.dispose()

        info = {"db_id": "DEMO", "db_type": "3", "db_name": str(db_path)}
        monkeypatch.setenv("DB_DEMO", _encode_conn(info))

        db = data.connection("DEMO")
        query = "SELECT * FROM iris"
        df = db.load(query)

        pd.testing.assert_frame_equal(df, sample_df)
        # No password / host / user in the source_uri
        source_uri = df.attrs[SOURCE_URI_ATTR]
        assert source_uri.startswith("dbref://DEMO#")
        for forbidden in ("password", "p@ss", str(db_path)):
            assert forbidden not in source_uri

    def test_different_queries_get_distinct_hashes(
        self, sample_df, monkeypatch, tmp_path
    ):
        db_path = tmp_path / "demo.db"
        engine = create_engine(f"sqlite:///{db_path}")
        sample_df.to_sql("iris", engine, index=False)
        engine.dispose()

        info = {"db_id": "DEMO", "db_type": "3", "db_name": str(db_path)}
        monkeypatch.setenv("DB_DEMO", _encode_conn(info))

        db = data.connection("DEMO")
        df1 = db.load("SELECT * FROM iris WHERE a = 1")
        df2 = db.load("SELECT * FROM iris WHERE a = 2")

        assert df1.attrs[SOURCE_URI_ATTR] != df2.attrs[SOURCE_URI_ATTR]
        assert df1.attrs[SOURCE_URI_ATTR].startswith("dbref://DEMO#")
        assert df2.attrs[SOURCE_URI_ATTR].startswith("dbref://DEMO#")

    def test_missing_connection_raises(self, monkeypatch, tmp_path):
        monkeypatch.delenv("DB_NOPE", raising=False)
        monkeypatch.setenv(
            "JUPYTERLAB_SQL_EXPLORER_DB_CONF", str(tmp_path / "nope.json")
        )
        with pytest.raises(KeyError, match="NOPE"):
            data.connection("NOPE")


class TestLoadUnsupportedExtension:
    def test_json_file_raises(self, tmp_path):
        p = tmp_path / "data.json"
        p.write_text('[{"a": 1}]')
        with pytest.raises(ValueError, match="Unsupported file extension"):
            data.load(str(p))

    def test_xlsx_file_raises(self, tmp_path):
        p = tmp_path / "data.xlsx"
        p.write_bytes(b"PK\x03\x04")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            data.load(str(p))

    def test_no_extension_raises(self, tmp_path):
        p = tmp_path / "data_no_ext"
        p.write_text("a,b\n1,2\n")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            data.load(str(p))


class TestLoadUnsupportedScheme:
    def test_unsupported_scheme_raises(self):
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            data.load("ftp://example.com/file.csv")

    def test_postgresql_uri_no_longer_supported(self):
        # SQL by raw URI was intentionally removed; see Data.py docstring.
        with pytest.raises(ValueError, match="Unsupported URI scheme"):
            data.load("postgresql://user:pw@host/db")

    def test_empty_scheme_treated_as_path(self):
        # Bare paths are valid (treated as local file); we surface the
        # underlying FileNotFoundError instead of a scheme error.
        with pytest.raises(FileNotFoundError):
            data.load("/no/scheme/here.csv")


# ---------------------------------------------------------------------------
# split()
# ---------------------------------------------------------------------------


@pytest.fixture
def split_df():
    return pd.DataFrame(
        {
            "x": list(range(100)),
            "target": [i % 2 for i in range(100)],
        }
    )


class TestSplitBasic:
    def test_two_way_split_sizes(self, split_df):
        result = data.split(
            split_df, {"training": 0.8, "evaluation": 0.2}, shuffle=False
        )
        assert set(result.keys()) == {"training", "evaluation"}
        assert len(result["training"]) == 80
        assert len(result["evaluation"]) == 20

    def test_three_way_split_sizes(self, split_df):
        result = data.split(
            split_df,
            {"training": 0.7, "evaluation": 0.2, "test": 0.1},
            shuffle=False,
        )
        assert len(result["training"]) == 70
        assert len(result["evaluation"]) == 20
        assert len(result["test"]) == 10

    def test_no_shuffle_preserves_order(self, split_df):
        result = data.split(
            split_df, {"training": 0.5, "evaluation": 0.5}, shuffle=False
        )
        assert list(result["training"]["x"]) == list(range(50))
        assert list(result["evaluation"]["x"]) == list(range(50, 100))

    def test_total_rows_covered(self, split_df):
        result = data.split(split_df, {"a": 0.33, "b": 0.33, "c": 0.34}, shuffle=False)
        total = sum(len(v) for v in result.values())
        assert total == len(split_df)


class TestSplitShuffle:
    def test_shuffle_determinism_with_random_state(self, split_df):
        out1 = data.split(
            split_df, {"a": 0.5, "b": 0.5}, shuffle=True, random_state=42
        )
        out2 = data.split(
            split_df, {"a": 0.5, "b": 0.5}, shuffle=True, random_state=42
        )
        pd.testing.assert_frame_equal(out1["a"], out2["a"])
        pd.testing.assert_frame_equal(out1["b"], out2["b"])

    def test_different_seeds_differ(self, split_df):
        out1 = data.split(
            split_df, {"a": 0.5, "b": 0.5}, shuffle=True, random_state=1
        )
        out2 = data.split(
            split_df, {"a": 0.5, "b": 0.5}, shuffle=True, random_state=2
        )
        assert not out1["a"].equals(out2["a"])


class TestSplitSourceUri:
    def test_source_prefix_sets_per_key_attrs(self, split_df):
        result = data.split(
            split_df,
            {"training": 0.5, "evaluation": 0.5},
            shuffle=False,
            source_prefix="s3://bucket/dataset",
        )
        assert result["training"].attrs[SOURCE_URI_ATTR] == "s3://bucket/dataset/training"
        assert result["evaluation"].attrs[SOURCE_URI_ATTR] == "s3://bucket/dataset/evaluation"

    def test_inherits_input_source_uri(self, split_df):
        split_df.attrs[SOURCE_URI_ATTR] = "s3://bucket/data.csv"
        result = data.split(split_df, {"a": 0.5, "b": 0.5}, shuffle=False)
        assert result["a"].attrs[SOURCE_URI_ATTR] == "s3://bucket/data.csv#a"
        assert result["b"].attrs[SOURCE_URI_ATTR] == "s3://bucket/data.csv#b"

    def test_nests_inside_existing_fragment(self, split_df):
        # dbref:// URIs already carry a #query_hash fragment; split
        # nests with "/" to avoid an ugly double-hash form.
        split_df.attrs[SOURCE_URI_ATTR] = "dbref://MYDB#a1b2c3d4"
        result = data.split(split_df, {"a": 0.5, "b": 0.5}, shuffle=False)
        assert result["a"].attrs[SOURCE_URI_ATTR] == "dbref://MYDB#a1b2c3d4/a"
        assert result["b"].attrs[SOURCE_URI_ATTR] == "dbref://MYDB#a1b2c3d4/b"

    def test_memory_fallback_when_no_source(self, split_df):
        result = data.split(split_df, {"a": 0.5, "b": 0.5}, shuffle=False)
        assert result["a"].attrs[SOURCE_URI_ATTR] == "memory://a"
        assert result["b"].attrs[SOURCE_URI_ATTR] == "memory://b"


class TestSplitValidation:
    def test_empty_ratios_raises(self, split_df):
        with pytest.raises(ValueError, match="non-empty"):
            data.split(split_df, {})

    def test_non_positive_ratio_raises(self, split_df):
        with pytest.raises(ValueError, match="must be positive"):
            data.split(split_df, {"a": 0.5, "b": 0.5, "c": 0.0})

    def test_negative_ratio_raises(self, split_df):
        with pytest.raises(ValueError, match="must be positive"):
            data.split(split_df, {"a": 1.5, "b": -0.5})

    def test_ratios_not_summing_to_one_raises(self, split_df):
        with pytest.raises(ValueError, match="sum to 1.0"):
            data.split(split_df, {"a": 0.5, "b": 0.3})

    def test_empty_df_raises(self):
        empty = pd.DataFrame({"x": []})
        with pytest.raises(ValueError, match="Cannot split an empty"):
            data.split(empty, {"a": 1.0})


class TestSplitIntoTrainPipeline:
    def test_round_trip_with_train(self, tmp_path, split_df):
        # Verify the typical flow: load() → split() yields dict suitable for train()
        csv = tmp_path / "data.csv"
        split_df.to_csv(csv, index=False)
        loaded = data.load(f"file://{csv}")
        result = data.split(
            loaded, {"training": 0.8, "evaluation": 0.2}, shuffle=False
        )
        assert set(result.keys()) == {"training", "evaluation"}
        for key, sub in result.items():
            assert SOURCE_URI_ATTR in sub.attrs
            assert sub.attrs[SOURCE_URI_ATTR].endswith(f"#{key}")
