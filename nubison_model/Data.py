"""Data loading and splitting utilities.

- ``load(uri)``: fetch a DataFrame from ``s3://`` / ``file://`` URIs or a
  bare local path. S3 credentials are resolved from environment
  (consumer's PodDefault injects STS WebIdentity).
- ``connection(name)``: bind to a JupyterLab SQL Explorer DB connection
  by name. Returns an object whose ``.load(query)`` runs a query and
  returns a DataFrame with credential-free lineage.
- ``split(df, ratios)``: auto-split a DataFrame into multiple subsets by
  ratio with optional shuffle. Each output carries a derived
  ``attrs["source_uri"]`` so downstream ``train()`` keeps the lineage.

All entry points record the source URI in ``df.attrs["source_uri"]`` so
``train()`` picks it up automatically for ``mlflow.log_input(source=...)``
lineage. The user does not name the URI twice. For ``connection().load()``
the recorded URI is ``dbref://<name>#<query_hash>`` — the password is
never logged.

SQL loading by raw URI (``postgresql://user:pw@host/db``) is intentionally
not part of the public surface — it would force users to embed
credentials in notebook cells. Use ``data.connection(name)`` instead.
"""

import base64
import hashlib
import io
import json
import logging
import math
import pathlib
from os import getenv
from typing import TYPE_CHECKING, Dict, Optional, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from nubison_model.Storage import ENV_VAR_AWS_ENDPOINT_URL

if TYPE_CHECKING:
    from sqlalchemy import URL

logger = logging.getLogger(__name__)

SOURCE_URI_ATTR = "source_uri"
RATIO_SUM_TOLERANCE = 1e-6
SUPPORTED_FILE_EXTENSIONS = (".csv", ".parquet")


def _read_by_extension(path_or_buf, source: str) -> pd.DataFrame:
    """Dispatch on file extension; raise ValueError on unsupported types.

    ``source`` is the URI / path used purely for the error message.
    """
    src_lower = source.lower()
    if src_lower.endswith(".parquet"):
        return pd.read_parquet(path_or_buf)
    if src_lower.endswith(".csv"):
        return pd.read_csv(path_or_buf)
    raise ValueError(
        f"Unsupported file extension for {source!r}. "
        f"Supported: {SUPPORTED_FILE_EXTENSIONS}"
    )

# JupyterLab SQL Explorer integration -----------------------------------
# - Pod injects a saved connection as ``DB_<NAME>`` env var (base64 of
#   the JSON dict) OR
# - the user adds it via the UI, which writes the same dict to
#   ``~/.local/share/jupyterlab-sql-explorer/db_conf.json``.
ENV_VAR_DB_PREFIX = "DB_"
ENV_VAR_DB_CONF_PATH = "JUPYTERLAB_SQL_EXPLORER_DB_CONF"
DEFAULT_DB_CONF_PATH = "~/.local/share/jupyterlab-sql-explorer/db_conf.json"

# jupyterlab-sql-explorer encodes DB types as integer-strings. mlplatform
# patches the explorer's ``_getSQL_engine`` to handle each value; this
# mapping mirrors the codes (assumed once the multi-DB patch lands).
DB_TYPE_TO_SCHEME: Dict[str, str] = {
    "1": "mysql+pymysql",
    "2": "postgresql+psycopg2",
    "3": "sqlite",
    "4": "oracle+cx_oracle",
    "5": "mssql+pyodbc",
    "8": "hive",  # Spark via Thrift Server
}


def load(uri: str) -> pd.DataFrame:
    """Load a DataFrame from a URI.

    The resulting DataFrame has ``df.attrs["source_uri"] = uri`` so
    downstream ``train()`` can pick it up for mlflow lineage.

    Args:
        uri: Source URI. Supported forms:
            - ``s3://bucket/key`` — boto3 with optional ``AWS_ENDPOINT_URL``
              env (used by mlplatform's PodDefault for STS WebIdentity).
              File format is detected from the extension (.parquet or .csv).
            - ``file:///abs/path/to/file.csv`` or a bare path
              (``"/tmp/data.csv"``, ``"data/train.parquet"``) —
              pandas.read_csv / pandas.read_parquet (extension-detected).

    Returns:
        DataFrame with ``attrs["source_uri"] = uri``.

    Raises:
        ValueError: Unsupported URI scheme, or the path's extension is
            not in ``SUPPORTED_FILE_EXTENSIONS`` (``.csv``, ``.parquet``).
    """
    scheme = urlparse(uri).scheme
    if scheme == "s3":
        df = _load_s3(uri)
    elif scheme in ("file", ""):
        df = _load_file(uri)
    else:
        raise ValueError(
            f"Unsupported URI scheme: {scheme!r}. "
            "Supported: s3://, file:// (or bare path)"
        )
    df.attrs[SOURCE_URI_ATTR] = uri
    return df


def split(
    df: pd.DataFrame,
    ratios: Dict[str, float],
    *,
    shuffle: bool = True,
    random_state: Optional[int] = None,
    source_prefix: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Auto-split a DataFrame into multiple subsets by ratio.

    Args:
        df: The input DataFrame.
        ratios: Mapping ``{name: ratio}`` whose values must sum to
            approximately ``1.0`` (within ``1e-6``). Keys become the
            output dict keys. Example:
            ``{"train": 0.8, "val": 0.2}``.
        shuffle: If ``True`` (default), shuffle the rows before
            splitting. The original ``df`` is not mutated.
        random_state: Seed for the shuffle. ``None`` (default) is
            non-deterministic.
        source_prefix: If provided, each output's
            ``attrs["source_uri"]`` is set to ``f"{source_prefix}/{key}"``.
            If omitted, falls back to ``"{input_source_uri}#{key}"`` when
            the input has a ``source_uri`` attr; otherwise to
            ``f"memory://{key}"``.

    Returns:
        Mapping ``{name: DataFrame}`` matching the keys of ``ratios``.
        Each output DataFrame carries an ``attrs["source_uri"]``
        derived as described above.

    Raises:
        ValueError: ``ratios`` is empty, contains non-positive entries,
            its values do not sum to ``1.0`` within tolerance, or
            ``df`` is empty.
    """
    if not ratios:
        raise ValueError("`ratios` must be a non-empty mapping")
    if any(v <= 0 for v in ratios.values()):
        raise ValueError(
            f"All ratios must be positive (got: {dict(ratios)})"
        )
    total = sum(ratios.values())
    if not math.isclose(total, 1.0, abs_tol=RATIO_SUM_TOLERANCE):
        raise ValueError(
            f"Ratios must sum to 1.0 (got: {total} for {dict(ratios)})"
        )

    n = len(df)
    if n == 0:
        raise ValueError("Cannot split an empty DataFrame")

    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n)
        shuffled = df.iloc[indices].reset_index(drop=True)
    else:
        shuffled = df.reset_index(drop=True)

    input_source_uri = df.attrs.get(SOURCE_URI_ATTR)

    keys = list(ratios.keys())
    sizes = [int(ratios[k] * n) for k in keys]
    # Assign any rounding remainder to the last key so the splits cover
    # exactly len(df) rows.
    remainder = n - sum(sizes)
    if sizes:
        sizes[-1] += remainder

    result: Dict[str, pd.DataFrame] = {}
    start = 0
    for key, size in zip(keys, sizes):
        subset = shuffled.iloc[start : start + size].reset_index(drop=True)
        if source_prefix is not None:
            subset.attrs[SOURCE_URI_ATTR] = f"{source_prefix}/{key}"
        elif input_source_uri is not None:
            # Nest inside an existing fragment with "/" to avoid an
            # ugly double-hash form like "dbref://X#hash#training".
            sep = "/" if "#" in input_source_uri else "#"
            subset.attrs[SOURCE_URI_ATTR] = f"{input_source_uri}{sep}{key}"
        else:
            subset.attrs[SOURCE_URI_ATTR] = f"memory://{key}"
        result[key] = subset
        start += size

    return result


def _resolve_db_info(name: str) -> dict:
    """Resolve a SQL Explorer connection by its user-facing name.

    Lookup order:
    1. ``DB_<NAME>`` env var (base64 of the JSON connection dict). The
       env-var convention is reused by mlplatform PodDefault injection.
    2. ``~/.local/share/jupyterlab-sql-explorer/db_conf.json`` (path
       overridable via ``JUPYTERLAB_SQL_EXPLORER_DB_CONF`` env).
       The SQL Explorer UI keys entries by numeric id with the
       connection name in an inner ``name`` field — match by that
       field, not by the outer key.
    """
    env_var = f"{ENV_VAR_DB_PREFIX}{name}"
    encoded = getenv(env_var)
    if encoded:
        return json.loads(base64.b64decode(encoded))

    conf_path = pathlib.Path(
        getenv(ENV_VAR_DB_CONF_PATH, DEFAULT_DB_CONF_PATH)
    ).expanduser()
    if conf_path.exists():
        cfg = json.loads(conf_path.read_text())
        for entry in cfg.values():
            if isinstance(entry, dict) and entry.get("name") == name:
                return entry

    raise KeyError(
        f"SQL Explorer connection {name!r} not found "
        f"(checked env {env_var} and {conf_path})"
    )


def _build_sqlalchemy_uri(info: dict) -> "URL":
    """Convert a SQL Explorer connection dict to a SQLAlchemy URL.

    Returns a :class:`sqlalchemy.URL` rather than a plain string so the
    password is masked in ``str(url)`` / ``repr(url)`` and therefore in
    any exception traceback that surfaces the URL.
    """
    from sqlalchemy import URL

    db_type = str(info.get("db_type", ""))
    scheme = DB_TYPE_TO_SCHEME.get(db_type)
    if not scheme:
        raise ValueError(
            f"Unsupported db_type {db_type!r}. "
            f"Known: {sorted(DB_TYPE_TO_SCHEME)}"
        )

    database = info.get("db_name", "") or ""
    if scheme == "sqlite":
        return URL.create(drivername="sqlite", database=database)

    raw_port = info.get("db_port") or None
    port: Optional[int]
    if raw_port in (None, ""):
        port = None
    else:
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            raise ValueError(
                f"db_port must be an integer, got {raw_port!r}"
            ) from None

    return URL.create(
        drivername=scheme,
        username=info.get("db_user") or None,
        password=info.get("db_pass") or None,
        host=info.get("db_host") or None,
        port=port,
        database=database,
    )


def _hash_query(query: str) -> str:
    """Short stable hash of a SQL query for lineage identifiers."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:8]


class _DBConnection:
    """A bound JupyterLab SQL Explorer connection.

    Instances are created via :func:`connection`. Each ``.load(query)``
    call returns a DataFrame whose ``source_uri`` is
    ``dbref://<name>#<query_hash>`` — credentials are never logged to
    mlflow.
    """

    def __init__(self, name: str):
        self._name = name
        info = _resolve_db_info(name)
        self._uri = _build_sqlalchemy_uri(info)

    def load(self, query: str) -> pd.DataFrame:
        df = _load_sql(self._uri, query)
        df.attrs[SOURCE_URI_ATTR] = f"dbref://{self._name}#{_hash_query(query)}"
        return df


def connection(name: str) -> _DBConnection:
    """Bind to a SQL Explorer connection by name.

    Looks up ``DB_<name>`` (base64 JSON) then
    ``~/.local/share/jupyterlab-sql-explorer/db_conf.json``. Returns an
    object with ``.load(query) -> DataFrame``.

    Example::

        db = data.connection("MYDB")
        train_df = db.load("SELECT * FROM features WHERE date >= '...'")
        eval_df  = db.load("SELECT * FROM features WHERE date >= '...'")

    Each ``.load`` records ``dbref://<name>#<query_hash>`` as the
    DataFrame's ``source_uri`` — the password never leaves the connection
    object.
    """
    return _DBConnection(name)


def _load_s3(uri: str) -> pd.DataFrame:
    import boto3

    bucket, _, key = uri[len("s3://") :].partition("/")
    if not bucket or not key:
        raise ValueError(f"Invalid s3:// URI: {uri!r}")

    endpoint_url = getenv(ENV_VAR_AWS_ENDPOINT_URL) or None
    s3 = boto3.client("s3", endpoint_url=endpoint_url)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    buf = io.BytesIO(body)

    return _read_by_extension(buf, uri)


def _load_sql(uri: Union[str, "URL"], query: str) -> pd.DataFrame:
    from sqlalchemy import create_engine

    engine = create_engine(uri)
    try:
        return pd.read_sql(query, engine)
    finally:
        engine.dispose()


def _load_file(uri: str) -> pd.DataFrame:
    parsed = urlparse(uri)
    # For "file://..." URIs urlparse populates `.path`; for a bare path
    # ("/tmp/x.csv", "data/x.csv") `.scheme` is "" and `.path == uri`.
    path = parsed.path if parsed.scheme == "file" else uri
    if not path:
        raise ValueError(f"Invalid file URI: {uri!r}")

    return _read_by_extension(path, uri)
