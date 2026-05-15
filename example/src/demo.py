"""Demo scaffolding for ``example/train.ipynb``.

Materializes the iris dataset into a SQLite database and registers it as
a JupyterLab SQL Explorer connection (``DB_<NAME>`` env var, base64 JSON)
so ``data.connection(name)`` resolves it. None of this is shipped with
``register()``; it's only here to keep the notebook self-contained.
"""

import base64
import json
import os
import pathlib
import sqlite3

from sklearn.datasets import load_iris

WORK_DIR = pathlib.Path("/tmp/nubison-train-example")
SQLITE_DB = WORK_DIR / "iris.db"
DB_CONNECTION_NAME = "IRIS"


def prepare() -> str:
    """Write iris to a SQLite DB and inject the SQL Explorer connection
    info into ``DB_IRIS``. Returns the connection name."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    df = load_iris(as_frame=True).frame.rename(
        columns=lambda c: c.replace(" (cm)", "").replace(" ", "_")
    )
    with sqlite3.connect(SQLITE_DB) as conn:
        df.to_sql("iris", conn, index=False, if_exists="replace")

    payload = {
        "db_id": DB_CONNECTION_NAME,
        "db_type": "3",  # sqlite
        "db_name": str(SQLITE_DB),
    }
    os.environ[f"DB_{DB_CONNECTION_NAME}"] = base64.b64encode(
        json.dumps(payload).encode()
    ).decode()

    return DB_CONNECTION_NAME
