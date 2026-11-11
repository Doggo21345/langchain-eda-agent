import sqlite3

import pandas as pd
import pytest

from eda_agent import cli
from eda_agent.data import load_dataset, load_sql, redact_url


@pytest.fixture
def db(messy_csv, tmp_path):
    path = tmp_path / "shop.db"
    df = pd.read_csv(messy_csv, keep_default_na=False, na_values=[""])
    with sqlite3.connect(path) as con:
        df.to_sql("events", con, index=False)
    return f"sqlite:///{path}"


def test_table_matches_file_scan(db, messy_csv):
    from_db = load_sql(db, table="events", chunk_rows=300)
    from_file = load_dataset(messy_csv)
    assert from_db.name == "events"
    assert from_db.scan.row_count == from_file.scan.row_count
    assert from_db.scan.null_counts == from_file.scan.null_counts
    assert from_db.scan.duplicate_rows == 10


def test_query_source(db):
    ds = load_sql(db, query="SELECT region, x FROM events WHERE region = 'EU'")
    assert set(ds.sample["region"]) == {"EU"} and list(ds.sample.columns) == ["region", "x"]


def test_unknown_table_rejected(db):
    with pytest.raises(ValueError, match="not found"):
        load_sql(db, table="events; DROP TABLE events")


def test_password_redacted():
    assert "hunter2" not in redact_url("postgresql://me:hunter2@db.example.com:5432/shop")


def test_cli_profiles_table_and_logs_timing(db, tmp_path, monkeypatch):
    monkeypatch.setenv("EDA_DB_URL", db)
    out = tmp_path / "out"
    assert cli.main(["--out", str(out), "--log-level", "WARNING", "profile", "--table", "events", "--no-plots"]) == 0
    assert (out / "events_report.md").exists()
    (run,) = [__import__("json").loads(line) for line in (out / "runs.jsonl").read_text().splitlines()]
    assert run["rows"] == 2010 and run["run_seconds"] >= run["load_seconds"] > 0
    assert "table=events" in run["path"]
