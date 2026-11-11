import pandas as pd

from eda_agent.data import load_dataset
from eda_agent.run_log import RunLogger


def test_detects_row_drop_schema_change_and_null_jump(messy_csv, tmp_path):
    logger = RunLogger(tmp_path / "runs.jsonl")
    first = logger.record_run(load_dataset(messy_csv))
    assert first["changes"] == []

    df = pd.read_csv(messy_csv, keep_default_na=False, na_values=[""]).head(500).drop(columns=["constant"])
    df.loc[:200, "y"] = None
    df.to_csv(messy_csv, index=False)

    second = RunLogger(tmp_path / "runs.jsonl").record_run(load_dataset(messy_csv))
    kinds = {c["kind"] for c in second["changes"]}
    assert {"row_drop", "columns_removed", "null_jump"} <= kinds
    assert len(logger.history("messy")) == 2


def test_records_unavailable_dataset(tmp_path):
    logger = RunLogger(tmp_path / "runs.jsonl")
    logger.record_missing("orders", tmp_path / "orders.csv", "file not found")
    (row,) = logger.history("orders")
    assert row["available"] is False and row["changes"][0]["kind"] == "unavailable"
