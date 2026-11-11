from eda_agent import profiling
from eda_agent.data import load_dataset


def issues_for(q, column):
    return [i["issue"] for i in q["issues"] if i["column"] == column]


def test_full_scan_is_exact_even_when_sampled(messy_csv):
    full = load_dataset(messy_csv)
    sampled = load_dataset(messy_csv, sample_rows=300, chunk_rows=250)
    assert sampled.sampled and len(sampled.sample) == 300
    assert sampled.scan.row_count == full.scan.row_count == 2010
    assert sampled.scan.null_counts == full.scan.null_counts
    assert sampled.scan.duplicate_rows == full.scan.duplicate_rows == 10
    assert sampled.scan.numeric_min["x"] == -999.0


def test_na_string_is_not_treated_as_null(messy_csv):
    ds = load_dataset(messy_csv)
    assert ds.scan.null_counts["region"] == 0
    assert "NA" in set(ds.sample["region"])


def test_null_audit(messy_csv):
    n = profiling.null_audit(load_dataset(messy_csv))
    assert n["by_column"][0]["column"] == "mostly_null"
    assert "mostly_null" in n["high_null_columns"]


def test_correlation_finds_linear_pair(messy_csv):
    c = profiling.correlations(load_dataset(messy_csv), method="spearman")  # robust to the -999 sentinels
    top = c["top_pairs"][0]
    assert {top["a"], top["b"]} == {"x", "y"}


def test_quality_checks(messy_csv):
    q = profiling.quality_issues(load_dataset(messy_csv))
    assert any("null" in i for i in issues_for(q, "mostly_null"))
    assert "constant value" in issues_for(q, "constant")
    assert "numeric values stored as text" in issues_for(q, "price_text")
    assert any("dates stored as text" in i for i in issues_for(q, "day"))
    assert any("negative" in i for i in issues_for(q, "x"))
    assert any("likely an ID" in i for i in issues_for(q, "id"))
    assert any("duplicate" in i["issue"] for i in q["issues"])
    assert 0 <= q["quality_score"] < 100


def test_group_aggregate(messy_csv):
    g = profiling.group_aggregate(load_dataset(messy_csv), "region", "y", "count")
    assert {r["group"] for r in g["rows"]} == {"NA", "EU"}
