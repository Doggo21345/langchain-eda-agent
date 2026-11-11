"""Deterministic profiling functions.

Every function takes a `Dataset` and returns plain, JSON-serialisable dicts so
the same results can feed the LLM agent, the Markdown report, and the run log.
Exact counts (rows, nulls, min/max/mean) come from the full-file scan; shape
statistics (quantiles, skew, correlations) come from the in-memory sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Dataset

HIGH_NULL_PCT = 20.0
HIGH_CORR = 0.8
HIGH_CARDINALITY_RATIO = 0.9


def _r(x, nd: int = 4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), nd)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include="number").select_dtypes(exclude="bool").columns)


def categorical_columns(df: pd.DataFrame) -> list[str]:
    num = set(numeric_columns(df))
    return [c for c in df.columns if c not in num and not pd.api.types.is_datetime64_any_dtype(df[c])]


def schema(ds: Dataset) -> dict:
    df = ds.sample
    cols = []
    for c in df.columns:
        cols.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "kind": "numeric" if c in numeric_columns(df) else "categorical",
            "example": None if df[c].dropna().empty else str(df[c].dropna().iloc[0])[:60],
        })
    return {
        "dataset": ds.name,
        "rows": ds.scan.row_count,
        "columns": len(df.columns),
        "sampled": ds.sampled,
        "sample_rows": len(df),
        "source": ds.path,
        "file_mb": _r(ds.file_bytes / 1e6, 2) if ds.file_bytes else None,
        "schema_hash": ds.schema_hash,
        "schema": cols,
    }


def null_audit(ds: Dataset) -> dict:
    n = max(ds.scan.row_count, 1)
    rows = []
    for c in ds.sample.columns:
        k = ds.scan.null_counts.get(c, 0)
        rows.append({"column": c, "nulls": k, "null_pct": _r(100 * k / n, 2)})
    rows.sort(key=lambda r: r["nulls"], reverse=True)
    total_cells = n * max(len(ds.sample.columns), 1)
    return {
        "rows": ds.scan.row_count,
        "overall_null_pct": _r(100 * sum(r["nulls"] for r in rows) / total_cells, 2),
        "columns_with_nulls": sum(1 for r in rows if r["nulls"]),
        "fully_null_columns": [r["column"] for r in rows if r["nulls"] == ds.scan.row_count and n],
        "high_null_columns": [r["column"] for r in rows if (r["null_pct"] or 0) >= HIGH_NULL_PCT],
        "by_column": rows,
    }


def numeric_distribution(ds: Dataset, column: str) -> dict:
    s = ds.sample[column].dropna().astype(float)
    if s.empty:
        return {"column": column, "error": "no non-null values"}
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    outliers = int(((s < lo) | (s > hi)).sum())
    skew = s.skew()
    return {
        "column": column,
        "count_non_null": int(ds.scan.row_count - ds.scan.null_counts.get(column, 0)),
        "mean": _r(ds.scan.numeric_mean.get(column, s.mean())),
        "std": _r(s.std()),
        "min": _r(ds.scan.numeric_min.get(column, s.min())),
        "p01": _r(s.quantile(0.01)),
        "p25": _r(q1),
        "median": _r(s.median()),
        "p75": _r(q3),
        "p99": _r(s.quantile(0.99)),
        "max": _r(ds.scan.numeric_max.get(column, s.max())),
        "skew": _r(skew, 3),
        "shape": "right-skewed" if skew > 1 else "left-skewed" if skew < -1 else "roughly symmetric",
        "zeros_pct": _r(100 * (s == 0).mean(), 2),
        "negative_pct": _r(100 * (s < 0).mean(), 2),
        "iqr_outliers": outliers,
        "iqr_outlier_pct": _r(100 * outliers / len(s), 2),
        "distinct": int(s.nunique()),
    }


def categorical_summary(ds: Dataset, column: str, top_n: int = 10) -> dict:
    s = ds.sample[column]
    non_null = s.dropna()
    vc = non_null.astype(str).value_counts()
    n = max(len(non_null), 1)
    return {
        "column": column,
        "distinct": int(vc.size),
        "cardinality_ratio": _r(vc.size / n),
        "likely_identifier": vc.size / n >= HIGH_CARDINALITY_RATIO and len(non_null) > 50,
        "top_values": [{"value": v, "count": int(c), "pct": _r(100 * c / n, 2)} for v, c in vc.head(top_n).items()],
        "single_value": vc.size == 1,
    }


def distributions(ds: Dataset) -> dict:
    df = ds.sample
    return {
        "numeric": [numeric_distribution(ds, c) for c in numeric_columns(df)],
        "categorical": [categorical_summary(ds, c, top_n=5) for c in categorical_columns(df)],
    }


def correlations(ds: Dataset, method: str = "pearson", top_n: int = 15) -> dict:
    num = ds.sample[numeric_columns(ds.sample)]
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] < 2:
        return {"method": method, "matrix": {}, "top_pairs": [], "note": "fewer than 2 varying numeric columns"}
    corr = num.corr(method=method)
    pairs = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = corr.loc[a, b]
            if pd.notna(v):
                pairs.append({"a": a, "b": b, "r": _r(v, 3)})
    pairs.sort(key=lambda p: abs(p["r"]), reverse=True)
    return {
        "method": method,
        "matrix": {a: {b: _r(corr.loc[a, b], 3) for b in cols} for a in cols},
        "top_pairs": pairs[:top_n],
        "highly_correlated": [p for p in pairs if abs(p["r"]) >= HIGH_CORR],
    }


def duplicates(ds: Dataset) -> dict:
    d = ds.scan.duplicate_rows
    return {
        "duplicate_rows": d,
        "duplicate_pct": _r(100 * d / max(ds.scan.row_count, 1), 2),
        "note": "exact, across all rows",
    }


def quality_issues(ds: Dataset) -> dict:
    """Rule-based data quality checks, each with a severity."""
    issues: list[dict] = []
    nulls = null_audit(ds)
    for c in nulls["fully_null_columns"]:
        issues.append({"severity": "high", "column": c, "issue": "column is entirely null"})
    for r in nulls["by_column"]:
        if r["column"] not in nulls["fully_null_columns"] and (r["null_pct"] or 0) >= HIGH_NULL_PCT:
            issues.append({"severity": "medium", "column": r["column"], "issue": f"{r['null_pct']}% null"})

    for c in numeric_columns(ds.sample):
        d = numeric_distribution(ds, c)
        if "error" in d:
            continue
        if d["distinct"] == 1:
            issues.append({"severity": "medium", "column": c, "issue": "constant value"})
        if (d["iqr_outlier_pct"] or 0) >= 5:
            issues.append({"severity": "low", "column": c, "issue": f"{d['iqr_outlier_pct']}% IQR outliers"})
        if 0 < (d["negative_pct"] or 0) < 1:
            issues.append({"severity": "medium", "column": c,
                           "issue": f"{d['negative_pct']}% negative values in a mostly positive column (min={d['min']}); possible sentinel or bad rows"})
        if abs(d["skew"] or 0) > 3:
            issues.append({"severity": "low", "column": c, "issue": f"heavily skewed (skew={d['skew']})"})

    for c in categorical_columns(ds.sample):
        cs = categorical_summary(ds, c)
        if cs["single_value"]:
            issues.append({"severity": "medium", "column": c, "issue": "single distinct value"})
        elif cs["likely_identifier"]:
            issues.append({"severity": "info", "column": c, "issue": "near-unique values (likely an ID)"})
        if _looks_numeric(ds.sample[c]):
            issues.append({"severity": "medium", "column": c, "issue": "numeric values stored as text"})
        elif _looks_datetime(ds.sample[c]):
            issues.append({"severity": "low", "column": c, "issue": "dates stored as text; parse before time-based analysis"})

    dup = duplicates(ds)
    if dup["duplicate_rows"]:
        issues.append({"severity": "medium", "column": None, "issue": f"{dup['duplicate_pct']}% duplicate rows"})

    for p in correlations(ds).get("highly_correlated", []):
        issues.append({"severity": "info", "column": f"{p['a']} ~ {p['b']}", "issue": f"highly correlated (r={p['r']})"})

    weights = {"high": 15, "medium": 5, "low": 2, "info": 0}
    score = max(0, 100 - sum(weights[i["severity"]] for i in issues))
    return {"quality_score": score, "issue_count": len(issues), "issues": issues}


def _looks_numeric(s: pd.Series) -> bool:
    vals = s.dropna().astype(str).str.strip().head(500)
    if vals.empty:
        return False
    parsed = pd.to_numeric(vals.str.replace(r"[,$%]", "", regex=True), errors="coerce")
    return parsed.notna().mean() > 0.95


def _looks_datetime(s: pd.Series) -> bool:
    vals = s.dropna().astype(str).str.strip().head(500)
    if vals.empty or not vals.str.contains(r"\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4}").mean() > 0.95:
        return False
    return pd.to_datetime(vals, errors="coerce", format="mixed").notna().mean() > 0.95


def group_aggregate(ds: Dataset, by: str, column: str, agg: str = "mean", top_n: int = 20) -> dict:
    allowed = {"mean", "median", "sum", "count", "min", "max", "std", "nunique"}
    if agg not in allowed:
        raise ValueError(f"agg must be one of {sorted(allowed)}")
    g = ds.sample.groupby(by, dropna=False)[column].agg(agg).sort_values(ascending=False).head(top_n)
    return {
        "by": by, "column": column, "agg": agg,
        "on_sample": ds.sampled,
        "rows": [{"group": str(k), "value": _r(v)} for k, v in g.items()],
    }


def full_profile(ds: Dataset) -> dict:
    return {
        "schema": schema(ds),
        "nulls": null_audit(ds),
        "distributions": distributions(ds),
        "correlations": correlations(ds),
        "duplicates": duplicates(ds),
        "quality": quality_issues(ds),
    }
