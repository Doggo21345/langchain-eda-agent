"""Markdown EDA report built from a full profile."""

from __future__ import annotations

import os
from pathlib import Path


def _table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_none_\n"
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = "".join("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |\n" for r in rows)
    return head + body


def render(profile: dict, plots: list[str], out_path: Path, run: dict | None = None,
           summary: str | None = None) -> Path:
    s, n, q = profile["schema"], profile["nulls"], profile["quality"]
    lines = [f"# EDA report: {s['dataset']}\n"]
    lines.append(
        f"**{s['rows']:,} rows x {s['columns']} columns** · "
        + (f"{s['file_mb']} MB · " if s.get("file_mb") is not None else f"`{s.get('source')}` · ")
        + f"quality score **{q['quality_score']}/100** · schema `{s['schema_hash']}`"
    )
    if s["sampled"]:
        lines.append(f"\nExact counts use all rows; shape statistics use a uniform sample of {s['sample_rows']:,} rows.")
    if summary:
        lines += ["\n## Agent summary\n", summary.strip()]

    if run and run.get("changes"):
        lines += ["\n## Changes since last run\n"]
        lines += [f"- **{c['severity']}** {c['message']}" for c in run["changes"]]

    lines += ["\n## Data quality issues\n", _table(q["issues"], ["severity", "column", "issue"])]
    lines += ["## Schema\n", _table(s["schema"], ["column", "dtype", "kind", "example"])]
    lines += [
        "## Null audit\n",
        f"Overall null rate {n['overall_null_pct']}%; {n['columns_with_nulls']} columns have nulls.\n",
        _table([r for r in n["by_column"] if r["nulls"]], ["column", "nulls", "null_pct"]),
    ]
    num = profile["distributions"]["numeric"]
    lines += ["## Numeric distributions\n", _table(
        [d for d in num if "error" not in d],
        ["column", "mean", "std", "min", "p25", "median", "p75", "max", "skew", "iqr_outlier_pct"],
    )]
    cat = profile["distributions"]["categorical"]
    lines += ["## Categorical columns\n", _table(
        [{"column": c["column"], "distinct": c["distinct"],
          "top": ", ".join(f"{t['value']} ({t['pct']}%)" for t in c["top_values"][:3])} for c in cat],
        ["column", "distinct", "top"],
    )]
    lines += ["## Strongest correlations\n", _table(profile["correlations"].get("top_pairs", [])[:10], ["a", "b", "r"])]
    lines += [f"Duplicate rows: {profile['duplicates']['duplicate_pct']}% ({profile['duplicates']['note']}).\n"]
    if plots:
        lines += ["## Charts\n"]
        lines += [f"![{Path(p).stem}]({os.path.relpath(p, out_path.parent)})" for p in plots]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return out_path
