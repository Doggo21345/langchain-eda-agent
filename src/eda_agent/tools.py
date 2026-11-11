"""LangChain tools the agent can call.

The tools wrap the deterministic functions in `profiling` and `plots`, so the
LLM decides *what* to look at while every number it reports comes from code.
Each call is timed and recorded by the `RunLogger`.
"""

from __future__ import annotations

import functools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.tools import BaseTool, tool

from . import plots, profiling, report
from .data import Dataset
from .run_log import RunLogger

MAX_CHARS = 6000


@dataclass
class Workspace:
    datasets: dict[str, Dataset]
    out_dir: Path
    run_logger: RunLogger
    artifacts: list[str] = field(default_factory=list)

    def get(self, name: str | None) -> Dataset:
        if not name:
            return next(iter(self.datasets.values()))
        if name not in self.datasets:
            raise KeyError(f"unknown dataset {name!r}; available: {list(self.datasets)}")
        return self.datasets[name]

    def column(self, ds: Dataset, column: str) -> str:
        if column not in ds.sample.columns:
            raise KeyError(f"unknown column {column!r} in {ds.name}; columns: {list(ds.sample.columns)}")
        return column


def _dump(obj) -> str:
    text = json.dumps(obj, default=str)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + '... [truncated; ask about specific columns for detail]'
    return text


def make_tools(ws: Workspace) -> list[BaseTool]:
    def logged(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kwargs)
                ws.run_logger.record_tool(fn.__name__, kwargs, time.perf_counter() - t0, ok=True)
                return _dump(out)
            except Exception as e:  # tools report errors back to the model instead of crashing the run
                ws.run_logger.record_tool(fn.__name__, kwargs, time.perf_counter() - t0, ok=False, error=str(e))
                return _dump({"error": f"{type(e).__name__}: {e}"})
        return wrapper

    @tool
    @logged
    def list_datasets() -> dict:
        """List loaded datasets with row/column counts. Call this first."""
        return {n: {"rows": d.scan.row_count, "columns": len(d.sample.columns), "sampled": d.sampled}
                for n, d in ws.datasets.items()}

    @tool
    @logged
    def describe_schema(dataset: str = "") -> dict:
        """Column names, dtypes, numeric vs categorical, and an example value for each column."""
        return profiling.schema(ws.get(dataset))

    @tool
    @logged
    def null_audit(dataset: str = "") -> dict:
        """Exact null counts and percentages per column across all rows, plus high-null and fully-null columns."""
        return profiling.null_audit(ws.get(dataset))

    @tool
    @logged
    def distribution_overview(dataset: str = "") -> dict:
        """Summary statistics for every numeric column and top values for every categorical column."""
        d = profiling.distributions(ws.get(dataset))
        return {
            "numeric": [{k: v.get(k) for k in ("column", "mean", "median", "std", "min", "max", "skew", "shape", "iqr_outlier_pct")}
                        for v in d["numeric"]],
            "categorical": [{"column": c["column"], "distinct": c["distinct"], "top_values": c["top_values"][:3]}
                            for c in d["categorical"]],
        }

    @tool
    @logged
    def profile_column(column: str, dataset: str = "") -> dict:
        """Detailed profile of one column: quantiles, skew, outliers for numeric; value counts for categorical."""
        ds = ws.get(dataset)
        col = ws.column(ds, column)
        base = {"nulls": ds.scan.null_counts.get(col, 0)}
        if col in profiling.numeric_columns(ds.sample):
            return base | profiling.numeric_distribution(ds, col)
        return base | profiling.categorical_summary(ds, col, top_n=15)

    @tool
    @logged
    def correlation_analysis(method: str = "pearson", dataset: str = "") -> dict:
        """Correlation between numeric columns. method is 'pearson' or 'spearman'. Returns the strongest pairs."""
        if method not in {"pearson", "spearman"}:
            raise ValueError("method must be 'pearson' or 'spearman'")
        c = profiling.correlations(ws.get(dataset), method=method)
        c.pop("matrix", None)
        return c

    @tool
    @logged
    def data_quality_check(dataset: str = "") -> dict:
        """Rule-based quality checks (nulls, constants, outliers, skew, duplicates, numbers stored as text) with a 0-100 score."""
        return profiling.quality_issues(ws.get(dataset))

    @tool
    @logged
    def group_aggregate(by: str, column: str, agg: str = "mean", dataset: str = "") -> dict:
        """Aggregate `column` grouped by `by`. agg: mean, median, sum, count, min, max, std, nunique."""
        ds = ws.get(dataset)
        return profiling.group_aggregate(ds, ws.column(ds, by), ws.column(ds, column), agg)

    @tool
    @logged
    def plot(kind: str, column: str = "", y: str = "", dataset: str = "") -> dict:
        """Save a PNG chart. kind: 'histogram' (needs column), 'scatter' (column=x, y=y), 'correlation', 'nulls'."""
        ds = ws.get(dataset)
        out = ws.out_dir / "plots"
        if kind == "histogram":
            path = plots.histogram(ds, ws.column(ds, column), out)
        elif kind == "scatter":
            path = plots.scatter(ds, ws.column(ds, column), ws.column(ds, y), out)
        elif kind == "correlation":
            path = plots.correlation_heatmap(ds, out)
        elif kind == "nulls":
            path = plots.null_bars(ds, out)
        else:
            raise ValueError("kind must be histogram, scatter, correlation, or nulls")
        if path:
            ws.artifacts.append(path)
        return {"saved": path}

    @tool
    @logged
    def run_history(dataset: str = "") -> dict:
        """Previous profiling runs of this dataset: rows, null rate, quality score, and flagged changes."""
        ds = ws.get(dataset)
        hist = ws.run_logger.history(ds.name)[-10:]
        return {"runs": [{k: r.get(k) for k in ("timestamp", "available", "rows", "overall_null_pct", "quality_score", "changes")}
                         for r in hist]}

    @tool
    @logged
    def write_report(summary: str, dataset: str = "") -> dict:
        """Write the final Markdown report. `summary` is your written findings; tables and charts are added automatically."""
        ds = ws.get(dataset)
        prof = profiling.full_profile(ds)
        charts = sorted(set(ws.artifacts)) or plots.standard_plots(ds, ws.out_dir / "plots")
        path = report.render(prof, charts, ws.out_dir / f"{ds.name}_report.md", summary=summary)
        ws.artifacts.append(str(path))
        return {"saved": str(path)}

    return [list_datasets, describe_schema, null_audit, distribution_overview, profile_column,
            correlation_analysis, data_quality_check, group_aggregate, plot, run_history, write_report]
