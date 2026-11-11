"""Run-over-run logging of data availability and quality.

Each profiling run appends one JSON line to `runs.jsonl` describing what data
was available (file present, size, rows, schema) and how healthy it looked
(null rates, quality score). Comparing a run with the previous run of the same
dataset surfaces pipeline problems: a missing file, a row-count drop, a schema
change, or a jump in nulls.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .data import Dataset
from . import profiling

log = logging.getLogger(__name__)

ROW_DROP_PCT = 20.0
NULL_JUMP_PTS = 5.0


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    for noisy in ("httpx", "httpcore", "anthropic", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@dataclass
class RunLogger:
    path: Path
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    tool_calls: list[dict] = field(default_factory=list)

    def record_tool(self, name: str, args: dict, seconds: float, ok: bool, error: str | None = None) -> None:
        entry = {"tool": name, "args": args, "seconds": round(seconds, 3), "ok": ok}
        if error:
            entry["error"] = error
        self.tool_calls.append(entry)
        log.info("tool %s ok=%s %.2fs args=%s", name, ok, seconds, args)

    def history(self, dataset: str | None = None) -> list[dict]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        return [r for r in rows if dataset is None or r.get("dataset") == dataset]

    def snapshot(self, ds: Dataset) -> dict:
        nulls = profiling.null_audit(ds)
        quality = profiling.quality_issues(ds)
        return {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "dataset": ds.name,
            "path": str(ds.path),
            "available": True,
            "file_bytes": ds.file_bytes,
            "rows": ds.scan.row_count,
            "columns": list(ds.sample.columns),
            "schema_hash": ds.schema_hash,
            "overall_null_pct": nulls["overall_null_pct"],
            "null_pct_by_column": {r["column"]: r["null_pct"] for r in nulls["by_column"]},
            "quality_score": quality["quality_score"],
            "issue_count": quality["issue_count"],
        }

    def record_run(self, ds: Dataset) -> dict:
        snap = self.snapshot(ds)
        prev = self.history(ds.name)
        snap["changes"] = compare(prev[-1], snap) if prev else []
        snap["tool_calls"] = self.tool_calls
        self._append(snap)
        for c in snap["changes"]:
            log.warning("[%s] %s", ds.name, c["message"])
        return snap

    def record_missing(self, name: str, path: Path, error: str) -> dict:
        entry = {
            "run_id": self.run_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "dataset": name,
            "path": str(path),
            "available": False,
            "error": error,
            "changes": [{"kind": "unavailable", "severity": "high", "message": f"dataset unavailable: {error}"}],
        }
        self._append(entry)
        log.error("[%s] dataset unavailable: %s", name, error)
        return entry

    def _append(self, entry: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(entry, default=str) + "\n")


def compare(prev: dict, cur: dict) -> list[dict]:
    """Differences between two run snapshots that are worth flagging."""
    out: list[dict] = []
    if not prev.get("available", True):
        out.append({"kind": "recovered", "severity": "info", "message": "dataset available again after a failed run"})
        return out

    p_rows, c_rows = prev.get("rows") or 0, cur.get("rows") or 0
    if p_rows:
        change = 100 * (c_rows - p_rows) / p_rows
        if change <= -ROW_DROP_PCT:
            out.append({"kind": "row_drop", "severity": "high",
                        "message": f"row count fell {abs(change):.1f}% ({p_rows:,} -> {c_rows:,})"})
        elif abs(change) >= ROW_DROP_PCT:
            out.append({"kind": "row_change", "severity": "info",
                        "message": f"row count changed {change:+.1f}% ({p_rows:,} -> {c_rows:,})"})

    p_cols, c_cols = set(prev.get("columns", [])), set(cur.get("columns", []))
    if added := sorted(c_cols - p_cols):
        out.append({"kind": "columns_added", "severity": "medium", "message": f"new columns: {added}"})
    if removed := sorted(p_cols - c_cols):
        out.append({"kind": "columns_removed", "severity": "high", "message": f"missing columns: {removed}"})
    if not added and not removed and prev.get("schema_hash") != cur.get("schema_hash"):
        out.append({"kind": "dtype_change", "severity": "medium", "message": "column types changed"})

    p_null, c_null = prev.get("null_pct_by_column", {}), cur.get("null_pct_by_column", {})
    for col in sorted(p_cols & c_cols):
        a, b = p_null.get(col) or 0, c_null.get(col) or 0
        if b - a >= NULL_JUMP_PTS:
            out.append({"kind": "null_jump", "severity": "medium",
                        "message": f"{col}: null rate {a:.1f}% -> {b:.1f}%"})

    if (q0 := prev.get("quality_score")) is not None and (q1 := cur.get("quality_score")) is not None and q1 < q0:
        out.append({"kind": "quality_drop", "severity": "medium" if q0 - q1 >= 10 else "low",
                    "message": f"quality score {q0} -> {q1}"})
    return out
