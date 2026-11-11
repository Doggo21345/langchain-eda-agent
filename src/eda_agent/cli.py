"""Command line entry point.

    eda-agent profile data.csv            # deterministic profile + report, no LLM
    eda-agent ask data.csv "question"     # LangChain agent answers / writes a report
    eda-agent profile --table orders      # database table (URL from $EDA_DB_URL)
    eda-agent history [dataset]           # run-over-run availability & quality log
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import plots, profiling, report
import os

from .data import DEFAULT_SAMPLE_ROWS, load_dataset, load_sql, redact_url
from .run_log import RunLogger, setup_logging
from .tools import Workspace

DEFAULT_QUESTION = "Do a full exploratory analysis of this data and write the report."


def _load(args, logger: RunLogger):
    datasets = {}
    for p in args.paths:
        path = Path(p)
        try:
            ds = load_dataset(path, sample_rows=args.sample_rows)
            datasets[ds.name] = ds
        except (FileNotFoundError, ValueError, OSError) as e:
            logger.record_missing(path.stem, path, str(e))

    if args.table or args.query:
        url = os.environ.get(args.db_url_env)
        if not url:
            raise SystemExit(f"set {args.db_url_env} to a SQLAlchemy database URL (e.g. sqlite:///data/orders.db)")
        safe_url = redact_url(url)
        sources = [("table", t) for t in args.table] + [("query", q) for q in args.query]
        for i, (kind, value) in enumerate(sources):
            name = value.rpartition(".")[2] if kind == "table" else f"query{i}" if len(args.query) > 1 else "query"
            try:
                ds = load_sql(url, **{kind: value}, name=name, sample_rows=args.sample_rows)
                datasets[ds.name] = ds
            except Exception as e:  # driver errors vary by database
                msg = str(e).replace(url, safe_url).splitlines()[0]
                logger.record_missing(name, f"{safe_url} {kind}", msg)

    if not datasets and not (args.paths or args.table or args.query):
        raise SystemExit("give at least one file path, --table, or --query")
    return datasets


def _add_sources(sp):
    sp.add_argument("paths", nargs="*", help="CSV/TSV/JSON/JSONL/Parquet files")
    sp.add_argument("--table", action="append", default=[], help="database table to profile (repeatable)")
    sp.add_argument("--query", action="append", default=[], help="SELECT query to profile (repeatable)")
    sp.add_argument("--db-url-env", default="EDA_DB_URL",
                    help="environment variable holding the SQLAlchemy URL (default: EDA_DB_URL)")
    sp.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)


def cmd_profile(args) -> int:
    out = Path(args.out)
    logger = RunLogger(out / "runs.jsonl")
    datasets = _load(args, logger)
    if not datasets:
        return 1
    for ds in datasets.values():
        t0 = time.perf_counter()
        prof = profiling.full_profile(ds)
        charts = [] if args.no_plots else plots.standard_plots(ds, out / "plots" / ds.name)
        run = logger.record_run(ds, work_seconds=time.perf_counter() - t0)
        path = report.render(prof, charts, out / f"{ds.name}_report.md", run=run)
        (out / f"{ds.name}_profile.json").write_text(json.dumps(prof, indent=2, default=str))
        print(f"{ds.name}: {ds.scan.row_count:,} rows, quality {prof['quality']['quality_score']}/100, "
              f"{len(run['changes'])} change(s) vs last run, {run['run_seconds']}s -> {path}")
    return 0


def cmd_ask(args) -> int:
    from .agent import ask  # imported lazily so `profile` works without LLM credentials

    out = Path(args.out)
    logger = RunLogger(out / "runs.jsonl")
    datasets = _load(args, logger)
    if not datasets:
        return 1
    ws = Workspace(datasets=datasets, out_dir=out, run_logger=logger)
    answer = ask(ws, args.question, model=args.model)
    for ds in datasets.values():
        logger.record_run(ds)
    print(answer)
    if ws.artifacts:
        print("\nArtifacts:\n" + "\n".join(f"  {a}" for a in sorted(set(ws.artifacts))))
    return 0


def cmd_history(args) -> int:
    rows = RunLogger(Path(args.out) / "runs.jsonl").history(args.dataset)
    if not rows:
        print("no runs logged yet")
        return 0
    for r in rows[-args.limit:]:
        status = (f"rows={r.get('rows'):,} null%={r.get('overall_null_pct')} quality={r.get('quality_score')}"
                  if r.get("available") else f"UNAVAILABLE: {r.get('error')}")
        print(f"{r['timestamp']}  {r['dataset']:<20} {status}")
        for c in r.get("changes", []):
            print(f"    [{c['severity']}] {c['message']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eda-agent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="eda_output", help="output directory (default: eda_output)")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("profile", help="profile datasets without an LLM")
    _add_sources(pr)
    pr.add_argument("--no-plots", action="store_true")
    pr.set_defaults(fn=cmd_profile)

    a = sub.add_parser("ask", help="let the LangChain agent analyse the data")
    _add_sources(a)
    a.add_argument("-q", "--question", default=DEFAULT_QUESTION)
    a.add_argument("--model", default=None, help="LangChain provider:model, e.g. anthropic:claude-sonnet-5-5")
    a.set_defaults(fn=cmd_ask)

    h = sub.add_parser("history", help="show logged runs")
    h.add_argument("dataset", nargs="?")
    h.add_argument("--limit", type=int, default=20)
    h.set_defaults(fn=cmd_history)

    args = p.parse_args(argv)
    setup_logging(args.log_level, Path(args.out) / "eda_agent.log")
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
