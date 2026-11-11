# LangChain Automated EDA Agent

A LangChain agent that does exploratory data analysis on large datasets. You point it at a file
and ask a question, or ask for a full EDA. The LLM decides what to look at. Every number it
reports comes from deterministic profiling tools: null audits, distribution analysis,
correlation matrices, data quality checks, and charts. Each run logs data availability and
quality to a run log, so a bad pipeline day (missing file, row-count drop, dropped column,
null spike) gets flagged automatically.

```
eda-agent ask data/orders.csv -q "Is this data trustworthy, and what drives churn?"
```

## How it works

```
             ┌───────────────────── LangChain agent (create_agent) ─────────────────────┐
question ──▶ │  LLM plans the analysis ──▶ calls tools ──▶ reads JSON results ──▶ answer │ ──▶ report.md + charts
             └───────────────────────────────────┬──────────────────────────────────────┘
                                                 │
     ┌───────────────────────────────────────────┴───────────────────────────────────────────┐
     │ tools.py  describe_schema · null_audit · distribution_overview · profile_column ·     │
     │           correlation_analysis · data_quality_check · group_aggregate · plot ·         │
     │           run_history · write_report          (each call timed + logged)               │
     ├───────────────────────────────────────────────────────────────────────────────────────┤
     │ profiling.py / plots.py   deterministic pandas + matplotlib, JSON-serialisable output  │
     ├───────────────────────────────────────────────────────────────────────────────────────┤
     │ data.py   chunked full-file scan (exact counts) + uniform random sample (shape stats)  │
     ├───────────────────────────────────────────────────────────────────────────────────────┤
     │ run_log.py   runs.jsonl: availability, rows, schema hash, null rates, quality score,   │
     │              and the diff against the previous run of the same dataset                 │
     └───────────────────────────────────────────────────────────────────────────────────────┘
```

**Scaling to large files.** `data.py` streams CSV, TSV, JSONL, and Parquet in chunks (250k rows by
default). Row counts, null counts, min/max/mean, and duplicate rows (via 8-byte row hashes)
are exact across every row. Quantiles, skew, correlations, and plots use a uniform random sample
(100k rows by default; each row gets a random key and the smallest keys are kept). The report
says when a statistic comes from the sample.

**Why tools instead of letting the LLM write pandas.** Tools keep numbers reproducible and
stop the model from inventing statistics. Tool errors go back to the model as messages, so it can
recover, for example after asking for a column that doesn't exist.

### Data quality checks

| Check | Severity |
|---|---|
| Column entirely null | high |
| Column ≥20% null | medium |
| Constant / single-value column | medium |
| Numbers stored as text (`"$4.20"`) | medium |
| Small share of negatives in a positive column (sentinels like `-1`) | medium |
| Exact duplicate rows | medium |
| ≥5% IQR outliers, heavy skew | low |
| Dates stored as text | low |
| Near-unique values (likely an ID), highly correlated pairs | info |

These roll up into a 0–100 quality score.

### Run-over-run checks

Each run is compared with the previous run of the same dataset:

| Change | Severity |
|---|---|
| Dataset missing or unreadable | high |
| Row count fell ≥20% | high |
| Column removed | high |
| Column added / dtype changed | medium |
| Column null rate up ≥5 points | medium |
| Quality score dropped | low/medium |

## Quick start

```bash
git clone https://github.com/Doggo21345/langchain-eda-agent && cd langchain-eda-agent
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# synthetic 500k-row orders dataset with planted problems
python examples/make_sample_data.py --rows 500000 --out data/orders.csv

# 1) deterministic profile, no LLM or API key needed
eda-agent profile data/orders.csv

# 2) the agent (needs an API key)
export ANTHROPIC_API_KEY=...
eda-agent ask data/orders.csv                              # full EDA + report
eda-agent ask data/orders.csv -q "Which device has the worst latency?"
eda-agent ask data/orders.csv --model openai:gpt-5         # any LangChain provider:model (pip install ".[openai]")

# 3) simulate a bad pipeline day, then look at the run log
python examples/make_sample_data.py --rows 300000 --out day2/orders.csv --degrade
eda-agent profile day2/orders.csv
eda-agent history orders
```

Outputs land in `eda_output/`: `<name>_report.md`, `<name>_profile.json`, `plots/`,
`runs.jsonl`, and `eda_agent.log`.

## Example output

[`docs/example/orders_report.md`](docs/example/orders_report.md) is the report from the degraded
"day 2" run. It flags the 40% row drop, the missing `discount_rate` column, and the jump in
`app_latency_ms` nulls from 0% to 25%. It also catches the planted problems: `-1.0` sentinel
order values, duplicate rows, a single-value `currency` column, and dates stored as text.

```
$ eda-agent history orders
2026-11-11T17:53:27  orders   rows=501,000 null%=8.21 quality=73
2026-11-11T17:53:31  orders   rows=300,600 null%=10.77 quality=68
    [high] row count fell 40.0% (501,000 -> 300,600)
    [high] missing columns: ['discount_rate']
    [medium] app_latency_ms: null rate 0.0% -> 25.1%
    [low] quality score 73 -> 68
```

<p>
<img src="docs/example/plots/orders/nulls.png" width="48%">
<img src="docs/example/plots/orders/corr_pearson.png" width="40%">
</p>

One gotcha it handles: by default pandas reads the string `"NA"` as missing, which silently
nulls real values like the North America region code. The loader only treats explicit markers
(`""`, `null`, `NaN`, `None`, `N/A`, ...) as missing.

## Use as a library

```python
from eda_agent import load_dataset, RunLogger, Workspace
from eda_agent.agent import ask

ds = load_dataset("data/orders.csv")
ws = Workspace(datasets={ds.name: ds}, out_dir="eda_output", run_logger=RunLogger("eda_output/runs.jsonl"))
print(ask(ws, "Summarise data quality risks"))
```

## Tests

```bash
pytest
```

The tests cover exact-vs-sampled statistics, the quality rules, run-over-run diffing, and an
end-to-end agent run. The agent test drives the real LangChain `create_agent` loop with a
scripted chat model, so it needs no API key.

## Project layout

```
src/eda_agent/
  data.py        chunked loading, full-file scan, uniform sampling
  profiling.py   schema, null audit, distributions, correlations, quality rules
  plots.py       histograms, null bars, correlation heatmap, scatter
  run_log.py     logging setup, runs.jsonl, run-over-run comparison
  tools.py       LangChain tools wrapping the above
  agent.py       create_agent + system prompt
  report.py      Markdown report
  cli.py         eda-agent profile | ask | history
examples/make_sample_data.py
tests/
```

## License

MIT
