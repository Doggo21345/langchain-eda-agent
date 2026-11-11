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

> Full deep dive with diagrams for every layer: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**

```mermaid
flowchart LR
    SRC[("Files: CSV · Parquet · JSONL<br/>or a SQL table / query")] --> LOAD
    subgraph LOAD["1 · Load (data.py)"]
        direction TB
        SCAN["Full scan<br/>exact rows, nulls,<br/>min/max/mean, duplicates"]
        SAMP["Uniform random sample<br/>for quantiles, correlations, charts"]
    end
    LOAD --> PROF["2 · Profile (profiling.py, plots.py)<br/>schema · null audit · distributions<br/>correlations · quality rules · charts"]
    PROF --> TOOLS["3 · Tools (tools.py)<br/>11 LangChain tools<br/>timed + logged"]
    TOOLS <--> LLM{{"LLM agent<br/>picks what to analyse,<br/>explains findings"}}
    PROF --> LOG[("4 · Run log (runs.jsonl)<br/>diff vs last run:<br/>row drops · schema changes · null spikes")]
    TOOLS --> OUT["5 · Report<br/>report.md + charts + profile.json"]
    LOG --> OUT
```

**The core idea: the LLM decides, code computes.** The model never writes pandas code and
never states a number it worked out itself. It picks from a fixed set of tools (`null_audit`,
`profile_column`, `correlation_analysis`, `plot`, ...). Each tool runs tested pandas/matplotlib
code and returns compact JSON. The model reads the JSON, decides what to look at next, and
writes up the findings. So the numbers are reproducible, there's no arbitrary code execution,
and the same profiling also runs with **no LLM at all** (`eda-agent profile`).

### The agent loop

```mermaid
sequenceDiagram
    actor U as You
    participant A as LLM agent
    participant T as Tools
    U->>A: "Is this data trustworthy? What drives churn?"
    A->>T: describe_schema()
    T-->>A: 14 columns, types, examples
    A->>T: null_audit() · data_quality_check()
    T-->>A: coupon_code 85% null, -1.0 sentinels, 0.2% duplicates
    A->>T: correlation_analysis() · profile_column("app_latency_ms")
    T-->>A: churn vs sessions r=-0.24, latency r=0.19, right-skewed
    A->>T: plot("scatter", ...) · run_history()
    T-->>A: chart saved · last run had 501k rows
    A->>T: write_report(summary)
    A-->>U: answer + report.md + charts
```

The agent is built with LangChain 1.x `create_agent` (a LangGraph tool-calling loop, capped at
40 steps). Tool errors, such as a misspelled column, go back to the model as messages that list
the valid options, so it can correct itself. Every tool call is timed and saved in the run log,
so you can audit what the agent looked at before it reached its conclusions.

**Scaling to large files and tables.** `data.py` streams CSV, TSV, JSONL, Parquet, and database
tables or queries (any SQLAlchemy database: Postgres, MySQL, Redshift, Snowflake, SQLite, ...)
in chunks (250k rows by default). Row counts, null counts, min/max/mean, and duplicate rows (via 8-byte row hashes)
are exact across every row. Quantiles, skew, correlations, and plots use a uniform random sample
(100k rows by default; each row gets a random key and the smallest keys are kept). The report
says when a statistic comes from the sample.

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

# 3) a database table or query instead of a file (pip install ".[sql]")
python examples/make_sample_data.py --rows 200000 --out data/shop.db
export EDA_DB_URL=sqlite:///data/shop.db          # e.g. postgresql://user:pass@host/db
eda-agent profile --table orders
eda-agent ask --query "SELECT device, app_latency_ms, churned_90d FROM orders" -q "Does latency differ by device?"

# 4) simulate a bad pipeline day, then look at the run log
python examples/make_sample_data.py --rows 300000 --out day2/orders.csv --degrade
eda-agent profile day2/orders.csv
eda-agent history orders
```

The database URL is read from an environment variable (`EDA_DB_URL`, or pick another with
`--db-url-env`), and the password is redacted everywhere it's logged. Only the table or query you
pass is run. The agent never writes SQL, and table names are checked against the database
before use.

Each run log entry records `load_seconds` and `run_seconds`, so you can track profiling time
across runs.

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

The tests cover exact-vs-sampled statistics, the quality rules, run-over-run diffing, the SQL
source (it must match the file scan exactly, reject unknown tables, and redact passwords), and an
end-to-end agent run. The agent test drives the real LangChain `create_agent` loop with a
scripted chat model, so it needs no API key.

## Project layout

```
src/eda_agent/
  data.py        chunked file/SQL loading, full scan, uniform sampling
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
