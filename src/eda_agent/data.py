"""Dataset loading.

Large files are handled in two passes:

1. A streaming pass reads the whole file in chunks and computes exact,
   cheap statistics (row count, null counts, numeric min/max/sum).
2. A uniform random sample (reservoir-style, per chunk) is kept in memory
   for the heavier work: quantiles, correlations, and plots.

Small files skip the sampling and are loaded whole.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

SUPPORTED = {".csv", ".tsv", ".parquet", ".json", ".jsonl"}
DEFAULT_SAMPLE_ROWS = 100_000
DEFAULT_CHUNK_ROWS = 250_000
_KEY, _POS = "__eda_sample_key", "__eda_row_pos"
# pandas treats "NA" as missing by default, which silently nulls real codes such
# as the North America region. Only these markers count as missing.
NA_VALUES = ["", "null", "NULL", "Null", "NaN", "nan", "None", "none", "N/A", "n/a", "#N/A"]


@dataclass
class FullScan:
    """Exact statistics from a pass over every row of the file."""

    row_count: int
    null_counts: dict[str, int]
    numeric_min: dict[str, float] = field(default_factory=dict)
    numeric_max: dict[str, float] = field(default_factory=dict)
    numeric_mean: dict[str, float] = field(default_factory=dict)
    duplicate_rows: int = 0


@dataclass
class Dataset:
    name: str
    path: Path
    sample: pd.DataFrame
    scan: FullScan
    file_bytes: int
    modified_at: float
    sampled: bool

    @property
    def schema_hash(self) -> str:
        sig = "|".join(f"{c}:{t}" for c, t in self.sample.dtypes.astype(str).items())
        return hashlib.sha256(sig.encode()).hexdigest()[:12]


def _reader(path: Path, chunksize: int | None):
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep, chunksize=chunksize, low_memory=False,
                           keep_default_na=False, na_values=NA_VALUES)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True, chunksize=chunksize)
    raise ValueError(f"streaming not supported for {suffix}")


def _iter_chunks(path: Path, chunk_rows: int):
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=chunk_rows):
            yield batch.to_pandas()
    elif suffix == ".json":
        yield pd.read_json(path)
    else:
        yield from _reader(path, chunk_rows)


def load_dataset(
    path: str | Path,
    name: str | None = None,
    sample_rows: int = DEFAULT_SAMPLE_ROWS,
    chunk_rows: int = DEFAULT_CHUNK_ROWS,
    seed: int = 0,
) -> Dataset:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in SUPPORTED:
        raise ValueError(f"unsupported file type {path.suffix}; use one of {sorted(SUPPORTED)}")

    rng = np.random.default_rng(seed)
    row_count = 0
    nulls: pd.Series | None = None
    mins: dict[str, float] = {}
    maxs: dict[str, float] = {}
    sums: dict[str, float] = {}
    non_null: dict[str, int] = {}
    kept: list[pd.DataFrame] = []
    row_hashes: list[np.ndarray] = []
    n_chunks = 0

    for chunk in _iter_chunks(path, chunk_rows):
        n_chunks += 1
        row_count += len(chunk)
        chunk_nulls = chunk.isna().sum()
        nulls = chunk_nulls if nulls is None else nulls.add(chunk_nulls, fill_value=0)
        # 8 bytes per row: exact duplicate detection across chunks.
        row_hashes.append(pd.util.hash_pandas_object(chunk, index=False).to_numpy())

        num = chunk.select_dtypes(include="number").select_dtypes(exclude="bool")
        for col in num.columns:
            s = num[col].dropna()
            if s.empty:
                continue
            mins[col] = min(mins.get(col, np.inf), float(s.min()))
            maxs[col] = max(maxs.get(col, -np.inf), float(s.max()))
            sums[col] = sums.get(col, 0.0) + float(s.sum())
            non_null[col] = non_null.get(col, 0) + len(s)

        # Uniform sample without knowing the row count up front: give every
        # row a random key and keep the `sample_rows` smallest keys seen so far.
        chunk = chunk.assign(**{_KEY: rng.random(len(chunk)), _POS: np.arange(len(chunk)) + row_count - len(chunk)})
        kept.append(chunk)
        if sum(len(k) for k in kept) > 2 * sample_rows:
            kept = [pd.concat(kept).nsmallest(sample_rows, _KEY)]

    sample = pd.concat(kept) if kept else pd.DataFrame(columns=[_KEY, _POS])
    if len(sample) > sample_rows:
        sample = sample.nsmallest(sample_rows, _KEY)
    sampled = len(sample) < row_count
    sample = sample.sort_values(_POS).drop(columns=[_KEY, _POS]).reset_index(drop=True)

    hashes = np.concatenate(row_hashes) if row_hashes else np.array([], dtype=np.uint64)
    duplicate_rows = int(len(hashes) - len(np.unique(hashes)))

    scan = FullScan(
        duplicate_rows=duplicate_rows,
        row_count=row_count,
        null_counts={k: int(v) for k, v in (nulls if nulls is not None else pd.Series()).items()},
        numeric_min=mins,
        numeric_max=maxs,
        numeric_mean={c: sums[c] / non_null[c] for c in sums if non_null.get(c)},
    )
    stat = os.stat(path)
    log.info(
        "loaded %s: %d rows in %d chunk(s), sample=%d rows (sampled=%s)",
        path.name, row_count, n_chunks, len(sample), sampled,
    )
    return Dataset(
        name=name or path.stem,
        path=path,
        sample=sample,
        scan=scan,
        file_bytes=stat.st_size,
        modified_at=stat.st_mtime,
        sampled=sampled,
    )
