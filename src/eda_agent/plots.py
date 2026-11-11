"""Static PNG visualisations written to the output directory."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .data import Dataset  # noqa: E402
from .profiling import categorical_columns, numeric_columns  # noqa: E402

INK, MUTED, ACCENT, ACCENT2 = "#1F2937", "#6B7280", "#2563EB", "#DC2626"


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", s).strip("_")[:60] or "col"


def _style(ax, title: str):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)


def _save(fig, out: Path) -> str:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return str(out)


def histogram(ds: Dataset, column: str, out_dir: Path, bins: int = 40) -> str:
    s = ds.sample[column].dropna()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    if column in numeric_columns(ds.sample):
        ax.hist(s.astype(float), bins=bins, color=ACCENT)
        ax.axvline(s.median(), color=ACCENT2, lw=1, ls="--", label=f"median {s.median():.3g}")
        ax.legend(frameon=False, fontsize=8)
    else:
        vc = s.astype(str).value_counts().head(15)[::-1]
        ax.barh(vc.index, vc.values, color=ACCENT)
    _style(ax, f"{column} distribution" + (" (sample)" if ds.sampled else ""))
    return _save(fig, out_dir / f"hist_{_slug(column)}.png")


def null_bars(ds: Dataset, out_dir: Path) -> str:
    n = max(ds.scan.row_count, 1)
    pct = {c: 100 * ds.scan.null_counts.get(c, 0) / n for c in ds.sample.columns}
    items = sorted(pct.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.28 * len(items))))
    ax.barh([k for k, _ in items], [v for _, v in items], color=[ACCENT2 if v >= 20 else ACCENT for _, v in items])
    ax.set_xlabel("% null", color=MUTED, fontsize=8)
    ax.set_xlim(0, 100)
    _style(ax, "Null rate by column (all rows)")
    return _save(fig, out_dir / "nulls.png")


def correlation_heatmap(ds: Dataset, out_dir: Path, method: str = "pearson") -> str | None:
    num = ds.sample[numeric_columns(ds.sample)]
    num = num.loc[:, num.nunique() > 1]
    if num.shape[1] < 2:
        return None
    corr = num.corr(method=method)
    k = len(corr)
    fig, ax = plt.subplots(figsize=(1 + 0.6 * k, 0.8 + 0.6 * k))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(k), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(k), corr.columns)
    if k <= 15:
        for i in range(k):
            for j in range(k):
                v = corr.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _style(ax, f"{method.title()} correlation")
    return _save(fig, out_dir / f"corr_{method}.png")


def scatter(ds: Dataset, x: str, y: str, out_dir: Path, max_points: int = 5000) -> str:
    df = ds.sample[[x, y]].dropna()
    if len(df) > max_points:
        df = df.sample(max_points, random_state=0)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(df[x], df[y], s=6, alpha=0.4, color=ACCENT)
    ax.set_xlabel(x, color=MUTED, fontsize=8)
    ax.set_ylabel(y, color=MUTED, fontsize=8)
    if len(df) > 2:
        m, b = np.polyfit(df[x].astype(float), df[y].astype(float), 1)
        xs = np.linspace(df[x].min(), df[x].max(), 50)
        ax.plot(xs, m * xs + b, color=ACCENT2, lw=1)
    _style(ax, f"{y} vs {x}")
    return _save(fig, out_dir / f"scatter_{_slug(x)}__{_slug(y)}.png")


def standard_plots(ds: Dataset, out_dir: Path, max_hist: int = 12) -> list[str]:
    paths = [null_bars(ds, out_dir)]
    c = correlation_heatmap(ds, out_dir)
    if c:
        paths.append(c)
    cols = numeric_columns(ds.sample)[:max_hist]
    cols += [c for c in categorical_columns(ds.sample) if ds.sample[c].nunique() <= 50][: max(0, max_hist - len(cols))]
    paths += [histogram(ds, c, out_dir) for c in cols]
    return paths
