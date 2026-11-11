"""Generate a synthetic e-commerce orders dataset with realistic data problems.

    python examples/make_sample_data.py --rows 500000 --out data/orders.csv
    python examples/make_sample_data.py --rows 300000 --out data/orders_day2.csv --degrade
    python examples/make_sample_data.py --rows 200000 --out data/shop.db   # SQLite table "orders"

`--degrade` simulates a bad pipeline day (fewer rows, a dropped column, more
nulls) so you can see the run-over-run checks fire.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make(rows: int, seed: int, degrade: bool) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    region = rng.choice(["NA", "EU", "APAC", "LATAM"], rows, p=[0.45, 0.3, 0.18, 0.07])
    device = rng.choice(["ios", "android", "web"], rows, p=[0.35, 0.4, 0.25])
    sessions = rng.poisson(6, rows) + 1
    pages = np.round(sessions * rng.gamma(3, 1.5, rows)).astype(int)
    latency = rng.lognormal(5.5, 0.5, rows) * np.where(device == "android", 1.3, 1.0)
    order_value = np.round(np.exp(rng.normal(3.6, 0.7, rows)) + 0.8 * pages, 2)
    discount = np.clip(rng.normal(0.08, 0.05, rows), 0, 0.5).round(3)
    churned = (rng.random(rows) < 1 / (1 + np.exp(-(latency / 400 - sessions / 4)))).astype(int)

    df = pd.DataFrame({
        "order_id": [f"o{i:08d}" for i in range(rows)],
        "order_date": pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 300, rows), unit="D"),
        "region": region,
        "device": device,
        "sessions_30d": sessions,
        "pages_viewed": pages,
        "app_latency_ms": latency.round(1),
        "order_value": order_value,
        "discount_rate": discount,
        "coupon_code": np.where(rng.random(rows) < 0.85, None, rng.choice(["SAVE10", "WELCOME", "VIP"], rows)),
        "rating": np.where(rng.random(rows) < 0.3, np.nan, rng.integers(1, 6, rows)),
        "currency": "USD",
        "zip_code": rng.integers(10000, 99999, rows).astype(str),
        "churned_90d": churned,
    })
    # A few corrupt values and some duplicate rows.
    bad = rng.choice(rows, max(1, rows // 2000), replace=False)
    df.loc[bad, "order_value"] = -1.0
    df = pd.concat([df, df.sample(frac=0.002, random_state=seed)], ignore_index=True)

    if degrade:
        df = df.drop(columns=["discount_rate"])
        df.loc[rng.random(len(df)) < 0.25, "app_latency_ms"] = np.nan
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--out", default="data/orders.csv")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--degrade", action="store_true")
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df = make(a.rows, a.seed, a.degrade)
    if a.out.endswith(".db"):
        import sqlite3

        with sqlite3.connect(a.out) as con:
            df.to_sql("orders", con, if_exists="replace", index=False)
    else:
        (df.to_parquet if a.out.endswith(".parquet") else df.to_csv)(a.out, index=False)
    print(f"wrote {len(df):,} rows x {df.shape[1]} cols -> {a.out}")
