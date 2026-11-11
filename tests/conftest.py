import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def messy_csv(tmp_path):
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(10, 2, n)
    df = pd.DataFrame({
        "id": [f"u{i}" for i in range(n)],
        "region": rng.choice(["NA", "EU"], n),
        "x": x,
        "y": 3 * x + rng.normal(0, 0.5, n),
        "mostly_null": np.where(rng.random(n) < 0.6, np.nan, 1.0),
        "constant": 5,
        "price_text": [f"${v:.2f}" for v in rng.uniform(1, 9, n)],
        "day": pd.date_range("2026-01-01", periods=n, freq="h").strftime("%Y-%m-%d"),
    })
    df.loc[:4, "x"] = -999.0  # sentinel values
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)  # 10 duplicate rows
    path = tmp_path / "messy.csv"
    df.to_csv(path, index=False)
    return path
