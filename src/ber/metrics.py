"""The challenge metric (macro F0.5 for each S1 record) and a bootstrap confidence interval.

Conventions
- truth: DataFrame q, d  (true links of the evaluated S1 set; an S1 with no row is a singleton)
- pred:  DataFrame q, d  (predicted links)
- queries: DataFrame q   (the evaluated S1 set, singletons included)
"""
from __future__ import annotations

import numpy as np
import polars as pl


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, queries: pl.DataFrame, return_per_entity: bool = False):
    """Exact challenge metric. pred: q, d (predicted links). Singletons score 1 iff predicted empty."""
    p = pred.select("q", "d").unique()
    tp = p.join(truth.select("q", "d"), on=["q", "d"]).group_by("q").len().rename({"len": "tp"})
    npred = p.group_by("q").len().rename({"len": "np"})
    ntrue = truth.group_by("q").len().rename({"len": "nt"})
    e = (queries.select("q").join(npred, on="q", how="left").join(ntrue, on="q", how="left").join(tp, on="q", how="left")
         .with_columns(pl.col("np", "nt", "tp").fill_null(0)))
    prec = pl.when(pl.col("np") > 0).then(pl.col("tp") / pl.col("np")).otherwise(0.0)
    rec = pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(0.0)
    e = e.with_columns(prec.alias("P"), rec.alias("R"))
    f = (pl.when((pl.col("nt") == 0) & (pl.col("np") == 0)).then(1.0)
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * pl.col("P") * pl.col("R") / (0.25 * pl.col("P") + pl.col("R"))))
    e = e.with_columns(f.alias("f05"))
    score = float(e["f05"].mean())
    return (score, e) if return_per_entity else score


def bootstrap_ci(values: np.ndarray, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
