"""Supervised pair matcher (LightGBM, BCE objective) + macro-F0.5 decision rules.

Decision rules (both tuned only on development out-of-fold predictions, then frozen)
  threshold : keep candidate j of S1 i iff p_ij >= t
  expected_f: per S1, choose the probability-sorted prefix S maximising the plug-in expected F0.5
              E[F](S) ~= 1.25 * sum_{j in S} p_j / (0.25 * sum_{all j} p_j + |S|)
              and compare with the empty set, whose expected score is prod_j (1 - p_j)
              (a singleton scores 1 only if predicted empty). A probability scale factor `alpha`
              (calibration slack) and a floor `t_min` are tuned on OOF.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import polars as pl

from .data import THREADS
from .metrics import macro_f05

LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=THREADS, seed=20260926)


def train_lgb(X: pl.DataFrame, y: np.ndarray, groups: np.ndarray, n_folds: int = 4, rounds: int = 1200,
              params: dict | None = None) -> tuple[list, np.ndarray]:
    """Group K-fold by S1 (query) -> OOF probabilities + fold models. Returns (models, oof)."""
    params = {**LGB_PARAMS, **(params or {})}
    Xn = X.to_numpy().astype(np.float32)
    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    fold_of = dict(zip(ug, rng.integers(0, n_folds, len(ug))))
    f = np.vectorize(fold_of.get)(groups)
    oof = np.zeros(len(y), np.float32)
    models = []
    for k in range(n_folds):
        tr, va = f != k, f == k
        dtr = lgb.Dataset(Xn[tr], y[tr], feature_name=X.columns, free_raw_data=True)
        dva = lgb.Dataset(Xn[va], y[va], reference=dtr)
        m = lgb.train(params, dtr, rounds, valid_sets=[dva], callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[va] = m.predict(Xn[va], num_iteration=m.best_iteration)
        models.append(m)
    return models, oof


def predict(models, X: pl.DataFrame) -> np.ndarray:
    Xn = X.to_numpy().astype(np.float32)
    ms = models if isinstance(models, list) else [models]
    return np.mean([m.predict(Xn, num_iteration=getattr(m, "best_iteration", None) or None) for m in ms], axis=0)


# --------------------------------------------------------------------------- decision rules
def decide_threshold(scored: pl.DataFrame, t: float) -> pl.DataFrame:
    return scored.filter(pl.col("p") >= t).select("q", "d")


def decide_expected_f(scored: pl.DataFrame, alpha: float = 1.0, t_min: float = 0.0) -> pl.DataFrame:
    s = scored.with_columns((pl.col("p") * alpha).clip(0, 1).alias("pp")).sort(["q", "pp"], descending=[False, True])
    s = s.with_columns(
        pl.col("pp").sum().over("q").alias("mu"),
        pl.col("pp").cum_sum().over("q").alias("cs"),
        (pl.int_range(pl.len()).over("q") + 1).alias("k"),
        (1 - pl.col("pp")).log().sum().over("q").exp().alias("p_empty"),
    ).with_columns((1.25 * pl.col("cs") / (0.25 * pl.col("mu") + pl.col("k"))).alias("ef"))
    best = s.group_by("q").agg(pl.col("ef").max().alias("ef_best"), pl.col("p_empty").first())
    s = s.join(best, on="q").filter((pl.col("ef_best") > pl.col("p_empty")))
    kbest = s.filter(pl.col("ef") == pl.col("ef_best")).group_by("q").agg(pl.col("k").min().alias("kb"))
    return s.join(kbest, on="q").filter((pl.col("k") <= pl.col("kb")) & (pl.col("p") >= t_min)).select("q", "d")


def tune_threshold(scored: pl.DataFrame, truth: pl.DataFrame, queries: pl.DataFrame, grid=None) -> tuple[float, pl.DataFrame]:
    grid = grid if grid is not None else np.round(np.arange(0.10, 0.96, 0.025), 3)
    rows = [(t, macro_f05(decide_threshold(scored, t), truth, queries)) for t in grid]
    df = pl.DataFrame(rows, schema=["t", "macro_f05"], orient="row")
    return float(df.sort("macro_f05", descending=True)["t"][0]), df


def tune_expected_f(scored: pl.DataFrame, truth: pl.DataFrame, queries: pl.DataFrame,
                    alphas=(0.8, 0.9, 1.0, 1.1, 1.2), t_mins=(0.0, 0.1, 0.2, 0.3)) -> tuple[dict, pl.DataFrame]:
    rows = [(a, tm, macro_f05(decide_expected_f(scored, a, tm), truth, queries)) for a in alphas for tm in t_mins]
    df = pl.DataFrame(rows, schema=["alpha", "t_min", "macro_f05"], orient="row").sort("macro_f05", descending=True)
    return {"alpha": float(df["alpha"][0]), "t_min": float(df["t_min"][0])}, df


# --------------------------------------------------------------------------- stage 2 + calibration
def context_features(q: np.ndarray, p: np.ndarray) -> pl.DataFrame:
    """Within-S1 context of a first-stage probability: how this candidate compares with the S1's others."""
    df = pl.DataFrame({"q": q, "p1": p.astype(np.float32)})
    return df.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("p1_rank_q"),
        pl.col("p1").max().over("q").alias("p1_max_q"),
    ).with_columns(
        # runner-up probability (null for single-candidate S1); plain window max -> works on every polars version
        pl.when(pl.col("p1_rank_q") > 1).then(pl.col("p1")).otherwise(None).max().over("q").alias("p1_second_q"),
        (pl.col("p1") > 0.5).sum().over("q").cast(pl.Float32).alias("p1_n_strong_q"),
        pl.col("p1").sum().over("q").alias("p1_sum_q"),
        pl.len().over("q").cast(pl.Float32).alias("p1_n_q"),
    ).with_columns(
        # gap to the best *other* candidate (for the top candidate that is the runner-up)
        (pl.col("p1") - pl.when(pl.col("p1_rank_q") == 1).then(pl.col("p1_second_q")).otherwise(pl.col("p1_max_q"))).alias("p1_gap_other_q"),
    ).drop("q", "p1_second_q")


class TwoStage:
    """Stage 1: LightGBM on pair features. Stage 2: LightGBM on pair features + within-S1 context of the
    stage-1 probability (trained on stage-1 OOF). Isotonic calibration on stage-2 OOF."""

    def __init__(self, feats: list, calibrate: bool = True):
        self.feats = feats
        self.calibrate = calibrate

    def fit_oof(self, F: pl.DataFrame, y: np.ndarray, n_folds: int = 4):
        from sklearn.isotonic import IsotonicRegression
        q = F["q"].to_numpy()
        self.m1, oof1 = train_lgb(F.select(self.feats), y, q, n_folds)
        C = context_features(q, oof1)
        X2 = pl.concat([F.select(self.feats), C], how="horizontal")
        self.feats2 = X2.columns
        self.m2, oof2 = train_lgb(X2, y, q, n_folds)
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(oof2, y)
        self.oof1, self.oof2, self.oof_cal = oof1, oof2, self.iso.predict(oof2).astype(np.float32)
        return self

    def predict(self, F: pl.DataFrame) -> np.ndarray:
        """Mean of the grouped-fold models of each stage, then the isotonic calibration."""
        q = F["q"].to_numpy()
        X1 = F.select(self.feats).to_numpy().astype(np.float32)
        p1 = np.mean([m.predict(X1, num_iteration=m.best_iteration) for m in self.m1], axis=0)
        X2 = pl.concat([F.select(self.feats), context_features(q, p1)], how="horizontal").to_numpy().astype(np.float32)
        p2 = np.mean([m.predict(X2, num_iteration=m.best_iteration) for m in self.m2], axis=0)
        return self.iso.predict(p2).astype(np.float32) if self.calibrate else p2.astype(np.float32)


# --------------------------------------------------------------------------- per-S1 emptiness gate
# The plug-in rule's empty-set probability prod(1 - p_j) assumes independent candidates; siblings are strongly
# correlated, so it is miscalibrated exactly where singleton false positives arise. A dedicated per-S1 model
# predicts P(no true candidate) from the whole probability profile plus the top candidate's own evidence.
def set_features(scored: pl.DataFrame, F: pl.DataFrame | None = None, top_feats: list | None = None, n_top: int = 8) -> pl.DataFrame:
    """scored: q, d, p (one S1's candidates together) -> one row per q."""
    s = scored.select("q", "d", "p").sort(["q", "p"], descending=[False, True])
    s = s.with_columns(pl.int_range(pl.len()).over("q").alias("r"))
    top = s.filter(pl.col("r") < n_top).pivot(on="r", index="q", values="p").sort("q")
    top = top.rename({c: f"p_top{int(c) + 1}" for c in top.columns if c != "q"})
    for i in range(1, n_top + 1):
        if f"p_top{i}" not in top.columns:
            top = top.with_columns(pl.lit(0.0).alias(f"p_top{i}"))
    agg = s.group_by("q").agg(
        pl.col("p").sum().alias("p_sum"), pl.len().cast(pl.Float32).alias("n_cands"),
        (pl.col("p") > 0.5).sum().cast(pl.Float32).alias("n_gt50"), (pl.col("p") > 0.2).sum().cast(pl.Float32).alias("n_gt20"),
        (1 - pl.col("p").clip(0, 0.999999)).log().sum().alias("log_p_none_indep"),
        (-(pl.col("p").clip(1e-6, 1) * pl.col("p").clip(1e-6, 1).log())).sum().alias("p_entropy"))
    out = top.join(agg, on="q").with_columns((pl.col("p_top1") - pl.col("p_top2").fill_null(0)).alias("gap12"))
    if F is not None and top_feats:
        best = s.filter(pl.col("r") == 0).select("q", "d")
        tf = best.join(F.select(["q", "d"] + top_feats), on=["q", "d"], how="left").drop("d")
        out = out.join(tf.rename({c: f"top1_{c}" for c in top_feats}), on="q", how="left")
    return out.with_columns(pl.exclude("q").cast(pl.Float32).fill_null(0.0)).sort("q")


class EmptyGate:
    """LightGBM on per-S1 set features -> P(S1 has no true candidate). An S1 whose decision would accept something
    is emptied when P0 >= t0 (t0 tuned on OOF macro F0.5; t0 > 1 disables the gate)."""

    def __init__(self, top_feats: list | None = None):
        self.top_feats = top_feats or []

    def fit_oof(self, scored_oof: pl.DataFrame, y: np.ndarray, F: pl.DataFrame | None = None):
        S = set_features(scored_oof, F, self.top_feats)
        pos = scored_oof.select("q").with_columns(pl.Series("y", y)).group_by("q").agg(pl.col("y").max())
        yq = S.select("q").join(pos, on="q", how="left", maintain_order="left")["y"].fill_null(0).to_numpy()
        self.feats = [c for c in S.columns if c != "q"]
        self.models, oof_nonempty = train_lgb(S.select(self.feats), yq, S["q"].to_numpy(), n_folds=4, rounds=800,
                                              params={"num_leaves": 31, "min_data_in_leaf": 40})
        self.oof = S.select("q").with_columns(pl.Series("p0", (1 - oof_nonempty).astype(np.float32)))
        return self

    def predict(self, scored: pl.DataFrame, F: pl.DataFrame | None = None) -> pl.DataFrame:
        S = set_features(scored, F, self.top_feats)
        for c in self.feats:
            if c not in S.columns:
                S = S.with_columns(pl.lit(0.0).alias(c))
        return S.select("q").with_columns(pl.Series("p0", (1 - predict(self.models, S.select(self.feats))).astype(np.float32)))


def apply_gate(accepted: pl.DataFrame, p0: pl.DataFrame, t0: float) -> pl.DataFrame:
    return accepted.join(p0.filter(pl.col("p0") >= t0).select("q"), on="q", how="anti")


def tune_gate(accepted_oof: pl.DataFrame, p0_oof: pl.DataFrame, truth: pl.DataFrame, queries: pl.DataFrame,
              grid=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.96, 0.98, 1.01)) -> tuple[float, pl.DataFrame]:
    rows = [(t0, macro_f05(apply_gate(accepted_oof, p0_oof, t0), truth, queries)) for t0 in grid]
    df = pl.DataFrame(rows, schema=["t0", "macro_f05"], orient="row").sort("macro_f05", descending=True)
    return float(df["t0"][0]), df
