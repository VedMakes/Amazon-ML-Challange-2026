"""Precomputed dense (learned-embedding) blocking arm.

Step 1 (step1_dense_blocker.py) fine-tunes a multilingual bi-encoder on S1 records with
eid % DENSE_TRAIN_MOD == 0 and writes, for every S1 of a split, its top-K S2 and S3 neighbours:
    dense_{split}.parquet : s1_eid (i64), t_eid (i64), src (i8: 2|3), score (f32), rank (u16, per source)
This module maps those rows onto the integer universe (q, d) so the arm behaves like every other arm.
Those training S1 records are excluded from every development sample (see DevSplits.sample_queries).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .settings import DENSE_TRAIN_MOD


def dense_training_s1(eid1: np.ndarray) -> np.ndarray:
    return (eid1 % DENSE_TRAIN_MOD) == 0


def load_dense(path: Path, U, q_idx: np.ndarray, k: int | None = None) -> pl.DataFrame:
    """Rows of the precomputed dense arm for S1 rows q_idx -> q, d, score, rank (rank per source)."""
    qmap = pl.DataFrame({"s1_eid": U.eid1[q_idx].astype(np.int64), "q": q_idx.astype(np.uint32)})
    df = (pl.scan_parquet(path).filter(pl.col("s1_eid").is_in(qmap["s1_eid"].implode()))
          .filter(pl.col("rank") <= (k if k is not None else 10_000)).collect())
    df = df.join(qmap, on="s1_eid")
    parts = []
    for code, lo in ((2, 0), (3, U.n2)):
        eids = U.eidT[lo:lo + (U.n2 if code == 2 else U.n3)]
        tmap = pl.DataFrame({"t_eid": eids.astype(np.int64), "d": (np.arange(len(eids)) + lo).astype(np.uint32)})
        parts.append(df.filter(pl.col("src") == code).join(tmap, on="t_eid"))
    out = pl.concat(parts) if parts else df
    return out.select("q", "d", pl.col("score").cast(pl.Float32), pl.col("rank").cast(pl.UInt16))


def _lookup(keys: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Positions of x in the (unsorted, unique) key array; -1 where absent."""
    o = np.argsort(keys, kind="stable")
    pos = np.clip(np.searchsorted(keys, x, sorter=o), 0, len(keys) - 1)
    idx = o[pos]
    return np.where(keys[idx] == x, idx, -1)


def dense_reverse(path: Path, U, R: int = 5, k: int = 20) -> pl.DataFrame:
    """Reverse evidence from the forward dense lists of EVERY S1 of the split: for each target, the S1 records whose
    top-k contains it, ranked by cosine (symmetric similarity -> the target's own top-R S1 among those that list it).
    Returns q, d, score, rank -- the frame ber.experiment.attach_reverse expects. numpy throughout (~50 B per row peak)."""
    df = (pl.scan_parquet(path).filter(pl.col("rank") <= k)
          .select(pl.col("s1_eid").cast(pl.Int64), pl.col("t_eid").cast(pl.Int64), pl.col("src").cast(pl.Int8), pl.col("score").cast(pl.Float32))
          .collect())
    q = _lookup(U.eid1.astype(np.int64), df["s1_eid"].to_numpy())
    te, src, sc = df["t_eid"].to_numpy(), df["src"].to_numpy(), df["score"].to_numpy()
    del df
    d = np.full(len(te), -1, np.int64)
    for code, lo, n in ((2, 0, U.n2), (3, U.n2, U.n3)):
        m = src == code
        pos = _lookup(U.eidT[lo:lo + n].astype(np.int64), te[m])
        d[m] = np.where(pos >= 0, pos + lo, -1)
    del te, src
    ok = (q >= 0) & (d >= 0)
    q, d, sc = q[ok], d[ok], sc[ok]
    order = np.lexsort((-sc, d))                       # by target, best cosine first
    q, d, sc = q[order], d[order], sc[order]
    del order
    i = np.arange(len(d))
    first = np.r_[True, d[1:] != d[:-1]]
    rank = i - np.maximum.accumulate(np.where(first, i, 0)) + 1
    keep = rank <= R
    return pl.DataFrame({"q": q[keep].astype(np.uint32), "d": d[keep].astype(np.uint32), "score": sc[keep],
                         "rank": rank[keep].astype(np.uint16)})
