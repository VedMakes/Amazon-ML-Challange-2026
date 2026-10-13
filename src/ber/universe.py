"""Record universe of one split and the development splits.

Index conventions (per split, all integer)
  q : row index into the Source-1 file                     (u32)
  d : global target index; S2 rows first, then S3 rows     (u32); d_src in {2, 3}
  country : integer code into Universe.countries (built from the data -> open set, France included)
"""
from __future__ import annotations

import gc
import time

import numpy as np
import polars as pl
import psutil

from . import store
from .data import load_raw
from .dense import dense_training_s1
from .split import assign_splits, build_components, leakage_audit


def rss_mb() -> int:
    return psutil.Process().memory_info().rss // 2**20


def gt_edges_int(eid1: np.ndarray, eidT: np.ndarray, t_src: np.ndarray) -> pl.DataFrame:
    """Ground truth -> integer edges (q, d). eidT/t_src describe the concatenated target table."""
    gt = load_raw("train", "gt")
    e = (gt.select(pl.col("source1_entity_id").str.slice(3).cast(pl.Int64).alias("e1"),
                   pl.col("matched_entity_ids").str.split(",").alias("t"))
         .explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
         .select("e1", pl.col("t").str.slice(0, 2).alias("src"), pl.col("t").str.slice(3).cast(pl.Int64).alias("et")))
    del gt
    o1 = np.argsort(eid1)
    q = o1[np.searchsorted(eid1, e["e1"].to_numpy(), sorter=o1)]
    d = np.empty(e.height, dtype=np.int64)
    for code, s in ((2, "S2"), (3, "S3")):
        m = (e["src"] == s).to_numpy()
        base = np.flatnonzero(t_src == code)
        ids = eidT[base]
        o = np.argsort(ids)
        d[m] = base[o[np.searchsorted(ids, e["et"].to_numpy()[m], sorter=o)]]
    assert (eid1[q] == e["e1"].to_numpy()).all() and (eidT[d] == e["et"].to_numpy()).all()
    return pl.DataFrame({"q": q.astype(np.uint32), "d": d.astype(np.uint32), "src": np.where(t_src[d] == 2, "S2", "S3")})


class Universe:
    """All records of one split (train or test) as compact integer arrays."""

    def __init__(self, split: str):
        self.split = split
        ids = store.load_ids(split)
        self.countries = ids["countries"]
        self.eid1, self.q_country = ids["eid1"], ids["c1"]
        self.eidT = np.concatenate([ids["eid2"], ids["eid3"]])
        self.d_country = np.concatenate([ids["c2"], ids["c3"]])
        self.n1, self.n2, self.n3 = len(ids["eid1"]), len(ids["eid2"]), len(ids["eid3"])
        self.d_src = np.concatenate([np.full(self.n2, 2, np.int8), np.full(self.n3, 3, np.int8)])

    @property
    def nT(self) -> int:
        return self.n2 + self.n3


class DevSplits:
    """Training split: connected components of the label graph, component-disjoint splits, S1 samples."""

    def __init__(self, seed: int = 20260926, outer_frac: float = 0.2, n_folds: int = 5, val_fold: int = 0):
        t0 = time.time()
        self.seed = seed
        self.U = U = Universe("train")
        self.edges = gt_edges_int(U.eid1, U.eidT, U.d_src)
        eq, ed = self.edges["q"].to_numpy().astype(np.int64), self.edges["d"].to_numpy().astype(np.int64)
        lab = build_components(U.n1, U.nT, eq, ed)
        self.q_mult = np.bincount(eq, minlength=U.n1)
        self.comp_split = assign_splits(U.q_country, self.q_mult, lab[:U.n1], seed, outer_frac, n_folds)
        cs = self.comp_split.sort("component")
        comp_ids = cs["component"].to_numpy()
        pos = np.searchsorted(comp_ids, lab)
        pos = np.minimum(pos, len(comp_ids) - 1)
        has = comp_ids[pos] == lab                       # node belongs to a component that contains an S1
        split_arr = np.where(has, cs["split"].to_numpy()[pos], "none")
        fold_arr = np.where(has, cs["fold"].to_numpy()[pos], -2)
        linked = np.zeros(U.n1 + U.nT, bool)
        linked[:U.n1] = True
        linked[U.n1 + ed] = True
        split_arr = np.where(linked, split_arr, "none")
        self.audit = leakage_audit(lab, split_arr, fold_arr)
        self.q_split, self.q_fold = split_arr[:U.n1], fold_arr[:U.n1]
        self.d_split, self.d_fold = split_arr[U.n1:], fold_arr[U.n1:]
        self.audit["components_with_more_than_one_S1"] = int((np.bincount(lab[:U.n1]) > 1).sum())
        self.val_fold = val_fold
        del lab, split_arr, fold_arr, linked
        gc.collect()
        self.build_seconds = time.time() - t0

    # ------------------------------------------------------------------ samples
    def sample_queries(self, where: str, n: int, seed: int) -> np.ndarray:
        """where: 'val' (inner val fold), 'train' (inner train folds), 'outer'. Stratified by country."""
        if where == "val":
            pool = np.flatnonzero((self.q_split == "dev") & (self.q_fold == self.val_fold))
        elif where == "train":
            pool = np.flatnonzero((self.q_split == "dev") & (self.q_fold != self.val_fold))
        else:
            pool = np.flatnonzero(self.q_split == "outer")
        pool = pool[~dense_training_s1(self.U.eid1[pool])]     # S1 reserved for the bi-encoder and the cross-encoder
        rng = np.random.default_rng(seed)
        out = []
        for c in np.unique(self.U.q_country[pool]):
            p = pool[self.U.q_country[pool] == c]
            out.append(rng.choice(p, size=min(len(p), int(round(n * len(p) / len(pool)))), replace=False))
        return np.sort(np.concatenate(out))

    def truth(self, q_idx: np.ndarray) -> pl.DataFrame:
        return self.edges.join(pl.DataFrame({"q": q_idx.astype(np.uint32)}), on="q")


def attach_reverse(pairs: pl.DataFrame, rev: pl.DataFrame, R: int) -> pl.DataFrame:
    """Add reverse-retrieval evidence to candidate pairs (q, d): where does this S1 rank among the S1 records
    that the target itself retrieves, and how far is it from the target's best S1."""
    best = rev.group_by("d").agg(pl.col("score").max().alias("rev_best"), pl.len().cast(pl.Float32).alias("rev_n"))
    out = (pairs.join(rev.select("d", "q", pl.col("score").alias("rev_score"), pl.col("rank").cast(pl.Float32).alias("rev_rank")),
                      on=["q", "d"], how="left", maintain_order="left")
           .join(best, on="d", how="left", maintain_order="left"))
    return out.with_columns(
        pl.col("rev_rank").fill_null(float(R + 1)),
        (pl.col("rev_best") - pl.col("rev_score")).alias("rev_gap"),
        (pl.col("rev_score") / pl.col("rev_best")).alias("rev_rel"),
        (pl.col("rev_rank") == 1).cast(pl.Float32).fill_null(0.0).alias("rev_is_top"),
    )
