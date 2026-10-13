"""Sibling propagation over each S1's candidate graph (stage 3).

Half of the remaining loss is S1s that miss some of their true siblings: the missing S2/S3 records are usually IN
the candidate set but look weak in isolation (renamed/DBA name, script switch, number noise). Siblings of one
business resemble each other, so we decide each candidate relative to the S1's *accepted core*:

  core(q)  = candidates of q with stage-1 probability >= t_core (top `max_core` by probability)
  edges    = every candidate d of q  x  every core member c of q (c != d): target-target similarities
  features = best similarity of d to the core (name token-set, address token-set, numbers, phonetic, legal),
             probability of that best core member, #core members it strongly resembles, cross-source flag
             (S2<->S3 resemblance is the strongest sibling signal), core size / mean probability, d's own p / rank.

A stage-3 LightGBM re-decides every candidate from these + its stage-1 probability (trained on stage-1 OOF
probabilities, so no in-sample leakage).
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

from .features import SideStore, _set_overlap

SIB_TEXT = ["name_norm_key", "addr_norm_key", "addr_num", "name_phon_key", "legal_key"]
GRAPH_FEATS = ["g_is_core", "g_p_rank", "g_n_core", "g_core_mean_p", "g_best_sim", "g_best_name", "g_best_addr",
               "g_best_num", "g_best_phon_eq", "g_best_legal_eq", "g_best_p", "g_best_cross_src", "g_n_strong",
               "g_max_name", "g_max_addr", "g_best_rank_in_d"]


def sibling_graph_features(split: str, scored: pl.DataFrame, n2: int, t_core: float, max_core: int = 6,
                           dside: SideStore | None = None) -> pl.DataFrame:
    """scored: q, d, p (stage-1 probability). Returns q, d, p + GRAPH_FEATS, in the input row order."""
    s = scored.select("q", "d", "p").with_row_index("row")
    s = s.with_columns(pl.col("p").rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("g_p_rank"))
    core = (s.filter(pl.col("p") >= t_core).filter(pl.col("g_p_rank") <= max_core)
            .select("q", pl.col("d").alias("c"), pl.col("p").alias("pc")))
    stats = core.group_by("q").agg(pl.len().cast(pl.Float32).alias("g_n_core"), pl.col("pc").mean().alias("g_core_mean_p"))
    e = s.select("row", "q", "d").join(core, on="q").filter(pl.col("d") != pl.col("c"))
    out = s.join(stats, on="q", how="left", maintain_order="left").with_columns(
        (pl.col("p") >= t_core).cast(pl.Float32).alias("g_is_core"), pl.col("g_n_core").fill_null(0.0))
    if e.height == 0:
        for c in GRAPH_FEATS:
            if c not in out.columns:
                out = out.with_columns(pl.lit(None, dtype=pl.Float32).alias(c))
        return out.select("q", "d", "p", *GRAPH_FEATS)
    ud = np.unique(np.concatenate([e["d"].to_numpy(), e["c"].to_numpy()])).astype(np.int64)
    side = dside if dside is not None else SideStore(split, "d", ud, n2, SIB_TEXT)
    A = side.gather(e["d"].to_numpy().astype(np.int64))
    C = side.gather(e["c"].to_numpy().astype(np.int64))
    nm = cpdist(A["name_norm_key"].fill_null("").to_list(), C["name_norm_key"].fill_null("").to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100.0
    ad = cpdist(A["addr_norm_key"].fill_null("").to_list(), C["addr_norm_key"].fill_null("").to_list(), scorer=fuzz.token_set_ratio, workers=-1) / 100.0
    _, nj, _ = _set_overlap(A["addr_num"], C["addr_num"])
    ph = (A["name_phon_key"] == C["name_phon_key"]).fill_null(False).to_numpy().astype(np.float32)
    lg = (A["legal_key"] == C["legal_key"]).fill_null(False).to_numpy().astype(np.float32)
    cross = ((e["d"].to_numpy() < n2) != (e["c"].to_numpy() < n2)).astype(np.float32)
    ad_empty = (A["addr_norm_key"].fill_null("") == "").to_numpy() | (C["addr_norm_key"].fill_null("") == "").to_numpy()
    sim = np.where(ad_empty, nm, 0.5 * nm + 0.5 * ad).astype(np.float32)          # address-less: name only
    e = e.with_columns(pl.Series("nm", nm.astype(np.float32)), pl.Series("ad", ad.astype(np.float32)),
                       pl.Series("nj", np.nan_to_num(nj, nan=-1.0).astype(np.float32)), pl.Series("ph", ph), pl.Series("lg", lg),
                       pl.Series("cross", cross), pl.Series("sim", sim))
    best = (e.sort(["row", "sim"], descending=[False, True]).group_by("row", maintain_order=True)
            .agg(pl.col("sim").first().alias("g_best_sim"), pl.col("nm").first().alias("g_best_name"),
                 pl.col("ad").first().alias("g_best_addr"), pl.col("nj").first().alias("g_best_num"),
                 pl.col("ph").first().alias("g_best_phon_eq"), pl.col("lg").first().alias("g_best_legal_eq"),
                 pl.col("pc").first().alias("g_best_p"), pl.col("cross").first().alias("g_best_cross_src"),
                 (pl.col("sim") >= 0.85).sum().cast(pl.Float32).alias("g_n_strong"),
                 pl.col("nm").max().alias("g_max_name"), pl.col("ad").max().alias("g_max_addr")))
    out = out.join(best, on="row", how="left", maintain_order="left")
    # how does this target rank among ALL candidates by similarity to the S1's best core member? (1 = closest)
    top = core.sort(["q", "pc"], descending=[False, True]).unique("q", keep="first").select("q", pl.col("c").alias("c1"))
    e1 = e.join(top, on="q").filter(pl.col("c") == pl.col("c1")).select("row", "q", "sim")
    e1 = e1.with_columns(pl.col("sim").rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("g_best_rank_in_d"))
    out = out.join(e1.select("row", "g_best_rank_in_d"), on="row", how="left", maintain_order="left")
    return out.select("q", "d", "p", *GRAPH_FEATS)
