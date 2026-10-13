"""Test inference for one chunk of S1 records: reverse evidence + features (child processes) -> matcher ->
sibling-graph stage -> decision rule. The result is cached, so an interrupted run continues at the next chunk."""
from __future__ import annotations

import numpy as np
import polars as pl

from . import jobs
from . import matcher as MT
from .runcache import cached_run
from .universe import attach_reverse


def _decide(scored: pl.DataFrame, decision: dict) -> pl.DataFrame:
    if decision["rule"] == "threshold":
        return MT.decide_threshold(scored, decision["t"])
    return MT.decide_expected_f(scored, decision["alpha"], decision["t_min"])


def match_chunk(tag: str, union: pl.DataFrame, model: MT.TwoStage, feats: list, decision: dict, max_pairs: int = 500_000,
                reverse: dict | None = None, graph: dict | None = None) -> pl.DataFrame:
    """One test chunk -> q, d, p, accepted. `union` = candidate pairs q, d (+ score / rank columns).
    `reverse` = {'parts': [parquet of q, d, score, rank], 'R': k} adds the reverse evidence.
    `graph` = {'model': fold models, 't_core': float} adds the sibling-graph stage."""
    def go():
        c = union.sort("q")
        if reverse is not None:
            ud = pl.Series(np.unique(c["d"].to_numpy()).astype(np.uint32)).implode()
            rev = pl.scan_parquet(reverse["parts"]).filter(pl.col("d").is_in(ud)).collect()
            c = attach_reverse(c, rev, reverse["R"])
        qs = c["q"].to_numpy()
        out, lo = [], 0
        while lo < len(qs):
            hi = min(lo + max_pairs, len(qs))
            while hi < len(qs) and qs[hi] == qs[hi - 1]:
                hi += 1
            F, _ = jobs.spawn(f"test-features-{tag}-{lo}", {"kind": "features", "split": "test"}, {"pairs": c.slice(lo, hi - lo)})
            for col in feats:
                if col not in F.columns:
                    F = F.with_columns(pl.lit(None, dtype=pl.Float32).alias(col))
            p = model.predict(F)                 # mean of the grouped-fold models
            scored = F.select("q", "d").with_columns(pl.Series("p", p))
            if graph is not None:        # stage 3: sibling graph features around the S1's accepted core, then re-decide
                from .graph import GRAPH_FEATS
                gid = f"test-graph-{tag}-{lo}"
                G, _ = jobs.spawn(gid, {"kind": "graph", "split": "test", "t_core": graph["t_core"]}, {"scored": scored})
                X3 = pl.concat([F.select(feats), G.select(["p"] + GRAPH_FEATS)], how="horizontal")
                scored = scored.with_columns(pl.Series("p", MT.predict(graph["model"], X3)))
            acc = _decide(scored, decision)
            if decision.get("gate") is not None:     # per-S1 emptiness gate (P(no true candidate) >= t0 -> predict empty)
                acc = MT.apply_gate(acc, decision["gate"].predict(scored, F), decision["t0"])
            acc = acc.with_columns(pl.lit(True).alias("accepted"))
            out.append(scored.join(acc, on=["q", "d"], how="left", maintain_order="left").with_columns(pl.col("accepted").fill_null(False)))
            lo = hi
        if not out:
            return pl.DataFrame(schema={"q": pl.UInt32, "d": pl.UInt32, "p": pl.Float64, "accepted": pl.Boolean})
        return pl.concat(out)
    return cached_run(f"test-scored-{tag}", go)[0]
