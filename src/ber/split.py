"""P1-P2: ground-truth graph, connected components, component-disjoint splits, leakage audit.

All ids are integers here: q = S1 row, d = global target row (S2 then S3).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def build_components(n_s1: int, n_t: int, eq: np.ndarray, ed: np.ndarray) -> np.ndarray:
    """Undirected graph over every training record (S1 nodes 0..n_s1-1, target nodes n_s1..).

    Edges = labelled S1->S2/S3 links. Returns component label per node.
    Unlinked targets form singleton components (never labels, only distractors).
    """
    n = n_s1 + n_t
    g = sp.coo_matrix((np.ones(len(eq), dtype=np.int8), (eq, ed + n_s1)), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    return lab


def assign_splits(q_country: np.ndarray, q_mult: np.ndarray, q_comp: np.ndarray, seed: int,
                  outer_frac: float = 0.2, n_folds: int = 5) -> pl.DataFrame:
    """Assign each component that contains an S1 to 'outer' or dev fold 0..n_folds-1.

    Stratified by (country, multiplicity bucket) with a seeded permutation inside each stratum.
    Assignment is done per *component*, so every record of a component shares one split.
    """
    comp = (pl.DataFrame({"component": q_comp, "country": q_country, "m": q_mult})
            .group_by("component").agg(pl.col("country").first(), pl.col("m").sum()))
    comp = comp.with_columns(pl.col("m").cut([0, 1, 3], labels=["0", "1", "2-3", "4+"]).cast(pl.String).alias("mult_bucket"))
    rng = np.random.default_rng(seed)
    comp = comp.sort("component").with_columns(pl.Series("u", rng.random(comp.height)))
    comp = comp.with_columns(pl.col("u").rank("ordinal").over(["country", "mult_bucket"]).alias("r"),
                             pl.len().over(["country", "mult_bucket"]).alias("nstrat"))
    is_outer = pl.col("r") <= (pl.col("nstrat") * outer_frac).round()
    return comp.with_columns(
        pl.when(is_outer).then(pl.lit("outer")).otherwise(pl.lit("dev")).alias("split"),
        pl.when(is_outer).then(-1).otherwise((pl.col("r") % n_folds).cast(pl.Int64)).alias("fold"),
    ).select("component", "country", "m", "mult_bucket", "split", "fold")


def leakage_audit(node_comp: np.ndarray, node_split: np.ndarray, node_fold: np.ndarray) -> dict:
    """Checks that no connected component spans two splits or two folds (labelled nodes only)."""
    df = pl.DataFrame({"c": node_comp, "s": node_split, "f": node_fold}).filter(pl.col("s") != "none")
    per = df.group_by("c").agg(pl.col("s").n_unique().alias("ns"), pl.col("f").n_unique().alias("nf"))
    return {
        "labelled_records": df.height,
        "components": per.height,
        "components_crossing_split": int((per["ns"] > 1).sum()),
        "components_crossing_fold": int((per["nf"] > 1).sum()),
    }
