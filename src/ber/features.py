"""Pairwise matcher features -- country-agnostic, no country one-hot.

Groups
  name     : rapidfuzz similarities on normalised / core / phonetic names, exact-key agreement,
             legal-form agreement, length features, script flags
  address  : rapidfuzz similarities, missingness, numeric / house / alphanumeric / postal agreement
  context  : target source, name ambiguity (how many records in the searched corpus share the
             normalised name -- label-free), candidate-set size
  retrieval: per-arm score / rank, provenance count, score relative to the query's best candidate
"""
from __future__ import annotations

import gc

import numpy as np
import polars as pl
from rapidfuzz import distance, fuzz
from rapidfuzz.process import cpdist

from . import store

Q_COLS = ["name_norm_key", "name_core_sorted", "name_phon_key", "addr_norm_key", "addr_num", "addr_alnum",
          "addr_postal", "addr_house", "legal_key", "name_nonascii", "addr_missing", "has_dba"]


def gather_side(split: str, which: str, idx: np.ndarray, n2: int, cols=Q_COLS) -> pl.DataFrame:
    """Gather meta columns for S1 rows (which='q') or global target rows (which='d'), column by column.
    Target rows are gathered per source file (no 10M-row concatenation)."""
    idx = np.asarray(idx, dtype=np.int64)
    out = {}
    if which == "d":
        in3 = idx >= n2
        order = np.argsort(in3, kind="stable")          # S2 rows first, then S3, original order inside
        inv = np.empty_like(order)
        inv[order] = np.arange(len(order))
    for c in cols:
        if which == "q":
            out[c] = store.load_meta(split, 1, [c])[c].gather(idx)
        else:
            p2 = store.load_meta(split, 2, [c])[c].gather(idx[~in3])
            p3 = store.load_meta(split, 3, [c])[c].gather(idx[in3] - n2)
            out[c] = pl.concat([p2, p3]).gather(inv)
            del p2, p3
        gc.collect()
    return pl.DataFrame(out)


class SideStore:
    """Preloaded meta columns for a sorted subset of rows (e.g. one country's targets) so that many
    chunks of pairs can be featurised without re-reading 10M-row columns each time."""

    def __init__(self, split: str, which: str, rows: np.ndarray, n2: int, cols=Q_COLS):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.frame = gather_side(split, which, self.rows, n2, cols)

    def gather(self, idx: np.ndarray) -> pl.DataFrame:
        pos = np.searchsorted(self.rows, idx)
        assert (self.rows[np.minimum(pos, len(self.rows) - 1)] == idx).all(), "row not in SideStore"
        return self.frame[pos]


def name_frequency(split: str, countries: list[str], key: str = "name_norm_key") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Label-free ambiguity statistics of the searched universe (#S1 / #targets sharing a normalised
    name), keyed by name hash + country code. Cached per split."""
    return store.load_name_freq(split, countries, key)


def _sim(fn, a: list, b: list) -> np.ndarray:
    return cpdist(a, b, scorer=fn, workers=-1, dtype=np.float32) / (100.0 if fn not in (distance.JaroWinkler.normalized_similarity,) else 1.0)


def _set_overlap(a: pl.Series, b: pl.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = pl.DataFrame({"a": a, "b": b})
    inter = df.select(pl.col("a").list.set_intersection(pl.col("b")).list.len()).to_series().to_numpy()
    la = a.list.unique().list.len().to_numpy()
    lb = b.list.unique().list.len().to_numpy()
    union = la + lb - inter
    jac = np.where(union > 0, inter / np.maximum(union, 1), np.nan).astype(np.float32)
    return inter.astype(np.float32), jac, np.minimum(la, lb).astype(np.float32)


def pair_features(split: str, pairs: pl.DataFrame, n2: int, freq: tuple | None = None, q_country=None,
                  qside: SideStore | None = None, dside: SideStore | None = None) -> pl.DataFrame:
    """pairs: q, d (+ optional retrieval columns score_*, rank_*, from_*, n_arms, rank_min)."""
    q = pairs["q"].to_numpy().astype(np.int64)
    d = pairs["d"].to_numpy().astype(np.int64)
    A = qside.gather(q) if qside is not None else gather_side(split, "q", q, n2)
    B = dside.gather(d) if dside is not None else gather_side(split, "d", d, n2)
    q_name_key = A["name_norm_key"]
    name_keys_d = B["name_norm_key"]
    f = {}
    an, bn = A["name_norm_key"].fill_null("").to_list(), B["name_norm_key"].fill_null("").to_list()
    f["n_ratio"] = _sim(fuzz.ratio, an, bn)
    f["n_tset"] = _sim(fuzz.token_set_ratio, an, bn)
    f["n_tsort"] = _sim(fuzz.token_sort_ratio, an, bn)
    f["n_partial"] = _sim(fuzz.partial_ratio, an, bn)
    f["n_jw"] = _sim(distance.JaroWinkler.normalized_similarity, an, bn)
    f["n_exact"] = (A["name_norm_key"] == B["name_norm_key"]).fill_null(False).to_numpy().astype(np.float32)
    ac, bc = A["name_core_sorted"].fill_null("").to_list(), B["name_core_sorted"].fill_null("").to_list()
    f["core_ratio"] = _sim(fuzz.ratio, ac, bc)
    f["core_tset"] = _sim(fuzz.token_set_ratio, ac, bc)
    f["core_exact"] = (A["name_core_sorted"] == B["name_core_sorted"]).fill_null(False).to_numpy().astype(np.float32)
    ap, bp = A["name_phon_key"].fill_null("").to_list(), B["name_phon_key"].fill_null("").to_list()
    f["phon_ratio"] = _sim(fuzz.ratio, ap, bp)
    f["phon_tset"] = _sim(fuzz.token_set_ratio, ap, bp)
    f["phon_exact"] = (A["name_phon_key"] == B["name_phon_key"]).fill_null(False).to_numpy().astype(np.float32)
    del an, bn, ac, bc, ap, bp
    la, lb = A["legal_key"].fill_null(""), B["legal_key"].fill_null("")
    f["legal_both"] = ((la != "") & (lb != "")).to_numpy().astype(np.float32)
    f["legal_equal"] = np.where(f["legal_both"] > 0, (la == lb).to_numpy(), np.nan).astype(np.float32)
    f["legal_any"] = ((la != "") | (lb != "")).to_numpy().astype(np.float32)
    f["q_name_len"] = A["name_norm_key"].str.len_chars().to_numpy().astype(np.float32)
    f["d_name_len"] = B["name_norm_key"].str.len_chars().to_numpy().astype(np.float32)
    f["q_nonascii"] = A["name_nonascii"].to_numpy().astype(np.float32)
    f["d_nonascii"] = B["name_nonascii"].to_numpy().astype(np.float32)
    f["d_dba"] = B["has_dba"].to_numpy().astype(np.float32)
    # address
    aa, ba = A["addr_norm_key"].fill_null("").to_list(), B["addr_norm_key"].fill_null("").to_list()
    f["a_ratio"] = _sim(fuzz.ratio, aa, ba)
    f["a_tset"] = _sim(fuzz.token_set_ratio, aa, ba)
    f["a_partial_tset"] = _sim(fuzz.partial_token_set_ratio, aa, ba)
    del aa, ba
    f["q_addr_missing"] = A["addr_missing"].to_numpy().astype(np.float32)
    f["d_addr_missing"] = B["addr_missing"].to_numpy().astype(np.float32)
    for k, (fa, fb) in {"num": ("addr_num", "addr_num"), "alnum": ("addr_alnum", "addr_alnum"), "postal": ("addr_postal", "addr_postal")}.items():
        inter, jac, mn = _set_overlap(A[fa], B[fb])
        f[f"{k}_inter"] = inter
        f[f"{k}_jac"] = jac
        f[f"{k}_conflict"] = np.where(mn > 0, (inter == 0).astype(np.float32), np.nan).astype(np.float32)
    hq, hd = A["addr_house"], B["addr_house"]
    both = (hq.is_not_null() & hd.is_not_null()).to_numpy()
    f["house_equal"] = np.where(both, (hq == hd).fill_null(False).to_numpy(), np.nan).astype(np.float32)
    # mutation-aware evidence: distractors in this data are near-copies with ONE controlled mutation (legal form,
    # house number, first name token) while true matches carry NOISE edits (digit split/drop, number words, typos)
    f.update(mutation_features(A, B))
    # sibling consistency: true matches of one S1 (its S2 and S3 records) resemble each other. The anchor is the
    # candidate with the highest name+address similarity to the S1; every candidate is compared with the anchor
    # (the anchor itself with the runner-up). A weak-looking true sibling scores high against a strong anchor.
    qa = pairs["q"].to_numpy()
    strength = (f["n_tset"] + np.nan_to_num(f["a_tset"])).astype(np.float32)
    g = pl.DataFrame({"q": qa, "s": strength}).with_row_index("i").with_columns(
        pl.col("s").rank("ordinal", descending=True).over("q").alias("r"))
    a1 = g.filter(pl.col("r") == 1).select("q", pl.col("i").alias("a1"), pl.col("s").alias("s1"))
    a2 = g.filter(pl.col("r") == 2).select("q", pl.col("i").alias("a2"))
    g = g.join(a1, on="q", how="left", maintain_order="left").join(a2, on="q", how="left", maintain_order="left")
    is_anchor = (g["i"] == g["a1"]).to_numpy()
    ref = np.where(is_anchor, g["a2"].fill_null(-1).cast(pl.Int64).to_numpy(), g["a1"].cast(pl.Int64).to_numpy())
    has = ref >= 0
    refc = np.maximum(ref, 0)
    bn = np.asarray(B["name_norm_key"].fill_null("").to_list(), dtype=object)
    ba = np.asarray(B["addr_norm_key"].fill_null("").to_list(), dtype=object)
    f["sib_name_tset"] = np.where(has, _sim(fuzz.token_set_ratio, bn.tolist(), bn[refc].tolist()), np.nan).astype(np.float32)
    f["sib_addr_tset"] = np.where(has, _sim(fuzz.token_set_ratio, ba.tolist(), ba[refc].tolist()), np.nan).astype(np.float32)
    _, sjac, _ = _set_overlap(B["addr_num"], B["addr_num"].gather(refc))
    f["sib_num_jac"] = np.where(has, sjac, np.nan).astype(np.float32)
    f["sib_is_anchor"] = is_anchor.astype(np.float32)
    f["sib_anchor_strength"] = g["s1"].to_numpy().astype(np.float32)
    del A, B, bn, ba, g
    gc.collect()
    out = pl.DataFrame(f)
    # context
    out = out.with_columns(pl.Series("d_is_S3", (d >= n2).astype(np.float32)))
    if freq is not None and q_country is not None:
        s1f, tf = freq
        keys = pl.DataFrame({"h": q_name_key.hash(seed=store.SEED), "country": q_country[q]})
        keys = keys.join(s1f, on=["h", "country"], how="left", maintain_order="left").join(tf, on=["h", "country"], how="left", maintain_order="left")
        out = out.with_columns(
            pl.Series("log_s1_same_name", np.log1p(keys["n_s1_same_name"].fill_null(0).to_numpy()).astype(np.float32)),
            pl.Series("log_t_same_name", np.log1p(keys["n_t_same_name"].fill_null(0).to_numpy()).astype(np.float32)))
    # retrieval provenance
    rcols = [c for c in pairs.columns if c.startswith(("score_", "rank_", "from_", "rev_")) or c in ("n_arms", "rank_min")]
    if rcols:
        r = pairs.select(["q"] + rcols).with_columns([pl.col(c).cast(pl.Float32) for c in rcols])
        rel = []
        for c in [c for c in rcols if c.startswith("score_")]:
            rel.append((pl.col(c) / pl.col(c).max().over("q")).alias(f"rel_{c}"))
        r = r.with_columns(rel + [pl.len().over("q").cast(pl.Float32).alias("n_cands_q")])
        out = pl.concat([out, r.drop("q")], how="horizontal")
    # rare-token weighted overlap (label-free token DF of the searched universe)
    out = pl.concat([out, rare_token_features(split, q, d, n2)], how="horizontal")
    # hybrid token-level fuzzy similarity (Monge-Elkan, soft TF-IDF) on name tokens
    out = pl.concat([out, fuzzy_token_features(split, q_name_key, name_keys_d)], how="horizontal")
    # within-query relative name/address evidence (is this the best-looking candidate for this S1?)
    out = pl.concat([pairs.select("q"), out], how="horizontal").with_columns(
        (pl.col("n_tset") - pl.col("n_tset").max().over("q")).alias("n_tset_gap"),
        (pl.col("a_tset") - pl.col("a_tset").max().over("q")).alias("a_tset_gap"),
        pl.col("n_tset").rank("dense", descending=True).over("q").cast(pl.Float32).alias("n_tset_rank"),
    ).drop("q")
    return out


def _gather_rows(split: str, view: str, idx: np.ndarray, n2: int, which: str):
    """Rows of the (memory-mapped) count matrix for S1 rows or global target rows, in the order of idx."""
    import scipy.sparse as sp
    if which == "q":
        return store.load_matrix(split, 1, view)[idx]
    in3 = idx >= n2
    order = np.argsort(in3, kind="stable")
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    m = sp.vstack([store.load_matrix(split, 2, view)[idx[~in3]], store.load_matrix(split, 3, view)[idx[in3] - n2]], format="csr")
    return m[inv]


def rare_token_features(split: str, q: np.ndarray, d: np.ndarray, n2: int) -> pl.DataFrame:
    """For name words, address words and numbers:
      ovl_q / ovl_d   IDF-weighted share of the S1 / target tokens that are matched
      miss_q / miss_d IDF of the rarest *unmatched* token on each side (a rare token present on one side only)
      distinct        sum of IDF / (#candidates of this S1 sharing the token): a match counts less when many of the
                      S1's other candidates share it (candidate-set-relative distinctiveness)
    """
    import scipy.sparse as sp
    f = {}
    for view in store.TOKEN_DF_VIEWS:
        dfa, n = store.load_token_df(split, view)
        A = _gather_rows(split, view, q, n2, "q").astype(np.float32)
        B = _gather_rows(split, view, d, n2, "d").astype(np.float32)
        A.data = np.log((n + 1) / (np.asarray(dfa[A.indices], dtype=np.float32) + 1)).astype(np.float32)
        B.data = np.log((n + 1) / (np.asarray(dfa[B.indices], dtype=np.float32) + 1)).astype(np.float32)
        Bb = B.copy(); Bb.data[:] = 1
        Mt = A.multiply(Bb).tocsr()                      # matched tokens, weighted by IDF
        wq, wd, wm = (np.asarray(x.sum(axis=1)).ravel() for x in (A, B, Mt))
        f[f"{view}_ovl_q"] = np.where(wq > 0, wm / np.maximum(wq, 1e-9), np.nan).astype(np.float32)
        f[f"{view}_ovl_d"] = np.where(wd > 0, wm / np.maximum(wd, 1e-9), np.nan).astype(np.float32)
        f[f"{view}_miss_q"] = np.asarray((A - Mt).max(axis=1).todense()).ravel().astype(np.float32)
        Ab = A.copy(); Ab.data[:] = 1
        f[f"{view}_miss_d"] = np.asarray((B - B.multiply(Ab)).max(axis=1).todense()).ravel().astype(np.float32)
        coo = Mt.tocoo()
        t = pl.DataFrame({"pair": coo.row.astype(np.int64), "tok": coo.col.astype(np.int64), "w": coo.data, "q": q[coo.row]})
        t = t.with_columns(pl.len().over(["q", "tok"]).alias("c"))
        dist = t.group_by("pair").agg((pl.col("w") / pl.col("c")).sum().alias("v"))
        arr = np.zeros(len(q), dtype=np.float32)
        arr[dist["pair"].to_numpy()] = dist["v"].to_numpy()
        f[f"{view}_distinct"] = arr
        del A, B, Bb, Mt, Ab, t
        gc.collect()
    return pl.DataFrame(f)


_NUMWORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen "
    "eighteen nineteen twenty".split())}
_NUMWORDS.update({w: str(i + 1) for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth "
    "sixteenth seventeenth eighteenth nineteenth twentieth".split())})


def _digits_expr(col: pl.Expr) -> pl.Expr:
    """Address -> list of numbers, number words converted to digits, leading zeros stripped."""
    return (col.fill_null("").str.split(" ").list.eval(pl.element().replace(_NUMWORDS))
            .list.eval(pl.element().filter(pl.element().str.contains(r"^\d+$")).str.strip_chars_start("0")))


def mutation_features(A: pl.DataFrame, B: pl.DataFrame) -> dict:
    from rapidfuzz.distance import Levenshtein
    f = {}
    df = pl.DataFrame({"a": A["addr_norm_key"], "b": B["addr_norm_key"], "na": A["name_norm_key"].fill_null(""),
                       "nb": B["name_norm_key"].fill_null(""), "la": A["legal_key"].fill_null(""), "lb": B["legal_key"].fill_null("")})
    df = df.with_columns(_digits_expr(pl.col("a")).alias("da"), _digits_expr(pl.col("b")).alias("db"))
    df = df.with_columns(
        pl.col("da").list.join("").alias("ca"), pl.col("db").list.join("").alias("cb"),
        pl.col("da").list.first().alias("ha"), pl.col("db").list.first().alias("hb"),
        pl.col("da").list.set_intersection(pl.col("db")).list.len().alias("ni"),
        pl.col("da").list.unique().list.len().alias("la_n"), pl.col("db").list.unique().list.len().alias("lb_n"),
    )
    ca, cb = df["ca"].fill_null("").to_list(), df["cb"].fill_null("").to_list()
    ha, hb = df["ha"].fill_null("").to_list(), df["hb"].fill_null("").to_list()
    both_c = (df["ca"].str.len_chars() > 0) & (df["cb"].str.len_chars() > 0)
    both_h = (df["ha"].fill_null("").str.len_chars() > 0) & (df["hb"].fill_null("").str.len_chars() > 0)
    lev_c = cpdist(ca, cb, scorer=Levenshtein.distance, workers=-1).astype(np.float32)
    lev_h = cpdist(ha, hb, scorer=Levenshtein.distance, workers=-1).astype(np.float32)
    bc, bh = both_c.to_numpy(), both_h.to_numpy()
    f["num_concat_equal"] = np.where(bc, (df["ca"] == df["cb"]).to_numpy(), np.nan).astype(np.float32)   # '1 40' == '140'
    f["num_concat_lev"] = np.where(bc, lev_c, np.nan).astype(np.float32)                                 # digit noise vs change
    f["house_lev"] = np.where(bh, lev_h, np.nan).astype(np.float32)
    hs = df.select(pl.col("ha").fill_null(""), pl.col("hb").fill_null(""))
    contain = hs.select(((pl.col("ha").str.contains(pl.col("hb"), literal=True) & (pl.col("hb") != "")) |
                         (pl.col("hb").str.contains(pl.col("ha"), literal=True) & (pl.col("ha") != ""))).alias("c"))["c"].to_numpy()
    f["house_contains"] = np.where(bh, contain, np.nan).astype(np.float32)                              # 4404 ~ 404
    f["house_len_diff"] = np.where(bh, np.abs(df["ha"].fill_null("").str.len_chars().to_numpy().astype(np.int32)
                                              - df["hb"].fill_null("").str.len_chars().to_numpy().astype(np.int32)), np.nan).astype(np.float32)
    mn = np.minimum(df["la_n"].to_numpy(), df["lb_n"].to_numpy())
    f["numw_inter"] = df["ni"].to_numpy().astype(np.float32)                                            # with number words
    f["numw_conflict"] = np.where(mn > 0, (df["ni"].to_numpy() == 0).astype(np.float32), np.nan).astype(np.float32)
    # legal forms: both present and disjoint -> a controlled mutation (LLC vs Corp)
    la, lb = df["la"], df["lb"]
    lsa, lsb = la.str.split(" "), lb.str.split(" ")
    inter = pl.DataFrame({"x": lsa, "y": lsb}).select(pl.col("x").list.set_intersection(pl.col("y")).list.len())["x"].to_numpy()
    bothl = ((la != "") & (lb != "")).to_numpy()
    f["legal_conflict"] = np.where(bothl, (inter == 0).astype(np.float32), np.nan).astype(np.float32)
    # first name token (distractors swap the leading token: 'Peters, Norman and Wong' -> 'Hanson, Norman and Wong')
    fa, fb = df["na"].str.split(" ").list.first().fill_null(""), df["nb"].str.split(" ").list.first().fill_null("")
    f["name_first_eq"] = (fa == fb).to_numpy().astype(np.float32)
    f["name_first_jw"] = cpdist(fa.to_list(), fb.to_list(), scorer=distance.JaroWinkler.normalized_similarity, workers=-1).astype(np.float32)
    f["name_last_eq"] = (df["na"].str.split(" ").list.last().fill_null("") == df["nb"].str.split(" ").list.last().fill_null("")).to_numpy().astype(np.float32)
    # pseudo-word rename: target name is one alphabetic token sharing nothing with the S1 name (DBA / replaced name)
    ntok_b = df["nb"].str.split(" ").list.len().to_numpy()
    f["d_name_single_token"] = (ntok_b == 1).astype(np.float32)
    f["d_name_len"] = df["nb"].str.len_chars().to_numpy().astype(np.float32)
    return f


def fuzzy_token_features(split: str, a_keys: pl.Series, b_keys: pl.Series, theta: float = 0.9) -> pl.DataFrame:
    """Hybrid token-level similarities between two names (token lists), robust to typos AND reordering:
      n_me_ab / n_me_ba  Monge-Elkan: mean over the tokens of one side of the best Jaro-Winkler match on the other
      n_me_idf           IDF-weighted Monge-Elkan (S1 side): rare tokens must find a close partner
      n_soft_tfidf       soft TF-IDF (Cohen et al.): IDF-weighted cosine where tokens with JW >= theta count as shared
    IDF comes from the label-free token DF of the searched universe (same hashing as the 'nw' view)."""
    n = len(a_keys)
    dfa, ndocs = store.load_token_df(split, "nw")
    base = pl.DataFrame({"pid": np.arange(n, dtype=np.int64), "a": a_keys.fill_null("").str.split(" "), "b": b_keys.fill_null("").str.split(" ")})
    at = base.select("pid", pl.col("a").alias("t")).explode("t").filter(pl.col("t") != "").with_row_index("ai")
    bt = base.select("pid", pl.col("b").alias("t")).explode("t").filter(pl.col("t") != "").with_row_index("bi")
    def idf(tok: pl.Series) -> np.ndarray:
        h = pl.DataFrame({"t": tok}).select(((pl.lit("nw:") + pl.col("t")).hash(seed=store.SEED) % store.DIM).alias("h"))["h"]
        return np.log((ndocs + 1) / (np.asarray(dfa[h.cast(pl.Int64).to_numpy()], dtype=np.float64) + 1)).astype(np.float32)
    at = at.with_columns(pl.Series("w", idf(at["t"])))
    bt = bt.with_columns(pl.Series("w", idf(bt["t"])))
    x = at.join(bt, on="pid", suffix="_b")                      # all token pairs within each record pair
    out = {k: np.full(n, np.nan, dtype=np.float32) for k in ("n_me_ab", "n_me_ba", "n_me_idf", "n_soft_tfidf")}
    if x.height:
        jw = cpdist(x["t"].to_list(), x["t_b"].to_list(), scorer=distance.JaroWinkler.normalized_similarity, workers=-1, dtype=np.float32)
        x = x.with_columns(pl.Series("jw", jw))
        best_a = x.group_by("ai").agg(pl.col("pid").first(), pl.col("w").first(), pl.col("jw").max().alias("m"),
                                      pl.col("w_b").sort_by("jw", descending=True).first().alias("wb"))
        best_b = x.group_by("bi").agg(pl.col("pid").first(), pl.col("jw").max().alias("m"))
        me_ab = best_a.group_by("pid").agg(pl.col("m").mean().alias("v"), ((pl.col("m") * pl.col("w")).sum() / pl.col("w").sum()).alias("vi"))
        me_ba = best_b.group_by("pid").agg(pl.col("m").mean().alias("v"))
        # soft TF-IDF: sum over S1 tokens with a close partner of w_a * w_b * jw, divided by the two L2 norms
        na = at.group_by("pid").agg((pl.col("w") ** 2).sum().sqrt().alias("na"))
        nb = bt.group_by("pid").agg((pl.col("w") ** 2).sum().sqrt().alias("nb"))
        st = (best_a.filter(pl.col("m") >= theta).group_by("pid").agg((pl.col("w") * pl.col("wb") * pl.col("m")).sum().alias("s"))
              .join(na, on="pid").join(nb, on="pid").with_columns((pl.col("s") / (pl.col("na") * pl.col("nb"))).clip(0, 1).alias("v")))
        for key, frame, col in (("n_me_ab", me_ab, "v"), ("n_me_idf", me_ab, "vi"), ("n_me_ba", me_ba, "v"), ("n_soft_tfidf", st, "v")):
            out[key][frame["pid"].to_numpy()] = frame[col].to_numpy()
        has_both = np.zeros(n, bool)
        has_both[np.intersect1d(at["pid"].to_numpy(), bt["pid"].to_numpy())] = True
        out["n_soft_tfidf"] = np.where(has_both & np.isnan(out["n_soft_tfidf"]), 0.0, out["n_soft_tfidf"]).astype(np.float32)
    return pl.DataFrame(out)


def pair_features_chunked(split: str, pairs: pl.DataFrame, n2: int, freq: tuple, q_country, chunk: int = 150_000) -> pl.DataFrame:
    """Featurise many pairs with bounded memory: sides preloaded once, chunks cut at S1 boundaries
    (within-query features need all of a query's candidates together)."""
    pairs = pairs.sort("q")
    qside = SideStore(split, "q", np.unique(pairs["q"].to_numpy()), n2)
    dside = SideStore(split, "d", np.unique(pairs["d"].to_numpy()), n2)
    qs = pairs["q"].to_numpy()
    starts = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1]])
    out, lo = [], 0
    while lo < len(qs):
        hi = min(lo + chunk, len(qs))
        if hi < len(qs):
            hi = starts[np.searchsorted(starts, hi)] if np.searchsorted(starts, hi) < len(starts) else len(qs)
        part = pairs.slice(lo, hi - lo)
        F = pair_features(split, part, n2, freq, q_country, qside=qside, dside=dside)
        out.append(pl.concat([part.select("q", "d"), F], how="horizontal"))
        lo = hi
        gc.collect()
    return pl.concat(out, how="diagonal")
