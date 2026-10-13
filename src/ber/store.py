"""Disk-backed feature store: one hashed sparse *count* matrix per (split, source, view).

Counts are label-free and weighting-free, so they can be precomputed once for every record.
The document frequencies for the rare-token features are computed later from all
records of one split without labels (load_token_df).

Feature id = hash(namespace + ":" + token) mod 2^NBITS (polars xxhash, fixed seed).
"""
from __future__ import annotations

import gc

import numpy as np
import polars as pl
import scipy.sparse as sp

from .data import CACHE, load_views
from .normalize import LEGAL_FORMS, NAME_FILLERS

NBITS = 25
DIM = 1 << NBITS
SEED = 20260926

# view name -> (source column, mode). The matcher features read these three views.
VIEWS = {
    "nw": ("name_tok", "tok"),     # name words (legal aliases canonicalised)
    "aw": ("addr_tok", "tok"),     # address words
    "an": ("addr_num", "tok"),     # numeric tokens (zero-stripped)
}
META_COLS = ["entity_id", "country", "business_name", "name_nonascii", "name_core_sorted", "addr_house", "addr_postal", "addr_alnum", "addr_num"]


def _mat_path(split, src, view):
    d = CACHE / "mat"
    d.mkdir(exist_ok=True, parents=True)
    return d / f"{split}_s{src}_{view}.npz"


def _explode(df: pl.DataFrame, col: str, mode: str, ns: str) -> pl.DataFrame:
    if mode == "tok":
        ex = df.select("row", pl.col(col).alias("f")).explode("f")
    elif mode == "rawtok":
        ex = df.select("row", pl.col(col).fill_null("").str.split(" ").alias("f")).explode("f")
    else:
        n = int(mode[-1])
        s = " " + pl.col(col).list.join(" ") + " "
        ex = (df.select("row", s.alias("s"))
              .with_columns(pl.int_ranges(0, (pl.col("s").str.len_chars() - n + 1).clip(0)).alias("pos"))
              .explode("pos").select("row", pl.col("s").str.slice(pl.col("pos"), n).alias("f")))
    return (ex.filter(pl.col("f").is_not_null() & (pl.col("f") != ""))
            .select("row", ((pl.lit(ns + ":") + pl.col("f")).hash(seed=SEED) % DIM).cast(pl.UInt32).alias("c")))


def build_matrix(split: str, src: int, view: str, chunk: int = 500_000) -> sp.csr_matrix:
    p = _mat_path(split, src, view)
    if p.exists() or _mm_dir(split, src, view).exists():
        return load_matrix(split, src, view)
    col, mode = VIEWS[view]
    v = load_views(split, src, [col])
    blocks = []
    for off in range(0, v.height, chunk):
        part = v.slice(off, chunk).with_row_index("row")
        ex = _explode(part, col, mode, view).group_by(["row", "c"]).len()
        m = sp.csr_matrix((ex["len"].cast(pl.UInt16).to_numpy(), (ex["row"].to_numpy(), ex["c"].to_numpy())),
                          shape=(part.height, DIM), dtype=np.uint16)
        m.sort_indices()
        blocks.append(m)
        del ex, part
        gc.collect()
    m = sp.vstack(blocks, format="csr", dtype=np.uint16)
    m.indices = m.indices.astype(np.int32)
    tmp = p.with_name(p.stem + ".tmp.npz")
    sp.save_npz(tmp, m, compressed=False)
    tmp.rename(p)
    return m


def _mm_dir(split, src, view):
    return CACHE / "mat" / f"{split}_s{src}_{view}"


def _to_mmap(split: str, src: int, view: str) -> None:
    """Convert the .npz count matrix to three raw .npy arrays that can be memory-mapped."""
    d = _mm_dir(split, src, view)
    m = sp.load_npz(_mat_path(split, src, view)).tocsr()
    tmp = d.with_suffix(".tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    np.save(tmp / "data.npy", m.data.astype(np.uint16))
    np.save(tmp / "indices.npy", m.indices.astype(np.int32))
    np.save(tmp / "indptr.npy", m.indptr.astype(np.int64 if m.nnz >= 2**31 else np.int32))
    np.save(tmp / "shape.npy", np.array(m.shape, dtype=np.int64))
    tmp.rename(d)
    _mat_path(split, src, view).unlink()


def load_matrix(split: str, src: int, view: str) -> sp.csr_matrix:
    """Memory-mapped CSR count matrix: pages are file-backed (reclaimable), not anonymous RAM."""
    d = _mm_dir(split, src, view)
    if not d.exists():
        if not _mat_path(split, src, view).exists():
            build_matrix(split, src, view)
        _to_mmap(split, src, view)
    data = np.load(d / "data.npy", mmap_mode="r")
    indices = np.load(d / "indices.npy", mmap_mode="r")
    indptr = np.load(d / "indptr.npy", mmap_mode="r")
    shape = tuple(np.load(d / "shape.npy"))
    m = sp.csr_matrix(shape, dtype=np.uint16)
    m.data, m.indices, m.indptr = data, indices, indptr   # attach without copy or validation
    return m


def load_meta(split: str, src: int, columns: list[str] | None = None) -> pl.DataFrame:
    """Per-record scalar/list keys (row order == matrix row order). Load only needed columns."""
    p = CACHE / f"{split}_s{src}_meta.parquet"
    if not p.exists():
        _build_meta(split, src)
    return pl.read_parquet(p, columns=columns)


def _build_meta(split: str, src: int, batch: int = 400_000) -> None:
    """Per-record keys, built batch by batch from the view cache (bounded memory)."""
    import pyarrow.parquet as pq
    from .data import load_views
    p = CACHE / f"{split}_s{src}_meta.parquet"
    if p.exists():
        return
    load_views(split, src, ["entity_id"])            # make sure the view cache exists
    legal = sorted(LEGAL_FORMS - NAME_FILLERS)
    cols = META_COLS + ["name_tok", "addr_tok", "name_phon", "name_core"]
    tmp = p.with_suffix(".tmp")
    writer = None
    for rb in pq.ParquetFile(CACHE / f"{split}_s{src}_views.parquet").iter_batches(batch_size=batch, columns=cols):
        v = pl.from_arrow(rb).with_columns(
            pl.col("name_tok").list.join(" ").alias("name_norm_key"),
            pl.col("addr_tok").list.join(" ").alias("addr_norm_key"),
            pl.col("addr_num").list.sort().list.join("|").alias("num_sig"),
            (pl.col("addr_num").list.len() > 0).alias("has_num"),
            (pl.col("addr_tok").list.len() == 0).alias("addr_missing"),
            pl.col("addr_tok").list.len().alias("addr_ntok"),
            pl.col("name_phon").list.sort().list.join(" ").alias("name_phon_key"),
            pl.when(pl.col("addr_house").is_not_null() & (pl.col("name_core").list.len() > 0))
            .then(pl.col("addr_house") + "|" + pl.col("name_core").list.sort().list.first()).alias("house_name_key"),
            pl.col("name_tok").list.eval(pl.element().filter(pl.element().is_in(legal)))
            .list.unique().list.sort().list.join(" ").alias("legal_key"),
            pl.col("name_tok").list.eval(pl.element().filter(pl.element() == "dba")).list.len().gt(0).alias("has_dba"),
            pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("eid"),
        ).drop("name_tok", "addr_tok", "name_phon", "name_core")
        t = v.to_arrow()
        writer = writer or pq.ParquetWriter(tmp, t.schema, compression="zstd")
        writer.write_table(t)
    writer.close()
    tmp.rename(p)


def load_ids(split: str) -> dict:
    """eid (int64) and country code (int16) per source + the country label list, cached as .npz.
    Country codes index a sorted list built from the data itself (open set). Read from the raw files, so the
    GPU steps do not need the feature store."""
    from .data import load_raw
    p = CACHE / f"{split}_ids.npz"
    if not p.exists():
        cols = {s: load_raw(split, s).select(pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("eid"), "country") for s in (1, 2, 3)}
        countries = sorted(set().union(*[set(c["country"].unique().to_list()) for c in cols.values()]))
        cmap = {c: i for i, c in enumerate(countries)}
        out = {"countries": np.array(countries, dtype=str)}
        for s, c in cols.items():
            out[f"eid{s}"] = c["eid"].to_numpy()
            out[f"c{s}"] = c["country"].replace_strict(cmap, return_dtype=pl.Int16).to_numpy()
        np.savez(p, **out)
    z = np.load(p, allow_pickle=False)
    return {"countries": [str(c) for c in z["countries"]], **{k: z[k] for k in z.files if k != "countries"}}


def load_name_freq(split: str, countries: list[str], key: str = "name_norm_key") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Label-free name-ambiguity tables keyed by (xxhash64 of the normalised name, country code):
    how many S1 records / how many targets in this universe share the name. Cached per split."""
    p1, pt = CACHE / f"{split}_namefreq_s1.parquet", CACHE / f"{split}_namefreq_t.parquet"
    if not (p1.exists() and pt.exists()):
        ids = load_ids(split)
        def tab(s):
            k = load_meta(split, s, [key])[key].hash(seed=SEED)
            return pl.DataFrame({"h": k, "country": ids[f"c{s}"]})
        tab(1).group_by(["h", "country"]).len().rename({"len": "n_s1_same_name"}).write_parquet(p1)
        pl.concat([tab(2), tab(3)]).group_by(["h", "country"]).len().rename({"len": "n_t_same_name"}).write_parquet(pt)
    return pl.read_parquet(p1), pl.read_parquet(pt)


TOKEN_DF_VIEWS = ["nw", "aw", "an"]


def load_token_df(split: str, view: str) -> tuple[np.ndarray, int]:
    """Label-free document frequency of every hashed token of `view` over all records of one split
    (S1+S2+S3). Used for rare-token-weighted overlap features. Returns (memory-mapped df, n_docs)."""
    p = CACHE / "mat" / f"{split}_df_{view}.npy"
    if not p.exists():
        df = np.zeros(DIM, dtype=np.int32)
        n = 0
        for s in (1, 2, 3):
            m = load_matrix(split, s, view)
            for o in range(0, m.nnz, 50_000_000):
                u, c = np.unique(np.asarray(m.indices[o:o + 50_000_000]), return_counts=True)
                df[u] += c.astype(np.int32)
            n += m.shape[0]
        np.save(p, df)
        (CACHE / "mat" / f"{split}_df_n.txt").write_text(str(n))
    return np.load(p, mmap_mode="r"), int((CACHE / "mat" / f"{split}_df_n.txt").read_text())


def build_all(views=TOKEN_DF_VIEWS, splits=("train", "test"), log=print) -> list[dict]:
    """Build the complete feature store from the raw TSVs: parquet cache, representation views, per-record
    meta keys, compact ids, name-frequency tables and one memory-mapped count matrix per (source, view).
    Idempotent: finished artefacts are skipped."""
    import time
    rows = []
    for split in splits:
        for s in (1, 2, 3):
            t = time.time()
            load_views(split, s, ["entity_id"])
            load_meta(split, s, ["eid"])
            tv = time.time() - t
            for v in views:
                t2 = time.time()
                m = load_matrix(split, s, v)
                rows.append({"split": split, "source": s, "view": v, "rows": m.shape[0], "nnz": int(m.nnz),
                             "seconds": round(time.time() - t2, 1)})
                del m
            log(f"{split} source{s}: views+meta {tv:.0f}s, matrices {sum(r['seconds'] for r in rows if r['split']==split and r['source']==s):.0f}s")
        load_ids(split)
        load_name_freq(split, None)
        for v in TOKEN_DF_VIEWS:
            load_token_df(split, v)
    return rows
