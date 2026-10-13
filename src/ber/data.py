"""Data loading: TSV -> parquet cache -> view cache. Memory-lean (pyarrow/polars, chunked)."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from .normalize import build_views

from .paths import CACHE, DATA, THREADS  # noqa: F401  (re-exported)

COLS = ["entity_id", "business_name", "business_address", "country"]

_PO = pacsv.ParseOptions(delimiter="\t", quote_char=False)
_CO = pacsv.ConvertOptions(
    column_types={c: pa.string() for c in COLS + ["source1_entity_id", "matched_entity_ids"]},
    strings_can_be_null=False,
)


def raw_path(split: str, src: int | str) -> Path:
    name = "ground_truth" if src == "gt" else f"source{src}"
    return DATA / split / f"{split}_{name}.tsv"


def load_raw(split: str, src: int | str) -> pl.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    tag = "gt" if src == "gt" else f"s{src}"
    p = CACHE / f"{split}_{tag}.parquet"
    if not p.exists():
        pq.write_table(pacsv.read_csv(raw_path(split, src), parse_options=_PO, convert_options=_CO), p, compression="zstd")
    return pl.read_parquet(p)


def load_views(split: str, src: int, columns: list[str] | None = None, chunk: int = 250_000) -> pl.DataFrame:
    """Views for one source file; built once in chunks and cached."""
    p = CACHE / f"{split}_s{src}_views.parquet"
    if not p.exists():
        from .normalize import _PHON_CACHE
        raw = load_raw(split, src)
        tmp = p.with_suffix(".tmp")
        writer = None
        for off in range(0, raw.height, chunk):
            part = build_views(raw.slice(off, chunk)).to_arrow()
            writer = writer or pq.ParquetWriter(tmp, part.schema, compression="zstd")
            writer.write_table(part)
        writer.close()
        tmp.rename(p)                                   # atomic: a killed build never leaves a "finished" file
        del raw
        _PHON_CACHE.clear()
    return pl.read_parquet(p, columns=columns)
