"""On-disk run cache: each step result is stored once, so an interrupted run continues where it stopped."""
from __future__ import annotations

import json
import os
import time

import polars as pl

from .data import CACHE
from .universe import rss_mb

RUNS = CACHE / "runs" / os.environ.get("BER_RUN_TAG", "final")


def cached_run(run_id: str, fn, force: bool = False) -> tuple[pl.DataFrame, dict]:
    """Execute fn() -> DataFrame once; persist parquet + runtime metadata; reload on later calls."""
    RUNS.mkdir(parents=True, exist_ok=True)
    p, m = RUNS / f"{run_id}.parquet", RUNS / f"{run_id}.json"
    if p.exists() and m.exists() and not force:
        return pl.read_parquet(p), json.loads(m.read_text())
    t = time.time()
    res = fn()
    df, child = res if isinstance(res, tuple) else (res, {})
    meta = {"runtime_sec": round(time.time() - t, 1), "peak_ram_mb": child.get("peak_ram_mb", rss_mb()),
            **({"child_runtime_sec": child["runtime_sec"]} if "runtime_sec" in child else {})}
    df.write_parquet(p)
    m.write_text(json.dumps(meta))
    return df, meta
