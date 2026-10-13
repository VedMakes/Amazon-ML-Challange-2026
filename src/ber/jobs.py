"""Out-of-process jobs for the heavy pair computations (features, sibling graph).

Each job runs in a short-lived child process that writes its result to parquet and exits. All of its memory goes
back to the operating system, and many jobs can run in parallel.

    child entry point:  python -m ber.jobs <job_dir>
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import polars as pl

from .data import CACHE

JOBS = CACHE / "jobs" / os.environ.get("BER_RUN_TAG", "final")


def spawn(job_id: str, spec: dict, frames: dict) -> tuple[pl.DataFrame, dict]:
    """Run one job in a child process. spec is JSON-able; frames are written as parquet next to it."""
    d = JOBS / job_id
    d.mkdir(parents=True, exist_ok=True)
    for k, f in frames.items():
        f.write_parquet(d / f"{k}.parquet")
    (d / "spec.json").write_text(json.dumps(spec))
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1]) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    t = time.time()
    r = subprocess.run([sys.executable, "-m", "ber.jobs", str(d)], env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"job {job_id} failed (exit {r.returncode}):\n{r.stderr[-3000:]}")
    meta = json.loads((d / "meta.json").read_text())
    meta["wall_sec"] = round(time.time() - t, 1)
    out = pl.read_parquet(d / "out.parquet")
    shutil.rmtree(d, ignore_errors=True)         # the result is in memory (and in the run cache) -> free the disk
    return out, meta


def _vm_hwm_mb() -> int:
    """Peak resident set of this process (Linux)."""
    try:
        for line in open("/proc/self/status"):
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def _child(d: Path) -> None:
    from .universe import Universe

    spec = json.loads((d / "spec.json").read_text())
    t = time.time()
    U = Universe(spec["split"])
    if spec["kind"] == "graph":
        from .graph import sibling_graph_features
        out = sibling_graph_features(spec["split"], pl.read_parquet(d / "scored.parquet"), U.n2, spec["t_core"], spec.get("max_core", 6))
    elif spec["kind"] == "features":
        from .features import name_frequency, pair_features_chunked
        pairs = pl.read_parquet(d / "pairs.parquet")
        out = pair_features_chunked(spec["split"], pairs, U.n2, name_frequency(spec["split"], U.countries), U.q_country)
    else:
        raise ValueError(spec["kind"])
    out.write_parquet(d / "out.parquet")
    (d / "meta.json").write_text(json.dumps({"runtime_sec": round(time.time() - t, 1), "peak_ram_mb": _vm_hwm_mb()}))


if __name__ == "__main__":
    _child(Path(sys.argv[1]))
