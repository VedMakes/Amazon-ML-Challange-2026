"""Folders and thread count. Each value comes from an environment variable or from a default.

  BER_DATA_DIR   folder that contains train/ and test/ (the challenge TSV files)
  BER_WORK_DIR   intermediate files (cache, dense candidates, cross-encoder scores, matcher model)
  BER_CACHE_DIR  feature store and run cache (default: <work>/cache)
  BER_OUT_DIR    the two submission TSV files (default: the output/ folder at the top of the ZIP)
  BER_THREADS    CPU threads (default: the CPUs this process can use)

The step scripts set these variables from their command-line options (see ber/cli.py). The resolved values are
written back to the environment, so the child processes of ber/jobs.py use the same folders.
"""
from __future__ import annotations

import os
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]            # .../code/business_entity_resolution
ZIP_ROOT = PKG_ROOT.parents[1]                            # the folder that contains code/ and output/


def _data_dir() -> Path:
    if os.environ.get("BER_DATA_DIR"):
        return Path(os.environ["BER_DATA_DIR"])
    for c in (PKG_ROOT / "dataset", ZIP_ROOT / "dataset", Path.cwd() / "dataset"):
        if (c / "train" / "train_source1.tsv").exists():
            return c
    return PKG_ROOT / "dataset"


def _cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


DATA = _data_dir()
WORK = Path(os.environ.get("BER_WORK_DIR", PKG_ROOT / "work"))
CACHE = Path(os.environ.get("BER_CACHE_DIR", WORK / "cache"))
OUT = Path(os.environ.get("BER_OUT_DIR", ZIP_ROOT / "output"))
THREADS = int(os.environ.get("BER_THREADS", _cpus()))
DENSE_DIR = WORK / "dense"        # step 1: dense_train.parquet, dense_test.parquet, model/
CE_DIR = WORK / "ce"              # step 2: ce_model/, ce_dev.parquet, ce_test_<chunk>.parquet
MATCHER_DIR = WORK / "matcher"    # step 3: matcher.pkl, scored_test_<chunk>.parquet, reports

os.environ.update({"BER_DATA_DIR": str(DATA), "BER_WORK_DIR": str(WORK), "BER_CACHE_DIR": str(CACHE),
                   "BER_OUT_DIR": str(OUT), "BER_THREADS": str(THREADS)})


def check_data() -> None:
    need = [DATA / s / f"{s}_source{i}.tsv" for s in ("train", "test") for i in (1, 2, 3)] + [DATA / "train" / "train_ground_truth.tsv"]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError("Challenge data not found. Use --data-dir or BER_DATA_DIR. Missing files:\n  " + "\n  ".join(missing))
