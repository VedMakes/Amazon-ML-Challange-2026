#!/usr/bin/env python
"""Check the two output files and make <team>_submission.zip (output/, code/, Documentation_template.md).

  python code/business_entity_resolution/src/make_submission_zip.py --team COD-i [--test-dir dataset/test]

Run it from any folder. It uses the folder that contains code/ and output/ as the root of the ZIP.
Only the Python standard library is necessary.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]          # code/business_entity_resolution
ROOT = PKG.parents[1]
SKIP_DIRS = {"work", "__pycache__", "dataset", ".ipynb_checkpoints"}


def check_tsv(path: Path, header: str) -> tuple[int, int]:
    rows = ids = 0
    with open(path, encoding="utf-8") as f:
        first = f.readline().rstrip("\n")
        if first != header:
            sys.exit(f"{path.name}: header is {first!r}, expected {header!r}")
        for line in f:
            rows += 1
            lst = line.rstrip("\n").split("\t")[1]
            ids += lst.count(",") + 1 if lst else 0
    return rows, ids


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--team", default="COD-i")
    ap.add_argument("--test-dir", help="dataset/test folder: if given, the official validator runs before the ZIP is made")
    ap.add_argument("--validator", help="path of validate_submission.py (default: utils/ next to the dataset folder)")
    a = ap.parse_args()
    m, c, doc = ROOT / "output" / "matching_results.tsv", ROOT / "output" / "candidate_pairs.tsv", ROOT / "Documentation_template.md"
    for p in (m, c, doc, PKG / "README.md", PKG / "requirements.txt"):
        if not p.exists():
            sys.exit(f"missing: {p}")
    rm, im = check_tsv(m, "source1_entity_id\tmatched_entity_ids")
    rc, ic = check_tsv(c, "source1_entity_id\tcandidate_entity_ids")
    print(f"matching_results.tsv : {rm:,} S1 rows, {im:,} matched IDs ({im / max(rm, 1):.3f} for each S1)")
    print(f"candidate_pairs.tsv  : {rc:,} S1 rows, {ic:,} candidate pairs ({ic / max(rc, 1):.2f} for each S1)")
    if rm != rc:
        sys.exit("the two files do not have the same number of rows")
    if a.test_dir:
        test_dir = Path(a.test_dir)
        val = Path(a.validator) if a.validator else test_dir.parent.parent / "utils" / "validate_submission.py"
        if not val.exists():
            sys.exit(f"validator not found: {val} (use --validator)")
        r = subprocess.run([sys.executable, str(val), "--matching", str(m), "--candidate", str(c), "--test-dir", str(test_dir)])
        if r.returncode != 0:
            sys.exit("the official validator reported problems: no ZIP made")
    out = ROOT / f"{a.team}_submission.zip"
    pdf = ROOT / 'Documentation_template.pdf'
    files = [m, c, doc] + ([pdf] if pdf.exists() else []) + sorted(p for p in PKG.rglob("*") if p.is_file() and not (set(p.relative_to(PKG).parts) & SKIP_DIRS) and p.suffix != ".pyc")
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in files:
            z.write(p, p.relative_to(ROOT).as_posix())
    print(f"{out}  ({out.stat().st_size / 2**20:.0f} MB, {len(files)} files)")
    for p in files:
        print("  ", p.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
