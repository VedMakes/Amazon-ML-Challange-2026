#!/usr/bin/env python
"""Write output/candidate_pairs.tsv from the step 1 result (dense_test.parquet) without the matcher.

The candidate set of the matcher is the dense top 20 of Source 2 + the dense top 20 of Source 3 for each S1 record.
Step 3 writes the same set (sorted by matcher probability; this script sorts by cosine score).

  python src/make_candidate_pairs.py --dense /path/to/dense_test.parquet --test-s1 /path/to/dataset/test/test_source1.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

K = 20                                                   # candidates for each source (settings.K)
ROOT = Path(__file__).resolve().parents[3]               # the folder that contains code/ and output/


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dense", required=True, help="dense_test.parquet of step 1")
    ap.add_argument("--test-s1", required=True, help="test_source1.tsv")
    ap.add_argument("--out", default=str(ROOT / "output" / "candidate_pairs.tsv"))
    ap.add_argument("--chunk", type=int, default=200_000, help="S1 records for each pass (memory control)")
    a = ap.parse_args()
    s1 = pl.read_csv(a.test_s1, separator="\t", quote_char=None, infer_schema_length=0, columns=["entity_id"])
    s1 = s1.with_columns(pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("s1_eid"))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs = empty = 0
    with open(out, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for o in range(0, s1.height, a.chunk):
            part = s1.slice(o, a.chunk)
            d = (pl.scan_parquet(a.dense).filter(pl.col("s1_eid").is_in(part["s1_eid"].implode()) & (pl.col("rank") <= K))
                 .select(pl.col("s1_eid").cast(pl.Int64), "t_eid", "src", "score").collect()
                 .unique(["s1_eid", "t_eid", "src"]).sort(["s1_eid", "score"], descending=[False, True])
                 .with_columns((pl.when(pl.col("src") == 2).then(pl.lit("S2-")).otherwise(pl.lit("S3-")) + pl.col("t_eid").cast(pl.String)).alias("tid")))
            pairs += d.height
            lists = d.group_by("s1_eid", maintain_order=True).agg(pl.col("tid").str.join(",").alias("c"))
            rows = part.join(lists, on="s1_eid", how="left", maintain_order="left").with_columns(pl.col("c").fill_null(""))
            empty += int((rows["c"] == "").sum())
            f.write("\n".join((rows["entity_id"] + "\t" + rows["c"]).to_list()) + "\n")
            print(f"{min(o + a.chunk, s1.height):,}/{s1.height:,} S1 records", flush=True)
    print(f"{out}: {s1.height:,} rows, {pairs:,} candidate pairs ({pairs / s1.height:.2f} for each S1), {empty:,} S1 without candidates")


if __name__ == "__main__":
    main()
