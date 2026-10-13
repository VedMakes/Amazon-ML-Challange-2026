#!/usr/bin/env python
"""Step 1 (GPU): blocking with a fine-tuned bi-encoder -> the candidate set.

1. Fine-tune `intfloat/multilingual-e5-small` (MIT licence, 118 M parameters) with a contrastive loss.
   Training records: S1 records with eid % 10 == 0 (10 % of the training S1). The matcher never uses these records.
   Each training row = (S1 record, one true target, one hard negative). In-batch negatives are used also.
2. Encode all records. For each S1, find the 20 nearest targets in Source 2 and the 20 nearest in Source 3
   (exact search on the GPU, same country only). Do this for the training data and for the test data.

Outputs in <work>/dense: model/, dense_train.parquet, dense_test.parquet, dense_report.json
Columns of the parquet files: s1_eid, t_eid, src (2 or 3), score (cosine), rank (1..20 for each source).

  python src/step1_dense_blocker.py            # finished parts are not done again; use --force to do them again
"""
from __future__ import annotations

import gc
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber import cli  # noqa: E402

ARGS = cli.parse(__doc__, lambda ap: ap.add_argument("--force", action="store_true", help="train and search again even if the outputs exist"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses  # noqa: E402
from sentence_transformers.training_args import BatchSamplers  # noqa: E402

from ber import settings as S  # noqa: E402
from ber.paths import DATA, DENSE_DIR, check_data  # noqa: E402

MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_LEN = 64                  # tokens for each record ("query: name | address")
BATCH_TRAIN = 256             # one GPU (the submitted model was trained on one GPU)
EPOCHS, LR = 1, 5e-5
ENC_BATCH = 1024
Q_CHUNK, D_BLOCK = 1024, 1_000_000        # tiles of the exact search (about 2 GB for each score block in fp16)
CUDA = torch.cuda.is_available()
DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
MODELS: dict = {}


def log(*a):
    print(*a, flush=True)


def read(split, name):
    return pl.read_csv(DATA / split / f"{split}_{name}.tsv", separator="\t", quote_char=None, infer_schema_length=0)


def records(split):
    out = {}
    for s in (1, 2, 3):
        d = read(split, f"source{s}")
        out[s] = d.select(pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("eid"), "country",
                          ("query: " + pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")).alias("text"))
    return out


def ground_truth():
    gt = read("train", "ground_truth")
    return (gt.select(pl.col("source1_entity_id").str.slice(3).cast(pl.Int64).alias("s1"), pl.col("matched_entity_ids").str.split(",").alias("t"))
            .explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
            .select("s1", pl.col("t").str.slice(1, 1).cast(pl.Int8).alias("src"), pl.col("t").str.slice(3).cast(pl.Int64).alias("t_eid")))


def training_triplets(TR, E):
    """(S1 text, true target text, hard negative text) for the training S1 records."""
    tr_s1 = TR[1].filter(pl.col("eid") % S.DENSE_TRAIN_MOD == 0)
    tgt = pl.concat([TR[2].with_columns(pl.lit(2, dtype=pl.Int8).alias("src")), TR[3].with_columns(pl.lit(3, dtype=pl.Int8).alias("src"))])
    pairs = (E.join(tr_s1.select(pl.col("eid").alias("s1"), pl.col("text").alias("a"), "country"), on="s1")
             .join(tgt.select(pl.col("eid").alias("t_eid"), "src", pl.col("text").alias("p")), on=["t_eid", "src"]))
    # hard negative: a record of a different business whose name starts with the same word (easy to confuse)
    first = lambda c: pl.col(c).str.slice(7).str.split(" ").list.first().str.to_lowercase()   # noqa: E731
    owner = E.select(pl.col("t_eid"), "src", pl.col("s1").alias("owner"))
    neg_pool = (tgt.rename({"eid": "t_eid"}).join(owner, on=["t_eid", "src"], how="left").with_columns(first("text").alias("ft"))
                .select("ft", "country", "owner", pl.col("text").alias("n"))
                .sample(fraction=1.0, shuffle=True, seed=S.SEED)
                .group_by(["ft", "country"]).head(20))                      # at most 20 for each group (no join explosion)
    pairs = pairs.with_columns(first("a").alias("ft")).with_row_index("i")
    cand = (pairs.select("i", "ft", "country", "s1").join(neg_pool, on=["ft", "country"])
            .filter(pl.col("owner").is_null() | (pl.col("owner") != pl.col("s1")))       # never a record of the same S1
            .unique("i", keep="any"))
    pairs = pairs.join(cand.select("i", "n"), on="i", how="left")
    rand_txt = tgt["text"].sample(n=pairs.height, with_replacement=True, seed=S.SEED + 1)   # fallback: a random target
    pairs = pairs.with_columns(pl.Series("rn", rand_txt)).with_columns(pl.coalesce("n", "rn").alias("n")).drop("rn")
    log(f"training triplets {pairs.height:,} from {tr_s1.height:,} S1 | hard negatives {cand.height / pairs.height:.1%}")
    return pairs


def train(pairs) -> int:
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=DEVICES[0])
    model.max_seq_length = MAX_LEN
    ds = Dataset.from_dict({"anchor": pairs["a"].to_list(), "positive": pairs["p"].to_list(), "negative": pairs["n"].to_list()})
    args = SentenceTransformerTrainingArguments(
        output_dir=str(DENSE_DIR / "ckpt"), num_train_epochs=EPOCHS, per_device_train_batch_size=BATCH_TRAIN,
        learning_rate=LR, warmup_ratio=0.05, fp16=CUDA, batch_sampler=BatchSamplers.NO_DUPLICATES,   # siblings share the anchor
        logging_steps=200, save_strategy="no", report_to="none", dataloader_num_workers=2, seed=S.SEED)
    args._n_gpu = 1                                           # one GPU (as in the submitted run)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=losses.MultipleNegativesRankingLoss(model))
    trainer.train()
    model.save(str(DENSE_DIR / "model"))
    del trainer, model, ds
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()
    sec = round(time.time() - t0)
    log(f"fine-tuned on {pairs.height:,} triplets in {sec} s -> {DENSE_DIR / 'model'}")
    return sec


def load_models():
    for dev in DEVICES:                                       # one copy of the model for each GPU
        m = SentenceTransformer(str(DENSE_DIR / "model"), device=dev)
        m.max_seq_length = MAX_LEN
        if CUDA:
            m.half()
        m.eval()
        MODELS[dev] = m
    log("encode / search devices:", DEVICES)


def _encode_on(dev, texts, out, lo, t0, total):
    step = 250_000
    for o in range(0, len(texts), step):
        chunk = texts[o:o + step]
        out[lo + o:lo + o + len(chunk)] = MODELS[dev].encode(chunk, batch_size=ENC_BATCH, normalize_embeddings=True,
                                                             convert_to_numpy=True, show_progress_bar=False).astype(np.float16)
        if dev == DEVICES[0]:
            done = min(o + step, len(texts)) * len(DEVICES)
            log(f"    encoded ~{min(done, total):,}/{total:,} | {done / max(time.time() - t0, 1):.0f}/s")


def encode(texts):
    """All GPUs in parallel: one contiguous part of the list for each device."""
    out = np.empty((len(texts), MODELS[DEVICES[0]].get_sentence_embedding_dimension()), dtype=np.float16)
    bounds = np.linspace(0, len(texts), len(DEVICES) + 1).astype(int)
    t0 = time.time()
    with ThreadPoolExecutor(len(DEVICES)) as ex:
        list(ex.map(lambda i: _encode_on(DEVICES[i], texts[bounds[i]:bounds[i + 1]], out, bounds[i], t0, len(texts)), range(len(DEVICES))))
    return out


@torch.no_grad()
def _search_on(dev, Q, D, k):
    """Exact top-k inner product on one device, in tiles of queries and documents."""
    cast = (lambda x: x) if dev != "cpu" else (lambda x: x.float())
    Dg = cast(torch.from_numpy(D).to(dev))
    best_s = np.full((len(Q), k), -1, np.float32)
    best_i = np.full((len(Q), k), -1, np.int64)
    for qo in range(0, len(Q), Q_CHUNK):
        q = cast(torch.from_numpy(Q[qo:qo + Q_CHUNK]).to(dev))
        s_acc, i_acc = None, None
        for do in range(0, len(D), D_BLOCK):
            sc = q @ Dg[do:do + D_BLOCK].T
            v, ix = torch.topk(sc, min(k, sc.shape[1]), dim=1)
            ix = ix + do
            s_acc = v if s_acc is None else torch.cat([s_acc, v], 1)
            i_acc = ix if i_acc is None else torch.cat([i_acc, ix], 1)
            if s_acc.shape[1] > k:
                v, j = torch.topk(s_acc, k, dim=1)
                s_acc, i_acc = v, torch.gather(i_acc, 1, j)
        n = s_acc.shape[1]
        best_s[qo:qo + len(q), :n] = s_acc.float().cpu().numpy()
        best_i[qo:qo + len(q), :n] = i_acc.cpu().numpy()
    del Dg
    if dev != "cpu":
        with torch.cuda.device(dev):
            torch.cuda.empty_cache()
    return best_i, best_s


def search(Q, D, k):
    parts = np.array_split(np.arange(len(Q)), len(DEVICES))
    with ThreadPoolExecutor(len(DEVICES)) as ex:
        res = list(ex.map(lambda i: _search_on(DEVICES[i], Q[parts[i]], D, k), range(len(DEVICES))))
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


def dense_split(split, R):
    """Top-K targets of each source for each S1 of the split (same country only) -> dense_<split>.parquet."""
    t0 = time.time()
    K = S.K
    e1 = encode(R[1]["text"].to_list())
    parts = []
    for s in (2, 3):
        et = encode(R[s]["text"].to_list())
        for c in R[1]["country"].unique().sort().to_list():
            qi = np.flatnonzero((R[1]["country"] == c).to_numpy())
            di = np.flatnonzero((R[s]["country"] == c).to_numpy())
            if len(qi) == 0 or len(di) == 0:
                continue
            idx, sc = search(e1[qi], et[di], K)
            ok = idx >= 0
            rows = np.repeat(qi, K).reshape(-1, K)
            parts.append(pl.DataFrame({"s1_eid": R[1]["eid"].to_numpy()[rows[ok]], "t_eid": R[s]["eid"].to_numpy()[di[idx[ok]]],
                                       "src": np.full(ok.sum(), s, np.int8), "score": sc[ok],
                                       "rank": np.tile(np.arange(1, K + 1, dtype=np.uint16), (len(qi), 1))[ok]}))
            log(f"  {split} S{s} {c}: {len(qi):,} S1 x {len(di):,} targets | {time.time() - t0:.0f} s")
        del et
        gc.collect()
    out = pl.concat(parts)
    tmp = DENSE_DIR / f"dense_{split}.tmp"
    out.write_parquet(tmp)
    tmp.rename(DENSE_DIR / f"dense_{split}.parquet")
    log(f"{split}: {out.height:,} rows -> dense_{split}.parquet | {time.time() - t0:.0f} s")
    return out


def main():
    check_data()
    DENSE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(S.SEED)
    log(json.dumps({"data": str(DATA), "output": str(DENSE_DIR), "model": MODEL_NAME, "devices": DEVICES}, indent=1))
    rep_path = DENSE_DIR / "dense_report.json"
    report = json.loads(rep_path.read_text()) if rep_path.exists() and not ARGS.force else {"model": MODEL_NAME, "K": S.K}
    need_train = ARGS.force or not (DENSE_DIR / "model" / "config.json").exists()
    need = {s: ARGS.force or not (DENSE_DIR / f"dense_{s}.parquet").exists() for s in ("train", "test")}
    if need_train or need["train"]:
        TR, E = records("train"), ground_truth()
        log({s: d.height for s, d in TR.items()}, "| links", E.height)
        if need_train:
            pairs = training_triplets(TR, E)
            report.update({"train_triplets": pairs.height, "train_seconds": train(pairs)})
            del pairs
        if need["train"]:
            load_models()
            D = dense_split("train", TR)
            # recall on S1 records that did not train the model (eid % 10 != 0)
            held = D.filter(pl.col("s1_eid") % S.DENSE_TRAIN_MOD != 0)
            Eh = E.filter(pl.col("s1") % S.DENSE_TRAIN_MOD != 0).rename({"s1": "s1_eid"})
            for k in (5, 10, 20):
                hit = Eh.join(held.filter(pl.col("rank") <= k).select("s1_eid", "t_eid", "src"), on=["s1_eid", "t_eid", "src"]).height
                report[f"heldout_pair_recall@{k}_per_source"] = round(hit / Eh.height, 4)
            log(report)
            del D, held, Eh
        del TR, E
        gc.collect()
    if need["test"]:
        if not MODELS:
            load_models()
        report["test_rows"] = dense_split("test", records("test")).height
    rep_path.write_text(json.dumps(report, indent=1))
    log(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
