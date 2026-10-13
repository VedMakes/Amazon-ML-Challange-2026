#!/usr/bin/env python
"""Step 2 (GPU): cross-encoder scores for the candidate pairs. The matcher uses the scores as features.

A cross-encoder reads the two records together ("S1 name | address [SEP] target name | address") and gives one
match score. It can see the one small change (house number, legal form, first word) that makes a near-copy
a different business.

Stages : train  fine-tune the bi-encoder weights of step 1 with a new classification head (1 epoch).
                Training records: S1 with eid % 10 == 0 (not used by the matcher) -> all true links + the 10 hardest
                wrong candidates of each S1.
         dev    scores for the development samples of the matcher (the same splits and seeds) -> ce_dev.parquet
         test   scores for the test chunks (the same chunks as step 3) -> ce_test_<chunk>.parquet
The cross-encoder scores the best 20 dense candidates of each S1 (settings.CE_TOPK).

  python src/step2_cross_encoder.py                      # all stages; finished parts are not done again
  python src/step2_cross_encoder.py --stage test --shard 0/2
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


def _options(ap):
    ap.add_argument("--stage", choices=["all", "train", "dev", "test"], default="all")
    ap.add_argument("--shard", default="0/1", help="i/n: this run scores every n-th test chunk, starting at chunk i")
    ap.add_argument("--force", action="store_true", help="do the selected stages again even if their outputs exist")


ARGS = cli.parse(__doc__, _options)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from ber import settings as S  # noqa: E402
from ber.dense import load_dense  # noqa: E402
from ber.paths import CE_DIR, DATA, DENSE_DIR, check_data  # noqa: E402
from ber.universe import DevSplits, Universe  # noqa: E402

N_NEG = 10                    # hardest wrong dense candidates for each training S1
MAX_LEN = 128                 # tokens for each pair
BATCH = 256                   # total training batch (all GPUs together)
EPOCHS, LR, SMOOTH, HOLDOUT = 1, 5e-5, 0.05, 0.02
SCORE_BATCH = 1024
CUDA = torch.cuda.is_available()
DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
# bf16 on Ampere and newer GPUs (no loss scaling), fp16 on older GPUs (T4), fp32 on CPU
AMP_DTYPE = torch.bfloat16 if CUDA and torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
DENSE_TRAIN, DENSE_TEST = DENSE_DIR / "dense_train.parquet", DENSE_DIR / "dense_test.parquet"
MODEL_DIR = CE_DIR / "ce_model"
TEXTS: dict = {}
T_START = time.time()
tok = None


def log(*a):
    print(*a, flush=True)


def read(split, name):
    return pl.read_csv(DATA / split / f"{split}_{name}.tsv", separator="\t", quote_char=None, infer_schema_length=0)


def load_texts(split):
    """eid -> "name | address" for each source. A space follows each digit (12b -> "1 2 b"), so the tokenizer
    gives one token to each digit and a changed house number is visible."""
    if TEXTS.get("split") == split:
        return
    TEXTS.clear()
    gc.collect()
    for s in (1, 2, 3):
        t = pl.col("business_name").fill_null("") + " | " + pl.col("business_address").fill_null("")
        d = read(split, f"source{s}").select(pl.col("entity_id").str.slice(3).cast(pl.Int64).alias("eid"), t.alias("text"))
        TEXTS[s] = d.with_columns(pl.col("text").str.replace_all(r"(\d)", r"$1 ").str.replace_all(r"\s+", " ").str.strip_chars())
    TEXTS["split"] = split


def attach_texts(P):
    """P: s1_eid, t_eid, src -> + a (S1 text), b (target text). The row order does not change."""
    P = P.with_row_index("_i")
    a = P.select("_i", "s1_eid").join(TEXTS[1].rename({"eid": "s1_eid", "text": "a"}), on="s1_eid", how="left")
    b = pl.concat([P.filter(pl.col("src") == s).select("_i", "t_eid").join(TEXTS[s].rename({"eid": "t_eid", "text": "b"}), on="t_eid", how="left")
                   for s in (2, 3)])
    return (P.join(a.select("_i", "a"), on="_i", how="left").join(b.select("_i", "b"), on="_i", how="left").sort("_i").drop("_i")
            .with_columns(pl.col("a").fill_null(""), pl.col("b").fill_null("")))


def gt_links():
    gt = read("train", "ground_truth")
    return (gt.select(pl.col("source1_entity_id").str.slice(3).cast(pl.Int64).alias("s1_eid"), pl.col("matched_entity_ids").str.split(",").alias("t"))
            .explode("t").filter(pl.col("t").is_not_null() & (pl.col("t") != ""))
            .select("s1_eid", pl.col("t").str.slice(3).cast(pl.Int64).alias("t_eid"), pl.col("t").str.slice(1, 1).cast(pl.Int8).alias("src")))


def dense_top(U, path, q_idx):
    """The candidates of the matcher (dense top-K for each source) -> for each S1 the best CE_TOPK by dense score."""
    D = load_dense(path, U, q_idx, k=S.K)
    q = D["q"].to_numpy().astype(np.int64)
    d = D["d"].to_numpy().astype(np.int64)
    P = pl.DataFrame({"s1_eid": U.eid1[q].astype(np.int64), "t_eid": U.eidT[d].astype(np.int64), "src": np.where(d < U.n2, 2, 3).astype(np.int8),
                      "score": D["score"].to_numpy()})
    return P.sort(["s1_eid", "score"], descending=[False, True]).group_by("s1_eid", maintain_order=True).head(S.CE_TOPK).drop("score")


def collate(batch):
    a, b, y = zip(*batch)
    return tok(list(a), list(b), truncation="longest_first", max_length=MAX_LEN, padding=True, return_tensors="pt"), torch.tensor(y, dtype=torch.float32)


@torch.inference_mode()
def _score_on(dev, model, a, b, out, lo):
    order = np.argsort([len(x) + len(y) for x, y in zip(a, b)])          # batches of equal length: little padding
    for o in range(0, len(order), SCORE_BATCH):
        idx = order[o:o + SCORE_BATCH]
        enc = tok([a[i] for i in idx], [b[i] for i in idx], truncation="longest_first", max_length=MAX_LEN, padding=True, return_tensors="pt").to(dev)
        with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=dev != "cpu"):
            out[lo + idx] = model(**enc).logits.float().squeeze(-1).cpu().numpy()


def score_pairs(a, b, models):
    out = np.empty(len(a), np.float32)
    bounds = np.linspace(0, len(a), len(models) + 1).astype(int)
    with ThreadPoolExecutor(len(models)) as ex:
        list(ex.map(lambda i: _score_on(DEVICES[i], models[i], a[bounds[i]:bounds[i + 1]], b[bounds[i]:bounds[i + 1]], out, bounds[i]), range(len(models))))
    return out


def auc(y, s):
    o = np.argsort(s)
    r = np.empty(len(s))
    r[o] = np.arange(1, len(s) + 1)
    npos = y.sum()
    return float((r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * (len(y) - npos)))


# --------------------------------------------------------------------------------------- stage: train
def stage_train():
    global tok
    t0 = time.time()
    init = DENSE_DIR / "model"
    if not (init / "config.json").exists():
        raise FileNotFoundError(f"{init} not found: run step 1 first")
    load_texts("train")
    E = gt_links().filter(pl.col("s1_eid") % S.DENSE_TRAIN_MOD == 0)
    s1_train = TEXTS[1].filter(pl.col("eid") % S.DENSE_TRAIN_MOD == 0)["eid"]
    neg = (pl.scan_parquet(DENSE_TRAIN).filter(pl.col("s1_eid") % S.DENSE_TRAIN_MOD == 0)
           .select(pl.col("s1_eid").cast(pl.Int64), pl.col("t_eid").cast(pl.Int64), pl.col("src").cast(pl.Int8), "score").collect()
           .join(E, on=["s1_eid", "t_eid", "src"], how="anti")
           .sort(["s1_eid", "score"], descending=[False, True]).group_by("s1_eid", maintain_order=True).head(N_NEG))
    pairs = pl.concat([E.select("s1_eid", "t_eid", "src").with_columns(pl.lit(1, dtype=pl.Int8).alias("y")),
                       neg.select("s1_eid", "t_eid", "src", pl.lit(0, dtype=pl.Int8).alias("y"))])
    hold = np.random.default_rng(S.SEED).choice(s1_train.to_numpy(), int(HOLDOUT * len(s1_train)), replace=False)
    pairs = attach_texts(pairs.with_columns(pl.col("s1_eid").is_in(pl.Series(hold).implode()).alias("hold")))
    TR = pairs.filter(~pl.col("hold")).sample(fraction=1.0, shuffle=True, seed=S.SEED)
    HO = pairs.filter(pl.col("hold"))
    log(f"{len(s1_train):,} training S1 | pairs: train {TR.height:,} ({TR['y'].mean() * 100:.1f} % positive), holdout {HO.height:,}")
    del pairs, neg, E
    gc.collect()

    tok = AutoTokenizer.from_pretrained(init)
    model = AutoModelForSequenceClassification.from_pretrained(init, num_labels=1, ignore_mismatched_sizes=True).to(DEVICES[0])
    ds = list(zip(TR["a"].to_list(), TR["b"].to_list(), TR["y"].to_list()))
    n_gpu = torch.cuda.device_count()
    lens = np.fromiter((len(a) + len(b) for a, b, _ in ds), dtype=np.int32, count=len(ds))

    class LengthGrouped(torch.utils.data.Sampler):
        """Shuffle, sort by length in groups of 50 batches, shuffle the batches: little padding, random order."""
        def __init__(self):
            self.seed = S.SEED

        def __len__(self):
            return len(ds) // BATCH

        def __iter__(self):
            rng = np.random.default_rng(self.seed)
            self.seed += 1
            idx = rng.permutation(len(ds))
            mb = 50 * BATCH
            batches = []
            for o in range(0, len(idx) - BATCH + 1, mb):
                chunk = idx[o:o + mb]
                chunk = chunk[np.argsort(lens[chunk], kind="stable")]
                batches += [chunk[i:i + BATCH] for i in range(0, len(chunk) - BATCH + 1, BATCH)]
            for i in rng.permutation(len(batches)):
                yield batches[i].tolist()

    loader = torch.utils.data.DataLoader(ds, batch_sampler=LengthGrouped(), collate_fn=collate, num_workers=4, pin_memory=CUDA, persistent_workers=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * len(loader)
    sched = get_linear_schedule_with_warmup(opt, int(0.05 * steps), steps)
    scaler = torch.cuda.amp.GradScaler(enabled=CUDA and AMP_DTYPE == torch.float16)   # bf16 needs no loss scaling
    lossf = nn.BCEWithLogitsLoss()
    net = nn.DataParallel(model) if n_gpu > 1 else model
    log(f"{steps:,} steps, batch {BATCH}, {max(n_gpu, 1)} GPU(s), {str(AMP_DTYPE).split('.')[-1] if CUDA else 'float32 (CPU)'}")
    step, ema = 0, None
    for _ in range(EPOCHS):
        net.train()
        for enc, y in loader:
            enc = {k: v.to(DEVICES[0], non_blocking=True) for k, v in enc.items()}
            y = y.to(DEVICES[0])
            with torch.autocast("cuda", dtype=AMP_DTYPE, enabled=CUDA):
                logits = net(**enc).logits.squeeze(-1)
                loss = lossf(logits.float(), y * (1 - SMOOTH) + SMOOTH / 2)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            ema = loss.item() if ema is None else 0.98 * ema + 0.02 * loss.item()
            if step % 500 == 0:
                log(f"step {step:,}/{steps:,} | loss {ema:.4f} | {(time.time() - t0) / 60:.1f} min")
    model.eval()
    s_ho = score_pairs(HO["a"].to_list(), HO["b"].to_list(), [model])
    y_ho = HO["y"].to_numpy()
    top1 = (HO.with_columns(pl.Series("s", s_ho)).filter(pl.col("y").max().over("s1_eid") == 1)
            .sort("s", descending=True).group_by("s1_eid", maintain_order=True).first())
    model.save_pretrained(MODEL_DIR)
    tok.save_pretrained(MODEL_DIR)
    rep = {"init": str(init), "train_pairs": len(ds), "final_loss": round(ema, 4) if ema is not None else None,
           "train_minutes": round((time.time() - t0) / 60, 1), "holdout_auc": round(auc(y_ho, s_ho), 5),
           "holdout_top1_true_%": round(100 * float(top1["y"].mean()), 2)}
    (CE_DIR / "ce_train_report.json").write_text(json.dumps(rep, indent=1))
    log(rep)
    del net, model, opt, ds, loader, TR, HO
    gc.collect()
    if CUDA:
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------- scoring
def load_scorers():
    global tok
    if not (MODEL_DIR / "config.json").exists():
        raise FileNotFoundError(f"{MODEL_DIR} not found: run the train stage first")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    ms = []
    for dev in DEVICES:
        m = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(dev)
        ms.append((m.to(AMP_DTYPE) if dev != "cpu" else m).eval())
    log("scoring with", MODEL_DIR, "on", DEVICES)
    return ms


def score_frame(P, out_path, scorers):
    t0 = time.time()
    P = attach_texts(P)
    s = score_pairs(P["a"].to_list(), P["b"].to_list(), scorers)
    tmp = out_path.with_suffix(".tmp")
    P.select("s1_eid", "t_eid", "src").with_columns(pl.Series("ce_logit", s)).write_parquet(tmp)
    tmp.rename(out_path)
    r = {"file": out_path.name, "pairs": P.height, "pairs_per_s": round(P.height / max(time.time() - t0, 1e-6)), "minutes": round((time.time() - t0) / 60, 1)}
    log(r)
    return r


def stage_dev(scorers):
    X = DevSplits(seed=S.SEED)
    U = X.U
    Q = {"train": X.sample_queries("train", S.N_TRAIN, seed=11), "val": X.sample_queries("val", S.N_VAL, seed=1),
         "outer": X.sample_queries("outer", S.N_OUTER, seed=3)}
    load_texts("train")
    P_dev = pl.concat([dense_top(U, DENSE_TRAIN, Q[t]) for t in ("train", "val", "outer")]).unique(["s1_eid", "t_eid", "src"])
    stat = score_frame(P_dev, CE_DIR / "ce_dev.parquet", scorers)
    # check on inner validation: the cross-encoder alone
    E = gt_links()
    qv = pl.Series(U.eid1[Q["val"]].astype(np.int64))
    V = (pl.read_parquet(CE_DIR / "ce_dev.parquet").filter(pl.col("s1_eid").is_in(qv.implode()))
         .join(E.with_columns(pl.lit(1).alias("y")), on=["s1_eid", "src", "t_eid"], how="left").with_columns(pl.col("y").fill_null(0)))
    n_true = E.filter(pl.col("s1_eid").is_in(qv.implode())).height
    stat.update({"val_pair_auc": round(auc(V["y"].to_numpy(), V["ce_logit"].to_numpy()), 5), "val_true_links": n_true,
                 "val_true_links_scored": int(V["y"].sum()), "n_train": S.N_TRAIN, "ce_topk": S.CE_TOPK})
    log(f"val: pair AUC {stat['val_pair_auc']} | true links in the top-{S.CE_TOPK}: {stat['val_true_links_scored']:,} of {n_true:,} "
        f"({100 * stat['val_true_links_scored'] / max(n_true, 1):.2f} %)")
    (CE_DIR / "ce_dev_report.json").write_text(json.dumps(stat, indent=1))


def stage_test(scorers, shard):
    TU = Universe("test")
    chunks = []
    for ci, c in enumerate(TU.countries):
        q_all = np.flatnonzero(TU.q_country == ci)
        for i, o in enumerate(range(0, len(q_all), S.TEST_QCHUNK)):
            chunks.append((f"{c}_{i:03d}", q_all[o:o + S.TEST_QCHUNK]))
    mine = [c for i, c in enumerate(chunks) if i % shard[1] == shard[0]]
    log(f"{len(chunks)} test chunks; this run: {len(mine)}")
    load_texts("test")
    for tag, qc in mine:
        outp = CE_DIR / f"ce_test_{tag}.parquet"
        if outp.exists() and not ARGS.force:
            continue
        score_frame(dense_top(TU, DENSE_TEST, qc), outp, scorers)
        log(f"  elapsed {(time.time() - T_START) / 3600:.2f} h")


def main():
    check_data()
    CE_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(S.SEED)
    np.random.seed(S.SEED)
    stage = ARGS.stage
    shard = tuple(int(x) for x in ARGS.shard.split("/"))
    log(json.dumps({"data": str(DATA), "dense": str(DENSE_DIR), "output": str(CE_DIR), "devices": DEVICES, "stage": stage, "shard": ARGS.shard}, indent=1))
    for p in (DENSE_TRAIN, DENSE_TEST):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found: run step 1 first")
    if stage in ("all", "train") and (ARGS.force or not (MODEL_DIR / "config.json").exists()):
        stage_train()
    if stage in ("all", "dev", "test"):
        scorers = load_scorers()
        if stage in ("all", "dev") and (ARGS.force or not (CE_DIR / "ce_dev.parquet").exists()):
            stage_dev(scorers)
        if stage in ("all", "test"):
            stage_test(scorers, shard)
    log(f"done in {(time.time() - T_START) / 3600:.2f} h")


if __name__ == "__main__":
    main()
