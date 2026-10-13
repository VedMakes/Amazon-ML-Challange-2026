#!/usr/bin/env python
"""Step 3 (CPU): features + LightGBM matcher -> output/matching_results.tsv and output/candidate_pairs.tsv.

Inputs : the challenge data, <work>/dense (step 1), <work>/ce (step 2).
Stages : dev      development sets (train / val / outer), matcher fit, outer-holdout check -> matcher.pkl
         test     scores the test chunks (37 chunks of 50,000 S1 records) -> scored_test_<chunk>.parquet
         finalize one S1 for each target, the two TSV files, the official validator
Each stage keeps its results on disk. If a run stops, start the same command again: finished work is not repeated.

Examples
  python src/step3_matcher.py                       # all stages
  python src/step3_matcher.py --stage test --shard 0/4   # one of four parallel test sessions
"""
from __future__ import annotations

import gc
import json
import pickle
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber import cli  # noqa: E402


def _options(ap):
    ap.add_argument("--stage", choices=["all", "dev", "test", "finalize"], default="all")
    ap.add_argument("--shard", default="0/1", help="i/n: this run scores every n-th test chunk, starting at chunk i (default 0/1 = all)")
    ap.add_argument("--jobs", type=int, help="parallel feature jobs / test chunks (default: from CPU count and RAM)")
    ap.add_argument("--no-ce", action="store_true", help="train and score without the cross-encoder features (step 2 not necessary)")
    ap.add_argument("--validator", help="path of validate_submission.py (default: utils/ next to the dataset folder)")


ARGS = cli.parse(__doc__, _options)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import psutil  # noqa: E402

from ber import jobs, matcher as MT, metrics as M, settings as S, store  # noqa: E402
from ber.dense import dense_reverse, load_dense  # noqa: E402
from ber.graph import GRAPH_FEATS  # noqa: E402
from ber.paths import CACHE, CE_DIR, DATA, DENSE_DIR, MATCHER_DIR, OUT, THREADS, check_data  # noqa: E402
from ber.pipeline import match_chunk  # noqa: E402
from ber.runcache import RUNS, cached_run  # noqa: E402
from ber.universe import DevSplits, Universe, attach_reverse, rss_mb  # noqa: E402

T_START = time.time()
KEEP_MARGIN = 0.0005          # a later stage is kept only if it raises the validation macro F0.5 by more than this
RAM_GB = psutil.virtual_memory().total / 2**30
N_PAR = ARGS.jobs or max(1, min(12, THREADS // 6, int(RAM_GB // 10)))
DENSE_TRAIN, DENSE_TEST = DENSE_DIR / "dense_train.parquet", DENSE_DIR / "dense_test.parquet"
CE_DEV = CE_DIR / "ce_dev.parquet"


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------------------- shared helpers
def candidates(U, path, q_idx):
    """Dense top-K for each target source -> q, d + dense score / rank (these become matcher features)."""
    return load_dense(path, U, q_idx, k=S.K).select("q", "d", pl.col("score").alias("score_E5-dense"),
                                                    pl.col("rank").cast(pl.Float32).alias("rank_E5-dense")).sort("q")


def with_ce(U, C, ce_path):
    """Cross-encoder features: probability and rank in the S1 (null for candidates that the cross-encoder did not score)."""
    q = C["q"].to_numpy().astype(np.int64)
    d = C["d"].to_numpy().astype(np.int64)
    keys = pl.DataFrame({"s1_eid": U.eid1[q].astype(np.int64), "t_eid": U.eidT[d].astype(np.int64), "src": np.where(d < U.n2, 2, 3).astype(np.int8)})
    ce = (pl.scan_parquet(ce_path).filter(pl.col("s1_eid").is_in(pl.Series(np.unique(keys["s1_eid"].to_numpy())).implode()))
          .select(pl.col("s1_eid").cast(pl.Int64), pl.col("t_eid").cast(pl.Int64), pl.col("src").cast(pl.Int8), "ce_logit").collect())
    lg = keys.join(ce, on=["s1_eid", "t_eid", "src"], how="left", maintain_order="left")["ce_logit"]
    return C.with_columns((1 / (1 + (-lg).exp())).cast(pl.Float32).alias("score_CE")).with_columns(
        pl.col("score_CE").rank("ordinal", descending=True).over("q").cast(pl.Float32).alias("rank_CE"))


def reverse_table(split, U, path):
    """For each target: the S1 records that list it, in cosine order (top R). Built one time for each split."""
    rid = f"dense-reverse-{split}"
    cached_run(rid, lambda: dense_reverse(path, U, R=S.R, k=S.K))
    return RUNS / f"{rid}.parquet"


def featurise(tag, split, cands):
    """Pair features in child processes: batches of <= MAX_PAIRS pairs that do not split an S1, N_PAR at a time."""
    def go():
        c = cands.sort("q")
        qs = c["q"].to_numpy()
        spans, lo = [], 0
        while lo < len(qs):
            hi = min(lo + S.MAX_PAIRS, len(qs))
            while hi < len(qs) and qs[hi] == qs[hi - 1]:
                hi += 1
            spans.append((lo, hi))
            lo = hi

        def one(span):
            f, _ = jobs.spawn(f"features-{tag}-{span[0]}", {"kind": "features", "split": split}, {"pairs": c.slice(span[0], span[1] - span[0])})
            log(f"  features {tag}: pairs {span[0]:,}-{span[1]:,} of {len(qs):,} | {(time.time() - T_START) / 60:.0f} min")
            return f
        with ThreadPoolExecutor(N_PAR) as ex:
            out = list(ex.map(one, spans))
        return pl.concat(out, how="diagonal")
    return cached_run(f"features-{tag}", go)[0]


def decide_with(scored, rule):
    return MT.decide_threshold(scored, rule["t"]) if rule["rule"] == "threshold" else MT.decide_expected_f(scored, rule["alpha"], rule["t_min"])


def tune_rule(oof_scored, T, QF, allow_expected_f):
    t, curve = MT.tune_threshold(oof_scored, T, QF)
    best = ({"rule": "threshold", "t": t}, float(curve.filter(pl.col("t") == t)["macro_f05"][0]))
    if allow_expected_f:
        ef, grid = MT.tune_expected_f(oof_scored, T, QF)
        if grid["macro_f05"][0] > best[1]:
            best = ({"rule": "expected_f", **ef}, float(grid["macro_f05"][0]))
    return best


def graph_job(tag, scored, t_core):
    return cached_run(f"graph-{tag}", lambda: jobs.spawn(f"graph-{tag}", {"kind": "graph", "split": "train", "t_core": float(t_core)}, {"scored": scored}))[0]


def test_chunks(TU):
    out = []
    for ci, c in enumerate(TU.countries):
        q_all = np.flatnonzero(TU.q_country == ci)
        for i, o in enumerate(range(0, len(q_all), S.TEST_QCHUNK)):
            out.append((f"{c}_{i:03d}", q_all[o:o + S.TEST_QCHUNK]))
    return out


def write_report(name, report):
    (MATCHER_DIR / f"report_{name}.json").write_text(json.dumps(report, indent=1, default=str))


# --------------------------------------------------------------------------------------- stage: dev
def stage_dev(use_ce: bool) -> None:
    report = {}
    X = DevSplits(seed=S.SEED)
    U = X.U
    log("leakage audit:", X.audit)
    Q = {"train": X.sample_queries("train", S.N_TRAIN, seed=11), "val": X.sample_queries("val", S.N_VAL, seed=1),
         "outer": X.sample_queries("outer", S.N_OUTER, seed=3)}
    rev_all = pl.read_parquet(reverse_table("train", U, DENSE_TRAIN))
    FS, Y, T, QF = {}, {}, {}, {}
    for t in ("val", "outer", "train"):
        t0 = time.time()
        C = candidates(U, DENSE_TRAIN, Q[t])
        if use_ce:
            C = with_ce(U, C, CE_DEV)
        C = attach_reverse(C, rev_all.filter(pl.col("d").is_in(pl.Series(np.unique(C["d"].to_numpy())).implode())), S.R)
        FS[t] = featurise(t, "train", C)
        del C
        Y[t] = FS[t].select("q", "d").join(X.edges.select("q", "d").with_columns(pl.lit(1).alias("y")), on=["q", "d"],
                                           how="left", maintain_order="left")["y"].fill_null(0).to_numpy()
        T[t], QF[t] = X.truth(Q[t]), pl.DataFrame({"q": Q[t].astype(np.uint32)})
        cover = int(Y[t].sum()) / max(T[t].height, 1)
        log(f"{t}: {len(Q[t]):,} S1 | {FS[t].height:,} pairs ({FS[t].height / len(Q[t]):.1f}/S1) | pair recall {cover:.4f} | "
            f"{(time.time() - t0) / 60:.1f} min | RSS {rss_mb()} MB")
        report[f"set_{t}"] = {"S1": len(Q[t]), "pairs": FS[t].height, "pair_recall": round(cover, 5)}
    del rev_all
    FEATS = [c for c in FS["train"].columns if c not in ("q", "d")]
    for t in ("val", "outer"):
        for c in FEATS:
            if c not in FS[t].columns:
                FS[t] = FS[t].with_columns(pl.lit(None, dtype=pl.Float32).alias(c))
    ce_feats = [f for f in FEATS if f.endswith("_CE")]
    log(f"{len(FEATS)} features | cross-encoder features: {ce_feats or 'none (--no-ce)'}")
    gc.collect()

    steps = []

    def val_score(p, rule, gate=None):
        sc = FS["val"].select("q", "d").with_columns(pl.Series("p", p))
        acc = decide_with(sc, rule)
        if gate is not None:
            acc = MT.apply_gate(acc, gate[0].predict(sc, FS["val"]), gate[1])
        f, per = M.macro_f05(acc, T["val"], QF["val"], return_per_entity=True)
        return f, float(per["P"].mean()), float(per["R"].mean())

    def add_step(name, rule, f_oof, f_val, p_, r_, kept, t0):
        steps.append({"step": name, "rule": json.dumps(rule), "OOF_F05": round(f_oof, 5), "val_F05": round(f_val, 5),
                      "val_P": round(p_, 4), "val_R": round(r_, 4), "kept": bool(kept), "minutes": round((time.time() - t0) / 60, 1)})
        log(steps[-1])

    # stage 1 + stage 2 (context in the S1) + isotonic calibration
    t0 = time.time()
    model = MT.TwoStage(FEATS, calibrate=True).fit_oof(FS["train"], Y["train"])
    oof, pval = model.oof_cal, model.predict(FS["val"])
    log("early-stopping rounds for each fold (cap 1200) | stage 1:", [m.best_iteration for m in model.m1], "| stage 2:", [m.best_iteration for m in model.m2])
    rule, f_oof = tune_rule(FS["train"].select("q", "d").with_columns(pl.Series("p", oof)), T["train"], QF["train"], allow_expected_f=True)
    f_base, p_, r_ = val_score(pval, rule)
    add_step("two-stage LightGBM + isotonic", rule, f_oof, f_base, p_, r_, True, t0)

    # stage 3: sibling graph
    t0 = time.time()
    t_core = float(rule["t"]) if rule["rule"] == "threshold" else 0.5
    G_tr = graph_job("train", FS["train"].select("q", "d").with_columns(pl.Series("p", oof)), t_core)
    G_va = graph_job("val", FS["val"].select("q", "d").with_columns(pl.Series("p", pval)), t_core)
    X3_tr = pl.concat([FS["train"].select(FEATS), G_tr.select(["p"] + GRAPH_FEATS)], how="horizontal")
    G3, oof3 = MT.train_lgb(X3_tr, Y["train"], FS["train"]["q"].to_numpy())
    del X3_tr
    rule3, f3_oof = tune_rule(FS["train"].select("q", "d").with_columns(pl.Series("p", oof3)), T["train"], QF["train"], allow_expected_f=False)
    p3 = MT.predict(G3, pl.concat([FS["val"].select(FEATS), G_va.select(["p"] + GRAPH_FEATS)], how="horizontal"))
    f3, p_, r_ = val_score(p3, rule3)
    use_graph = f3 > f_base + KEEP_MARGIN
    add_step("+ sibling graph stage 3", rule3, f3_oof, f3, p_, r_, use_graph, t0)
    oof_c, pval_c, rule_c, f_c = (oof3, p3, rule3, f3) if use_graph else (oof, pval, rule, f_base)

    # emptiness gate: P(the S1 has no true candidate)
    t0 = time.time()
    imp = pl.DataFrame({"feature": model.feats2, "gain": np.mean([m.feature_importance("gain") for m in model.m2], axis=0)})
    gate_feats = [f for f in imp.sort("gain", descending=True)["feature"].to_list() if f in FEATS][:12]
    tr_sc = FS["train"].select("q", "d").with_columns(pl.Series("p", oof_c))
    gate = MT.EmptyGate(gate_feats).fit_oof(tr_sc, Y["train"], FS["train"])
    t_gate, gcurve = MT.tune_gate(decide_with(tr_sc, rule_c), gate.oof, T["train"], QF["train"])
    fg, p_, r_ = val_score(pval_c, rule_c, (gate, t_gate))
    use_gate = t_gate <= 1 and fg > f_c + KEEP_MARGIN
    add_step("+ emptiness gate", {**rule_c, "t0": t_gate}, float(gcurve["macro_f05"][0]), fg, p_, r_, use_gate, t0)
    print(pl.DataFrame(steps))
    imp = imp.with_columns((100 * pl.col("gain") / pl.col("gain").sum()).round(2).alias("gain_%")).sort("gain", descending=True)
    print(imp.head(25))
    decision = {**rule_c, "gate": gate if use_gate else None, "t0": t_gate}
    graph = {"model": G3, "t_core": t_core} if use_graph else None
    report["steps"] = steps
    report["feature_importance_top25"] = imp.head(25).to_dicts()

    # outer holdout: frozen model and rule (not used for a decision before this point)
    sc = FS["outer"].select("q", "d").with_columns(pl.Series("p", model.predict(FS["outer"])))
    if use_graph:
        G_o = graph_job("outer", sc, t_core)
        sc = sc.with_columns(pl.Series("p", MT.predict(G3, pl.concat([FS["outer"].select(FEATS), G_o.select(["p"] + GRAPH_FEATS)], how="horizontal"))))
    acc = decide_with(sc, decision)
    if use_gate:
        acc = MT.apply_gate(acc, gate.predict(sc, FS["outer"]), t_gate)
    f_out, per = M.macro_f05(acc, T["outer"], QF["outer"], return_per_entity=True)
    lo, hi = M.bootstrap_ci(per["f05"].to_numpy())
    qo = per.select("q", "f05").with_columns(pl.Series("country", np.asarray(U.countries, dtype=object)[U.q_country[per["q"].to_numpy()]], dtype=pl.String))
    by_country = {r["country"]: round(r["F05"], 4) for r in qo.group_by("country").agg(pl.col("f05").mean().alias("F05")).sort("country").to_dicts()}
    report["outer"] = {"macro_F05": round(f_out, 5), "CI95": [round(lo, 5), round(hi, 5)], "P": round(float(per["P"].mean()), 5),
                       "R": round(float(per["R"].mean()), 5), "by_country": by_country, "pair_recall": report["set_outer"]["pair_recall"]}
    log("OUTER HOLDOUT:", report["outer"])
    pickle.dump({"model": model, "features": FEATS, "decision": decision, "graph": graph}, open(MATCHER_DIR / "matcher.pkl", "wb"))
    report["minutes"] = round((time.time() - T_START) / 60, 1)
    write_report("dev", report)
    log("saved", MATCHER_DIR / "matcher.pkl")


# --------------------------------------------------------------------------------------- stage: test
def stage_test(shard: tuple[int, int]) -> None:
    match = pickle.load(open(MATCHER_DIR / "matcher.pkl", "rb"))
    use_ce = "score_CE" in match["features"]
    TU = Universe("test")
    chunks = test_chunks(TU)
    mine = [c for i, c in enumerate(chunks) if i % shard[1] == shard[0]]
    log(f"{len(chunks)} test chunks; this run: {len(mine)} | rule { {k: v for k, v in match['decision'].items() if k != 'gate'} } "
        f"| graph stage {match['graph'] is not None} | gate {match['decision'].get('gate') is not None} | cross-encoder {use_ce}")
    rev = reverse_table("test", TU, DENSE_TEST)
    stats = []

    def score_chunk(item):
        tag, qc = item
        outp = MATCHER_DIR / f"scored_test_{tag}.parquet"
        if outp.exists():
            return
        t0 = time.time()
        C = candidates(TU, DENSE_TEST, qc)
        if use_ce:
            ce_p = CE_DIR / f"ce_test_{tag}.parquet"
            if not ce_p.exists():
                raise FileNotFoundError(f"{ce_p} not found: run step 2 (or copy its results) before this stage")
            C = with_ce(TU, C, ce_p)
        Sc = match_chunk(tag, C, match["model"], match["features"], match["decision"], max_pairs=S.MAX_PAIRS,
                         reverse={"parts": [rev], "R": S.R}, graph=match["graph"])
        q = Sc["q"].to_numpy().astype(np.int64)
        d = Sc["d"].to_numpy().astype(np.int64)
        tmp = outp.with_suffix(".tmp")
        pl.DataFrame({"s1_eid": TU.eid1[q], "t_eid": TU.eidT[d], "src": np.where(d < TU.n2, 2, 3).astype(np.int8),
                      "p": Sc["p"].to_numpy().astype(np.float32), "accepted": Sc["accepted"].to_numpy().astype(bool)}).write_parquet(tmp)
        tmp.rename(outp)
        stats.append({"chunk": tag, "S1": len(qc), "pairs": C.height, "accepted": int(Sc["accepted"].sum()), "minutes": round((time.time() - t0) / 60, 1)})
        log(stats[-1], f"| {len(stats)}/{len(mine)} | elapsed {(time.time() - T_START) / 3600:.2f} h")

    with ThreadPoolExecutor(N_PAR) as ex:
        list(ex.map(score_chunk, mine))
    write_report(f"test_{shard[0]}of{shard[1]}", {"chunks": stats})


# --------------------------------------------------------------------------------------- stage: finalize
def stage_finalize(validator: str | None) -> None:
    TU = Universe("test")
    expected = [MATCHER_DIR / f"scored_test_{tag}.parquet" for tag, _ in test_chunks(TU)]
    missing = [p.name for p in expected if not p.exists()]
    log(f"scored chunks {len(expected) - len(missing)}/{len(expected)}")
    if missing:
        raise FileNotFoundError(f"test chunks without scores (run the test stage / copy the files of all shards): {missing[:10]}")
    Sc = pl.concat([pl.read_parquet(p).with_columns(pl.col("s1_eid").cast(pl.Int64), pl.col("t_eid").cast(pl.Int64), pl.col("src").cast(pl.Int8),
                                                    pl.col("p").cast(pl.Float32), pl.col("accepted").cast(pl.Boolean)) for p in expected])
    # one S1 for each target (the labels never link a target to two S1 records): the S1 with the highest probability keeps it
    win = (Sc.filter(pl.col("accepted")).sort("p", descending=True).unique(["t_eid", "src"], keep="first")
           .select("s1_eid", "t_eid", "src").with_columns(pl.lit(True).alias("win")))
    n_conf = int(Sc["accepted"].sum()) - win.height
    Sc = (Sc.join(win, on=["s1_eid", "t_eid", "src"], how="left").with_columns((pl.col("accepted") & pl.col("win").fill_null(False)).alias("accepted"))
          .drop("win").with_columns((pl.when(pl.col("src") == 2).then(pl.lit("S2-")).otherwise(pl.lit("S3-")) + pl.col("t_eid").cast(pl.String)).alias("tid"))
          .sort(["s1_eid", "p"], descending=[False, True]))
    cl = Sc.group_by("s1_eid", maintain_order=True).agg(pl.col("tid").str.join(",").alias("c"))
    ml = Sc.filter(pl.col("accepted")).group_by("s1_eid", maintain_order=True).agg(pl.col("tid").str.join(",").alias("m"))
    rows = (pl.DataFrame({"s1_eid": TU.eid1.astype(np.int64)}).join(cl, on="s1_eid", how="left", maintain_order="left")
            .join(ml, on="s1_eid", how="left", maintain_order="left").with_columns(pl.col("c", "m").fill_null(""))
            .with_columns(("S1-" + pl.col("s1_eid").cast(pl.String)).alias("sid")))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "candidate_pairs.tsv").write_text("\n".join(["source1_entity_id\tcandidate_entity_ids"] + (rows["sid"] + "\t" + rows["c"]).to_list()) + "\n")
    (OUT / "matching_results.tsv").write_text("\n".join(["source1_entity_id\tmatched_entity_ids"] + (rows["sid"] + "\t" + rows["m"]).to_list()) + "\n")
    fin = {"S1_rows": rows.height, "candidate_pairs": Sc.height, "mean_candidates_per_S1": round(Sc.height / rows.height, 2),
           "matched_links": int(Sc["accepted"].sum()), "mean_matches_per_S1": round(int(Sc["accepted"].sum()) / rows.height, 3),
           "S1_predicted_no_match_%": round(100 * float((rows["m"] == "").mean()), 2), "targets_claimed_by_more_than_one_S1": n_conf}
    log(fin)
    log("written:", OUT / "matching_results.tsv", "|", OUT / "candidate_pairs.tsv")
    cands = [Path(validator)] if validator else [DATA.parent / "utils" / "validate_submission.py", DATA / "utils" / "validate_submission.py"]
    val = next((c for c in cands if c.exists()), None)
    if val is None:
        log("official validator not found (use --validator <path of validate_submission.py>)")
    else:
        r = subprocess.run([sys.executable, str(val), "--matching", str(OUT / "matching_results.tsv"),
                            "--candidate", str(OUT / "candidate_pairs.tsv"), "--test-dir", str(DATA / "test")], capture_output=True, text=True)
        log((r.stdout + r.stderr).strip()[-3000:], f"\n(validator exit code {r.returncode})")
        fin["validator_exit_code"] = r.returncode
    write_report("finalize", fin)


# --------------------------------------------------------------------------------------- main
def main() -> None:
    check_data()
    MATCHER_DIR.mkdir(parents=True, exist_ok=True)
    stage = ARGS.stage
    use_ce = not ARGS.no_ce
    shard = tuple(int(x) for x in ARGS.shard.split("/"))
    free_gb = shutil.disk_usage(CACHE if CACHE.exists() else CACHE.parent.parent).free / 2**30 if CACHE.parent.parent.exists() else 0
    log(json.dumps({"data": str(DATA), "dense": str(DENSE_DIR), "cross_encoder": str(CE_DIR) if use_ce else "not used", "work": str(MATCHER_DIR),
                    "output": str(OUT), "threads": THREADS, "RAM_GB": round(RAM_GB), "parallel_jobs": N_PAR, "free_disk_GB": round(free_gb),
                    "stage": stage, "shard": ARGS.shard}, indent=1))
    for p in (DENSE_TRAIN, DENSE_TEST) + ((CE_DEV,) if use_ce and stage in ("all", "dev") else ()):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found: run {'step 1' if 'dense' in p.name else 'step 2'} first (or use --no-ce)")
    if stage != "finalize":
        t0 = time.time()
        cached_run("feature-store-manifest", lambda: pl.DataFrame(store.build_all(log=log)))
        log(f"feature store ready in {time.time() - t0:.0f} s")
    if stage in ("all", "dev"):
        stage_dev(use_ce)
        gc.collect()
    if stage in ("all", "test"):
        stage_test(shard)
    if stage in ("all", "finalize"):
        stage_finalize(ARGS.validator)
    log(f"done in {(time.time() - T_START) / 3600:.2f} h")


if __name__ == "__main__":
    main()
