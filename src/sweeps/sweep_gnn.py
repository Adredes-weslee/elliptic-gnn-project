# src/sweeps/sweep_gnn.py
import argparse
import copy
import csv
import itertools
import json
import os
import shutil
import time
from datetime import datetime

import yaml

from src.train_gnn import main as train_main


# --------------- helpers ---------------
def product_dict(**kwargs):
    keys = list(kwargs.keys())
    vals = list(kwargs.values())
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


def slug(v):
    if isinstance(v, bool):
        return "t" if v else "f"
    if isinstance(v, float):
        s = f"{v:.10g}"
        return s.replace(".", "p").replace("-", "m")
    return str(v)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def read_json(p, default=None):
    try:
        with open(p, "r") as f:
            return json.load(f)
    except Exception:
        return default


def read_metrics(run_name):
    outdir = os.path.join("outputs", "gnn", run_name)
    rec = {"run_name": run_name, "outdir": outdir}
    mpath = os.path.join(outdir, "metrics.json")
    cpath = os.path.join(outdir, "config_used.yaml")
    if os.path.exists(mpath):
        rec.update(read_json(mpath, {}))
    if os.path.exists(cpath):
        with open(cpath, "r") as f:
            cfg_used = yaml.safe_load(f)
        for k in [
            "arch",
            "hidden_dim",
            "layers",
            "dropout",
            "lr",
            "weight_decay",
            "train_window_k",
            "time_embed_dim",
            "time_embed_type",
            "time_embed_l2",
            "time_loss_weighting",
            "symmetrize_edges",
            "patience",
        ]:
            if k in cfg_used:
                rec[k] = cfg_used[k]
    return rec


def extract_per_timestep(rec):
    for k in [
        "test_pr_auc_by_time",
        "pr_auc_test_by_time",
        "pr_auc_illicit_by_time",
        "per_timestep_pr_auc_test",
        "per_timestep_pr_auc",
    ]:
        if k in rec:
            v = rec[k]
            if isinstance(v, list):
                return v
            if isinstance(v, dict):
                items = list(v.items())
                try:
                    items = sorted(
                        ((int(t), s) for t, s in v.items()), key=lambda x: x[0]
                    )
                except Exception:
                    pass
                return [s for _, s in items]
    return None


def add_recent_summaries(rec, per_ts):
    if not per_ts:
        return rec
    n = len(per_ts)
    if n >= 1:
        rec["pr_auc_last1"] = float(per_ts[-1])
    if n >= 3:
        rec["pr_auc_last3"] = float(sum(per_ts[-3:]) / 3.0)
        rec["pr_auc_first3"] = float(sum(per_ts[:3]) / 3.0)
    if n >= 5:
        rec["pr_auc_last5"] = float(sum(per_ts[-5:]) / 5.0)
    return rec


def point_best_to(outdir):
    tgt = os.path.abspath(outdir)
    best_dir = os.path.join("outputs", "gnn", "best")
    if os.path.islink(best_dir):
        try:
            os.unlink(best_dir)
        except Exception:
            pass
    elif os.path.exists(best_dir):
        try:
            shutil.rmtree(best_dir)
        except Exception:
            pass
    try:
        os.symlink(tgt, best_dir, target_is_directory=True)
        return "symlink"
    except Exception:
        ensure_dir(best_dir)
        with open(os.path.join(best_dir, "POINTER.txt"), "w") as f:
            f.write(f"Best run:\n{tgt}\nGenerated: {datetime.now().isoformat()}\n")
        for fn in ["metrics.json", "config_used.yaml"]:
            src = os.path.join(tgt, fn)
            if os.path.exists(src):
                try:
                    shutil.copy2(src, os.path.join(best_dir, fn))
                except Exception:
                    pass
        return "pointer"


# --------------- sweep core ---------------
def run_sweep(base_cfg, sweep_grid, rank_key="pr_auc_illicit"):
    combos = list(product_dict(**sweep_grid))

    # normalize set (e.g., time_embed_dim==0 -> disable embed)
    sane_combos = []
    for c in combos:
        c = dict(c)
        if c.get("time_embed_dim", 0) == 0:
            c["time_embed_type"] = "none"
            c["time_embed_l2"] = 0.0
        if c.get("time_embed_type") == "sin" and c.get("time_embed_dim") not in (2, 4):
            continue
        sane_combos.append(c)

    # De-duplicate combos after normalization
    unique = []
    seen = set()
    for combo in sane_combos:
        key = json.dumps(combo, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        unique.append(combo)
    print(
        f"[SWEEP] {len(sane_combos)} combinations ({len(unique)} unique after normalization)"
    )
    sane_combos = unique

    rows, durations_txt, per_timestep_map = [], [], {}

    for i, combo in enumerate(sane_combos, 1):
        cfg = copy.deepcopy(base_cfg)
        cfg["symmetrize_edges"] = True
        for k, v in combo.items():
            cfg[k] = v

        # run_name
        tag = {
            "hidden_dim": "hid",
            "layers": "lay",
            "dropout": "dro",
            "lr": "lr",
            "weight_decay": "wei",
            "train_window_k": "tra",
            "time_embed_dim": "tim",
            "time_embed_type": "tmt",
            "time_embed_l2": "tel2",
            "time_loss_weighting": "tlw",
            "patience": "pat",
        }
        rn = cfg.get("arch", "model")
        if "sage" in rn:
            rn = "sage_resbn"
        rn += "".join([f"_{tag[k]}{slug(cfg[k])}" for k in tag if k in cfg])
        cfg["run_name"] = rn

        # Skip if already finished
        outdir = os.path.join("outputs", "gnn", rn)
        mpath = os.path.join(outdir, "metrics.json")
        if os.path.exists(mpath):
            print(f"[SKIP] {rn} already has metrics.json — skipping")
            rec = read_metrics(rn)
            rec["dt_seconds"] = 0.0
            rows.append(rec)
            per_ts = extract_per_timestep(rec)
            if per_ts:
                per_timestep_map[rn] = per_ts
                add_recent_summaries(rec, per_ts)
            continue

        print(f"\n[{i}/{len(sane_combos)}] run_name={rn}")
        t0 = time.time()
        try:
            train_main(cfg)
        except Exception as e:
            print(f"[ERROR] {rn}: {e}")
        dt = time.time() - t0
        durations_txt.append((rn, dt))

        rec = read_metrics(rn)
        rec["dt_seconds"] = round(dt, 2)
        per_ts = extract_per_timestep(rec)
        if per_ts:
            per_timestep_map[rn] = per_ts
            add_recent_summaries(rec, per_ts)

        rows.append(rec)
        print(f"[DONE] {rn} in {dt / 60:.1f} min")

    ensure_dir("outputs/sweeps")

    # Rich human-readable TXT (includes scores, not just durations)
    with open("outputs/sweeps/last_sweep.txt", "w") as f:
        f.write(
            "run_name\tpr_auc_test\tval_pr_auc\tpr_auc_last3\troc_auc\tf1_at_thr\tdt_seconds\n"
        )
        for r in rows:
            f.write(
                f"{r.get('run_name', '')}\t"
                f"{r.get('pr_auc_illicit', '')}\t"
                f"{r.get('best_val_pr_auc', '')}\t"
                f"{r.get('pr_auc_last3', '')}\t"
                f"{r.get('roc_auc', '')}\t"
                f"{r.get('f1_illicit_at_thr', '')}\t"
                f"{r.get('dt_seconds', '')}\n"
            )

    # summary tsv
    tsv_path = "outputs/sweeps/last_sweep.tsv"
    fields = [
        "run_name",
        "arch",
        "hidden_dim",
        "layers",
        "dropout",
        "lr",
        "weight_decay",
        "train_window_k",
        "time_embed_dim",
        "time_embed_type",
        "time_embed_l2",
        "time_loss_weighting",
        "symmetrize_edges",
        "patience",
        "best_val_pr_auc",
        "pr_auc_illicit",
        "roc_auc",
        "f1_illicit_at_thr",
        "threshold",
        "precision_at_k",
        "recall_at_precision",
        "ece",
        "n_test",
        "pr_auc_last1",
        "pr_auc_last3",
        "pr_auc_last5",
        "pr_auc_first3",
        "dt_seconds",
        "outdir",
    ]
    with open(tsv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    print(f"[WROTE] {tsv_path}")

    # jsonl
    jsonl_path = "outputs/sweeps/last_sweep.jsonl"
    with open(jsonl_path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[WROTE] {jsonl_path}")

    # per-timestep
    if per_timestep_map:
        max_T = max(len(v) for v in per_timestep_map.values())
        per_tsv = "outputs/sweeps/last_sweep_per_timestep.tsv"
        with open(per_tsv, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["run_name"] + [f"t{t}" for t in range(max_T)])
            for rn, seq in per_timestep_map.items():
                row = [rn] + [f"{x:.6f}" for x in seq] + [""] * (max_T - len(seq))
                w.writerow(row)
        print(f"[WROTE] {per_tsv}")

    # leaderboard + best/
    def score_of(r):
        return r.get(rank_key, r.get("pr_auc_illicit", float("-inf")))

    rows_scored = [r for r in rows if isinstance(score_of(r), (int, float))]
    rows_scored.sort(key=score_of, reverse=True)

    if rows_scored:
        lead_path = "outputs/sweeps/leaderboard.tsv"
        with open(lead_path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(
                [
                    "rank",
                    "run_name",
                    rank_key,
                    "test_pr_auc",
                    "val_pr_auc",
                    "dt_seconds",
                    "outdir",
                ]
            )
            for idx, r in enumerate(rows_scored, 1):
                w.writerow(
                    [
                        idx,
                        r["run_name"],
                        f"{score_of(r):.6f}",
                        f"{r.get('pr_auc_illicit', float('nan')):.6f}",
                        f"{r.get('best_val_pr_auc', float('nan')):.6f}",
                        r.get("dt_seconds", ""),
                        r.get("outdir", ""),
                    ]
                )
        print(f"[WROTE] {lead_path}")

        top = rows_scored[0]
        mode = point_best_to(top["outdir"])
        print(f"[BEST] {top['run_name']} ({rank_key}={score_of(top):.6f}) [{mode}]")
        print("Top 5 by rank key:")
        for r in rows_scored[:5]:
            print(
                f"  {r['run_name'][:55]:55s}  {rank_key}={score_of(r):.4f}  "
                f"test_pr_auc={r.get('pr_auc_illicit', float('nan')):.4f}  "
                f"val_pr_auc={r.get('best_val_pr_auc', float('nan')):.4f}"
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        type=str,
        required=True,
        help="base YAML (e.g., configs/sage_resbn_k8_sin2.yaml)",
    )
    ap.add_argument(
        "--rank_key",
        type=str,
        default="pr_auc_illicit",
        help="metric to rank by (e.g., pr_auc_last3)",
    )
    args = ap.parse_args()

    with open(args.base, "r") as f:
        base_cfg = yaml.safe_load(f)

    # Narrow grid around current champion (fast iteration)
    sweep_grid = dict(
        hidden_dim=[64],
        layers=[3],
        dropout=[0.2, 0.25],
        lr=[5e-4, 7e-4],
        weight_decay=[5e-5, 1e-4],
        train_window_k=[8, 9, 10],
        time_embed_dim=[0, 2, 4],  # 0 disables time features
        time_embed_type=["sin", "none"],  # focus on sin; "none" as control
        time_embed_l2=[0.0, 1e-4],
        time_loss_weighting=["none", "sqrt", "linear"],
        symmetrize_edges=[True],
        patience=[30],
    )

    run_sweep(base_cfg, sweep_grid, rank_key=args.rank_key)
