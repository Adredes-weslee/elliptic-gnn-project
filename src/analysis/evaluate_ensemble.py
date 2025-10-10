"""
Ensemble two trained runs (probability- or logit-average), recompute thresholds on val,
and write artifacts/metrics just like a normal run directory.

Usage:
  python -m src.analysis.evaluate_ensemble \
    --run_a outputs/gnn/<RUN_A> \
    --run_b outputs/gnn/<RUN_B> \
    --out_dir outputs/gnn/<OUT_DIR> \
    --mode logit \
    --precision_target 0.0 \
    --topk 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.utils.common import ensure_dir
from src.utils.metrics import (
    expected_calibration_error,
    f1_at_threshold,
    pick_threshold_for_precision,
    pick_threshold_max_f1,
    pr_auc_illicit,
    precision_at_k,
    recall_at_precision,
    roc_auc_illicit,
)


def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _load_split(run_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # returns (scores, labels, node_idx)
    s = np.load(run_dir / f"scores_{split}.npy")
    y = np.load(run_dir / f"y_{split}.npy")
    idx = np.load(run_dir / f"node_idx_{split}.npy")
    return s, y, idx


def _load_timestep(run_dir: Path, split: str) -> np.ndarray:
    return np.load(run_dir / f"timestep_{split}.npy")


def _align(
    base_ids: np.ndarray, other_ids: np.ndarray, other_vals: np.ndarray
) -> np.ndarray:
    """Return 'other_vals' reordered to match 'base_ids' ordering."""
    if np.array_equal(base_ids, other_ids):
        return other_vals
    pos = {nid: i for i, nid in enumerate(other_ids)}
    return np.asarray([other_vals[pos[n]] for n in base_ids])


def _ensemble(pa: np.ndarray, pb: np.ndarray, mode: str = "logit") -> np.ndarray:
    if mode == "prob":
        return 0.5 * (pa + pb)
    if mode == "logit":
        za, zb = _logit(pa), _logit(pb)
        ze = 0.5 * (za + zb)
        return _sigmoid(ze)
    raise ValueError(f"Unknown mode={mode!r}. Use 'prob' or 'logit'.")


def _select_threshold(
    y_val: np.ndarray,
    p_val: np.ndarray,
    precision_target: float,
) -> float:
    yb = (y_val == 1).astype(int)
    if precision_target and precision_target > 0:
        thr = pick_threshold_for_precision(yb, p_val, precision_target)
    else:
        thr, _ = pick_threshold_max_f1(yb, p_val)
    return float(thr)


def _metrics_for_test(
    y_test: np.ndarray,
    p_test: np.ndarray,
    thr: float,
    topk: int,
    timestep_test: np.ndarray | None = None,
) -> Dict:
    yb = (y_test == 1).astype(int)
    pr = pr_auc_illicit(yb, p_test)
    roc = roc_auc_illicit(yb, p_test)
    f1 = f1_at_threshold(yb, p_test, thr)
    p_at_k = precision_at_k(yb, p_test, topk)
    rec_at_p = recall_at_precision(yb, p_test, 0.90)  # same default as train_gnn
    ece = expected_calibration_error(yb, p_test)

    out = dict(
        pr_auc_illicit=pr,
        roc_auc=roc,
        f1_illicit_at_thr=f1,
        threshold=float(thr),
        precision_at_k=p_at_k,
        recall_at_precision=rec_at_p,
        ece=ece,
        n_test=int(len(y_test)),
    )

    # Optional: per-timestep PR-AUC (mirrors train_gnn)
    if timestep_test is not None and timestep_test.size > 0:
        uniq = sorted(set(int(t) for t in timestep_test.tolist()))
        pr_by_t = []
        for t in uniq:
            m = timestep_test == t
            if m.sum() == 0:
                pr_by_t.append(float("nan"))
            else:
                pr_by_t.append(pr_auc_illicit((y_test[m] == 1).astype(int), p_test[m]))

        def _avg_tail(arr, k):
            return float(sum(arr[-k:]) / k) if len(arr) >= k else None

        out["test_pr_auc_by_time"] = pr_by_t
        out["pr_auc_last1"] = float(pr_by_t[-1]) if pr_by_t else None
        if len(pr_by_t) >= 3:
            out["pr_auc_last3"] = _avg_tail(pr_by_t, 3)
        if len(pr_by_t) >= 5:
            out["pr_auc_last5"] = _avg_tail(pr_by_t, 5)

    return out


def main():
    ap = argparse.ArgumentParser(
        description="Ensemble two trained runs and write metrics/artifacts."
    )
    ap.add_argument("--run_a", type=Path, required=True)
    ap.add_argument("--run_b", type=Path, required=True)
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Where to write ensemble outputs (default: <run_b>/ensemble)",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="logit",
        choices=["prob", "logit"],
        help="Average probabilities or logits",
    )
    ap.add_argument(
        "--precision_target",
        type=float,
        default=0.0,
        help="If >0, pick threshold to hit this precision on val",
    )
    ap.add_argument("--topk", type=int, default=100)
    args = ap.parse_args()

    run_a = args.run_a.resolve()
    run_b = args.run_b.resolve()

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = run_b / "ensemble"
    out_dir = out_dir.resolve()
    ensure_dir(str(out_dir))

    # --- Load VAL splits and align A->B ---
    pva, yva, ida = _load_split(run_a, "val")
    pvb, yvb, idb = _load_split(run_b, "val")
    if not np.array_equal(np.sort(ida), np.sort(idb)):
        raise RuntimeError("VAL node indices differ between runs.")
    pva = _align(idb, ida, pva)  # reorder pa to B's order
    assert np.array_equal(yva := _align(idb, ida, yva), yvb), (
        "VAL labels differ after alignment."
    )

    # --- Ensemble on VAL and select threshold ---
    pve = _ensemble(pva, pvb, mode=args.mode)
    thr = _select_threshold(yvb, pve, args.precision_target)

    # --- Load TEST splits and align, then ensemble ---
    pta, yta, ida_t = _load_split(run_a, "test")
    ptb, ytb, idb_t = _load_split(run_b, "test")
    if not np.array_equal(np.sort(ida_t), np.sort(idb_t)):
        raise RuntimeError("TEST node indices differ between runs.")
    pta = _align(idb_t, ida_t, pta)
    yta = _align(idb_t, ida_t, yta)
    if not np.array_equal(yta, ytb):
        raise RuntimeError("TEST labels differ after alignment.")
    pte = _ensemble(pta, ptb, mode=args.mode)

    # --- Timestep (for per-time PR-AUC) ---
    t_val = (
        _load_timestep(run_b, "val") if (run_b / "timestep_val.npy").exists() else None
    )
    t_test = (
        _load_timestep(run_b, "test")
        if (run_b / "timestep_test.npy").exists()
        else None
    )

    # --- Save ensemble artifacts (names match normal run layout) ---
    np.save(out_dir / "scores_val.npy", pve)
    np.save(out_dir / "y_val.npy", yvb)
    np.save(out_dir / "node_idx_val.npy", idb)
    if t_val is not None:
        np.save(out_dir / "timestep_val.npy", t_val)

    np.save(out_dir / "scores_test.npy", pte)
    np.save(out_dir / "y_test.npy", ytb)
    np.save(out_dir / "node_idx_test.npy", idb_t)
    if t_test is not None:
        np.save(out_dir / "timestep_test.npy", t_test)

    # --- Compute metrics on TEST (threshold chosen on VAL above) ---
    metrics = _metrics_for_test(ytb, pte, thr, args.topk, timestep_test=t_test)

    # Set a simple run_name & carry a note about sources
    metrics["ensemble_sources"] = {
        "run_a": str(run_a),
        "run_b": str(run_b),
        "mode": args.mode,
    }

    # Best-effort carry-over: read best_val_pr_auc if present
    for src in (run_a, run_b):
        try:
            with open(src / "metrics.json", "r", encoding="utf-8") as fh:
                m = json.load(fh)
            if "best_val_pr_auc" in m:
                metrics.setdefault("best_val_pr_auc", float(m["best_val_pr_auc"]))
                metrics["best_val_pr_auc"] = max(
                    metrics["best_val_pr_auc"], float(m["best_val_pr_auc"])
                )
        except Exception:
            pass

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    # minimal config echo so analysis scripts can find basics
    cfg_echo = {
        "run_name": f"ensemble_of_{run_a.name}_and_{run_b.name}",
        "precision_target": float(args.precision_target),
        "topk": int(args.topk),
        "ensemble_mode": args.mode,
    }
    with open(out_dir / "config_used.yaml", "w", encoding="utf-8") as fh:
        # tiny YAML without importing yaml (keep deps minimal)
        fh.write("\n".join(f"{k}: {v}" for k, v in cfg_echo.items()))

    print(json.dumps(metrics, indent=2))
    print(f"[DONE] Ensemble written to: {out_dir}")


if __name__ == "__main__":
    main()
