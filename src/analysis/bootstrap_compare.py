import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from src.utils.metrics import pr_auc_illicit, precision_at_k


def load_run(run_dir: Path) -> Dict[str, np.ndarray]:
    y = np.load(run_dir / "y_test.npy").astype(int)
    scores = np.load(run_dir / "scores_test.npy")
    node_idx_path = run_dir / "node_idx_test.npy"
    node_idx = (
        np.load(node_idx_path).astype(int)
        if node_idx_path.exists()
        else np.arange(len(y), dtype=int)
    )
    return {"y": y, "scores": scores, "node_idx": node_idx}


def align_runs(
    run_a: Dict[str, np.ndarray], run_b: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    idx_a = run_a["node_idx"]
    idx_b = run_b["node_idx"]
    if idx_a.shape == idx_b.shape and np.array_equal(idx_a, idx_b):
        return run_a, run_b

    common, a_idx, b_idx = np.intersect1d(
        idx_a, idx_b, assume_unique=False, return_indices=True
    )
    if common.size == 0:
        raise ValueError("No overlapping nodes between the provided runs.")

    aligned_a = {
        "y": run_a["y"][a_idx],
        "scores": run_a["scores"][a_idx],
        "node_idx": idx_a[a_idx],
    }
    aligned_b = {
        "y": run_b["y"][b_idx],
        "scores": run_b["scores"][b_idx],
        "node_idx": idx_b[b_idx],
    }
    return aligned_a, aligned_b


def compute_metrics(y: np.ndarray, scores: np.ndarray, topk: int) -> Dict[str, float]:
    k = min(int(topk), len(y))
    return {
        "pr_auc": pr_auc_illicit(y, scores),
        "precision_at_k": precision_at_k(y, scores, k),
    }


def paired_bootstrap(
    y_a: np.ndarray,
    scores_a: np.ndarray,
    y_b: np.ndarray,
    scores_b: np.ndarray,
    topk: int,
    n_boot: int,
    seed: int | None = 42,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_a)
    k = min(int(topk), n)
    pr_deltas = []
    p_at_k_deltas = []

    for _ in range(int(n_boot)):
        sample_idx = rng.choice(n, size=n, replace=True)
        y_a_s, s_a_s = y_a[sample_idx], scores_a[sample_idx]
        y_b_s, s_b_s = y_b[sample_idx], scores_b[sample_idx]

        pr_deltas.append(pr_auc_illicit(y_b_s, s_b_s) - pr_auc_illicit(y_a_s, s_a_s))
        p_at_k_deltas.append(
            precision_at_k(y_b_s, s_b_s, k) - precision_at_k(y_a_s, s_a_s, k)
        )

    pr_deltas = np.asarray(pr_deltas)
    p_at_k_deltas = np.asarray(p_at_k_deltas)

    def summarize(d: np.ndarray) -> Dict[str, float]:
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"delta": float(np.mean(d)), "ci_low": float(lo), "ci_high": float(hi)}

    return {
        "delta_pr_auc": summarize(pr_deltas),
        "delta_p_at_k": summarize(p_at_k_deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap comparison between two runs"
    )
    parser.add_argument(
        "--run_a", type=Path, required=True, help="Baseline run directory"
    )
    parser.add_argument(
        "--run_b", type=Path, required=True, help="Comparison run directory"
    )
    parser.add_argument("--topk", type=int, default=100, help="Top-k for precision")
    parser.add_argument(
        "--n_boot", type=int, default=1000, help="Number of bootstrap samples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out_dir", type=Path, default=None, help="Optional extra output directory"
    )
    args = parser.parse_args()

    run_a = load_run(args.run_a)
    run_b = load_run(args.run_b)
    run_a, run_b = align_runs(run_a, run_b)

    y_a, s_a = run_a["y"], run_a["scores"]
    y_b, s_b = run_b["y"], run_b["scores"]

    m_a = compute_metrics(y_a, s_a, args.topk)
    m_b = compute_metrics(y_b, s_b, args.topk)
    boots = paired_bootstrap(y_a, s_a, y_b, s_b, args.topk, args.n_boot, args.seed)

    n = int(len(y_a))
    k = int(min(args.topk, n))
    output = {
        "run_a": str(args.run_a),
        "run_b": str(args.run_b),
        "n_samples": n,
        "n_bootstrap": int(args.n_boot),
        "top_k": k,
        # deltas for Streamlit:
        **boots,
        # raw metrics for reference:
        "pr_auc": {"run_a": float(m_a["pr_auc"]), "run_b": float(m_b["pr_auc"])},
        "precision_at_k": {
            "run_a": float(m_a["precision_at_k"]),
            "run_b": float(m_b["precision_at_k"]),
        },
    }

    # --- Write where the dashboard expects ---
    run_b_dir = args.run_b
    run_a_dir = args.run_a
    run_b_dir.mkdir(parents=True, exist_ok=True)
    run_a_dir.mkdir(parents=True, exist_ok=True)

    # Primary file (Compare tab will find this):
    (run_b_dir / "bootstrap_compare.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    # Helpful symmetric filenames:
    (run_b_dir / f"bootstrap_compare_{run_a_dir.name}.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    (run_a_dir / f"bootstrap_compare_{run_b_dir.name}.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )

    # Optional extra copy:
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / "bootstrap_compare.json").write_text(
            json.dumps(output, indent=2), encoding="utf-8"
        )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
