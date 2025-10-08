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
    if node_idx_path.exists():
        node_idx = np.load(node_idx_path).astype(int)
    else:
        node_idx = np.arange(len(y), dtype=int)
    return {"y": y, "scores": scores, "node_idx": node_idx}


def align_runs(
    run_a: Dict[str, np.ndarray], run_b: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    idx_a = run_a["node_idx"]
    idx_b = run_b["node_idx"]
    if idx_a.shape == idx_b.shape and np.array_equal(idx_a, idx_b):
        return run_a, run_b

    common, a_idx, b_idx = np.intersect1d(idx_a, idx_b, assume_unique=False, return_indices=True)
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
    k = min(topk, len(y))
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
) -> Dict[str, Dict[str, float]]:
    n = len(y_a)
    k = min(topk, n)
    pr_deltas = []
    p_at_k_deltas = []

    for _ in range(n_boot):
        sample_idx = np.random.choice(n, size=n, replace=True)
        y_a_s = y_a[sample_idx]
        scores_a_s = scores_a[sample_idx]
        y_b_s = y_b[sample_idx]
        scores_b_s = scores_b[sample_idx]

        pr_a = pr_auc_illicit(y_a_s, scores_a_s)
        pr_b = pr_auc_illicit(y_b_s, scores_b_s)
        pr_deltas.append(pr_b - pr_a)

        p_a = precision_at_k(y_a_s, scores_a_s, k)
        p_b = precision_at_k(y_b_s, scores_b_s, k)
        p_at_k_deltas.append(p_b - p_a)

    pr_deltas = np.array(pr_deltas)
    p_at_k_deltas = np.array(p_at_k_deltas)

    def summarize(deltas: np.ndarray) -> Dict[str, float]:
        lower, upper = np.percentile(deltas, [2.5, 97.5])
        return {
            "delta_mean": float(np.mean(deltas)),
            "delta_ci_lower": float(lower),
            "delta_ci_upper": float(upper),
        }

    return {
        "pr_auc": summarize(pr_deltas),
        "precision_at_k": summarize(p_at_k_deltas),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired bootstrap comparison between two runs")
    parser.add_argument("--run_a", type=Path, required=True, help="Baseline run directory")
    parser.add_argument("--run_b", type=Path, required=True, help="Comparison run directory")
    parser.add_argument("--topk", type=int, default=100, help="Top-k for precision")
    parser.add_argument("--n_boot", type=int, default=1000, help="Number of bootstrap samples")
    args = parser.parse_args()

    run_a = load_run(args.run_a)
    run_b = load_run(args.run_b)
    run_a, run_b = align_runs(run_a, run_b)

    y_a, scores_a = run_a["y"], run_a["scores"]
    y_b, scores_b = run_b["y"], run_b["scores"]

    metrics_a = compute_metrics(y_a, scores_a, args.topk)
    metrics_b = compute_metrics(y_b, scores_b, args.topk)
    bootstrap_summary = paired_bootstrap(y_a, scores_a, y_b, scores_b, args.topk, args.n_boot)

    output = {
        "run_a": str(args.run_a),
        "run_b": str(args.run_b),
        "n_samples": int(len(y_a)),
        "n_bootstrap": int(args.n_boot),
        "top_k": int(min(args.topk, len(y_a))),
        "metrics": {
            "pr_auc": {
                "run_a": float(metrics_a["pr_auc"]),
                "run_b": float(metrics_b["pr_auc"]),
                **bootstrap_summary["pr_auc"],
            },
            "precision_at_k": {
                "run_a": float(metrics_a["precision_at_k"]),
                "run_b": float(metrics_b["precision_at_k"]),
                **bootstrap_summary["precision_at_k"],
            },
        },
    }

    output_path = Path("bootstrap_compare.json")
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    main()
