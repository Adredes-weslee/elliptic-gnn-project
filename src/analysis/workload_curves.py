import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")


def compute_precision_curve(y: np.ndarray, scores: np.ndarray, k_values):
    order = np.argsort(-scores)
    y_sorted = y[order]
    cumulative = np.cumsum(y_sorted)
    precisions = []
    for k in k_values:
        k_eff = min(k, len(y_sorted))
        if k_eff == 0:
            precisions.append(float("nan"))
            continue
        precision = cumulative[k_eff - 1] / k_eff
        precisions.append(float(precision))
    return precisions


def build_k_values(n: int, k_max: int):
    if n == 0:
        return []
    limit = min(k_max, n)
    if limit < 10:
        return [limit]
    values = list(range(10, limit + 1, 10))
    if values[-1] != limit:
        values.append(limit)
    return values


def main(run_dir: Path, k_max: int) -> None:
    y = np.load(run_dir / "y_test.npy").astype(int)
    scores = np.load(run_dir / "scores_test.npy")

    k_values = build_k_values(len(y), k_max)
    precisions = compute_precision_curve(y, scores, k_values)

    out_csv = run_dir / "workload_curve.csv"
    with open(out_csv, "w") as f:
        f.write("k,precision_at_k\n")
        for k, p in zip(k_values, precisions):
            f.write(f"{k},{p}\n")

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, precisions, marker="o")
    plt.xlabel("k")
    plt.ylabel("Precision@k")
    plt.title("Workload Precision Curve")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(run_dir / "workload_curve.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Workload precision curves")
    parser.add_argument("--run_dir", type=Path, required=True, help="Directory containing evaluation artifacts")
    parser.add_argument("--k_max", type=int, default=5000, help="Maximum k to evaluate")
    args = parser.parse_args()

    main(args.run_dir, args.k_max)
