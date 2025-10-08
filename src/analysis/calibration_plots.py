import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.metrics import expected_calibration_error

plt.switch_backend("Agg")


def compute_reliability(y: np.ndarray, scores: np.ndarray, bins: int = 15):
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_centers = []
    accuracies = []
    confidences = []

    for i in range(bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i < bins - 1:
            mask = (scores >= lo) & (scores < hi)
        else:
            mask = (scores >= lo) & (scores <= hi)
        if not np.any(mask):
            continue
        scores_bin = scores[mask]
        y_bin = y[mask]
        bin_centers.append((lo + hi) / 2.0)
        confidences.append(scores_bin.mean())
        accuracies.append(y_bin.mean())
    return np.array(bin_centers), np.array(accuracies), np.array(confidences)


def main(run_dir: Path) -> None:
    y = np.load(run_dir / "y_test.npy").astype(int)
    scores = np.load(run_dir / "scores_test.npy")

    bin_centers, accs, confs = compute_reliability(y, scores)
    ece = expected_calibration_error(y, scores, bins=15)

    plt.figure(figsize=(7, 5))
    width = 1.0 / 15
    if len(bin_centers) > 0:
        plt.bar(bin_centers, accs, width=width, alpha=0.6, label="Empirical accuracy", align="center")
        plt.plot(bin_centers, confs, marker="o", linestyle="--", color="black", label="Mean confidence")
    plt.plot([0, 1], [0, 1], color="red", linestyle=":", label="Perfect calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(f"Calibration Curve (ECE={ece:.4f})")
    plt.legend()
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "calibration_curve.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration plots")
    parser.add_argument("--run_dir", type=Path, required=True, help="Directory containing evaluation artifacts")
    args = parser.parse_args()

    main(args.run_dir)
