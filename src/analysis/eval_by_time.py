import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, f1_score

plt.switch_backend("Agg")


def compute_metrics_per_timestep(y: np.ndarray, scores: np.ndarray, timesteps: np.ndarray, threshold: float) -> List[dict]:
    results = []
    unique_timesteps = np.unique(timesteps)
    for timestep in unique_timesteps:
        mask = timesteps == timestep
        y_t = y[mask]
        scores_t = scores[mask]
        n = int(mask.sum())

        pr_auc = np.nan
        if n > 0:
            try:
                pr_auc = float(average_precision_score(y_t, scores_t))
            except ValueError:
                pr_auc = float("nan")
        f1 = (
            float(f1_score(y_t, (scores_t >= threshold).astype(int), zero_division=0))
            if n > 0
            else float("nan")
        )

        results.append(
            dict(
                timestep=int(timestep),
                pr_auc_illicit=pr_auc,
                f1_illicit_at_thr=f1,
                n=n,
            )
        )
    return results


def main(run_dir: Path) -> None:
    scores = np.load(run_dir / "scores_test.npy")
    y = np.load(run_dir / "y_test.npy")
    timesteps = np.load(run_dir / "timestep_test.npy")
    with open(run_dir / "metrics.json", "r") as f:
        metrics = json.load(f)
    threshold = float(metrics.get("threshold"))

    y = y.astype(int)
    results = compute_metrics_per_timestep(y, scores, timesteps, threshold)

    out_csv = run_dir / "by_time.csv"
    with open(out_csv, "w") as f:
        f.write("timestep,pr_auc_illicit,f1_illicit_at_thr,n\n")
        for row in results:
            f.write(f"{row['timestep']},{row['pr_auc_illicit']},{row['f1_illicit_at_thr']},{row['n']}\n")

    # Plot PR-AUC vs timestep
    timesteps_ordered = [row["timestep"] for row in results]
    pr_values = [row["pr_auc_illicit"] for row in results]

    plt.figure(figsize=(8, 5))
    plt.plot(timesteps_ordered, pr_values, marker="o")
    plt.xlabel("Timestep")
    plt.ylabel("PR-AUC (Illicit)")
    plt.title("PR-AUC over Timestep")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(run_dir / "by_time_pr_auc.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate metrics by timestep")
    parser.add_argument("--run_dir", type=Path, required=True, help="Directory containing evaluation artifacts")
    args = parser.parse_args()

    main(args.run_dir)
