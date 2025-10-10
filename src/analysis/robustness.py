# src/analysis/robustness.py
import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml
from torch_geometric.data import Data

from src.data.dataset_elliptic import make_temporal_masks
from src.train_gnn import build_model, load_cached
from src.utils.calibrate import TemperatureScaler
from src.utils.common import gpu_available, set_seed
from src.utils.metrics import (
    expected_calibration_error,
    f1_at_threshold,
    pr_auc_illicit,
    precision_at_k,
    recall_at_precision,
    roc_auc_illicit,
)


def _model_uses_time_embed(model) -> bool:
    return (
        hasattr(model, "time_embed_dim")
        and int(getattr(model, "time_embed_dim", 0)) > 0
    )


def prepare_data(cfg: dict, processed_dir: Path, device: torch.device) -> Data:
    data = load_cached(str(processed_dir))

    # Optional rolling window (match training)
    window_k = cfg.get("train_window_k")
    if window_k is not None:
        window_k = int(window_k)
        train_ts = data.timestep[data.train_mask]
        if train_ts.numel() == 0:
            raise RuntimeError("Train mask is empty; cannot apply rolling window.")
        t_train_end = int(train_ts.max().item())
        val_ts = data.timestep[data.val_mask]
        if val_ts.numel() == 0:
            raise RuntimeError("Validation mask is empty; cannot infer t_val_end.")
        t_val_end = int(val_ts.max().item())
        data = make_temporal_masks(data, t_train_end, t_val_end, window_k)

    # Optional scalar time (only if used during training and no time embedding)
    if cfg.get("use_time_scalar", False) and int(cfg.get("time_embed_dim", 0)) == 0:
        tnorm = (data.timestep.float() / float(data.timestep.max())).unsqueeze(1)
        data.x = torch.cat([data.x, tnorm], dim=1)

    return data.to(device)


def build_edge_index(cfg: dict, data: Data) -> torch.Tensor:
    edge_index = data.edge_index
    if cfg.get("symmetrize_edges", False):
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return edge_index


def drop_edges(edge_index: torch.Tensor, drop_frac: float) -> Tuple[torch.Tensor, int]:
    drop_frac = float(drop_frac)
    if drop_frac < 0 or drop_frac > 1:
        raise ValueError("drop_frac must be within [0, 1]")
    if drop_frac <= 0:
        return edge_index, 0

    num_edges = edge_index.size(1)
    drop_count = int(round(drop_frac * float(num_edges)))
    drop_count = min(drop_count, num_edges)
    if drop_count == 0:
        return edge_index, 0
    if drop_count >= num_edges:
        raise RuntimeError("Dropping all edges would leave an empty graph.")

    perm = torch.randperm(num_edges, device=edge_index.device)
    keep_idx = perm[drop_count:]
    return edge_index[:, keep_idx], drop_count


def maybe_add_noise(x: torch.Tensor, noise_std: float) -> torch.Tensor:
    noise_std = float(noise_std)
    if noise_std <= 0:
        return x
    noise = torch.randn_like(x) * noise_std
    return x + noise


def compute_probs(
    model: torch.nn.Module,
    x_input: torch.Tensor,
    data: Data,
    edge_index: torch.Tensor,
    use_temp_scale: bool,
) -> Tuple[np.ndarray, float]:
    # --- forward with (optional) time embedding ---
    with torch.no_grad():
        t_idx = data.timestep if _model_uses_time_embed(model) else None
        logits = model(x_input, edge_index, t_idx)

    temperature = 1.0
    if use_temp_scale:
        val_mask = data.val_mask
        if val_mask.sum().item() == 0:
            use_temp_scale = False
        else:
            ts = TemperatureScaler().to(logits.device)
            _ = ts.fit(logits[val_mask], data.y[val_mask])
            logits = logits / ts.T
            temperature = float(ts.T.detach().cpu().item())

    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    return probs, temperature


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate robustness of a trained GNN under noise and edge drop"
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to the trained GNN run directory",
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing processed graph.pt",
    )
    parser.add_argument(
        "--drop_frac",
        type=float,
        default=0.10,
        help="Fraction of edges to drop uniformly at random",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.01,
        help="Stddev of Gaussian noise added to node features",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)

    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected config at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Expected metrics at {metrics_path}")
    with metrics_path.open("r", encoding="utf-8") as f:
        base_metrics = json.load(f)
    if "threshold" not in base_metrics:
        raise KeyError("metrics.json does not contain 'threshold'")
    threshold = float(base_metrics["threshold"])

    processed_dir = (
        args.processed_dir
        if args.processed_dir is not None
        else Path(cfg.get("processed_dir", "data/processed"))
    )
    graph_path = processed_dir / "graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Processed graph not found at {graph_path}")

    device = torch.device("cuda" if gpu_available() else "cpu")
    data = prepare_data(cfg, processed_dir, device)

    # Build model exactly as in training (includes time-embed configuration)
    in_dim = data.x.size(1)
    model = build_model(cfg["arch"], in_dim, cfg).to(device)
    state_dict = torch.load(run_dir / "best.ckpt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Edge drop + feature noise
    edge_index_original = build_edge_index(cfg, data)
    edge_index, drop_count = drop_edges(edge_index_original, args.drop_frac)
    x_input = maybe_add_noise(data.x, args.noise_std)

    # Temperature scaling consistent with training pipeline
    use_temp_scale = bool(cfg.get("calibrate_temperature", True))
    probs, temperature = compute_probs(model, x_input, data, edge_index, use_temp_scale)

    # Metrics on TEST
    y_np = data.y.detach().cpu().numpy()
    test_mask = data.test_mask.detach().cpu().numpy().astype(bool)
    y_test = y_np[test_mask]
    p_test = probs[test_mask]
    y_bin = (y_test == 1).astype(int)

    topk = int(cfg.get("topk", 100))
    precision_target = float(cfg.get("precision_target", 0.90))

    n_edges_original = int(edge_index_original.size(1))
    metrics = {
        "pr_auc_illicit": pr_auc_illicit(y_bin, p_test),
        "roc_auc": roc_auc_illicit(y_bin, p_test),
        "f1_illicit_at_thr": f1_at_threshold(y_bin, p_test, threshold),
        "threshold": threshold,
        "precision_at_k": precision_at_k(y_bin, p_test, topk),
        "recall_at_precision": recall_at_precision(y_bin, p_test, precision_target),
        "ece": expected_calibration_error(y_bin, p_test),
        "n_test": int(len(y_test)),
        "drop_frac_requested": float(args.drop_frac),
        "drop_frac_effective": float(drop_count / n_edges_original)
        if n_edges_original
        else 0.0,
        "edges_dropped": int(drop_count),
        "n_edges_original": n_edges_original,
        "n_edges_after_drop": int(edge_index.size(1)),
        "noise_std": float(args.noise_std),
        "temperature": float(temperature),
        "seed": int(args.seed),
    }

    output_name = f"robustness_drop{args.drop_frac}_noise{args.noise_std}.json"
    output_path = run_dir / output_name
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
