import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import yaml

from src.data.dataset_elliptic import make_temporal_masks
from src.train_gnn import build_model, load_cached
from src.utils.calibrate import TemperatureScaler
from src.utils.common import gpu_available
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

# ...existing code...


def prepare_data(cfg: dict, device: torch.device):
    data = load_cached(cfg["processed_dir"])

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

    if cfg.get("use_time_scalar", False):
        tnorm = (data.timestep.float() / float(data.timestep.max())).unsqueeze(1)
        data.x = torch.cat([data.x, tnorm], dim=1)

    edge_index = data.edge_index
    if cfg.get("symmetrize_edges", False):
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        data.edge_index = edge_index

    return data.to(device)


def build_edge_index_ablated(
    edge_index: torch.Tensor, num_nodes: int, frac: float, device: torch.device
) -> Tuple[torch.Tensor, int]:
    frac = float(frac)
    num_hubs = int(frac * float(num_nodes))
    edge_index_cpu = edge_index.detach().cpu()
    deg_src = torch.bincount(edge_index_cpu[0], minlength=num_nodes)
    deg_dst = torch.bincount(edge_index_cpu[1], minlength=num_nodes)
    deg = deg_src + deg_dst
    hubs = torch.zeros(num_nodes, dtype=torch.bool)
    if num_hubs > 0:
        topk = torch.topk(deg, num_hubs)
        hubs[topk.indices] = True
    edge_mask = ~(hubs[edge_index_cpu[0]] | hubs[edge_index_cpu[1]])
    edge_index_abl = edge_index_cpu[:, edge_mask].to(device)
    return edge_index_abl, int(num_hubs)


def compute_threshold(
    cfg: dict, y_val: np.ndarray, p_val: np.ndarray, y_te: np.ndarray, p_te: np.ndarray
) -> float:
    if cfg.get("use_val_for_thresholds", True):
        if cfg.get("precision_target", 0.0) and cfg["precision_target"] > 0:
            return pick_threshold_for_precision(
                (y_val == 1).astype(int), p_val, cfg["precision_target"]
            )
        thr, _ = pick_threshold_max_f1((y_val == 1).astype(int), p_val)
        return thr
    thr, _ = pick_threshold_max_f1((y_te == 1).astype(int), p_te)
    return thr


def _infer_expected_in_features_from_state(model: torch.nn.Module) -> int | None:
    # Try common GraphSAGE linear weight keys first, else first 2D weight we find.
    sd = model.state_dict()
    for suffix in ("lin_l.weight", "lin_rel.weight", "lin.weight"):
        for k, w in sd.items():
            if k.endswith(suffix) and w.dim() == 2:
                return int(w.shape[1])
    for _, w in sd.items():
        if w.dim() == 2:
            return int(w.shape[1])
    return None


def _align_features_to_model(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    expected_in = _infer_expected_in_features_from_state(model)
    cur_in = int(x.size(1))
    if expected_in is None or expected_in == cur_in:
        return x
    if cur_in < expected_in:
        pad = torch.zeros(
            x.size(0), expected_in - cur_in, device=x.device, dtype=x.dtype
        )
        x = torch.cat([x, pad], dim=1)
        print(f"[ABL] padded features {cur_in} -> {expected_in}")
    else:
        x = x[:, :expected_in]
        print(f"[ABL] truncated features {cur_in} -> {expected_in}")
    return x


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate hub ablation for a trained GNN run"
    )
    parser.add_argument(
        "--run_dir", type=Path, required=True, help="Path to the GNN run directory"
    )
    parser.add_argument(
        "--frac", type=float, default=0.01, help="Fraction of nodes to ablate as hubs"
    )
    args = parser.parse_args()

    run_dir = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    config_path = run_dir / "config_used.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Expected config at {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if gpu_available() else "cpu")
    data = prepare_data(cfg, device)

    # Build model and load weights
    in_dim = data.x.size(1)
    model = build_model(cfg["arch"], in_dim, cfg).to(device)
    state_dict = torch.load(run_dir / "best.ckpt", map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    # Align features to the trained model's expected input width (handles feature-engineering mismatches)
    data.x = _align_features_to_model(model, data.x)

    edge_index = data.edge_index
    with torch.no_grad():
        logits_base = model(data.x, edge_index)
    ts = None
    if cfg.get("calibrate_temperature", True):
        ts = TemperatureScaler().to(device)
        _ = ts.fit(logits_base[data.val_mask], data.y[data.val_mask])
        logits_base = logits_base / ts.T
    probs_base = torch.softmax(logits_base, dim=1)[:, 1].detach().cpu().numpy()

    y_np = data.y.detach().cpu().numpy()
    val_mask = data.val_mask.detach().cpu().numpy()
    test_mask = data.test_mask.detach().cpu().numpy()

    y_val = y_np[val_mask]
    p_val = probs_base[val_mask]
    y_te = y_np[test_mask]
    p_te = probs_base[test_mask]

    thr = compute_threshold(cfg, y_val, p_val, y_te, p_te)
    edge_index_abl, n_hubs = build_edge_index_ablated(
        edge_index, data.num_nodes, args.frac, device
    )

    with torch.no_grad():
        logits_abl = model(data.x, edge_index_abl)
        if ts is not None:
            logits_abl = logits_abl / ts.T
        probs_abl = torch.softmax(logits_abl, dim=1)[:, 1].detach().cpu().numpy()

    y_bin = (y_te == 1).astype(int)
    p_te_abl = probs_abl[test_mask]

    metrics = {
        "pr_auc_illicit": pr_auc_illicit(y_bin, p_te_abl),
        "roc_auc": roc_auc_illicit(y_bin, p_te_abl),
        "f1_illicit_at_thr": f1_at_threshold(y_bin, p_te_abl, thr),
        "threshold": thr,
        "precision_at_k": precision_at_k(y_bin, p_te_abl, cfg.get("topk", 100)),
        "recall_at_precision": recall_at_precision(
            y_bin, p_te_abl, cfg.get("precision_target", 0.90)
        ),
        "ece": expected_calibration_error(y_bin, p_te_abl),
        "n_test": int(len(y_te)),
        "n_hubs": int(n_hubs),
        "hub_fraction": float(args.frac),
        "n_edges_remaining": int(edge_index_abl.size(1)),
    }

    frac_str = str(args.frac).replace(".", "p")
    output_path = run_dir / f"metrics_hub_removed_{frac_str}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
