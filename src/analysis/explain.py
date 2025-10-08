"""Model explainability utilities for baselines and GNNs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.nn.models import GNNExplainer
from xgboost import XGBClassifier

plt.switch_backend("Agg")

try:  # Optional at runtime, required for xgb mode
    import shap
except ImportError as exc:  # pragma: no cover - handled at runtime
    shap = None  # type: ignore[assignment]
    _SHAP_IMPORT_ERROR = exc
else:
    _SHAP_IMPORT_ERROR = None


DEFAULT_XGB_PARAMS: Dict[str, object] = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "tree_method": "auto",
}
MODEL_FILENAME = "xgb_model.json"


@dataclass
class XGBArtifacts:
    model: XGBClassifier
    X_test: np.ndarray
    feature_names: List[str]


def read_config_file(path: Path) -> Dict:
    if path.suffix in {".yaml", ".yml"}:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    raise ValueError(f"Unsupported config format: {path.suffix}")


def find_run_config(run_dir: Path) -> Optional[Dict]:
    for name in ("config.yaml", "config.yml", "config.json", "config_used.yaml"):
        candidate = run_dir / name
        if candidate.exists():
            return read_config_file(candidate)
    return None


def pick_tree_method() -> str:
    try:
        return "gpu_hist" if torch.cuda.is_available() else "hist"
    except Exception:  # pragma: no cover - torch import/runtime issues
        return "hist"


def load_processed_graph(processed_dir: Path) -> Data:
    graph_path = processed_dir / "graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Processed graph not found: {graph_path}")
    data = torch.load(graph_path, map_location="cpu")
    if not isinstance(data, Data):
        raise TypeError(f"Expected torch_geometric.data.Data, got {type(data)!r}")
    return data


def labeled_split_arrays(data: Data) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    timestep = data.timestep.cpu().numpy()
    labeled_idx = np.where(y >= 0)[0]

    feats = x[labeled_idx]
    labels = y[labeled_idx]
    t = timestep[labeled_idx]

    if not hasattr(data, "train_mask") or not hasattr(data, "val_mask"):
        raise RuntimeError("Graph is missing train/val masks. Build data with src.data.build_graph first.")

    train_ts = data.timestep[data.train_mask].cpu().numpy()
    val_ts = data.timestep[data.val_mask].cpu().numpy()
    if train_ts.size == 0 or val_ts.size == 0:
        raise RuntimeError("Cannot infer train/val timesteps from masks.")
    t_train_end = int(train_ts.max())
    t_val_end = int(val_ts.max())

    train_mask = t <= t_train_end
    val_mask = (t > t_train_end) & (t <= t_val_end)
    test_mask = t > t_val_end
    return feats, labels, train_mask, val_mask, test_mask, labeled_idx, t


def apply_train_window(train_mask: np.ndarray, timesteps: np.ndarray, window_k: Optional[int]) -> np.ndarray:
    if window_k is None:
        return train_mask
    if train_mask.sum() == 0:
        raise RuntimeError("Train mask is empty; cannot apply rolling window.")
    train_ts = timesteps[train_mask]
    if train_ts.size == 0:
        raise RuntimeError("Train mask is empty after filtering labeled nodes.")
    T = int(train_ts.max())
    t_lo = max(1, T - int(window_k) + 1)
    window_mask = (timesteps >= t_lo) & (timesteps <= T)
    return train_mask & window_mask


def ensure_shap_import() -> None:
    if shap is None:
        raise RuntimeError(
            "shap is required for xgb explanations but is not installed"
        ) from _SHAP_IMPORT_ERROR


def load_or_fit_xgb(run_dir: Path, cfg: Optional[Dict]) -> XGBArtifacts:
    ensure_shap_import()

    processed_dir = Path(cfg.get("processed_dir", "data/processed")) if cfg else Path("data/processed")
    data = load_processed_graph(processed_dir)
    feats, labels, train_mask, _, test_mask, _, timesteps = labeled_split_arrays(data)

    window_k = None
    if cfg and cfg.get("train_window_k") is not None:
        window_k = int(cfg["train_window_k"])
    train_mask = apply_train_window(train_mask, timesteps, window_k)

    X_train, y_train = feats[train_mask], labels[train_mask]
    X_test = feats[test_mask]

    if X_train.size == 0:
        raise RuntimeError("No training data available for XGB model refit.")

    feature_names = [f"f{i}" for i in range(feats.shape[1])]

    model_path = run_dir / MODEL_FILENAME
    if model_path.exists():
        model = XGBClassifier()
        model.load_model(str(model_path))
        return XGBArtifacts(model=model, X_test=X_test, feature_names=feature_names)

    params = DEFAULT_XGB_PARAMS.copy()
    if cfg:
        params.update({k: cfg[k] for k in DEFAULT_XGB_PARAMS.keys() if k in cfg})

    if params.get("tree_method") == "auto":
        params["tree_method"] = pick_tree_method()

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    model.save_model(str(model_path))
    return XGBArtifacts(model=model, X_test=X_test, feature_names=feature_names)


def summarize_shap_values(shap_values: np.ndarray, feature_names: List[str], top_k: int = 20) -> List[Dict[str, float]]:
    if shap_values.ndim == 3:  # multi-class -> pick positive class (index 1)
        shap_values = shap_values[1]
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1][:top_k]
    return [
        {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
        for i in order
    ]


def run_xgb(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg = find_run_config(run_dir)
    artifacts = load_or_fit_xgb(run_dir, cfg)

    max_plots = max(int(args.max_plots), 1)

    rng = np.random.default_rng(seed=42)
    subset_size = min(200, artifacts.X_test.shape[0])
    if subset_size == 0:
        raise RuntimeError("No test samples available for SHAP analysis.")
    subset_idx = rng.choice(artifacts.X_test.shape[0], size=subset_size, replace=False)
    subset = artifacts.X_test[subset_idx]

    explainer = shap.TreeExplainer(artifacts.model)
    shap_values = explainer.shap_values(subset)

    shap.summary_plot(
        shap_values,
        subset,
        feature_names=artifacts.feature_names,
        plot_type="bar",
        show=False,
        max_display=max_plots,
    )
    plt.tight_layout()
    plt.savefig(run_dir / "shap_summary.png", dpi=200)
    plt.close()

    top_features = summarize_shap_values(np.array(shap_values), artifacts.feature_names)
    with open(run_dir / "xgb_top_features.json", "w", encoding="utf-8") as fh:
        json.dump(top_features, fh, indent=2)


class LogProbModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits = self.model(x, edge_index)
        return torch.log_softmax(logits, dim=-1)


def build_gnn_model(cfg: Dict, in_dim: int) -> torch.nn.Module:
    from src.models.gnn import GATNet, GCNNet, SAGENet

    arch = cfg["arch"].lower()
    hidden_dim = cfg["hidden_dim"]
    layers = cfg["layers"]
    dropout = cfg["dropout"]

    if arch == "gcn":
        model = GCNNet(in_dim, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    elif arch == "sage":
        model = SAGENet(in_dim, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    elif arch == "gat":
        heads = cfg.get("heads", 4)
        model = GATNet(in_dim, hidden_dim=hidden_dim, layers=layers, heads=heads, dropout=dropout)
    else:
        raise ValueError(f"Unknown GNN architecture: {cfg['arch']}")
    return model


def maybe_add_time_feature(data: Data, cfg: Dict) -> Data:
    if cfg.get("use_time_scalar", False):
        tnorm = (data.timestep.float() / float(data.timestep.max())).unsqueeze(1)
        data.x = torch.cat([data.x, tnorm], dim=1)
    return data


def maybe_symmetrize_edges(data: Data, cfg: Dict) -> Data:
    if cfg.get("symmetrize_edges", False):
        edge_index = torch.cat([data.edge_index, data.edge_index.flip(0)], dim=1)
        data.edge_index = edge_index
    return data


def prepare_data_for_gnn(cfg: Dict) -> Data:
    processed_dir = Path(cfg["processed_dir"])
    data = load_processed_graph(processed_dir)

    if cfg.get("train_window_k") is not None:
        from src.data.dataset_elliptic import make_temporal_masks

        window_k = int(cfg["train_window_k"])
        train_ts = data.timestep[data.train_mask]
        if train_ts.numel() == 0:
            raise RuntimeError("Train mask empty; cannot apply rolling window.")
        t_train_end = int(train_ts.max().item())
        val_ts = data.timestep[data.val_mask]
        if val_ts.numel() == 0:
            raise RuntimeError("Validation mask empty; cannot infer t_val_end.")
        t_val_end = int(val_ts.max().item())
        data = make_temporal_masks(data, t_train_end, t_val_end, window_k)

    data = maybe_add_time_feature(data, cfg)
    data = maybe_symmetrize_edges(data, cfg)
    return data


def pick_node_to_explain(run_dir: Path, node_override: Optional[int]) -> Tuple[int, Dict[str, float]]:
    scores_path = run_dir / "scores_test.npy"
    labels_path = run_dir / "y_test.npy"
    node_idx_path = run_dir / "node_idx_test.npy"
    if not (scores_path.exists() and labels_path.exists() and node_idx_path.exists()):
        raise FileNotFoundError("Required test score artifacts not found in run directory.")

    scores = np.load(scores_path)
    labels = np.load(labels_path)
    node_ids = np.load(node_idx_path)

    if node_override is not None:
        node_override = int(node_override)
        if node_override not in node_ids:
            raise ValueError(f"Requested node {node_override} not in test set indices.")
        idx = np.where(node_ids == node_override)[0][0]
        return node_override, {
            "selection": "user",
            "score": float(scores[idx]),
            "label": int(labels[idx]),
        }

    threshold = 0.5
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as fh:
            metrics = json.load(fh)
        threshold = float(metrics.get("threshold", threshold))

    preds = (scores >= threshold).astype(int)
    order = np.argsort(scores)[::-1]

    tp_candidates = [i for i in order if preds[i] == 1 and labels[i] == 1]
    fp_candidates = [i for i in order if preds[i] == 1 and labels[i] == 0]

    if tp_candidates:
        idx = tp_candidates[0]
        return int(node_ids[idx]), {
            "selection": "auto_true_positive",
            "score": float(scores[idx]),
            "label": int(labels[idx]),
        }
    if fp_candidates:
        idx = fp_candidates[0]
        return int(node_ids[idx]), {
            "selection": "auto_false_positive",
            "score": float(scores[idx]),
            "label": int(labels[idx]),
        }

    idx = order[0]
    return int(node_ids[idx]), {
        "selection": "auto_high_score",
        "score": float(scores[idx]),
        "label": int(labels[idx]),
    }


def run_gnn(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    cfg = find_run_config(run_dir)
    if cfg is None:
        raise FileNotFoundError("Could not locate config_used.yaml in GNN run directory.")

    data = prepare_data_for_gnn(cfg)
    device = torch.device("cpu")
    data = data.to(device)

    in_dim = data.x.size(1)
    model = build_gnn_model(cfg, in_dim).to(device)

    ckpt_path = run_dir / "best.ckpt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    node_id, selection_info = pick_node_to_explain(run_dir, args.node)

    wrapped_model = LogProbModel(model)
    explainer = GNNExplainer(wrapped_model, epochs=args.epochs, return_type="log_prob")

    node_feat_mask, edge_mask = explainer.explain_node(node_id, data.x, data.edge_index, target=1)

    _, ax = explainer.visualize_subgraph(node_id, data.edge_index, edge_mask, y=data.y)
    ax.set_title(f"GNNExplainer node {node_id} ({selection_info['selection']})")
    plt.tight_layout()
    plt.savefig(run_dir / f"gnn_explainer_node_{node_id}.png", dpi=200)
    plt.close()

    edge_mask_np = edge_mask.detach().cpu().numpy()
    edge_index_np = data.edge_index.cpu().numpy()
    top_edge_idx = np.argsort(edge_mask_np)[::-1][:20]
    edge_importance = [
        {
            "source": int(edge_index_np[0, i]),
            "target": int(edge_index_np[1, i]),
            "importance": float(edge_mask_np[i]),
        }
        for i in top_edge_idx
    ]

    feat_mask_np = node_feat_mask.detach().cpu().numpy()
    feature_names = [f"f{i}" for i in range(feat_mask_np.shape[0])]
    if cfg.get("use_time_scalar", False):
        feature_names[-1] = "time_scalar"

    feature_importance = [
        {"feature": feature_names[i], "importance": float(feat_mask_np[i])}
        for i in np.argsort(feat_mask_np)[::-1][: min(20, len(feature_names))]
    ]

    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()

    result = {
        "node_id": int(node_id),
        "selection": selection_info,
        "probability_illicit": float(probs[node_id]),
        "feature_importance": feature_importance,
        "edge_importance": edge_importance,
    }

    with open(run_dir / "gnn_explainer_importance.json", "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model explainability utilities")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    xgb_parser = subparsers.add_parser("xgb", help="Explain XGBoost baseline with SHAP")
    xgb_parser.add_argument("--run_dir", type=Path, required=True, help="Baseline run directory")
    xgb_parser.add_argument("--max_plots", type=int, default=20, help="Number of features to show in SHAP bar plot")
    xgb_parser.set_defaults(func=run_xgb)

    gnn_parser = subparsers.add_parser("gnn", help="Explain GNN predictions with GNNExplainer")
    gnn_parser.add_argument("--run_dir", type=Path, required=True, help="GNN run directory")
    gnn_parser.add_argument("--node", type=int, default=None, help="Node id to explain (defaults to auto-picked TP/FP)")
    gnn_parser.add_argument("--epochs", type=int, default=200, help="GNNExplainer optimization epochs")
    gnn_parser.set_defaults(func=run_gnn)

    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
