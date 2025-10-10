# src/analysis/explain.py
"""Model explainability utilities for baselines and GNNs.

Subcommands:
  - xgb : explain an XGBoost baseline with SHAP.
  - gnn : explain a trained GNN with GNNExplainer.

Robust across torch_geometric versions:
- Uses the old `GNNExplainer(model=..., return_type=...)` API if available.
- Otherwise uses the new `Explainer(algorithm=..., model_config=...)` API.
  On new API, it supports BOTH:
    * explainer.explain_node(...)
    * explainer(x=..., edge_index=..., index=..., target=...)
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx
from xgboost import XGBClassifier

plt.switch_backend("Agg")

# Optional at runtime, required only for xgb subcommand
try:
    import shap  # type: ignore
except ImportError as exc:  # pragma: no cover
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


# ---------------- Common helpers ----------------
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
    except Exception:  # pragma: no cover
        return "hist"


def load_processed_graph(processed_dir: Path) -> Data:
    """Load processed graph, handling PyTorch>=2.6 weights_only default safely."""
    graph_path = processed_dir / "graph.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Processed graph not found: {graph_path}")

    kwargs = {"map_location": "cpu"}
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        kwargs["weights_only"] = False  # trusted local artifact

    try:
        data = torch.load(graph_path, **kwargs)
    except Exception as e:
        raise RuntimeError(
            "Failed to load processed graph. If you trust this file, call torch.load with "
            "weights_only=False (PyTorch 2.6+ default changed). "
            f"Original error: {type(e).__name__}: {e}"
        ) from e

    if not isinstance(data, Data):
        raise TypeError(f"Expected torch_geometric.data.Data, got {type(data)!r}")
    return data


def labeled_split_arrays(
    data: Data,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    x = data.x.cpu().numpy()
    y = data.y.cpu().numpy()
    timestep = data.timestep.cpu().numpy()
    labeled_idx = np.where(y >= 0)[0]

    feats = x[labeled_idx]
    labels = y[labeled_idx]
    t = timestep[labeled_idx]

    if not hasattr(data, "train_mask") or not hasattr(data, "val_mask"):
        raise RuntimeError(
            "Graph is missing train/val masks. Build data with src.data.build_graph first."
        )

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


def apply_train_window(
    train_mask: np.ndarray, timesteps: np.ndarray, window_k: Optional[int]
) -> np.ndarray:
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


# ---------------- XGB + SHAP ----------------
def ensure_shap_import() -> None:
    if shap is None:
        raise RuntimeError(
            "shap is required for xgb explanations but is not installed"
        ) from _SHAP_IMPORT_ERROR


def load_or_fit_xgb(run_dir: Path, cfg: Optional[Dict]) -> XGBArtifacts:
    ensure_shap_import()

    processed_dir = (
        Path(cfg.get("processed_dir", "data/processed"))
        if cfg
        else Path("data/processed")
    )
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


def summarize_shap_values(
    shap_values: np.ndarray, feature_names: List[str], top_k: int = 20
) -> List[Dict[str, float]]:
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

    explainer = shap.TreeExplainer(artifacts.model)  # type: ignore[arg-type]
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


# ---------------- GNN + GNNExplainer ----------------
class ProbModel(torch.nn.Module):
    """Wrapper to optionally pass timestep to the model.

    IMPORTANT:
      * Returns a SINGLE LOGIT per node for binary tasks if the base model outputs 2 logits.
      * That makes PyG's binary loss happy across versions of the new Explainer API.
    """

    def __init__(self, model: torch.nn.Module, t_idx: Optional[torch.Tensor] = None):
        super().__init__()
        self.model = model
        self.t_idx = t_idx
        self._use_time = (
            hasattr(model, "time_embed_dim") and getattr(model, "time_embed_dim", 0) > 0
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Get raw logits from the original model
        if self._use_time and self.t_idx is not None:
            logits = self.model(x, edge_index, self.t_idx)  # shape [N, C]
        else:
            logits = self.model(x, edge_index)  # shape [N, C]

        # If the base model is binary with 2 logits, convert to ONE binary logit:
        #   sigmoid(z_pos - z_neg) == softmax([z_neg, z_pos])_pos
        if logits.dim() == 2 and logits.size(-1) == 2:
            z = logits[:, 1] - logits[:, 0]  # shape [N]
            return z.unsqueeze(1)  # shape [N, 1] -> y_hat[index] -> [1]

        # Otherwise, leave as-is (e.g., already single-logit or multiclass)
        return logits


def build_gnn_model(cfg: Dict, in_dim: int) -> torch.nn.Module:
    from src.models.gnn import GATNet, GCNNet, SAGENet, SAGEResBNNet

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
        model = GATNet(
            in_dim, hidden_dim=hidden_dim, layers=layers, heads=heads, dropout=dropout
        )
    elif arch in ("sage_resbn", "sage_bn", "sage_res"):
        model = SAGEResBNNet(
            in_dim,
            hidden_dim=cfg.get("hidden_dim", 128),
            layers=cfg.get("layers", 3),
            dropout=cfg.get("dropout", 0.2),
            num_classes=2,
            use_bn=cfg.get("use_bn", True),
            residual=cfg.get("residual", True),
            time_embed_dim=cfg.get("time_embed_dim", 0),
            time_embed_type=cfg.get("time_embed_type", "learned"),
            max_timestep=cfg.get("max_timestep", 49),
        )
    else:
        raise ValueError(f"Unknown GNN architecture: {cfg['arch']}")
    return model


def maybe_add_time_feature(data: Data, cfg: Dict) -> Data:
    # Mirror training behavior: only add scalar time if no embedding is used.
    if cfg.get("use_time_scalar", False) and int(cfg.get("time_embed_dim", 0)) == 0:
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


def pick_node_to_explain(
    run_dir: Path, node_override: Optional[int]
) -> Tuple[int, Dict[str, float]]:
    scores_path = run_dir / "scores_test.npy"
    labels_path = run_dir / "y_test.npy"
    node_idx_path = run_dir / "node_idx_test.npy"
    if not (scores_path.exists() and labels_path.exists() and node_idx_path.exists()):
        raise FileNotFoundError(
            "Required test score artifacts not found in run directory."
        )

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


# -------- Explainer builder & robust calling --------
def _build_explainer_robust(wrapped_model, epochs: int):
    """Return (api_kind, explainer_or_cls, expl_kwargs)."""
    # Find GNNExplainer
    GNNExplainer = None
    try:
        from torch_geometric.nn import GNNExplainer as _GNNExplainer

        GNNExplainer = _GNNExplainer
    except Exception:
        try:
            from torch_geometric.nn.models import GNNExplainer as _GNNExplainer

            GNNExplainer = _GNNExplainer
        except Exception:
            try:
                from torch_geometric.explain import GNNExplainer as _GNNExplainer

                GNNExplainer = _GNNExplainer
            except Exception as e:
                raise ImportError(
                    "GNNExplainer is not available in your torch_geometric installation."
                ) from e

    # OLD API? Constructor accepts 'model'
    sig = inspect.signature(GNNExplainer.__init__)
    param_names = [p for p in sig.parameters.keys() if p not in ("self",)]
    if "model" in param_names or (len(param_names) > 0 and param_names[0] == "model"):
        explainer = GNNExplainer(wrapped_model, epochs=epochs, return_type="raw")
        return "old", explainer, None

    # NEW API: algorithm + Explainer + ModelConfig
    from torch_geometric.explain import Explainer as _Explainer

    algo = GNNExplainer(epochs=epochs)

    expl_kwargs = dict(model=wrapped_model, algorithm=algo)
    expl_sig = inspect.signature(_Explainer.__init__)
    if "node_mask_type" in expl_sig.parameters:
        expl_kwargs["node_mask_type"] = "attributes"
    if "edge_mask_type" in expl_sig.parameters:
        expl_kwargs["edge_mask_type"] = "object"
    if "explanation_type" in expl_sig.parameters:
        expl_kwargs["explanation_type"] = "model"  # avoid passing target

    # Build ModelConfig robustly
    try:
        from torch_geometric.explain import ModelConfig as _ModelConfig  # type: ignore

        try:
            from torch_geometric.explain.config import (
                ModelMode,
                ModelReturnType,
                ModelTaskLevel,
            )
        except Exception:
            ModelMode = getattr(_ModelConfig, "ModelMode", None)  # type: ignore
            ModelReturnType = getattr(_ModelConfig, "ModelReturnType", None)  # type: ignore
            ModelTaskLevel = getattr(_ModelConfig, "ModelTaskLevel", None)  # type: ignore

        def _values(enum_cls, defaults):
            try:
                return {e.value for e in enum_cls}
            except Exception:
                return set(defaults)

        modes = _values(
            ModelMode, ["binary_classification", "classification", "regression"]
        )
        returns = _values(
            ModelReturnType, ["logits", "raw", "log_prob", "probabilities", "prob"]
        )
        tasks = _values(ModelTaskLevel, ["node", "edge", "graph"])

        # Either mode works now that wrapped_model emits one logit in the 2-class case.
        mode_str = (
            "binary_classification"
            if "binary_classification" in modes
            else ("classification" if "classification" in modes else next(iter(modes)))
        )
        ret_str = next(
            (
                r
                for r in ["logits", "raw", "log_prob", "probabilities", "prob"]
                if r in returns
            ),
            next(iter(returns)),
        )
        task_str = "node" if "node" in tasks else next(iter(tasks))

        try:
            model_config = _ModelConfig(
                mode=ModelMode(mode_str),
                task_level=ModelTaskLevel(task_str),
                return_type=ModelReturnType(ret_str),
            )
        except Exception:
            model_config = _ModelConfig(
                mode=mode_str, task_level=task_str, return_type=ret_str
            )

        if "model_config" in expl_sig.parameters:
            expl_kwargs["model_config"] = model_config
    except Exception:
        # Fallback config
        if "model_config" in expl_sig.parameters:
            expl_kwargs["model_config"] = dict(
                mode="binary_classification",
                task_level="node",
                return_type="logits",
            )

    return "new", _Explainer, expl_kwargs


def _run_explainer_node(explainer, data: Data, node_id: int):
    """Call the explainer in a version-robust way and return an Explanation-like object."""
    # Prefer calling WITHOUT target for explanation_type='model' (new API)
    if hasattr(explainer, "explain_node"):
        try:
            return explainer.explain_node(node_id, x=data.x, edge_index=data.edge_index)
        except TypeError:
            # Old signatures may require target
            return explainer.explain_node(
                node_id, x=data.x, edge_index=data.edge_index, target=1
            )
    else:
        try:
            return explainer(x=data.x, edge_index=data.edge_index, index=node_id)
        except TypeError:
            return explainer(
                x=data.x, edge_index=data.edge_index, index=node_id, target=1
            )


def _unpack_masks_from_explanation(explanation, data: Data):
    """Return (node_feat_mask, edge_mask) tensors from various Explanation flavors."""
    # Edge mask
    edge_mask = None
    for name in ("edge_mask", "edge_mask_logits", "mask"):
        if hasattr(explanation, name):
            edge_mask = getattr(explanation, name)
            break
    if edge_mask is None:
        raise RuntimeError("GNNExplainer did not return an edge_mask.")

    # Node feature mask
    node_feat_mask = None
    for name in ("node_feat_mask", "feature_mask", "feat_mask", "x_mask", "node_mask"):
        if hasattr(explanation, name):
            node_feat_mask = getattr(explanation, name)
            break
    if node_feat_mask is None:
        node_feat_mask = torch.zeros(
            data.x.size(1), dtype=torch.float, device=edge_mask.device
        )

    return node_feat_mask, edge_mask


# ---------------- Run GNN explain ----------------
def run_gnn(args: argparse.Namespace) -> None:
    run_dir = args.run_dir
    cfg = find_run_config(run_dir)
    if cfg is None:
        raise FileNotFoundError(
            "Could not locate config_used.yaml in GNN run directory."
        )

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

    wrapped_model = ProbModel(
        model,
        t_idx=data.timestep
        if (
            hasattr(model, "time_embed_dim") and getattr(model, "time_embed_dim", 0) > 0
        )
        else None,
    )

    api_kind, explainer_or_cls, expl_kwargs = _build_explainer_robust(
        wrapped_model, args.epochs
    )

    # Explain node
    if api_kind == "old":
        explainer = explainer_or_cls  # instance
        node_feat_mask, edge_mask = explainer.explain_node(
            node_id, data.x, data.edge_index, target=1
        )
        png_path = run_dir / f"gnn_explainer_node_{node_id}.png"
        _save_subgraph_png(node_id, data, edge_mask, png_path, k=2)
        # Optional viz if available
        try:
            _, ax = explainer.visualize_subgraph(  # type: ignore[attr-defined]
                node_id, data.edge_index, edge_mask, y=data.y
            )
            ax.set_title(f"GNNExplainer node {node_id} ({selection_info['selection']})")
            plt.tight_layout()
            plt.savefig(run_dir / f"gnn_explainer_node_{node_id}.png", dpi=200)
            plt.close()
        except Exception:
            pass
    else:
        # Construct Explainer and call robustly
        ExplainerCls = explainer_or_cls  # class
        explainer = ExplainerCls(**expl_kwargs)  # type: ignore[misc]
        explanation = _run_explainer_node(explainer, data, node_id)
        node_feat_mask, edge_mask = _unpack_masks_from_explanation(explanation, data)
        png_path = run_dir / f"gnn_explainer_node_{node_id}.png"
        _save_subgraph_png(node_id, data, edge_mask, png_path, k=2)
        # Best-effort visualization
        try:
            if hasattr(explanation, "visualize_subgraph"):
                _, ax = explanation.visualize_subgraph(node_id, y=data.y)
            else:
                ax = None
            if ax is not None:
                ax.set_title(
                    f"GNNExplainer node {node_id} ({selection_info['selection']})"
                )
                plt.tight_layout()
                plt.savefig(run_dir / f"gnn_explainer_node_{node_id}.png", dpi=200)
                plt.close()
        except Exception:
            pass

    # -------- Export importances (robust to 1D or 2D feature masks) --------
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

    # node_feat_mask can be:
    #  - 1D: [F]
    #  - 2D: [N, F]  (most new-API cases)
    #  - weird shapes: collapse to [F]
    feat_mask_np = node_feat_mask.detach().cpu().numpy()
    if feat_mask_np.ndim == 1:
        feat_vec = feat_mask_np
    elif feat_mask_np.ndim == 2:
        # aggregate across nodes
        feat_vec = np.mean(np.abs(feat_mask_np), axis=0)
    else:
        # e.g., [something, F] -> flatten nodes/anything and average on last dim
        feat_vec = np.reshape(feat_mask_np, (-1, feat_mask_np.shape[-1]))
        feat_vec = np.mean(np.abs(feat_vec), axis=0)

    # Make feature names match the feature dimension (last dim)
    num_feats = int(feat_vec.shape[0])
    feature_names = [f"f{i}" for i in range(num_feats)]
    if cfg.get("use_time_scalar", False) and int(cfg.get("time_embed_dim", 0)) == 0:
        # last column is the appended normalized timestep
        feature_names[-1] = "time_scalar"

    order = np.argsort(feat_vec)[::-1][: min(20, num_feats)]
    feature_importance = [
        {"feature": feature_names[int(i)], "importance": float(feat_vec[int(i)])}
        for i in order
    ]

    # NOTE: Use the ORIGINAL model (2 logits) to compute probabilities for the report
    with torch.no_grad():
        if hasattr(model, "time_embed_dim") and getattr(model, "time_embed_dim", 0) > 0:
            logits = model(data.x, data.edge_index, data.timestep)
        else:
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


# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Model explainability utilities")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    xgb_parser = subparsers.add_parser("xgb", help="Explain XGBoost baseline with SHAP")
    xgb_parser.add_argument(
        "--run_dir", type=Path, required=True, help="Baseline run directory"
    )
    xgb_parser.add_argument(
        "--max_plots",
        type=int,
        default=20,
        help="Number of features to show in SHAP bar plot",
    )
    xgb_parser.set_defaults(func=run_xgb)

    gnn_parser = subparsers.add_parser(
        "gnn", help="Explain GNN predictions with GNNExplainer"
    )
    gnn_parser.add_argument(
        "--run_dir", type=Path, required=True, help="GNN run directory"
    )
    gnn_parser.add_argument(
        "--node",
        type=int,
        default=None,
        help="Node id to explain (defaults to auto-picked TP/FP)",
    )
    gnn_parser.add_argument(
        "--epochs", type=int, default=200, help="GNNExplainer optimization epochs"
    )
    gnn_parser.set_defaults(func=run_gnn)

    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    return args


def _save_subgraph_png(node_id, data, edge_mask, out_path, k=2):
    try:
        import networkx as nx  # optional
    except ImportError:
        print("[WARN] networkx not installed; skipping PNG")
        return

    # Ensure parent folder exists (e.g., run_dir)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Build k-hop subgraph and a boolean mask over *global* edges
    subset, sub_edge_index, _, edge_mask_global_bool = k_hop_subgraph(
        node_id, k, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes
    )

    # Align explainer edge scores to the subgraph edges
    # (filter the global edge_mask by the boolean selector returned above)
    sub_edge_scores = edge_mask[edge_mask_global_bool].detach().cpu().numpy()
    if sub_edge_scores.ndim > 1:
        sub_edge_scores = np.squeeze(sub_edge_scores)

    # Construct a NetworkX graph from the subgraph
    G = to_networkx(
        Data(num_nodes=subset.size(0), edge_index=sub_edge_index),
        to_undirected=True,
    )

    # Normalize widths by score (optional, but looks nicer)
    if sub_edge_scores.size > 0:
        s = sub_edge_scores - sub_edge_scores.min()
        denom = s.max() if s.max() > 0 else 1.0
        widths = (0.5 + 3.0 * (s / denom)).tolist()
    else:
        widths = [0.5] * sub_edge_index.size(1)

    # Center node in red; others default
    node_colors = [
        "#ff6b6b" if i == 0 else "#1f77b4" for i in range(G.number_of_nodes())
    ]

    pos = nx.spring_layout(G, seed=0)

    plt.figure(figsize=(7, 6))
    nx.draw(
        G,
        pos=pos,
        with_labels=False,
        node_size=60,
        node_color=node_colors,
        width=widths,
    )
    plt.title(f"GNNExplainer subgraph for node {node_id}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
