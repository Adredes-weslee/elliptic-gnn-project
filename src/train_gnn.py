# src/train_gnn.py
import argparse
import inspect
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.loader import NeighborLoader

from src.data.dataset_elliptic import make_temporal_masks
from src.models.gnn import GATNet, GCNNet, SAGENet, SAGEResBNNet
from src.utils.calibrate import TemperatureScaler
from src.utils.common import (
    ensure_dir,
    gpu_available,
    log_device_info,
    save_json,
    set_seed,
)
from src.utils.logger import RunLogger
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


def _autocast_ctx(device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled)
    else:
        return torch.cuda.amp.autocast(enabled=enabled)


def _make_grad_scaler(device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device=device.type, enabled=enabled)
    else:
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_cached(processed_dir):
    # Safe load for PyTorch>=2.6 (weights_only default True)
    path = os.path.join(processed_dir, "graph.pt")
    sig = inspect.signature(torch.load)
    kwargs = {"map_location": "cpu"}
    if "weights_only" in sig.parameters:
        kwargs["weights_only"] = False  # local trusted artifact
    try:
        data = torch.load(path, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {path}. Ensure graph was created with torch.save(Data(...)). "
            f"Original error: {e}"
        ) from e
    return data


def build_model(arch, in_dim, cfg):
    if arch == "gcn":
        return GCNNet(
            in_dim,
            hidden_dim=cfg["hidden_dim"],
            layers=cfg["layers"],
            dropout=cfg["dropout"],
        )
    elif arch == "sage":
        return SAGENet(
            in_dim,
            hidden_dim=cfg["hidden_dim"],
            layers=cfg["layers"],
            dropout=cfg["dropout"],
        )
    elif arch == "gat":
        return GATNet(
            in_dim,
            hidden_dim=cfg["hidden_dim"],
            layers=cfg["layers"],
            heads=cfg.get("heads", 4),
            dropout=cfg["dropout"],
        )
    elif arch in ("sage_resbn", "sage_bn", "sage_res"):
        return SAGEResBNNet(
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
        raise ValueError("Unknown arch")


def get_device(cfg):
    forced = cfg.get("device", "auto")
    if forced == "cuda" and gpu_available():
        return torch.device("cuda")
    if forced == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if gpu_available() else "cpu")


def class_weight(train_y):
    pos = (train_y == 1).sum().item()
    neg = (train_y == 0).sum().item()
    if pos == 0 or neg == 0:
        return torch.tensor([1.0, 1.0], dtype=torch.float32)
    w_pos = (pos + neg) / (2.0 * pos)
    w_neg = (pos + neg) / (2.0 * neg)
    return torch.tensor([w_neg, w_pos], dtype=torch.float32)


def _model_uses_time_embed(model):
    return hasattr(model, "time_embed_dim") and getattr(model, "time_embed_dim") > 0


# -------- time-aware loss weighting + optional embedding L2 --------
def _norm_train_time(t_vec, t_min, t_max):
    denom = max(float(t_max - t_min), 1.0)
    return (t_vec.float() - float(t_min)) / denom


def _make_loss_fn(cfg, cw, model, t_min, t_max):
    """
    Returns a callable: (logits, target, t_idx) -> scalar loss
    - focal_loss supported (cfg.focal_loss)
    - time weighting: cfg.time_loss_weighting in {'none','linear','sqrt'}
    - optional L2 on learned time embedding (cfg.time_embed_l2)
    """
    use_scheme = str(
        cfg.get("time_loss_weighting", "none")
    )  # "none" | "linear" | "sqrt"
    embed_l2 = float(cfg.get("time_embed_l2", 0.0))
    use_focal = bool(cfg.get("focal_loss", False))
    gamma = float(cfg.get("focal_gamma", 2.0))

    def loss_fn(logits, target, t_idx=None):
        # base per-sample loss (vector)
        if use_focal:
            ce = F.cross_entropy(logits, target, reduction="none")
            pt = torch.softmax(logits, dim=1)[
                torch.arange(len(target), device=logits.device), target
            ]
            loss_vec = ((1 - pt) ** gamma) * ce
        else:
            loss_vec = F.cross_entropy(
                logits, target, weight=cw.to(logits.device), reduction="none"
            )

        # optional time reweight
        if use_scheme != "none" and t_idx is not None:
            wt = _norm_train_time(t_idx, t_min, t_max).to(logits.device)
            if use_scheme == "sqrt":
                wt = torch.sqrt(torch.clamp(wt, min=0.0))
            elif use_scheme == "linear":
                pass  # already linear in [0,1]
            else:
                raise ValueError(f"unknown time_loss_weighting={use_scheme}")
            wt = torch.clamp(wt, min=1e-3)
            loss_vec = loss_vec * wt

        loss = loss_vec.mean()

        # optional L2 on learned time embedding matrix
        if embed_l2 > 0.0 and hasattr(model, "time_emb") and model.time_emb is not None:
            loss = loss + embed_l2 * (model.time_emb.weight.pow(2).mean())

        return loss

    return loss_fn


# ---------------- training loops ----------------
def train_epoch(
    model, data, edge_index, optimizer, loss_fn, scaler, use_amp, cfg, device
):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with _autocast_ctx(device, use_amp):
        logits = model(
            data.x, edge_index, data.timestep if _model_uses_time_embed(model) else None
        )
        t_idx = (
            data.timestep[data.train_mask]
            if cfg.get("time_loss_weighting", "none") != "none"
            else None
        )
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask], t_idx)
    scaler.scale(loss).backward()
    if cfg.get("grad_clip", 0) and cfg["grad_clip"] > 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.item())


def train_epoch_minibatch(
    model, loader, optimizer, loss_fn, scaler, use_amp, cfg, device
):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with _autocast_ctx(device, use_amp):
            logits = model(
                batch.x,
                batch.edge_index,
                batch.timestep if _model_uses_time_embed(model) else None,
            )
            target = batch.y[: batch.batch_size]
            t_idx = (
                batch.timestep[: batch.batch_size]
                if cfg.get("time_loss_weighting", "none") != "none"
                else None
            )
            loss = loss_fn(logits[: batch.batch_size], target, t_idx)
        scaler.scale(loss).backward()
        if cfg.get("grad_clip", 0) and cfg["grad_clip"] > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        total_loss += loss.item() * int(batch.batch_size)
        total_examples += int(batch.batch_size)
    if total_examples == 0:
        return 0.0
    return float(total_loss / total_examples)


@torch.no_grad()
def eval_split(model, data, edge_index, mask):
    model.eval()
    logits = model(
        data.x, edge_index, data.timestep if _model_uses_time_embed(model) else None
    )
    probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    y = data.y.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    return y[mask_np], probs[mask_np], logits


@torch.no_grad()
def eval_val_minibatch(model, loader, device):
    model.eval()
    y_batches, p_batches = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(
            batch.x,
            batch.edge_index,
            batch.timestep if _model_uses_time_embed(model) else None,
        )
        logits = logits[: batch.batch_size]
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu()
        y_batches.append(batch.y[: batch.batch_size].detach().cpu())
        p_batches.append(probs)
    if not y_batches:
        return np.array([]), np.array([])
    y = torch.cat(y_batches).numpy()
    probs = torch.cat(p_batches).numpy()
    return y, probs


def main(cfg):
    set_seed(cfg.get("seed", 42))
    outdir = os.path.join("outputs", "gnn", cfg["run_name"])
    ensure_dir(outdir)
    logger = RunLogger(outdir)

    device = get_device(cfg)
    log_device_info()
    print(f"[RUN] Using device: {device}")
    use_amp = (device.type == "cuda") and bool(cfg.get("amp", True))
    scaler = _make_grad_scaler(device, use_amp)

    data = load_cached(cfg["processed_dir"])
    if not hasattr(data, "train_mask"):
        raise RuntimeError(
            "Build graph first: python -m src.data.build_graph --config configs/split.yaml"
        )

    # Optional rolling window
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

    # Optional scalar time (kept for compatibility; disabled if embedding is on)
    if cfg.get("use_time_scalar", False) and cfg.get("time_embed_dim", 0) == 0:
        tnorm = (data.timestep.float() / float(data.timestep.max())).unsqueeze(1)
        data.x = torch.cat([data.x, tnorm], dim=1)

    # Edges
    edge_index_used = data.edge_index
    if cfg.get("symmetrize_edges", False):
        ei = torch.cat([edge_index_used, edge_index_used.flip(0)], dim=1)
        edge_index_used = ei
        data.edge_index = edge_index_used
    else:
        edge_index_used = data.edge_index

    # Mini-batch or full-batch
    use_mini_batch = bool(cfg.get("mini_batch", False))
    if use_mini_batch:
        train_idx = torch.where(data.train_mask)[0]
        val_idx = torch.where(data.val_mask)[0]
        fanout = cfg.get("fanout", [10, 10])
        batch_size = cfg.get("batch_size", 8192)
        train_loader = NeighborLoader(
            data,
            num_neighbors=fanout,
            batch_size=batch_size,
            input_nodes=train_idx,
            shuffle=True,
        )
        val_loader = NeighborLoader(
            data,
            num_neighbors=fanout,
            batch_size=batch_size,
            input_nodes=val_idx,
            shuffle=False,
        )
    else:
        data = data.to(device)
        train_loader, val_loader = None, None
        edge_index_used = data.edge_index

    # Model / Optimizer
    in_dim = data.x.size(1)
    model = build_model(cfg["arch"], in_dim, cfg).to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )

    # Class weights (for CE path)
    if cfg.get("class_weight_pos", "auto") == "auto":
        cw = class_weight(data.y[data.train_mask])
    else:
        cw = torch.tensor([1.0, float(cfg["class_weight_pos"])], dtype=torch.float32)

    # Train-time timestep bounds for weighting
    t_train = data.timestep[data.train_mask]
    t_min, t_max = int(t_train.min().item()), int(t_train.max().item())

    # Loss (focal/CE + optional time weighting + embed L2)
    loss_fn = _make_loss_fn(cfg, cw, model, t_min, t_max)

    # Train
    best_val = -1.0
    best_state = None
    patience = cfg.get("patience", 20)
    bad = 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        if use_mini_batch:
            loss = train_epoch_minibatch(
                model, train_loader, opt, loss_fn, scaler, use_amp, cfg, device
            )
            y_val, p_val = eval_val_minibatch(model, val_loader, device)
        else:
            loss = train_epoch(
                model, data, edge_index_used, opt, loss_fn, scaler, use_amp, cfg, device
            )
            y_val, p_val, _ = eval_split(model, data, edge_index_used, data.val_mask)

        pr_val = (
            0.0 if y_val.size == 0 else pr_auc_illicit((y_val == 1).astype(int), p_val)
        )
        RunLogger(outdir).log_epoch(epoch, loss, pr_val)  # quick log

        if pr_val > best_val:
            best_val = pr_val
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad = 0
        else:
            bad += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | loss {loss:.4f} | val PR-AUC(illicit) {pr_val:.4f} (best {best_val:.4f})"
            )

        if bad >= patience:
            print("Early stopping.")
            break

    # Load best
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    if use_mini_batch:
        data = data.to(device)
        edge_index_used = data.edge_index

    # Optional temperature scaling (old API: divide logits by learned T)
    ts = None
    use_temp_scale = bool(cfg.get("calibrate_temperature", True))
    if use_temp_scale:
        _, _, logits_val = eval_split(model, data, edge_index_used, data.val_mask)
        ts = TemperatureScaler().to(device)
        _ = ts.fit(logits_val[data.val_mask].to(device), data.y[data.val_mask])

    def get_probs(edge_index_eval):
        model.eval()
        with torch.no_grad():
            logits = model(
                data.x,
                edge_index_eval,
                data.timestep if _model_uses_time_embed(model) else None,
            )
            if use_temp_scale and ts is not None:
                logits = logits / ts.T
            probs = torch.softmax(logits, dim=1)[:, 1]
        return probs.detach().cpu().numpy(), logits

    # Final eval
    probs, logits = get_probs(edge_index_used)
    y_np = data.y.detach().cpu().numpy()
    val_mask = data.val_mask.detach().cpu().numpy()
    test_mask = data.test_mask.detach().cpu().numpy()
    timestep_np = data.timestep.detach().cpu().numpy()

    y_val, p_val = y_np[val_mask], probs[val_mask]
    y_te, p_te = y_np[test_mask], probs[test_mask]

    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    np.save(os.path.join(outdir, "scores_val.npy"), p_val)
    np.save(os.path.join(outdir, "y_val.npy"), y_val)
    np.save(os.path.join(outdir, "node_idx_val.npy"), val_indices)
    np.save(os.path.join(outdir, "timestep_val.npy"), timestep_np[val_mask])
    np.save(os.path.join(outdir, "scores_test.npy"), p_te)
    np.save(os.path.join(outdir, "y_test.npy"), y_te)
    np.save(os.path.join(outdir, "node_idx_test.npy"), test_indices)
    np.save(os.path.join(outdir, "timestep_test.npy"), timestep_np[test_mask])

    # Threshold selection
    if cfg.get("use_val_for_thresholds", True):
        if cfg.get("precision_target", 0.0) and cfg["precision_target"] > 0:
            thr = pick_threshold_for_precision(
                (y_val == 1).astype(int), p_val, cfg["precision_target"]
            )
        else:
            thr, _ = pick_threshold_max_f1((y_val == 1).astype(int), p_val)
    else:
        thr, _ = pick_threshold_max_f1((y_te == 1).astype(int), p_te)

    # Metrics on test
    y_bin = (y_te == 1).astype(int)
    pr = pr_auc_illicit(y_bin, p_te)
    roc = roc_auc_illicit(y_bin, p_te)
    f1 = f1_at_threshold(y_bin, p_te, thr)
    p_at_k = precision_at_k(y_bin, p_te, cfg.get("topk", 100))
    rec_at_p = recall_at_precision(y_bin, p_te, cfg.get("precision_target", 0.90))
    ece = expected_calibration_error(y_bin, p_te)

    metrics = dict(
        pr_auc_illicit=pr,
        roc_auc=roc,
        f1_illicit_at_thr=f1,
        threshold=thr,
        precision_at_k=p_at_k,
        recall_at_precision=rec_at_p,
        ece=ece,
        n_test=int(len(y_te)),
        best_val_pr_auc=best_val,
    )

    # -------- per-timestep PR-AUC on test --------
    test_ts = timestep_np[test_mask]
    if test_ts.size > 0:
        # preserve chronological order
        uniq = sorted(set(int(t) for t in test_ts.tolist()))
        pr_by_t = []
        for t in uniq:
            idx = test_ts == t
            if idx.sum() == 0:
                pr_by_t.append(float("nan"))
            else:
                pr_by_t.append(pr_auc_illicit((y_te[idx] == 1).astype(int), p_te[idx]))

        def _avg_tail(arr, k):
            return float(sum(arr[-k:]) / k) if len(arr) >= k else None

        metrics["test_pr_auc_by_time"] = pr_by_t
        if pr_by_t:
            metrics["pr_auc_last1"] = float(pr_by_t[-1])
            if len(pr_by_t) >= 3:
                metrics["pr_auc_last3"] = _avg_tail(pr_by_t, 3)
            if len(pr_by_t) >= 5:
                metrics["pr_auc_last5"] = _avg_tail(pr_by_t, 5)

    ensure_dir(outdir)
    torch.save(model.state_dict(), os.path.join(outdir, "best.ckpt"))
    save_json(os.path.join(outdir, "metrics.json"), metrics)

    # Optional hub ablation (unchanged)
    frac = float(cfg.get("ablate_hubs_frac", 0.0))
    if frac > 0:
        num_nodes = data.num_nodes
        num_hubs = int(frac * float(num_nodes))
        edge_index_cpu = edge_index_used.detach().cpu()
        deg_src = torch.bincount(edge_index_cpu[0], minlength=num_nodes)
        deg_dst = torch.bincount(edge_index_cpu[1], minlength=num_nodes)
        deg = deg_src + deg_dst
        hubs = torch.zeros(num_nodes, dtype=torch.bool)
        if num_hubs > 0:
            topk = torch.topk(deg, num_hubs)
            hubs[topk.indices] = True
        edge_mask = ~(hubs[edge_index_cpu[0]] | hubs[edge_index_cpu[1]])
        edge_index_abl = edge_index_cpu[:, edge_mask].to(edge_index_used.device)
        probs_abl, _ = get_probs(edge_index_abl)
        p_te_abl = probs_abl[test_mask]
        y_bin = (y_te == 1).astype(int)
        metrics_hub_removed = dict(
            pr_auc_illicit=pr_auc_illicit(y_bin, p_te_abl),
            roc_auc=roc_auc_illicit(y_bin, p_te_abl),
            f1_illicit_at_thr=f1_at_threshold(y_bin, p_te_abl, thr),
            threshold=thr,
            precision_at_k=precision_at_k(y_bin, p_te_abl, cfg.get("topk", 100)),
            recall_at_precision=recall_at_precision(
                y_bin, p_te_abl, cfg.get("precision_target", 0.90)
            ),
            ece=expected_calibration_error(y_bin, p_te_abl),
            n_test=int(len(y_te)),
            n_hubs=int(num_hubs),
            hub_fraction=frac,
            n_edges_remaining=int(edge_index_abl.size(1)),
        )
        save_json(os.path.join(outdir, "metrics_hub_removed.json"), metrics_hub_removed)

    with open(os.path.join(outdir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    RunLogger(outdir).close()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
