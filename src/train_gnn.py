
import os, argparse, yaml, json, numpy as np, torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from src.utils.common import ensure_dir, save_json, set_seed, log_device_info, gpu_available
from src.utils.metrics import (
    pr_auc_illicit, roc_auc_illicit, precision_at_k, pick_threshold_max_f1,
    pick_threshold_for_precision, f1_at_threshold, expected_calibration_error, recall_at_precision
)
from src.utils.calibrate import TemperatureScaler
from src.models.gnn import GCNNet, SAGENet, GATNet
from src.utils.logger import RunLogger

def load_cached(processed_dir):
    data = torch.load(os.path.join(processed_dir, "graph.pt"))
    return data

def build_model(arch, in_dim, cfg):
    if arch == "gcn":
        return GCNNet(in_dim, hidden_dim=cfg["hidden_dim"], layers=cfg["layers"], dropout=cfg["dropout"])
    elif arch == "sage":
        return SAGENet(in_dim, hidden_dim=cfg["hidden_dim"], layers=cfg["layers"], dropout=cfg["dropout"])
    elif arch == "gat":
        return GATNet(in_dim, hidden_dim=cfg["hidden_dim"], layers=cfg["layers"], heads=cfg.get("heads",4), dropout=cfg["dropout"])
    else:
        raise ValueError("Unknown arch")

def get_device(cfg):
    import torch
    forced = cfg.get("device","auto")
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
    # weights inversely proportional to frequency
    w_pos = (pos + neg) / (2.0 * pos)
    w_neg = (pos + neg) / (2.0 * neg)
    return torch.tensor([w_neg, w_pos], dtype=torch.float32)

def train_epoch(model, data, optimizer, loss_fn, scaler, use_amp, cfg):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=use_amp):
        logits = model(data.x, data.edge_index)
        loss = loss_fn(logits[data.train_mask], data.y[data.train_mask])
    scaler.scale(loss).backward()
    if cfg.get("grad_clip", 0) and cfg["grad_clip"] > 0:
        scaler.unscale_(optimizer)
        import torch as _torch
        _torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return float(loss.item())

def train_epoch_minibatch(model, loader, optimizer, loss_fn, scaler, use_amp, cfg, device):
    model.train()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(batch.x, batch.edge_index)
            target = batch.y[: batch.batch_size]
            loss = loss_fn(logits[: batch.batch_size], target)
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
def eval_split(model, data, mask):
    model.eval()
    logits = model(data.x, data.edge_index)
    probs = torch.softmax(logits, dim=1)[:,1].detach().cpu().numpy()
    y = data.y.detach().cpu().numpy()
    return y[mask.cpu().numpy()], probs[mask.cpu().numpy()], logits

@torch.no_grad()
def eval_val_minibatch(model, loader, device):
    model.eval()
    y_batches = []
    p_batches = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    data = load_cached(cfg["processed_dir"])
    if not hasattr(data, "train_mask"):
        raise RuntimeError("Build graph first: python -m src.data.build_graph --config configs/split.yaml")

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
        train_loader = None
        val_loader = None

    in_dim = data.x.size(1)
    model = build_model(cfg["arch"], in_dim, cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])

    # Loss
    if cfg.get("focal_loss", False):
        gamma = 2.0
        def focal_loss(logits, target):
            ce = F.cross_entropy(logits, target, reduction="none")
            pt = torch.softmax(logits, dim=1)[torch.arange(len(target)), target]
            loss = ((1-pt)**gamma) * ce
            return loss.mean()
        loss_fn = lambda logits, target: focal_loss(logits, target)
    else:
        if cfg.get("class_weight_pos","auto") == "auto":
            cw = class_weight(data.y[data.train_mask])
        else:
            cw = torch.tensor([1.0, float(cfg["class_weight_pos"])], dtype=torch.float32)
        loss_fn = lambda logits, target: F.cross_entropy(logits, target, weight=cw.to(logits.device))

    best_val = -1.0
    best_state = None
    patience = cfg.get("patience", 20)
    bad = 0

    for epoch in range(1, cfg["max_epochs"]+1):
        if use_mini_batch:
            loss = train_epoch_minibatch(model, train_loader, opt, loss_fn, scaler, use_amp, cfg, device)
            y_val, p_val = eval_val_minibatch(model, val_loader, device)
        else:
            loss = train_epoch(model, data, opt, loss_fn, scaler, use_amp, cfg)
            y_val, p_val, _ = eval_split(model, data, data.val_mask)
        if y_val.size == 0:
            pr_val = 0.0
        else:
            pr_val = pr_auc_illicit((y_val==1).astype(int), p_val)

        logger.log_epoch(epoch, loss, pr_val)

        if pr_val > best_val:
            best_val = pr_val
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | loss {loss:.4f} | val PR-AUC(illicit) {pr_val:.4f} (best {best_val:.4f})")

        if bad >= patience:
            print("Early stopping.")
            break

    # Load best
    if best_state is not None:
        model.load_state_dict({k:v.to(device) for k,v in best_state.items()})

    if use_mini_batch:
        data = data.to(device)

    # Optionally temperature-scale on validation logits
    if cfg.get("calibrate_temperature", True):
        y_val, _, logits_val = eval_split(model, data, data.val_mask)
        ts = TemperatureScaler().to(device)
        _ = ts.fit(logits_val[data.val_mask].to(device), data.y[data.val_mask])
        def predict_probs():
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                logits = logits / ts.T
                probs = torch.softmax(logits, dim=1)[:,1]
            return probs.detach().cpu().numpy(), logits
        get_probs = predict_probs
    else:
        def predict_probs():
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                probs = torch.softmax(logits, dim=1)[:,1]
            return probs.detach().cpu().numpy(), logits
        get_probs = predict_probs

    # Final eval
    probs, logits = get_probs()
    y_np = data.y.detach().cpu().numpy()
    val_mask = data.val_mask.detach().cpu().numpy()
    test_mask = data.test_mask.detach().cpu().numpy()

    y_val, p_val = y_np[val_mask], probs[val_mask]
    y_te, p_te = y_np[test_mask], probs[test_mask]

    # Threshold selection on val
    if cfg.get("use_val_for_thresholds", True):
        if cfg.get("precision_target",0.0) and cfg["precision_target"] > 0:
            thr = pick_threshold_for_precision((y_val==1).astype(int), p_val, cfg["precision_target"])
        else:
            thr, _ = pick_threshold_max_f1((y_val==1).astype(int), p_val)
    else:
        thr, _ = pick_threshold_max_f1((y_te==1).astype(int), p_te)

    # Metrics on test
    y_bin = (y_te==1).astype(int)
    pr = pr_auc_illicit(y_bin, p_te)
    roc = roc_auc_illicit(y_bin, p_te)
    f1 = f1_at_threshold(y_bin, p_te, thr)
    p_at_k = precision_at_k(y_bin, p_te, cfg.get("topk",100))
    rec_at_p = recall_at_precision(y_bin, p_te, cfg.get("precision_target",0.90))
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

    ensure_dir(outdir)
    torch.save(model.state_dict(), os.path.join(outdir, "best.ckpt"))
    save_json(os.path.join(outdir, "metrics.json"), metrics)
    with open(os.path.join(outdir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    logger.close()
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
