import argparse
import inspect
import json
import os

import numpy as np
import torch
import xgboost as xgb
import yaml
from packaging import version as _pkg_version
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.utils.common import ensure_dir, save_json, set_seed
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


def pick_tree_method():
    """Legacy helper (kept for config compatibility)."""
    try:
        return "gpu_hist" if torch.cuda.is_available() else "hist"
    except Exception:
        return "hist"


def pick_xgb_params():
    """Handle XGBoost >=2.0 device API vs older versions; avoid deprecation warnings."""
    ver = _pkg_version.parse(xgb.__version__)
    use_cuda = torch.cuda.is_available()
    params = {}
    if ver >= _pkg_version.parse("2.0.0"):
        params["tree_method"] = "hist"
        params["device"] = "cuda" if use_cuda else "cpu"
        # Do NOT set predictor explicitly (avoids 'not used' warning; auto-selected)
    else:
        params["tree_method"] = "gpu_hist" if use_cuda else "hist"
    return params


def make_calibrator(calibration: str, p_val, y_val):
    from src.utils.calibrate import calibrate_isotonic, calibrate_platt

    calibration = (calibration or "none").lower()
    if calibration == "isotonic":
        cal = calibrate_isotonic(p_val, y_val)
        return cal, (lambda s: cal.transform(s))
    if calibration == "platt":
        cal = calibrate_platt(p_val, y_val)
        return cal, (lambda s: cal.predict_proba(s.reshape(-1, 1))[:, 1])
    return None, (lambda s: s)


def load_cached(processed_dir):
    """Load processed graph.pt handling PyTorch>=2.6 weights_only default."""
    path = os.path.join(processed_dir, "graph.pt")
    sig = inspect.signature(torch.load)
    kwargs = {"map_location": "cpu"}
    if "weights_only" in sig.parameters:
        kwargs["weights_only"] = False  # trusted local artifact
    try:
        return torch.load(path, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {path}. Ensure it was created with torch.save(Data(...)). "
            f"Original error: {e}"
        ) from e


def get_split_arrays(data):
    """Extract labeled node features/labels & derive temporal train/val/test masks."""
    x = data.x.numpy()
    y = data.y.numpy()
    labeled_idx = np.where(y >= 0)[0]
    feats = x[labeled_idx]
    labels = y[labeled_idx]
    t = data.timestep.numpy()[labeled_idx]

    t_train_end = int(torch.max(data.timestep[data.train_mask]).item())
    t_val_end = int(torch.max(data.timestep[data.val_mask]).item())

    train = t <= t_train_end
    val = (t > t_train_end) & (t <= t_val_end)
    test = t > t_val_end
    return feats, labels, train, val, test, labeled_idx, t


def main(cfg):
    set_seed(cfg.get("seed", 42))
    outdir = os.path.join("outputs", "baselines", cfg["run_name"])
    ensure_dir(outdir)

    data = load_cached(cfg["processed_dir"])
    X, y, train_mask, val_mask, test_mask, labeled_idx, timesteps = get_split_arrays(
        data
    )

    window_k = cfg.get("train_window_k")
    if window_k is not None:
        window_k = int(window_k)
        train_timesteps = timesteps[train_mask]
        if train_timesteps.size == 0:
            raise RuntimeError("Train mask is empty; cannot apply rolling window.")
        T = int(train_timesteps.max())
        t_lo = max(1, T - window_k + 1)
        in_window = (timesteps >= t_lo) & (timesteps <= T)
        train_mask = train_mask & in_window

    Xtr, ytr = X[train_mask], y[train_mask]
    Xva, yva = X[val_mask], y[val_mask]
    Xte, yte = X[test_mask], y[test_mask]

    model_name = cfg["model"]
    if model_name == "logistic_regression":
        model = Pipeline(
            [
                ("scaler", StandardScaler(with_mean=True)),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=cfg.get("max_iter", 2000),
                        C=cfg.get("C", 1.0),
                        class_weight=cfg.get("class_weight", None),
                        n_jobs=cfg.get("n_jobs", None),
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    elif model_name == "xgboost":
        xgb_params = pick_xgb_params()
        print(f"[XGB] params={xgb_params}")
        model = XGBClassifier(
            n_estimators=cfg.get("n_estimators", 600),
            max_depth=cfg.get("max_depth", 6),
            learning_rate=cfg.get("learning_rate", 0.05),
            subsample=cfg.get("subsample", 0.8),
            colsample_bytree=cfg.get("colsample_bytree", 0.8),
            eval_metric=cfg.get("eval_metric", "logloss"),
            scale_pos_weight=cfg.get("scale_pos_weight", 1.0),
            **xgb_params,
        )
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")

    model.fit(Xtr, ytr)

    def predict_proba(m, X_):
        if isinstance(m, XGBClassifier):
            booster = m.get_booster()
            dm = xgb.DMatrix(X_)
            return booster.predict(dm)
        if hasattr(m, "predict_proba"):
            return m.predict_proba(X_)[:, 1]
        if hasattr(m, "decision_function"):
            from scipy.special import expit

            return expit(m.decision_function(X_))
        raise RuntimeError("Model lacks probability interface")

    p_tr = predict_proba(model, Xtr)
    p_va = predict_proba(model, Xva)
    p_te = predict_proba(model, Xte)

    print(f"[CAL] calibration={cfg.get('calibration', 'none')}")
    _, transform = make_calibrator(cfg.get("calibration", "none"), p_va, yva)
    p_va_cal = transform(p_va)
    p_te_cal = transform(p_te)

    np.save(os.path.join(outdir, "scores_val.npy"), p_va_cal)
    np.save(os.path.join(outdir, "y_val.npy"), yva)
    np.save(os.path.join(outdir, "node_idx_val.npy"), labeled_idx[val_mask])
    np.save(os.path.join(outdir, "timestep_val.npy"), timesteps[val_mask])
    np.save(os.path.join(outdir, "scores_test.npy"), p_te_cal)
    np.save(os.path.join(outdir, "y_test.npy"), yte)
    np.save(os.path.join(outdir, "node_idx_test.npy"), labeled_idx[test_mask])
    np.save(os.path.join(outdir, "timestep_test.npy"), timesteps[test_mask])

    if cfg.get("use_val_for_thresholds", True):
        if cfg.get("precision_target", 0.0) and cfg["precision_target"] > 0:
            thr = pick_threshold_for_precision(yva, p_va_cal, cfg["precision_target"])
        else:
            thr, _ = pick_threshold_max_f1(yva, p_va_cal)
    else:
        thr, _ = pick_threshold_max_f1(yte, p_te_cal)

    pr = pr_auc_illicit(yte, p_te_cal)
    roc = roc_auc_illicit(yte, p_te_cal)
    f1 = f1_at_threshold(yte, p_te_cal, thr)
    p_at_k = precision_at_k(yte, p_te_cal, cfg.get("topk", 100))
    rec_at_p = recall_at_precision(yte, p_te_cal, cfg.get("precision_target", 0.90))
    ece = expected_calibration_error(yte, p_te_cal)

    metrics = dict(
        pr_auc_illicit=pr,
        roc_auc=roc,
        f1_illicit_at_thr=f1,
        threshold=thr,
        precision_at_k=p_at_k,
        recall_at_precision=rec_at_p,
        ece=ece,
        n_test=int(len(yte)),
    )

    save_json(os.path.join(outdir, "metrics.json"), metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
