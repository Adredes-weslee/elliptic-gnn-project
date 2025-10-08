
import os, argparse, yaml, json, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import torch
from src.utils.common import ensure_dir, save_json, set_seed
from src.utils.metrics import (
    pr_auc_illicit, roc_auc_illicit, precision_at_k, pick_threshold_max_f1,
    pick_threshold_for_precision, f1_at_threshold, expected_calibration_error, recall_at_precision
)
from src.utils.calibrate import calibrate_isotonic, calibrate_platt

def load_cached(processed_dir):
    data = torch.load(os.path.join(processed_dir, "graph.pt"), map_location="cpu")
    return data

def get_split_arrays(data):
    x = data.x.numpy()
    y = data.y.numpy()
    labeled = y >= 0
    feats = x[labeled]
    labels = y[labeled]
    t = data.timestep.numpy()[labeled]

    t_train_end = int(torch.max(data.timestep[data.train_mask]).item())
    t_val_end = int(torch.max(data.timestep[data.val_mask]).item())

    train = t <= t_train_end
    val = (t > t_train_end) & (t <= t_val_end)
    test = t > t_val_end
    return feats, labels, train, val, test

def main(cfg):
    set_seed(cfg.get("seed", 42))
    outdir = os.path.join("outputs", "baselines", cfg["run_name"])
    ensure_dir(outdir)

    data = load_cached(cfg["processed_dir"])
    X, y, train_mask, val_mask, test_mask = get_split_arrays(data)

    Xtr, ytr = X[train_mask], y[train_mask]
    Xva, yva = X[val_mask], y[val_mask]
    Xte, yte = X[test_mask], y[test_mask]

    model_name = cfg["model"]
    if model_name == "logistic_regression":
        model = Pipeline([
            ("scaler", StandardScaler(with_mean=True)),
            ("clf", LogisticRegression(max_iter=cfg.get("max_iter",2000), C=cfg.get("C",1.0)))
        ])
    elif model_name == "xgboost":
        model = XGBClassifier(
            n_estimators=cfg.get("n_estimators",600),
            max_depth=cfg.get("max_depth",6),
            learning_rate=cfg.get("learning_rate",0.05),
            subsample=cfg.get("subsample",0.8),
            colsample_bytree=cfg.get("colsample_bytree",0.8),
            eval_metric=cfg.get("eval_metric","logloss"),
            tree_method=cfg.get("tree_method","auto")
        )
    else:
        raise ValueError("Unknown baseline model")

    model.fit(Xtr, ytr)

    def predict_proba(m, X):
        if hasattr(m, "predict_proba"):
            return m.predict_proba(X)[:,1]
        elif hasattr(m, "decision_function"):
            from scipy.special import expit
            return expit(m.decision_function(X))
        else:
            raise RuntimeError("Model has neither predict_proba nor decision_function")

    p_tr = predict_proba(model, Xtr)
    p_va = predict_proba(model, Xva)
    p_te = predict_proba(model, Xte)

    # Calibration (optional)
    cal = cfg.get("calibration","none")
    if cal == "isotonic":
        calibrator = calibrate_isotonic(p_va, yva)
        p_te_cal = calibrator.transform(p_te)
        p_va_cal = calibrator.transform(p_va)
    elif cal == "platt":
        calibrator = calibrate_platt(p_va, yva)
        p_te_cal = calibrator.predict_proba(p_te.reshape(-1,1))[:,1]
        p_va_cal = calibrator.predict_proba(p_va.reshape(-1,1))[:,1]
    else:
        calibrator = None
        p_te_cal = p_te
        p_va_cal = p_va

    # Threshold selection on validation
    if cfg.get("use_val_for_thresholds", True):
        if cfg.get("precision_target", 0.0) and cfg["precision_target"] > 0:
            thr = pick_threshold_for_precision(yva, p_va_cal, cfg["precision_target"])
        else:
            thr, _ = pick_threshold_max_f1(yva, p_va_cal)
    else:
        thr, _ = pick_threshold_max_f1(yte, p_te_cal)

    # Metrics on test
    pr = pr_auc_illicit(yte, p_te_cal)
    roc = roc_auc_illicit(yte, p_te_cal)
    f1 = f1_at_threshold(yte, p_te_cal, thr)
    p_at_k = precision_at_k(yte, p_te_cal, cfg.get("topk",100))
    rec_at_p = recall_at_precision(yte, p_te_cal, cfg.get("precision_target",0.90))
    ece = expected_calibration_error(yte, p_te_cal)

    metrics = dict(
        pr_auc_illicit=pr,
        roc_auc=roc,
        f1_illicit_at_thr=f1,
        threshold=thr,
        precision_at_k=p_at_k,
        recall_at_precision=rec_at_p,
        ece=ece,
        n_test=int(len(yte))
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
