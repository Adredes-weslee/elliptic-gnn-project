from __future__ import annotations

import os
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

LABEL_MAPS = {
    "class1": 1,
    "1": 1,
    1: 1,
    "illicit": 1,
    "class2": 0,
    "2": 0,
    2: 0,
    "licit": 0,
    "unknown": -1,
    -1: -1,
}


def _map_label(v):
    s = str(v).strip().lower()
    return LABEL_MAPS.get(s, LABEL_MAPS.get(v, -1))


def _looks_like_timestep(col: pd.Series) -> bool:
    """Heuristic: column looks like the Elliptic time index (integers in [1..49])."""
    if not np.issubdtype(col.dtype, np.number):
        # try coercion if strings
        try:
            col = pd.to_numeric(col, errors="coerce")
        except Exception:
            return False
    vals = col.dropna().astype(float)
    if vals.empty:
        return False
    return (
        (vals.min() >= 1)
        and (vals.max() <= 49)
        and (vals.round().eq(vals).mean() > 0.95)
    )


def load_elliptic_as_graph(
    data_dir: str,
    features_csv: str = "elliptic_txs_features.csv",
    classes_csv: str = "elliptic_txs_classes.csv",
    edgelist_csv: str = "elliptic_txs_edgelist.csv",
) -> Tuple[Data, Dict]:
    """
    Build a single graph with:
      data.x: [N, F], data.y: [N] (0/1/-1), data.edge_index: [2, E], data.timestep: [N] in 1..49.

    Robust to:
      - classes.csv without a time column (derive timestep from features' 2nd column)
      - edgelist headers 'txId1,txId2' or no header
    """
    f_path = os.path.join(data_dir, features_csv)
    c_path = os.path.join(data_dir, classes_csv)
    e_path = os.path.join(data_dir, edgelist_csv)

    # ----------------------------
    # Read CLASSES: txId + class (+ optional time_step/timestep)
    # ----------------------------
    df_cls = pd.read_csv(c_path)
    df_cls.columns = [c.strip() for c in df_cls.columns]

    # Normalize txId column name if needed
    if "txId" not in df_cls.columns:
        for col in df_cls.columns:
            if col.lower().startswith("tx"):
                df_cls = df_cls.rename(columns={col: "txId"})
                break

    # Normalize time column if present
    has_cls_ts = False
    if "time_step" in df_cls.columns:
        df_cls = df_cls.rename(columns={"time_step": "timestep"})
        has_cls_ts = True
    elif "timestep" in df_cls.columns:
        has_cls_ts = True
    else:
        has_cls_ts = False  # we'll try to infer from features CSV

    # Normalize class column name if needed
    if "class" not in df_cls.columns:
        for col in df_cls.columns:
            if col.lower().startswith("class"):
                df_cls = df_cls.rename(columns={col: "class"})
                break

    # Coerce dtypes
    df_cls["txId"] = pd.to_numeric(df_cls["txId"], errors="raise").astype(np.int64)
    if has_cls_ts:
        df_cls["timestep"] = pd.to_numeric(df_cls["timestep"], errors="raise").astype(
            np.int64
        )
    df_cls["label"] = df_cls["class"].apply(_map_label)
    # Keep only what we need from classes
    keep_cols = ["txId", "label"] + (["timestep"] if has_cls_ts else [])
    df_cls = df_cls[keep_cols]

    # ----------------------------
    # Read FEATURES: headerless -> first column is txId, second may be timestep, rest are features
    # ----------------------------
    df_feat = pd.read_csv(f_path, header=None)
    if df_feat.shape[1] < 2:
        raise ValueError(
            "features CSV appears malformed (needs at least txId + 1 column)."
        )

    # First column must be txId
    df_feat = df_feat.copy()
    df_feat.iloc[:, 0] = pd.to_numeric(df_feat.iloc[:, 0], errors="raise").astype(
        np.int64
    )

    # Detect if the 2nd column looks like 'timestep'
    feat_has_ts = _looks_like_timestep(df_feat.iloc[:, 1])

    # If 2nd column is timestep, pull it out (we'll decide whether to use it below)
    feat_timestep = None
    if feat_has_ts:
        feat_timestep = pd.to_numeric(df_feat.iloc[:, 1], errors="raise").astype(
            np.int64
        )

    # Build feature columns
    if feat_has_ts:
        # columns: txId | timestep | F features
        num_feat = df_feat.shape[1] - 2
        feat_cols = [f"f{i}" for i in range(num_feat)]
        # Drop the timestep column from the raw features before naming
        df_feat = df_feat.drop(columns=df_feat.columns[1])
        df_feat.columns = ["txId"] + feat_cols
    else:
        # columns: txId | F features
        num_feat = df_feat.shape[1] - 1
        feat_cols = [f"f{i}" for i in range(num_feat)]
        df_feat.columns = ["txId"] + feat_cols

    # If classes didn't have timestep but features did, attach timestep from features
    if not has_cls_ts and feat_has_ts:
        df_feat["timestep"] = feat_timestep
        used_ts_source = "FEATURES"
    elif has_cls_ts:
        used_ts_source = "CLASSES"
    else:
        # No timestep anywhere -> cannot build temporal masks; fail early
        raise ValueError(
            "No timestep column found in classes and features did not contain a valid timestep column.\n"
            "Expected either classes.csv to have 'time_step'/'timestep' OR features.csv column 2 to be 1..49."
        )

    # ----------------------------
    # Join features with labels (and timestep if present on either side)
    # ----------------------------
    df = df_feat.merge(df_cls, on="txId", how="left")

    # If both sides provided a timestep, prefer classes (dataset-of-record).
    # (In that case pandas would have suffixed columns; reconcile to 'timestep'.)
    if "timestep_x" in df.columns and "timestep_y" in df.columns:
        # Assume _x from features, _y from classes
        df["timestep"] = df["timestep_y"]
        df = df.drop(columns=["timestep_x", "timestep_y"])
        used_ts_source = "CLASSES"
    elif "timestep" not in df.columns and has_cls_ts:
        # classes had timestep but features didn't; bring it in via another merge
        df = df.merge(df_cls[["txId", "timestep"]], on="txId", how="left")
        used_ts_source = "CLASSES"

    if "timestep" not in df.columns:
        raise ValueError("Failed to construct 'timestep' from available CSVs.")

    # Fill missing labels with -1 (unknown)
    if "label" not in df.columns:
        df["label"] = -1
    df["label"] = df["label"].fillna(-1).astype(int)

    print(f"[TS] using timestep from: {used_ts_source}")

    # ----------------------------
    # Tensors
    # ----------------------------
    x = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.int64)
    timestep = torch.tensor(df["timestep"].values, dtype=torch.int64)

    # Build node index (txId -> row index)
    tx_ids = df["txId"].values.astype(np.int64)
    tx_to_idx = {int(tx): i for i, tx in enumerate(tx_ids)}

    # ----------------------------
    # Read EDGELIST robustly: header 'txId1,txId2' OR no header
    # ----------------------------
    try:
        sniff = pd.read_csv(e_path, nrows=5)
        if sniff.shape[1] >= 2 and not np.issubdtype(sniff.dtypes.iloc[0], np.number):
            df_edge = pd.read_csv(e_path, header=0)
        else:
            df_edge = pd.read_csv(e_path, header=None)
    except Exception:
        df_edge = pd.read_csv(e_path, header=None)

    # Normalize to two cols named src,dst
    if {"txId1", "txId2"}.issubset(set(df_edge.columns)):
        df_edge = df_edge[["txId1", "txId2"]].copy()
        df_edge.columns = ["src", "dst"]
    else:
        df_edge = df_edge.iloc[:, :2].copy()
        df_edge.columns = ["src", "dst"]

    df_edge["src"] = pd.to_numeric(df_edge["src"], errors="coerce").astype("Int64")
    df_edge["dst"] = pd.to_numeric(df_edge["dst"], errors="coerce").astype("Int64")
    df_edge = df_edge.dropna().astype({"src": "int64", "dst": "int64"})
    edges_total = len(df_edge)

    # Map to node indices (drop edges whose endpoints aren't present)
    keep = df_edge["src"].isin(tx_to_idx) & df_edge["dst"].isin(tx_to_idx)
    kept = df_edge[keep]
    if kept.empty:
        warnings.warn(
            "No edges mapped to known txIds. If you are testing with a small/partial features CSV, "
            "this is expected. Use the full features file to see edges."
        )

    src_idx = np.array([tx_to_idx[int(s)] for s in kept["src"].values], dtype=np.int64)
    dst_idx = np.array([tx_to_idx[int(d)] for d in kept["dst"].values], dtype=np.int64)

    # Enforce intra-timestep edges (dataset guarantee)
    if len(src_idx) > 0:
        src_t = timestep[torch.from_numpy(src_idx)]
        dst_t = timestep[torch.from_numpy(dst_idx)]
        same_t = (src_t == dst_t).cpu().numpy()
        src_idx = src_idx[same_t]
        dst_idx = dst_idx[same_t]
    else:
        same_t = np.array([], dtype=bool)

    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    print(
        f"[EDGES] total_in_csv={edges_total} mapped={len(kept)} "
        f"same_t={int(same_t.sum())} kept_in_graph={edge_index.size(1)}"
    )

    data = Data(x=x, edge_index=edge_index, y=y)
    data.timestep = timestep  # [N]

    meta = {
        "num_nodes": int(x.size(0)),
        "num_edges": int(edge_index.size(1)),
        "num_features": int(x.size(1)),
        "label_counts": {
            "-1": int((y == -1).sum()),
            "0": int((y == 0).sum()),
            "1": int((y == 1).sum()),
        },
    }
    return data, meta


def make_temporal_masks(
    data: Data,
    t_train_end: int,
    t_val_end: int,
    train_window_k: int | None = None,
):
    """Build boolean masks for train/val/test over labeled nodes only."""
    y = data.y
    t = data.timestep
    labeled = y >= 0

    train_mask = (t <= t_train_end) & labeled
    val_mask = (t > t_train_end) & (t <= t_val_end) & labeled
    test_mask = (t > t_val_end) & labeled

    if train_window_k is not None:
        t_lo = max(1, t_train_end - train_window_k + 1)
        train_mask = (t >= t_lo) & (t <= t_train_end) & labeled

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data
