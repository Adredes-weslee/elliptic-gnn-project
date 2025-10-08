
from __future__ import annotations

import os
import pandas as pd
import torch
from torch_geometric.data import Data
from typing import Dict, Tuple

LABEL_MAPS = {
    "class1": 1, "1": 1, 1: 1, "illicit": 1,
    "class2": 0, "2": 0, 2: 0, "licit": 0,
    "unknown": -1, -1: -1
}

def _map_label(v):
    s = str(v).strip().lower()
    return LABEL_MAPS.get(s, LABEL_MAPS.get(v, -1))

def load_elliptic_as_graph(
    data_dir: str,
    features_csv: str = "elliptic_txs_features.csv",
    classes_csv: str = "elliptic_txs_classes.csv",
    edgelist_csv: str = "elliptic_txs_edgelist.csv",
) -> Tuple[Data, Dict]:
    # Returns a single global graph with:
    # x: [N, F], y: [N] (0=licit,1=illicit,-1=unknown), edge_index: [2, E], timestep: [N]
    f_path = os.path.join(data_dir, features_csv)
    c_path = os.path.join(data_dir, classes_csv)
    e_path = os.path.join(data_dir, edgelist_csv)

    df_feat = pd.read_csv(f_path, header=None)
    # Expected: col0=txId, col1=timestep, cols[2:]=features
    if df_feat.shape[1] < 3:
        raise ValueError("elliptic_txs_features.csv appears truncated or malformed.")
    df_feat.columns = ["txId", "timestep"] + [f"f{i}" for i in range(df_feat.shape[1]-2)]

    df_cls = pd.read_csv(c_path)
    df_cls.columns = [c.strip() for c in df_cls.columns]
    if "class" not in df_cls.columns:
        # handle variants like 'class_label'
        for col in df_cls.columns:
            if col.lower().startswith("class"):
                df_cls = df_cls.rename(columns={col: "class"})
                break
    df_cls["label"] = df_cls["class"].apply(_map_label)
    df_cls = df_cls.drop(columns=["class"])

    # Join to get labels aligned with features
    df = df_feat.merge(df_cls, on="txId", how="left")
    if "label" not in df.columns:
        df["label"] = -1
    df["label"] = df["label"].fillna(-1).astype(int)

    # Build node index mapping
    tx_ids = df["txId"].values
    tx_to_idx = {tx: i for i, tx in enumerate(tx_ids)}

    # Features/labels/timestep tensors
    feat_cols = [c for c in df.columns if c.startswith("f")]
    x = torch.tensor(df[feat_cols].values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.int64)
    timestep = torch.tensor(df["timestep"].values, dtype=torch.int64)

    # Edges
    df_edge = pd.read_csv(e_path, header=None)
    if df_edge.shape[1] < 2:
        raise ValueError("edgelist CSV malformed.")
    df_edge.columns = ["src", "dst"]

    # Map to indices (drop edges with unknown endpoints)
    src_idx, dst_idx = [], []
    for s, d in zip(df_edge["src"].values, df_edge["dst"].values):
        if s in tx_to_idx and d in tx_to_idx:
            src_idx.append(tx_to_idx[s])
            dst_idx.append(tx_to_idx[d])
    edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, y=y)
    data.timestep = timestep  # [N]

    meta = {
        "num_nodes": int(x.size(0)),
        "num_edges": int(edge_index.size(1)),
        "num_features": int(x.size(1)),
        "label_counts": {
            "-1": int((y==-1).sum()),
            "0": int((y==0).sum()),
            "1": int((y==1).sum()),
        }
    }
    return data, meta

def make_temporal_masks(
    data: Data,
    t_train_end: int,
    t_val_end: int,
    train_window_k: int | None = None,
):
    # Build boolean masks over nodes for train/val/test using labeled nodes only.
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
