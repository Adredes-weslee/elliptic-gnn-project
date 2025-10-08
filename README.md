You’re super close. The two snippets you shared (EDA + baselines) look solid. The weak spot that explains your earlier `num_edges: 0` is almost certainly the **dataset loader** (not shown here): it likely assumes the second column of `elliptic_txs_features.csv` is `timestep`. In the official dump, the **timestep lives in `elliptic_txs_classes.csv`** (often as `time_step`) and the **features file is `txId + 166 features`**. If you treat a *feature* column as timestep, your edgelist-to-node mapping goes sideways and you can end up dropping every edge.

Below is (1) the **updated README** block with step-by-step “what to run” + what to expect, and (2) a **Codex patch** to make the dataset loader bullet-proof against headers/dtypes and to source `timestep` from the classes CSV (plus safe edge filtering). I also included a tiny prompt to relax baseline thresholds for smoke tests.

---

# ✅ Updated README (drop-in)

Replace your README’s quickstart/run instructions with this block.

```markdown
# Catching Illicit Crypto Flows — Elliptic GNN Project

Implements the ISyE 6740 project: detecting illicit Bitcoin transactions on the Elliptic dataset using feature-only baselines (LR, XGBoost) and GNNs (GCN, GraphSAGE, GAT) with leakage-safe temporal splits, calibration, and operations-focused metrics.

## Repo layout

```

.
├── configs/
│   ├── split.yaml
│   ├── baseline_lr.yaml
│   ├── baseline_xgb.yaml
│   ├── gcn.yaml
│   ├── sage.yaml
│   └── gat.yaml
├── data/
│   └── raw/          # <- put the three Elliptic CSVs here
├── data/processed/   # <- created by build_graph
├── outputs/          # <- run artifacts
├── src/
│   ├── analysis/     # by-time, calibration, workload, bootstrap, hub-ablation, sweep, etc.
│   ├── data/         # build_graph.py, dataset_elliptic.py
│   ├── models/       # gnn.py
│   ├── utils/        # common.py, metrics.py, calibrate.py, logger.py
│   ├── train_baselines.py
│   └── train_gnn.py
├── app/
│   └── streamlit_app.py
└── environment.yml

````

## 0) Requirements

- Conda/Mamba. CUDA GPU optional (CPU works but slower).
- Data files in `data/raw/`:
  - `elliptic_txs_features.csv`  (txId + 166 numeric feature columns)
  - `elliptic_txs_classes.csv`   (txId + class + time_step)
  - `elliptic_txs_edgelist.csv`  (two columns: src, dst)

## 1) Environment

```bash
conda env create -f environment.yml
conda activate elliptic-gnn
# If torch-geometric fails for your CUDA, see:
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
````

## 2) Build the graph (and verify edges)

```bash
python -m src.data.build_graph --config configs/split.yaml
# EXPECT: a summary with ~203,769 nodes and ~234,355 edges (give or take).
# Files: data/processed/graph.pt and data/processed/meta.json
```

## 3) EDA & leakage check

```bash
python -m src.analysis.eda --processed_dir data/processed --assert_no_cross_time_edges
# Files: degree_hist.csv, labels_by_time.csv
# EXPECT: “no cross-timestep edges” (dataset guarantees this).
```

## 4) Baselines

Logistic Regression:

```bash
python -m src.train_baselines --config configs/baseline_lr.yaml
```

XGBoost (auto-CPU/GPU):

```bash
python -m src.train_baselines --config configs/baseline_xgb.yaml
```

Outputs (per run under `outputs/baselines/<run_name>/`):

* `metrics.json` (PR-AUC_illicit, ROC-AUC, F1@thr, Precision@K, Recall@Precision, ECE, threshold)
* `scores_(val|test).npy`, `y_(val|test).npy`, `timestep_(val|test).npy`, `node_idx_(val|test).npy`
* optional `training_log.csv`

> Tip (smoke test): set `precision_target: 0.0` to use **max-F1** instead of forcing 90% precision.

## 5) GNNs (GCN / SAGE / GAT)

```bash
python -m src.train_gnn --config configs/gcn.yaml
python -m src.train_gnn --config configs/sage.yaml
python -m src.train_gnn --config configs/gat.yaml
```

You should see:

```
[GPU] CUDA available: N device(s) -> [...]
[RUN] Using device: cuda
```

Key toggles in configs:

* `device: auto|cpu|cuda`, `amp: true`, `grad_clip: 1.0`
* `symmetrize_edges: false|true`
* `use_time_scalar: false|true` (append normalized timestep)
* `train_window_k: null|<int>` (rolling window over train period)
* `ablate_hubs_frac: 0.0..0.1` (hub removal eval)

Outputs (per run under `outputs/gnn/<run_name>/`):

* `best.ckpt`, `metrics.json`, `config_used.yaml`
* `scores_*`, `y_*`, `timestep_*`, `node_idx_*`
* `training_log.csv`, TensorBoard logs in `tb/`

TensorBoard:

```bash
tensorboard --logdir outputs/gnn/<run_name>/tb
```

## 6) Analysis & plots

Per-timestep drift:

```bash
python -m src.analysis.eval_by_time --run_dir outputs/gnn/gcn_h64
# by_time.csv, by_time_pr_auc.png
```

Calibration curve:

```bash
python -m src.analysis.calibration_plots --run_dir outputs/gnn/gcn_h64
```

Workload (Precision@K):

```bash
python -m src.analysis.workload_curves --run_dir outputs/gnn/gcn_h64 --k_max 5000
```

Paired bootstrap (baseline vs GNN):

```bash
python -m src.analysis.bootstrap_compare \
  --run_a outputs/baselines/xgb_default \
  --run_b outputs/gnn/gcn_h64 \
  --topk 100 --n_boot 1000
```

Hub-removed ablation:

```bash
python -m src.analysis.hub_ablation --run_dir outputs/gnn/gcn_h64 --frac 0.01
```

## 7) Interpretability

XGBoost + SHAP:

```bash
python -m src.analysis.explain --mode xgb --run_dir outputs/baselines/xgb_default --max_plots 1
```

GNNExplainer:

```bash
python -m src.analysis.explain --mode gnn --run_dir outputs/gnn/gcn_h64
```

## 8) Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
```

Browse runs under `outputs/…`:

* Metrics tables, calibration curve, workload curve
* By-timestep drift plot
* Optional bootstrap deltas & hub-removed metrics (if present)

## What to do first

1. Build graph → confirm edges ≈ 234k
2. Run LR and XGB → get baseline metrics
3. Run GCN → confirm GPU/AMP + better PR-AUC than baselines
4. Generate plots + Streamlit to review
5. Then tune/sweeps (hidden_dim, dropout, class weighting / focal) and run paired bootstrap vs baselines.

```

---

# 🔧 Codex patch — robust dataset loader (features/timestep/edgelist)

Paste this to Codex. It fixes the common pitfalls:

- **timestep** is read from `elliptic_txs_classes.csv` (`time_step` → `timestep`), not from the features CSV.
- **txId/src/dst** are forced to integers.
- **edgelist** header or no-header handled cleanly.
- **intra-timestep** edge enforcement (no leakage).
- Helpful `[EDGES] …` print.

```

Open the repo and patch the dataset loader to robustly parse Elliptic.

FILE: src/data/dataset_elliptic.py

GOAL

* Features CSV: txId + 166 feature columns (no timestep).
* Classes CSV: txId + class + time_step → normalize to 'timestep' and map labels.
* Edgelist CSV: handle header/no-header, force int ids, map to node indices.
* Enforce intra-timestep edges; print counts.

EDITS (replace the current load_elliptic_as_graph implementation with this one):

import os
import json
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, Tuple
import warnings

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
"""
Returns a single graph:
data.x: [N, F], data.y: [N] (0/1/-1), data.edge_index: [2, E]
data.timestep: [N] in 1..49
"""

```
f_path = os.path.join(data_dir, features_csv)
c_path = os.path.join(data_dir, classes_csv)
e_path = os.path.join(data_dir, edgelist_csv)

# --- FEATURES: txId + 166 features (no timestep here)
df_feat = pd.read_csv(f_path, header=None)
if df_feat.shape[1] < 2:
    raise ValueError("features CSV appears malformed.")
# Expect 1 + 166 = 167 columns most commonly
feat_cols = [f"f{i}" for i in range(df_feat.shape[1] - 1)]
df_feat.columns = ["txId"] + feat_cols
df_feat["txId"] = pd.to_numeric(df_feat["txId"], errors="raise").astype(np.int64)

# --- CLASSES: txId + class + time_step (normalize names)
df_cls = pd.read_csv(c_path)
df_cls.columns = [c.strip() for c in df_cls.columns]

# normalize columns
if "txId" not in df_cls.columns:
    for col in df_cls.columns:
        if col.lower().startswith("tx"):
            df_cls = df_cls.rename(columns={col: "txId"})
            break
if "time_step" in df_cls.columns:
    df_cls = df_cls.rename(columns={"time_step": "timestep"})
if "timestep" not in df_cls.columns:
    for col in df_cls.columns:
        if col.lower().startswith("time"):
            df_cls = df_cls.rename(columns={col: "timestep"})
            break
if "class" not in df_cls.columns:
    for col in df_cls.columns:
        if col.lower().startswith("class"):
            df_cls = df_cls.rename(columns={col: "class"})
            break

df_cls["txId"] = pd.to_numeric(df_cls["txId"], errors="raise").astype(np.int64)
df_cls["timestep"] = pd.to_numeric(df_cls["timestep"], errors="raise").astype(np.int64)
df_cls["label"] = df_cls["class"].apply(_map_label)
df_cls = df_cls.drop(columns=["class"])

# --- JOIN: ensure every feature row gets label + timestep (unknown if missing)
df = df_feat.merge(df_cls[["txId","timestep","label"]], on="txId", how="left")
if "label" not in df.columns:
    df["label"] = -1
df["label"] = df["label"].fillna(-1).astype(int)

# build node index
tx_ids = df["txId"].values.astype(np.int64)
tx_to_idx = {int(tx): i for i, tx in enumerate(tx_ids)}

# tensors
x = torch.tensor(df[feat_cols].values, dtype=torch.float32)
y = torch.tensor(df["label"].values, dtype=torch.int64)
timestep = torch.tensor(df["timestep"].values, dtype=torch.int64)

# --- EDGELIST: header/no-header robust, force ints
try:
    sniff = pd.read_csv(e_path, nrows=5)
    if sniff.shape[1] >= 2 and not np.issubdtype(sniff.dtypes.iloc[0], np.number):
        df_edge = pd.read_csv(e_path, header=0)
    else:
        df_edge = pd.read_csv(e_path, header=None)
except Exception:
    df_edge = pd.read_csv(e_path, header=None)

df_edge = df_edge.iloc[:, :2].copy()
df_edge.columns = ["src","dst"]
df_edge["src"] = pd.to_numeric(df_edge["src"], errors="coerce").astype("Int64")
df_edge["dst"] = pd.to_numeric(df_edge["dst"], errors="coerce").astype("Int64")
df_edge = df_edge.dropna().astype({"src":"int64","dst":"int64"})
edges_total = len(df_edge)

# map to indices
keep = df_edge["src"].isin(tx_to_idx) & df_edge["dst"].isin(tx_to_idx)
kept = df_edge[keep]
if kept.empty:
    warnings.warn("No edges matched txId mapping. Check id dtypes/headers.")
src_idx = np.array([tx_to_idx[int(s)] for s in kept["src"].values], dtype=np.int64)
dst_idx = np.array([tx_to_idx[int(d)] for d in kept["dst"].values], dtype=np.int64)

# enforce intra-timestep edges
src_t = timestep[torch.from_numpy(src_idx)]
dst_t = timestep[torch.from_numpy(dst_idx)]
same_t = (src_t == dst_t).cpu().numpy()
src_idx = src_idx[same_t]
dst_idx = dst_idx[same_t]

edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

print(f"[EDGES] total_in_csv={edges_total} mapped={len(kept)} "
      f"same_t={int(same_t.sum())} kept_in_graph={edge_index.size(1)}")

data = Data(x=x, edge_index=edge_index, y=y)
data.timestep = timestep

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
```

````

**Acceptance check:**

```bash
python -m src.data.build_graph --config configs/split.yaml
# Expect: [EDGES] ... kept_in_graph > 0  and meta num_edges ≈ 234k
python -m src.analysis.eda --processed_dir data/processed --assert_no_cross_time_edges
````

---

# 🧪 (Optional) Codex micro-prompt — smoke-test thresholds for baselines

If your smoke tests keep producing F1=0 because `precision_target: 0.90` is too strict, paste this to Codex to tweak the default configs:

```
Open configs/baseline_lr.yaml and configs/baseline_xgb.yaml and set:
precision_target: 0.0    # use max-F1 for smoke tests
# keep a commented line with your final target (e.g., 0.90) to restore later
```

(You can restore your 0.90 target once models start producing good precision.)

---

## Should you implement more now, or test first?

**Test first.** After the loader patch, rebuild the graph and verify **edges > 0**. Then run LR/XGB and one GNN (GCN). Generate the analysis plots and open your Streamlit app. Once the plumbing looks good, move on to:

* **Rolling window (`train_window_k`)** sensitivity
* **Symmetrize edges** toggle
* **Hub-removed ablation** (`ablate_hubs_frac`)
* **Paired bootstrap** (GNN vs best baseline)

If anything still looks off (e.g., edges still 0), share the first 5 rows + dtypes of each CSV and we’ll adjust the loader immediately.
