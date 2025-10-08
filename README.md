
---

# ✅ Updated README (drop-in)

Replace your README’s quickstart/run instructions with this block.

# Catching Illicit Crypto Flows — Elliptic GNN Project

Implements the ISyE 6740 project: detecting illicit Bitcoin transactions on the Elliptic dataset using feature-only baselines (LR, XGBoost) and GNNs (GCN, GraphSAGE, GAT) with leakage-safe temporal splits, calibration, and operations-focused metrics.

## Repo layout

```
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
```

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
