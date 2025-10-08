
# Catching Illicit Crypto Flows — Elliptic GNN Project

Implements the experiments described in the proposal **“Catching Illicit Crypto Flows: Temporal Graph Neural Networks on the Elliptic Bitcoin Dataset.”**

Includes:
- Leakage-safe pipeline from Elliptic CSVs to a PyTorch Geometric graph with temporal masks
- **Baselines:** Logistic Regression and XGBoost (with optional isotonic/Platt calibration)
- **GNNs:** GCN, GraphSAGE, GAT with class-weighted loss and early stopping on **validation PR-AUC (illicit)**
- Metrics: PR-AUC (illicit), ROC-AUC, F1 (illicit), Precision@K, Recall@Precision
- Temperature scaling for GNN calibration
- Reproducibility (seeded runs), YAML-configured experiments

> **Dataset placement**: Put the 3 CSVs in `data/raw/`:
> - `elliptic_txs_features.csv`
> - `elliptic_txs_classes.csv`
> - `elliptic_txs_edgelist.csv`

## 1) Environment (conda)

```bash
conda env create -f environment.yml
conda activate elliptic-gnn
```

If `torch-geometric` wheels fail for your CUDA setup, see:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html  
For CPU-only, replace `pytorch-cuda=11.8` with `cpuonly` in `environment.yml`.

## 2) Prepare data

Build the cached graph and masks (train/val/test by timestep):

```bash
python -m src.data.build_graph --config configs/split.yaml
```

This writes `data/processed/graph.pt` and `data/processed/meta.json`.

## EDA & Checks

Generate quick sanity tables (degree histogram and labels by timestep) and
optionally enforce that every edge stays within a single timestep:

```bash
python -m src.analysis.eda --processed_dir data/processed --assert_no_cross_time_edges
```

The script writes `degree_hist.csv` and `labels_by_time.csv` alongside the
processed graph and prints a short summary.

## 3) Baselines

Logistic regression:

```bash
python -m src.train_baselines --config configs/baseline_lr.yaml
```

XGBoost (GPU if available):

```bash
python -m src.train_baselines --config configs/baseline_xgb.yaml
```

Artifacts: `outputs/baselines/<run_name>/metrics.json`.

## 4) GNNs

GCN / GraphSAGE / GAT:

```bash
python -m src.train_gnn --config configs/gcn.yaml
python -m src.train_gnn --config configs/sage.yaml
python -m src.train_gnn --config configs/gat.yaml
```

On a CUDA machine you should see startup logs similar to:

```
[GPU] CUDA available: 1 device(s) -> ['NVIDIA ...'], torch.version.cuda=...
[RUN] Using device: cuda
```

Artifacts: `outputs/gnn/<run_name>/metrics.json` and `best.ckpt`.

View TensorBoard with:

```
tensorboard --logdir outputs/gnn/<run_name>/tb
```

## Analysis

After training, run the analysis scripts on a directory that contains
`metrics.json`, `scores_test.npy`, `y_test.npy`, and `timestep_test.npy`
(for example `outputs/gnn/<run_name>/`). Each script writes a CSV summary
and saves a PNG plot alongside the run artifacts.

```bash
# Per-timestep drift analysis
python src/analysis/eval_by_time.py --run_dir outputs/gnn/<run_name>

# Calibration reliability curve
python src/analysis/calibration_plots.py --run_dir outputs/gnn/<run_name>

# Precision-vs-workload curve (adjust --k_max as needed)
python src/analysis/workload_curves.py --run_dir outputs/gnn/<run_name> --k_max 5000
```

## 5) Repo layout

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
│   └── raw/   # <- put CSVs here
├── outputs/   # run artifacts
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── build_graph.py
│   │   └── dataset_elliptic.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── gnn.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── calibrate.py
│   │   ├── metrics.py
│   │   └── common.py
│   ├── train_baselines.py
│   └── train_gnn.py
├── environment.yml
└── README.md
```

## Notes

- Unknown nodes are never treated as negatives; they only provide graph context.
- Edges are intra-timestep only, per dataset description.
- Early stopping monitors **validation PR-AUC (illicit)**.
- PR curves and Precision@K use **illicit** as the positive class.
