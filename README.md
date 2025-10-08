
 Catching Illicit Crypto Flows — Elliptic GNN Project

Implements the ISyE 6740 project: detecting illicit Bitcoin transactions on the **Elliptic** dataset using **feature-only baselines** (LR, XGBoost) and **GNNs** (GCN, SAGE, GAT) with leakage-safe temporal splits, calibration, and operations-focused metrics.

## Repo layout

```

.
├── configs/
│   ├── baseline_lr.yaml
│   ├── baseline_xgb.yaml
│   ├── gcn.yaml
│   ├── sage.yaml
│   ├── gat.yaml
│   └── split.yaml
├── data/
│   └── raw/         # <- place the three Elliptic CSVs here
├── outputs/         # run artifacts (created)
├── src/
│   ├── analysis/    # eval_by_time, calibration_plots, workload_curves, bootstrap_compare, hub_ablation, sweep, (robustness if added)
│   ├── data/        # build_graph.py, dataset_elliptic.py
│   ├── models/      # gnn.py
│   ├── utils/       # common.py, metrics.py, calibrate.py, logger.py
│   ├── train_baselines.py
│   └── train_gnn.py
├── environment.yml
└── README.md

````

## 0) Requirements

- Conda/Mamba (recommended), CUDA GPU optional (CPU works but slower).
- Three Elliptic CSVs:
  - `elliptic_txs_features.csv`
  - `elliptic_txs_classes.csv`
  - `elliptic_txs_edgelist.csv`

## 1) Environment

```bash
# create and activate
conda env create -f environment.yml
conda activate elliptic-gnn

# if torch-geometric fails for your CUDA, see:
# https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
# CPU fallback: set pytorch-cuda to 'cpuonly' in environment.yml and recreate
````

## 2) Put data & build graph

Place the CSVs in `data/raw/` and then:

```bash
python -m src.data.build_graph --config configs/split.yaml
```

This creates `data/processed/graph.pt` & `meta.json`. Splits are temporal (train ≤ 34, val 35–43, test 44–49 by default).

## 3) EDA & checks (sanity + no leakage)

```bash
python -m src.analysis.eda --processed_dir data/processed --assert_no_cross_time_edges
```

Outputs:

* `degree_hist.csv` and `labels_by_time.csv`
* Guards that **all edges are intra-timestep** (no leakage).

## 4) Baselines

Logistic regression:

```bash
python -m src.train_baselines --config configs/baseline_lr.yaml
```

XGBoost (auto-selects `gpu_hist` when CUDA is visible):

```bash
python -m src.train_baselines --config configs/baseline_xgb.yaml
```

Artifacts (per run in `outputs/baselines/<run_name>/`):

* `metrics.json` (PR-AUC_illicit, ROC-AUC, F1@thr, Precision@K, Recall@Precision, ECE, chosen threshold)
* `training_log.csv` (if implemented)
* Per-node outputs: `scores_(val|test).npy`, `y_(val|test).npy`, `timestep_(val|test).npy`, `node_idx_(val|test).npy`

## 5) GNNs (GCN / SAGE / GAT)

```bash
# Shows GPU info + uses AMP on CUDA
python -m src.train_gnn --config configs/gcn.yaml
python -m src.train_gnn --config configs/sage.yaml
python -m src.train_gnn --config configs/gat.yaml
```

You should see:

```
[GPU] CUDA available: N device(s) -> [...]
[RUN] Using device: cuda
```

Key config toggles:

* `device: auto|cpu|cuda`
* `amp: true` (mixed precision)
* `symmetrize_edges: false|true`
* `use_time_scalar: false|true` (append normalized timestep as a feature)
* `train_window_k: null|<int>` (rolling window over train period)
* `ablate_hubs_frac: 0.0..0.1` (hub removal eval)

Artifacts (per run in `outputs/gnn/<run_name>/`):

* `best.ckpt`, `metrics.json`, `config_used.yaml`
* `training_log.csv`, TensorBoard logs at `tb/`
* Per-node arrays: `scores/y/timestep/node_idx` for val & test

TensorBoard:

```bash
tensorboard --logdir outputs/gnn/<run_name>/tb
```

## 6) Analysis scripts

By-timestep drift:

```bash
python -m src.analysis.eval_by_time --run_dir outputs/gnn/gcn_h64
# writes by_time.csv and by_time_pr_auc.png
```

Calibration curve + ECE:

```bash
python -m src.analysis.calibration_plots --run_dir outputs/gnn/gcn_h64
```

Workload curves (Precision@K):

```bash
python -m src.analysis.workload_curves --run_dir outputs/gnn/gcn_h64 --k_max 5000
```

Paired bootstrap (baseline vs GNN):

```bash
python -m src.analysis.bootstrap_compare \
  --run_a outputs/baselines/xgb_default \
  --run_b outputs/gnn/gcn_h64 \
  --topk 100 --n_boot 1000
# writes bootstrap_compare.json (ΔPR-AUC, ΔP@K with 95% CIs)
```

Hub-removed ablation (if enabled in config, writes `metrics_hub_removed.json`);
you can also use the standalone script if included:

```bash
python -m src.analysis.hub_ablation --run_dir outputs/gnn/gcn_h64 --frac 0.01
```

Robustness (random edge drop + feature noise):

```bash
python -m src.analysis.robustness --run_dir outputs/gnn/gcn_h64 --drop_frac 0.10 --noise_std 0.01
```

## 7) Interpretability

XGBoost + SHAP:

```bash
python -m src.analysis.explain --mode xgb --run_dir outputs/baselines/xgb_default --max_plots 1
# shap_summary.png, xgb_top_features.json
```

GNNExplainer:

```bash
# auto-selects a node or pass --node <id>
python -m src.analysis.explain --mode gnn --run_dir outputs/gnn/gcn_h64
# gnn_explainer_node_<id>.png, gnn_explainer_importance.json
```

## 8) Small sweeps

```bash
python -m src.analysis.sweep \
  --template configs/gcn.yaml \
  --out_csv outputs/sweeps/gcn_small.csv \
  --param hidden_dim 64 128 \
  --param dropout 0.2 0.5 \
  --param focal_loss false true \
  --param focal_gamma 1.0 2.0
```

## Notes

* Unlabeled nodes are never treated as negatives; they provide graph context.
* Early stopping monitors **val PR-AUC (illicit)**.
* Thresholds are set on **validation** (Precision target or max-F1) and applied on test.

````

---

## ✅ What to run first (quick checklist)

1) **Create env & activate**
```bash
conda env create -f environment.yml && conda activate elliptic-gnn
````

2. **Put CSVs** into `data/raw/` and **build graph**

```bash
python -m src.data.build_graph --config configs/split.yaml
```

3. **Sanity checks**

```bash
python -m src.analysis.eda --processed_dir data/processed --assert_no_cross_time_edges
```

4. **Baselines (LR, XGB)**

```bash
python -m src.train_baselines --config configs/baseline_lr.yaml
python -m src.train_baselines --config configs/baseline_xgb.yaml
```

5. **GNNs**

```bash
python -m src.train_gnn --config configs/gcn.yaml
```

Expect to see GPU lines. After ~few epochs you’ll get `outputs/gnn/<run>/metrics.json`.

6. **Analysis & plots**

```bash
python -m src.analysis.eval_by_time --run_dir outputs/gnn/<run>
python -m src.analysis.calibration_plots --run_dir outputs/gnn/<run>
python -m src.analysis.workload_curves --run_dir outputs/gnn/<run> --k_max 5000
```

7. **Compare models (bootstrap)**

```bash
python -m src.analysis.bootstrap_compare \
  --run_a outputs/baselines/xgb_default \
  --run_b outputs/gnn/gcn_h64 \
  --topk 100 --n_boot 1000
```

8. **Interpretability**

```bash
python -m src.analysis.explain --mode xgb --run_dir outputs/baselines/xgb_default
python -m src.analysis.explain --mode gnn --run_dir outputs/gnn/gcn_h64
```

---


---

### Should you add more code right now?

My recommendation: **test first** with a small run to confirm the plumbing (GPU/AMP lines, metrics saved, analysis scripts produce CSV/PNGs). Then add the **robustness** script (Prompt N) and the **Streamlit UI** (Prompt UI) — both directly support your evaluation section and make your results easy to present.

If you want, I can also give you prompts to add a Dockerfile + Makefile + pre-commit (ruff/black) for smooth grading and reproducibility.

### Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```

Select a run under outputs/ to explore metrics, curves, and comparisons.

