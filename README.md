
---

# ✅ Catching Illicit Crypto Flows — Elliptic GNN Project

Detects illicit Bitcoin transactions on the Elliptic dataset using feature-only baselines (LR, XGBoost) and GNNs (GCN, GraphSAGE, GAT) with leakage-safe temporal splits, temperature calibration, robustness checks, and ops-oriented metrics (Precision@K, PR-AUC by timestep, etc.).

## Repo layout

```
├── configs/
│   ├── split.yaml
│   ├── baseline_lr.yaml
│   ├── baseline_xgb.yaml
│   ├── gcn.yaml
│   ├── sage.yaml
│   ├── gat.yaml
│   ├── rec_k8.yaml              # recommended: SAGE-ResBN (k=8, sin2)
│   └── rec_k9.yaml              # recommended: SAGE-ResBN (k=9, sin2)
├── data/
│   └── raw/                     # <- put the three Elliptic CSVs here
├── data/processed/              # <- created by build_graph
├── outputs/                     # <- run artifacts (baselines/, gnn/, …)
├── src/
│   ├── analysis/                # by-time, calibration, workload, bootstrap, robustness, explain, ensemble, …
│   ├── sweeps/                  # sweep_gnn.py (hyperparam grid)
│   ├── data/                    # build_graph.py, dataset_elliptic.py
│   ├── models/                  # gnn.py
│   ├── utils/                   # common.py, metrics.py, calibrate.py, logger.py
│   ├── train_baselines.py
│   └── train_gnn.py
├── app/
│   └── streamlit_app.py
└── environment.yml
```

## 0) Requirements

* Conda (or mamba). CUDA GPU optional (CPU works, slower).
* Elliptic CSVs in `data/raw/`:

  * `elliptic_txs_features.csv`  (txId + 166 numeric features)
  * `elliptic_txs_classes.csv`   (txId + class + time_step)
  * `elliptic_txs_edgelist.csv`  (two columns: src, dst)

## 1) Environment

```bash
conda env create -f environment.yml
conda activate elliptic-gnn
# If torch-geometric build fails for your CUDA, see their install notes.
```

## 2) Build the graph

```bash
python -m src.data.build_graph --config configs/split.yaml
# EXPECT: ~203,769 nodes and ~234,355 edges.
# Files: data/processed/graph.pt, data/processed/meta.json
```

## 3) EDA & leakage check

```bash
python -m src.analysis.eda --processed_dir data/processed --assert_no_cross_time_edges
# Files: degree_hist.csv, labels_by_time.csv
# Expect dataset has no cross-timestep edges.
```

## 4) Baselines

Logistic Regression

```bash
python -m src.train_baselines --config configs/baseline_lr.yaml
```

XGBoost (auto CPU/GPU)

```bash
python -m src.train_baselines --config configs/baseline_xgb.yaml
```

Outputs per run (`outputs/baselines/<run_name>/`):

* `metrics.json` (PR-AUC_illicit, ROC-AUC, F1@thr, Precision@K, Recall@Precision, ECE, threshold)
* `scores_(val|test).npy`, `y_(val|test).npy`, `timestep_(val|test).npy`, `node_idx_(val|test).npy`
* `training_log.csv` (optional)

> Tip: set `precision_target: 0.0` in a config to use **max-F1** instead of enforcing a precision target.

## 5) GNNs (GCN / SAGE / GAT)

```bash
python -m src.train_gnn --config configs/gcn.yaml
python -m src.train_gnn --config configs/sage.yaml
python -m src.train_gnn --config configs/gat.yaml
```

Config toggles you’ll use a lot:

* `device: auto|cpu|cuda`, `amp: true`, `grad_clip: 1.0`
* `symmetrize_edges: true|false`
* `use_time_scalar: true|false` (append normalized timestep if no embedding)
* `time_embed_dim`, `time_embed_type` (e.g., `sin`), `train_window_k`
* `precision_target` (0.0 for max-F1; >0 to pick threshold at that precision)
* `ablate_hubs_frac` (hub removal during eval)

Outputs per run (`outputs/gnn/<run_name>/`):

* `best.ckpt`, `metrics.json`, `config_used.yaml`
* `scores_*`, `y_*`, `timestep_*`, `node_idx_*`
* `training_log.csv`, TensorBoard logs in `tb/`

TensorBoard:

```bash
tensorboard --logdir outputs/gnn/<run_name>/tb
```

### Recommended configs

The two configs below reflect the settings that performed best in your experiments:

```bash
python -m src.train_gnn --config configs/rec_k8.yaml
python -m src.train_gnn --config configs/rec_k9.yaml
```

(They use SAGE-ResBN, `time_embed_type: sin`, `time_embed_dim: 2`, `dropout: 0.2`, `lr: 5e-4`, `weight_decay: 5e-5`, `train_window_k: 8 or 9`, no time-loss weighting.)

## 6) Analysis & plots

Per-timestep drift:

```bash
python -m src.analysis.eval_by_time --run_dir outputs/gnn/rec_sage_resbn_k8_sin2
# Creates: by_time.csv, by_time_pr_auc.png
```

Calibration curve:

```bash
python -m src.analysis.calibration_plots --run_dir outputs/gnn/rec_sage_resbn_k8_sin2
# Creates: calibration_curve.png
```

Workload (Precision@K):

```bash
python -m src.analysis.workload_curves --run_dir outputs/gnn/rec_sage_resbn_k8_sin2 --k_max 5000
# Creates: workload_curve.png, workload_curve.csv
```

Paired bootstrap (e.g., baseline vs GNN):

```bash
python -m src.analysis.bootstrap_compare \
  --run_a outputs/baselines/xgb_default \
  --run_b outputs/gnn/rec_sage_resbn_k9_sin2 \
  --topk 100 --n_boot 1000
```

This writes:

* `outputs/gnn/rec_sage_resbn_k9_sin2/bootstrap_compare.json` (primary)
* `outputs/gnn/rec_sage_resbn_k9_sin2/bootstrap_compare_<run_a_name>.json`
* `outputs/baselines/xgb_default/bootstrap_compare_<run_b_name>.json`

Hub-removed ablation:

```bash
python -m src.analysis.hub_ablation --run_dir outputs/gnn/rec_sage_resbn_k8_sin2 --frac 0.01
# Creates: metrics_hub_removed_0p01.json
```

Robustness (random edge drop + feature noise):

```bash
python -m src.analysis.robustness --run_dir outputs/gnn/rec_sage_resbn_k8_sin2 \
  --drop_frac 0.10 --noise_std 0.01
# Creates: robustness_drop0.1_noise0.01.json
```

## 7) Interpretability

XGBoost + SHAP:

```bash
python -m src.analysis.explain xgb --run_dir outputs/baselines/xgb_default --max_plots 20
# Creates: shap_summary.png, xgb_top_features.json
```

GNNExplainer (version-adaptive for PyG):

```bash
python -m src.analysis.explain gnn --run_dir outputs/gnn/rec_sage_resbn_k8_sin2 --epochs 200
# Creates: gnn_explainer_node_<id>.png, gnn_explainer_importance.json
```

> If your PyG version exposes the algorithm-only API, the script wraps it with `Explainer` and auto-configures return type.

## 8) Ensembling (k=8 + k=9)

Average probabilities or logits across two runs and emit a normal `metrics.json`:

```bash
python -m src.analysis.evaluate_ensemble \
  --run_a outputs/gnn/rec_sage_resbn_k8_sin2 \
  --run_b outputs/gnn/rec_sage_resbn_k9_sin2 \
  --out_dir outputs/gnn/rec_sage_resbn_k9_sin2/ensemble \
  --mode logit \
  --topk 100 \
  --precision_target 0.0
```

Outputs in `.../ensemble/`: `metrics.json`, `scores_test.npy`, etc.

## 9) Hyperparameter sweeps

Grid sweep (rank by late-timestep performance):

```bash
python -m src.sweeps.sweep_gnn --base configs/rec_k8.yaml --rank_key pr_auc_last3
# Writes each child run under outputs/gnn/<run_name> and a sweep summary to outputs/sweeps/
```

## 10) Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

Pick any run under `outputs/…` and browse:

* **Overview**: primary metrics table (PR-AUC_illicit, ROC-AUC, F1@thr, P@K, Recall@P, ECE, threshold)
* **Curves**: PR-AUC by time (png)
* **Drift**: interactive PR-AUC over timestep (from `by_time.csv`)
* **Calibration**: reliability diagram (png)
* **Workload**: Precision@K curve (png + interactive from `workload_curve.csv`)
* **Compare**: paired bootstrap deltas if `bootstrap_compare.json` is present in either run’s folder
* **Artifacts**: quick downloads (metrics, config, plots)

---

### Notes & tips

* **Where artifacts live:** each analysis script writes **into the run folder** you pass via `--run_dir` (or `--run_b` for bootstrap). The Streamlit app looks for those files right there.
* **Thresholding for ops:** set `precision_target` in the config if you need a fixed-precision operating point. Otherwise set `precision_target: 0.0` to use max-F1.
* **Repro:** configs include `seed` and deterministic CuDNN settings by default.
* **Speed:** AMP (`amp: true`) + GPU gives ~2–4× speedups on typical laptops/desktops.


