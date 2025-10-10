"""Streamlit dashboard for exploring model run outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st

RUN_TYPES = ["baselines", "gnn", "all"]
OUTPUT_ROOT = Path("outputs")


# -------------------- discovery --------------------
def scan_run_directories(root: Path = OUTPUT_ROOT) -> Dict[str, List[Path]]:
    """Return a mapping of run type to available run directories."""
    runs: Dict[str, List[Path]] = {run_type: [] for run_type in RUN_TYPES}
    if not root.exists():
        return runs

    discovered: List[Path] = []
    for metrics_file in root.rglob("metrics.json"):
        discovered.append(metrics_file.parent)

    discovered = sorted(set(discovered))
    for run_dir in discovered:
        relative = run_dir.relative_to(root)
        top_level = relative.parts[0] if relative.parts else ""
        if top_level in ("baselines", "gnn"):
            runs[top_level].append(run_dir)
        runs["all"].append(run_dir)

    for key in runs:
        runs[key] = sorted(set(runs[key]))
    return runs


def format_run_option(run_dir: Path) -> str:
    try:
        return str(run_dir.relative_to(OUTPUT_ROOT))
    except ValueError:
        return str(run_dir)


# -------------------- io helpers --------------------
def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        st.error(f"Failed to decode JSON from {path}: {exc}")
        return None


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to read CSV {path}: {exc}")
        return None


def nested_get(mapping: Dict, keys: Iterable[str]) -> Optional[float]:
    cur = mapping
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    if isinstance(cur, (int, float)):
        return float(cur)
    return None


# -------------------- metrics table --------------------
def metrics_table(metrics: Dict) -> pd.DataFrame:
    rows: List[Tuple[str, Optional[float]]] = []

    # Map display label -> possible keys in your various writers
    value_map = {
        "PR-AUC (illicit)": ["pr_auc_illicit", "pr_auc"],
        "ROC-AUC": ["roc_auc"],
        "F1@thr (illicit)": ["f1_illicit_at_thr", "f1_at_threshold", "f1"],
        "P@K": ["precision_at_k", "p_at_k", "precision_at_k_value"],
        "Recall@P": ["recall_at_precision", "recall_at_p"],
        "ECE": ["ece", "expected_calibration_error"],
        "thr": ["threshold", "thr", "decision_threshold"],
        "best_val_pr_auc": ["best_val_pr_auc"],
        "pr_auc_last1": ["pr_auc_last1"],
        "pr_auc_last3": ["pr_auc_last3"],
        "pr_auc_last5": ["pr_auc_last5"],
        "n_test": ["n_test"],
    }

    for label, keys in value_map.items():
        value = None
        for key in keys:
            if key in metrics:
                v = metrics.get(key)
                if isinstance(v, dict):
                    # handle nested {"value": ...} or similar shapes
                    value = v.get("value") or v.get("score") or v.get("mean")
                elif isinstance(v, (int, float)):
                    value = v
                if value is not None:
                    break
        if value is None:
            value = nested_get(metrics, [keys[0], "value"])
        rows.append((label, None if value is None else float(value)))

    return pd.DataFrame(rows, columns=["Metric", "Value"])


def load_hub_ablation(run_dir: Path) -> Optional[pd.DataFrame]:
    # Your training script writes "metrics_hub_removed.json"
    path = run_dir / "metrics_hub_removed.json"
    data = load_json(path)
    if not data:
        return None
    # Normalize to a one-row table
    if isinstance(data, dict):
        return pd.DataFrame([data])
    try:
        return pd.DataFrame(data)
    except Exception:  # noqa: BLE001
        return None


# -------------------- tabs --------------------
def show_overview_tab(
    run_dir: Path, threshold_source: str, custom_threshold: float
) -> None:
    st.markdown(f"**Run:** `{run_dir}`")

    metrics_path = run_dir / "metrics.json"
    metrics = load_json(metrics_path)
    if not metrics:
        st.warning("metrics.json was not found or could not be parsed for this run.")
        return

    # Some writers nest under "metrics"/"test"/"results" – flatten if present
    metrics_section = metrics
    for candidate in ("metrics", "test", "results"):
        if isinstance(metrics.get(candidate), dict):
            metrics_section = metrics[candidate]
            break

    df = metrics_table(metrics_section)
    st.subheader("Primary metrics")
    st.dataframe(df.set_index("Metric"))

    if threshold_source == "Use slider":
        st.caption(f"Using custom decision threshold: {custom_threshold:.3f}")
    else:
        thr_row = df.loc[df["Metric"] == "thr"]
        if not thr_row.empty and pd.notnull(thr_row["Value"].iloc[0]):
            st.caption(f"Run decision threshold: {thr_row['Value'].iloc[0]:.3f}")

    hub_ablation = load_hub_ablation(run_dir)
    if hub_ablation is not None and not hub_ablation.empty:
        st.subheader("Hub ablation metrics")
        st.dataframe(hub_ablation)


def show_curves_tab(run_dir: Path) -> None:
    # quick visuals that are typically present
    for name, caption in [
        ("by_time_pr_auc.png", "PR-AUC by time"),
        ("calibration_curve.png", "Calibration curve"),
        ("workload_curve.png", "Workload Precision@K curve"),
    ]:
        p = run_dir / name
        if p.exists():
            st.image(str(p), caption=caption)


def show_drift_tab(run_dir: Path, enabled: bool) -> None:
    if not enabled:
        st.info("Enable 'Show by timestep' in the sidebar to view drift plots.")
        return

    csv_path = run_dir / "by_time.csv"
    df = load_csv(csv_path)
    if df is None or df.empty:
        st.info("No by_time.csv file available for drift analysis.")
        return

    metric_column = None
    for candidate in ("pr_auc", "pr-auc", "pr_auc_score"):
        if candidate in df.columns:
            metric_column = candidate
            break

    timestep_column = None
    for candidate in ("timestep", "time", "step"):
        if candidate in df.columns:
            timestep_column = candidate
            break

    if not metric_column or not timestep_column:
        st.warning(
            "by_time.csv does not contain recognizable timestep or PR-AUC columns."
        )
        st.dataframe(df.head())
        return

    fig = px.line(df, x=timestep_column, y=metric_column, title="PR-AUC over time")
    st.plotly_chart(fig, use_container_width=True)


def show_calibration_tab(run_dir: Path, enabled: bool) -> None:
    if not enabled:
        st.info("Enable 'Show calibration' in the sidebar to view calibration plots.")
        return

    image_path = run_dir / "calibration_curve.png"
    if image_path.exists():
        st.image(str(image_path), caption="Calibration curve")
    else:
        st.info("No calibration curve available.")


def show_workload_tab(run_dir: Path, enabled: bool) -> None:
    if not enabled:
        st.info("Enable 'Show workload' in the sidebar to view workload plots.")
        return

    image_path = run_dir / "workload_curve.png"
    csv_path = run_dir / "workload_curve.csv"  # <-- your writer uses this name

    if image_path.exists():
        st.image(str(image_path), caption="Workload curve")
    else:
        st.info("No workload curve image available.")

    df = load_csv(csv_path)
    if df is not None and not df.empty:
        # Column names you produce: typically 'k' and 'precision'
        x_col = next((c for c in ("k", "K", "top_k") if c in df.columns), None)
        y_col = next(
            (c for c in ("precision", "precision_at_k", "p@k") if c in df.columns), None
        )
        if x_col and y_col:
            fig = px.line(df, x=x_col, y=y_col, title="Precision vs K (from CSV)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                "workload_curve.csv does not contain recognizable columns for plotting."
            )
            st.dataframe(df.head())


def extract_delta(
    data: Dict, keys: Iterable[str]
) -> Optional[Tuple[float, Optional[float], Optional[float]]]:
    for key in keys:
        if key not in data:
            continue
        entry = data[key]
        if isinstance(entry, dict):
            delta = (
                entry.get("delta")
                or entry.get("estimate")
                or entry.get("mean")
                or entry.get("value")
            )
            lower = entry.get("ci_low") or entry.get("lower") or entry.get("low")
            upper = entry.get("ci_high") or entry.get("upper") or entry.get("high")
            if delta is not None:
                return float(delta), _optional_float(lower), _optional_float(upper)
        elif isinstance(entry, (int, float)):
            return float(entry), None, None
    return None


def _optional_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def show_compare_tab(run_dir: Path, comparison_dir: Optional[Path]) -> None:
    candidates = [run_dir / "bootstrap_compare.json"]
    if comparison_dir is not None:
        candidates.append(comparison_dir / "bootstrap_compare.json")
        candidates.append(run_dir / f"bootstrap_compare_{comparison_dir.name}.json")
        candidates.append(comparison_dir / f"bootstrap_compare_{run_dir.name}.json")

    compare_data = None
    chosen_path = None
    for path in candidates:
        if path.exists():
            compare_data = load_json(path)
            chosen_path = path
            if compare_data:
                break

    if not compare_data:
        st.info("No bootstrap comparison data available for the selected runs.")
        return

    if chosen_path:
        try:
            caption_path = chosen_path.relative_to(Path.cwd())
        except ValueError:
            caption_path = chosen_path
        st.caption(f"Loaded comparison from {caption_path}")

    rows = []
    pr_auc = extract_delta(compare_data, ["delta_pr_auc", "pr_auc", "pr-auc"])
    if pr_auc:
        rows.append(("ΔPR-AUC",) + pr_auc)
    p_at_k = extract_delta(compare_data, ["delta_p_at_k", "p_at_k", "precision_at_k"])
    if p_at_k:
        rows.append(("ΔP@K",) + p_at_k)

    if not rows:
        st.warning("Bootstrap comparison JSON does not contain ΔPR-AUC or ΔP@K data.")
        st.json(compare_data)
        return

    df = pd.DataFrame(rows, columns=["Metric", "Δ", "CI Low", "CI High"])
    st.table(df.set_index("Metric"))


def show_artifacts_tab(run_dir: Path) -> None:
    # Prioritize human-readable first
    priority = [
        "metrics.json",
        "config_used.yaml",
        "by_time.csv",
        "workload_curve.csv",
        "training_log.csv",
        "metrics_hub_removed.json",
        "robustness_*.json",
        "bootstrap_compare*.json",
        "*.png",
        "*.csv",
        "*.npy",
        "best.ckpt",
    ]

    files: List[Path] = []
    for pattern in priority:
        files.extend(sorted(run_dir.glob(pattern)))

    # De-duplicate while preserving order
    seen = set()
    artifact_paths = []
    for p in files:
        if p.exists() and p.is_file() and p not in seen:
            seen.add(p)
            artifact_paths.append(p)

    if not artifact_paths:
        st.info("No downloadable artifacts found in the run directory.")
        return

    for path in artifact_paths:
        with path.open("rb") as file:
            st.download_button(
                label=f"Download {path.name}",
                data=file.read(),
                file_name=path.name,
                mime="application/octet-stream",
            )


# -------------------- app --------------------
def main() -> None:
    st.set_page_config(page_title="Run Dashboard", layout="wide")
    st.title("Run Metrics Dashboard")

    run_directories = scan_run_directories()

    with st.sidebar:
        st.header("Run selection")
        run_type = st.selectbox("Run type", RUN_TYPES, index=RUN_TYPES.index("all"))

        options = (
            run_directories.get(run_type, [])
            if run_type != "all"
            else run_directories["all"]
        )
        option_labels = [format_run_option(path) for path in options]
        label_to_path = dict(zip(option_labels, options))
        if option_labels:
            selected_label = st.selectbox("Run", option_labels)
            selected_dir = label_to_path.get(selected_label)
        else:
            st.warning("No runs found for the selected run type.")
            selected_dir = None

        comparison_dir: Optional[Path] = None
        comparison_options = run_directories["all"]
        if comparison_options:
            comparison_labels = ["None"] + [
                format_run_option(path) for path in comparison_options
            ]
            selected_comparison = st.selectbox(
                "Comparison run (optional)", comparison_labels, index=0
            )
            if selected_comparison != "None":
                comparison_map = dict(zip(comparison_labels[1:], comparison_options))
                comparison_dir = comparison_map.get(selected_comparison)

        threshold_source = st.radio(
            "Threshold source", ["Use run threshold", "Use slider"], index=0
        )
        custom_threshold = st.slider(
            "Custom threshold",
            0.0,
            1.0,
            0.5,
            0.01,
            disabled=threshold_source != "Use slider",
        )
        show_by_timestep = st.checkbox("Show by timestep", value=True)
        show_calibration = st.checkbox("Show calibration", value=True)
        show_workload = st.checkbox("Show workload", value=True)

    if selected_dir is None:
        if run_directories["all"]:
            st.info(
                "Select a different run type or generate metrics for the chosen category."
            )
        else:
            st.info(
                "No runs found. Add run outputs under the `outputs/` directory to begin."
            )
        return

    tabs = st.tabs(
        [
            "Overview",
            "Curves",
            "Drift",
            "Calibration",
            "Workload",
            "Compare",
            "Artifacts",
        ]
    )

    with tabs[0]:
        show_overview_tab(selected_dir, threshold_source, custom_threshold)
    with tabs[1]:
        show_curves_tab(selected_dir)
    with tabs[2]:
        show_drift_tab(selected_dir, show_by_timestep)
    with tabs[3]:
        show_calibration_tab(selected_dir, show_calibration)
    with tabs[4]:
        show_workload_tab(selected_dir, show_workload)
    with tabs[5]:
        show_compare_tab(selected_dir, comparison_dir)
    with tabs[6]:
        show_artifacts_tab(selected_dir)


if __name__ == "__main__":
    main()
