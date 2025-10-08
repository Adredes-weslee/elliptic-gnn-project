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


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as exc:
        st.error(f"Failed to decode JSON from {path}: {exc}")
        return None


def nested_get(mapping: Dict, keys: Iterable[str]) -> Optional[float]:
    for key in keys:
        if mapping is None:
            return None
        if isinstance(mapping, dict):
            mapping = mapping.get(key)
        else:
            return None
    if isinstance(mapping, (int, float)):
        return float(mapping)
    return None


def metrics_table(metrics: Dict) -> pd.DataFrame:
    rows: List[Tuple[str, Optional[float]]] = []

    value_map = {
        "PR-AUC": ["pr_auc"],
        "ROC-AUC": ["roc_auc"],
        "F1@thr": ["f1_at_threshold", "f1"],
        "P@K": ["precision_at_k", "p_at_k", "precision_at_k_value"],
        "Recall@P": ["recall_at_precision", "recall_at_p"],
        "ECE": ["ece", "expected_calibration_error"],
        "thr": ["threshold", "thr", "decision_threshold"],
    }

    for label, keys in value_map.items():
        value = None
        for key in keys:
            if key in metrics:
                candidate = metrics.get(key)
                if isinstance(candidate, dict):
                    value = (
                        candidate.get("value")
                        or candidate.get("score")
                        or candidate.get("precision")
                        or candidate.get("recall")
                    )
                elif isinstance(candidate, (int, float)):
                    value = candidate
                if value is not None:
                    break
        if value is None:
            # Attempt nested access such as precision_at_k -> value.
            value = nested_get(metrics, [keys[0], "value"])
        rows.append((label, value))

    df = pd.DataFrame(rows, columns=["Metric", "Value"])
    return df


def load_hub_ablation(run_dir: Path) -> Optional[pd.DataFrame]:
    path = run_dir / "hub_ablation_metrics.json"
    data = load_json(path)
    if not data:
        return None
    return pd.DataFrame(data)


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pylint: disable=broad-except
        st.error(f"Failed to read CSV {path}: {exc}")
        return None


def show_overview_tab(run_dir: Path, threshold_source: str, custom_threshold: float) -> None:
    metrics_path = run_dir / "metrics.json"
    metrics = load_json(metrics_path)

    if not metrics:
        st.warning("metrics.json was not found or could not be parsed for this run.")
        return

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
    elif "thr" in df["Metric"].values:
        thr_value = df.loc[df["Metric"] == "thr", "Value"].iloc[0]
        if thr_value is not None:
            st.caption(f"Run decision threshold: {thr_value:.3f}")

    hub_ablation = load_hub_ablation(run_dir)
    if hub_ablation is not None and not hub_ablation.empty:
        st.subheader("Hub ablation metrics")
        st.dataframe(hub_ablation)


def show_curves_tab(run_dir: Path) -> None:
    image_path = run_dir / "by_time_pr_auc.png"
    if image_path.exists():
        st.image(str(image_path), caption="PR-AUC by time")
    else:
        st.info("No PR-AUC by time image available.")


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
        st.warning("by_time.csv does not contain recognizable timestep or PR-AUC columns.")
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
    csv_path = run_dir / "workload.csv"

    if image_path.exists():
        st.image(str(image_path), caption="Workload curve")
    else:
        st.info("No workload curve image available.")

    df = load_csv(csv_path)
    if df is not None and not df.empty:
        x_col = None
        y_col = None
        for candidate in ("k", "K", "top_k"):
            if candidate in df.columns:
                x_col = candidate
                break
        for candidate in ("precision", "precision_at_k", "p@k"):
            if candidate in df.columns:
                y_col = candidate
                break
        if x_col and y_col:
            fig = px.line(df, x=x_col, y=y_col, title="Precision vs K")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("workload.csv does not contain recognizable columns for plotting precision vs K.")


def extract_delta(data: Dict, keys: Iterable[str]) -> Optional[Tuple[float, Optional[float], Optional[float]]]:
    for key in keys:
        if key not in data:
            continue
        entry = data[key]
        if isinstance(entry, dict):
            delta = entry.get("delta") or entry.get("estimate") or entry.get("mean") or entry.get("value")
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
    artifact_paths = []
    for candidate in ("metrics.json", "config_used.yaml"):
        path = run_dir / candidate
        if path.exists():
            artifact_paths.append(path)

    artifact_paths.extend(sorted(run_dir.glob("*.png")))

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


def main() -> None:
    st.set_page_config(page_title="Run Dashboard", layout="wide")
    st.title("Run Metrics Dashboard")

    run_directories = scan_run_directories()

    with st.sidebar:
        st.header("Run selection")
        run_type = st.selectbox("Run type", RUN_TYPES, index=RUN_TYPES.index("all"))

        options = run_directories.get(run_type, []) if run_type != "all" else run_directories["all"]
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
            comparison_labels = ["None"] + [format_run_option(path) for path in comparison_options]
            selected_comparison = st.selectbox("Comparison run (optional)", comparison_labels, index=0)
            if selected_comparison != "None":
                comparison_map = dict(zip(comparison_labels[1:], comparison_options))
                comparison_dir = comparison_map.get(selected_comparison)

        threshold_source = st.radio("Threshold source", ["Use run threshold", "Use slider"], index=0)
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
            st.info("Select a different run type or generate metrics for the chosen category.")
        else:
            st.info("No runs found. Add run outputs under the `outputs/` directory to begin.")
        return

    tabs = st.tabs(["Overview", "Curves", "Drift", "Calibration", "Workload", "Compare", "Artifacts"])

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
