import argparse
import copy
import csv
import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import yaml


def parse_param_entries(entries: Sequence[Sequence[str]]) -> Tuple[List[str], List[List[Any]]]:
    names: List[str] = []
    values: List[List[Any]] = []
    for entry in entries:
        if not entry:
            continue
        if len(entry) < 2:
            raise ValueError("Each --param must include a name and at least one value")
        name, *raw_values = entry
        parsed_values = [yaml.safe_load(v) for v in raw_values]
        names.append(name)
        values.append(parsed_values)
    return names, values


def set_config_value(cfg: Dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = cfg
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def sanitize_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value).replace(".", "p")
    if value is None:
        return "none"
    return str(value).replace(" ", "")


def build_row(run_name: str, param_names: Sequence[str], param_values: Sequence[Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"run_name": run_name}
    for name, value in zip(param_names, param_values):
        row[name] = value
    return row


def gather_fieldnames(rows: Iterable[Dict[str, Any]]) -> List[str]:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def run_sweep(
    template_path: Path,
    out_csv: Path,
    param_entries: Sequence[Sequence[str]],
) -> None:
    with template_path.open("r", encoding="utf-8") as f:
        template_cfg = yaml.safe_load(f)

    if template_cfg is None:
        raise ValueError(f"Template {template_path} is empty or invalid")

    param_names, param_values = parse_param_entries(param_entries)
    if not param_names:
        raise ValueError("At least one --param must be provided")

    combos = list(itertools.product(*param_values))
    total = len(combos)

    base_run_name = template_cfg.get("run_name", "sweep")

    results: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for idx, combo in enumerate(combos, start=1):
            cfg = copy.deepcopy(template_cfg)
            for name, value in zip(param_names, combo):
                set_config_value(cfg, name, value)

            suffix = "_".join(
                f"{name}-{sanitize_value(value)}" for name, value in zip(param_names, combo)
            )
            cfg["run_name"] = f"{base_run_name}_{suffix}" if suffix else base_run_name

            config_path = Path(tmpdir) / f"config_{idx}.yaml"
            with config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f)

            print(f"[SWEEP] ({idx}/{total}) Running {cfg['run_name']}...")

            row = build_row(cfg["run_name"], param_names, combo)

            try:
                subprocess.run(
                    [sys.executable, "-m", "src.train_gnn", "--config", str(config_path)],
                    check=True,
                )
            except subprocess.CalledProcessError as exc:
                print(
                    f"[SWEEP] Run {cfg['run_name']} failed with return code {exc.returncode}",
                    file=sys.stderr,
                )
                row["run_status"] = "failed"
                row["error"] = str(exc)
            else:
                metrics_path = Path("outputs") / "gnn" / cfg["run_name"] / "metrics.json"
                if metrics_path.exists():
                    with metrics_path.open("r", encoding="utf-8") as f:
                        metrics = json.load(f)
                    if not isinstance(metrics, dict):
                        row.update({"metrics_parse_error": "metrics.json did not contain a dict"})
                        row["run_status"] = "metrics_parse_error"
                    else:
                        row.update(metrics)
                        row["run_status"] = "ok"
                else:
                    print(
                        f"[SWEEP] metrics.json not found for {cfg['run_name']} at {metrics_path}",
                        file=sys.stderr,
                    )
                    row["run_status"] = "missing_metrics"
                    row["error"] = "metrics.json not found"

            results.append(row)

    if out_csv.parent != Path(""):
        os.makedirs(out_csv.parent, exist_ok=True)

    fieldnames = gather_fieldnames(results)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[SWEEP] Wrote results to {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run small hyperparameter sweeps")
    parser.add_argument("--template", type=Path, required=True, help="Base YAML config template")
    parser.add_argument("--out_csv", type=Path, required=True, help="Path to output CSV file")
    parser.add_argument(
        "--param",
        action="append",
        nargs="+",
        default=[],
        help="Parameter sweep definition: --param name value1 value2 ...",
    )
    args = parser.parse_args()

    run_sweep(args.template, args.out_csv, args.param)


if __name__ == "__main__":
    main()
