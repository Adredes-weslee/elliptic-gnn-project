"""Exploratory data analysis utilities for the processed Elliptic graph."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch


def load_processed_graph(
    processed_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """Load the cached graph tensors from ``processed_dir``.

    Returns:
        edge_index ([2, E] long tensor), y (long), timestep (long), meta (dict)

    Notes:
        PyTorch 2.6+ defaults torch.load(weights_only=True), which blocks unpickling
        torch_geometric Data objects. We explicitly set weights_only=False (trusted local file).
    """
    graph_path = processed_dir / "graph.pt"
    meta_path = processed_dir / "meta.json"

    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph tensor at {graph_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta information at {meta_path}")

    # Handle PyTorch 2.6+ change (weights_only default). Backwards compatible.
    import inspect

    load_sig = inspect.signature(torch.load)
    load_kwargs = {"map_location": "cpu"}
    if "weights_only" in load_sig.parameters:
        load_kwargs["weights_only"] = False

    try:
        data = torch.load(graph_path, **load_kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load {graph_path}. If this persists, ensure the file was created "
            f"with torch.save(Data(...)). Under PyTorch>=2.6 we force weights_only=False. Original error: {e}"
        ) from e

    with open(meta_path, "r") as f:
        meta = json.load(f)

    y_size = int(data.y.numel()) if hasattr(data, "y") else None
    loaded_num_nodes = int(data.num_nodes) if hasattr(data, "num_nodes") else y_size
    if (
        "num_nodes" in meta
        and loaded_num_nodes is not None
        and int(meta["num_nodes"]) != int(loaded_num_nodes)
    ):
        size_display = y_size if y_size is not None else loaded_num_nodes
        print(
            f"[WARN] meta.num_nodes={meta.get('num_nodes')} differs from loaded tensors (y size={size_display})."
        )
    # (we'll compare edges later after we compute degs)

    if not hasattr(data, "edge_index"):
        raise AttributeError("Loaded data object has no 'edge_index' attribute")
    if not hasattr(data, "y"):
        raise AttributeError("Loaded data object has no 'y' attribute")
    if not hasattr(data, "timestep"):
        raise AttributeError("Loaded data object has no 'timestep' attribute")

    edge_index = data.edge_index.to(torch.long)
    y = data.y.to(torch.long)
    timestep = data.timestep.to(torch.long)

    return edge_index, y, timestep, meta


def write_degree_histogram(
    edge_index: torch.Tensor, num_nodes: int, out_path: Path
) -> None:
    """Compute degree counts (treating edges as undirected) and persist to CSV."""

    if edge_index.numel() == 0:
        degrees = torch.zeros(num_nodes, dtype=torch.long)
    else:
        degrees = torch.bincount(edge_index.view(-1), minlength=num_nodes)

    unique_degrees, counts = torch.unique(degrees, return_counts=True, sorted=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["degree", "count"])
        for degree_value, count in zip(unique_degrees.tolist(), counts.tolist()):
            writer.writerow([int(degree_value), int(count)])


def write_labels_by_time(
    y: torch.Tensor, timestep: torch.Tensor, out_path: Path
) -> Tuple[int, int, int]:
    """Aggregate label counts per timestep and persist to CSV."""

    unique_timesteps = torch.unique(timestep, sorted=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "n_illicit", "n_licit", "n_unknown"])
        for t in unique_timesteps.tolist():
            mask = timestep == t
            y_t = y[mask]
            n_illicit = int((y_t == 1).sum())
            n_licit = int((y_t == 0).sum())
            n_unknown = int((y_t < 0).sum())
            writer.writerow([int(t), n_illicit, n_licit, n_unknown])

    total_illicit = int((y == 1).sum())
    total_licit = int((y == 0).sum())
    total_unknown = int((y < 0).sum())
    return total_illicit, total_licit, total_unknown


def assert_no_cross_time_edges(
    edge_index: torch.Tensor, timestep: torch.Tensor
) -> None:
    """Ensure every edge connects nodes from the same timestep."""

    if edge_index.numel() == 0:
        return

    src_ts = timestep[edge_index[0]]
    dst_ts = timestep[edge_index[1]]
    mismatch = torch.nonzero(src_ts != dst_ts, as_tuple=False).view(-1)

    if mismatch.numel() == 0:
        return

    print("Found cross-timestep edges (node_u, node_v, timestep_u, timestep_v):")
    max_report = 20
    for idx in mismatch.tolist()[:max_report]:
        u = int(edge_index[0, idx])
        v = int(edge_index[1, idx])
        tu = int(src_ts[idx])
        tv = int(dst_ts[idx])
        print(f"  ({u}, {v}, {tu}, {tv})")
    remaining = mismatch.numel() - max_report
    if remaining > 0:
        print(f"  ... and {remaining} more")
    sys.exit(1)


def main(processed_dir: Path, check_cross_time: bool) -> None:
    edge_index, y, timestep, meta = load_processed_graph(processed_dir)
    num_nodes = int(meta.get("num_nodes", y.size(0)))
    num_edges = int(meta.get("num_edges", edge_index.size(1)))

    if num_edges == 0:
        print(
            "[INFO] edge_index has 0 edges. This can happen if you built the graph with a partial "
            "features CSV (only a few txIds) or if edgelist txIds didn't map. "
            "Use the full elliptic_txs_features.csv to see ~234k edges."
        )
    # sanity: compare edge count to edge_index
    calc_edges = int(edge_index.size(1))
    if calc_edges != num_edges:
        print(
            f"[WARN] meta.num_edges={num_edges} but edge_index has {calc_edges}. Using edge_index count for analysis."
        )
        num_edges = calc_edges

    if check_cross_time:
        assert_no_cross_time_edges(edge_index, timestep)

    degree_csv = processed_dir / "degree_hist.csv"
    labels_csv = processed_dir / "labels_by_time.csv"

    write_degree_histogram(edge_index, num_nodes, degree_csv)
    totals = write_labels_by_time(y, timestep, labels_csv)

    print(
        "Graph summary: "
        f"nodes={num_nodes}, edges={num_edges}; "
        f"labels(illicit={totals[0]}, licit={totals[1]}, unknown={totals[2]}). "
        f"Wrote {degree_csv.name} and {labels_csv.name}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EDA and sanity checks for the processed Elliptic graph"
    )
    parser.add_argument(
        "--processed_dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing graph.pt and meta.json",
    )
    parser.add_argument(
        "--assert_no_cross_time_edges",
        action="store_true",
        help="Exit with error if any edge connects nodes from different timesteps",
    )
    args = parser.parse_args()

    main(args.processed_dir, args.assert_no_cross_time_edges)
