
import os, argparse, yaml, torch
from .dataset_elliptic import load_elliptic_as_graph, make_temporal_masks
from ..utils.common import ensure_dir, save_json, set_seed

def main(cfg):
    set_seed(cfg.get("seed", 42))

    data_dir = cfg["data_dir"]
    processed_dir = cfg["processed_dir"]
    ensure_dir(processed_dir)

    data, meta = load_elliptic_as_graph(
        data_dir=data_dir,
        features_csv=cfg.get("features_csv","elliptic_txs_features.csv"),
        classes_csv=cfg.get("classes_csv","elliptic_txs_classes.csv"),
        edgelist_csv=cfg.get("edgelist_csv","elliptic_txs_edgelist.csv"),
    )

    data = make_temporal_masks(
        data,
        t_train_end=cfg["t_train_end"],
        t_val_end=cfg["t_val_end"],
    )

    # Save
    graph_path = os.path.join(processed_dir, "graph.pt")
    torch.save(data, graph_path)
    meta["graph_pt"] = graph_path
    save_json(os.path.join(processed_dir, "meta.json"), meta)
    print(f"Saved graph to {graph_path}")
    print(meta)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
