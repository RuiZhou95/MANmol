from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import torch


def edge_reverse_stats(edge_index: torch.Tensor) -> tuple[int, int]:
    if edge_index.numel() == 0:
        return 0, 0
    edges = {(int(src), int(dst)) for src, dst in edge_index.t().tolist()}
    missing = sum(1 for src, dst in edges if (dst, src) not in edges)
    return missing, len(edges)


def summarize(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "min": float(tensor.min()),
        "median": float(tensor.median()),
        "mean": float(tensor.mean()),
        "max": float(tensor.max()),
    }


def audit_graphs(graph_pkl: Path) -> dict:
    with graph_pkl.open("rb") as handle:
        rows = pickle.load(handle)

    num_nodes = []
    num_edges = []
    node_dims = set()
    edge_dims = set()
    empty_edge_graphs = 0
    invalid_edge_graphs = 0
    missing_reverse_edges = 0
    total_unique_edges = 0
    graphs_with_missing_reverse = 0
    node_min = []
    node_max = []
    edge_attr_min = []
    edge_attr_max = []

    for row in rows:
        x, edge_index, edge_attr = row["graph"]
        x = x.float()
        edge_index = edge_index.long()
        edge_attr = edge_attr.float() if edge_attr is not None else None

        num_nodes.append(int(x.size(0)))
        num_edges.append(int(edge_index.size(1)))
        node_dims.add(int(x.size(1)) if x.dim() > 1 else 1)
        if edge_attr is None or edge_attr.numel() == 0:
            edge_dims.add(0)
            empty_edge_graphs += 1
        else:
            edge_dims.add(int(edge_attr.size(1)) if edge_attr.dim() > 1 else 1)
            edge_attr_min.append(float(edge_attr.min()))
            edge_attr_max.append(float(edge_attr.max()))

        if edge_index.numel() == 0:
            empty_edge_graphs += 1
        elif int(edge_index.max()) >= x.size(0) or int(edge_index.min()) < 0:
            invalid_edge_graphs += 1

        missing, total = edge_reverse_stats(edge_index)
        missing_reverse_edges += missing
        total_unique_edges += total
        if missing:
            graphs_with_missing_reverse += 1

        node_min.append(float(x.min()))
        node_max.append(float(x.max()))

    return {
        "graph_pkl": str(graph_pkl),
        "num_graphs": len(rows),
        "num_nodes": summarize(num_nodes),
        "num_edges": summarize(num_edges),
        "node_feature_dims": sorted(node_dims),
        "edge_feature_dims": sorted(edge_dims),
        "empty_edge_graphs": empty_edge_graphs,
        "invalid_edge_graphs": invalid_edge_graphs,
        "graphs_with_missing_reverse_edges": graphs_with_missing_reverse,
        "missing_reverse_edges": missing_reverse_edges,
        "total_unique_edges": total_unique_edges,
        "missing_reverse_edge_fraction": missing_reverse_edges / total_unique_edges if total_unique_edges else 0.0,
        "node_feature_min": min(node_min),
        "node_feature_max": max(node_max),
        "edge_attr_min": min(edge_attr_min) if edge_attr_min else None,
        "edge_attr_max": max(edge_attr_max) if edge_attr_max else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-pkl", default="data/all_data_ID-smiles-graph-13320.pkl")
    parser.add_argument("--output", default="outputs/audits/graph_data_audit.json")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    graph_pkl = Path(args.graph_pkl)
    output = Path(args.output)
    if not graph_pkl.is_absolute():
        graph_pkl = root / graph_pkl
    if not output.is_absolute():
        output = root / output

    result = audit_graphs(graph_pkl)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
