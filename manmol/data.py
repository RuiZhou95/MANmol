from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Batch, Data

from manmol.graph_features import smiles_to_rich_graph


class MultiModalDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer, max_length: int):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        tokens = self.tokenizer(
            row["smiles"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        graph = row["graph"].clone()
        graph.y = torch.tensor([row["target"]], dtype=torch.float32)
        graph.mol_id = int(row["ID"])
        return {
            "id": int(row["ID"]),
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "descriptors": torch.tensor(row["descriptors"], dtype=torch.float32),
            "graph": graph,
            "target": torch.tensor(row["target"], dtype=torch.float32),
        }


def collate_multimodal(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "ids": torch.tensor([item["id"] for item in batch], dtype=torch.long),
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "descriptors": torch.stack([item["descriptors"] for item in batch]),
        "graphs": Batch.from_data_list([item["graph"] for item in batch]),
        "targets": torch.stack([item["target"] for item in batch]),
    }


def make_graph_undirected(edge_index: torch.Tensor, edge_attr: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
    if edge_index.numel() == 0:
        return edge_index, edge_attr
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    if edge_attr is not None and edge_attr.numel():
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    return edge_index, edge_attr


def load_graphs(graph_pkl: str | Path, make_undirected: bool = False) -> dict[int, Data]:
    with open(graph_pkl, "rb") as handle:
        graph_rows = pickle.load(handle)

    graphs: dict[int, Data] = {}
    for row in graph_rows:
        x, edge_index, edge_attr = row["graph"]
        edge_index = edge_index.long()
        edge_attr = edge_attr.float() if edge_attr is not None and edge_attr.numel() else None
        if make_undirected:
            edge_index, edge_attr = make_graph_undirected(edge_index, edge_attr)
        graph = Data(
            x=x.float(),
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
        graphs[int(row["ID"])] = graph
    return graphs


def load_clean_rows(config: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], list[str], StandardScaler]:
    data_cfg = config["data"]
    descriptor_df = pd.read_csv(data_cfg["descriptor_csv"])
    split_df = pd.read_csv(data_cfg["split_csv"])
    graph_cfg = config.get("graph", {})

    id_col = data_cfg.get("id_column", "ID")
    target_col = data_cfg.get("target_column", "target")
    smiles_col = data_cfg.get("smiles_column", "smiles")
    graph_by_id = None
    if graph_cfg.get("source", "legacy") == "legacy":
        graph_by_id = load_graphs(data_cfg["graph_pkl"], make_undirected=graph_cfg.get("make_undirected", False))
    exclude = set(data_cfg.get("descriptor_exclude_columns", []))
    descriptor_columns = [col for col in descriptor_df.columns if col not in exclude]

    merged = descriptor_df.merge(split_df, on=id_col, how="inner")
    if graph_by_id is not None:
        merged = merged[merged[id_col].map(lambda mol_id: int(mol_id) in graph_by_id)].copy()

    train_mask = merged["split"] == "train"
    scaler = StandardScaler()
    scaler.fit(merged.loc[train_mask, descriptor_columns].astype(float))
    merged.loc[:, descriptor_columns] = scaler.transform(merged[descriptor_columns].astype(float))

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    for record in merged.to_dict("records"):
        mol_id = int(record[id_col])
        smiles = record[smiles_col]
        graph = graph_by_id[mol_id] if graph_by_id is not None else smiles_to_rich_graph(smiles)
        split_rows[record["split"]].append(
            {
                "ID": mol_id,
                "smiles": smiles,
                "target": float(record[target_col]),
                "descriptors": [float(record[col]) for col in descriptor_columns],
                "graph": graph,
            }
        )
    return split_rows, descriptor_columns, scaler


def make_loaders(config: dict[str, Any], tokenizer) -> tuple[dict[str, torch.utils.data.DataLoader], list[str]]:
    split_rows, descriptor_columns, _ = load_clean_rows(config)
    train_cfg = config["training"]
    datasets = {
        split: MultiModalDataset(rows, tokenizer, train_cfg["max_length"])
        for split, rows in split_rows.items()
    }
    loaders = {
        "train": torch.utils.data.DataLoader(
            datasets["train"],
            batch_size=train_cfg["batch_size"],
            shuffle=True,
            num_workers=train_cfg.get("num_workers", 0),
            collate_fn=collate_multimodal,
        ),
        "val": torch.utils.data.DataLoader(
            datasets["val"],
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 0),
            collate_fn=collate_multimodal,
        ),
        "test": torch.utils.data.DataLoader(
            datasets["test"],
            batch_size=train_cfg["batch_size"],
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 0),
            collate_fn=collate_multimodal,
        ),
    }
    return loaders, descriptor_columns
