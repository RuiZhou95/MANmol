from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv, global_mean_pool
from transformers import AutoModel

from manmol.models import GraphEncoder


class DescriptorMLP(nn.Module):
    def __init__(self, descriptor_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(descriptor_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, batch: dict):
        return self.network(batch["descriptors"].float()).squeeze(-1)


class GraphGINERegressor(nn.Module):
    def __init__(self, node_input_dim: int, hidden_size: int, edge_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.encoder = GraphEncoder(node_input_dim, hidden_size, edge_dim, num_layers, dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, batch: dict):
        graph_repr = self.encoder(batch["graphs"])
        return self.regressor(graph_repr).squeeze(-1)


class GraphSAGERegressor(nn.Module):
    def __init__(self, node_input_dim: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.node_proj = nn.Linear(node_input_dim, hidden_size)
        self.convs = nn.ModuleList([SAGEConv(hidden_size, hidden_size) for _ in range(num_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, batch: dict):
        graphs = batch["graphs"]
        x = self.node_proj(graphs.x.float())
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, graphs.edge_index)
            x = norm(torch.relu(x) + residual)
            x = self.dropout(x)
        graph_repr = global_mean_pool(x, graphs.batch)
        return self.regressor(graph_repr).squeeze(-1)


class SmilesChemBERTaRegressor(nn.Module):
    def __init__(self, smiles_model_name: str, hidden_size: int, dropout: float, freeze_smiles_encoder: bool = False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(smiles_model_name)
        if freeze_smiles_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        encoder_hidden = self.encoder.config.hidden_size
        self.attention = nn.MultiheadAttention(encoder_hidden, num_heads=8, batch_first=True)
        self.regressor = nn.Sequential(
            nn.Linear(encoder_hidden, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, batch: dict):
        attention_mask = batch["attention_mask"]
        outputs = self.encoder(input_ids=batch["input_ids"], attention_mask=attention_mask).last_hidden_state
        attn_output, _ = self.attention(
            outputs,
            outputs,
            outputs,
            key_padding_mask=~attention_mask.bool(),
            need_weights=False,
        )
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (attn_output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.regressor(pooled).squeeze(-1)


def build_baseline(config: dict, descriptor_size: int) -> nn.Module:
    baseline = config["baseline"]
    model_cfg = config["model"]
    if baseline == "descriptor_mlp":
        return DescriptorMLP(descriptor_size, model_cfg["hidden_size"], model_cfg["dropout"])
    if baseline == "graph_gine":
        return GraphGINERegressor(
            node_input_dim=model_cfg.get("node_input_dim", 3),
            hidden_size=model_cfg["hidden_size"],
            edge_dim=model_cfg["gnn_edge_dim"],
            num_layers=model_cfg["gnn_layers"],
            dropout=model_cfg["dropout"],
        )
    if baseline == "graph_sage":
        return GraphSAGERegressor(
            node_input_dim=model_cfg.get("node_input_dim", 3),
            hidden_size=model_cfg["hidden_size"],
            num_layers=model_cfg["gnn_layers"],
            dropout=model_cfg["dropout"],
        )
    if baseline == "smiles_chemberta":
        return SmilesChemBERTaRegressor(
            smiles_model_name=model_cfg["smiles_model_name"],
            hidden_size=model_cfg["hidden_size"],
            dropout=model_cfg["dropout"],
            freeze_smiles_encoder=model_cfg.get("freeze_smiles_encoder", False),
        )
    raise ValueError(f"Unsupported baseline: {baseline}")
