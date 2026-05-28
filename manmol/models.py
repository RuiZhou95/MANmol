from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool
from transformers import AutoModel


ALL_MODALITIES = ("smiles", "descriptor", "graph")
FUSION_ALIASES = {"weighted_sum": "gated", "equal_weight": "equal"}


class GraphEncoder(nn.Module):
    def __init__(self, node_input_dim: int, hidden_size: int, edge_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.node_proj = nn.Linear(node_input_dim, hidden_size)
        self.edge_proj = nn.Linear(edge_dim, hidden_size)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            self.convs.append(GINEConv(mlp, edge_dim=hidden_size))
            self.norms.append(nn.LayerNorm(hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, graph_batch):
        x = self.node_proj(graph_batch.x.float())
        edge_attr = graph_batch.edge_attr
        if edge_attr is None:
            edge_attr = torch.ones((graph_batch.edge_index.size(1), 1), device=x.device)
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        edge_attr = self.edge_proj(edge_attr.float())

        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = conv(x, graph_batch.edge_index, edge_attr)
            x = norm(torch.relu(x) + residual)
            x = self.dropout(x)
        return global_mean_pool(x, graph_batch.batch)


class CrossModalAttention(nn.Module):
    """Cross-modal attention: each modality attends to all others.

    For N modalities, creates N cross-attention layers. Each layer
    uses one modality as query and the concatenation of all other
    modalities as key/value.
    """

    def __init__(self, hidden_size: int, num_modalities: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_modalities = num_modalities
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads=num_heads, batch_first=True, dropout=dropout
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, representations: list[torch.Tensor]) -> list[torch.Tensor]:
        """Each modality representation cross-attends to all others.

        Args:
            representations: list of [batch, hidden_size] tensors

        Returns:
            list of cross-attended [batch, hidden_size] tensors
        """
        n = len(representations)
        if n < 2:
            return representations

        updated = []
        for i in range(n):
            query = representations[i].unsqueeze(1)  # [batch, 1, hidden]
            others = torch.stack(
                [representations[j] for j in range(n) if j != i], dim=1
            )  # [batch, n-1, hidden]

            attended, _ = self.cross_attn(query, others, others, need_weights=False)
            attended = attended.squeeze(1)  # [batch, hidden]
            updated.append(self.norm(representations[i] + self.dropout(attended)))

        return updated


class MANmolClean(nn.Module):
    def __init__(
        self,
        smiles_model_name: str,
        descriptor_size: int,
        node_input_dim: int = 3,
        hidden_size: int = 128,
        gnn_layers: int = 3,
        gnn_edge_dim: int = 1,
        dropout: float = 0.1,
        freeze_smiles_encoder: bool = False,
        modalities: list[str] | None = None,
        fusion: str = "gated",
        gate_temperature: float = 1.0,
        modality_layer_norm: bool = False,
        modality_dropout: float = 0.0,
    ):
        super().__init__()
        self.modalities = self._normalize_modalities(modalities)
        self.fusion = FUSION_ALIASES.get(fusion, fusion)
        if self.fusion not in {"gated", "equal", "concat", "channel_gate", "cross_attend"}:
            raise ValueError(f"Unsupported fusion mode: {fusion}")
        if gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")
        self.gate_temperature = gate_temperature
        self._last_gate = None
        self.hidden_size = hidden_size
        self.modality_layer_norm = modality_layer_norm
        self.modality_dropout = modality_dropout

        if "smiles" in self.modalities:
            self.smiles_encoder = AutoModel.from_pretrained(smiles_model_name)
            smiles_hidden = self.smiles_encoder.config.hidden_size
            if freeze_smiles_encoder:
                for parameter in self.smiles_encoder.parameters():
                    parameter.requires_grad = False
            self.smiles_attention = nn.MultiheadAttention(smiles_hidden, num_heads=8, batch_first=True)
            self.smiles_proj = nn.Sequential(
                nn.Linear(smiles_hidden, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        if "descriptor" in self.modalities:
            self.descriptor_encoder = nn.Sequential(
                nn.Linear(descriptor_size, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, hidden_size),
                nn.ReLU(),
            )
        if "graph" in self.modalities:
            self.graph_encoder = GraphEncoder(node_input_dim, hidden_size, gnn_edge_dim, gnn_layers, dropout)

        # Per-modality LayerNorm
        if modality_layer_norm:
            self.modality_norms = nn.ModuleDict()
            for m in self.modalities:
                self.modality_norms[m] = nn.LayerNorm(hidden_size)

        if self.fusion == "gated":
            self.gate = nn.Sequential(
                nn.Linear(hidden_size * len(self.modalities), hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, len(self.modalities)),
            )
        elif self.fusion == "channel_gate":
            concat_dim = hidden_size * len(self.modalities)
            self.gate = nn.Sequential(
                nn.Linear(concat_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, concat_dim),
            )
        elif self.fusion == "cross_attend":
            self.cross_attn = CrossModalAttention(
                hidden_size=hidden_size,
                num_modalities=len(self.modalities),
                num_heads=4,
                dropout=dropout,
            )

        regressor_input_dim = hidden_size * len(self.modalities) if self.fusion in ("concat", "channel_gate", "cross_attend") else hidden_size
        self.regressor = nn.Sequential(
            nn.Linear(regressor_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    @staticmethod
    def _normalize_modalities(modalities: list[str] | None) -> list[str]:
        if modalities is None:
            return list(ALL_MODALITIES)
        if not modalities:
            raise ValueError("At least one modality must be active")
        unknown = sorted(set(modalities) - set(ALL_MODALITIES))
        if unknown:
            raise ValueError(f"Unknown modalities: {unknown}")
        if len(set(modalities)) != len(modalities):
            raise ValueError("Duplicate modalities are not allowed")
        return [name for name in ALL_MODALITIES if name in modalities]

    def encode_smiles(self, input_ids, attention_mask):
        smiles_output = self.smiles_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        smiles_attn, _ = self.smiles_attention(
            smiles_output,
            smiles_output,
            smiles_output,
            key_padding_mask=~attention_mask.bool(),
            need_weights=False,
        )
        mask = attention_mask.unsqueeze(-1).float()
        smiles_repr = (smiles_attn * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.smiles_proj(smiles_repr)

    def encode_modalities(self, input_ids, attention_mask, descriptors, graphs) -> list[torch.Tensor]:
        representations = []
        for modality in self.modalities:
            if modality == "smiles":
                representations.append(self.encode_smiles(input_ids, attention_mask))
            elif modality == "descriptor":
                representations.append(self.descriptor_encoder(descriptors.float()))
            else:
                representations.append(self.graph_encoder(graphs))
        return representations

    def forward(self, input_ids, attention_mask, descriptors, graphs):
        representations = self.encode_modalities(input_ids, attention_mask, descriptors, graphs)

        # Per-modality LayerNorm
        if self.modality_layer_norm:
            for i, m in enumerate(self.modalities):
                representations[i] = self.modality_norms[m](representations[i])

        # Modality dropout (training only)
        if self.training and self.modality_dropout > 0:
            for i in range(len(representations)):
                if torch.rand(1, device=representations[i].device).item() < self.modality_dropout:
                    representations[i] = torch.zeros_like(representations[i])

        if self.fusion == "concat":
            fused = torch.cat(representations, dim=-1)
            weights = torch.full(
                (fused.size(0), len(representations)),
                1.0 / len(representations),
                device=fused.device,
            )
        elif self.fusion == "channel_gate":
            concat = torch.cat(representations, dim=-1)
            gate = torch.sigmoid(self.gate(concat))
            self._last_gate = gate
            fused = gate * concat
            splits = [self.hidden_size] * len(self.modalities)
            gate_splits = torch.split(gate, splits, dim=-1)
            weights = torch.stack([g.mean(dim=-1) for g in gate_splits], dim=-1)
        elif self.fusion == "cross_attend":
            cross_reprs = self.cross_attn(representations)
            fused = torch.cat(cross_reprs, dim=-1)
            weights = torch.full(
                (fused.size(0), len(representations)),
                1.0 / len(representations),
                device=fused.device,
            )
        else:
            stacked = torch.stack(representations, dim=1)
            if self.fusion == "equal":
                weights = torch.full(
                    (stacked.size(0), len(representations)),
                    1.0 / len(representations),
                    device=stacked.device,
                )
            else:
                gate_logits = self.gate(torch.cat(representations, dim=-1))
                weights = torch.softmax(gate_logits / self.gate_temperature, dim=-1)
            fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        prediction = self.regressor(fused).squeeze(-1)
        return prediction, weights
