# src/models/gnn.py
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


# ------------------------------
# Baselines you already had
# ------------------------------
class GCNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, layers=3, dropout=0.2, num_classes=2):
        super().__init__()
        assert layers >= 2
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes))

    def forward(self, x, edge_index, t_idx: Optional[torch.Tensor] = None):
        h = x
        for conv in self.convs[:-1]:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.convs[-1](h, edge_index)
        return out


class SAGENet(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, layers=3, dropout=0.2, num_classes=2):
        super().__init__()
        assert layers >= 2
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, num_classes))

    def forward(self, x, edge_index, t_idx: Optional[torch.Tensor] = None):
        h = x
        for conv in self.convs[:-1]:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.convs[-1](h, edge_index)
        return out


class GATNet(nn.Module):
    def __init__(
        self, in_dim, hidden_dim=128, layers=3, dropout=0.2, num_classes=2, heads=4
    ):
        super().__init__()
        assert layers >= 2
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_dim, hidden_dim // heads, heads=heads))
        for _ in range(layers - 2):
            self.convs.append(GATConv(hidden_dim, hidden_dim // heads, heads=heads))
        self.convs.append(GATConv(hidden_dim, num_classes, heads=1, concat=False))

    def forward(self, x, edge_index, t_idx: Optional[torch.Tensor] = None):
        h = x
        for conv in self.convs[:-1]:
            h = conv(h, edge_index)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        out = self.convs[-1](h, edge_index)
        return out


# ------------------------------
# New: SAGE with Residual + BN + Time Embeds
# ------------------------------
class SAGEResBNNet(nn.Module):
    """
    SAGE with residual connections and BatchNorm. Optional timestep embedding concatenated to inputs:
      - time_embed_type='learned' -> nn.Embedding(max_timestep, time_embed_dim)
      - time_embed_type='sin'     -> fixed sinusoidal features (no params)
      - time_embed_type='none'    -> disabled (or time_embed_dim=0)
    """

    def __init__(
        self,
        in_dim,
        hidden_dim=128,
        layers=3,
        dropout=0.2,
        num_classes=2,
        use_bn=True,
        residual=True,
        time_embed_dim=0,
        time_embed_type="learned",
        max_timestep=50,
    ):
        super().__init__()
        assert layers >= 2
        self.dropout = float(dropout)
        self.use_bn = bool(use_bn)
        self.residual = bool(residual)
        self.time_embed_dim = int(time_embed_dim)
        self.time_embed_type = str(time_embed_type)
        self.max_timestep = int(max_timestep)

        # Time embedding (if any)
        self.time_emb = None
        if self.time_embed_dim > 0 and self.time_embed_type == "learned":
            self.time_emb = nn.Embedding(self.max_timestep, self.time_embed_dim)
            in_dim = in_dim + self.time_embed_dim
        elif self.time_embed_dim > 0 and self.time_embed_type == "sin":
            in_dim = in_dim + self.time_embed_dim
        else:
            self.time_embed_dim = 0
            self.time_embed_type = "none"

        # Convs
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, num_classes))

        # BNs for hidden layers only
        self.bns = nn.ModuleList()
        if self.use_bn:
            for _ in range(layers - 1):
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        # Residual projections where dims change
        self.res_projs = nn.ModuleList()
        in_dims = [in_dim] + [hidden_dim] * (layers - 2)
        out_dims = [hidden_dim] * (layers - 1)
        for d_in, d_out in zip(in_dims, out_dims):
            self.res_projs.append(
                nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out, bias=False)
            )

    # ----- helpers -----
    def _sinusoid(self, t_idx: torch.Tensor) -> torch.Tensor:
        # map t_idx in [1..max_timestep] to [0,1]
        t = torch.clamp(t_idx.long() - 1, 0, self.max_timestep - 1).to(
            dtype=torch.float32, device=t_idx.device
        )
        t = t / max(float(self.max_timestep - 1), 1.0)
        dim = self.time_embed_dim
        if dim <= 0:
            return None
        half = dim // 2
        freqs = torch.arange(1, half + 1, device=t.device, dtype=t.dtype) * (
            2.0 * math.pi
        )
        angles = t.unsqueeze(1) * freqs.unsqueeze(0)
        feat = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [N, 2*half]
        if feat.size(1) < dim:
            pad = torch.zeros(
                (feat.size(0), dim - feat.size(1)), device=feat.device, dtype=feat.dtype
            )
            feat = torch.cat([feat, pad], dim=1)
        return feat

    def _inject_time(self, x, t_idx):
        if self.time_embed_dim <= 0 or t_idx is None:
            return x
        if self.time_embed_type == "learned":
            tidx = torch.clamp(t_idx.long() - 1, 0, self.max_timestep - 1)
            te = self.time_emb(tidx)
            return torch.cat([x, te], dim=1)
        elif self.time_embed_type == "sin":
            te = self._sinusoid(t_idx)
            return torch.cat([x, te], dim=1)
        else:
            return x

    # ----- forward -----
    def forward(self, x, edge_index, t_idx: Optional[torch.Tensor] = None):
        x = self._inject_time(x, t_idx)
        h = x
        for li, conv in enumerate(self.convs[:-1]):
            h_in = h
            h = conv(h, edge_index)
            if self.use_bn:
                h = self.bns[li](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + self.res_projs[li](h_in)  # residual
        out = self.convs[-1](h, edge_index)
        return out
