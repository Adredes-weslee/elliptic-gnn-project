import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


class GCNNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        for _ in range(layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, num_classes))

    def forward(self, x, edge_index, t_idx=None):  # keep signature compatible
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x  # logits


class SAGENet(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, layers=2, dropout=0.3, num_classes=2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, num_classes))

    def forward(self, x, edge_index, t_idx=None):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATNet(nn.Module):
    def __init__(
        self, in_dim, hidden_dim=64, layers=2, heads=4, dropout=0.3, num_classes=2
    ):
        super().__init__()
        self.dropout = dropout
        self.layers = layers
        self.heads = heads

        self.gats = nn.ModuleList()
        self.gats.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        for _ in range(layers - 2):
            self.gats.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )
        self.gats.append(
            GATConv(
                hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout
            )
        )

    def forward(self, x, edge_index, t_idx=None):
        for conv in self.gats[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gats[-1](x, edge_index)
        return x


# Upgraded: SAGE with BatchNorm + Residual + optional learned/sinusoidal time embedding
class SAGEResBNNet(nn.Module):
    """
    - Residual connections between hidden SAGE layers
    - BatchNorm after each hidden layer
    - Optional timestep embedding concatenated to input:
        * time_embed_type='learned' -> nn.Embedding
        * time_embed_type='sin'     -> fixed sinusoidal features
        * time_embed_type='none'    -> disabled (or time_embed_dim=0)
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
        max_timestep=49,
    ):
        super().__init__()
        assert layers >= 2, "Use >=2 layers for residuals to matter."

        self.dropout = dropout
        self.use_bn = use_bn
        self.residual = residual
        self.time_embed_dim = int(time_embed_dim)
        self.time_embed_type = str(time_embed_type)
        self.max_timestep = int(max_timestep)

        # time features
        self.time_emb = None
        if self.time_embed_dim > 0 and self.time_embed_type == "learned":
            self.time_emb = nn.Embedding(self.max_timestep, self.time_embed_dim)
            in_dim = in_dim + self.time_embed_dim
        elif self.time_embed_dim > 0 and self.time_embed_type == "sin":
            # no parameters; still expand input dims
            in_dim = in_dim + self.time_embed_dim
        else:
            self.time_embed_dim = 0
            self.time_embed_type = "none"

        # convs
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
        self.convs.append(SAGEConv(hidden_dim, num_classes))

        # batch norms for hidden layers only
        self.bns = nn.ModuleList()
        if self.use_bn:
            for _ in range(layers - 1):  # no BN on final logits
                self.bns.append(nn.BatchNorm1d(hidden_dim))

        # residual projections when dims mismatch (input->hidden only)
        self.res_projs = nn.ModuleList()
        in_dims = [in_dim] + [hidden_dim] * (layers - 2)
        out_dims = [hidden_dim] * (layers - 1)
        for d_in, d_out in zip(in_dims, out_dims):
            self.res_projs.append(
                nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out, bias=False)
            )

    def _sinusoid(self, t_idx):
        # t_idx in [1..max_timestep] -> normalize to [0,1]
        if self.time_embed_dim <= 0:
            return None
        t = torch.clamp(t_idx.long() - 1, 0, self.max_timestep - 1).to(
            dtype=torch.float32, device=t_idx.device
        )
        t = t / max(float(self.max_timestep - 1), 1.0)  # [N]
        dim = self.time_embed_dim
        half = dim // 2
        freqs = torch.arange(1, half + 1, device=t.device, dtype=t.dtype) * (
            2.0 * math.pi
        )
        angles = t.unsqueeze(1) * freqs.unsqueeze(0)  # [N, half]
        sin = torch.sin(angles)
        cos = torch.cos(angles)
        feat = torch.cat([sin, cos], dim=1)  # [N, 2*half]
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
            t = torch.clamp(t_idx.long() - 1, 0, self.max_timestep - 1)
            te = self.time_emb(t)  # [N, D]
            return torch.cat([x, te], dim=1)
        elif self.time_embed_type == "sin":
            te = self._sinusoid(t_idx)
            return torch.cat([x, te], dim=1)
        else:
            return x

    def forward(self, x, edge_index, t_idx=None):
        x = self._inject_time(x, t_idx)

        h = x
        # hidden blocks
        for li, conv in enumerate(self.convs[:-1]):
            h_in = h
            h = conv(h, edge_index)
            if self.use_bn:
                h = self.bns[li](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            if self.residual:
                h = h + self.res_projs[li](h_in)

        # logits
        out = self.convs[-1](h, edge_index)
        return out
