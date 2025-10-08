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

    def forward(self, x, edge_index):
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

    def forward(self, x, edge_index):
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
        # first
        self.gats.append(GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout))
        # middle
        for _ in range(layers - 2):
            self.gats.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout)
            )
        # final
        self.gats.append(
            GATConv(
                hidden_dim * heads, num_classes, heads=1, concat=False, dropout=dropout
            )
        )

    def forward(self, x, edge_index):
        for conv in self.gats[:-1]:
            x = conv(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gats[-1](x, edge_index)
        return x
