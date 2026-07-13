"""GridFM v0.1 model and physics-informed loss.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv, GPSConv, GINEConv

from .dataset import pbe_residual, N_ELECTRICAL


class GridFMv01(nn.Module):
    def __init__(self, in_dim=9, per_head=32, heads=4, num_layers=4,
                 edge_dim=2, out_dim=N_ELECTRICAL, dropout=0.1):
        super().__init__()
        hidden = per_head * heads
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([
            TransformerConv(hidden, per_head, heads=heads, edge_dim=edge_dim,
                            dropout=dropout, concat=True)
            for _ in range(num_layers)
        ])
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x, edge_index, edge_attr, batch=None):
        # batch is accepted (and ignored) so v0.1 and v0.2 share the training loop
        h = self.embed(x)
        for conv in self.convs:
            h = self.drop(self.act(conv(h, edge_index, edge_attr)))
        return self.decoder(h)          # (N, 6): Pd, Qd, Pg, Qg, Vm, Va


class GridFMv02(nn.Module):
    """GridFM v0.2: GPSConv stack
    """
    def __init__(self, in_dim=25, hidden=80, heads=4, num_layers=4,
                 edge_dim=2, out_dim=N_ELECTRICAL, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList([
            GPSConv(hidden,
                    GINEConv(nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                           nn.Linear(hidden, hidden)),
                             edge_dim=edge_dim),
                    heads=heads, dropout=dropout, attn_type="multihead")
            for _ in range(num_layers)
        ])
        self.decoder = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x, edge_index, edge_attr, batch=None):
        # batch tells the global-attention channel where each graph in a
        # mini-batch ends, so buses never attend across scenarios
        h = self.embed(x)
        for conv in self.convs:
            h = conv(h, edge_index, batch=batch, edge_attr=edge_attr)
        return self.decoder(h)          # (N, 6): Pd, Qd, Pg, Qg, Vm, Va


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def gridfm_loss(pred, x_true, mask, ybus_index, ybus_g, ybus_b,
                w1=0.01, w2=0.99):
    """L = w1 * masked MSE + w2 * PBE 
    """
    m6 = mask[:, :N_ELECTRICAL]
    mse = ((pred - x_true[:, :N_ELECTRICAL])[m6] ** 2).mean()

    x_mix = torch.cat([torch.where(m6, pred, x_true[:, :N_ELECTRICAL]),
                       x_true[:, N_ELECTRICAL:]], dim=1)
    dp, dq = pbe_residual(x_mix, ybus_index, ybus_g, ybus_b)
    pbe = (dp ** 2 + dq ** 2).mean()

    total = w1 * mse + w2 * pbe
    return total, {"mse": float(mse.detach()), "pbe": float(pbe.detach()),
                   "total": float(total.detach())}
