"""Graph dataset construction for the GridFM recreation.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

NODE_FEATURES = ["Pd", "Qd", "Pg", "Qg", "Vm", "Va", "PQ", "PV", "REF"]
N_ELECTRICAL = 6           # first 6 features are maskable
PP_BASE_MVA = 100.0        # pandapower/MATPOWER system base the raw files use


# ----------------------------------------------------------------------------
# BaseMVA normalization
# ----------------------------------------------------------------------------
def compute_base_mva(node_df):
    """Grid-specific base: the maximum power-related magnitude in the dataset.
    """
    return float(node_df[["Pd", "Qd", "Pg", "Qg"]].abs().values.max())


def build_graphs(raw_dir, grid, base_mva=None):
    """Load the Milestone-1 CSVs of one grid and build a list of PyG Data objects.
    """
    raw_dir = Path(raw_dir)
    node = pd.read_csv(raw_dir / grid / "node_features.csv")
    edge = pd.read_csv(raw_dir / grid / "edge_features.csv")
    ybus = pd.read_csv(raw_dir / grid / "ybus.csv")
    meta = pd.read_csv(raw_dir / grid / "meta.csv").set_index("scenario")

    if base_mva is None:
        base_mva = compute_base_mva(node)
    y_scale = PP_BASE_MVA / base_mva

    edge_g = {s: d for s, d in edge.groupby("scenario")}
    ybus_g = {s: d for s, d in ybus.groupby("scenario")}

    graphs = []
    for sid, nd in node.groupby("scenario"):
        nd = nd.sort_values("bus")
        assert (nd.bus.values == np.arange(len(nd))).all(), "non-contiguous bus index"

        x = np.stack([
            nd.Pd.values / base_mva, nd.Qd.values / base_mva,
            nd.Pg.values / base_mva, nd.Qg.values / base_mva,
            nd.Vm.values, nd.Va.values,
            nd.PQ.values.astype(float), nd.PV.values.astype(float),
            nd.REF.values.astype(float),
        ], axis=1)

        ed = edge_g[sid]
        # undirected grid -> store both orientations (thesis: E and E_R)
        src = np.concatenate([ed.from_bus.values, ed.to_bus.values])
        dst = np.concatenate([ed.to_bus.values, ed.from_bus.values])
        ea = np.concatenate([ed[["g", "b"]].values, ed[["g", "b"]].values]) * y_scale

        yb = ybus_g[sid]
        data = Data(
            x=torch.tensor(x, dtype=torch.float32),
            edge_index=torch.tensor(np.stack([src, dst]), dtype=torch.long),
            edge_attr=torch.tensor(ea, dtype=torch.float32),
            ybus_index=torch.tensor(yb[["i", "j"]].values.T, dtype=torch.long),
            ybus_g=torch.tensor(yb.G.values * y_scale, dtype=torch.float32),
            ybus_b=torch.tensor(yb.B.values * y_scale, dtype=torch.float32),
            scenario=int(sid),
            group=int(sid) // 4,          # 4 variants share one load scenario
            scale=float(meta.loc[sid, "scale"]),
            n_removed=int(meta.loc[sid, "n_removed"]),
            base_mva=float(base_mva),
        )
        graphs.append(data)
    return graphs, base_mva


# ----------------------------------------------------------------------------
# Masking
# ----------------------------------------------------------------------------
def random_mask(x, p=0.5, generator=None):
    """Pre-training mask: each electrical feature masked by an independent coin flip.
    """
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[:, :N_ELECTRICAL] = torch.rand(
        (x.shape[0], N_ELECTRICAL), generator=generator, device=x.device) < p
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


def pf_mask(x):
    """Structured power-flow mask

    PQ bus:  hide |V|, Va       PV bus:  hide Qg, Va       REF bus: hide Pg, Qg
    """
    pq, pv, ref = x[:, 6] == 1, x[:, 7] == 1, x[:, 8] == 1
    mask = torch.zeros_like(x, dtype=torch.bool)
    mask[pq, 4] = True; mask[pq, 5] = True
    mask[pv, 3] = True; mask[pv, 5] = True
    mask[ref, 2] = True; mask[ref, 3] = True
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask


# ----------------------------------------------------------------------------
# Power balance residual, works on batched graphs
# ----------------------------------------------------------------------------
def pbe_residual(x, ybus_index, ybus_g, ybus_b):
    """Per-node active/reactive power balance residual in p.u.

    Computes S = diag(V) conj(Ybus V) via sparse scatter and subtracts the
    net injection (Pg - Pd, Qg - Qd). Differentiable -> reused as the
    physics-informed loss term in Milestone 3.
    """
    vm, va = x[:, 4], x[:, 5]
    vr, vi = vm * torch.cos(va), vm * torch.sin(va)
    i, j = ybus_index[0], ybus_index[1]
    ir = torch.zeros_like(vr).index_add_(0, i, ybus_g * vr[j] - ybus_b * vi[j])
    ii = torch.zeros_like(vr).index_add_(0, i, ybus_g * vi[j] + ybus_b * vr[j])
    p_calc = vr * ir + vi * ii
    q_calc = vi * ir - vr * ii
    dp = p_calc - (x[:, 2] - x[:, 0])
    dq = q_calc - (x[:, 3] - x[:, 1])
    return dp, dq


# ----------------------------------------------------------------------------
# Splits and caching
# ----------------------------------------------------------------------------
def split_graphs(graphs, frac=(0.8, 0.1, 0.1), seed=42):
    """Group-aware train/val/test split.
    """
    rng = np.random.RandomState(seed)
    groups = sorted({g.group for g in graphs})
    rng.shuffle(groups)
    n = len(groups)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)
    lab = {}
    for k, grp in enumerate(groups):
        lab[grp] = "train" if k < n_tr else ("val" if k < n_tr + n_va else "test")
    out = {"train": [], "val": [], "test": []}
    for g in graphs:
        out[lab[g.group]].append(g)
    return out


def save_processed(splits, out_dir, grid):
    out = Path(out_dir) / grid
    out.mkdir(parents=True, exist_ok=True)
    for name, gl in splits.items():
        torch.save(gl, out / ("%s.pt" % name))


def load_processed(out_dir, grid, split):
    return torch.load(Path(out_dir) / grid / ("%s.pt" % split), weights_only=False)
