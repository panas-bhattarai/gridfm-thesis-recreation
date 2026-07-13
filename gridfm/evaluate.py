"""Power-flow evaluation of a (pre-)trained GridFM model.
"""
import numpy as np
import pandas as pd
import pandapower as pp
import torch

from . import dataset as ds
from .datagen import build_net, apply_load_scale, run_opf, transfer_dispatch


# ----------------------------------------------------------------------------
# Model evaluation with the structured PF mask
# ----------------------------------------------------------------------------
def pf_evaluate(model, graphs, device="cpu"):
    """Returns (per_graph DataFrame, detail dict for scatter plots)."""
    model = model.to(device).eval()
    rows = []
    det = {k: [] for k in ["vm_true", "vm_pred", "va_true", "va_pred",
                           "qg_true", "qg_pred", "grid_base"]}
    per_bus_dp = []  # p.u. residual per bus of the first graph (for the map figure)
    for g in graphs:
        g = g.to(device)
        xm, m = ds.pf_mask(g.x)
        with torch.no_grad():
            pred = model(xm, g.edge_index, g.edge_attr)
        m6 = m[:, :ds.N_ELECTRICAL]
        x_mix = torch.cat([torch.where(m6, pred, g.x[:, :ds.N_ELECTRICAL]),
                           g.x[:, ds.N_ELECTRICAL:]], dim=1)
        dp, dq = ds.pbe_residual(x_mix, g.ybus_index, g.ybus_g, g.ybus_b)
        base = float(g.base_mva)
        rows.append({"scenario": int(g.scenario),
                     "p_res_mw": float(dp.abs().mean()) * base,
                     "q_res_mvar": float(dq.abs().mean()) * base,
                     "p_res_max_mw": float(dp.abs().max()) * base})
        if not per_bus_dp:
            per_bus_dp.append((dp.abs().cpu().numpy() * base, int(g.scenario)))
        # masked-feature details (Vm/Va at PQ buses, Qg at PV buses)
        pq = m[:, 4]
        det["vm_true"] += g.x[pq, 4].tolist(); det["vm_pred"] += pred[pq, 4].tolist()
        va = m[:, 5]
        det["va_true"] += g.x[va, 5].tolist(); det["va_pred"] += pred[va, 5].tolist()
        qg = m[:, 3]
        det["qg_true"] += (g.x[qg, 3] * base).tolist()
        det["qg_pred"] += (pred[qg, 3] * base).tolist()
        det["grid_base"].append(base)
    return pd.DataFrame(rows), det, per_bus_dp[0]


# ----------------------------------------------------------------------------
# DC power flow baseline (multiprocessing worker)
# ----------------------------------------------------------------------------
def dc_state_worker(task):
    """Rebuild one stored scenario, redo OPF dispatch, run DC PF.

    Returns the DC state (theta per bus) and the DC injections (MW per bus),
    or a failure marker. task = (grid_key, scenario_id, scale, removed_str).
    """
    grid, sid, scale, removed_str = task
    net = build_net(grid)
    apply_load_scale(net, scale)
    if isinstance(removed_str, str) and removed_str:
        for tok in removed_str.split(";"):
            et, idx = tok.split(":")
            net[et].at[int(idx), "in_service"] = False
    if not run_opf(net):
        return {"status": "opf_fail", "scenario": sid}
    transfer_dispatch(net)
    try:
        pp.rundcpp(net)
    except Exception:
        return {"status": "dc_fail", "scenario": sid}

    nb = len(net.bus)
    pos = {int(b): i for i, b in enumerate(net.bus.index)}
    theta = np.deg2rad(net.res_bus.va_degree.values)

    pd_mw = np.zeros(nb)
    lm = net.load.in_service.values
    for b, p in zip(net.load.loc[lm, "bus"], net.load.loc[lm, "p_mw"]):
        pd_mw[pos[int(b)]] += p
    pg_mw = np.zeros(nb)
    gm = net.gen.in_service.values
    for b, p in zip(net.gen.loc[gm, "bus"], net.res_gen.loc[gm, "p_mw"]):
        pg_mw[pos[int(b)]] += p
    for b, p in zip(net.ext_grid.bus, net.res_ext_grid.p_mw):
        pg_mw[pos[int(b)]] += p
    if len(net.sgen):
        sm = net.sgen.in_service.values
        for b, p in zip(net.sgen.loc[sm, "bus"], net.res_sgen.loc[sm, "p_mw"]):
            pg_mw[pos[int(b)]] += p
    return {"status": "ok", "scenario": sid, "theta": theta,
            "p_inj_mw": pg_mw - pd_mw}


def dc_residual_mw(theta, p_inj_mw, ybus_df, base_mva_raw=100.0):
    """Mean |ΔP| in MW of the DC state under the full AC power balance.

    V = 1.0 p.u. flat (the DC assumption), theta from the DC solution, exact
    AC Ybus from storage (raw 100 MVA base).
    """
    nb = len(theta)
    V = np.exp(1j * theta)
    Y = np.zeros((nb, nb), dtype=complex)
    for i, j, G, B in zip(ybus_df.i, ybus_df.j, ybus_df.G, ybus_df.B):
        Y[int(i), int(j)] = G + 1j * B
    p_calc = (V * np.conj(Y @ V)).real * base_mva_raw
    return float(np.abs(p_calc - p_inj_mw).mean())
