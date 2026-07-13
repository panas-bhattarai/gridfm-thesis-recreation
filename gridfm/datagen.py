"""Data generation pipeline
"""
import logging

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn
import pandapower.topology as top

logging.getLogger("pandapower").setLevel(logging.CRITICAL)

# grid key -> pandapower.networks constructor name
GRIDS = {
    "case24": "case24_ieee_rts",
    "case30": "case30",
    "case118": "case118",
    "case39": "case39",   # held out for fine-tuning / zero-shot (never pre-trained on)
}

BASE_MVA = 100.0  # pandapower/MATPOWER system base


# ----------------------------------------------------------------------------
# Load profile (ASSUMPTION: synthetic EIA-like profile; see notebook)
# ----------------------------------------------------------------------------
def make_load_profile(n_steps=8760, seed=42):
    """Synthetic aggregated hourly load profile for one year, max-normalized to 1.
    """
    rng = np.random.RandomState(seed)
    hours = np.arange(n_steps)
    hour_of_day = hours % 24
    day_of_year = (hours // 24) % 365
    day_of_week = (hours // 24) % 7

    seasonal = 0.85 + 0.15 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    daily = (0.72
             + 0.14 * np.exp(-((hour_of_day - 8.5) ** 2) / 7.0)
             + 0.22 * np.exp(-((hour_of_day - 18.5) ** 2) / 9.0))
    weekly = np.where(day_of_week >= 5, 0.92, 1.0)
    noise = 1.0 + 0.02 * rng.randn(n_steps)

    profile = seasonal * daily * weekly * noise
    return profile / profile.max()


def scenario_scale_factors(profile, n_scenarios):
    """Sample n_scenarios scaling factors evenly across the year (linear interp)."""
    t = np.linspace(0, len(profile) - 1, n_scenarios)
    return np.interp(t, np.arange(len(profile)), profile)


# ----------------------------------------------------------------------------
# Network manipulation
# ----------------------------------------------------------------------------
def build_net(grid_key):
    return getattr(pn, GRIDS[grid_key])()


def apply_load_scale(net, factor):
    """Scale all loads by a single global factor.
    """
    net.load["p_mw"] *= factor
    net.load["q_mvar"] *= factor


def perturb_topology(net, rng, max_outages=3):
    """Remove 1..max_outages random elements (lines, trafos, generators).

    Returns the list of removed (element_type, index), or None if the
    perturbation disconnects the grid (sample must be discarded).
    """
    candidates = ([("line", i) for i in net.line.index[net.line.in_service]]
                  + [("trafo", i) for i in net.trafo.index[net.trafo.in_service]]
                  + [("gen", i) for i in net.gen.index[net.gen.in_service]]
                  + [("sgen", i) for i in net.sgen.index[net.sgen.in_service]])
    k = rng.randint(1, max_outages + 1)
    k = min(k, len(candidates))
    chosen = [candidates[i] for i in rng.choice(len(candidates), size=k, replace=False)]
    for etype, idx in chosen:
        net[etype].at[idx, "in_service"] = False
    # connectivity check: every bus must still be supplied from the slack
    if len(top.unsupplied_buses(net)) > 0:
        return None
    return chosen


# ----------------------------------------------------------------------------
# Solving and feature extraction
# ----------------------------------------------------------------------------
def run_opf(net):
    """AC-OPF with a fallback initialization. Returns True if converged."""
    try:
        pp.runopp(net)
        return True
    except Exception:
        try:
            pp.runopp(net, init="pf")
            return True
        except Exception:
            return False


def transfer_dispatch(net):
    """Copy the OPF dispatch into the net as PF setpoints."""
    gmask = net.gen.in_service.values
    net.gen.loc[gmask, "p_mw"] = net.res_gen.loc[gmask, "p_mw"].values
    net.gen.loc[gmask, "vm_pu"] = net.res_bus.loc[net.gen.loc[gmask, "bus"], "vm_pu"].values
    net.ext_grid["vm_pu"] = net.res_bus.loc[net.ext_grid.bus, "vm_pu"].values
    # static generators (e.g., case24 models 22 of the RTS units as sgen):
    # OPF dispatches them too; they enter the PF as fixed PQ injections
    if len(net.sgen):
        smask = net.sgen.in_service.values
        net.sgen.loc[smask, "p_mw"] = net.res_sgen.loc[smask, "p_mw"].values
        net.sgen.loc[smask, "q_mvar"] = net.res_sgen.loc[smask, "q_mvar"].values


def solve_opf_pf(net):
    """AC-OPF for the dispatch, then Newton-Raphson AC-PF for the final state."""
    if not run_opf(net):
        return "opf_fail"
    transfer_dispatch(net)
    try:
        pp.runpp(net)
    except Exception:
        return "pf_fail"
    if not net.converged:
        return "pf_fail"
    return "ok"


def extract_features(net, scenario_id):
    """Build node, edge and Ybus tables for one solved scenario.
    """
    from pandapower.pypower.idx_brch import F_BUS, T_BUS, BR_R, BR_X, BR_STATUS

    buses = net.bus.index.values
    nb = len(buses)
    pos = {int(b): i for i, b in enumerate(buses)}

    pd_mw = np.zeros(nb); qd_mvar = np.zeros(nb)
    lmask = net.load.in_service.values
    for b, p, q in zip(net.load.loc[lmask, "bus"], net.load.loc[lmask, "p_mw"],
                       net.load.loc[lmask, "q_mvar"]):
        pd_mw[pos[int(b)]] += p; qd_mvar[pos[int(b)]] += q

    pg_mw = np.zeros(nb); qg_mvar = np.zeros(nb)
    gmask = net.gen.in_service.values
    for b, p, q in zip(net.gen.loc[gmask, "bus"], net.res_gen.loc[gmask, "p_mw"],
                       net.res_gen.loc[gmask, "q_mvar"]):
        pg_mw[pos[int(b)]] += p; qg_mvar[pos[int(b)]] += q
    for b, p, q in zip(net.ext_grid.bus, net.res_ext_grid.p_mw, net.res_ext_grid.q_mvar):
        pg_mw[pos[int(b)]] += p; qg_mvar[pos[int(b)]] += q
    if len(net.sgen):
        smask = net.sgen.in_service.values
        for b, p, q in zip(net.sgen.loc[smask, "bus"], net.res_sgen.loc[smask, "p_mw"],
                           net.res_sgen.loc[smask, "q_mvar"]):
            pg_mw[pos[int(b)]] += p; qg_mvar[pos[int(b)]] += q

    vm = net.res_bus.vm_pu.values.copy()
    va = np.deg2rad(net.res_bus.va_degree.values)

    is_ref = np.zeros(nb, dtype=int)
    for b in net.ext_grid.bus:
        is_ref[pos[int(b)]] = 1
    is_pv = np.zeros(nb, dtype=int)
    for b in net.gen.loc[gmask, "bus"]:
        if not is_ref[pos[int(b)]]:
            is_pv[pos[int(b)]] = 1
    is_pq = ((is_ref == 0) & (is_pv == 0)).astype(int)

    node_df = pd.DataFrame({
        "scenario": scenario_id, "bus": buses,
        "Pd": pd_mw, "Qd": qd_mvar, "Pg": pg_mw, "Qg": qg_mvar,
        "Vm": vm, "Va": va, "PQ": is_pq, "PV": is_pv, "REF": is_ref,
    })

    # edges from the ppc branch table (in-service branches only)
    br = net._ppc["branch"]
    lookup = net._pd2ppc_lookups["bus"]
    inv = {int(lookup[int(b)]): int(b) for b in buses}
    on = br[:, BR_STATUS].real > 0
    f = [inv[int(i.real)] for i in br[on, F_BUS]]
    t = [inv[int(i.real)] for i in br[on, T_BUS]]
    ys = 1.0 / (br[on, BR_R].real + 1j * br[on, BR_X].real)
    edge_df = pd.DataFrame({
        "scenario": scenario_id, "from_bus": f, "to_bus": t,
        "g": ys.real, "b": ys.imag,
    })

    # sparse Ybus triplets in pandapower bus numbering
    Y = net._ppc["internal"]["Ybus"].tocoo()
    ybus_df = pd.DataFrame({
        "scenario": scenario_id,
        "i": [inv[int(r)] for r in Y.row], "j": [inv[int(c)] for c in Y.col],
        "G": Y.data.real, "B": Y.data.imag,
    })
    return node_df, edge_df, ybus_df


def physics_residual_mw(node_df, ybus_df):
    """Max nodal apparent-power mismatch |S_bus(V) + S_load - S_gen| in MVA.
    """
    nb = len(node_df)
    pos = {int(b): i for i, b in enumerate(node_df.bus.values)}
    V = node_df.Vm.values * np.exp(1j * node_df.Va.values)
    Y = np.zeros((nb, nb), dtype=complex)
    for i, j, G, B in zip(ybus_df.i, ybus_df.j, ybus_df.G, ybus_df.B):
        Y[pos[int(i)], pos[int(j)]] = G + 1j * B
    s_calc = V * np.conj(Y @ V) * BASE_MVA
    s_inj = (node_df.Pg.values - node_df.Pd.values) + 1j * (node_df.Qg.values - node_df.Qd.values)
    return float(np.abs(s_calc - s_inj).max())


# ----------------------------------------------------------------------------
# Worker for multiprocessing
# ----------------------------------------------------------------------------
def generate_one(task):
    """One scenario: (grid_key, scenario_id, perturb_id, scale_factor, seed).

    perturb_id == 0 keeps the intact topology (load-only scenario);
    perturb_id >= 1 applies a random topology perturbation.
    """
    grid_key, scenario_id, perturb_id, scale, seed = task
    rng = np.random.RandomState(seed)
    net = build_net(grid_key)
    apply_load_scale(net, scale)

    removed = []
    if perturb_id > 0:
        removed = perturb_topology(net, rng)
        if removed is None:
            return {"status": "island", "grid": grid_key, "scenario": scenario_id}

    status = solve_opf_pf(net)
    if status != "ok":
        return {"status": status, "grid": grid_key, "scenario": scenario_id}

    node_df, edge_df, ybus_df = extract_features(net, scenario_id)
    resid = physics_residual_mw(node_df, ybus_df)
    if resid > 0.5:  # MW-scale tolerance; NR solutions sit at ~1e-6
        return {"status": "physics_fail", "grid": grid_key, "scenario": scenario_id}

    meta = pd.DataFrame([{
        "scenario": scenario_id, "grid": grid_key, "scale": scale,
        "n_removed": len(removed),
        "removed": ";".join("%s:%d" % (e, i) for e, i in removed),
        "residual_mva": resid,
    }])
    return {"status": "ok", "grid": grid_key, "scenario": scenario_id,
            "node": node_df, "edge": edge_df, "ybus": ybus_df, "meta": meta}
