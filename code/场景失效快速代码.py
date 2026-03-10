# -*- coding: utf-8 -*-
"""
mcm2026B_competition_solver_final.py
------------------------------------
Competition-ready final solver for MCM 2026 Problem B.

Core fixes vs your previous "standard" version:
A) Replace fragile Stage-1 "min tau by IPOPT" with a robust feasibility bisection:
   - tau_star_bisect = minimal feasible build time (build + ops-year supply).
   - This is the correct, defensible "shortest time-to-build" metric.

B) Epsilon-constraint cost optimization (time-focused policy):
   - Solve: minimize cost+env+smooth subject to tau <= (1+eps_tau)*tau_star_bisect.

C) Add a cost-opt policy (no tau upper bound) to show long-horizon advantages:
   - Solve: minimize cost+env+smooth + tiny_tau_weight*tau.
   - This makes SE disruptions at t=20 actually matter (if tau gets long).

D) Report rigor upgrades:
   - Split costs: Cost_build, Cost_ops1yr, Cost_total
   - Total time: Time_total = tau + 1
   - Trigger flags: SE_Triggered, Rocket_Triggered (whether disruption window overlaps [0, tau+1])
   - NumPy trapz compatibility.

Outputs:
  mcm_final_outputs/final_summary.csv
  mcm_final_outputs/timeseries_<label>__<policy>.csv

Dependencies: numpy, pandas, casadi (ipopt)
No solver_core required.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import casadi as ca

warnings.filterwarnings("ignore")
OUTDIR = "mcm_final_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# -------------------------
# NumPy trapezoid compat
# -------------------------
def trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    if hasattr(np, "trapz"):
        return float(np.trapz(y, x))
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    if y.size < 2:
        return 0.0
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])))

# -------------------------
# Scaling
# -------------------------
@dataclass(frozen=True)
class Scales:
    mass: float = 1e6     # tons -> Mt
    launch: float = 1e3   # launches -> k-launch
    money: float = 1e9    # dollars -> $B

# -------------------------
# CasADi-safe sigmoid / windows
# -------------------------
def sigmoid_k(x, k=10.0):
    z = ca.fmin(ca.fmax(k * x, -50), 50)
    return 1.0 / (1.0 + ca.exp(-z))

def window_health(t, t0, dur, severity, k_step=10.0):
    w = sigmoid_k(t - t0, k_step) - sigmoid_k(t - (t0 + dur), k_step)
    return 1.0 - severity * w

def capacity_logistic(t, K_max, alpha, r, delta_eff,
                      failure_active=False, fail_t_start=0.0, fail_duration=0.0, fail_severity=0.0):
    a = float(max(min(alpha, 0.999999), 1e-6))
    base = K_max / (1.0 + ((1.0 - a) / a) * ca.exp(-float(r) * t))
    base = base * float(delta_eff)
    if failure_active:
        base = base * window_health(t, float(fail_t_start), float(fail_duration), float(fail_severity), k_step=10.0)
    return ca.fmax(base, 0.0)

def rocket_health(t, active=False, fail_t_start=0.0, fail_duration=0.0, fail_severity=0.0):
    if not active:
        return ca.SX(1.0)
    h = window_health(t, float(fail_t_start), float(fail_duration), float(fail_severity), k_step=10.0)
    return ca.fmin(ca.fmax(h, 0.0), 1.0)

# -------------------------
# Params
# -------------------------
@dataclass
class Params:
    # Construction goal
    M_goal_tons: float = 1e8  # 100 million tons

    # Ops year water demand
    P_pop: float = 1e5
    w_ls_kg_per_person_day: float = 1.5
    ops_supply_factor: float = 1.0  # water-only = 1.0 ; water+other supplies -> >1

    # Scenario: mixed | se_only | rocket_only
    scenario: str = "mixed"

    # Discretization
    N_build: int = 250
    N_ops: int = 60

    # Build time bounds (years)
    tau_min: float = 1.0
    tau_max: float = 400.0

    # Space elevator capacity
    K_max_total_tpy: float = 3.0 * 179000.0
    alpha: float = 0.02
    r: float = 0.18
    delta_eff: float = 0.974

    # Vehicles
    L_R_ton_per_launch: float = 125.0
    FE_max_launch_per_yr: float = 100000.0
    FA_max_launch_per_yr: float = 200000.0

    # Apex
    A_max_tons: float = 5e5
    A_gate_tons: float = 1e3

    # Costs
    C_E_dollars_per_launch: float = 1.0e8
    C_A_dollars_per_launch: float = 2.0e7
    C_SE_dollars_per_ton: float = 250.0

    # Environment index
    e_E_per_launch: float = 1275.0
    e_A_per_launch: float = 450.0
    e_SE_per_ton: float = 2.0
    gamma_env: float = 1e-4  # quadratic penalty coefficient

    # Disruptions
    se_failure_active: bool = False
    fail_t_start: float = 20.0
    fail_duration: float = 3.0
    fail_severity: float = 0.90

    rocket_failure_active: bool = False
    rocket_fail_t_start: float = 22.0
    rocket_fail_duration: float = 2.0
    rocket_fail_severity: float = 0.50

    # Smoothness & terminal regularization
    w_smooth: float = 0.5
    w_end: float = 1e2

    # Ops requirement mode
    ops_require_rate_each_step: bool = False

    # IPOPT
    ipopt_max_iter: int = 50000
    ipopt_tol: float = 1e-8
    ipopt_acceptable_tol: float = 1e-4
    ipopt_acceptable_constr_viol_tol: float = 1e-6
    ipopt_acceptable_iter: int = 1200
    ipopt_bound_relax_factor: float = 1e-10

    # Feasibility reporting
    tol_viol_scaled: float = 1e-6

    # epsilon constraint for time-focused policy
    eps_tau: float = 0.05

    # tiny tau weight for cost-opt policy (prevents drifting to tau_max)
    tiny_tau_weight: float = 1e-6

# -------------------------
# Scaling
# -------------------------
def scaled(p: Params) -> Dict[str, float]:
    sc = Scales()
    scenario = str(p.scenario).lower().strip()
    if scenario not in ("mixed", "se_only", "rocket_only"):
        raise ValueError("scenario must be mixed|se_only|rocket_only")

    # ops-year water demand (tons/year)
    W_year_tons = p.P_pop * p.w_ls_kg_per_person_day * 365.0 / 1000.0
    W_year_tons *= float(max(p.ops_supply_factor, 0.0))

    sp = {
        "mass": sc.mass,
        "launch": sc.launch,
        "money": sc.money,

        "M_goal": p.M_goal_tons / sc.mass,            # Mt
        "W_year": W_year_tons / sc.mass,             # Mt
        "W_req": (W_year_tons / 1.0) / sc.mass,      # Mt/yr average demand

        "A_max": p.A_max_tons / sc.mass,             # Mt
        "A_gate": p.A_gate_tons / sc.mass,           # Mt
        "L_R": p.L_R_ton_per_launch / sc.mass,       # Mt/launch

        "FE_cap": p.FE_max_launch_per_yr / sc.launch,  # k-launch/yr
        "FA_cap": p.FA_max_launch_per_yr / sc.launch,  # k-launch/yr
        "K_max": p.K_max_total_tpy / sc.mass,          # Mt/yr

        "C_E": (p.C_E_dollars_per_launch * sc.launch) / sc.money,  # $B/(k-launch)
        "C_A": (p.C_A_dollars_per_launch * sc.launch) / sc.money,  # $B/(k-launch)
        "C_SE": (p.C_SE_dollars_per_ton * sc.mass) / sc.money,     # $B/Mt

        # emissions index per rate unit
        "e_E": p.e_E_per_launch * sc.launch,
        "e_A": p.e_A_per_launch * sc.launch,
        "e_SE": p.e_SE_per_ton * sc.mass,
    }

    if scenario == "se_only":
        sp["FE_cap"] = 0.0
    if scenario == "rocket_only":
        sp["FA_cap"] = 0.0
        sp["K_max"] = 0.0

    return sp

# -------------------------
# Scaled constraint violation metric
# -------------------------
def viol_scaled(gval, lbg, ubg, g_scale) -> float:
    gval = np.asarray(gval, float).flatten()
    lbg = np.asarray(lbg, float).flatten()
    ubg = np.asarray(ubg, float).flatten()
    gs = np.asarray(g_scale, float).flatten()

    low = np.maximum(lbg - gval, 0.0)
    high = np.maximum(gval - ubg, 0.0)
    high[np.isinf(ubg)] = 0.0
    low[np.isinf(lbg)] = 0.0
    v = low + high

    if v.size == 0:
        return 0.0
    if gs.size != gval.size:
        return float(np.max(v))
    denom = np.maximum(gs, 1e-12)
    return float(np.max(v / denom))

# -------------------------
# Trigger flags (for reporting credibility)
# -------------------------
def _window_overlaps(h0: float, h1: float, w0: float, w1: float) -> bool:
    return (max(h0, w0) <= min(h1, w1))

def triggered_flags(p: Params, tau: float) -> Dict[str, str]:
    horizon0 = 0.0
    horizon1 = float(tau) + 1.0  # build + 1 year ops
    se_trig = False
    rk_trig = False
    if p.se_failure_active:
        se_trig = _window_overlaps(horizon0, horizon1, float(p.fail_t_start), float(p.fail_t_start + p.fail_duration))
    if p.rocket_failure_active:
        rk_trig = _window_overlaps(horizon0, horizon1, float(p.rocket_fail_t_start), float(p.rocket_fail_t_start + p.rocket_fail_duration))
    return {"SE_Triggered": "YES" if se_trig else "NO", "Rocket_Triggered": "YES" if rk_trig else "NO"}

# -------------------------
# NLP builder
# policy:
#   - "time_opt": tau has upper bound tau_ub (epsilon-constrained)
#   - "cost_opt": tau free in [tau_min, tau_max] (tiny penalty added)
# -------------------------
def build_nlp_variable_tau(
    p: Params,
    sp: Dict[str, float],
    policy: str,
    tau_ub: Optional[float] = None
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    Nb = int(p.N_build)
    No = int(p.N_ops)

    tau = ca.SX.sym("tau", 1)
    hb = tau[0] / Nb
    ho = 1.0 / No

    # Build nodes
    Ab = ca.SX.sym("Ab", Nb + 1)
    Mb = ca.SX.sym("Mb", Nb + 1)

    # Ops nodes
    Ao = ca.SX.sym("Ao", No + 1)
    So = ca.SX.sym("So", No + 1)

    # Controls
    FEb = ca.SX.sym("FEb", Nb + 1)
    FAb = ca.SX.sym("FAb", Nb + 1)
    Ub  = ca.SX.sym("Ub",  Nb + 1)

    FEo = ca.SX.sym("FEo", No + 1)
    FAo = ca.SX.sym("FAo", No + 1)
    Uo  = ca.SX.sym("Uo",  No + 1)

    x = ca.vertcat(tau, Ab, Mb, Ao, So, FEb, FAb, Ub, FEo, FAo, Uo)

    # Index
    offset = 0
    idx = {}
    idx["tau"] = slice(offset, offset+1); offset += 1
    idx["Ab"]  = slice(offset, offset+Nb+1); offset += Nb+1
    idx["Mb"]  = slice(offset, offset+Nb+1); offset += Nb+1
    idx["Ao"]  = slice(offset, offset+No+1); offset += No+1
    idx["So"]  = slice(offset, offset+No+1); offset += No+1
    idx["FEb"] = slice(offset, offset+Nb+1); offset += Nb+1
    idx["FAb"] = slice(offset, offset+Nb+1); offset += Nb+1
    idx["Ub"]  = slice(offset, offset+Nb+1); offset += Nb+1
    idx["FEo"] = slice(offset, offset+No+1); offset += No+1
    idx["FAo"] = slice(offset, offset+No+1); offset += No+1
    idx["Uo"]  = slice(offset, offset+No+1); offset += No+1

    # Bounds
    lbx, ubx = [], []
    # tau bounds (possibly tightened)
    tau_lb = float(p.tau_min)
    tau_ub_final = float(p.tau_max) if tau_ub is None else float(min(p.tau_max, tau_ub))
    lbx += [tau_lb]; ubx += [tau_ub_final]

    # States bounds
    lbx += [0.0]*(Nb+1); ubx += [float(sp["A_max"])]*(Nb+1)
    lbx += [0.0]*(Nb+1); ubx += [10.0*float(sp["M_goal"])]*(Nb+1)

    lbx += [0.0]*(No+1); ubx += [float(sp["A_max"])]*(No+1)
    lbx += [0.0]*(No+1); ubx += [10.0*float(sp["W_year"])]*(No+1)

    # Controls bounds
    lbx += [0.0]*(Nb+1); ubx += [float(sp["FE_cap"])]*(Nb+1)
    lbx += [0.0]*(Nb+1); ubx += [float(sp["FA_cap"])]*(Nb+1)
    lbx += [0.0]*(Nb+1); ubx += [float(sp["K_max"])]*(Nb+1)

    lbx += [0.0]*(No+1); ubx += [float(sp["FE_cap"])]*(No+1)
    lbx += [0.0]*(No+1); ubx += [float(sp["FA_cap"])]*(No+1)
    lbx += [0.0]*(No+1); ubx += [float(sp["K_max"])]*(No+1)

    lbx = np.array(lbx, float)
    ubx = np.array(ubx, float)

    # Gate slope
    gate_slope = float(sp["FA_cap"]/sp["A_gate"]) if (sp["A_gate"] > 0 and sp["FA_cap"] > 0) else 0.0

    # Constraints
    g, lbg, ubg, g_scale = [], [], [], []

    # Initial conditions build
    g += [Ab[0], Mb[0]]
    lbg += [0.0, 0.0]; ubg += [0.0, 0.0]
    g_scale += [max(float(sp["A_max"]), 1.0), max(float(sp["M_goal"]), 1.0)]

    def time_build(i):
        return tau[0] * (float(i)/float(Nb))

    def time_ops(j):
        return tau[0] + float(j)/float(No)

    def delivered_rate(Ai, FEi, FAi, t_abs):
        outflow = sp["L_R"] * (FAi * sp["launch"])  # Mt/yr
        rh = rocket_health(
            t_abs,
            active=p.rocket_failure_active,
            fail_t_start=p.rocket_fail_t_start,
            fail_duration=p.rocket_fail_duration,
            fail_severity=p.rocket_fail_severity
        )
        arrive_E = sp["L_R"] * rh * (FEi * sp["launch"])
        arrive_A = rh * outflow
        return outflow, arrive_E + arrive_A, rh

    # Build segment Euler
    for i in range(Nb):
        t_i = time_build(i)
        Phi_i = capacity_logistic(
            t_i, sp["K_max"], p.alpha, p.r, p.delta_eff,
            failure_active=p.se_failure_active,
            fail_t_start=p.fail_t_start,
            fail_duration=p.fail_duration,
            fail_severity=p.fail_severity
        )
        # U <= Phi
        g += [Ub[i] - Phi_i]; lbg += [-np.inf]; ubg += [0.0]
        g_scale += [max(float(sp["K_max"]), 1.0)]

        # gate
        if gate_slope > 0:
            g += [FAb[i] - gate_slope*Ab[i]]; lbg += [-np.inf]; ubg += [0.0]
            g_scale += [max(float(sp["FA_cap"]), 1.0)]

        outflow, dM, _ = delivered_rate(Ab[i], FEb[i], FAb[i], t_i)

        gA = Ab[i+1] - (Ab[i] + hb*(Ub[i] - outflow))
        gM = Mb[i+1] - (Mb[i] + hb*dM)
        g += [gA, gM]; lbg += [0.0, 0.0]; ubg += [0.0, 0.0]
        g_scale += [max(float(sp["A_max"]), 1.0), max(float(sp["M_goal"]), 1.0)]

    # Build completion
    g += [Mb[-1] - sp["M_goal"]]; lbg += [0.0]; ubg += [np.inf]
    g_scale += [max(float(sp["M_goal"]), 1.0)]

    # Connect to ops
    g += [Ao[0] - Ab[-1], So[0]]
    lbg += [0.0, 0.0]; ubg += [0.0, 0.0]
    g_scale += [max(float(sp["A_max"]), 1.0), max(float(sp["W_year"]), 1.0)]

    # Ops segment Euler
    for j in range(No):
        t_j = time_ops(j)
        Phi_j = capacity_logistic(
            t_j, sp["K_max"], p.alpha, p.r, p.delta_eff,
            failure_active=p.se_failure_active,
            fail_t_start=p.fail_t_start,
            fail_duration=p.fail_duration,
            fail_severity=p.fail_severity
        )
        g += [Uo[j] - Phi_j]; lbg += [-np.inf]; ubg += [0.0]
        g_scale += [max(float(sp["K_max"]), 1.0)]

        if gate_slope > 0:
            g += [FAo[j] - gate_slope*Ao[j]]; lbg += [-np.inf]; ubg += [0.0]
            g_scale += [max(float(sp["FA_cap"]), 1.0)]

        outflow_o, dM_o, _ = delivered_rate(Ao[j], FEo[j], FAo[j], t_j)

        gA = Ao[j+1] - (Ao[j] + ho*(Uo[j] - outflow_o))
        gS = So[j+1] - (So[j] + ho*dM_o)
        g += [gA, gS]; lbg += [0.0, 0.0]; ubg += [0.0, 0.0]
        g_scale += [max(float(sp["A_max"]), 1.0), max(float(sp["W_year"]), 1.0)]

        if p.ops_require_rate_each_step:
            g += [dM_o - sp["W_req"]]; lbg += [0.0]; ubg += [np.inf]
            g_scale += [max(float(sp["W_req"]), 1.0)]

    # Ops-year cumulative
    g += [So[-1] - sp["W_year"]]; lbg += [0.0]; ubg += [np.inf]
    g_scale += [max(float(sp["W_year"]), 1.0)]

    # Objective scaling
    tau_nom = 50.0
    FE_ref = min(float(sp["FE_cap"]), float(sp["M_goal"]) / max(tau_nom * float(sp["L_R"]) * float(sp["launch"]), 1e-12)) if sp["FE_cap"] > 0 else 0.0
    FA_ref = min(float(sp["FA_cap"]), float(sp["M_goal"]) / max(tau_nom * float(sp["L_R"]) * float(sp["launch"]), 1e-12)) if sp["FA_cap"] > 0 else 0.0
    U_ref  = min(float(sp["K_max"]), float(sp["M_goal"]) / max(tau_nom, 1e-12)) if sp["K_max"] > 0 else 0.0
    cost_scale = max(1e-3, tau_nom * (sp["C_E"]*FE_ref + sp["C_A"]*FA_ref + sp["C_SE"]*U_ref))

    # Integrated cost+env+smooth across both segments
    J = 0.0
    for i in range(Nb):
        cost_rate = sp["C_E"]*FEb[i] + sp["C_A"]*FAb[i] + sp["C_SE"]*Ub[i]
        env_rate  = sp["e_E"]*FEb[i] + sp["e_A"]*FAb[i] + sp["e_SE"]*Ub[i]
        env_pen   = p.gamma_env * (env_rate**2)

        dFE = (FEb[i+1] - FEb[i]) / (hb + 1e-12)
        dFA = (FAb[i+1] - FAb[i]) / (hb + 1e-12)
        dU  = (Ub[i+1]  - Ub[i])  / (hb + 1e-12)
        smooth = p.w_smooth*(dFE**2 + dFA**2 + dU**2)

        J += hb * (cost_rate + env_pen + smooth) / cost_scale

    for j in range(No):
        cost_rate = sp["C_E"]*FEo[j] + sp["C_A"]*FAo[j] + sp["C_SE"]*Uo[j]
        env_rate  = sp["e_E"]*FEo[j] + sp["e_A"]*FAo[j] + sp["e_SE"]*Uo[j]
        env_pen   = p.gamma_env * (env_rate**2)

        dFE = (FEo[j+1] - FEo[j]) / ho
        dFA = (FAo[j+1] - FAo[j]) / ho
        dU  = (Uo[j+1]  - Uo[j])  / ho
        smooth = p.w_smooth*(dFE**2 + dFA**2 + dU**2)

        J += ho * (cost_rate + env_pen + smooth) / cost_scale

    J += p.w_end * (FEb[-1]**2 + FAb[-1]**2 + Ub[-1]**2 + FEo[-1]**2 + FAo[-1]**2 + Uo[-1]**2) / cost_scale

    # policy-specific objective tweak
    if policy == "cost_opt":
        J = J + float(p.tiny_tau_weight) * tau[0]  # tiny tie-breaker
    elif policy == "time_opt":
        # tau upper bound already in lbx/ubx; no extra needed
        pass
    else:
        raise ValueError("policy must be 'time_opt' or 'cost_opt'")

    nlp = {"x": x, "f": J, "g": ca.vertcat(*g)}
    meta = {
        "idx": idx,
        "Nb": Nb, "No": No,
        "lbx": lbx, "ubx": ubx,
        "lbg": np.array(lbg, float), "ubg": np.array(ubg, float),
        "g_scale": np.array(g_scale, float),
        "cost_scale": float(cost_scale),
        "gate_slope": gate_slope
    }
    return nlp, meta

# -------------------------
# Initial guess (consistent Euler with simple clipping)
# -------------------------
def initial_guess(p: Params, sp: Dict[str, float], meta: Dict[str, Any], tau0: float) -> np.ndarray:
    Nb = meta["Nb"]; No = meta["No"]
    ii = meta["idx"]

    x0 = np.zeros(meta["ubx"].shape[0], float)
    tau0 = float(np.clip(tau0, p.tau_min, p.tau_max))
    x0[ii["tau"]] = tau0

    # nominal avg launch needed
    rh_nom = 0.95
    avg_klaunch = float(sp["M_goal"] / max(tau0 * sp["L_R"] * sp["launch"] * rh_nom, 1e-12))

    scenario = str(p.scenario).lower().strip()

    FEb = np.zeros(Nb+1); FAb = np.zeros(Nb+1); Ub = np.zeros(Nb+1)
    FEo = np.zeros(No+1); FAo = np.zeros(No+1); Uo = np.zeros(No+1)

    if scenario == "rocket_only":
        FEb[:] = min(sp["FE_cap"], avg_klaunch)
        FAb[:] = 0.0
        Ub[:]  = 0.0
    elif scenario == "se_only":
        FEb[:] = 0.0
        FAb[:] = min(sp["FA_cap"], avg_klaunch)
        Ub[:]  = float(sp["K_max"])
    else:
        FEb[:] = min(sp["FE_cap"], 0.35*avg_klaunch)
        FAb[:] = min(sp["FA_cap"], 0.65*avg_klaunch)
        Ub[:]  = float(sp["K_max"])

    # ops: modest (water is small relative to construction)
    FEo[:] = min(sp["FE_cap"], FEb[-1])
    FAo[:] = min(sp["FA_cap"], FAb[-1])
    Uo[:]  = min(sp["K_max"], Ub[-1])

    gate_slope = meta["gate_slope"]
    hb = tau0 / Nb
    ho = 1.0 / No

    Ab = np.zeros(Nb+1); Mb = np.zeros(Nb+1)
    Ao = np.zeros(No+1); So = np.zeros(No+1)

    # build simulate (assume rh≈1, Phi not enforced here; IPOPT will fix)
    for i in range(Nb):
        if gate_slope > 0:
            FAb[i] = min(FAb[i], gate_slope * Ab[i])
        outflow = sp["L_R"] * (FAb[i] * sp["launch"])
        max_outflow = (Ab[i]/hb) + Ub[i]
        if outflow > max_outflow and outflow > 0:
            s = max_outflow / outflow
            FAb[i] *= s
            outflow = max_outflow
        Ab[i+1] = float(np.clip(Ab[i] + hb*(Ub[i]-outflow), 0.0, sp["A_max"]))
        Mb[i+1] = float(max(0.0, Mb[i] + hb*(sp["L_R"]*(FEb[i]*sp["launch"]) + outflow)))

    Ao[0] = Ab[-1]
    So[0] = 0.0
    for j in range(No):
        if gate_slope > 0:
            FAo[j] = min(FAo[j], gate_slope * Ao[j])
        outflow = sp["L_R"] * (FAo[j] * sp["launch"])
        max_outflow = (Ao[j]/ho) + Uo[j]
        if outflow > max_outflow and outflow > 0:
            s = max_outflow / outflow
            FAo[j] *= s
            outflow = max_outflow
        Ao[j+1] = float(np.clip(Ao[j] + ho*(Uo[j]-outflow), 0.0, sp["A_max"]))
        delivered = sp["L_R"]*(FEo[j]*sp["launch"]) + outflow
        So[j+1] = float(max(0.0, So[j] + ho*delivered))

    # pack
    x0[ii["Ab"]]  = Ab
    x0[ii["Mb"]]  = Mb
    x0[ii["Ao"]]  = Ao
    x0[ii["So"]]  = So
    x0[ii["FEb"]] = FEb
    x0[ii["FAb"]] = FAb
    x0[ii["Ub"]]  = Ub
    x0[ii["FEo"]] = FEo
    x0[ii["FAo"]] = FAo
    x0[ii["Uo"]]  = Uo

    return np.clip(x0, meta["lbx"], meta["ubx"])

# -------------------------
# Solve one NLP (given policy + optional tau upper bound)
# -------------------------
def solve_policy(
    label: str,
    p: Params,
    policy: str,
    tau_ub: Optional[float] = None,
    x0_override: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    sp = scaled(p)
    nlp, meta = build_nlp_variable_tau(p, sp, policy=policy, tau_ub=tau_ub)

    ipopt_opts = {
        "print_level": 0,
        "max_iter": int(p.ipopt_max_iter),
        "tol": float(p.ipopt_tol),
        "mu_strategy": "adaptive",
        "acceptable_tol": float(p.ipopt_acceptable_tol),
        "acceptable_constr_viol_tol": float(p.ipopt_acceptable_constr_viol_tol),
        "acceptable_iter": int(p.ipopt_acceptable_iter),
        "bound_relax_factor": float(p.ipopt_bound_relax_factor),
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, {"ipopt": ipopt_opts, "print_time": False})

    # heuristic tau0
    if sp["FE_cap"] > 0:
        tau_lb = float(sp["M_goal"] / max(sp["L_R"]*sp["launch"]*sp["FE_cap"], 1e-12))
    else:
        tau_lb = 1.0
    tau0 = float(np.clip(max(8.0, tau_lb*1.2), p.tau_min, p.tau_max))
    if tau_ub is not None:
        tau0 = float(np.clip(tau0, p.tau_min, float(tau_ub)))

    x0 = x0_override if x0_override is not None else initial_guess(p, sp, meta, tau0=tau0)

    sol = solver(x0=x0, lbx=meta["lbx"], ubx=meta["ubx"], lbg=meta["lbg"], ubg=meta["ubg"])
    stats = solver.stats()
    status = str(stats.get("return_status", ""))
    success = bool(stats.get("success", False)) or ("Solve_Succeeded" in status) or ("Solved_To_Acceptable_Level" in status)

    x = np.array(sol["x"]).astype(float).flatten()
    g = np.array(sol["g"]).astype(float).flatten()
    v = viol_scaled(g, meta["lbg"], meta["ubg"], meta["g_scale"])

    ii = meta["idx"]
    Nb = meta["Nb"]; No = meta["No"]
    tau = float(x[ii["tau"]][0])

    Ab = x[ii["Ab"]]; Mb = x[ii["Mb"]]
    Ao = x[ii["Ao"]]; So = x[ii["So"]]
    FEb = x[ii["FEb"]]; FAb = x[ii["FAb"]]; Ub = x[ii["Ub"]]
    FEo = x[ii["FEo"]]; FAo = x[ii["FAo"]]; Uo = x[ii["Uo"]]

    # time grids
    t_build = np.linspace(0.0, tau, Nb+1)
    t_ops = np.linspace(tau, tau+1.0, No+1)
    time_total = tau + 1.0

    # Split costs
    cost_rate_b = sp["C_E"]*FEb + sp["C_A"]*FAb + sp["C_SE"]*Ub
    cost_rate_o = sp["C_E"]*FEo + sp["C_A"]*FAo + sp["C_SE"]*Uo
    cost_build = trapz_compat(cost_rate_b, t_build)
    cost_ops = trapz_compat(cost_rate_o, t_ops)
    cost_total = cost_build + cost_ops

    # Linear env index (more report-friendly) + quadratic penalty (as in objective)
    env_rate_b = sp["e_E"]*FEb + sp["e_A"]*FAb + sp["e_SE"]*Ub
    env_rate_o = sp["e_E"]*FEo + sp["e_A"]*FAo + sp["e_SE"]*Uo
    E_total = trapz_compat(env_rate_b, t_build) + trapz_compat(env_rate_o, t_ops)
    env_pen = trapz_compat(p.gamma_env*(env_rate_b**2), t_build) + trapz_compat(p.gamma_env*(env_rate_o**2), t_ops)

    # Launches & throughput split
    FE_launch_build = trapz_compat(FEb, t_build) * sp["launch"]
    FE_launch_ops = trapz_compat(FEo, t_ops) * sp["launch"]
    FA_launch_build = trapz_compat(FAb, t_build) * sp["launch"]
    FA_launch_ops = trapz_compat(FAo, t_ops) * sp["launch"]
    SE_tons_build = trapz_compat(Ub, t_build) * sp["mass"]
    SE_tons_ops = trapz_compat(Uo, t_ops) * sp["mass"]

    A_peak_tons = float(max(np.max(Ab), np.max(Ao)) * sp["mass"])

    # feasibility checks
    build_ok = float(Mb[-1]) >= float(sp["M_goal"]) - 1e-9
    ops_ok = float(So[-1]) >= float(sp["W_year"]) - 1e-9
    constraint_ok = v <= float(p.tol_viol_scaled)

    trig = triggered_flags(p, tau)

    # Save timeseries
    df_build = pd.DataFrame({
        "Segment": ["build"]*(Nb+1),
        "t_yr": t_build,
        "A_Mt": Ab,
        "M_build_Mt": Mb,
        "S_ops_Mt": np.zeros(Nb+1),
        "FE_klaunch_per_yr": FEb,
        "FA_klaunch_per_yr": FAb,
        "U_se_Mt_per_yr": Ub,
    })
    df_ops = pd.DataFrame({
        "Segment": ["ops"]*(No+1),
        "t_yr": t_ops,
        "A_Mt": Ao,
        "M_build_Mt": np.full(No+1, float(Mb[-1])),
        "S_ops_Mt": So,
        "FE_klaunch_per_yr": FEo,
        "FA_klaunch_per_yr": FAo,
        "U_se_Mt_per_yr": Uo,
    })
    df_ts = pd.concat([df_build, df_ops], ignore_index=True)
    ts_path = os.path.join(OUTDIR, f"timeseries_{label}__{policy}.csv")
    df_ts.to_csv(ts_path, index=False)

    return {
        "ScenarioLabel": label,
        "Policy": policy,
        "ScenarioMode": p.scenario,
        "SE_Disrupt": "ON" if p.se_failure_active else "OFF",
        "Rocket_Disrupt": "ON" if p.rocket_failure_active else "OFF",
        **trig,

        "SolverSuccess": "YES" if success else "NO",
        "ConstraintOK": "YES" if constraint_ok else "NO",
        "BuildOK": "YES" if build_ok else "NO",
        "OpsOK": "YES" if ops_ok else "NO",
        "Viol_scaled": float(v),

        "tau_build (yr)": float(tau),
        "Time_total (yr)": float(time_total),
        "M_build_end [Mt]": float(Mb[-1]),
        "S_ops_end [Mt]": float(So[-1]),

        "Cost_build ($B)": float(cost_build),
        "Cost_ops1yr ($B)": float(cost_ops),
        "Cost_total ($B)": float(cost_total),

        "E_total (index)": float(E_total),
        "EnvPenalty": float(env_pen),

        "FE_launch_build": float(FE_launch_build),
        "FE_launch_ops": float(FE_launch_ops),
        "FA_launch_build": float(FA_launch_build),
        "FA_launch_ops": float(FA_launch_ops),
        "SE_tons_build": float(SE_tons_build),
        "SE_tons_ops": float(SE_tons_ops),

        "Apex_Inventory_Peak (tons)": float(A_peak_tons),
        "ReturnStatus": status,
        "TimeseriesCSV": ts_path,

        # Return the raw solution vector as well (for warm-start in caller)
        "_x": x,
    }

# -------------------------
# Feasibility bisection for tau_star (robust shortest time)
# We compute minimal tau s.t. constraints feasible (BuildOK & OpsOK & ConstraintOK).
# -------------------------
def tau_star_bisect(label: str, p: Params, tol_tau: float = 0.1, max_iter: int = 40) -> Tuple[float, Optional[np.ndarray]]:
    sp = scaled(p)

    # Step 1: find a feasible high (start from tau_max)
    hi = float(p.tau_max)
    # Solve a time_opt with tau_ub=hi (effectively just feasible solve), no need for cost optimal
    res_hi = solve_policy(label, p, policy="time_opt", tau_ub=hi)
    if not (res_hi["SolverSuccess"] == "YES" and res_hi["ConstraintOK"] == "YES" and res_hi["BuildOK"] == "YES" and res_hi["OpsOK"] == "YES"):
        raise RuntimeError(f"[Bisection] No feasible solution even at tau_max={hi}. Check parameters or constraints.")
    x_warm = res_hi["_x"]

    lo = float(p.tau_min)

    # Ensure lo is infeasible or feasible; if feasible, tau_star may be tau_min
    try:
        res_lo = solve_policy(label, p, policy="time_opt", tau_ub=lo, x0_override=x_warm)
        lo_feas = (res_lo["SolverSuccess"] == "YES" and res_lo["ConstraintOK"] == "YES" and res_lo["BuildOK"] == "YES" and res_lo["OpsOK"] == "YES")
    except Exception:
        lo_feas = False

    if lo_feas:
        return lo, x_warm

    # Bisection: invariant => hi feasible, lo infeasible
    for _ in range(max_iter):
        if hi - lo <= tol_tau:
            break
        mid = 0.5*(hi + lo)

        # Solve with tau_ub=mid; warm start from last feasible x
        try:
            res_mid = solve_policy(label, p, policy="time_opt", tau_ub=mid, x0_override=x_warm)
            mid_feas = (res_mid["SolverSuccess"] == "YES" and res_mid["ConstraintOK"] == "YES" and res_mid["BuildOK"] == "YES" and res_mid["OpsOK"] == "YES")
        except Exception:
            mid_feas = False

        if mid_feas:
            hi = mid
            x_warm = res_mid["_x"]
        else:
            lo = mid

    return hi, x_warm

# -------------------------
# High-level: run both policies
# -------------------------
def solve_scenario(label: str, p: Params) -> List[Dict[str, Any]]:
    rows = []

    # 1) Shortest time via bisection (defensible tau*)
    tau_star, x_warm = tau_star_bisect(label, p, tol_tau=0.05, max_iter=40)

    # 2) Time-focused cost optimization under epsilon constraint
    tau_ub = (1.0 + float(p.eps_tau)) * float(tau_star)
    res_time = solve_policy(label, p, policy="time_opt", tau_ub=tau_ub, x0_override=x_warm)
    res_time["tau_star_bisect (yr)"] = float(tau_star)
    res_time["tau_ub_eps (yr)"] = float(tau_ub)
    rows.append(res_time)

    # 3) Cost-opt (no tau upper bound) to show long-horizon behavior
    res_cost = solve_policy(label, p, policy="cost_opt", tau_ub=None, x0_override=x_warm)
    res_cost["tau_star_bisect (yr)"] = float(tau_star)
    res_cost["tau_ub_eps (yr)"] = np.nan
    rows.append(res_cost)

    # remove internal _x key for CSV
    for r in rows:
        if "_x" in r:
            del r["_x"]

    return rows

# -------------------------
# Main
# -------------------------
def main():
    print(">>> Running FINAL competition solver (tau* bisection + epsilon time-opt + cost-opt)...")

    base = Params(
        M_goal_tons=1e8,
        scenario="mixed",
        N_build=250,
        N_ops=60,
        tau_min=1.0,
        tau_max=400.0,
        ops_supply_factor=1.0,
        ops_require_rate_each_step=False,  # change to True if you want stricter ops delivery (rate constraint)
        eps_tau=0.05
    )

    cases: List[Tuple[str, Params]] = [
        ("rocket_only_base", Params(**{**base.__dict__, "scenario": "rocket_only"})),
        ("se_only_base",     Params(**{**base.__dict__, "scenario": "se_only"})),
        ("mixed_base",       Params(**{**base.__dict__, "scenario": "mixed"})),

        # Late disruptions (may NOT trigger for time-opt if tau small; trigger flags will show that)
        ("mixed_sefail_late",   Params(**{**base.__dict__, "scenario": "mixed", "se_failure_active": True})),
        ("mixed_bothfail_late", Params(**{**base.__dict__, "scenario": "mixed", "se_failure_active": True, "rocket_failure_active": True})),

        # OPTIONAL: Early disruptions to demonstrate sensitivity during fast builds (recommended for writeup)
        ("mixed_sefail_early", Params(**{**base.__dict__, "scenario": "mixed",
                                         "se_failure_active": True, "fail_t_start": 4.0, "fail_duration": 1.0, "fail_severity": 0.6})),
        ("mixed_bothfail_early", Params(**{**base.__dict__, "scenario": "mixed",
                                           "se_failure_active": True, "fail_t_start": 4.0, "fail_duration": 1.0, "fail_severity": 0.6,
                                           "rocket_failure_active": True, "rocket_fail_t_start": 5.0, "rocket_fail_duration": 0.5, "rocket_fail_severity": 0.4})),
    ]

    rows = []
    for label, cfg in cases:
        print(f"\n--- Solving {label} ---")
        out = solve_scenario(label, cfg)
        rows.extend(out)

    df = pd.DataFrame(rows)

    # Friendly columns ordering
    show_cols = [
        "ScenarioLabel", "Policy", "ScenarioMode", "SE_Disrupt", "Rocket_Disrupt", "SE_Triggered", "Rocket_Triggered",
        "SolverSuccess", "ConstraintOK", "BuildOK", "OpsOK", "Viol_scaled",
        "tau_star_bisect (yr)", "tau_ub_eps (yr)", "tau_build (yr)", "Time_total (yr)",
        "Cost_build ($B)", "Cost_ops1yr ($B)", "Cost_total ($B)",
        "E_total (index)", "EnvPenalty",
        "FE_launch_build", "FE_launch_ops", "FA_launch_build", "FA_launch_ops",
        "SE_tons_build", "SE_tons_ops",
        "Apex_Inventory_Peak (tons)",
        "ReturnStatus", "TimeseriesCSV"
    ]

    out_csv = os.path.join(OUTDIR, "final_summary.csv")
    df.to_csv(out_csv, index=False)

    # print concise view
    print("\n" + df[show_cols].to_string(index=False))
    print(f"\n[Output] {out_csv}")
    print(">>> Done.")

if __name__ == "__main__":
    main()
