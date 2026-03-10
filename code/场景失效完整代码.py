# -*- coding: utf-8 -*-
"""
mcm2026B_competition_solver_final_relfail.py
--------------------------------------------
FINAL (competition-ready) version to STABLY answer MCM 2026 Problem B - Question 2:
"How does the solution change when the system is imperfect?"

What this version adds (while KEEPING all prior functionality):
✅ Relative-time disruptions (DO NOT miss the mission timeline):
   - SE failure window and rocket failure window are defined as FRACTIONS of build time τ
   - So failures always occur during the build/ops timeline (no more “failure at year 20” while build ends at year 8)

Everything else preserved / still inside optimization (not post-processing):
- Variable build completion time tau (decision variable)
- Build + 1-year ops segment with ops supply requirement enforced inside NLP
- Two-policy epsilon-constraint:
    * tau_star via feasibility bisection (minimal feasible τ)
    * cost_opt: minimize cost+env+smooth subject to τ <= (1+eps_tau)*tau_star
  (Optional: time_opt included as a tau-minimization solve for reporting consistency)
- Scenarios: rocket_only | se_only | mixed
- Disruption toggles: se_failure_active / rocket_failure_active
- Robust metrics and time-series CSV outputs for plotting/report
- NumPy trapezoid compatibility

Outputs:
  mcm_final_outputs/final_relfail_summary.csv
  mcm_final_outputs/timeseries_<label>__<policy>.csv

Notes on Q2:
- Because failure windows depend on τ, “mixed_sefail / mixed_bothfail” will now ACTUALLY differ from base.
- This creates meaningful comparisons: delay Δτ, cost increase ΔCost, policy shifts in FE/FA/U during failure window.

Dependencies:
  numpy, pandas, casadi
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


# =========================
# NumPy integral compat
# =========================
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


# =========================
# Scaling
# =========================
@dataclass(frozen=True)
class Scales:
    mass: float = 1e6     # tons -> Mt
    launch: float = 1e3   # launches -> k-launch
    money: float = 1e9    # dollars -> $B


# =========================
# Smooth sigmoid (CasADi-safe and NumPy)
# =========================
def _sigmoid_ca(x, k=10.0):
    z = ca.fmin(ca.fmax(k * x, -50), 50)
    return 1.0 / (1.0 + ca.exp(-z))


def _sigmoid_np(x: float, k=10.0) -> float:
    z = float(np.clip(k * x, -50.0, 50.0))
    return float(1.0 / (1.0 + np.exp(-z)))


def window_health_ca(t, t0, dur, severity, k_step=10.0):
    # w = sig(t-t0) - sig(t-(t0+dur))
    w = _sigmoid_ca(t - t0, k_step) - _sigmoid_ca(t - (t0 + dur), k_step)
    return 1.0 - severity * w


def window_health_np(t: float, t0: float, dur: float, severity: float, k_step=10.0) -> float:
    w = _sigmoid_np(t - t0, k_step) - _sigmoid_np(t - (t0 + dur), k_step)
    return float(1.0 - float(severity) * w)


# =========================
# SE capacity and rocket health (symbolic + numeric)
# =========================
def capacity_logistic_ca(t, K_max, alpha, r, delta_eff,
                         failure_active=False, fail_t_start=0.0, fail_duration=0.0, fail_severity=0.0):
    # K(t) = K_max / (1 + ((1-a)/a)*exp(-r t)) * delta
    a = float(max(min(alpha, 0.999999), 1e-6))
    base = K_max / (1.0 + ((1.0 - a) / a) * ca.exp(-float(r) * t))
    base = base * float(delta_eff)
    if failure_active:
        base = base * window_health_ca(t, fail_t_start, fail_duration, float(fail_severity), k_step=10.0)
    return ca.fmax(base, 0.0)


def capacity_logistic_np(t: float, K_max: float, alpha: float, r: float, delta_eff: float,
                         failure_active=False, fail_t_start=0.0, fail_duration=0.0, fail_severity=0.0) -> float:
    a = max(min(float(alpha), 0.999999), 1e-6)
    base = float(K_max) / (1.0 + ((1.0 - a) / a) * np.exp(-float(r) * float(t)))
    base *= float(delta_eff)
    if failure_active:
        base *= window_health_np(float(t), float(fail_t_start), float(fail_duration), float(fail_severity), k_step=10.0)
    return float(max(base, 0.0))


def rocket_health_ca(t, active=False, fail_t_start=0.0, fail_duration=0.0, fail_severity=0.0):
    if not active:
        return ca.SX(1.0)
    h = window_health_ca(t, fail_t_start, fail_duration, float(fail_severity), k_step=10.0)
    return ca.fmin(ca.fmax(h, 0.0), 1.0)


def rocket_health_np(t: float, active=False, fail_t_start=0.0, fail_duration=0.0, fail_severity=0.0) -> float:
    if not active:
        return 1.0
    h = window_health_np(float(t), float(fail_t_start), float(fail_duration), float(fail_severity), k_step=10.0)
    return float(np.clip(h, 0.0, 1.0))


# =========================
# Params
# =========================
@dataclass
class Params:
    # Goal (construction mass)
    M_goal_tons: float = 1e8  # 100 million tons

    # Population & water (ops year)
    P_pop: float = 1e5
    w_ls_kg_per_person_day: float = 1.5
    ops_supply_factor: float = 1.0

    # Scenario: mixed | se_only | rocket_only
    scenario: str = "mixed"

    # Time discretization
    N_build: int = 250
    N_ops: int = 60

    # Build time variable bounds (years)
    tau_min: float = 1.0
    tau_max: float = 400.0

    # SE capacity
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

    # Emissions index (quadratic penalty)
    e_E_per_launch: float = 1275.0
    e_A_per_launch: float = 450.0
    e_SE_per_ton: float = 2.0
    gamma_env: float = 1e-4

    # Disruptions ON/OFF
    se_failure_active: bool = False
    rocket_failure_active: bool = False

    # -------- NEW: Relative failure windows (stable for Q2) --------
    # Failures are scheduled as fractions of build time tau (not absolute years).
    # Example: theta=0.35 means failure starts at 0.35*tau, duration=0.15*tau.
    se_fail_theta_build: float = 0.35
    se_fail_frac_build: float = 0.15
    se_fail_severity: float = 0.90

    rocket_fail_theta_build: float = 0.55
    rocket_fail_frac_build: float = 0.10
    rocket_fail_severity: float = 0.50

    # Optional: also apply failures during ops year (relative to ops window of 1 year)
    # If these fractions are <=0, ops failures are disabled (default OFF).
    se_fail_theta_ops: float = 0.0
    se_fail_frac_ops: float = 0.0
    rocket_fail_theta_ops: float = 0.0
    rocket_fail_frac_ops: float = 0.0

    # Smoothness & terminal regularization
    w_smooth: float = 0.5
    w_end: float = 1e2

    # Ops requirement mode
    ops_require_rate_each_step: bool = False  # False: cumulative >= W_year; True: enforce per-step rate >= W_req

    # IPOPT
    ipopt_max_iter: int = 40000
    ipopt_tol: float = 1e-8
    ipopt_acceptable_tol: float = 1e-4
    ipopt_acceptable_constr_viol_tol: float = 1e-6
    ipopt_acceptable_iter: int = 800
    ipopt_bound_relax_factor: float = 1e-10

    # Feasibility reporting
    tol_viol_scaled: float = 1e-6

    # Stage-2 epsilon on tau
    eps_tau: float = 0.05

    # Regularization coefficients
    stage1_reg: float = 1e-3


def scaled(p: Params) -> Dict[str, float]:
    sc = Scales()
    scenario = str(p.scenario).lower().strip()
    if scenario not in ("mixed", "se_only", "rocket_only"):
        raise ValueError("scenario must be mixed|se_only|rocket_only")

    # ops year water demand (tons/year)
    W_year_tons = p.P_pop * p.w_ls_kg_per_person_day * 365.0 / 1000.0
    W_year_tons *= float(max(p.ops_supply_factor, 0.0))

    sp = {
        "mass": sc.mass,
        "launch": sc.launch,
        "money": sc.money,

        "M_goal": p.M_goal_tons / sc.mass,        # Mt
        "W_year": W_year_tons / sc.mass,          # Mt within ops year
        "W_req": (W_year_tons / 1.0) / sc.mass,   # Mt/yr average requirement

        "A_max": p.A_max_tons / sc.mass,          # Mt
        "A_gate": p.A_gate_tons / sc.mass,        # Mt
        "L_R": p.L_R_ton_per_launch / sc.mass,    # Mt/launch

        "FE_cap": p.FE_max_launch_per_yr / sc.launch,  # k-launch/yr
        "FA_cap": p.FA_max_launch_per_yr / sc.launch,  # k-launch/yr
        "K_max": p.K_max_total_tpy / sc.mass,          # Mt/yr

        "C_E": (p.C_E_dollars_per_launch * sc.launch) / sc.money,  # $B/(k-launch)
        "C_A": (p.C_A_dollars_per_launch * sc.launch) / sc.money,  # $B/(k-launch)
        "C_SE": (p.C_SE_dollars_per_ton * sc.mass) / sc.money,     # $B/Mt

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


# =========================
# Constraint violation metric (scaled)
# =========================
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


# =========================
# NEW: relative failure window builder (symbolic, depends on tau)
# =========================
def _rel_window_build_ca(theta: float, frac: float, tau):
    # t0 = theta*tau, dur = frac*tau, clipped to [0, tau]
    # (clipping softly not necessary; assume theta, frac chosen in [0,1], theta+frac<=1 recommended)
    t0 = float(theta) * tau
    dur = float(frac) * tau
    return t0, dur


def _rel_window_ops_ca(theta_ops: float, frac_ops: float, tau):
    # ops time is [tau, tau+1]
    if theta_ops <= 0.0 or frac_ops <= 0.0:
        return None
    t0 = tau + float(theta_ops) * 1.0
    dur = float(frac_ops) * 1.0
    return t0, dur


# =========================
# NLP builder (variable tau) with relative failures
# policies:
#   "tau_star" : feasibility solve with tau fixed externally (used for bisection)
#   "time_opt" : minimize tau (stage1)
#   "cost_opt" : minimize cost under tau <= (1+eps)*tau_star (stage2)
# =========================
def build_nlp_variable_tau(p: Params, sp: Dict[str, float], policy: str,
                           tau_star: Optional[float] = None,
                           tau_fixed: Optional[float] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    Nb = int(p.N_build)
    No = int(p.N_ops)

    # decision variable tau unless fixed
    tau = ca.SX.sym("tau", 1)  # years
    hb = tau[0] / Nb
    ho = 1.0 / No  # ops year fixed 1 year

    # States
    Ab = ca.SX.sym("Ab", Nb + 1)   # Apex inventory
    Mb = ca.SX.sym("Mb", Nb + 1)   # build delivered
    Ao = ca.SX.sym("Ao", No + 1)   # Apex inventory ops
    So = ca.SX.sym("So", No + 1)   # ops-year cumulative delivered

    # Controls
    FEb = ca.SX.sym("FEb", Nb + 1)
    FAb = ca.SX.sym("FAb", Nb + 1)
    Ub  = ca.SX.sym("Ub",  Nb + 1)

    FEo = ca.SX.sym("FEo", No + 1)
    FAo = ca.SX.sym("FAo", No + 1)
    Uo  = ca.SX.sym("Uo",  No + 1)

    x = ca.vertcat(tau, Ab, Mb, Ao, So, FEb, FAb, Ub, FEo, FAo, Uo)

    # Index map
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
    lbx = []
    ubx = []

    # tau bounds (or fixed)
    if tau_fixed is None:
        lbx += [float(p.tau_min)]
        ubx += [float(p.tau_max)]
    else:
        lbx += [float(tau_fixed)]
        ubx += [float(tau_fixed)]

    # build states
    lbx += [0.0]*(Nb+1); ubx += [float(sp["A_max"])]*(Nb+1)
    lbx += [0.0]*(Nb+1); ubx += [10.0*float(sp["M_goal"])]*(Nb+1)

    # ops states
    lbx += [0.0]*(No+1); ubx += [float(sp["A_max"])]*(No+1)
    lbx += [0.0]*(No+1); ubx += [10.0*float(sp["W_year"])]*(No+1)

    # controls bounds (rates)
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

    # Relative failure windows (build)
    se_t0_b, se_dur_b = _rel_window_build_ca(p.se_fail_theta_build, p.se_fail_frac_build, tau[0])
    rk_t0_b, rk_dur_b = _rel_window_build_ca(p.rocket_fail_theta_build, p.rocket_fail_frac_build, tau[0])

    # Optional ops windows
    se_ops = _rel_window_ops_ca(p.se_fail_theta_ops, p.se_fail_frac_ops, tau[0])
    rk_ops = _rel_window_ops_ca(p.rocket_fail_theta_ops, p.rocket_fail_frac_ops, tau[0])

    # Constraints
    g, lbg, ubg, g_scale = [], [], [], []

    # Initial conditions build
    g += [Ab[0], Mb[0]]
    lbg += [0.0, 0.0]
    ubg += [0.0, 0.0]
    g_scale += [max(float(sp["A_max"]), 1.0), max(float(sp["M_goal"]), 1.0)]

    # time mapping
    def t_build(i):
        return tau[0] * (float(i)/float(Nb))  # in [0, tau]

    def t_ops(j):
        return tau[0] + float(j)/float(No)   # in [tau, tau+1]

    def rocket_health_at(t_abs):
        # choose window based on whether time is in build or ops
        # For symbolic, we apply build window always; ops window optionally multiplies additional effect.
        rh = ca.SX(1.0)
        if p.rocket_failure_active:
            rh = rh * rocket_health_ca(t_abs, active=True,
                                       fail_t_start=rk_t0_b, fail_duration=rk_dur_b,
                                       fail_severity=float(p.rocket_fail_severity))
            if rk_ops is not None:
                rh = rh * rocket_health_ca(t_abs, active=True,
                                           fail_t_start=rk_ops[0], fail_duration=rk_ops[1],
                                           fail_severity=float(p.rocket_fail_severity))
        return ca.fmin(ca.fmax(rh, 0.0), 1.0)

    def phi_at(t_abs):
        if not p.se_failure_active:
            return capacity_logistic_ca(t_abs, sp["K_max"], p.alpha, p.r, p.delta_eff,
                                        failure_active=False)
        # build window always applies
        base = capacity_logistic_ca(t_abs, sp["K_max"], p.alpha, p.r, p.delta_eff,
                                    failure_active=True,
                                    fail_t_start=se_t0_b, fail_duration=se_dur_b,
                                    fail_severity=float(p.se_fail_severity))
        # ops window optionally multiplies further
        if se_ops is not None:
            base = base * window_health_ca(t_abs, se_ops[0], se_ops[1], float(p.se_fail_severity), k_step=10.0)
        return ca.fmax(base, 0.0)

    # Build dynamics (Euler)
    for i in range(Nb):
        ti = t_build(i)

        Phi_i = phi_at(ti)
        g += [Ub[i] - Phi_i]
        lbg += [-np.inf]; ubg += [0.0]
        g_scale += [max(float(sp["K_max"]), 1.0)]

        if gate_slope > 0:
            g += [FAb[i] - gate_slope * Ab[i]]
            lbg += [-np.inf]; ubg += [0.0]
            g_scale += [max(float(sp["FA_cap"]), 1.0)]

        outflow = sp["L_R"] * (FAb[i] * sp["launch"])  # Mt/yr
        rh = rocket_health_at(ti)
        arrive_E = sp["L_R"] * rh * (FEb[i] * sp["launch"])
        arrive_A = rh * outflow
        dM = arrive_E + arrive_A  # Mt/yr delivered

        gA = Ab[i+1] - (Ab[i] + hb * (Ub[i] - outflow))
        g += [gA]; lbg += [0.0]; ubg += [0.0]
        g_scale += [max(float(sp["A_max"]), 1.0)]

        gM = Mb[i+1] - (Mb[i] + hb * dM)
        g += [gM]; lbg += [0.0]; ubg += [0.0]
        g_scale += [max(float(sp["M_goal"]), 1.0)]

    # Build completion
    g += [Mb[-1] - sp["M_goal"]]
    lbg += [0.0]; ubg += [np.inf]
    g_scale += [max(float(sp["M_goal"]), 1.0)]

    # Connect to ops
    g += [Ao[0] - Ab[-1], So[0]]
    lbg += [0.0, 0.0]; ubg += [0.0, 0.0]
    g_scale += [max(float(sp["A_max"]), 1.0), max(float(sp["W_year"]), 1.0)]

    # Ops dynamics
    for j in range(No):
        tj = t_ops(j)

        Phi_j = phi_at(tj)
        g += [Uo[j] - Phi_j]
        lbg += [-np.inf]; ubg += [0.0]
        g_scale += [max(float(sp["K_max"]), 1.0)]

        if gate_slope > 0:
            g += [FAo[j] - gate_slope * Ao[j]]
            lbg += [-np.inf]; ubg += [0.0]
            g_scale += [max(float(sp["FA_cap"]), 1.0)]

        outflow_o = sp["L_R"] * (FAo[j] * sp["launch"])
        rh_o = rocket_health_at(tj)
        arrive_Eo = sp["L_R"] * rh_o * (FEo[j] * sp["launch"])
        arrive_Ao = rh_o * outflow_o
        dS = arrive_Eo + arrive_Ao

        gA = Ao[j+1] - (Ao[j] + ho * (Uo[j] - outflow_o))
        g += [gA]; lbg += [0.0]; ubg += [0.0]
        g_scale += [max(float(sp["A_max"]), 1.0)]

        gS = So[j+1] - (So[j] + ho * dS)
        g += [gS]; lbg += [0.0]; ubg += [0.0]
        g_scale += [max(float(sp["W_year"]), 1.0)]

        if p.ops_require_rate_each_step:
            g += [dS - sp["W_req"]]
            lbg += [0.0]; ubg += [np.inf]
            g_scale += [max(float(sp["W_req"]), 1.0)]

    # Ops year requirement
    g += [So[-1] - sp["W_year"]]
    lbg += [0.0]; ubg += [np.inf]
    g_scale += [max(float(sp["W_year"]), 1.0)]

    # Epsilon constraint for cost_opt
    if policy == "cost_opt":
        if tau_star is None:
            raise ValueError("cost_opt requires tau_star")
        g += [tau[0] - float((1.0 + p.eps_tau) * tau_star)]
        lbg += [-np.inf]; ubg += [0.0]
        g_scale += [max(float(p.tau_max), 1.0)]

    # Objective: integrate cost+env+smooth over build+ops
    # scale
    tau_nom = 50.0
    FE_ref = min(float(sp["FE_cap"]), float(sp["M_goal"]) / max(tau_nom * float(sp["L_R"]) * float(sp["launch"]), 1e-12)) if sp["FE_cap"] > 0 else 0.0
    FA_ref = min(float(sp["FA_cap"]), float(sp["M_goal"]) / max(tau_nom * float(sp["L_R"]) * float(sp["launch"]), 1e-12)) if sp["FA_cap"] > 0 else 0.0
    U_ref = min(float(sp["K_max"]), float(sp["M_goal"]) / max(tau_nom, 1e-12)) if sp["K_max"] > 0 else 0.0
    cost_scale = max(1e-3, tau_nom * (sp["C_E"]*FE_ref + sp["C_A"]*FA_ref + sp["C_SE"]*U_ref))

    J = 0.0
    # build
    for i in range(Nb):
        cost_rate = sp["C_E"]*FEb[i] + sp["C_A"]*FAb[i] + sp["C_SE"]*Ub[i]
        env_rate  = sp["e_E"]*FEb[i] + sp["e_A"]*FAb[i] + sp["e_SE"]*Ub[i]
        env_pen   = p.gamma_env * (env_rate**2)

        dFE = (FEb[i+1] - FEb[i]) / (hb + 1e-12)
        dFA = (FAb[i+1] - FAb[i]) / (hb + 1e-12)
        dU  = (Ub[i+1]  - Ub[i])  / (hb + 1e-12)
        smooth = p.w_smooth*(dFE**2 + dFA**2 + dU**2)
        J += hb * (cost_rate + env_pen + smooth) / cost_scale

    # ops
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

    # policy objective
    if policy == "time_opt":
        # minimize tau primarily
        J = tau[0] + float(p.stage1_reg) * J
    elif policy in ("cost_opt", "tau_star"):
        # tau_star (feasibility) can minimize small regularization; cost_opt minimizes full J
        # (tau_star objective doesn't matter since tau is fixed in bisection calls)
        J = J
    else:
        raise ValueError("policy must be one of: tau_star, time_opt, cost_opt")

    nlp = {"x": x, "f": J, "g": ca.vertcat(*g)}
    meta = {
        "idx": idx,
        "Nb": Nb,
        "No": No,
        "lbx": lbx, "ubx": ubx,
        "lbg": np.array(lbg, float), "ubg": np.array(ubg, float),
        "g_scale": np.array(g_scale, float),
        "cost_scale": float(cost_scale),
    }
    return nlp, meta


# =========================
# Initial guess (consistent Euler, fast)
# =========================
def initial_guess(p: Params, sp: Dict[str, float], meta: Dict[str, Any], tau0: float) -> np.ndarray:
    Nb = meta["Nb"]; No = meta["No"]
    ii = meta["idx"]

    x0 = np.zeros(meta["ubx"].shape[0], float)
    x0[ii["tau"]] = np.clip(tau0, p.tau_min, p.tau_max)

    # nominal avg rate
    rh_nom = 0.9
    avg_klaunch = float(sp["M_goal"] / max(tau0 * sp["L_R"] * sp["launch"] * rh_nom, 1e-12))

    FEb = np.zeros(Nb+1); FAb = np.zeros(Nb+1); Ub = np.zeros(Nb+1)
    FEo = np.zeros(No+1); FAo = np.zeros(No+1); Uo = np.zeros(No+1)

    scenario = str(p.scenario).lower().strip()
    if scenario == "rocket_only":
        FEb[:] = min(sp["FE_cap"], avg_klaunch)
        FAb[:] = 0.0
        Ub[:] = 0.0
    elif scenario == "se_only":
        FEb[:] = 0.0
        FAb[:] = min(sp["FA_cap"], avg_klaunch)
        Ub[:] = min(sp["K_max"], sp["K_max"])
    else:
        FEb[:] = min(sp["FE_cap"], 0.35*avg_klaunch)
        FAb[:] = min(sp["FA_cap"], 0.65*avg_klaunch)
        Ub[:]  = min(sp["K_max"], sp["K_max"])

    # ops: small vs construction; keep steady
    FEo[:] = min(sp["FE_cap"], FEb[-1])
    FAo[:] = min(sp["FA_cap"], FAb[-1])
    Uo[:]  = min(sp["K_max"], Ub[-1])

    Ab = np.zeros(Nb+1); Mb = np.zeros(Nb+1)
    Ao = np.zeros(No+1); So = np.zeros(No+1)

    gate_slope = float(sp["FA_cap"]/sp["A_gate"]) if (sp["A_gate"]>0 and sp["FA_cap"]>0) else 0.0
    hb = tau0 / Nb
    ho = 1.0 / No

    # build forward sim
    for i in range(Nb):
        if gate_slope > 0:
            FAb[i] = min(FAb[i], gate_slope * Ab[i])
        outflow = sp["L_R"] * (FAb[i] * sp["launch"])
        max_outflow = (Ab[i]/hb) + Ub[i]
        if outflow > max_outflow and outflow > 0:
            scale = max_outflow / outflow
            FAb[i] *= scale
            outflow = max_outflow
        Ab[i+1] = np.clip(Ab[i] + hb*(Ub[i]-outflow), 0.0, sp["A_max"])
        Mb[i+1] = max(0.0, Mb[i] + hb*(sp["L_R"]*(FEb[i]*sp["launch"]) + outflow))

    Ao[0] = Ab[-1]
    So[0] = 0.0
    for j in range(No):
        if gate_slope > 0:
            FAo[j] = min(FAo[j], gate_slope * Ao[j])
        outflow = sp["L_R"] * (FAo[j] * sp["launch"])
        max_outflow = (Ao[j]/ho) + Uo[j]
        if outflow > max_outflow and outflow > 0:
            scale = max_outflow / outflow
            FAo[j] *= scale
            outflow = max_outflow
        Ao[j+1] = np.clip(Ao[j] + ho*(Uo[j]-outflow), 0.0, sp["A_max"])
        delivered = sp["L_R"]*(FEo[j]*sp["launch"]) + outflow
        So[j+1] = max(0.0, So[j] + ho*delivered)

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


# =========================
# Physical postprocess (for plots/metrics only)
# =========================
def postprocess_physical(A: np.ndarray, M: np.ndarray, S: np.ndarray,
                         FE: np.ndarray, FA: np.ndarray, U: np.ndarray,
                         Phi: np.ndarray, eps: float = 1e-12):
    A_pp = np.maximum(A, 0.0)
    M_pp = np.maximum.accumulate(np.maximum(M, 0.0))
    S_pp = np.maximum.accumulate(np.maximum(S, 0.0))
    FE_pp = np.maximum(FE, 0.0)
    FA_pp = np.maximum(FA, 0.0)
    U_pp = np.minimum(np.maximum(U, 0.0), Phi)
    # tiny eps
    for arr in (A_pp, M_pp, S_pp, FE_pp, FA_pp, U_pp):
        arr[np.abs(arr) < eps] = 0.0
    return A_pp, M_pp, S_pp, FE_pp, FA_pp, U_pp


# =========================
# Solve a single NLP instance
# =========================
def solve_policy(p: Params, label: str, policy: str,
                 tau_star: Optional[float] = None,
                 tau_fixed: Optional[float] = None) -> Dict[str, Any]:
    sp = scaled(p)
    nlp, meta = build_nlp_variable_tau(p, sp, policy=policy, tau_star=tau_star, tau_fixed=tau_fixed)

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

    # initial tau guess
    if tau_fixed is not None:
        tau0 = float(tau_fixed)
    else:
        if sp["FE_cap"] > 0:
            tau_lb = float(sp["M_goal"] / max(sp["L_R"]*sp["launch"]*sp["FE_cap"], 1e-12))
        else:
            tau_lb = 1.0
        tau0 = float(np.clip(max(8.0, tau_lb*1.2), p.tau_min, p.tau_max))

    x0 = initial_guess(p, sp, meta, tau0=tau0)

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
    t_ops   = np.linspace(tau, tau+1.0, No+1)

    # Merge series for single CSV
    t_all = np.concatenate([t_build, t_ops[1:]])
    A_all = np.concatenate([Ab, Ao[1:]])
    M_all = np.concatenate([Mb, np.full(No, float(Mb[-1]))])
    S_all = np.concatenate([np.zeros(Nb+1), So[1:]])
    FE_all = np.concatenate([FEb, FEo[1:]])
    FA_all = np.concatenate([FAb, FAo[1:]])
    U_all  = np.concatenate([Ub,  Uo[1:]])

    # Postcompute Phi and rocket_health numerically for plotting/interpretation
    # Use the SAME relative-failure logic as in NLP.
    # Build windows:
    se_t0_b = p.se_fail_theta_build * tau
    se_dur_b = p.se_fail_frac_build * tau
    rk_t0_b = p.rocket_fail_theta_build * tau
    rk_dur_b = p.rocket_fail_frac_build * tau
    # Ops windows (optional):
    se_ops = None
    rk_ops = None
    if p.se_fail_theta_ops > 0 and p.se_fail_frac_ops > 0:
        se_ops = (tau + p.se_fail_theta_ops*1.0, p.se_fail_frac_ops*1.0)
    if p.rocket_fail_theta_ops > 0 and p.rocket_fail_frac_ops > 0:
        rk_ops = (tau + p.rocket_fail_theta_ops*1.0, p.rocket_fail_frac_ops*1.0)

    Phi_all = np.zeros_like(t_all, float)
    rh_all  = np.ones_like(t_all, float)
    for k, tt in enumerate(t_all):
        # Phi
        if p.se_failure_active:
            Phi_val = capacity_logistic_np(tt, sp["K_max"], p.alpha, p.r, p.delta_eff,
                                           failure_active=True,
                                           fail_t_start=se_t0_b, fail_duration=se_dur_b,
                                           fail_severity=p.se_fail_severity)
            if se_ops is not None:
                Phi_val *= window_health_np(tt, se_ops[0], se_ops[1], p.se_fail_severity, k_step=10.0)
            Phi_all[k] = max(Phi_val, 0.0)
        else:
            Phi_all[k] = capacity_logistic_np(tt, sp["K_max"], p.alpha, p.r, p.delta_eff, failure_active=False)

        # rocket health
        if p.rocket_failure_active:
            rh = rocket_health_np(tt, active=True,
                                  fail_t_start=rk_t0_b, fail_duration=rk_dur_b,
                                  fail_severity=p.rocket_fail_severity)
            if rk_ops is not None:
                rh *= rocket_health_np(tt, active=True,
                                       fail_t_start=rk_ops[0], fail_duration=rk_ops[1],
                                       fail_severity=p.rocket_fail_severity)
            rh_all[k] = float(np.clip(rh, 0.0, 1.0))
        else:
            rh_all[k] = 1.0

    # Physical postprocess for reporting/plots
    A_pp, M_pp, S_pp, FE_pp, FA_pp, U_pp = postprocess_physical(
        A_all, M_all, S_all, FE_all, FA_all, U_all, Phi_all, eps=1e-12
    )

    # Metrics (use postprocessed)
    cost_rate = sp["C_E"]*FE_pp + sp["C_A"]*FA_pp + sp["C_SE"]*U_pp
    total_cost_B = trapz_compat(cost_rate, t_all)

    env_rate = sp["e_E"]*FE_pp + sp["e_A"]*FA_pp + sp["e_SE"]*U_pp
    E_total = trapz_compat(env_rate, t_all)  # index integral
    env_pen = trapz_compat(p.gamma_env*(env_rate**2), t_all)

    total_FE_launches = trapz_compat(FE_pp, t_all) * sp["launch"]
    total_FA_launches = trapz_compat(FA_pp, t_all) * sp["launch"]
    total_SE_tons = trapz_compat(U_pp, t_all) * sp["mass"]
    A_peak_tons = float(np.max(A_pp) * sp["mass"])

    build_ok = float(Mb[-1]) >= float(sp["M_goal"]) - 1e-9
    ops_ok = float(So[-1]) >= float(sp["W_year"]) - 1e-9
    constraint_ok = v <= float(p.tol_viol_scaled)

    # Save timeseries
    df_ts = pd.DataFrame({
        "ScenarioLabel": [label]*len(t_all),
        "Policy": [policy]*len(t_all),
        "ScenarioMode": [p.scenario]*len(t_all),
        "SE_Disrupt": ["ON" if p.se_failure_active else "OFF"]*len(t_all),
        "Rocket_Disrupt": ["ON" if p.rocket_failure_active else "OFF"]*len(t_all),
        "t_yr": t_all,

        "A_Mt_raw": A_all,
        "M_build_Mt_raw": M_all,
        "S_ops_Mt_raw": S_all,
        "FE_klaunch_per_yr_raw": FE_all,
        "FA_klaunch_per_yr_raw": FA_all,
        "U_se_Mt_per_yr_raw": U_all,

        "A_Mt": A_pp,
        "M_build_Mt": M_pp,
        "S_ops_Mt": S_pp,
        "FE_klaunch_per_yr": FE_pp,
        "FA_klaunch_per_yr": FA_pp,
        "U_se_Mt_per_yr": U_pp,

        "Phi_se_Mt_per_yr": Phi_all,
        "rocket_health": rh_all,
    })
    ts_path = os.path.join(OUTDIR, f"timeseries_{label}__{policy}.csv")
    df_ts.to_csv(ts_path, index=False)

    return {
        "ScenarioLabel": label,
        "Policy": policy,
        "ScenarioMode": p.scenario,
        "SE_Disrupt": "ON" if p.se_failure_active else "OFF",
        "Rocket_Disrupt": "ON" if p.rocket_failure_active else "OFF",
        "SolverSuccess": "YES" if success else "NO",
        "ConstraintOK": "YES" if constraint_ok else "NO",
        "BuildOK": "YES" if build_ok else "NO",
        "OpsOK": "YES" if ops_ok else "NO",
        "Viol_scaled": float(v),

        "tau_star_bisect (yr)": float(tau_fixed) if (policy == "tau_star" and tau_fixed is not None) else np.nan,
        "tau_build (yr)": float(tau),

        "TotalCost ($B)": float(total_cost_B),
        "E_total (index)": float(E_total),
        "EnvPenalty": float(env_pen),
        "Total_FE_Launches": float(total_FE_launches),
        "Total_FA_Launches": float(total_FA_launches),
        "Total_SE_Throughput (tons)": float(total_SE_tons),
        "Apex_Inventory_Peak (tons)": float(A_peak_tons),

        "ReturnStatus": status,
        "TimeseriesCSV": ts_path,
    }


# =========================
# Feasibility bisection for tau_star
# =========================
def tau_star_bisect(p: Params, label: str, max_iter: int = 24) -> float:
    sp = scaled(p)

    lo = float(p.tau_min)
    hi = float(p.tau_max)

    # quick helper: check feasibility at fixed tau
    def feasible(tau_val: float) -> bool:
        res = solve_policy(p, label=label, policy="tau_star", tau_fixed=tau_val)
        return (res["SolverSuccess"] == "YES"
                and res["ConstraintOK"] == "YES"
                and res["BuildOK"] == "YES"
                and res["OpsOK"] == "YES")

    # ensure hi feasible; if not, return tau_max (still usable but indicates tight system)
    if not feasible(hi):
        return float(p.tau_max)

    # find a feasible lo? if lo feasible then tau_star ~ lo
    if feasible(lo):
        return lo

    # bisection
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            hi = mid
        else:
            lo = mid

    return float(hi)


# =========================
# High-level solve per scenario: tau_star + time_opt + cost_opt
# =========================
def solve_all_policies(label: str, p: Params) -> List[Dict[str, Any]]:
    # 1) tau_star by bisection (guarantees timeline meaning)
    tau_star = tau_star_bisect(p, label=label)

    # record tau_star row by solving at fixed tau_star (for timeseries + metrics)
    res_tau = solve_policy(p, label=label, policy="tau_star", tau_fixed=tau_star)

    # 2) time_opt (minimize tau) — mainly for cross-check; may return ~tau_star
    res_time = solve_policy(p, label=label, policy="time_opt", tau_star=None, tau_fixed=None)
    res_time["tau_star_bisect (yr)"] = float(tau_star)

    # 3) cost_opt under tau <= (1+eps)*tau_star
    res_cost = solve_policy(p, label=label, policy="cost_opt", tau_star=tau_star, tau_fixed=None)
    res_cost["tau_star_bisect (yr)"] = float(tau_star)

    return [res_tau, res_time, res_cost]


def main():
    print(">>> Running FINAL solver with relative-time failures (stable Q2 comparisons)...")

    base = Params(
        M_goal_tons=1e8,
        scenario="mixed",
        N_build=250,
        N_ops=60,
        tau_min=1.0,
        tau_max=400.0,
        ops_supply_factor=1.0,
        ops_require_rate_each_step=False,

        # relative failure defaults (you can tune in report)
        se_fail_theta_build=0.35, se_fail_frac_build=0.15, se_fail_severity=0.90,
        rocket_fail_theta_build=0.55, rocket_fail_frac_build=0.10, rocket_fail_severity=0.50,
    )

    cases: List[Tuple[str, Params]] = [
        ("rocket_only_base", Params(**{**base.__dict__, "scenario": "rocket_only", "se_failure_active": False, "rocket_failure_active": False})),
        ("se_only_base",     Params(**{**base.__dict__, "scenario": "se_only",     "se_failure_active": False, "rocket_failure_active": False})),
        ("mixed_base",       Params(**{**base.__dict__, "scenario": "mixed",       "se_failure_active": False, "rocket_failure_active": False})),

        # Q2 scenarios (now ALWAYS inside timeline because relative to tau)
        ("mixed_sefail",     Params(**{**base.__dict__, "scenario": "mixed", "se_failure_active": True,  "rocket_failure_active": False})),
        ("mixed_rocketfail", Params(**{**base.__dict__, "scenario": "mixed", "se_failure_active": False, "rocket_failure_active": True})),
        ("mixed_bothfail",   Params(**{**base.__dict__, "scenario": "mixed", "se_failure_active": True,  "rocket_failure_active": True})),
    ]

    rows: List[Dict[str, Any]] = []
    for label, cfg in cases:
        print(f"\n--- Solving {label} ---")
        rows.extend(solve_all_policies(label, cfg))

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTDIR, "final_relfail_summary.csv")
    df.to_csv(out_csv, index=False)

    show_cols = [
        "ScenarioLabel", "Policy", "ScenarioMode", "SE_Disrupt", "Rocket_Disrupt",
        "SolverSuccess", "ConstraintOK", "BuildOK", "OpsOK",
        "Viol_scaled", "tau_star_bisect (yr)", "tau_build (yr)",
        "TotalCost ($B)", "E_total (index)", "EnvPenalty",
        "Total_FE_Launches", "Total_FA_Launches",
        "Total_SE_Throughput (tons)", "Apex_Inventory_Peak (tons)",
        "ReturnStatus", "TimeseriesCSV"
    ]
    print("\n" + df[show_cols].to_string(index=False))
    print(f"\n[Output] {out_csv}")
    print(">>> Done.")


if __name__ == "__main__":
    main()
