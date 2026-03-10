# -*- coding: utf-8 -*-
"""
MCM 2026 Problem B - Final Robust Solver (Parameterized + Feasibility Slack + Non-Stalling)
-----------------------------------------------------------------------------------------
Default: Base-test run (fast & stable).
Full run: python mcm_b_solver_final.py --full

Key upgrades:
1) Terminal slack s>=0: M(T)+s = M_goal, penalize s^2 with continuation slack_w schedule
2) Parameterized NLP (nlp['p']) so heatmap/MC/tornado reuse same compiled solver (per N,T,k)
3) IPOPT: max_cpu_time + acceptable_tol/iter + warm-start (x, lam_x, lam_g)
4) fast batch tasks: lower N, capped k schedule
5) control scaling FE/FA
6) multi-start + structured initial guesses
7) no seaborn; matplotlib-only plots

Outputs:
- mcm_final_outputs/Fig1_OptimalSwitching.png
- mcm_final_outputs/Fig2_Resilience.png
- mcm_final_outputs/Fig4_Sensitivity_Heatmap.png
- mcm_final_outputs/Fig5_MonteCarlo_Robustness.png
- mcm_final_outputs/Fig6_Tornado_Sensitivity.png
- mcm_final_outputs/final_summary.csv
- mcm_final_outputs/final_summary.tex
"""

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import casadi as ca
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------
# Global Style / Output
# ----------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "axes.grid": True
})
warnings.filterwarnings("ignore")

OUTDIR = "mcm_final_outputs"
os.makedirs(OUTDIR, exist_ok=True)


# ----------------------------
# Params
# ----------------------------
@dataclass
class Params:
    # Mission
    M_goal: float = 1e8          # tons
    P_pop: float = 1e5
    w_ls: float = 1.5            # kg/person/day
    W_req: float = 0.0           # tons/year computed

    # SE throughput logistic cap
    K_max_total: float = 1.5e7   # tons/year
    alpha: float = 0.01
    r: float = 0.18
    delta_eff: float = 0.974

    # Failure / resilience
    failure_active: bool = False
    fail_t_start: float = 20.0
    fail_duration: float = 3.0
    fail_severity: float = 0.90

    # Transport / bounds
    L_R: float = 125.0           # tons/launch
    FE_max: float = 100000.0     # 1/yr
    FA_max: float = 200000.0     # 1/yr
    A_max: float = 5e5           # tons

    # Cost / environment
    C_E: float = 1.0e8
    C_A: float = 1.0e7
    e_E: float = 1275.0
    gamma: float = 1e-6

    # Smoothing & penalties
    w_smooth: float = 1.0
    w_late: float = 1e5
    late_fraction: float = 0.30
    w_over: float = 1e3

    # Slack feasibility continuation
    slack_weight_schedule: Tuple[float, ...] = (1e6, 1e8, 1e10)
    slack_tol_tons: float = 1e5   # reporting feasible if s <= 1e5 tons

    # Horizon / grid
    T: float = 50.0
    N_main: int = 100
    N_fast: int = 40

    # Phase-out detection
    phaseout_eps: float = 10.0
    phaseout_confirm_n: int = 4

    # Homotopy sharpness (capped)
    k_schedule_main: Tuple[float, ...] = (1.0, 5.0, 15.0, 40.0)
    k_schedule_fast: Tuple[float, ...] = (10.0, 30.0)

    # IPOPT main
    ipopt_max_iter_main: int = 2500
    ipopt_tol_main: float = 1e-6
    ipopt_cpu_main: float = 30.0

    # IPOPT fast
    ipopt_max_iter_fast: int = 600
    ipopt_tol_fast: float = 2e-5
    ipopt_cpu_fast: float = 8.0

    # Control scaling
    FE_scale: float = 1e4
    FA_scale: float = 1e4


def finalize_params(p: Params) -> Params:
    p.W_req = p.P_pop * p.w_ls * 365.0 / 1000.0
    return p


# ----------------------------
# Smooth primitives
# ----------------------------
def sigma_k(z, k):
    return 1.0 / (1.0 + ca.exp(-ca.fmin(ca.fmax(k * z, -30), 30)))


def gk_relu(z, k):
    return z * sigma_k(z, k)


# ----------------------------
# Parameterized capacity logic
# ----------------------------
def capacity_logic_param(t, par8, is_numeric=False):
    """
    par8 layout:
      0 K_max_total
      1 alpha
      2 r
      3 delta_eff
      4 fail_active (0/1)
      5 fail_t_start
      6 fail_duration
      7 fail_severity
    """
    exp_fn = np.exp if is_numeric else ca.exp
    K = par8[0]
    a = par8[1]
    rr = par8[2]
    delta_eff = par8[3]

    base_cap = K / (1.0 + ((1.0 - a) / a) * exp_fn(-rr * t))

    fail_active = par8[4]
    fail_t_start = par8[5]
    fail_duration = par8[6]
    fail_severity = par8[7]

    k_step = 10.0
    if is_numeric:
        def num_sig(x): return 1.0 / (1.0 + np.exp(-np.clip(k_step * x, -30, 30)))
        start_step = num_sig(t - float(fail_t_start))
        end_step = num_sig(t - float(fail_t_start + fail_duration))
    else:
        start_step = sigma_k(t - fail_t_start, k_step)
        end_step = sigma_k(t - (fail_t_start + fail_duration), k_step)

    window = start_step - end_step
    health = 1.0 - fail_active * fail_severity * window
    return base_cap * delta_eff * health


# ----------------------------
# Build NLP (parameterized, includes slack s)
# ----------------------------
def build_hs_nlp_parameterized(p: Params, N: int, k: float) -> Dict[str, Any]:
    """
    States: A(t), M(t)
    Controls (scaled): FE_s(t), FA_s(t)
    Slack: s>=0, with terminal equality M(T)+s=M_goal

    NLP parameter vector par:
      [0:8]  capacity+failure params
      8      C_E
      9      C_A
      10     gamma
      11     L_R
      12     slack_w (penalty)
    """
    p = finalize_params(p)
    N = int(N)
    T = float(p.T)
    h = T / N
    tgrid = np.linspace(0.0, T, N + 1)

    # Decision variables
    A = ca.SX.sym("A", N + 1)
    M = ca.SX.sym("M", N + 1)
    FE_s = ca.SX.sym("FE_s", N + 1)
    FA_s = ca.SX.sym("FA_s", N + 1)
    s = ca.SX.sym("s", 1)

    # Scaled -> real controls
    FE = p.FE_scale * FE_s
    FA = p.FA_scale * FA_s

    x = ca.vertcat(A, M, FE_s, FA_s, s)
    idx = {
        "A": slice(0, N + 1),
        "M": slice(N + 1, 2 * N + 2),
        "FE_s": slice(2 * N + 2, 3 * N + 3),
        "FA_s": slice(3 * N + 3, 4 * N + 4),
        "s": slice(4 * N + 4, 4 * N + 5),
    }

    # Parameters
    K_max_total = ca.SX.sym("K_max_total")
    alpha = ca.SX.sym("alpha")
    rr = ca.SX.sym("r")
    delta_eff = ca.SX.sym("delta_eff")
    fail_active = ca.SX.sym("fail_active")
    fail_t_start = ca.SX.sym("fail_t_start")
    fail_duration = ca.SX.sym("fail_duration")
    fail_severity = ca.SX.sym("fail_severity")

    C_E = ca.SX.sym("C_E")
    C_A = ca.SX.sym("C_A")
    gamma = ca.SX.sym("gamma")
    L_R = ca.SX.sym("L_R")
    slack_w = ca.SX.sym("slack_w")

    par = ca.vertcat(
        K_max_total, alpha, rr, delta_eff,
        fail_active, fail_t_start, fail_duration, fail_severity,
        C_E, C_A, gamma, L_R, slack_w
    )

    # Bounds
    lbx = np.zeros(x.shape[0], dtype=float)
    ubx = np.concatenate([
        np.full(N + 1, p.A_max),
        np.full(N + 1, p.M_goal * 3.0),
        np.full(N + 1, p.FE_max / p.FE_scale),
        np.full(N + 1, p.FA_max / p.FA_scale),
        np.array([p.M_goal * 3.0])  # slack upper bound
    ])

    # Constraints
    g = []
    lbg = []
    ubg = []

    # initial conditions
    g += [A[0], M[0]]
    lbg += [0.0, 0.0]
    ubg += [0.0, 0.0]

    def f(Ax, Mx, FEsx, FAsx, t):
        FEx = p.FE_scale * FEsx
        FAx = p.FA_scale * FAsx

        Phi_se = capacity_logic_param(t, par[0:8], is_numeric=False)
        inflow = Phi_se * sigma_k(p.A_max - Ax, k)
        outflow = L_R * FAx * sigma_k(Ax, k)

        dA = inflow - outflow
        Phi_gross = L_R * (FEx + FAx)
        dM = gk_relu(Phi_gross - p.W_req, k)
        return dA, dM

    for i in range(N):
        ti = tgrid[i]
        dA_i, dM_i = f(A[i], M[i], FE_s[i], FA_s[i], ti)
        dA_ip1, dM_ip1 = f(A[i + 1], M[i + 1], FE_s[i + 1], FA_s[i + 1], ti + h)

        A_m = 0.5 * (A[i] + A[i + 1]) + (h / 8) * (dA_i - dA_ip1)
        M_m = 0.5 * (M[i] + M[i + 1]) + (h / 8) * (dM_i - dM_ip1)
        FE_sm = 0.5 * (FE_s[i] + FE_s[i + 1])
        FA_sm = 0.5 * (FA_s[i] + FA_s[i + 1])
        dA_m, dM_m = f(A_m, M_m, FE_sm, FA_sm, ti + 0.5 * h)

        g += [A[i + 1] - A[i] - (h / 6) * (dA_i + 4 * dA_m + dA_ip1)]
        g += [M[i + 1] - M[i] - (h / 6) * (dM_i + 4 * dM_m + dM_ip1)]
        lbg += [0.0, 0.0]
        ubg += [0.0, 0.0]

    # terminal equality with slack
    g_idx_terminal = len(g)
    g += [M[-1] + s[0] - p.M_goal]
    lbg += [0.0]
    ubg += [0.0]

    # Objective
    J = 0
    late_start_idx = int((1.0 - p.late_fraction) * N)

    for i in range(N):
        FEavg_s = 0.5 * (FE_s[i] + FE_s[i + 1])
        FAavg_s = 0.5 * (FA_s[i] + FA_s[i + 1])
        FEavg = p.FE_scale * FEavg_s
        FAavg = p.FA_scale * FAavg_s

        cost = C_E * FEavg + C_A * FAavg
        env = gamma * (p.e_E * FEavg) ** 2

        # IMPORTANT: smoothing on real FE/FA rate changes (scale back!)
        dFE_dt = p.FE_scale * (FE_s[i + 1] - FE_s[i]) / h
        dFA_dt = p.FA_scale * (FA_s[i + 1] - FA_s[i]) / h
        smooth = p.w_smooth * (dFE_dt ** 2 + dFA_dt ** 2)

        late_penalty = 0.0
        if i >= late_start_idx and p.w_late > 0:
            FEi = p.FE_scale * FE_s[i]
            FEip1 = p.FE_scale * FE_s[i + 1]
            late_penalty = p.w_late * (FEi ** 2 + FEip1 ** 2)

        J += h * (cost + env + smooth + late_penalty)

    # Slack penalty dominates feasibility
    J += slack_w * (s[0] ** 2)

    # Secondary overbuild (with equality, (M-Mgoal)^2 == s^2, but keep for paper/consistency)
    if p.w_over > 0:
        J += p.w_over * ((M[-1] - p.M_goal) ** 2)

    J_scale = 1e9
    nlp = {"x": x, "f": J / J_scale, "g": ca.vertcat(*g), "p": par}

    return {
        "nlp": nlp,
        "lbx": lbx,
        "ubx": ubx,
        "lbg": np.array(lbg, dtype=float),
        "ubg": np.array(ubg, dtype=float),
        "meta": {
            "tgrid": tgrid,
            "h": h,
            "N": N,
            "k": float(k),
            "idx": idx,
            "J_scale": float(J_scale),
            "g_idx_terminal": int(g_idx_terminal),
        }
    }


# ----------------------------
# Solver Cache
# ----------------------------
class SolverCache:
    """
    Cache by structure only: (N, T, k, fast_flag).
    Scenario params (K/gamma/L_R/failure/slack_w...) are passed via nlp['p'].
    """
    def __init__(self):
        self.cache: Dict[Tuple[int, float, float, bool], Dict[str, Any]] = {}
        self._solver_count = 0

    def get(self, p: Params, N: int, k: float, fast: bool) -> Dict[str, Any]:
        key = (int(N), float(p.T), float(k), bool(fast))
        if key in self.cache:
            return self.cache[key]

        built = build_hs_nlp_parameterized(p, N=N, k=k)

        opts = ipopt_opts(p, fast=fast)
        self._solver_count += 1
        solver_name = f"solver_{self._solver_count}"
        solver = ca.nlpsol(solver_name, "ipopt", built["nlp"], opts)

        built["solver"] = solver
        self.cache[key] = built
        return built


def ipopt_opts(p: Params, fast: bool) -> Dict[str, Any]:
    if fast:
        max_iter = int(p.ipopt_max_iter_fast)
        tol = float(p.ipopt_tol_fast)
        cpu = float(p.ipopt_cpu_fast)
        acceptable_tol = 1e-3
    else:
        max_iter = int(p.ipopt_max_iter_main)
        tol = float(p.ipopt_tol_main)
        cpu = float(p.ipopt_cpu_main)
        acceptable_tol = 1e-5

    return {
        "ipopt": {
            "print_level": 0,
            "max_iter": max_iter,
            "tol": tol,
            "max_cpu_time": cpu,
            "acceptable_tol": acceptable_tol,
            "acceptable_iter": 15,
            "mu_strategy": "adaptive",
            "warm_start_init_point": "yes",
        },
        "print_time": False
    }


# ----------------------------
# Structured initial guesses (multi-start)
# ----------------------------
def make_structured_x0(p: Params, meta: Dict[str, Any], style: int) -> np.ndarray:
    """
    style 0: front-loaded FE, moderate FA
    style 1: higher FE & FA
    style 2: low FE, higher FA (forces movement)
    """
    N = meta["N"]
    t = meta["tgrid"]
    idx = meta["idx"]

    x0 = np.zeros(4 * (N + 1) + 1, dtype=float)

    if style == 0:
        FE_hi = 0.50
        FA_lv = 0.15
    elif style == 1:
        FE_hi = 0.70
        FA_lv = 0.25
    else:
        FE_hi = 0.05
        FA_lv = 0.35

    t1 = 0.25 * p.T
    t2 = 0.65 * p.T

    FE0_s = np.zeros_like(t)
    FE0_s[t <= t1] = FE_hi * (p.FE_max / p.FE_scale)
    mid = (t > t1) & (t <= t2)
    if np.any(mid):
        FE0_s[mid] = np.linspace(FE_hi * (p.FE_max / p.FE_scale), 0.05 * (p.FE_max / p.FE_scale), mid.sum())
    FE0_s[t > t2] = 0.0

    FA0_s = np.full_like(t, FA_lv * (p.FA_max / p.FA_scale))

    x0[idx["FE_s"]] = FE0_s
    x0[idx["FA_s"]] = FA0_s

    # slack initial: assume none delivered
    x0[idx["s"]] = min(p.M_goal, 0.9 * p.M_goal)
    return x0


def clamp_x0(x0: np.ndarray, lbx: np.ndarray, ubx: np.ndarray) -> np.ndarray:
    return np.clip(x0, lbx, ubx)


# ----------------------------
# Phase-out detection
# ----------------------------
def detect_phaseout(FE: np.ndarray, t: np.ndarray, eps: float, m: int) -> Dict[str, Any]:
    if m <= 0:
        return {"phaseout_idx": None, "phaseout_t": None, "phaseout_confirm_window": m}

    active = FE > eps
    if not np.any(active):
        if np.all(FE[:min(m, len(FE))] <= eps):
            return {"phaseout_idx": 0, "phaseout_t": float(t[0]), "phaseout_confirm_window": m}
        return {"phaseout_idx": None, "phaseout_t": None, "phaseout_confirm_window": m}

    last_active_idx = np.where(active)[0][-1]
    confirm_start = last_active_idx + 1
    confirm_end = confirm_start + m
    if confirm_end <= len(FE) and np.all(FE[confirm_start:confirm_end] <= eps):
        return {"phaseout_idx": confirm_start, "phaseout_t": float(t[confirm_start]), "phaseout_confirm_window": m}

    return {"phaseout_idx": None, "phaseout_t": None, "phaseout_confirm_window": m}


# ----------------------------
# Pack parameter vector
# ----------------------------
def pack_par(p: Params, slack_w: float) -> np.ndarray:
    p = finalize_params(p)
    return np.array([
        p.K_max_total, p.alpha, p.r, p.delta_eff,
        1.0 if p.failure_active else 0.0, p.fail_t_start, p.fail_duration, p.fail_severity,
        p.C_E, p.C_A, p.gamma, p.L_R,
        float(slack_w)
    ], dtype=float)


# ----------------------------
# Solve with slack_w continuation + k continuation + multistart + warm-start
# ----------------------------
def solve_haat(p_in: Params, label: str, cache: SolverCache, fast: bool) -> Dict[str, Any]:
    p = finalize_params(p_in)

    N = p.N_fast if fast else p.N_main
    k_schedule = p.k_schedule_fast if fast else p.k_schedule_main
    slack_schedule = p.slack_weight_schedule
    verbose = (not fast) and (not label.startswith(("Heatmap", "MC_", "Tornado")))

    acceptable_status = {"Solve_Succeeded", "Solved_To_Acceptable_Level", "Feasible_Point_Found"}
    solver_counters = {"success_steps": 0, "failed_steps": 0, "retries": 0, "last_status": ""}

    warm_x = None
    warm_lam_x = None
    warm_lam_g = None
    last_pack = None

    if verbose:
        print(f"Solving {label} (N={N}) ...", end=" ")

    # continuation: slack_w outer, k inner
    for slack_w in slack_schedule:
        par_vec = pack_par(p, slack_w)

        for k in k_schedule:
            built = cache.get(p, N=N, k=float(k), fast=fast)
            solver = built["solver"]
            meta = built["meta"]
            lbx = built["lbx"]; ubx = built["ubx"]
            lbg = built["lbg"]; ubg = built["ubg"]

            attempts: List[np.ndarray] = []

            if warm_x is not None:
                attempts.append(clamp_x0(warm_x, lbx, ubx))
                attempts.append(clamp_x0(0.7 * warm_x, lbx, ubx))

            # structured multistart
            attempts.append(clamp_x0(make_structured_x0(p, meta, 0), lbx, ubx))
            attempts.append(clamp_x0(make_structured_x0(p, meta, 1), lbx, ubx))
            attempts.append(clamp_x0(make_structured_x0(p, meta, 2), lbx, ubx))
            attempts.append(np.zeros_like(attempts[0]))

            step_ok = False
            status = "unknown"

            for ai, x_try in enumerate(attempts):
                if ai > 0:
                    solver_counters["retries"] += 1

                kwargs = dict(x0=x_try, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=par_vec)
                if warm_lam_x is not None and warm_lam_g is not None:
                    kwargs["lam_x0"] = warm_lam_x
                    kwargs["lam_g0"] = warm_lam_g

                try:
                    sol = solver(**kwargs)
                    st = solver.stats()
                    status = st.get("return_status", "")
                    succ = bool(st.get("success", False))
                    step_ok = succ or (status in acceptable_status)

                    warm_x = np.array(sol["x"]).flatten()
                    warm_lam_x = np.array(sol["lam_x"]).flatten()
                    warm_lam_g = np.array(sol["lam_g"]).flatten()

                    last_pack = (sol, meta, slack_w, k)
                    if step_ok:
                        break

                except Exception:
                    # if lam warm start destabilizes, drop it
                    warm_lam_x, warm_lam_g = None, None
                    continue

            solver_counters["last_status"] = status

            if step_ok:
                solver_counters["success_steps"] += 1
            else:
                solver_counters["failed_steps"] += 1
                if last_pack is None:
                    if verbose:
                        print("FAILED.")
                    tgrid = np.linspace(0.0, p.T, N + 1)
                    zeros = np.zeros_like(tgrid)
                    return {
                        "label": label, "p": p, "t": tgrid,
                        "A": zeros.copy(), "M": zeros.copy(),
                        "FE": zeros.copy(), "FA": zeros.copy(),
                        "s": float(p.M_goal),
                        "obj_raw": 0.0, "phaseout_t": None,
                        "feasible": False, "solver_status": solver_counters,
                    }

            if verbose:
                print(f"[sw={slack_w:.0e}, k={k}]..", end=" ")

    if verbose:
        print("Done.")

    sol, meta, slack_w_final, k_final = last_pack
    x_opt = np.array(sol["x"]).flatten()
    idx = meta["idx"]

    A = x_opt[idx["A"]]
    M = x_opt[idx["M"]]
    FE = p.FE_scale * x_opt[idx["FE_s"]]
    FA = p.FA_scale * x_opt[idx["FA_s"]]
    s_val = float(x_opt[idx["s"]][0])

    # reporting feasible: slack small enough
    feasible = (s_val <= p.slack_tol_tons)

    res = {
        "label": label, "p": p, "t": meta["tgrid"],
        "A": A, "M": M,
        "FE": FE, "FA": FA,
        "s": s_val,
        "obj_raw": float(sol["f"]) * meta["J_scale"],
        "feasible": bool(feasible),
        "solver_status": solver_counters,
        "slack_w_final": float(slack_w_final),
        "k_final": float(k_final),
    }
    res.update(detect_phaseout(FE, res["t"], p.phaseout_eps, p.phaseout_confirm_n))
    return res


# ----------------------------
# Export table
# ----------------------------
def export_results_table(results: List[Dict[str, Any]]):
    rows = []
    discount_rate = 0.03

    def npv_cost_only(r: Dict[str, Any], rate: float) -> float:
        p = r["p"]
        t = r["t"]; FE = r["FE"]; FA = r["FA"]
        if len(t) < 2:
            return 0.0
        h = float(t[1] - t[0])
        npv = 0.0
        for i in range(len(t) - 1):
            fe_rate = 0.5 * (FE[i] + FE[i + 1])
            fa_rate = 0.5 * (FA[i] + FA[i + 1])
            interval_cash = (p.C_E * fe_rate + p.C_A * fa_rate) * h
            t_mid = 0.5 * (t[i] + t[i + 1])
            npv += interval_cash / ((1 + rate) ** t_mid)
        return npv

    for r in results:
        p = r["p"]
        st = r.get("solver_status", {})
        solver_status = f"ok:{st.get('success_steps',0)} fail:{st.get('failed_steps',0)} retry:{st.get('retries',0)}"

        total_cost = r["obj_raw"]
        npv = npv_cost_only(r, discount_rate)

        rows.append({
            "Scenario": r["label"],
            "Gamma": p.gamma,
            "Resilience": "ON" if p.failure_active else "OFF",
            "M(T) [Mt]": float(r["M"][-1] / 1e6) if len(r["M"]) else 0.0,
            "Slack s(T) [kt]": float(r.get("s", np.nan) / 1e3),
            "Phase-out (yr)": f"{r['phaseout_t']:.1f}" if r["phaseout_t"] is not None else "Never",
            "Feasible": "YES" if r["feasible"] else "NO",
            "Solver": solver_status,
            "Total Obj ($B)": f"{total_cost / 1e9:.2f}",
            "Financial NPV ($B)": f"{npv / 1e9:.2f}",
        })

    df = pd.DataFrame(rows)
    print("\n=== Experiment Summary ===")
    print(df.to_string(index=False))

    df.to_csv(os.path.join(OUTDIR, "final_summary.csv"), index=False)
    with open(os.path.join(OUTDIR, "final_summary.tex"), "w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, float_format="%.2f", caption="Comparison of Scenarios"))


# ----------------------------
# Plots
# ----------------------------
def plot_fig1_optimal_switching(res: Dict[str, Any]):
    t = res["t"]
    p = res["p"]
    par_vec = pack_par(p, slack_w=1.0)  # slack_w irrelevant for capacity
    se_cap_equiv = np.array([capacity_logic_param(float(ti), par_vec[0:8], True) / p.L_R for ti in t])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, res["FE"], linewidth=2.4, label=r"Rocket rate $F_E$")
    ax.plot(t, res["FA"], linewidth=1.8, alpha=0.9, label=r"Transfer rate $F_A$")
    ax.plot(t, se_cap_equiv, linestyle="-.", alpha=0.55, label="SE capacity (eq. launches/yr)")

    if res.get("phaseout_t") is not None:
        pt = res["phaseout_t"]
        ax.axvline(x=pt, color="gray", linestyle=":", alpha=0.65)
        ax.annotate(
            f"Phase-out @ {pt:.1f} yrs after 2050",
            xy=(pt, 0),
            xytext=(min(pt + 2, p.T - 1), max(1.0, np.max(res["FE"]) * 0.6)),
            arrowprops=dict(arrowstyle="->", color="black"),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.85)
        )

    ax.set_title("Fig 1: Optimal Mode Switching Strategy")
    ax.set_xlabel("Years after 2050")
    ax.set_ylabel("Rate (1/yr)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Fig1_OptimalSwitching.png"))
    plt.close(fig)


def plot_fig2_resilience(res_base: Dict[str, Any], res_fail: Dict[str, Any]):
    t = res_base["t"]
    p_f = res_fail["p"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax1.plot(t, res_base["FE"], linestyle="--", alpha=0.6, label="Baseline FE")
    ax1.plot(t, res_fail["FE"], linewidth=2.0, label="Resilience FE")
    ax1.axvspan(p_f.fail_t_start, p_f.fail_t_start + p_f.fail_duration, color="red", alpha=0.12, label="Failure window")
    ax1.set_ylabel("Rocket rate (1/yr)")
    ax1.set_title("Fig 2: System Resilience Analysis")
    ax1.legend()

    ax2.plot(t, res_base["M"] / 1e6, linewidth=2.0, label="Baseline M(t)")
    ax2.plot(t, res_fail["M"] / 1e6, linestyle="--", linewidth=2.0, label="Resilience M(t)")
    ax2.axhline(res_base["p"].M_goal / 1e6, color="gray", linestyle=":", label="Goal (100 Mt)")
    ax2.set_ylabel("Mass (Mt)")
    ax2.set_xlabel("Years after 2050")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Fig2_Resilience.png"))
    plt.close(fig)


# ----------------------------
# Fig4 Heatmap (fast)
# ----------------------------
def run_sensitivity_heatmap(cache: SolverCache, full: bool):
    print("\n>>> Running Sensitivity Heatmap ...")

    if full:
        gammas = [1e-7, 5e-7, 1e-6, 5e-6, 1e-5]
        se_caps = [5e6, 1e7, 1.5e7, 2e7]
    else:
        gammas = [1e-6, 5e-6, 1e-5]
        se_caps = [1e7, 1.5e7]

    Z = np.zeros((len(se_caps), len(gammas)), dtype=float)
    annot = [["" for _ in gammas] for __ in se_caps]

    total = len(se_caps) * len(gammas)
    done = 0
    bad = 0

    for i, Kcap in enumerate(se_caps):
        for j, g in enumerate(gammas):
            done += 1
            print(f"  Heatmap {done}/{total} (K={Kcap/1e6:.1f}Mt, gamma={g:g})")
            p_test = Params(gamma=g, K_max_total=Kcap)
            res = solve_haat(p_test, "Heatmap", cache=cache, fast=True)

            if not res.get("feasible", False):
                bad += 1

            t_out = res["phaseout_t"] if res["phaseout_t"] is not None else p_test.T
            Z[i, j] = t_out
            annot[i][j] = f"{t_out:.1f}" if res["phaseout_t"] is not None else "Never"

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(Z, aspect="auto")
    ax.set_xticks(np.arange(len(gammas)))
    ax.set_yticks(np.arange(len(se_caps)))
    ax.set_xticklabels([f"{x:g}" for x in gammas])
    ax.set_yticklabels([f"{x/1e6:.1f}Mt" for x in se_caps])

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            ax.text(j, i, annot[i][j], ha="center", va="center", fontsize=9)

    ax.set_xlabel("gamma (env penalty)")
    ax.set_ylabel("K_max_total (Mt/year)")
    ax.set_title(f"Fig 4: Rocket Phase-out Year Sensitivity (T={Params().T:.0f} => Never)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Phase-out year (years after 2050)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Fig4_Sensitivity_Heatmap.png"))
    plt.close(fig)

    print(f"[Output] Heatmap -> {OUTDIR}/Fig4_Sensitivity_Heatmap.png (slack-not-tight count: {bad})")


# ----------------------------
# Fig5 Monte Carlo (fast)
# ----------------------------
def run_monte_carlo(cache: SolverCache, n_sims: int, seed: int = 42):
    print(f"\n>>> Running Monte Carlo Robustness n={n_sims} ...")
    rng = np.random.default_rng(seed)

    base_K = 1.5e7
    base_L = 125.0
    results = []
    bad = 0

    for i in range(n_sims):
        print(f"  MC {i+1}/{n_sims}", end="\r")
        rand_k = rng.uniform(0.95, 1.05)
        rand_l = rng.uniform(0.95, 1.05)
        p_mc = Params(gamma=1e-6, K_max_total=base_K * rand_k, L_R=base_L * rand_l)

        res = solve_haat(p_mc, f"MC_{i}", cache=cache, fast=True)
        if res.get("feasible", False):
            t_out = res["phaseout_t"] if res["phaseout_t"] is not None else p_mc.T
            results.append(t_out)
        else:
            bad += 1

    print("\nDone.")

    fig, ax = plt.subplots(figsize=(7, 5))
    if len(results) > 0:
        ax.hist(results, bins=min(10, len(results)))
        mean_v = float(np.mean(results))
        ax.axvline(mean_v, linestyle="--", linewidth=1.5, label=f"Mean: {mean_v:.1f}")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No feasible samples", ha="center", va="center", transform=ax.transAxes)

    ax.set_title(f"Fig 5: Phase-out Robustness (±5% K & L_R)\nN={n_sims}, slack-not-tight={bad}")
    ax.set_xlabel("Phase-out year (years after 2050)")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Fig5_MonteCarlo_Robustness.png"))
    plt.close(fig)

    print(f"[Output] MonteCarlo -> {OUTDIR}/Fig5_MonteCarlo_Robustness.png")


# ----------------------------
# Fig6 Tornado (fast)
# ----------------------------
def run_tornado(cache: SolverCache, base_params: Optional[Params] = None, delta: float = 0.10):
    print("\n>>> Running Tornado Sensitivity (FAST)...")
    p0 = base_params if base_params is not None else Params(gamma=1e-6)

    base = solve_haat(p0, "Tornado_Base", cache=cache, fast=True)
    base_val = base.get("obj_raw", 0.0)

    params_to_test = [
        ("r", "SE Growth r"),
        ("gamma", "Env penalty gamma"),
        ("C_E", "Rocket cost C_E"),
        ("C_A", "Transfer cost C_A"),
        ("K_max_total", "SE capacity Kmax"),
        ("L_R", "Payload L_R"),
        ("w_late", "Late rocket penalty w_late"),
    ]

    impacts = []
    def safe_scale(v, s): return max(float(v) * s, 1e-12)

    for attr, lab in params_to_test:
        v0 = getattr(p0, attr)
        p_low = Params(**{**p0.__dict__, attr: safe_scale(v0, 1.0 - delta)})
        p_high = Params(**{**p0.__dict__, attr: safe_scale(v0, 1.0 + delta)})

        r_low = solve_haat(p_low, "Tornado_low", cache=cache, fast=True)
        r_high = solve_haat(p_high, "Tornado_high", cache=cache, fast=True)

        v_low = r_low.get("obj_raw", base_val)
        v_high = r_high.get("obj_raw", base_val)

        if abs(base_val) > 1e-9:
            low_imp = (v_low - base_val) / base_val * 100.0
            high_imp = (v_high - base_val) / base_val * 100.0
            span = abs(high_imp - low_imp)
        else:
            low_imp = v_low - base_val
            high_imp = v_high - base_val
            span = abs(high_imp - low_imp)

        impacts.append({"label": lab, "low": low_imp, "high": high_imp, "span": span})

    impacts.sort(key=lambda x: x["span"], reverse=True)
    labels = [x["label"] for x in impacts]
    lows = [x["low"] for x in impacts]
    highs = [x["high"] for x in impacts]
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(y, lows, alpha=0.85, label=f"-{int(delta*100)}%")
    ax.barh(y, highs, alpha=0.85, label=f"+{int(delta*100)}%")
    ax.axvline(0, color="k")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("% change in objective")
    ax.set_title("Fig 6: Tornado Sensitivity (Objective)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Fig6_Tornado_Sensitivity.png"))
    plt.close(fig)

    print(f"[Output] Tornado -> {OUTDIR}/Fig6_Tornado_Sensitivity.png")


# ----------------------------
# Main
# ----------------------------
def main():
    full = ("--full" in sys.argv)
    print(">>> Starting MCM Solver (Final Robust Version)")
    print(f"Mode: {'FULL' if full else 'BASE TEST'}")

    cache = SolverCache()

    # Main scenarios: use main resolution in full; fast in base test
    results: List[Dict[str, Any]] = []
    if full:
        results.append(solve_haat(Params(gamma=1e-6), "A_Baseline", cache=cache, fast=False))
        results.append(solve_haat(Params(gamma=5e-6), "B_Policy", cache=cache, fast=False))
        results.append(solve_haat(Params(gamma=1e-6, failure_active=True), "C_Resilience", cache=cache, fast=False))
    else:
        results.append(solve_haat(Params(gamma=1e-6), "A_Baseline", cache=cache, fast=True))
        results.append(solve_haat(Params(gamma=5e-6), "B_Policy", cache=cache, fast=True))
        results.append(solve_haat(Params(gamma=1e-6, failure_active=True), "C_Resilience", cache=cache, fast=True))

    export_results_table(results)

    # Fig1 choose policy if feasible else baseline
    pick = results[1] if results[1].get("feasible", False) else results[0]
    plot_fig1_optimal_switching(pick)
    plot_fig2_resilience(results[0], results[2])

    # Batch experiments
    run_sensitivity_heatmap(cache=cache, full=full)
    run_monte_carlo(cache=cache, n_sims=(30 if full else 10))
    run_tornado(cache=cache, base_params=Params(gamma=1e-6), delta=0.10)

    print(f"\n>>> Done. Outputs in: {OUTDIR}")
    print("Feasibility diagnostics (slack-tight):")
    for r in results:
        print(f" - {r['label']}: feasible={r['feasible']}  M(T)={r['M'][-1]/1e6:.3f}Mt  slack={r.get('s',np.nan)/1e3:.2f}kt  phaseout={r.get('phaseout_t',None)}")


if __name__ == "__main__":
    main()