# -*- coding: utf-8 -*-
"""
MCM 2026 Problem B: Self-Consistent Results Script (Convergent v6 - Deliverable)
--------------------------------------------------------------------------------
Deliverable upgrades added on top of v5:
  (A) Fix s_over ambiguity: if enable_terminal_over_slack=False, s_over is FIXED to 0 (lbx=ubx=0)
      so it never shows meaningless values in outputs.
  (B) Add terminal-band interpretability fields:
      M_goal, M_band_high, M_over, M_under
  (C) Output positive-sense consistency flags:
      phaseout_consistent, overbuild_within_tol, late_cap_satisfied
      (still keep original flags for audit)
  (D) Save full trajectories per scenario to CSV:
      trajectories_<Scenario>.csv with t, A, M, FE, FA, Phi_se, FE_cap_eq
  (E) Add phaseout_t_alt (presentation-only) to avoid "Never" under strict phaseout definition.

Dependencies: casadi, numpy, pandas, matplotlib

Run:
  python consistency_self_consistent_convergent_v6_deliverable.py
"""

import sys
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List, Optional

import numpy as np

print(f"[Env] Python: {sys.executable}")

try:
    import casadi as ca
    import pandas as pd
    import matplotlib.pyplot as plt
except ImportError as e:
    print("\n[CRITICAL ERROR] Missing libraries.")
    print(f"Error details: {e}")
    print("Please run: pip install casadi pandas matplotlib")
    sys.exit(1)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
})

OUTDIR = "mcm_final_outputs"
os.makedirs(OUTDIR, exist_ok=True)
warnings.filterwarnings("ignore")


# ============================================================
# 0) Embedded minimal core (no external dependency)
# ============================================================
@dataclass(frozen=True)
class Scales:
    mass: float = 1e6
    launch: float = 1e3
    money: float = 1e9


def sigma_k(z, k):
    return 1.0 / (1.0 + ca.exp(-ca.fmin(ca.fmax(k * z, -50), 50)))


def softplus_k(z, k):
    kz = ca.fmin(ca.fmax(k * z, -50), 50)
    return (1.0 / k) * ca.log(1.0 + ca.exp(kz))


def sigma_num(z, k):
    kz = np.clip(k * np.asarray(z, dtype=float), -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-kz))


def window_health(t, t0, dur, severity, k_step: float = 10.0, is_numeric: bool = False):
    if is_numeric:
        w = sigma_num(t - t0, k_step) - sigma_num(t - (t0 + dur), k_step)
        return 1.0 - severity * w
    w = sigma_k(t - t0, k_step) - sigma_k(t - (t0 + dur), k_step)
    return 1.0 - severity * w


def capacity_logistic(t,
                      K_max,
                      alpha: float,
                      r: float,
                      delta_eff: float,
                      failure_active: bool = False,
                      fail_t_start: float = 0.0,
                      fail_duration: float = 0.0,
                      fail_severity: float = 0.0,
                      k_step: float = 10.0,
                      is_numeric: bool = False):
    exp_fn = np.exp if is_numeric else ca.exp
    a = max(min(alpha, 0.999999), 1e-6)
    base = K_max / (1.0 + ((1.0 - a) / a) * exp_fn(-r * t))
    base = base * delta_eff
    if failure_active:
        return base * window_health(t, fail_t_start, fail_duration, fail_severity, k_step=k_step, is_numeric=is_numeric)
    return base


# ============================================================
# 1) Utilities
# ============================================================
def dm_to_1d(x) -> np.ndarray:
    if isinstance(x, (ca.DM, ca.SX, ca.MX)):
        return np.array(x.full()).reshape(-1)
    return np.array(x).reshape(-1)


def _clamp(x0: np.ndarray, lbx: np.ndarray, ubx: np.ndarray) -> np.ndarray:
    return np.clip(x0, lbx, ubx)


def _ipopt_ok(stats: Dict[str, Any]) -> bool:
    if stats is None:
        return False
    if bool(stats.get("success", False)):
        return True
    rs = str(stats.get("return_status", ""))
    return rs in {"Solved_To_Acceptable_Level", "Solve_Succeeded"}


def _finite(x, default=0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _constraint_max_violation(g_val: np.ndarray, lbg: np.ndarray, ubg: np.ndarray) -> float:
    if g_val is None or len(g_val) == 0:
        return float("inf")
    g_val = np.asarray(g_val).reshape(-1)
    lbg = np.asarray(lbg).reshape(-1)
    ubg = np.asarray(ubg).reshape(-1)

    v_low = np.where(np.isfinite(lbg), np.maximum(lbg - g_val, 0.0), 0.0)
    v_high = np.where(np.isfinite(ubg), np.maximum(g_val - ubg, 0.0), 0.0)

    return float(np.max(np.maximum(v_low, v_high)))


def _scaled_params(p: "Params") -> Dict[str, float]:
    sc = Scales()
    return {
        "mass": sc.mass,
        "launch": sc.launch,
        "money": sc.money,

        "M_goal": p.M_goal / sc.mass,
        "A_max": p.A_max / sc.mass,
        "L_R": p.L_R / sc.mass,
        "FE_max": p.FE_max / sc.launch,
        "FA_max": p.FA_max / sc.launch,
        "K_max_total": p.K_max_total / sc.mass,
        "W_req": p.W_req / sc.mass,

        "C_E": (p.C_E * sc.launch) / sc.money,
        "C_A": (p.C_A * sc.launch) / sc.money,
        "e_E": p.e_E * sc.launch,

        "phaseout_eps": p.phaseout_eps / sc.launch,
        "FE_late_cap_target": p.FE_late_cap_target / sc.launch,
    }


# ==========================================
# 2) Parameters
# ==========================================
@dataclass
class Params:
    # Mission
    M_goal: float = 1e8
    P_pop: float = 1e5
    w_ls: float = 1.5
    W_req: float = 0.0  # derived

    # Space elevator capacity
    K_max_total: float = 1.5e7
    alpha: float = 0.15
    r: float = 0.18
    delta_eff: float = 0.974

    # Failure
    failure_active: bool = False
    fail_t_start: float = 20.0
    fail_duration: float = 3.0
    fail_severity: float = 0.90

    # Controls
    L_R: float = 125.0
    FE_max: float = 40000.0
    FA_max: float = 200000.0
    A_max: float = 5e5

    # Cost / policy
    C_E: float = 1.0e8
    C_A: float = 1.0e7
    e_E: float = 1275.0
    gamma: float = 0.08

    # Discretization
    T: float = 50.0
    N: int = 250

    # Homotopy schedule (kept)
    k_schedule: Tuple[float, ...] = (1.0, 5.0, 20.0)

    # Smoothness
    w_smooth: float = 50.0

    # Phase-out detector
    phaseout_eps: float = 10.0
    phaseout_m: int = 4
    eps_relax: float = 1.5

    # Terminal band
    enable_terminal_band: bool = True
    terminal_over_tol: float = 0.01
    # IMPORTANT: default hard upper band (deliverable)
    enable_terminal_over_slack: bool = False

    # Late cap
    enable_late_cap: bool = True
    t_late_frac: float = 0.70
    FE_late_cap_target: float = -1.0

    # Slack weights (dimensionless multipliers)
    w_goal_slack: float = 1e4
    w_late_slack: float = 1e3
    w_over_slack: float = 5e4

    # Smooth positive-part epsilon (gating fix)
    eps_pos: float = 1e-8

    # Numerical feasibility tolerance
    feas_abs_tol: float = 1e-4
    feas_rel_tol: float = 2e-7

    # IPOPT
    ipopt_max_iter: int = 12000
    ipopt_tol: float = 1e-7
    ipopt_print_level: int = 0


def finalize_params(p: Params) -> Params:
    p.W_req = p.P_pop * p.w_ls * 365.0 / 1000.0
    if p.FE_late_cap_target < 0:
        p.FE_late_cap_target = float(p.phaseout_eps * p.eps_relax)
    return p


# ==========================================
# 3) NLP builder
# ==========================================
def pos_eps(z, eps, is_numeric: bool = False):
    """Smooth positive part: 0.5*(z + sqrt(z^2 + eps^2))"""
    if is_numeric:
        z = np.asarray(z, dtype=float)
        return 0.5 * (z + np.sqrt(z * z + eps * eps))
    return 0.5 * (z + ca.sqrt(z * z + eps * eps))


def build_hs_nlp(p: Params, k: float, late_cap: Optional[float]):
    sp = _scaled_params(p)
    N = int(p.N)
    T = float(p.T)
    h = T / N
    tgrid = np.linspace(0.0, T, N + 1)

    # decision variables
    A = ca.SX.sym("A", N + 1)
    M = ca.SX.sym("M", N + 1)
    FE = ca.SX.sym("FE", N + 1)
    FA = ca.SX.sym("FA", N + 1)
    s_goal = ca.SX.sym("s_goal", 1)
    s_late = ca.SX.sym("s_late", 1)
    s_over = ca.SX.sym("s_over", 1)  # kept for indexing stability; fixed to 0 if not used
    x = ca.vertcat(A, M, FE, FA, s_goal, s_late, s_over)

    idx = {
        "A": slice(0, N + 1),
        "M": slice(N + 1, 2 * N + 2),
        "FE": slice(2 * N + 2, 3 * N + 3),
        "FA": slice(3 * N + 3, 4 * N + 4),
        "s_goal": slice(4 * N + 4, 4 * N + 5),
        "s_late": slice(4 * N + 5, 4 * N + 6),
        "s_over": slice(4 * N + 6, 4 * N + 7),
    }

    # bounds
    lbx = np.concatenate([
        np.zeros(4 * (N + 1)),
        np.array([0.0, 0.0, 0.0]),
    ])
    ubx = np.concatenate([
        np.full(N + 1, sp["A_max"]),
        np.full(N + 1, sp["M_goal"] * 3.0),
        np.full(N + 1, sp["FE_max"]),
        np.full(N + 1, sp["FA_max"]),
        np.array([sp["M_goal"] * 3.0, sp["FE_max"], sp["M_goal"] * 3.0]),
    ])

    # Deliverable patch: if upper-band slack disabled, FIX s_over=0 to avoid meaningless output
    if not p.enable_terminal_over_slack:
        lbx[idx["s_over"]] = 0.0
        ubx[idx["s_over"]] = 0.0

    g, lbg, ubg = [], [], []

    # initial conditions
    g += [A[0], M[0]]
    lbg += [0.0, 0.0]
    ubg += [0.0, 0.0]

    # dynamics with gating fix
    def f(Ax, Mx, FEx, FAx, t):
        Phi_se = capacity_logistic(
            t,
            sp["K_max_total"],
            p.alpha,
            p.r,
            p.delta_eff,
            failure_active=p.failure_active,
            fail_t_start=p.fail_t_start,
            fail_duration=p.fail_duration,
            fail_severity=p.fail_severity,
            is_numeric=False,
        )

        eps = float(p.eps_pos)
        remain = pos_eps(sp["A_max"] - Ax, eps) / (sp["A_max"] + eps)  # ->0 at A=Amax
        avail = Ax / (Ax + eps)                                       # ->0 at A=0

        inflow = Phi_se * remain
        outflow = sp["L_R"] * (FAx * sp["launch"]) * avail
        dA = inflow - outflow

        Phi_arrive = sp["L_R"] * ((FEx + FAx) * sp["launch"])
        dM = Phi_arrive  # FE,FA>=0 => Phi_arrive>=0
        return dA, dM

    # Hermite–Simpson
    for i in range(N):
        ti = tgrid[i]
        dA_i, dM_i = f(A[i], M[i], FE[i], FA[i], ti)
        dA_ip1, dM_ip1 = f(A[i + 1], M[i + 1], FE[i + 1], FA[i + 1], ti + h)

        A_m = 0.5 * (A[i] + A[i + 1]) + (h / 8) * (dA_i - dA_ip1)
        M_m = 0.5 * (M[i] + M[i + 1]) + (h / 8) * (dM_i - dM_ip1)
        FEm = 0.5 * (FE[i] + FE[i + 1])
        FAm = 0.5 * (FA[i] + FA[i + 1])

        dA_m, dM_m = f(A_m, M_m, FEm, FAm, ti + 0.5 * h)

        g += [A[i + 1] - A[i] - (h / 6) * (dA_i + 4 * dA_m + dA_ip1)]
        g += [M[i + 1] - M[i] - (h / 6) * (dM_i + 4 * dM_m + dM_ip1)]
        lbg += [0.0, 0.0]
        ubg += [0.0, 0.0]

        # Monotonicity
        g += [M[i + 1] - M[i]]
        lbg += [0.0]
        ubg += [np.inf]

    # terminal lower bound
    g += [M[-1] + s_goal[0] - sp["M_goal"]]
    lbg += [0.0]
    ubg += [np.inf]

    # terminal upper band
    if p.enable_terminal_band:
        if p.enable_terminal_over_slack:
            g += [M[-1] - (1.0 + p.terminal_over_tol) * sp["M_goal"] - s_over[0]]
            lbg += [-np.inf]
            ubg += [0.0]
        else:
            g += [M[-1] - (1.0 + p.terminal_over_tol) * sp["M_goal"]]
            lbg += [-np.inf]
            ubg += [0.0]

    # late-cap
    t_late = p.t_late_frac * p.T
    cap_used = np.nan
    if p.enable_late_cap and late_cap is not None:
        cap_used = float(min(late_cap, sp["FE_max"]))
        for i in range(N + 1):
            if tgrid[i] >= t_late:
                g += [FE[i] - cap_used - s_late[0]]
                lbg += [-np.inf]
                ubg += [0.0]

    # Objective scaling
    cost_scale = (sp["C_E"] * sp["FE_max"] + sp["C_A"] * sp["FA_max"]) * p.T
    env_scale = p.gamma * (sp["e_E"] * sp["FE_max"]) ** 2 * p.T
    smooth_scale = p.w_smooth * ((sp["FE_max"] / max(p.T, 1e-9)) ** 2 + (sp["FA_max"] / max(p.T, 1e-9)) ** 2) * p.T
    J_scale = float(max(1.0, _finite(cost_scale + env_scale + smooth_scale, 1.0)))

    J = 0
    for i in range(N):
        FEavg = 0.5 * (FE[i] + FE[i + 1])
        FAavg = 0.5 * (FA[i] + FA[i + 1])

        cost = sp["C_E"] * FEavg + sp["C_A"] * FAavg
        env = p.gamma * (sp["e_E"] * FEavg) ** 2

        dFE_dt = (FE[i + 1] - FE[i]) / h
        dFA_dt = (FA[i + 1] - FA[i]) / h
        smooth = p.w_smooth * (dFE_dt ** 2 + dFA_dt ** 2)

        J += h * (cost + env + smooth)

    s_goal_frac = s_goal[0] / max(sp["M_goal"], 1e-12)
    s_late_frac = s_late[0] / max(sp["FE_max"], 1e-12)
    s_over_frac = s_over[0] / max(sp["M_goal"], 1e-12)

    # penalize slacks
    J += (p.w_goal_slack * J_scale) * (s_goal_frac ** 2)
    J += (p.w_late_slack * J_scale) * (s_late_frac ** 2)
    if p.enable_terminal_over_slack:
        J += (p.w_over_slack * J_scale) * (s_over_frac ** 2)

    nlp = {"x": x, "f": J / J_scale, "g": ca.vertcat(*g)}
    meta = {
        "tgrid": tgrid, "h": h, "N": N, "k": k,
        "idx": idx, "J_scale": J_scale,
        "t_late": float(t_late),
        "late_cap": float(cap_used) if np.isfinite(cap_used) else np.nan,
        "scales": sp,
        "ng": len(lbg),
        "nx": int(nlp["x"].shape[0]),
    }
    return nlp, lbx, ubx, np.array(lbg, dtype=float), np.array(ubg, dtype=float), meta


# ==========================================
# 4) Solver
# ==========================================
def _make_solver(nlp, p: Params):
    opts = {
        "ipopt": {
            "print_level": int(p.ipopt_print_level),
            "max_iter": int(p.ipopt_max_iter),
            "tol": float(p.ipopt_tol),
            "nlp_scaling_method": "gradient-based",
            "mu_strategy": "adaptive",
            "acceptable_tol": 1e-5,
            "acceptable_iter": 25,
            "linear_solver": "mumps",
            "warm_start_init_point": "yes",
            # deliverable stability (safe additions)
            "bound_relax_factor": 0.0,
        },
        "print_time": False,
        "error_on_fail": False,
    }
    return ca.nlpsol("solver", "ipopt", nlp, opts)


def _detect_phaseout_aligned(FE: np.ndarray, t: np.ndarray, eps: float, m: int, eps_relax: float) -> Dict[str, Any]:
    """
    Original stricter detector kept for audit:
    - find a block of m points with FE>eps (activity)
    - then confirm after that point FE stays <= eps*eps_relax (phase-out)
    """
    if len(FE) == 0:
        return {"phaseout_idx": None, "phaseout_t": None, "phaseout_confirm_window": m}

    above = FE > eps
    last_end = None
    for i in range(len(FE) - m + 1):
        if np.all(above[i:i + m]):
            last_end = i + m - 1

    if last_end is None:
        return {"phaseout_idx": None, "phaseout_t": None, "phaseout_confirm_window": m}

    cand = last_end + 1
    if cand >= len(FE):
        return {"phaseout_idx": None, "phaseout_t": None, "phaseout_confirm_window": m}

    thresh = eps * eps_relax
    if np.all(FE[cand:] <= thresh):
        return {"phaseout_idx": cand, "phaseout_t": float(t[cand]), "phaseout_confirm_window": m}

    return {"phaseout_idx": None, "phaseout_t": None, "phaseout_confirm_window": m}


def _detect_phaseout_alt(FE: np.ndarray, t: np.ndarray, eps: float, m: int, eps_relax: float) -> Optional[float]:
    """
    Presentation-only: a more intuitive phase-out time.
    Definition:
      Let j = last index where FE > eps (last "active" moment).
      Find earliest i >= j+1 such that FE[i:i+m] <= eps*eps_relax.
      Return t[i] if found else None.
    """
    if len(FE) < m:
        return None
    thresh = eps * eps_relax

    active_idx = np.where(FE > eps)[0]
    j = int(active_idx[-1]) if active_idx.size > 0 else -1
    start = max(0, j + 1)

    for i in range(start, len(FE) - m + 1):
        if np.all(FE[i:i + m] <= thresh):
            return float(t[i])
    return None


def _maybe_drop_duals(lam_x0, lam_g0, nx: int, ng: int):
    lx = lam_x0
    lg = lam_g0
    if lx is not None:
        lx = np.asarray(lx).reshape(-1)
        if lx.size != nx:
            lx = None
    if lg is not None:
        lg = np.asarray(lg).reshape(-1)
        if lg.size != ng:
            lg = None
    return lx, lg


def _solve_one(p: Params, label: str, late_cap: Optional[float],
               warm: Optional[Dict[str, np.ndarray]]) -> Tuple[Dict[str, Any], Optional[Dict[str, np.ndarray]]]:
    last_pack = None
    warm_pack = warm

    for k in p.k_schedule:
        nlp, lbx, ubx, lbg, ubg, meta = build_hs_nlp(p, k, late_cap=late_cap)
        solver = _make_solver(nlp, p)

        if warm_pack is None:
            x0 = np.zeros(meta["nx"])
            idx = meta["idx"]
            sp = meta["scales"]
            x0[idx["FE"]] = min(sp["FE_max"] * 0.15, sp["FE_max"])
            x0[idx["FA"]] = min(sp["FA_max"] * 0.08, sp["FA_max"])
            x0[idx["s_goal"]] = 0.0
            x0[idx["s_late"]] = 0.0
            x0[idx["s_over"]] = 0.0
            lam_x0 = None
            lam_g0 = None
        else:
            x0 = warm_pack["x"].copy()
            lam_x0 = warm_pack.get("lam_x", None)
            lam_g0 = warm_pack.get("lam_g", None)

        x0 = _clamp(x0, lbx, ubx)
        lam_x0, lam_g0 = _maybe_drop_duals(lam_x0, lam_g0, nx=meta["nx"], ng=meta["ng"])

        kwargs = dict(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        if lam_x0 is not None and lam_g0 is not None:
            kwargs["lam_x0"] = lam_x0
            kwargs["lam_g0"] = lam_g0

        try:
            sol = solver(**kwargs)
        except Exception:
            try:
                x1 = _clamp(0.7 * x0, lbx, ubx)
                sol = solver(x0=x1, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            except Exception:
                return ({
                    "label": label,
                    "p": p,
                    "feasible": False,
                    "ipopt_ok": False,
                    "solver_status": "crash",
                    "late_cap": late_cap if late_cap is not None else "",
                }, warm_pack)

        stats = solver.stats()
        ipopt_ok = _ipopt_ok(stats)

        x_sol = dm_to_1d(sol.get("x", np.zeros(meta["nx"])))
        lam_x_sol = dm_to_1d(sol.get("lam_x", np.zeros(meta["nx"])))
        g_sol = dm_to_1d(sol.get("g", np.array([])))
        lam_g_sol = dm_to_1d(sol.get("lam_g", np.zeros(meta["ng"]))) if meta["ng"] > 0 else np.array([])

        warm_pack = {"x": x_sol, "lam_x": lam_x_sol, "lam_g": lam_g_sol}
        last_pack = (sol, meta, stats, warm_pack, ipopt_ok, lbg, ubg, g_sol)

    if last_pack is None:
        return ({
            "label": label,
            "p": p,
            "feasible": False,
            "ipopt_ok": False,
            "solver_status": "no_solution",
            "late_cap": late_cap if late_cap is not None else "",
        }, warm_pack)

    sol, meta, stats, warm_pack, ipopt_ok, lbg, ubg, g_sol = last_pack
    idx = meta["idx"]
    t = meta["tgrid"]
    x = warm_pack["x"]
    sp = meta["scales"]

    A = x[idx["A"]]
    M = x[idx["M"]]
    FE = x[idx["FE"]]
    FA = x[idx["FA"]]
    s_goal = float(x[idx["s_goal"]][0])
    s_late = float(x[idx["s_late"]][0])
    s_over = float(x[idx["s_over"]][0])  # will be 0 if slack disabled

    term_slack_low = float(M[-1] + s_goal - sp["M_goal"])
    if p.enable_terminal_band:
        if p.enable_terminal_over_slack:
            term_slack_high = float((1.0 + p.terminal_over_tol) * sp["M_goal"] + s_over - M[-1])
        else:
            term_slack_high = float((1.0 + p.terminal_over_tol) * sp["M_goal"] - M[-1])
    else:
        term_slack_high = np.nan

    t_late = float(meta["t_late"])
    cap = _finite(meta.get("late_cap", np.nan), np.nan)
    late_mask = t >= t_late
    if p.enable_late_cap and late_cap is not None and np.any(late_mask) and np.isfinite(cap):
        late_cap_violation = float(np.max(FE[late_mask] - cap - s_late))
    else:
        late_cap_violation = 0.0

    max_constr_viol = _constraint_max_violation(g_sol, lbg, ubg)

    tol_M = p.feas_abs_tol + p.feas_rel_tol * sp["M_goal"]
    ok_low = term_slack_low >= -tol_M
    ok_high = (np.isnan(term_slack_high) or term_slack_high >= -tol_M)
    ok_cap = (late_cap_violation <= 1e-6 + 1e-9 * max(1.0, abs(_finite(cap, 1.0))))
    feasible = bool(ipopt_ok and ok_low and ok_high and ok_cap)

    ph = _detect_phaseout_aligned(FE, t, sp["phaseout_eps"], p.phaseout_m, p.eps_relax)

    res = {
        "label": label,
        "p": p,
        "solver_status": str(stats.get("return_status", "")),
        "ipopt_ok": bool(ipopt_ok),
        "feasible": bool(feasible),

        "t": t, "A": A, "M": M, "FE": FE, "FA": FA,
        "obj_raw": float(sol.get("f", 0.0)) * float(meta["J_scale"]),
        "t_late": t_late,
        "late_cap": cap,
        "late_cap_violation": late_cap_violation,

        "term_slack_low": term_slack_low,
        "term_slack_high": term_slack_high,
        "s_goal": s_goal,
        "s_late": s_late,
        "s_over": s_over,

        "max_constr_viol": max_constr_viol,
        **ph
    }
    return res, warm_pack


def _cap_schedule_from_solution(FE: np.ndarray, t: np.ndarray, t_late: float, target: float,
                                failure_active: bool, start_cap: Optional[float] = None) -> List[float]:
    target = float(target)
    if target <= 0 or not np.isfinite(target):
        return [target]

    late_mask = t >= float(t_late)
    late_max = float(np.max(FE[late_mask])) if np.any(late_mask) else float(np.max(FE))
    start = max(target, late_max)
    if start_cap is not None and np.isfinite(start_cap):
        start = max(start, float(start_cap))

    if start <= target * 1.001:
        return [target]

    ratio = 0.65 if failure_active else 0.5
    caps = [start]
    while caps[-1] > target * 1.05:
        next_cap = max(target, caps[-1] * ratio)
        if next_cap >= caps[-1] - 1e-12:
            break
        caps.append(next_cap)
        if len(caps) > 12:
            break

    if abs(caps[-1] - target) > 1e-12:
        caps.append(target)

    out = []
    for c in caps:
        c = float(c)
        if len(out) == 0 or c < out[-1] - 1e-12:
            out.append(c)
    out[-1] = target
    return out


def solve_with_cap_continuation(p: Params, label: str) -> Dict[str, Any]:
    p = finalize_params(p)
    sp = _scaled_params(p)

    if not p.enable_late_cap:
        res, _ = _solve_one(p, label, late_cap=None, warm=None)
        return res

    target = float(sp["FE_late_cap_target"])

    # seed with cap = FE_max (scaled), consistent units
    seed_cap = float(sp["FE_max"])
    warm = None
    seed_res, warm = _solve_one(p, label, late_cap=seed_cap, warm=warm)

    if not seed_res.get("ipopt_ok", False):
        seed_res["feasible"] = False
        seed_res["solver_status"] = f"seed_failed:{seed_res.get('solver_status', '')}"
        return seed_res

    schedule = _cap_schedule_from_solution(
        FE=np.array(seed_res["FE"]),
        t=np.array(seed_res["t"]),
        t_late=float(seed_res["t_late"]),
        target=target,
        failure_active=bool(p.failure_active),
        start_cap=seed_cap,
    )
    if len(schedule) >= 1 and abs(schedule[0] - seed_cap) <= 1e-12:
        schedule = schedule[1:]

    best = seed_res
    for cap in schedule:
        res, warm = _solve_one(p, label, late_cap=float(cap), warm=warm)
        if res.get("ipopt_ok", False):
            best = res
        else:
            best = dict(best)
            best["solver_status"] = f"cap_failed_at_{cap}:{res.get('solver_status', '')}"
            best["ipopt_ok"] = False
            best["feasible"] = False
            best["failed_cap"] = float(cap)
            best["failed_max_constr_viol"] = float(res.get("max_constr_viol", np.nan))
            return best

    return best


# ==========================================
# 5) Audit + reporting + plots
# ==========================================
def _audit_consistency(res: Dict[str, Any], overbuild_tol: float = 0.05) -> Dict[str, Any]:
    p: Params = res["p"]
    sp = _scaled_params(p)

    if "M" not in res or "FE" not in res or "t" not in res:
        return {
            "phaseout_inconsistent": False,
            "overbuild_excessive": False,
            "late_spike": False,
            "M_T": np.nan,
            "phaseout_t": None,
            "phaseout_t_alt": None,
            "late_window_start": np.nan,
            "late_cap": float(res.get("late_cap", np.nan)),
            "late_max_FE": np.nan,
            "late_median_FE": np.nan,
            "late_cap_violation": float(res.get("late_cap_violation", np.nan)),
            "term_slack_low": float(res.get("term_slack_low", np.nan)),
            "term_slack_high": float(res.get("term_slack_high", np.nan)),
            "ipopt_ok": False,
            "monotone_min_dM": np.nan,
            "max_constr_viol": float(res.get("max_constr_viol", np.inf)),
        }

    t = np.array(res["t"])
    FE = np.array(res["FE"])
    M = np.array(res["M"])

    eps = float(sp["phaseout_eps"])
    eps_relax = float(p.eps_relax)
    mT = float(M[-1]) if len(M) else np.nan

    # overbuild excessive (5% by default)
    overbuild_excessive = bool(np.isfinite(mT) and (mT > (1.0 + overbuild_tol) * sp["M_goal"]))

    # phaseout inconsistency check only if strict phaseout_t exists
    pt = res.get("phaseout_t", None)
    phaseout_inconsistent = False
    if isinstance(pt, (float, int)):
        mask = t >= float(pt)
        if np.any(FE[mask] > eps * eps_relax):
            phaseout_inconsistent = True

    # alternative phaseout time for presentation
    pt_alt = _detect_phaseout_alt(FE, t, eps=eps, m=int(p.phaseout_m), eps_relax=eps_relax)

    # late-cap spike
    t_late = float(res.get("t_late", p.t_late_frac * p.T))
    late_mask = t >= t_late
    late_max = float(np.max(FE[late_mask])) if np.any(late_mask) else float(np.max(FE))
    late_med = float(np.median(FE[late_mask])) if np.any(late_mask) else float(np.median(FE))

    cap = _finite(res.get("late_cap", np.nan), np.nan)
    cap_tol = 1e-6 + 1e-9 * max(1.0, abs(cap if np.isfinite(cap) else 1.0))
    late_spike = bool(np.isfinite(cap) and (late_max > cap + float(res.get("s_late", 0.0)) + cap_tol))

    # monotone check (min diff; should be >=0)
    monotone_violation = float(np.min(np.diff(M))) if len(M) > 1 else np.nan

    return {
        "phaseout_inconsistent": phaseout_inconsistent,
        "overbuild_excessive": overbuild_excessive,
        "late_spike": late_spike,
        "M_T": mT,
        "phaseout_t": pt if pt is not None else "",
        "phaseout_t_alt": pt_alt if pt_alt is not None else "",
        "late_window_start": t_late,
        "late_cap": cap,
        "late_max_FE": late_max,
        "late_median_FE": late_med,
        "late_cap_violation": float(res.get("late_cap_violation", np.nan)),
        "term_slack_low": float(res.get("term_slack_low", np.nan)),
        "term_slack_high": float(res.get("term_slack_high", np.nan)),
        "ipopt_ok": bool(res.get("ipopt_ok", False)),
        "monotone_min_dM": monotone_violation,
        "max_constr_viol": float(res.get("max_constr_viol", np.nan)),
    }


def export_results_table(results: List[Dict[str, Any]]):
    sc = Scales()
    rows = []

    for r in results:
        p: Params = r["p"]
        sp = _scaled_params(p)

        # terminal band in scaled mass units (Mt)
        M_goal = float(sp["M_goal"])
        M_band_high = float((1.0 + p.terminal_over_tol) * sp["M_goal"]) if p.enable_terminal_band else np.nan

        pt = r.get("phaseout_t", None)
        pt_show = f"{pt:.1f}" if isinstance(pt, (float, int)) else "Never"

        m_traj = r.get("M", [])
        m_final = float(np.array(m_traj)[-1]) if len(m_traj) > 0 else np.nan

        M_over = float(max(0.0, m_final - M_band_high)) if np.isfinite(M_band_high) and np.isfinite(m_final) else np.nan
        M_under = float(max(0.0, M_goal - m_final)) if np.isfinite(M_goal) and np.isfinite(m_final) else np.nan

        rows.append({
            "Scenario": r.get("label", ""),
            "Gamma": p.gamma,
            "Resilience": "ON" if p.failure_active else "OFF",
            "N": p.N,

            "M_goal(Mt)": M_goal,
            "M_band_high(Mt)": M_band_high,
            "M(T) [Mt]": m_final,
            "M_over(Mt)": M_over,
            "M_under(Mt)": M_under,

            "Phase-out (yr)": pt_show,
            "LateCap": float(r.get("late_cap", np.nan)) * sc.launch,
            "LateCapViol": float(r.get("late_cap_violation", np.nan)) * sc.launch,

            "Solver": r.get("solver_status", ""),
            "IPOPT_ok": "YES" if r.get("ipopt_ok", False) else "NO",
            "Feasible": "YES" if r.get("feasible", False) else "NO",

            "Slack_low": float(r.get("term_slack_low", np.nan)),
            "Slack_high": float(r.get("term_slack_high", np.nan)),

            # s_over is fixed to 0 when slack disabled; safe to output
            "s_over": float(r.get("s_over", np.nan)),

            "MaxViol": float(r.get("max_constr_viol", np.nan)),
            "Obj(scaled)": float(r.get("obj_raw", 0.0)),
        })

    df = pd.DataFrame(rows)
    print("\n=== Experiment Summary ===")
    print(df.to_string(index=False))

    df.to_csv(os.path.join(OUTDIR, "final_summary.csv"), index=False)
    with open(os.path.join(OUTDIR, "final_summary.tex"), "w") as f:
        f.write(df.to_latex(index=False, float_format="%.6g",
                            caption="Comparison of Scenarios (Convergent v6 - Deliverable)"))


def export_consistency_report(results: List[Dict[str, Any]]):
    sc = Scales()
    rows = []

    for r in results:
        audit = _audit_consistency(r, overbuild_tol=0.05)

        # Positive-sense flags (deliverable)
        phaseout_consistent = (not bool(audit["phaseout_inconsistent"]))
        overbuild_within_tol = (not bool(audit["overbuild_excessive"]))
        late_cap_satisfied = (not bool(audit["late_spike"]))

        rows.append({
            "Scenario": r.get("label", ""),

            # Positive flags
            "phaseout_consistent": phaseout_consistent,
            "overbuild_within_tol": overbuild_within_tol,
            "late_cap_satisfied": late_cap_satisfied,

            # Original flags kept for audit traceability
            "phaseout_inconsistent": audit["phaseout_inconsistent"],
            "overbuild_excessive": audit["overbuild_excessive"],
            "late_spike": audit["late_spike"],

            "Feasible": "YES" if r.get("feasible", False) else "NO",
            "IPOPT_ok": "YES" if audit["ipopt_ok"] else "NO",

            "phaseout_t": audit["phaseout_t"],
            "phaseout_t_alt": audit["phaseout_t_alt"],

            "M_T": audit["M_T"],
            "late_window_start": audit["late_window_start"],
            "late_cap": audit["late_cap"] * sc.launch,
            "late_max_FE": audit["late_max_FE"] * sc.launch,
            "late_median_FE": audit["late_median_FE"] * sc.launch,
            "late_cap_violation": audit["late_cap_violation"] * sc.launch,

            "term_slack_low": audit["term_slack_low"],
            "term_slack_high": audit["term_slack_high"],
            "s_over": float(r.get("s_over", np.nan)),

            "monotone_min_dM": audit["monotone_min_dM"],
            "max_constr_viol": audit["max_constr_viol"],

            "failed_cap": float(r.get("failed_cap", np.nan)),
            "failed_max_constr_viol": float(r.get("failed_max_constr_viol", np.nan)),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "consistency_report.csv"), index=False)
    print(f"\n[Output] Consistency report -> {OUTDIR}/consistency_report.csv")
    print(df.to_string(index=False))

    # small note file to avoid reviewer confusion
    note = (
        "NOTE: For boolean flags in consistency_report.csv:\n"
        "  - phaseout_consistent / overbuild_within_tol / late_cap_satisfied:\n"
        "    True means 'no violation detected'.\n"
        "  - phaseout_inconsistent / overbuild_excessive / late_spike:\n"
        "    True means 'violation detected'.\n"
    )
    with open(os.path.join(OUTDIR, "consistency_notes.txt"), "w", encoding="utf-8") as f:
        f.write(note)


def export_trajectories(results: List[Dict[str, Any]]):
    """Save full trajectories per scenario for audit and fast plotting."""
    sc = Scales()

    for r in results:
        if "t" not in r or "A" not in r or "M" not in r or "FE" not in r or "FA" not in r:
            continue

        p: Params = r["p"]
        sp = _scaled_params(p)

        t = np.array(r["t"], dtype=float)
        A = np.array(r["A"], dtype=float)
        M = np.array(r["M"], dtype=float)
        FE = np.array(r["FE"], dtype=float)
        FA = np.array(r["FA"], dtype=float)

        # capacity Phi_se (scaled mass/time)
        Phi_se = np.array([
            capacity_logistic(
                ti,
                sp["K_max_total"],
                p.alpha, p.r, p.delta_eff,
                failure_active=p.failure_active,
                fail_t_start=p.fail_t_start,
                fail_duration=p.fail_duration,
                fail_severity=p.fail_severity,
                is_numeric=True
            ) for ti in t
        ], dtype=float)

        # "equivalent FE capacity" for plotting reference
        # FE_cap_eq satisfies Phi_se ≈ L_R*(FE_cap_eq*launch)
        FE_cap_eq = Phi_se / (sp["L_R"] * sp["launch"] + 1e-18)

        df = pd.DataFrame({
            "t_years_after_2050": t,
            "A_Mt": A,                       # scaled Mt
            "M_Mt": M,                       # scaled Mt
            "FE_klaunch_per_year": FE * sc.launch,
            "FA_klaunch_per_year": FA * sc.launch,
            "Phi_se_Mt_per_year": Phi_se,    # scaled Mt/yr
            "FE_cap_eq_klaunch_per_year": FE_cap_eq * sc.launch,
        })

        fn = f"trajectories_{r.get('label','scenario')}.csv"
        df.to_csv(os.path.join(OUTDIR, fn), index=False)


def plot_combined_results(res_pol, res_fail):
    if "t" not in res_pol or "FE" not in res_pol or "M" not in res_pol:
        print(f"[Warn] Cannot plot: Policy scenario '{res_pol.get('label','')}' has no trajectory data.")
        return
    if "t" not in res_fail or "FE" not in res_fail or "M" not in res_fail:
        print(f"[Warn] Cannot plot: Resilience scenario '{res_fail.get('label','')}' has no trajectory data.")
        return

    t = np.array(res_pol["t"])
    sc = Scales()

    # Fig 1
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sp_pol = _scaled_params(res_pol["p"])

    se_cap_eq = []
    for ti in t:
        Phi_se = capacity_logistic(
            ti,
            sp_pol["K_max_total"],
            res_pol["p"].alpha,
            res_pol["p"].r,
            res_pol["p"].delta_eff,
            failure_active=res_pol["p"].failure_active,
            fail_t_start=res_pol["p"].fail_t_start,
            fail_duration=res_pol["p"].fail_duration,
            fail_severity=res_pol["p"].fail_severity,
            is_numeric=True,
        )
        se_cap_eq.append(Phi_se / (sp_pol["L_R"] * sp_pol["launch"] + 1e-18))

    ax1.plot(t, np.array(res_pol["FE"]) * sc.launch, label=r"Earth Rockets $F_E$")
    ax1.plot(t, np.array(se_cap_eq) * sc.launch, linestyle="-.", alpha=0.7, label="Elevator Capacity (eq)")
    ax1.axvline(float(res_pol.get("t_late", res_pol["p"].t_late_frac * res_pol["p"].T)), linestyle=":", alpha=0.5)
    ax1.set_title("Fig 1: Optimal Mode Switching Strategy (Convergent v6 - Deliverable)")
    ax1.set_xlabel("Years after 2050")
    ax1.set_ylabel("Launch Rate (1/yr)")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Fig1_OptimalSwitching.png"))
    plt.close()

    # Fig 2
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    ax2a.plot(t, np.array(res_pol["FE"]) * sc.launch, linestyle="--", label="Policy")
    ax2a.plot(t, np.array(res_fail["FE"]) * sc.launch, label="Resilience")
    p_f = res_fail["p"]
    ax2a.axvspan(p_f.fail_t_start, p_f.fail_t_start + p_f.fail_duration, alpha=0.15, label="Failure")
    ax2a.axvline(float(res_pol.get("t_late", p_f.t_late_frac * p_f.T)), linestyle=":", alpha=0.5,
                 label="Late window start")
    ax2a.set_title("Fig 2: System Resilience Analysis (Convergent v6 - Deliverable)")
    ax2a.set_ylabel("Launch Rate (1/yr)")
    ax2a.legend()

    ax2b.plot(t, np.array(res_pol["M"]), label="Policy mass")
    ax2b.plot(t, np.array(res_fail["M"]), linestyle="--", label="Resilience mass")
    goal_M_scaled = _scaled_params(res_pol["p"])["M_goal"]
    ax2b.axhline(goal_M_scaled, linestyle=":", label="Goal")
    ax2b.set_ylabel("Mass (Mt)")
    ax2b.set_xlabel("Years after 2050")
    ax2b.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Fig2_Resilience.png"))
    plt.close()


# ==========================================
# 6) Main
# ==========================================
def main():
    print(">>> Running CONVERGENT v6 DELIVERABLE (audit-ready outputs)")

    resA = solve_with_cap_continuation(Params(gamma=1e-5, failure_active=False), "A_Baseline")
    resB = solve_with_cap_continuation(Params(gamma=0.08, failure_active=False), "B_Policy")
    resC = solve_with_cap_continuation(Params(gamma=0.05, failure_active=True), "C_Resilience")

    for r in (resA, resB, resC):
        if "p" not in r:
            r["p"] = finalize_params(Params())

    export_results_table([resA, resB, resC])
    export_consistency_report([resA, resB, resC])
    export_trajectories([resA, resB, resC])
    plot_combined_results(resB, resC)

    print(f"\n>>> Done. Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()
