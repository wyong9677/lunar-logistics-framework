# -*- coding: utf-8 -*-
"""
revised_solver_core.py
----------------------
Single-file core numerical library for the paper.

This file functionally covers three conceptual layers in ONE script:

1) campaign_solver.py
   - Campaign-scale dynamic planning model
   - Params / preset_mode / solve_policy / tau_star_bisect
   - build_cases / solve_all_policies / solve_cost_only / run_tornado

2) reduced_regime_solver.py
   - Reduced paper-level / mature-regime diagnostic model
   - PaperParams / finalize_params / solve_haat
   - phase-out support metrics

3) shared_postprocess.py
   - shared numerical integration / constraint-violation / phase-out utilities

IMPORTANT
---------
These model layers are intentionally kept in one file for convenience, but they
must still be treated as DIFFERENT model layers in the manuscript:

- campaign-scale full model:
    Params + solve_policy + tau_star_bisect + build_cases + run_tornado
- reduced paper-level / mature-regime diagnostic model:
    PaperParams + solve_haat + detect_phaseout_year

They are not interchangeable in interpretation.
"""

from __future__ import annotations

import os
import json
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import casadi as ca
import numpy as np
import pandas as pd

try:
    from solver_core import Scales, capacity_logistic, rocket_health, window_health
except ImportError as e:
    raise ImportError(
        "Missing dependency 'solver_core.py'. "
        "This file must be present to run the campaign-scale solver."
    ) from e


# Reproducible output root (shared by campaign-scale and reduced-model scripts).
# Default is aligned with the paper pipeline; environment override keeps the core
# script usable from batch runners without hard-coding submission-time paths.
OUTDIR = os.environ.get("PAPER_OUTDIR", "paper_outputs")
os.makedirs(OUTDIR, exist_ok=True)

MODEL_LAYER_CAMPAIGN = "campaign_scale_full_model"
MODEL_LAYER_REDUCED = "reduced_mature_regime_diagnostic"

# Practical IPOPT statuses accepted by each model layer. These are kept explicit so
# that the manuscript and supplementary material can cite exactly how numerical
# acceptance was defined.
CAMPAIGN_ACCEPTED_RETURN_STATUS_SUBSTRINGS = (
    "Solve_Succeeded",
    "Solved_To_Acceptable_Level",
)
REDUCED_ACCEPTED_RETURN_STATUS_SUBSTRINGS = (
    "Solve_Succeeded",
    "Solved_To_Acceptable_Level",
    "Maximum_Iterations_Exceeded",
    "Feasible_Point_Found",
)


def _result_scope_note(model_layer: str) -> str:
    if model_layer == MODEL_LAYER_CAMPAIGN:
        return (
            "Primary architecture-comparison claims must be drawn from the "
            "campaign-scale full model only."
        )
    return (
        "Reduced-model outputs are mechanism-oriented diagnostics and must not "
        "be interpreted as full campaign-scale robustness evidence."
    )


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# =========================================================
# Shared postprocess / utility layer
# =========================================================

def trapz_compat(y: np.ndarray, x: np.ndarray) -> float:
    """Compatibility wrapper for numpy trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def viol_scaled(gval: np.ndarray, lbg: np.ndarray, ubg: np.ndarray, g_scale: np.ndarray) -> float:
    """Maximum scaled constraint violation for campaign-scale NLP."""
    low = np.maximum(lbg - gval, 0.0)
    high = np.maximum(gval - ubg, 0.0)
    high[np.isinf(ubg)] = 0.0
    low[np.isinf(lbg)] = 0.0
    v = low + high
    if v.size == 0:
        return 0.0
    return float(np.max(v / np.maximum(g_scale, 1e-12)))


def _constraint_violation_abs(gval: np.ndarray, lbg: np.ndarray, ubg: np.ndarray) -> float:
    """Maximum absolute constraint violation for reduced-model NLP."""
    low = np.maximum(lbg - gval, 0.0)
    high = np.maximum(gval - ubg, 0.0)
    high[np.isinf(ubg)] = 0.0
    low[np.isinf(lbg)] = 0.0
    v = low + high
    if v.size == 0:
        return 0.0
    return float(np.max(v))


def _window_overlap(h0: float, h1: float, w0: float, w1: float) -> bool:
    return max(h0, w0) < min(h1, w1)


def metric_JE(t: np.ndarray, FE: np.ndarray) -> float:
    """Reduced-model Earth-launch burden metric."""
    return trapz_compat(np.asarray(FE, dtype=float) ** 2, np.asarray(t, dtype=float))


def metric_NEM(t: np.ndarray, FE: np.ndarray) -> float:
    """Reduced-model cumulative Earth-launch count proxy."""
    return trapz_compat(np.asarray(FE, dtype=float), np.asarray(t, dtype=float))


def detect_phaseout_year(
    t: np.ndarray,
    FE: np.ndarray,
    eps_launch_per_year: float,
    window_years: float,
    relax: float = 1.0,
) -> Optional[float]:
    """
    Detect the first time after which the moving-average FE remains below threshold.
    Returns None if no finite-horizon phase-out is detected.
    """
    t = np.asarray(t, dtype=float)
    FE = np.asarray(FE, dtype=float)
    if t.size < 2:
        return None

    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return None

    win = max(1, int(round(window_years / dt)))
    kernel = np.ones(win, dtype=float) / float(win)
    FE_avg = np.convolve(FE, kernel, mode="valid")
    thr = float(eps_launch_per_year) * float(relax)

    ok = FE_avg <= thr
    if not np.any(ok):
        return None

    suffix_all = np.flip(np.cumprod(np.flip(ok.astype(int))).astype(bool))
    if not np.any(suffix_all):
        return None

    j = int(np.argmax(suffix_all))
    idx_t = j + win - 1
    if idx_t < 0 or idx_t >= t.size:
        return None
    return float(t[idx_t])


def classify_feasibility(res: Dict[str, Any]) -> str:
    """Reduced-model feasibility classifier used by paper-level scripts."""
    if str(res.get("solver_status", "")) == "crashed":
        return "crashed"
    if not bool(res.get("feasible", False)):
        return "infeasible"
    return "ok"


def _solver_practically_accepted(
    stats: Dict[str, Any],
    accepted_status_substrings: Tuple[str, ...],
) -> Tuple[bool, str]:
    """
    Centralized practical IPOPT acceptance rule used by both model layers.
    """
    return_status = str(stats.get("return_status", ""))
    success_flag = bool(stats.get("success", False))
    accepted = success_flag or any(s in return_status for s in accepted_status_substrings)
    return bool(accepted), return_status


def _archive_run_metadata(filename: str, payload: Dict[str, Any]) -> str:
    """Write lightweight JSON metadata next to CSV artifacts for reproducibility."""
    outpath = os.path.join(OUTDIR, filename)
    _write_json(outpath, payload)
    return outpath


# =========================================================
# Campaign-scale full model
# =========================================================

@dataclass
class Params:
    # Mission / demand
    M_goal_tons: float = 1e8
    P_pop: float = 1e5

    # One-year post-build water logistics (gross use with recovery)
    w_ls_kg_per_person_day: float = 3.75
    rho_recovery: float = 0.95
    ops_supply_factor: float = 1.0

    # Scenario: mixed | se_only | rocket_only
    scenario: str = "mixed"

    # Discretization + variable build-time bounds
    N_build: int = 120
    N_ops: int = 40
    tau_min: float = 1.0
    tau_max: float = 400.0

    # SE throughput logistic
    K_max_total_tpy: float = 3.0 * 179000.0
    alpha: float = 0.02
    r: float = 0.18
    delta_eff: float = 0.8714

    # Vehicles / stock
    L_R_ton_per_launch: float = 125.0
    FE_max_launch_per_yr: float = 100000.0
    FA_max_launch_per_yr: float = 200000.0
    A_max_tons: float = 5e5
    A_gate_tons: float = 1e3

    # Cost / Earth-launch burden proxy
    C_E_dollars_per_launch: float = 1.0e8
    C_A_dollars_per_launch: float = 2.0e7
    C_SE_dollars_per_ton: float = 250.0
    e_E_per_launch: float = 1275.0
    e_A_per_launch: float = 450.0
    e_SE_per_ton: float = 2.0
    gamma_env: float = 1e-4

    # Failure toggles
    se_failure_active: bool = False
    rocket_failure_active: bool = False

    # Failure timing mode: absolute | relative
    failure_timing_mode: str = "relative"

    # Absolute-time windows (years)
    se_fail_t_start: float = 20.0
    se_fail_duration: float = 3.0
    se_fail_severity: float = 0.90
    rocket_fail_t_start: float = 22.0
    rocket_fail_duration: float = 2.0
    rocket_fail_severity: float = 0.50

    # Relative-to-build windows
    se_fail_theta_build: float = 0.35
    se_fail_frac_build: float = 0.15
    rocket_fail_theta_build: float = 0.55
    rocket_fail_frac_build: float = 0.10

    # Optional relative-to-ops-year windows
    se_fail_theta_ops: float = 0.0
    se_fail_frac_ops: float = 0.0
    rocket_fail_theta_ops: float = 0.0
    rocket_fail_frac_ops: float = 0.0

    # Optimization controls
    w_smooth: float = 0.5
    w_end: float = 1e2
    eps_tau: float = 0.05
    stage1_reg: float = 1e-3
    ops_require_rate_each_step: bool = False

    # Solver controls
    ipopt_max_iter: int = 8000
    ipopt_tol: float = 1e-7
    ipopt_acceptable_tol: float = 1e-4
    ipopt_acceptable_constr_viol_tol: float = 1e-6
    ipopt_acceptable_iter: int = 600
    ipopt_bound_relax_factor: float = 1e-10

    # Feasibility reporting
    tol_viol_scaled: float = 1e-6
    model_layer: str = MODEL_LAYER_CAMPAIGN
    practical_accept_rule: Tuple[str, ...] = CAMPAIGN_ACCEPTED_RETURN_STATUS_SUBSTRINGS


def preset_mode(p: Params, run_mode: str) -> Params:
    """Convenience preset for fast/full campaign-scale runs."""
    if run_mode == "full":
        p.N_build = 250
        p.N_ops = 60
        p.ipopt_max_iter = 40000
        p.ipopt_tol = 1e-8
        p.ipopt_acceptable_iter = 800
    else:
        p.N_build = 120
        p.N_ops = 40
        p.ipopt_max_iter = 8000
        p.ipopt_tol = 1e-7
        p.ipopt_acceptable_iter = 600
    return p


def scaled(p: Params) -> Dict[str, float]:
    """Scale campaign parameters into numerically convenient units."""
    sc = Scales()
    scenario = p.scenario.strip().lower()
    if scenario not in ("mixed", "se_only", "rocket_only"):
        raise ValueError("scenario must be mixed|se_only|rocket_only")

    W_year_tons = (1.0 - float(p.rho_recovery)) * float(p.P_pop) * float(p.w_ls_kg_per_person_day) * 365.0 / 1000.0
    W_year_tons *= max(float(p.ops_supply_factor), 0.0)

    sp = {
        "mass": sc.mass,
        "launch": sc.launch,
        "money": sc.money,
        "M_goal": p.M_goal_tons / sc.mass,
        "W_year": W_year_tons / sc.mass,
        "W_req": W_year_tons / sc.mass,
        "A_max": p.A_max_tons / sc.mass,
        "A_gate": p.A_gate_tons / sc.mass,
        "L_R": p.L_R_ton_per_launch / sc.mass,
        "FE_cap": p.FE_max_launch_per_yr / sc.launch,
        "FA_cap": p.FA_max_launch_per_yr / sc.launch,
        "K_max": p.K_max_total_tpy / sc.mass,
        "C_E": (p.C_E_dollars_per_launch * sc.launch) / sc.money,
        "C_A": (p.C_A_dollars_per_launch * sc.launch) / sc.money,
        "C_SE": (p.C_SE_dollars_per_ton * sc.mass) / sc.money,
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


def _failure_windows_expr(p: Params, tau_expr):
    """Return build/ops failure windows as expressions or floats."""
    if p.failure_timing_mode == "absolute":
        se_start = float(p.se_fail_t_start)
        se_dur = float(p.se_fail_duration)
        rk_start = float(p.rocket_fail_t_start)
        rk_dur = float(p.rocket_fail_duration)
    else:
        se_start = float(p.se_fail_theta_build) * tau_expr
        se_dur = float(p.se_fail_frac_build) * tau_expr
        rk_start = float(p.rocket_fail_theta_build) * tau_expr
        rk_dur = float(p.rocket_fail_frac_build) * tau_expr

    se_ops = None
    rk_ops = None
    if p.se_fail_theta_ops > 0.0 and p.se_fail_frac_ops > 0.0:
        se_ops = (tau_expr + float(p.se_fail_theta_ops), float(p.se_fail_frac_ops))
    if p.rocket_fail_theta_ops > 0.0 and p.rocket_fail_frac_ops > 0.0:
        rk_ops = (tau_expr + float(p.rocket_fail_theta_ops), float(p.rocket_fail_frac_ops))
    return se_start, se_dur, rk_start, rk_dur, se_ops, rk_ops


def build_nlp_variable_tau(
    p: Params,
    sp: Dict[str, float],
    policy: str,
    tau_star: Optional[float] = None,
    tau_fixed: Optional[float] = None,
):
    """
    Build campaign-scale NLP with variable build horizon tau and one-year ops stage.
    """
    Nb, No = int(p.N_build), int(p.N_ops)

    tau = ca.SX.sym("tau", 1)
    hb = tau[0] / Nb
    ho = 1.0 / No

    Ab = ca.SX.sym("Ab", Nb + 1)
    Mb = ca.SX.sym("Mb", Nb + 1)
    Ao = ca.SX.sym("Ao", No + 1)
    So = ca.SX.sym("So", No + 1)

    FEb = ca.SX.sym("FEb", Nb + 1)
    FAb = ca.SX.sym("FAb", Nb + 1)
    Ub = ca.SX.sym("Ub", Nb + 1)

    FEo = ca.SX.sym("FEo", No + 1)
    FAo = ca.SX.sym("FAo", No + 1)
    Uo = ca.SX.sym("Uo", No + 1)

    x = ca.vertcat(tau, Ab, Mb, Ao, So, FEb, FAb, Ub, FEo, FAo, Uo)

    offset = 0
    idx = {}
    for k, n in [
        ("tau", 1), ("Ab", Nb + 1), ("Mb", Nb + 1), ("Ao", No + 1), ("So", No + 1),
        ("FEb", Nb + 1), ("FAb", Nb + 1), ("Ub", Nb + 1),
        ("FEo", No + 1), ("FAo", No + 1), ("Uo", No + 1),
    ]:
        idx[k] = slice(offset, offset + n)
        offset += n

    lbx, ubx = [], []
    if tau_fixed is None:
        lbx += [float(p.tau_min)]
        ubx += [float(p.tau_max)]
    else:
        lbx += [float(tau_fixed)]
        ubx += [float(tau_fixed)]

    lbx += [0.0] * (Nb + 1); ubx += [sp["A_max"]] * (Nb + 1)
    lbx += [0.0] * (Nb + 1); ubx += [10.0 * sp["M_goal"]] * (Nb + 1)
    lbx += [0.0] * (No + 1); ubx += [sp["A_max"]] * (No + 1)
    lbx += [0.0] * (No + 1); ubx += [10.0 * sp["W_year"]] * (No + 1)

    lbx += [0.0] * (Nb + 1); ubx += [sp["FE_cap"]] * (Nb + 1)
    lbx += [0.0] * (Nb + 1); ubx += [sp["FA_cap"]] * (Nb + 1)
    lbx += [0.0] * (Nb + 1); ubx += [sp["K_max"]] * (Nb + 1)
    lbx += [0.0] * (No + 1); ubx += [sp["FE_cap"]] * (No + 1)
    lbx += [0.0] * (No + 1); ubx += [sp["FA_cap"]] * (No + 1)
    lbx += [0.0] * (No + 1); ubx += [sp["K_max"]] * (No + 1)

    lbx = np.asarray(lbx, dtype=float)
    ubx = np.asarray(ubx, dtype=float)

    gate_slope = (sp["FA_cap"] / sp["A_gate"]) if (sp["A_gate"] > 0.0 and sp["FA_cap"] > 0.0) else 0.0
    se_start, se_dur, rk_start, rk_dur, se_ops, rk_ops = _failure_windows_expr(p, tau[0])

    def t_build(i):
        return tau[0] * (float(i) / float(Nb))

    def t_ops(j):
        return tau[0] + float(j) / float(No)

    def phi_at(t_abs):
        cap = capacity_logistic(
            t_abs,
            K_max=sp["K_max"],
            alpha=p.alpha,
            r=p.r,
            delta_eff=p.delta_eff,
            failure_active=bool(p.se_failure_active),
            fail_t_start=se_start,
            fail_duration=se_dur,
            fail_severity=float(p.se_fail_severity),
            is_numeric=False,
        )
        if p.se_failure_active and se_ops is not None:
            cap = cap * window_health(t_abs, se_ops[0], se_ops[1], p.se_fail_severity, is_numeric=False)
        return ca.fmax(cap, 0.0)

    def rocket_health_at(t_abs):
        rh = rocket_health(
            t_abs,
            failure_active=bool(p.rocket_failure_active),
            fail_t_start=rk_start,
            fail_duration=rk_dur,
            fail_severity=p.rocket_fail_severity,
            is_numeric=False,
        )
        if p.rocket_failure_active and rk_ops is not None:
            rh = rh * rocket_health(
                t_abs,
                failure_active=True,
                fail_t_start=rk_ops[0],
                fail_duration=rk_ops[1],
                fail_severity=p.rocket_fail_severity,
                is_numeric=False,
            )
        return ca.fmin(ca.fmax(rh, 0.0), 1.0)

    g, lbg, ubg, g_scale = [], [], [], []

    # Initial conditions
    g += [Ab[0], Mb[0]]
    lbg += [0.0, 0.0]
    ubg += [0.0, 0.0]
    g_scale += [max(sp["A_max"], 1.0), max(sp["M_goal"], 1.0)]

    # Build stage
    for i in range(Nb):
        ti = t_build(i)
        phi_i = phi_at(ti)

        g += [Ub[i] - phi_i]
        lbg += [-np.inf]
        ubg += [0.0]
        g_scale += [max(sp["K_max"], 1.0)]

        if gate_slope > 0.0:
            g += [FAb[i] - gate_slope * Ab[i]]
            lbg += [-np.inf]
            ubg += [0.0]
            g_scale += [max(sp["FA_cap"], 1.0)]

        outflow = sp["L_R"] * (FAb[i] * sp["launch"])
        rh = rocket_health_at(ti)
        arrive = sp["L_R"] * rh * (FEb[i] * sp["launch"]) + rh * outflow

        g += [Ab[i + 1] - (Ab[i] + hb * (Ub[i] - outflow))]
        lbg += [0.0]
        ubg += [0.0]
        g_scale += [max(sp["A_max"], 1.0)]

        g += [Mb[i + 1] - (Mb[i] + hb * arrive)]
        lbg += [0.0]
        ubg += [0.0]
        g_scale += [max(sp["M_goal"], 1.0)]

    g += [Mb[-1] - sp["M_goal"]]
    lbg += [0.0]
    ubg += [np.inf]
    g_scale += [max(sp["M_goal"], 1.0)]

    # Ops stage initial conditions
    g += [Ao[0] - Ab[-1], So[0]]
    lbg += [0.0, 0.0]
    ubg += [0.0, 0.0]
    g_scale += [max(sp["A_max"], 1.0), max(sp["W_year"], 1.0)]

    # Ops stage
    for j in range(No):
        tj = t_ops(j)
        phi_j = phi_at(tj)

        g += [Uo[j] - phi_j]
        lbg += [-np.inf]
        ubg += [0.0]
        g_scale += [max(sp["K_max"], 1.0)]

        if gate_slope > 0.0:
            g += [FAo[j] - gate_slope * Ao[j]]
            lbg += [-np.inf]
            ubg += [0.0]
            g_scale += [max(sp["FA_cap"], 1.0)]

        outflow = sp["L_R"] * (FAo[j] * sp["launch"])
        rh = rocket_health_at(tj)
        delivered = sp["L_R"] * rh * (FEo[j] * sp["launch"]) + rh * outflow

        g += [Ao[j + 1] - (Ao[j] + ho * (Uo[j] - outflow))]
        lbg += [0.0]
        ubg += [0.0]
        g_scale += [max(sp["A_max"], 1.0)]

        g += [So[j + 1] - (So[j] + ho * delivered)]
        lbg += [0.0]
        ubg += [0.0]
        g_scale += [max(sp["W_year"], 1.0)]

        if p.ops_require_rate_each_step:
            g += [delivered - sp["W_req"]]
            lbg += [0.0]
            ubg += [np.inf]
            g_scale += [max(sp["W_req"], 1.0)]

    g += [So[-1] - sp["W_year"]]
    lbg += [0.0]
    ubg += [np.inf]
    g_scale += [max(sp["W_year"], 1.0)]

    # Near-minimum-time constraint for cost_opt
    if policy == "cost_opt":
        if tau_star is None:
            raise ValueError("cost_opt needs tau_star")
        g += [tau[0] - (1.0 + p.eps_tau) * float(tau_star)]
        lbg += [-np.inf]
        ubg += [0.0]
        g_scale += [max(p.tau_max, 1.0)]

    # Objective scaling
    tau_nom = 50.0
    fe_ref = min(sp["FE_cap"], sp["M_goal"] / max(tau_nom * sp["L_R"] * sp["launch"], 1e-12)) if sp["FE_cap"] > 0 else 0.0
    fa_ref = min(sp["FA_cap"], sp["M_goal"] / max(tau_nom * sp["L_R"] * sp["launch"], 1e-12)) if sp["FA_cap"] > 0 else 0.0
    u_ref = min(sp["K_max"], sp["M_goal"] / max(tau_nom, 1e-12)) if sp["K_max"] > 0 else 0.0
    cost_scale = max(1e-3, tau_nom * (sp["C_E"] * fe_ref + sp["C_A"] * fa_ref + sp["C_SE"] * u_ref))

    J = 0.0
    for i in range(Nb):
        c = sp["C_E"] * FEb[i] + sp["C_A"] * FAb[i] + sp["C_SE"] * Ub[i]
        e = sp["e_E"] * FEb[i] + sp["e_A"] * FAb[i] + sp["e_SE"] * Ub[i]
        env = p.gamma_env * (e ** 2)
        dfe = (FEb[i + 1] - FEb[i]) / (hb + 1e-12)
        dfa = (FAb[i + 1] - FAb[i]) / (hb + 1e-12)
        du = (Ub[i + 1] - Ub[i]) / (hb + 1e-12)
        smooth = p.w_smooth * (dfe ** 2 + dfa ** 2 + du ** 2)
        J += hb * (c + env + smooth) / cost_scale

    for j in range(No):
        c = sp["C_E"] * FEo[j] + sp["C_A"] * FAo[j] + sp["C_SE"] * Uo[j]
        e = sp["e_E"] * FEo[j] + sp["e_A"] * FAo[j] + sp["e_SE"] * Uo[j]
        env = p.gamma_env * (e ** 2)
        dfe = (FEo[j + 1] - FEo[j]) / ho
        dfa = (FAo[j + 1] - FAo[j]) / ho
        du = (Uo[j + 1] - Uo[j]) / ho
        smooth = p.w_smooth * (dfe ** 2 + dfa ** 2 + du ** 2)
        J += ho * (c + env + smooth) / cost_scale

    J += p.w_end * (
        FEb[-1] ** 2 + FAb[-1] ** 2 + Ub[-1] ** 2 +
        FEo[-1] ** 2 + FAo[-1] ** 2 + Uo[-1] ** 2
    ) / cost_scale

    if policy == "time_opt":
        J = tau[0] + p.stage1_reg * J

    nlp = {"x": x, "f": J, "g": ca.vertcat(*g)}
    meta = {
        "idx": idx,
        "Nb": Nb,
        "No": No,
        "lbx": lbx,
        "ubx": ubx,
        "lbg": np.asarray(lbg, dtype=float),
        "ubg": np.asarray(ubg, dtype=float),
        "g_scale": np.asarray(g_scale, dtype=float),
    }
    return nlp, meta


def initial_guess(
    p: Params,
    sp: Dict[str, float],
    meta: Dict[str, Any],
    tau0: float,
    noise_scale: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Construct campaign-scale initial guess with optional random perturbation.
    """
    Nb, No = meta["Nb"], meta["No"]
    ii = meta["idx"]
    x0 = np.zeros(meta["ubx"].shape[0], dtype=float)
    x0[ii["tau"]] = np.clip(tau0, p.tau_min, p.tau_max)

    avg_klaunch = sp["M_goal"] / max(tau0 * sp["L_R"] * sp["launch"] * 0.9, 1e-12)
    FEb = np.zeros(Nb + 1); FAb = np.zeros(Nb + 1); Ub = np.zeros(Nb + 1)
    FEo = np.zeros(No + 1); FAo = np.zeros(No + 1); Uo = np.zeros(No + 1)

    if p.scenario == "rocket_only":
        FEb[:] = min(sp["FE_cap"], avg_klaunch)
    elif p.scenario == "se_only":
        FAb[:] = min(sp["FA_cap"], avg_klaunch)
        Ub[:] = sp["K_max"]
    else:
        FEb[:] = min(sp["FE_cap"], 0.35 * avg_klaunch)
        FAb[:] = min(sp["FA_cap"], 0.65 * avg_klaunch)
        Ub[:] = sp["K_max"]

    FEo[:] = FEb[-1]
    FAo[:] = FAb[-1]
    Uo[:] = Ub[-1]

    if noise_scale > 0.0:
        rr = rng if rng is not None else np.random.default_rng(0)
        for arr in (FEb, FAb, Ub, FEo, FAo, Uo):
            jitter = 1.0 + noise_scale * rr.standard_normal(arr.shape[0])
            arr[:] = np.maximum(arr * jitter, 0.0)
        FEb[:] = np.minimum(FEb, sp["FE_cap"])
        FAb[:] = np.minimum(FAb, sp["FA_cap"])
        Ub[:] = np.minimum(Ub, sp["K_max"])
        FEo[:] = np.minimum(FEo, sp["FE_cap"])
        FAo[:] = np.minimum(FAo, sp["FA_cap"])
        Uo[:] = np.minimum(Uo, sp["K_max"])

    Ab = np.zeros(Nb + 1)
    Mb = np.zeros(Nb + 1)
    Ao = np.zeros(No + 1)
    So = np.zeros(No + 1)

    hb = tau0 / Nb
    ho = 1.0 / No
    gate_slope = (sp["FA_cap"] / sp["A_gate"]) if (sp["A_gate"] > 0 and sp["FA_cap"] > 0) else 0.0

    for i in range(Nb):
        if gate_slope > 0:
            FAb[i] = min(FAb[i], gate_slope * Ab[i])
        outflow = sp["L_R"] * (FAb[i] * sp["launch"])
        max_out = (Ab[i] / hb) + Ub[i]
        if outflow > max_out > 0:
            outflow = max_out
        Ab[i + 1] = np.clip(Ab[i] + hb * (Ub[i] - outflow), 0.0, sp["A_max"])
        Mb[i + 1] = max(0.0, Mb[i] + hb * (sp["L_R"] * (FEb[i] * sp["launch"]) + outflow))

    Ao[0] = Ab[-1]
    for j in range(No):
        if gate_slope > 0:
            FAo[j] = min(FAo[j], gate_slope * Ao[j])
        outflow = sp["L_R"] * (FAo[j] * sp["launch"])
        max_out = (Ao[j] / ho) + Uo[j]
        if outflow > max_out > 0:
            outflow = max_out
        Ao[j + 1] = np.clip(Ao[j] + ho * (Uo[j] - outflow), 0.0, sp["A_max"])
        So[j + 1] = max(0.0, So[j] + ho * (sp["L_R"] * (FEo[j] * sp["launch"]) + outflow))

    x0[ii["Ab"]] = Ab
    x0[ii["Mb"]] = Mb
    x0[ii["Ao"]] = Ao
    x0[ii["So"]] = So
    x0[ii["FEb"]] = FEb
    x0[ii["FAb"]] = FAb
    x0[ii["Ub"]] = Ub
    x0[ii["FEo"]] = FEo
    x0[ii["FAo"]] = FAo
    x0[ii["Uo"]] = Uo
    return np.clip(x0, meta["lbx"], meta["ubx"])


def _campaign_failure_result(
    p: Params,
    label: str,
    policy: str,
    tau_star: Optional[float] = None,
    return_status: str = "exception",
) -> Dict[str, Any]:
    return {
        "ModelLayer": MODEL_LAYER_CAMPAIGN,
        "InterpretationScope": _result_scope_note(MODEL_LAYER_CAMPAIGN),
        "CampaignAcceptanceRule": list(tuple(p.practical_accept_rule)),
        "ScenarioLabel": label,
        "Policy": policy,
        "ScenarioMode": p.scenario,
        "FailureTiming": p.failure_timing_mode,
        "SE_Disrupt": "ON" if p.se_failure_active else "OFF",
        "Rocket_Disrupt": "ON" if p.rocket_failure_active else "OFF",
        "SE_Triggered": "NO",
        "Rocket_Triggered": "NO",
        "SolverSuccess": "NO",
        "PracticalAccepted": "NO",
        "ConstraintOK": "NO",
        "BuildOK": "NO",
        "OpsOK": "NO",
        "Viol_scaled": np.nan,
        "tau_ub_eps (yr)": float((1.0 + p.eps_tau) * tau_star) if (policy == "cost_opt" and tau_star is not None) else np.nan,
        "tau_build (yr)": np.nan,
        "Time_total (yr)": np.nan,
        "Cost_build ($B)": np.nan,
        "Cost_ops1yr ($B)": np.nan,
        "Cost_total ($B)": np.nan,
        "TotalCost ($B)": np.nan,
        "E_total (index)": np.nan,
        "EnvPenalty": np.nan,
        "FE_launch_build": np.nan,
        "FA_launch_build": np.nan,
        "FE_launch_ops": np.nan,
        "FA_launch_ops": np.nan,
        "SE_tons_build": np.nan,
        "SE_tons_ops": np.nan,
        "Total_FE_Launches": np.nan,
        "Total_FA_Launches": np.nan,
        "Total_SE_Throughput (tons)": np.nan,
        "Apex_Inventory_Peak (tons)": np.nan,
        "ReturnStatus": return_status,
        "OutputDir": OUTDIR,
        "TimeseriesCSV": "",
    }


def solve_policy(
    p: Params,
    label: str,
    policy: str,
    tau_star: Optional[float] = None,
    tau_fixed: Optional[float] = None,
    tau0_override: Optional[float] = None,
    init_noise_scale: float = 0.0,
    init_seed: Optional[int] = None,
    write_timeseries: bool = True,
) -> Dict[str, Any]:
    """
    Solve one campaign-scale policy instance and return summary metrics.
    """
    sp = scaled(p)
    nlp, meta = build_nlp_variable_tau(p, sp, policy=policy, tau_star=tau_star, tau_fixed=tau_fixed)

    solver = ca.nlpsol(
        "solver",
        "ipopt",
        nlp,
        {
            "ipopt": {
                "print_level": 0,
                "max_iter": int(p.ipopt_max_iter),
                "tol": float(p.ipopt_tol),
                "mu_strategy": "adaptive",
                "acceptable_tol": float(p.ipopt_acceptable_tol),
                "acceptable_constr_viol_tol": float(p.ipopt_acceptable_constr_viol_tol),
                "acceptable_iter": int(p.ipopt_acceptable_iter),
                "bound_relax_factor": float(p.ipopt_bound_relax_factor),
            },
            "print_time": False,
        },
    )

    if tau0_override is not None:
        tau0 = float(np.clip(tau0_override, p.tau_min, p.tau_max))
    else:
        tau0 = (
            float(tau_fixed)
            if tau_fixed is not None
            else float(np.clip(max(8.0, p.tau_min * 2.0), p.tau_min, p.tau_max))
        )

    rng = np.random.default_rng(init_seed) if init_seed is not None else None
    x0 = initial_guess(p, sp, meta, tau0, noise_scale=float(max(init_noise_scale, 0.0)), rng=rng)

    try:
        sol = solver(x0=x0, lbx=meta["lbx"], ubx=meta["ubx"], lbg=meta["lbg"], ubg=meta["ubg"])
    except Exception:
        return _campaign_failure_result(p, label, policy, tau_star=tau_star, return_status="exception")

    stats = solver.stats()
    success, status = _solver_practically_accepted(stats, tuple(p.practical_accept_rule))

    x = np.asarray(sol["x"], dtype=float).flatten()
    g = np.asarray(sol["g"], dtype=float).flatten()
    v = viol_scaled(g, meta["lbg"], meta["ubg"], meta["g_scale"])

    ii = meta["idx"]
    Nb, No = meta["Nb"], meta["No"]

    tau = float(x[ii["tau"]][0])
    Ab, Mb = x[ii["Ab"]], x[ii["Mb"]]
    Ao, So = x[ii["Ao"]], x[ii["So"]]
    FEb, FAb, Ub = x[ii["FEb"]], x[ii["FAb"]], x[ii["Ub"]]
    FEo, FAo, Uo = x[ii["FEo"]], x[ii["FAo"]], x[ii["Uo"]]

    t_build = np.linspace(0.0, tau, Nb + 1)
    t_ops = np.linspace(tau, tau + 1.0, No + 1)
    t_all = np.concatenate([t_build, t_ops[1:]])

    A_all = np.concatenate([Ab, Ao[1:]])
    M_all = np.concatenate([Mb, np.full(No, Mb[-1])])
    S_all = np.concatenate([np.zeros(Nb + 1), So[1:]])
    FE_all = np.concatenate([FEb, FEo[1:]])
    FA_all = np.concatenate([FAb, FAo[1:]])
    U_all = np.concatenate([Ub, Uo[1:]])

    se_start, se_dur, rk_start, rk_dur, se_ops, rk_ops = _failure_windows_expr(p, tau)
    Phi_all = np.zeros_like(t_all)
    rh_all = np.ones_like(t_all)

    for k, tt in enumerate(t_all):
        Phi = capacity_logistic(
            tt,
            K_max=sp["K_max"],
            alpha=p.alpha,
            r=p.r,
            delta_eff=p.delta_eff,
            failure_active=bool(p.se_failure_active),
            fail_t_start=float(se_start),
            fail_duration=float(se_dur),
            fail_severity=float(p.se_fail_severity),
            is_numeric=True,
        )
        if p.se_failure_active and se_ops is not None:
            Phi *= window_health(tt, float(se_ops[0]), float(se_ops[1]), float(p.se_fail_severity), is_numeric=True)
        Phi_all[k] = max(float(Phi), 0.0)

        rh = rocket_health(
            tt,
            failure_active=bool(p.rocket_failure_active),
            fail_t_start=float(rk_start),
            fail_duration=float(rk_dur),
            fail_severity=float(p.rocket_fail_severity),
            is_numeric=True,
        )
        if p.rocket_failure_active and rk_ops is not None:
            rh *= rocket_health(
                tt,
                failure_active=True,
                fail_t_start=float(rk_ops[0]),
                fail_duration=float(rk_ops[1]),
                fail_severity=float(p.rocket_fail_severity),
                is_numeric=True,
            )
        rh_all[k] = float(np.clip(rh, 0.0, 1.0))

    U_pp = np.minimum(np.maximum(U_all, 0.0), Phi_all)
    FE_pp = np.maximum(FE_all, 0.0)
    FA_pp = np.maximum(FA_all, 0.0)

    cost_rate = sp["C_E"] * FE_pp + sp["C_A"] * FA_pp + sp["C_SE"] * U_pp
    env_rate = sp["e_E"] * FE_pp + sp["e_A"] * FA_pp + sp["e_SE"] * U_pp

    total_cost = trapz_compat(cost_rate, t_all)
    e_total = trapz_compat(env_rate, t_all)
    env_pen = trapz_compat(p.gamma_env * (env_rate ** 2), t_all)
    delivered_rate = sp["L_R"] * rh_all * (FE_pp + FA_pp) * sp["launch"]

    cost_rate_build = sp["C_E"] * FEb + sp["C_A"] * FAb + sp["C_SE"] * Ub
    cost_rate_ops = sp["C_E"] * FEo + sp["C_A"] * FAo + sp["C_SE"] * Uo
    cost_build = trapz_compat(cost_rate_build, t_build)
    cost_ops = trapz_compat(cost_rate_ops, t_ops)

    fe_launch_build = trapz_compat(np.maximum(FEb, 0.0), t_build) * sp["launch"]
    fa_launch_build = trapz_compat(np.maximum(FAb, 0.0), t_build) * sp["launch"]
    fe_launch_ops = trapz_compat(np.maximum(FEo, 0.0), t_ops) * sp["launch"]
    fa_launch_ops = trapz_compat(np.maximum(FAo, 0.0), t_ops) * sp["launch"]

    se_tons_build = trapz_compat(np.maximum(Ub, 0.0), t_build) * sp["mass"]
    se_tons_ops = trapz_compat(np.maximum(Uo, 0.0), t_ops) * sp["mass"]

    horizon0, horizon1 = 0.0, tau + 1.0
    se_triggered = False
    rocket_triggered = False
    if p.se_failure_active:
        se_triggered = _window_overlap(horizon0, horizon1, float(se_start), float(se_start) + float(se_dur))
        if se_ops is not None:
            se_triggered = se_triggered or _window_overlap(
                horizon0, horizon1, float(se_ops[0]), float(se_ops[0]) + float(se_ops[1])
            )
    if p.rocket_failure_active:
        rocket_triggered = _window_overlap(horizon0, horizon1, float(rk_start), float(rk_start) + float(rk_dur))
        if rk_ops is not None:
            rocket_triggered = rocket_triggered or _window_overlap(
                horizon0, horizon1, float(rk_ops[0]), float(rk_ops[0]) + float(rk_ops[1])
            )

    build_ok = float(Mb[-1]) >= sp["M_goal"] - 1e-9
    ops_ok = float(So[-1]) >= sp["W_year"] - 1e-9
    constraint_ok = v <= float(p.tol_viol_scaled)

    ts_path = ""
    if write_timeseries:
        ts_path = os.path.join(OUTDIR, f"timeseries_{label}__{policy}.csv")
        pd.DataFrame(
            {
                "ScenarioLabel": [label] * len(t_all),
                "Policy": [policy] * len(t_all),
                "t_yr": t_all,
                "A_Mt": A_all,
                "M_build_Mt": M_all,
                "S_ops_Mt": S_all,
                "FE_klaunch_per_yr": FE_pp,
                "FA_klaunch_per_yr": FA_pp,
                "U_se_Mt_per_yr": U_pp,
                "Phi_se_Mt_per_yr": Phi_all,
                "rocket_health": rh_all,
                "Delivered_Mt_per_yr": delivered_rate,
            }
        ).to_csv(ts_path, index=False)

    return {
        "ModelLayer": MODEL_LAYER_CAMPAIGN,
        "InterpretationScope": _result_scope_note(MODEL_LAYER_CAMPAIGN),
        "CampaignAcceptanceRule": list(tuple(p.practical_accept_rule)),
        "ScenarioLabel": label,
        "Policy": policy,
        "ScenarioMode": p.scenario,
        "FailureTiming": p.failure_timing_mode,
        "SE_Disrupt": "ON" if p.se_failure_active else "OFF",
        "Rocket_Disrupt": "ON" if p.rocket_failure_active else "OFF",
        "SE_Triggered": "YES" if se_triggered else "NO",
        "Rocket_Triggered": "YES" if rocket_triggered else "NO",
        "SolverSuccess": "YES" if success else "NO",
        "PracticalAccepted": "YES" if success else "NO",
        "ConstraintOK": "YES" if constraint_ok else "NO",
        "BuildOK": "YES" if build_ok else "NO",
        "OpsOK": "YES" if ops_ok else "NO",
        "Viol_scaled": float(v),
        "tau_ub_eps (yr)": float((1.0 + p.eps_tau) * tau_star) if (policy == "cost_opt" and tau_star is not None) else np.nan,
        "tau_build (yr)": tau,
        "Time_total (yr)": tau + 1.0,
        "Cost_build ($B)": cost_build,
        "Cost_ops1yr ($B)": cost_ops,
        "Cost_total ($B)": total_cost,
        "TotalCost ($B)": total_cost,
        "E_total (index)": e_total,
        "EnvPenalty": env_pen,
        "FE_launch_build": fe_launch_build,
        "FA_launch_build": fa_launch_build,
        "FE_launch_ops": fe_launch_ops,
        "FA_launch_ops": fa_launch_ops,
        "SE_tons_build": se_tons_build,
        "SE_tons_ops": se_tons_ops,
        "Total_FE_Launches": trapz_compat(FE_pp, t_all) * sp["launch"],
        "Total_FA_Launches": trapz_compat(FA_pp, t_all) * sp["launch"],
        "Total_SE_Throughput (tons)": trapz_compat(U_pp, t_all) * sp["mass"],
        "Apex_Inventory_Peak (tons)": float(np.max(np.maximum(A_all, 0.0)) * sp["mass"]),
        "ReturnStatus": status,
        "OutputDir": OUTDIR,
        "TimeseriesCSV": ts_path,
    }


def tau_star_bisect(
    p: Params,
    label: str,
    tol_tau: float = 0.05,
    max_iter: int = 24,
    init_noise_scale: float = 0.0,
    init_seed_base: Optional[int] = None,
) -> float:
    """
    Bisection search for minimum feasible build horizon.
    Returns NaN if even tau_max is infeasible.
    """
    lo, hi = float(p.tau_min), float(p.tau_max)

    def feasible(tau_v: float) -> bool:
        seed = None if init_seed_base is None else int(init_seed_base + int(round(1000 * tau_v)))
        r = solve_policy(
            p,
            label,
            policy="tau_star",
            tau_fixed=tau_v,
            init_noise_scale=init_noise_scale,
            init_seed=seed,
            write_timeseries=False,
        )
        return (
            r["SolverSuccess"] == "YES"
            and r["ConstraintOK"] == "YES"
            and r["BuildOK"] == "YES"
            and r["OpsOK"] == "YES"
        )

    if not feasible(hi):
        return float("nan")
    if feasible(lo):
        return lo

    for _ in range(max_iter):
        if hi - lo <= tol_tau:
            break
        mid = 0.5 * (hi + lo)
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    return hi


def solve_all_policies(label: str, p: Params) -> List[Dict[str, Any]]:
    """Solve tau_star / time_opt / cost_opt for one campaign-scale case."""
    tau_star = tau_star_bisect(p, label=label)
    if not np.isfinite(tau_star):
        raise RuntimeError(f"tau_star_bisect failed for case '{label}'")

    r_tau = solve_policy(p, label, policy="tau_star", tau_fixed=tau_star)
    r_tau["tau_star_bisect (yr)"] = tau_star

    r_time = solve_policy(p, label, policy="time_opt")
    r_time["tau_star_bisect (yr)"] = tau_star

    r_cost = solve_policy(p, label, policy="cost_opt", tau_star=tau_star)
    r_cost["tau_star_bisect (yr)"] = tau_star

    return [r_tau, r_time, r_cost]


def solve_cost_only(label: str, p: Params, write_timeseries: bool = True) -> Dict[str, Any]:
    """Solve only the campaign-scale cost-oriented policy for one case."""
    tau_star = tau_star_bisect(p, label=label)
    if not np.isfinite(tau_star):
        raise RuntimeError(f"tau_star_bisect failed for case '{label}'")
    r_cost = solve_policy(
        p,
        label,
        policy="cost_opt",
        tau_star=tau_star,
        write_timeseries=write_timeseries,
    )
    r_cost["tau_star_bisect (yr)"] = tau_star
    return r_cost


def run_tornado(base: Params, delta: float = 0.10) -> pd.DataFrame:
    """
    Campaign-scale OAT sensitivity around the mixed baseline comparator.
    Returns cost, build-time, and Earth-launch-burden changes.
    """
    candidates = [
        "K_max_total_tpy", "alpha", "r", "delta_eff", "L_R_ton_per_launch",
        "C_E_dollars_per_launch", "C_A_dollars_per_launch", "C_SE_dollars_per_ton",
        "gamma_env", "se_fail_severity", "rocket_fail_severity",
    ]

    base_case = replace(base, scenario="mixed", se_failure_active=False, rocket_failure_active=False)
    base_row = solve_cost_only("mixed_base", base_case, write_timeseries=False)
    base_cost = float(base_row["TotalCost ($B)"])
    base_tau = float(base_row["tau_build (yr)"])
    base_env = float(base_row["E_total (index)"])

    rows = []
    for key in candidates:
        val = getattr(base, key)

        if key == "se_fail_severity" and not bool(base.se_failure_active):
            for sign in (-1.0, 1.0):
                rows.append(
                    {
                        "param": key,
                        "direction": "down" if sign < 0 else "up",
                        "value": np.nan,
                        "cost": np.nan,
                        "tau_build": np.nan,
                        "env_total": np.nan,
                        "delta_cost": np.nan,
                        "delta_tau": np.nan,
                        "delta_env": np.nan,
                        "status": "SKIPPED_INACTIVE_FAILURE",
                    }
                )
            continue

        if key == "rocket_fail_severity" and not bool(base.rocket_failure_active):
            for sign in (-1.0, 1.0):
                rows.append(
                    {
                        "param": key,
                        "direction": "down" if sign < 0 else "up",
                        "value": np.nan,
                        "cost": np.nan,
                        "tau_build": np.nan,
                        "env_total": np.nan,
                        "delta_cost": np.nan,
                        "delta_tau": np.nan,
                        "delta_env": np.nan,
                        "status": "SKIPPED_INACTIVE_FAILURE",
                    }
                )
            continue

        for sign in (-1.0, 1.0):
            new_val = val * (1.0 + sign * delta)
            if isinstance(val, float) and key in {"alpha", "delta_eff", "se_fail_severity", "rocket_fail_severity"}:
                new_val = float(np.clip(new_val, 1e-4, 0.999))

            p2 = replace(base, **{key: new_val})
            try:
                row = solve_cost_only(
                    f"mix_{key}_{'down' if sign < 0 else 'up'}",
                    replace(p2, scenario="mixed"),
                    write_timeseries=False,
                )
                cost = float(row["TotalCost ($B)"])
                tau = float(row["tau_build (yr)"])
                env = float(row["E_total (index)"])
                rows.append({
                    "param": key,
                    "direction": "down" if sign < 0 else "up",
                    "value": new_val,
                    "cost": cost,
                    "tau_build": tau,
                    "env_total": env,
                    "delta_cost": cost - base_cost,
                    "delta_tau": tau - base_tau,
                    "delta_env": env - base_env,
                    "rel_delta_cost": (cost - base_cost) / max(abs(base_cost), 1e-12),
                    "rel_delta_tau": (tau - base_tau) / max(abs(base_tau), 1e-12),
                    "rel_delta_env": (env - base_env) / max(abs(base_env), 1e-12),
                    "status": "OK",
                })
            except Exception as e:
                rows.append({
                    "param": key,
                    "direction": "down" if sign < 0 else "up",
                    "value": new_val,
                    "cost": np.nan,
                    "tau_build": np.nan,
                    "env_total": np.nan,
                    "delta_cost": np.nan,
                    "delta_tau": np.nan,
                    "delta_env": np.nan,
                    "error": str(e),
                    "status": "FAILED",
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUTDIR, "tornado_summary.csv")
    df.to_csv(csv_path, index=False)
    _archive_run_metadata(
        "tornado_summary_meta.json",
        {
            "model_layer": MODEL_LAYER_CAMPAIGN,
            "interpretation_scope": _result_scope_note(MODEL_LAYER_CAMPAIGN),
            "delta": float(delta),
            "baseline_label": "mixed_base",
            "output_csv": csv_path,
            "n_rows": int(len(df)),
        },
    )
    return df


def build_cases(base: Params) -> List[Tuple[str, Params]]:
    """Campaign-scale scenario cases used by batch runners."""
    b = base
    return [
        ("rocket_only_base", replace(b, scenario="rocket_only", se_failure_active=False, rocket_failure_active=False)),
        ("se_only_base", replace(b, scenario="se_only", se_failure_active=False, rocket_failure_active=False)),
        ("mixed_base", replace(b, scenario="mixed", se_failure_active=False, rocket_failure_active=False)),
        ("mixed_sefail", replace(b, scenario="mixed", se_failure_active=True, rocket_failure_active=False)),
        ("mixed_rocketfail", replace(b, scenario="mixed", se_failure_active=False, rocket_failure_active=True)),
        ("mixed_bothfail", replace(b, scenario="mixed", se_failure_active=True, rocket_failure_active=True)),
    ]


# =========================================================
# Reduced paper-level / mature-regime diagnostic model
# =========================================================

@dataclass
class PaperParams:
    # Mission
    M_goal: float = 1e8
    P_pop: float = 1e5
    w_gross: float = 3.75
    rho_recovery: float = 0.95
    W_req: float = 0.0

    # SE baseline (kept close to campaign baseline by default)
    K_max_total: float = 3.0 * 179000.0
    alpha: float = 0.02
    r: float = 0.18
    delta_eff: float = 0.8714

    # Failure
    failure_active: bool = False
    fail_t_start: float = 20.0
    fail_duration: float = 3.0
    fail_severity: float = 0.90

    # Transport
    L_R: float = 125.0
    FE_max: float = 100000.0
    FA_max: float = 200000.0
    A_max: float = 5e5

    # Economics / policy
    C_E: float = 1.0e8
    C_A: float = 2.0e7
    c_SE: float = 250.0
    e_E: float = 1275.0
    gamma: float = 0.08

    # Discretization
    T: float = 50.0
    N: int = 100
    k_schedule: Tuple[float, ...] = (1.0, 5.0, 20.0, 100.0)

    # Smoothness
    w_smooth: float = 50.0

    # Phaseout metric
    phaseout_eps: float = 1.0
    phaseout_window_yrs: float = 1.0
    phaseout_relax: float = 1.0

    # IPOPT
    ipopt_max_iter: int = 4500
    ipopt_tol: float = 1e-6
    ipopt_acceptable_tol: float = 1e-4
    ipopt_acceptable_constr_viol_tol: float = 1e-6
    ipopt_acceptable_iter: int = 600
    ipopt_bound_relax_factor: float = 1e-10

    # Practical reduced-model acceptance rules (reported explicitly in metadata).
    reduced_terminal_gap_tol_abs: float = 1e-2
    reduced_viol_tol_abs: float = 1e-2
    model_layer: str = MODEL_LAYER_REDUCED
    practical_accept_rule: Tuple[str, ...] = REDUCED_ACCEPTED_RETURN_STATUS_SUBSTRINGS


def finalize_params(p: PaperParams) -> PaperParams:
    """Fill derived reduced-model quantities."""
    p.W_req = (1.0 - p.rho_recovery) * p.P_pop * p.w_gross * 365.0 / 1000.0
    return p


def _sigma_k_paper(z, k):
    return 1.0 / (1.0 + ca.exp(-ca.fmin(ca.fmax(k * z, -50), 50)))


def _gk_relu_paper(z, k):
    return z * _sigma_k_paper(z, k)


def _reduced_solver_accepted(stats: Dict[str, Any], p: PaperParams) -> Tuple[bool, str]:
    """
    Practical IPOPT acceptance rule for the reduced diagnostic layer.
    """
    return _solver_practically_accepted(stats, tuple(p.practical_accept_rule))


def capacity_logic(t, p: PaperParams, is_numeric: bool = False):
    """
    Reduced-model effective SE capacity with optional failure window.
    """
    exp_fn = np.exp if is_numeric else ca.exp
    K = p.K_max_total
    a = p.alpha
    base = K / (1.0 + ((1.0 - a) / max(a, 1e-12)) * exp_fn(-p.r * t))
    base = base * p.delta_eff

    if p.failure_active:
        if is_numeric:
            start = 1.0 / (1.0 + np.exp(-np.clip(10.0 * (t - p.fail_t_start), -50, 50)))
            end = 1.0 / (1.0 + np.exp(-np.clip(10.0 * (t - (p.fail_t_start + p.fail_duration)), -50, 50)))
        else:
            start = _sigma_k_paper(t - p.fail_t_start, 10.0)
            end = _sigma_k_paper(t - (p.fail_t_start + p.fail_duration), 10.0)
        return base * (1.0 - p.fail_severity * (start - end))

    return base


def build_hs_nlp(p: PaperParams, k: float):
    """
    Reduced-model Hermite-Simpson transcription over a fixed mature-regime horizon.
    """
    N = int(p.N)
    T = float(p.T)
    h = T / N
    tgrid = np.linspace(0.0, T, N + 1)

    A = ca.SX.sym("A", N + 1)
    M = ca.SX.sym("M", N + 1)
    FE = ca.SX.sym("FE", N + 1)
    FA = ca.SX.sym("FA", N + 1)
    x = ca.vertcat(A, M, FE, FA)

    idx = {
        "A": slice(0, N + 1),
        "M": slice(N + 1, 2 * N + 2),
        "FE": slice(2 * N + 2, 3 * N + 3),
        "FA": slice(3 * N + 3, 4 * N + 4),
    }

    lbx = np.zeros(4 * (N + 1))
    ubx = np.concatenate([
        np.full(N + 1, p.A_max),
        np.full(N + 1, p.M_goal * 3.0),
        np.full(N + 1, p.FE_max),
        np.full(N + 1, p.FA_max),
    ])

    g, lbg, ubg = [], [], []

    g += [A[0], M[0]]
    lbg += [0.0, 0.0]
    ubg += [0.0, 0.0]

    def flows(Ax, FEx, FAx, t):
        phi_se = capacity_logic(t, p, is_numeric=False)
        inflow = phi_se * _sigma_k_paper(p.A_max - Ax, k)
        outflow = p.L_R * FAx * _sigma_k_paper(Ax, k)
        delivered = _gk_relu_paper(p.L_R * FEx + outflow, k)
        return inflow, outflow, delivered

    def f(Ax, Mx, FEx, FAx, t):
        inflow, outflow, delivered = flows(Ax, FEx, FAx, t)
        dA = inflow - outflow
        dM = delivered
        return dA, dM

    for i in range(N):
        ti = tgrid[i]
        dA_i, dM_i = f(A[i], M[i], FE[i], FA[i], ti)
        dA_ip1, dM_ip1 = f(A[i + 1], M[i + 1], FE[i + 1], FA[i + 1], ti + h)

        A_m = 0.5 * (A[i] + A[i + 1]) + (h / 8.0) * (dA_i - dA_ip1)
        M_m = 0.5 * (M[i] + M[i + 1]) + (h / 8.0) * (dM_i - dM_ip1)
        FEm = 0.5 * (FE[i] + FE[i + 1])
        FAm = 0.5 * (FA[i] + FA[i + 1])
        dA_m, dM_m = f(A_m, M_m, FEm, FAm, ti + 0.5 * h)

        g += [A[i + 1] - A[i] - (h / 6.0) * (dA_i + 4.0 * dA_m + dA_ip1)]
        g += [M[i + 1] - M[i] - (h / 6.0) * (dM_i + 4.0 * dM_m + dM_ip1)]
        lbg += [0.0, 0.0]
        ubg += [0.0, 0.0]

    g += [M[-1] - p.M_goal]
    lbg += [0.0]
    ubg += [np.inf]

    J = 0.0
    for i in range(N):
        ti = tgrid[i]
        tm = ti + 0.5 * h

        A_m = 0.5 * (A[i] + A[i + 1])
        FEavg = 0.5 * (FE[i] + FE[i + 1])
        FAavg = 0.5 * (FA[i] + FA[i + 1])

        inflow_m, _, _ = flows(A_m, FEavg, FAavg, tm)

        cost = p.C_E * FEavg + p.C_A * FAavg + p.c_SE * inflow_m
        env = p.gamma * (p.e_E * FEavg) ** 2
        dFE_dt = (FE[i + 1] - FE[i]) / h
        dFA_dt = (FA[i + 1] - FA[i]) / h
        smooth = p.w_smooth * (dFE_dt ** 2 + dFA_dt ** 2)
        J += h * (cost + env + smooth)

    J_scale = 1e9
    nlp = {"x": x, "f": J / J_scale, "g": ca.vertcat(*g)}
    meta = {
        "tgrid": tgrid,
        "h": h,
        "N": N,
        "k": k,
        "idx": idx,
        "J_scale": J_scale,
        "lbg": np.array(lbg, dtype=float),
        "ubg": np.array(ubg, dtype=float),
    }
    return nlp, lbx, ubx, np.array(lbg), np.array(ubg), meta


def _make_solver_paper(nlp, p: PaperParams):
    """Construct IPOPT solver for reduced-model NLP."""
    opts = {
        "ipopt": {
            "print_level": 0,
            "max_iter": int(p.ipopt_max_iter),
            "tol": float(p.ipopt_tol),
            "mu_strategy": "adaptive",
            "acceptable_tol": float(p.ipopt_acceptable_tol),
            "acceptable_constr_viol_tol": float(p.ipopt_acceptable_constr_viol_tol),
            "acceptable_iter": int(p.ipopt_acceptable_iter),
            "bound_relax_factor": float(p.ipopt_bound_relax_factor),
        },
        "print_time": False,
    }
    return ca.nlpsol("solver", "ipopt", nlp, opts)


def _clamp_paper(x0: np.ndarray, lbx: np.ndarray, ubx: np.ndarray) -> np.ndarray:
    """Clamp reduced-model initial guess into variable bounds."""
    return np.clip(x0, lbx, ubx)


def _initial_guess_paper(
    p: PaperParams,
    meta: Dict[str, Any],
    lbx: np.ndarray,
    ubx: np.ndarray,
    warm_x: Optional[np.ndarray] = None,
    mode: str = "balanced",
) -> np.ndarray:
    """
    Reduced-model initial guess.

    Modes
    -----
    balanced:
        choose FA to roughly absorb mature SE inflow, then let FE fill the
        remaining mass-delivery requirement over horizon T.
    rocket_heavy:
        let FE carry most of the target.
    se_heavy:
        use a larger FA seed while keeping FE positive if needed.
    """
    if warm_x is not None and warm_x.shape[0] == lbx.shape[0] and np.all(np.isfinite(warm_x)):
        return _clamp_paper(warm_x, lbx, ubx)

    idx = meta["idx"]
    t = np.asarray(meta["tgrid"], dtype=float)
    x0 = np.zeros_like(lbx, dtype=float)

    phi = np.asarray(capacity_logic(t, p, is_numeric=True), dtype=float)
    phi = np.clip(phi, 0.0, None)

    phi_mean = float(np.mean(phi))
    phi_q75 = float(np.quantile(phi, 0.75))
    target_rate = float(p.M_goal) / max(float(p.T), 1e-12)

    fa_bal = min(float(p.FA_max), phi_mean / max(float(p.L_R), 1e-12))
    fa_hi = min(float(p.FA_max), phi_q75 / max(float(p.L_R), 1e-12))

    fe_fill_bal = max(0.0, target_rate / max(float(p.L_R), 1e-12) - fa_bal)
    fe_fill_hi = max(0.0, target_rate / max(float(p.L_R), 1e-12) - fa_hi)

    if mode == "balanced":
        FA_seed = fa_bal
        FE_seed = min(float(p.FE_max), fe_fill_bal)
    elif mode == "rocket_heavy":
        FA_seed = 0.25 * fa_bal
        FE_seed = min(
            float(p.FE_max),
            max(0.0, target_rate / max(float(p.L_R), 1e-12) - FA_seed),
        )
    elif mode == "se_heavy":
        FA_seed = fa_hi
        FE_seed = min(float(p.FE_max), fe_fill_hi)
    else:
        raise ValueError(f"Unknown reduced-model initial-guess mode: {mode}")

    delivered_rate = float(p.L_R) * (FE_seed + FA_seed)
    if delivered_rate < 0.98 * target_rate:
        extra_fe = min(
            float(p.FE_max) - FE_seed,
            max(0.0, (target_rate - delivered_rate) / max(float(p.L_R), 1e-12)),
        )
        FE_seed += extra_fe

    A_guess = np.zeros_like(t)
    for i in range(len(t) - 1):
        dt = float(t[i + 1] - t[i])
        outflow_cap = float(p.L_R) * FA_seed
        outflow = min(outflow_cap, A_guess[i] / max(dt, 1e-12) + phi[i])
        A_guess[i + 1] = np.clip(
            A_guess[i] + dt * (phi[i] - outflow),
            0.0,
            float(p.A_max),
        )

    M_guess = np.minimum(float(p.L_R) * (FE_seed + FA_seed) * t, float(p.M_goal))

    x0[idx["A"]] = A_guess
    x0[idx["M"]] = M_guess
    x0[idx["FE"]] = FE_seed
    x0[idx["FA"]] = FA_seed

    return _clamp_paper(x0, lbx, ubx)


def _pack_reduced_result(
    p: PaperParams,
    label: str,
    sol: Any,
    meta: Dict[str, Any],
    stats: Dict[str, Any],
    stage_k: float,
) -> Dict[str, Any]:
    """
    Standardized reduced-model result packing and feasibility logic.
    """
    accepted_by_ipopt, return_status = _reduced_solver_accepted(stats, p)

    x_opt = np.array(sol["x"]).flatten()
    idx = meta["idx"]
    t = meta["tgrid"]

    A = x_opt[idx["A"]]
    M = x_opt[idx["M"]]
    FE = x_opt[idx["FE"]]
    FA = x_opt[idx["FA"]]

    gval = np.array(sol["g"]).flatten()
    viol_max = _constraint_violation_abs(gval, meta["lbg"], meta["ubg"])
    terminal_gap = float(M[-1] - p.M_goal)

    feasible = bool(
        accepted_by_ipopt
        and terminal_gap >= -float(p.reduced_terminal_gap_tol_abs)
        and viol_max <= float(p.reduced_viol_tol_abs)
    )

    return {
        "ModelLayer": MODEL_LAYER_REDUCED,
        "InterpretationScope": _result_scope_note(MODEL_LAYER_REDUCED),
        "ReducedAcceptanceRule": {
            "accepted_return_status_substrings": list(tuple(p.practical_accept_rule)),
            "terminal_gap_tol_abs": float(p.reduced_terminal_gap_tol_abs),
            "viol_tol_abs": float(p.reduced_viol_tol_abs),
        },
        "label": label,
        "solver_status": return_status if return_status else "unknown",
        "solver_success_flag": bool(stats.get("success", False)),
        "accepted_by_ipopt": bool(accepted_by_ipopt),
        "solver_return_status": return_status,
        "feasible": feasible,
        "viol_max": float(viol_max),
        "terminal_gap": float(terminal_gap),
        "stage_k": float(stage_k),
        "p": p,
        "t": t,
        "A": A,
        "M": M,
        "FE": FE,
        "FA": FA,
        "obj_raw": float(sol["f"]) * float(meta["J_scale"]),
        "JE": float(metric_JE(t, FE)),
        "N_EM": float(metric_NEM(t, FE)),
        "OutputDir": OUTDIR,
        "phaseout_t": detect_phaseout_year(
            t=t,
            FE=FE,
            eps_launch_per_year=float(p.phaseout_eps),
            window_years=float(p.phaseout_window_yrs),
            relax=float(p.phaseout_relax),
        ),
    }


def solve_haat(p: PaperParams, label: str = "run") -> Dict[str, Any]:
    """
    Solve reduced-model mature-regime problem via continuation over k_schedule.

    Key behavior:
    - accept IPOPT practical statuses (not only stats()['success']),
    - preserve the best feasible continuation stage found so far,
    - try multiple initial-guess families at each stage.
    """
    p = finalize_params(p)

    warm_x: Optional[np.ndarray] = None
    best_feasible: Optional[Dict[str, Any]] = None
    best_any: Optional[Dict[str, Any]] = None

    for k in p.k_schedule:
        nlp, lbx, ubx, lbg, ubg, meta = build_hs_nlp(p, float(k))
        solver = _make_solver_paper(nlp, p)

        seed_list: List[np.ndarray] = []

        if warm_x is not None and warm_x.shape[0] == lbx.shape[0] and np.all(np.isfinite(warm_x)):
            seed_list.append(_clamp_paper(warm_x, lbx, ubx))

        seed_list.append(_initial_guess_paper(p, meta, lbx, ubx, warm_x=None, mode="balanced"))
        seed_list.append(_initial_guess_paper(p, meta, lbx, ubx, warm_x=None, mode="rocket_heavy"))
        seed_list.append(_initial_guess_paper(p, meta, lbx, ubx, warm_x=None, mode="se_heavy"))
        seed_list.append(_clamp_paper(0.7 * seed_list[0], lbx, ubx))

        stage_res: Optional[Dict[str, Any]] = None
        solved = False

        for x0 in seed_list:
            try:
                sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
                stats = solver.stats()
                stage_res = _pack_reduced_result(p, label, sol, meta, stats, float(k))
                solved = True
                break
            except Exception:
                continue

        if not solved or stage_res is None:
            if best_feasible is not None:
                break
            if best_any is not None:
                break
            continue

        x_candidate = np.array(sol["x"]).flatten()
        if np.all(np.isfinite(x_candidate)):
            warm_x = x_candidate

        best_any = stage_res
        if bool(stage_res.get("feasible", False)):
            best_feasible = stage_res

    if best_feasible is not None:
        return best_feasible

    if best_any is not None:
        return best_any

    return {
        "ModelLayer": MODEL_LAYER_REDUCED,
        "InterpretationScope": _result_scope_note(MODEL_LAYER_REDUCED),
        "ReducedAcceptanceRule": {
            "accepted_return_status_substrings": list(tuple(p.practical_accept_rule)),
            "terminal_gap_tol_abs": float(p.reduced_terminal_gap_tol_abs),
            "viol_tol_abs": float(p.reduced_viol_tol_abs),
        },
        "label": label,
        "solver_status": "crashed",
        "solver_success_flag": False,
        "accepted_by_ipopt": False,
        "solver_return_status": "exception",
        "feasible": False,
        "viol_max": np.nan,
        "terminal_gap": np.nan,
        "stage_k": np.nan,
        "p": p,
        "t": np.array([0.0]),
        "A": np.array([0.0]),
        "M": np.array([0.0]),
        "FE": np.array([0.0]),
        "FA": np.array([0.0]),
        "obj_raw": 0.0,
        "JE": np.nan,
        "N_EM": np.nan,
        "OutputDir": OUTDIR,
        "phaseout_t": None,
    }
