# -*- coding: utf-8 -*-
"""
MCM 2026 B: SE Mixed Model + Slack-Robustified Baseline Audit / Tornado (FINAL++)
-------------------------------------------------------------------------------
Keeps ALL original features:
- Baseline credibility audit (slack + slack_share + viol_inf)
- Baseline double-check with stricter slack penalties
- FAST -> HQ auto-upgrade
- Tornado sensitivity with FAIL marking + table export + 2-panel plot

Fixes requested:
(1) Stability double-check: strict solve uses ONLY the FINAL k (no homotopy replay),
    warm-started from baseline solution => greatly reduces "basin switching".
(2) Fig 6 baseline consistency: tornado baseline is taken from a recorded baseline audit
    (baseline_for_tornado), and the figure title uses exactly that base_obj_core.
"""

import os
import json
import csv
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List, Iterable

import numpy as np
import casadi as ca
import matplotlib.pyplot as plt

# =========================================================
# 0) Global settings
# =========================================================
OUTDIR = "mcm_final_outputs"
os.makedirs(OUTDIR, exist_ok=True)
warnings.filterwarnings("ignore")

# -------------------------
# Run mode switches
# -------------------------
MODE = "baseline"      # "baseline" or "tornado"
FAST = True            # FAST attempt first; auto-upgrade if needed
DOUBLE_CHECK = True    # baseline: re-solve with stricter slack penalty

# Tornado settings
DELTA = 0.20
TOP_K: Optional[int] = None
SYMLINTHRESH = 5.0
EXCLUDE_TOP1_IN_ZOOM = True

# =========================
# Acceptance thresholds
# =========================
# Slack-based (publishable)
SLACK_NORM_TOL = 1e-5          # normalized slack norm threshold
SLACK_SHARE_TOL = 1e-4         # slack penalty share threshold
CORE_STABILITY_TOL = 0.005     # 0.5% relative change tolerance under stricter slack

# Feasibility / constraint violation thresholds
VIOL_INF_TOL = 1e-4            # for "credible_publishable"
VIOL_FILTER = 1e-3             # for candidate selection inside solve_robust


# =========================================================
# 1) Parameters
# =========================================================
@dataclass
class Params:
    # Mission
    M_goal: float = 1e8
    scenario: str = "mixed"   # mixed | se_only | rocket_only

    # SE throughput logistic
    K_max_total: float = 3.0 * 179000.0
    alpha: float = 0.02
    r: float = 0.18
    delta_eff: float = 0.974

    # Vehicles
    L_R: float = 125.0
    FE_max: float = 100000.0
    FA_max: float = 200000.0

    # Apex buffer
    A_max: float = 5e5
    A_gate: float = 1e3

    # Economics / emissions (SE included)
    C_E: float = 1.0e8
    C_A: float = 2.0e7
    C_SE: float = 250.0

    e_E: float = 1275.0
    e_A: float = 450.0
    e_SE: float = 2.0

    gamma: float = 1e-4

    # Failure toggles (off for baseline tornado)
    se_failure_active: bool = False
    fail_t_start: float = 20.0
    fail_duration: float = 3.0
    fail_severity: float = 0.90

    rocket_failure_active: bool = False
    rocket_fail_t_start: float = 22.0
    rocket_fail_duration: float = 2.0
    rocket_fail_severity: float = 0.50

    # Discretization / homotopy
    T: float = 50.0
    N: int = 100
    k_schedule: Tuple[float, ...] = (1.0, 5.0, 20.0, 100.0)

    # Regularization
    w_smooth: float = 1.0
    w_end: float = 1e4

    # Slack penalties
    w_slack_dyn: float = 1e12
    w_slack_goal: float = 1e14

    # IPOPT
    ipopt_max_iter: int = 6000
    ipopt_tol: float = 1e-6
    ipopt_max_cpu_time: float = 0.0  # 0 = unlimited


def finalize_params(p: Params) -> Params:
    p.scenario = str(p.scenario).lower().strip()
    if p.scenario not in ("mixed", "se_only", "rocket_only"):
        raise ValueError("scenario must be mixed|se_only|rocket_only")
    return p


# =========================================================
# 2) Math helpers
# =========================================================
def sigma_k(z, k):
    # logistic with clipping
    return 1.0 / (1.0 + ca.exp(-ca.fmin(ca.fmax(k * z, -50), 50)))

def gk_relu(z, k):
    return z * sigma_k(z, k)

def _window_health(t, t0, dur, severity, k_step=10.0, is_numeric=False):
    if is_numeric:
        def num_sig(x):
            return 1.0 / (1.0 + np.exp(-np.clip(k_step * x, -50, 50)))
        w = num_sig(t - t0) - num_sig(t - (t0 + dur))
        return 1.0 - severity * w
    w = sigma_k(t - t0, k_step) - sigma_k(t - (t0 + dur), k_step)
    return 1.0 - severity * w

def capacity_se(t, p: Params, is_numeric=False):
    exp_fn = np.exp if is_numeric else ca.exp
    K = p.K_max_total
    a = max(min(p.alpha, 0.999999), 1e-6)
    base = K / (1.0 + ((1.0 - a) / a) * exp_fn(-p.r * t))
    base *= p.delta_eff
    if p.se_failure_active:
        return base * _window_health(t, p.fail_t_start, p.fail_duration, p.fail_severity, is_numeric=is_numeric)
    return base

def rocket_health(t, p: Params, is_numeric=False):
    if not p.rocket_failure_active:
        return 1.0 if is_numeric else ca.SX(1.0)
    return _window_health(
        t, p.rocket_fail_t_start, p.rocket_fail_duration, p.rocket_fail_severity,
        is_numeric=is_numeric
    )


# =========================================================
# 3) NLP builder (HS + SLACK)
# =========================================================
def build_hs_nlp(p: Params, k: float):
    N, T = int(p.N), float(p.T)
    h = T / N
    tgrid = np.linspace(0.0, T, N + 1)

    A = ca.SX.sym("A", N + 1)
    M = ca.SX.sym("M", N + 1)
    FE = ca.SX.sym("FE", N + 1)
    FA = ca.SX.sym("FA", N + 1)

    sA = ca.SX.sym("sA", N)
    sM = ca.SX.sym("sM", N)
    sG = ca.SX.sym("sG", 1)  # >=0

    x = ca.vertcat(A, M, FE, FA, sA, sM, sG)

    off = 0
    sl_A = slice(off, off + (N + 1)); off += (N + 1)
    sl_M = slice(off, off + (N + 1)); off += (N + 1)
    sl_FE = slice(off, off + (N + 1)); off += (N + 1)
    sl_FA = slice(off, off + (N + 1)); off += (N + 1)
    sl_sA = slice(off, off + N); off += N
    sl_sM = slice(off, off + N); off += N
    sl_sG = slice(off, off + 1); off += 1

    # scenario caps
    FE_cap, FA_cap, K_cap = float(p.FE_max), float(p.FA_max), float(p.K_max_total)
    if p.scenario == "se_only":
        FE_cap = 0.0
    elif p.scenario == "rocket_only":
        K_cap = 0.0
        FA_cap = 0.0
    K_scale = (K_cap / max(p.K_max_total, 1e-12)) if p.K_max_total > 0 else 0.0

    big = 1e12
    lbx = np.concatenate([
        np.zeros(N + 1),                  # A >= 0
        np.zeros(N + 1),                  # M >= 0
        np.zeros(N + 1),                  # FE >= 0
        np.zeros(N + 1),                  # FA >= 0
        -big * np.ones(N),                # sA free
        -big * np.ones(N),                # sM free
        np.zeros(1),                      # sG >= 0
    ]).astype(float)

    ubx = np.concatenate([
        np.full(N + 1, p.A_max),
        np.full(N + 1, p.M_goal * 3.0),
        np.full(N + 1, FE_cap),
        np.full(N + 1, FA_cap),
        big * np.ones(N),
        big * np.ones(N),
        big * np.ones(1),
    ]).astype(float)

    g, lbg, ubg = [], [], []

    # initial
    g += [A[0], M[0]]
    lbg += [0.0, 0.0]
    ubg += [0.0, 0.0]

    def f(Ax, Mx, FEx, FAx, t):
        # SE inflow to Apex
        Phi_se = capacity_se(t, p, False) * K_scale
        inflow = Phi_se * sigma_k(p.A_max - Ax, k)

        # outflow gate: requires A > A_gate and A > 0
        gate = sigma_k(Ax - p.A_gate, k)
        avail = sigma_k(Ax, k)
        outflow = p.L_R * FAx * gate * avail
        dA = inflow - outflow

        # delivery to Moon
        rh = rocket_health(t, p, False)
        Phi_arrive = p.L_R * rh * (FEx + FAx)
        dM = gk_relu(Phi_arrive, k)
        return dA, dM

    # HS with slack
    for i in range(N):
        ti = tgrid[i]
        dA_i, dM_i = f(A[i], M[i], FE[i], FA[i], ti)
        dA_ip1, dM_ip1 = f(A[i + 1], M[i + 1], FE[i + 1], FA[i + 1], ti + h)

        A_m = 0.5 * (A[i] + A[i + 1]) + (h / 8) * (dA_i - dA_ip1)
        M_m = 0.5 * (M[i] + M[i + 1]) + (h / 8) * (dM_i - dM_ip1)
        FEm = 0.5 * (FE[i] + FE[i + 1])
        FAm = 0.5 * (FA[i] + FA[i + 1])
        dA_m, dM_m = f(A_m, M_m, FEm, FAm, ti + 0.5 * h)

        dynA = A[i + 1] - A[i] - (h / 6) * (dA_i + 4 * dA_m + dA_ip1)
        dynM = M[i + 1] - M[i] - (h / 6) * (dM_i + 4 * dM_m + dM_ip1)

        g += [dynA - sA[i], dynM - sM[i]]
        lbg += [0.0, 0.0]
        ubg += [0.0, 0.0]

    # terminal goal with nonnegative slack
    g += [M[-1] - p.M_goal + sG[0]]
    lbg += [0.0]
    ubg += [np.inf]

    # objective
    J_core = 0
    for i in range(N):
        t_mid = 0.5 * (tgrid[i] + tgrid[i + 1])
        FEavg = 0.5 * (FE[i] + FE[i + 1])
        FAavg = 0.5 * (FA[i] + FA[i + 1])
        Aavg = 0.5 * (A[i] + A[i + 1])

        Phi_se_mid = capacity_se(t_mid, p, False) * K_scale
        inflow_mid = Phi_se_mid * sigma_k(p.A_max - Aavg, k)

        cost = p.C_E * FEavg + p.C_A * FAavg + p.C_SE * inflow_mid
        env_rate = p.e_E * FEavg + p.e_A * FAavg + p.e_SE * inflow_mid
        env = p.gamma * (env_rate ** 2)

        dFE_dt = (FE[i + 1] - FE[i]) / h
        dFA_dt = (FA[i + 1] - FA[i]) / h
        smooth = p.w_smooth * (dFE_dt ** 2 + dFA_dt ** 2)

        J_core += h * (cost + env + smooth)

    if p.w_end > 0:
        J_core += p.w_end * (FE[-1] ** 2 + FA[-1] ** 2)

    J_slack = p.w_slack_dyn * (ca.dot(sA, sA) + ca.dot(sM, sM)) + p.w_slack_goal * (sG[0] ** 2)
    J_total = J_core + J_slack

    # scaling: objective is ~O(1e13), scale to ~O(1-10)
    J_scale = 1e13
    nlp = {"x": x, "f": J_total / J_scale, "g": ca.vertcat(*g)}

    core_fun = ca.Function("core_fun", [x], [J_core])
    slack_fun = ca.Function("slack_fun", [x], [J_slack])

    meta = {
        "tgrid": tgrid, "h": h, "N": N, "k": k, "J_scale": J_scale,
        "slices": {"A": sl_A, "M": sl_M, "FE": sl_FE, "FA": sl_FA, "sA": sl_sA, "sM": sl_sM, "sG": sl_sG},
        "core_fun": core_fun, "slack_fun": slack_fun,
    }
    return nlp, lbx, ubx, np.array(lbg, float), np.array(ubg, float), meta


# =========================================================
# 4) Robust solve
# =========================================================
def _constraint_violation(gval: np.ndarray, lbg: np.ndarray, ubg: np.ndarray) -> float:
    low = np.maximum(lbg - gval, 0.0)
    high = np.maximum(gval - ubg, 0.0)
    return float(np.max(low + high)) if gval.size else 0.0

def _demand_scaled_init(p: Params, dim: int) -> np.ndarray:
    """
    Creates a simple initial guess biased by "average launch rate needed".
    Layout: A,M,FE,FA,sA,sM,sG
    """
    x0 = np.zeros(dim, float)
    N = int(p.N)

    off_A = 0
    off_M = off_A + (N + 1)
    off_FE = off_M + (N + 1)
    off_FA = off_FE + (N + 1)

    avg_launch = float(p.M_goal / max(p.T * p.L_R, 1e-12))
    if p.scenario == "mixed":
        fe0, fa0 = 0.2 * avg_launch, 0.8 * avg_launch
    elif p.scenario == "se_only":
        fe0, fa0 = 0.0, 1.0 * avg_launch
    else:
        fe0, fa0 = 1.0 * avg_launch, 0.0

    x0[off_FE:off_FE + (N + 1)] = fe0
    x0[off_FA:off_FA + (N + 1)] = fa0
    return x0

def _make_solver(p: Params, nlp: Dict[str, Any]):
    ipopt_opts = {
        "print_level": 0,
        "max_iter": int(p.ipopt_max_iter),
        "tol": float(p.ipopt_tol),
        "mu_strategy": "adaptive",
        "acceptable_tol": 1e-5,
        "acceptable_iter": 10,
        "expect_infeasible_problem": "yes",
    }
    if float(p.ipopt_max_cpu_time) > 0:
        ipopt_opts["max_cpu_time"] = float(p.ipopt_max_cpu_time)
    return ca.nlpsol("solver", "ipopt", nlp, {"ipopt": ipopt_opts, "print_time": False})

def solve_robust(
    p: Params,
    verbose: bool = False,
    x0_override: Optional[np.ndarray] = None,
    k_schedule_override: Optional[Iterable[float]] = None
) -> Dict[str, Any]:
    """
    Homotopy over k_schedule (or k_schedule_override). At each k, tries multiple x0 seeds.
    Selection: feasible-first (viol_inf <= VIOL_FILTER), then viol, then slack, then obj_total.

    - x0_override: warm-start vector (used by strict check & tornado for consistency).
    - k_schedule_override: allow "final-k-only" solves (used to fix stability check).
    """
    p = finalize_params(p)
    warm_x = None
    best = None
    rng = np.random.default_rng(0)

    ks = tuple(k_schedule_override) if k_schedule_override is not None else tuple(p.k_schedule)
    x0_override_local = None if x0_override is None else np.array(x0_override, dtype=float).flatten()

    def better(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
        # 1) feasible-first
        if a["feasible"] != b["feasible"]:
            return a["feasible"] and (not b["feasible"])
        # 2) smaller violation
        if a["viol_inf"] != b["viol_inf"]:
            return a["viol_inf"] < b["viol_inf"]
        # 3) smaller slack
        if a["slack_norm_inf"] != b["slack_norm_inf"]:
            return a["slack_norm_inf"] < b["slack_norm_inf"]
        # 4) smaller objective
        return a["obj_total"] < b["obj_total"]

    for k in ks:
        nlp, lbx, ubx, lbg, ubg, meta = build_hs_nlp(p, float(k))
        solver = _make_solver(p, nlp)
        dim = int(nlp["x"].shape[0])

        # choose base initial guess
        if warm_x is not None and warm_x.size == dim:
            base_x0 = warm_x
        elif x0_override_local is not None and x0_override_local.size == dim:
            base_x0 = x0_override_local
        else:
            base_x0 = _demand_scaled_init(p, dim)

        base_x0 = np.clip(base_x0, lbx, ubx)
        jitter = 0.01 * rng.standard_normal(size=base_x0.shape)

        tries = [
            base_x0,
            np.clip(0.7 * base_x0, lbx, ubx),
            np.zeros_like(base_x0),
            np.clip(base_x0 + jitter, lbx, ubx),
        ]

        step_best = None
        for xt in tries:
            try:
                sol = solver(x0=xt, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
                stats = solver.stats()
                ok = bool(stats.get("success", False))
                status = str(stats.get("return_status", ""))
                iters = int(stats.get("iter_count", -1))

                x_opt = np.array(sol["x"]).flatten()
                g_opt = np.array(sol["g"]).flatten()
                viol = _constraint_violation(g_opt, lbg, ubg)

                sl = meta["slices"]
                sA = x_opt[sl["sA"]]
                sM = x_opt[sl["sM"]]
                sG = x_opt[sl["sG"]][0]

                A_scale = max(p.A_max, 1.0)
                M_scale = max(p.M_goal, 1.0)
                slack_norm = max(
                    float(np.max(np.abs(sA)) / A_scale) if sA.size else 0.0,
                    float(np.max(np.abs(sM)) / M_scale) if sM.size else 0.0,
                    float(sG / M_scale),
                )

                obj_total = float(sol["f"]) * meta["J_scale"]
                obj_core = float(meta["core_fun"](x_opt))
                obj_slack = float(meta["slack_fun"](x_opt))

                feasible = bool(np.isfinite(viol) and (viol <= VIOL_FILTER))

                cand = {
                    "x": x_opt,
                    "success": ok,
                    "return_status": status,
                    "iter_count": iters,
                    "viol_inf": float(viol),
                    "feasible": feasible,
                    "slack_norm_inf": float(slack_norm),
                    "obj_total": float(obj_total),
                    "obj_core": float(obj_core),
                    "obj_slack": float(obj_slack),
                    "meta": meta,
                }

                if (step_best is None) or better(cand, step_best):
                    step_best = cand

            except Exception:
                continue

        if step_best is None:
            continue

        # update warm start only if feasible or solver succeeded
        if step_best["feasible"] or step_best["success"]:
            warm_x = step_best["x"]

        if (best is None) or better(step_best, best):
            best = step_best

        if verbose and best is not None:
            print(
                f"  k={k:<6g} feas={best['feasible']} slack_norm={best['slack_norm_inf']:.2e} "
                f"viol={best['viol_inf']:.2e} obj_core={best['obj_core']:.3e} "
                f"success={best['success']} status={best['return_status']}"
            )

    if best is None:
        return {"ok": False, "reason": "no_candidate"}

    best["ok"] = True
    return best


# =========================================================
# 5) Baseline audit addon
# =========================================================
def _write_json(path: str, d: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)

def _write_csv_row(path: str, d: Dict[str, Any]):
    keys = list(d.keys())
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if not file_exists:
            w.writeheader()
        w.writerow(d)

def baseline_audit(
    p: Params,
    tag: str = "baseline",
    x0_override: Optional[np.ndarray] = None,
    k_schedule_override: Optional[Iterable[float]] = None
) -> Dict[str, Any]:
    """
    Publishable credibility:
      - viol_inf <= VIOL_INF_TOL
      - slack norms <= SLACK_NORM_TOL
      - slack_share <= SLACK_SHARE_TOL
    """
    res = solve_robust(p, verbose=True, x0_override=x0_override, k_schedule_override=k_schedule_override)
    if not res.get("ok", False):
        raise RuntimeError("Baseline solve failed: no candidate.")

    x = res["x"]
    meta = res["meta"]
    sl = meta["slices"]
    sA = x[sl["sA"]]
    sM = x[sl["sM"]]
    sG = float(x[sl["sG"]][0])

    A_scale = max(p.A_max, 1.0)
    M_scale = max(p.M_goal, 1.0)
    max_sA = float(np.max(np.abs(sA))) if sA.size else 0.0
    max_sM = float(np.max(np.abs(sM))) if sM.size else 0.0

    slack_A_norm = max_sA / A_scale
    slack_M_norm = max_sM / M_scale
    slack_G_norm = sG / M_scale

    obj_core = float(res["obj_core"])
    obj_slack = float(res["obj_slack"])
    slack_share = obj_slack / max(obj_core, 1e-12)

    viol_inf = float(res["viol_inf"])

    credible = (
        (viol_inf <= VIOL_INF_TOL) and
        (slack_A_norm <= SLACK_NORM_TOL) and
        (slack_M_norm <= SLACK_NORM_TOL) and
        (slack_G_norm <= SLACK_NORM_TOL) and
        (slack_share <= SLACK_SHARE_TOL)
    )

    report = {
        "tag": tag,
        "scenario": p.scenario,
        "N": int(p.N),
        "k_schedule": list(k_schedule_override) if k_schedule_override is not None else list(p.k_schedule),

        "success": bool(res["success"]),
        "return_status": str(res.get("return_status", "")),
        "iter_count": int(res.get("iter_count", -1)),

        "feasible_filter": bool(res.get("feasible", False)),
        "viol_inf": viol_inf,
        "VIOL_INF_TOL": float(VIOL_INF_TOL),

        "slack_norm_inf": float(res["slack_norm_inf"]),
        "slack_A_norm": float(slack_A_norm),
        "slack_M_norm": float(slack_M_norm),
        "slack_G_norm": float(slack_G_norm),

        "obj_core": obj_core,
        "obj_slack": obj_slack,
        "slack_share": float(slack_share),

        "credible_publishable": bool(credible),

        "w_slack_dyn": float(p.w_slack_dyn),
        "w_slack_goal": float(p.w_slack_goal),
        "SLACK_NORM_TOL": float(SLACK_NORM_TOL),
        "SLACK_SHARE_TOL": float(SLACK_SHARE_TOL),
        "VIOL_FILTER": float(VIOL_FILTER),
    }

    _write_json(os.path.join(OUTDIR, f"{tag}_audit.json"), report)
    csv_path = os.path.join(OUTDIR, f"{tag}_audit.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)
    _write_csv_row(csv_path, report)

    print("\n[Baseline Audit]")
    for k_, v_ in report.items():
        if isinstance(v_, float):
            print(f"  {k_:<20s}: {v_:.6e}")
        else:
            print(f"  {k_:<20s}: {v_}")

    if not credible:
        raise RuntimeError(
            f"Baseline NOT credible: viol_inf={viol_inf:.3e} (tol {VIOL_INF_TOL:.1e}), "
            f"slack_norm_inf={report['slack_norm_inf']:.3e}, slack_share={slack_share:.3e}"
        )

    # keep solution vector only in returned dict (NOT written to json/csv)
    report["_x_solution"] = x
    return report

def baseline_double_check(p: Params) -> Dict[str, Any]:
    """
    FIXED stability check:
    - baseline: full k_schedule (homotopy)
    - strict:   FINAL k only (no replay of early k), warm-started from baseline x*
    """
    rep1 = baseline_audit(p, tag="baseline")
    x_baseline = rep1.get("_x_solution", None)

    # strict penalties
    p2 = Params(**p.__dict__)
    p2.w_slack_dyn *= 10.0
    p2.w_slack_goal *= 10.0

    # FINAL-k-only solve to reduce basin switching (fix #1)
    k_last = float(p.k_schedule[-1])
    rep2 = baseline_audit(
        p2,
        tag="baseline_strict",
        x0_override=x_baseline,
        k_schedule_override=(k_last,)
    )

    # remove internal large arrays before returning
    rep1.pop("_x_solution", None)
    rep2.pop("_x_solution", None)

    rel_change = abs(rep2["obj_core"] - rep1["obj_core"]) / max(rep1["obj_core"], 1e-12)
    stable = (rel_change <= CORE_STABILITY_TOL)

    summary = {
        "rel_change_core_obj": float(rel_change),
        "stable_under_stricter_slack": bool(stable),
        "CORE_STABILITY_TOL": float(CORE_STABILITY_TOL),
        "strict_check_note": "baseline_strict uses final-k-only (no homotopy replay) warm-started from baseline",
        "final_k_used": k_last,
    }
    _write_json(os.path.join(OUTDIR, "baseline_double_check.json"), summary)

    print(f"\n[Baseline Stability] core objective relative change = {rel_change*100:.6f}%")
    print(f"[Baseline Stability] stable_under_stricter_slack = {'YES' if stable else 'NO'}")
    return {"baseline": rep1, "strict": rep2, "summary": summary}


# =========================================================
# 6) Tornado (feasible-only, with audit table)
# =========================================================
def run_tornado(
    base_params: Params,
    delta: float,
    top_k: Optional[int],
    base_val: Optional[float] = None,
    base_x: Optional[np.ndarray] = None
) -> Tuple[float, List[Dict[str, Any]]]:
    params_to_test = [
        ("L_R", "Payload/Launch"),
        ("C_E", "Earth->Moon Cost"),
        ("K_max_total", "SE Throughput"),
        ("r", "SE Growth Rate"),
        ("C_A", "Apex->Moon Cost"),
        ("gamma", "Env Penalty"),
        ("C_SE", "SE Unit Cost"),
    ]
    if top_k is not None:
        params_to_test = params_to_test[:top_k]

    # If base not supplied, solve it (still allowed)
    if base_val is None or base_x is None:
        print("[Tornado] base solve ...")
        base_res = solve_robust(base_params, verbose=True)
        if (not base_res.get("ok", False)) or (not base_res.get("feasible", False)):
            raise RuntimeError(
                f"Base solve infeasible: viol_inf={base_res.get('viol_inf', np.inf):.3e}, "
                f"VIOL_FILTER={VIOL_FILTER:.1e}. Upgrade settings first."
            )
        base_val = float(base_res["obj_core"])
        base_x = base_res.get("x", None)
        print(
            f"[Tornado] base obj_core={base_val:.4e} slack_norm={base_res['slack_norm_inf']:.2e} "
            f"viol={base_res['viol_inf']:.2e}"
        )
    else:
        print(f"[Tornado] using audited base obj_core={base_val:.4e} (consistent with baseline_for_tornado)")

    def safe_scale(v, s):
        return max(float(v) * float(s), 1e-12)

    rows = []
    for attr, name in params_to_test:
        v0 = getattr(base_params, attr)
        p_low = Params(**base_params.__dict__)
        p_high = Params(**base_params.__dict__)
        setattr(p_low, attr, safe_scale(v0, 1.0 - delta))
        setattr(p_high, attr, safe_scale(v0, 1.0 + delta))

        # warm-start perturbed solves from base solution for comparability
        r_low = solve_robust(p_low, verbose=False, x0_override=base_x)
        r_high = solve_robust(p_high, verbose=False, x0_override=base_x)

        def pct(res):
            if (not res.get("ok", False)) or (not res.get("feasible", False)):
                return np.nan
            val = float(res["obj_core"])
            return (val - base_val) / base_val * 100.0

        low_pct = pct(r_low)
        high_pct = pct(r_high)

        span = float(np.nanmax([abs(low_pct), abs(high_pct)])) if np.any(np.isfinite([low_pct, high_pct])) else np.nan

        rows.append({
            "name": name, "attr": attr,
            "low_pct": low_pct, "high_pct": high_pct, "span": span,

            # feasibility metrics for appendix / audit
            "low_viol": float(r_low.get("viol_inf", np.inf)),
            "high_viol": float(r_high.get("viol_inf", np.inf)),
            "low_slack": float(r_low.get("slack_norm_inf", np.inf)),
            "high_slack": float(r_high.get("slack_norm_inf", np.inf)),
            "low_feasible": bool(r_low.get("feasible", False)),
            "high_feasible": bool(r_high.get("feasible", False)),
            "low_status": str(r_low.get("return_status", "")),
            "high_status": str(r_high.get("return_status", "")),
        })

        lo_s = "FAIL" if (not np.isfinite(low_pct)) else f"{low_pct:+.2f}%"
        hi_s = "FAIL" if (not np.isfinite(high_pct)) else f"{high_pct:+.2f}%"
        print(f"  - {name:<16s} low={lo_s:<10s} high={hi_s:<10s}")

    rows.sort(key=lambda d: (-np.nan_to_num(d["span"], nan=-1e18)))

    # save tornado table
    out_json = os.path.join(OUTDIR, "tornado_table.json")
    _write_json(out_json, {"delta": delta, "base_obj_core": float(base_val), "rows": rows})

    out_csv = os.path.join(OUTDIR, "tornado_table.csv")
    if os.path.exists(out_csv):
        os.remove(out_csv)
    for r in rows:
        _write_csv_row(out_csv, r)

    return float(base_val), rows

def plot_tornado(base_val: float, rows: List[Dict[str, Any]], delta: float):
    labels = [r["name"] for r in rows]
    lows = [r["low_pct"] if np.isfinite(r["low_pct"]) else np.nan for r in rows]
    highs = [r["high_pct"] if np.isfinite(r["high_pct"]) else np.nan for r in rows]

    rows_zoom = rows.copy()
    if EXCLUDE_TOP1_IN_ZOOM and len(rows_zoom) >= 2:
        rows_zoom = rows_zoom[1:]
    labels_z = [r["name"] for r in rows_zoom]
    lows_z = [r["low_pct"] if np.isfinite(r["low_pct"]) else np.nan for r in rows_zoom]
    highs_z = [r["high_pct"] if np.isfinite(r["high_pct"]) else np.nan for r in rows_zoom]

    finite_vals = [abs(v) for v in lows_z + highs_z if np.isfinite(v)]
    zoom_lim = max(10.0, (max(finite_vals) * 1.2 if finite_vals else 10.0))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw={"height_ratios": [2.0, 1.4]})
    fig.subplots_adjust(left=0.24)

    # Full symlog
    y = np.arange(len(labels))
    ax1.barh(y, lows, alpha=0.75, label=f"Param -{int(delta*100)}%")
    ax1.barh(y, highs, alpha=0.75, label=f"Param +{int(delta*100)}%")
    ax1.axvline(0, color="k", linewidth=1)
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax1.set_xscale("symlog", linthresh=SYMLINTHRESH)
    ax1.set_xlabel("% Change in Objective (symlog)")
    ax1.set_title(f"Fig 6: Tornado Sensitivity (Objective)\nBaseline obj_core = {base_val:.3e}")
    ax1.legend(loc="best")

    for i, (lo, hi) in enumerate(zip(lows, highs)):
        for val, dx in [(lo, -1), (hi, +1)]:
            if np.isfinite(val):
                ax1.text(val + dx * 0.8, i, f"{val:+.1f}%",
                         va="center", ha="left" if dx > 0 else "right", fontsize=9)
            else:
                ax1.text(0, i, "FAIL", va="center", ha="center", fontsize=9)

    # Zoom
    yz = np.arange(len(labels_z))
    ax2.barh(yz, lows_z, alpha=0.75, label=f"-{int(delta*100)}%")
    ax2.barh(yz, highs_z, alpha=0.75, label=f"+{int(delta*100)}%")
    ax2.axvline(0, color="k", linewidth=1)
    ax2.set_yticks(yz)
    ax2.set_yticklabels(labels_z)
    ax2.invert_yaxis()
    ax2.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax2.set_xlim(-zoom_lim, zoom_lim)
    ax2.set_xlabel("% Change in Objective (zoom)")

    for i, (lo, hi) in enumerate(zip(lows_z, highs_z)):
        for val, dx in [(lo, -1), (hi, +1)]:
            if np.isfinite(val):
                ax2.text(val + dx * 0.6, i, f"{val:+.1f}%",
                         va="center", ha="left" if dx > 0 else "right", fontsize=9)
            else:
                ax2.text(0, i, "FAIL", va="center", ha="center", fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUTDIR, "Fig6_Tornado_Sensitivity.png")
    fig.savefig(out, dpi=220)
    print(f"[Output] {out}")


# =========================================================
# 7) Main
# =========================================================
def main():
    print(">>> MCM SE model (Slack-Robust)")

    # Baseline parameters (mixed)
    p0 = Params(gamma=1e-4, scenario="mixed")

    # FAST attempt config
    if FAST:
        p0.N = 40
        p0.k_schedule = (1.0, 5.0, 20.0)
        p0.ipopt_max_cpu_time = 25.0
        p0.ipopt_max_iter = 3000
        print(f"[FAST] N={p0.N}, k={p0.k_schedule}, cpu={p0.ipopt_max_cpu_time}s")

    def make_hq() -> Params:
        p1 = Params(gamma=1e-4, scenario="mixed")
        p1.N = 120
        p1.k_schedule = (1.0, 3.0, 10.0, 30.0, 80.0)
        p1.ipopt_max_cpu_time = 0.0
        p1.ipopt_max_iter = 9000
        p1.w_slack_dyn = 5e12
        p1.w_slack_goal = 5e14
        return p1

    if MODE == "baseline":
        try:
            if DOUBLE_CHECK:
                baseline_double_check(p0)
            else:
                baseline_audit(p0, tag="baseline")
        except Exception as e:
            print(f"[WARN] baseline FAST failed/unstable: {e}")
            print("[UPGRADE] switching to HQ settings ...")
            p1 = make_hq()
            print(f"[HQ] N={p1.N}, k={p1.k_schedule}, cpu=unlimited")
            if DOUBLE_CHECK:
                baseline_double_check(p1)
            else:
                baseline_audit(p1, tag="baseline")
        print(">>> Done (baseline mode).")
        return

    if MODE == "tornado":
        try:
            # FIX #2: generate/record a baseline specifically for tornado; figure will match it
            rep_base = baseline_audit(p0, tag="baseline_for_tornado")
            base_val = float(rep_base["obj_core"])
            base_x = np.array(rep_base["_x_solution"], dtype=float).flatten()

            base_val, rows = run_tornado(p0, delta=DELTA, top_k=TOP_K, base_val=base_val, base_x=base_x)
            plot_tornado(base_val, rows, delta=DELTA)

        except Exception as e:
            print(f"[WARN] tornado FAST failed: {e}")
            print("[UPGRADE] switching to HQ settings ...")
            p1 = make_hq()

            rep_base = baseline_audit(p1, tag="baseline_for_tornado")
            base_val = float(rep_base["obj_core"])
            base_x = np.array(rep_base["_x_solution"], dtype=float).flatten()

            base_val, rows = run_tornado(p1, delta=DELTA, top_k=TOP_K, base_val=base_val, base_x=base_x)
            plot_tornado(base_val, rows, delta=DELTA)

        print(">>> Done (tornado mode).")
        return

    raise ValueError("MODE must be 'baseline' or 'tornado'.")


if __name__ == "__main__":
    main()