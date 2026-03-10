# -*- coding: utf-8 -*-
"""
MCM 2026 B: SE-Included Scenario Solver (UNIQUE-ONLY VERSION)
-------------------------------------------------------------
Keeps ONLY the unique features from your "Ultimate" script:
- scenario: mixed | se_only | rocket_only  (3 required scenarios)
- SE cost & emission included in objective (C_SE, e_SE)
- Apex outflow gating fix: A_gate
- demand-scaled initialization (better solve success)
- metric: t_build (first time reaching M_goal)
- terminal spike suppression: w_end
- one-year water ops post-processing: compute_one_year_water_ops_cost
- (optional) Tornado sensitivity (OFF by default)

Removed as "already covered elsewhere":
- heatmap module
- Monte Carlo module
- phase-out reporting/plots
- paper Fig1/Fig2 plotting
"""

import os
import warnings
from dataclasses import dataclass, replace
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import casadi as ca
import pandas as pd
import matplotlib.pyplot as plt

OUTDIR = "mcm_final_outputs"
os.makedirs(OUTDIR, exist_ok=True)
warnings.filterwarnings("ignore")

# =========================
# Run switches
# =========================
RUN_TORNADO = False   # default OFF (expensive)

# =========================
# Parameters
# =========================
@dataclass
class Params:
    # Mission
    M_goal: float = 1e8     # tons
    P_pop: float = 1e5
    w_ls: float = 1.5       # kg/person/day
    W_req: float = 0.0      # derived tons/year

    # Scenario: "mixed" | "se_only" | "rocket_only"
    scenario: str = "mixed"

    # SE throughput logistic (tons/year)
    K_max_total: float = 3.0 * 179000.0
    alpha: float = 0.02
    r: float = 0.18
    delta_eff: float = 0.974

    # Vehicles
    L_R: float = 125.0      # tons/launch
    FE_max: float = 100000.0
    FA_max: float = 200000.0

    # Apex buffer (tons)
    A_max: float = 5e5
    A_gate: float = 1e3     # NEW: outflow only if A > A_gate

    # Economics / emissions
    C_E: float = 1.0e8
    C_A: float = 2.0e7
    C_SE: float = 250.0     # $/ton Earth->Apex via SE

    e_E: float = 1275.0     # idx per Earth->Moon launch
    e_A: float = 450.0
    e_SE: float = 2.0       # idx per ton Earth->Apex

    gamma: float = 1e-4     # env penalty weight

    # Resilience toggles (keep; unique supports dual-failure)
    se_failure_active: bool = False
    fail_t_start: float = 20.0
    fail_duration: float = 3.0
    fail_severity: float = 0.90

    rocket_failure_active: bool = False
    rocket_fail_t_start: float = 22.0
    rocket_fail_duration: float = 2.0
    rocket_fail_severity: float = 0.50

    # Discretization / homotopy
    k_schedule: Tuple[float, ...] = (1.0, 5.0, 20.0, 100.0)
    T: float = 50.0
    N: int = 100

    # Regularization (keep minimal)
    w_smooth: float = 1.0
    w_end: float = 1e4      # NEW: terminal spike suppression

    # IPOPT
    ipopt_max_iter: int = 5000
    ipopt_tol: float = 1e-6
    ipopt_max_cpu_time: float = 0.0  # 0 = unlimited


def finalize_params(p: Params) -> Params:
    p.W_req = p.P_pop * p.w_ls * 365.0 / 1000.0  # tons/year
    p.scenario = str(p.scenario).lower().strip()
    if p.scenario not in ("mixed", "se_only", "rocket_only"):
        raise ValueError("scenario must be mixed|se_only|rocket_only")
    return p

# =========================
# Math helpers
# =========================
def sigma_k(z, k):
    return 1.0 / (1.0 + ca.exp(-ca.fmin(ca.fmax(k * z, -50), 50)))

def gk_relu(z, k):
    return z * sigma_k(z, k)

def _window_health(t, t0, dur, severity, k_step=10.0, is_numeric=False):
    if is_numeric:
        def num_sig(x):
            return 1.0 / (1.0 + np.exp(-np.clip(k_step * x, -50, 50)))
        w = num_sig(t - t0) - num_sig(t - (t0 + dur))
        return 1.0 - severity * w
    else:
        w = sigma_k(t - t0, k_step) - sigma_k(t - (t0 + dur), k_step)
        return 1.0 - severity * w

def capacity_se(t, p: Params, is_numeric=False):
    exp_fn = np.exp if is_numeric else ca.exp
    K = p.K_max_total
    a = max(min(p.alpha, 0.999999), 1e-6)
    base = K / (1.0 + ((1.0 - a) / a) * exp_fn(-p.r * t))
    base = base * p.delta_eff
    if p.se_failure_active:
        return base * _window_health(t, p.fail_t_start, p.fail_duration, p.fail_severity, is_numeric=is_numeric)
    return base

def rocket_health(t, p: Params, is_numeric=False):
    if not p.rocket_failure_active:
        return 1.0 if is_numeric else ca.SX(1.0)
    return _window_health(t, p.rocket_fail_t_start, p.rocket_fail_duration, p.rocket_fail_severity,
                          is_numeric=is_numeric)

# =========================
# NLP builder
# =========================
def build_hs_nlp(p: Params, k: float):
    N, T = int(p.N), float(p.T)
    h = T / N
    tgrid = np.linspace(0.0, T, N + 1)

    A = ca.SX.sym("A", N + 1)
    M = ca.SX.sym("M", N + 1)
    FE = ca.SX.sym("FE", N + 1)
    FA = ca.SX.sym("FA", N + 1)
    x = ca.vertcat(A, M, FE, FA)

    idx = {"A": slice(0, N + 1),
           "M": slice(N + 1, 2 * N + 2),
           "FE": slice(2 * N + 2, 3 * N + 3),
           "FA": slice(3 * N + 3, 4 * N + 4)}

    # scenario caps
    FE_cap, FA_cap, K_cap = float(p.FE_max), float(p.FA_max), float(p.K_max_total)
    if p.scenario == "se_only":
        FE_cap = 0.0
    elif p.scenario == "rocket_only":
        K_cap = 0.0
        FA_cap = 0.0

    K_scale = (K_cap / max(p.K_max_total, 1e-12)) if p.K_max_total > 0 else 0.0

    lbx = np.zeros(4 * (N + 1), dtype=float)
    ubx = np.concatenate([
        np.full(N + 1, p.A_max),
        np.full(N + 1, p.M_goal * 3.0),
        np.full(N + 1, FE_cap),
        np.full(N + 1, FA_cap),
    ]).astype(float)

    g, lbg, ubg = [], [], []
    g += [A[0], M[0]]
    lbg += [0.0, 0.0]
    ubg += [0.0, 0.0]

    def f(Ax, Mx, FEx, FAx, t):
        Phi_se = capacity_se(t, p, False) * K_scale
        inflow = Phi_se * sigma_k(p.A_max - Ax, k)

        # UNIQUE: A_gate fix
        gate = sigma_k(Ax - p.A_gate, k)
        outflow = p.L_R * FAx * gate

        dA = inflow - outflow

        rh = rocket_health(t, p, False)
        Phi_arrive = p.L_R * rh * (FEx + FAx)
        dM = gk_relu(Phi_arrive, k)
        return dA, dM, inflow

    for i in range(N):
        ti = tgrid[i]
        dA_i, dM_i, _ = f(A[i], M[i], FE[i], FA[i], ti)
        dA_ip1, dM_ip1, _ = f(A[i + 1], M[i + 1], FE[i + 1], FA[i + 1], ti + h)

        A_m = 0.5 * (A[i] + A[i + 1]) + (h / 8) * (dA_i - dA_ip1)
        M_m = 0.5 * (M[i] + M[i + 1]) + (h / 8) * (dM_i - dM_ip1)
        FEm = 0.5 * (FE[i] + FE[i + 1])
        FAm = 0.5 * (FA[i] + FA[i + 1])

        dA_m, dM_m, _ = f(A_m, M_m, FEm, FAm, ti + 0.5 * h)

        g += [A[i + 1] - A[i] - (h / 6) * (dA_i + 4 * dA_m + dA_ip1)]
        g += [M[i + 1] - M[i] - (h / 6) * (dM_i + 4 * dM_m + dM_ip1)]
        lbg += [0.0, 0.0]
        ubg += [0.0, 0.0]

    # terminal: meet goal
    g += [M[-1] - p.M_goal]
    lbg += [0.0]
    ubg += [np.inf]

    # objective (UNIQUE: includes SE cost & emission)
    J = 0
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

        J += h * (cost + env + smooth)

    # UNIQUE: terminal spike suppression
    if p.w_end > 0:
        J += p.w_end * (FE[-1] ** 2 + FA[-1] ** 2)

    J_scale = 1e9
    nlp = {"x": x, "f": J / J_scale, "g": ca.vertcat(*g)}
    meta = {"tgrid": tgrid, "h": h, "N": N, "k": k, "idx": idx, "J_scale": J_scale,
            "scenario_caps": {"FE_cap": FE_cap, "FA_cap": FA_cap, "K_cap": K_cap}}
    return nlp, lbx, ubx, np.array(lbg, float), np.array(ubg, float), meta

# =========================
# Metrics
# =========================
def _first_hit_time(M: np.ndarray, t: np.ndarray, goal: float) -> Optional[float]:
    if M[-1] < goal:
        return None
    j = int(np.where(M >= goal)[0][0])
    if j == 0:
        return float(t[0])
    x0, x1 = M[j - 1], M[j]
    t0, t1 = t[j - 1], t[j]
    if abs(x1 - x0) < 1e-12:
        return float(t1)
    a = (goal - x0) / (x1 - x0)
    return float(t0 + a * (t1 - t0))

def compute_one_year_water_ops_cost(p: Params, t_op: float) -> Dict[str, float]:
    W = float(p.W_req)
    se_cap = float(capacity_se(t_op, p, True))
    rh = float(rocket_health(t_op, p, True))
    tons_per_launch = float(p.L_R * rh)

    scenario = p.scenario
    FE_cap = float(p.FE_max) if scenario != "se_only" else 0.0
    FA_cap = float(p.FA_max) if scenario != "rocket_only" else 0.0
    K_cap = float(p.K_max_total) if scenario != "rocket_only" else 0.0

    fa_tons_cap = tons_per_launch * FA_cap
    chain_cap = min(se_cap * (K_cap / max(p.K_max_total, 1e-12)) if p.K_max_total > 0 else 0.0,
                    fa_tons_cap)

    x_chain = min(W, chain_cap)
    x_rem = W - x_chain

    fe_tons_cap = tons_per_launch * FE_cap
    x_FE = min(x_rem, fe_tons_cap) if FE_cap > 0 else 0.0
    unmet = max(W - x_chain - x_FE, 0.0)

    FA_launches = x_chain / max(tons_per_launch, 1e-12)
    FE_launches = x_FE / max(tons_per_launch, 1e-12)
    SE_tons = x_chain

    base_cost = (p.C_A * FA_launches) + (p.C_E * FE_launches) + (p.C_SE * SE_tons)
    env_rate = (p.e_A * FA_launches) + (p.e_E * FE_launches) + (p.e_SE * SE_tons)
    env_cost = p.gamma * (env_rate ** 2)

    return {
        "ops_feasible": bool(unmet <= 1e-9),
        "ops_unmet_tons": float(unmet),
        "ops_water_tons": float(W),
        "ops_chain_cap_tpy": float(chain_cap),
        "ops_cost_usd": float(base_cost + env_cost),
    }

def _demand_scaled_init(p: Params, meta: Dict[str, Any], x0: np.ndarray) -> np.ndarray:
    ii = meta["idx"]
    FE_cap = float(meta["scenario_caps"]["FE_cap"])
    FA_cap = float(meta["scenario_caps"]["FA_cap"])
    avg_launch = float(p.M_goal / max(p.T * p.L_R, 1e-12))

    if p.scenario == "mixed":
        fa0 = min(FA_cap, 0.8 * avg_launch)
        fe0 = min(FE_cap, 0.2 * avg_launch)
    elif p.scenario == "se_only":
        fa0 = min(FA_cap, 1.0 * avg_launch)
        fe0 = 0.0
    else:
        fa0 = 0.0
        fe0 = min(FE_cap, 1.0 * avg_launch)

    x0[ii["FA"]] = fa0
    x0[ii["FE"]] = fe0
    return x0

def solve_haat(p: Params, label: str) -> Dict[str, Any]:
    p = finalize_params(p)
    warm_x = None
    last = None
    status = {"success_steps": 0, "failed_steps": 0, "retries": 0, "last_status": "none"}

    for k in p.k_schedule:
        nlp, lbx, ubx, lbg, ubg, meta = build_hs_nlp(p, k)

        ipopt_opts = {"print_level": 0, "max_iter": int(p.ipopt_max_iter), "tol": float(p.ipopt_tol)}
        if float(p.ipopt_max_cpu_time) > 0:
            ipopt_opts["max_cpu_time"] = float(p.ipopt_max_cpu_time)
        solver = ca.nlpsol("solver", "ipopt", nlp, {"ipopt": ipopt_opts, "print_time": False})

        x0 = warm_x if warm_x is not None else np.zeros(nlp["x"].shape[0], float)
        if warm_x is None:
            x0 = _demand_scaled_init(p, meta, x0)

        tries = [np.clip(x0, lbx, ubx), np.clip(0.7 * x0, lbx, ubx), np.zeros_like(x0)]
        ok = False
        for j, xt in enumerate(tries):
            if j > 0:
                status["retries"] += 1
            try:
                sol = solver(x0=xt, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
                ok = bool(solver.stats().get("success", False))
                warm_x = np.array(sol["x"]).flatten()
                last = (sol, meta)
                status["last_status"] = "success" if ok else "failed"
                if ok:
                    break
            except Exception:
                continue

        if ok:
            status["success_steps"] += 1
        else:
            status["failed_steps"] += 1
            if last is None:
                return {"label": label, "p": p, "feasible": False, "solver_status": status}

    sol, meta = last
    x_opt = np.array(sol["x"]).flatten()
    ii = meta["idx"]

    t = meta["tgrid"]
    A = x_opt[ii["A"]]
    M = x_opt[ii["M"]]
    FE = x_opt[ii["FE"]]
    FA = x_opt[ii["FA"]]

    t_build = _first_hit_time(M, t, p.M_goal)
    ops = compute_one_year_water_ops_cost(p, t_build if t_build is not None else p.T)

    return {
        "label": label, "p": p, "t": t, "A": A, "M": M, "FE": FE, "FA": FA,
        "t_build": t_build, "obj_raw": float(sol["f"]) * meta["J_scale"],
        "feasible": True, "solver_status": status, **ops
    }

# =========================
# (Optional) Tornado
# =========================
def run_tornado_sensitivity(base_params: Params, measure: str = "obj_raw", delta: float = 0.10):
    base_res = solve_haat(base_params, "Tornado_Base")
    base_val = float(base_res.get(measure, 0.0) or 0.0)

    params_to_test = [("r","SE Growth Rate"), ("gamma","Env Penalty"), ("C_E","Earth->Moon Cost"),
                      ("C_A","Apex->Moon Cost"), ("C_SE","SE Cost/ton"),
                      ("K_max_total","SE Throughput"), ("L_R","Payload/Launch")]

    impacts = []
    for attr, name in params_to_test:
        p_low = Params(**base_params.__dict__)
        p_high = Params(**base_params.__dict__)
        v0 = getattr(base_params, attr)
        setattr(p_low, attr, max(float(v0) * (1.0 - delta), 1e-12))
        setattr(p_high, attr, max(float(v0) * (1.0 + delta), 1e-12))

        v_low = float(solve_haat(p_low, "Tornado_low").get(measure, 0.0) or 0.0)
        v_high = float(solve_haat(p_high, "Tornado_high").get(measure, 0.0) or 0.0)

        low_imp = (v_low - base_val) / base_val * 100.0 if abs(base_val) > 1e-12 else (v_low - base_val)
        high_imp = (v_high - base_val) / base_val * 100.0 if abs(base_val) > 1e-12 else (v_high - base_val)
        impacts.append((name, low_imp, high_imp, abs(high_imp - low_imp)))

    impacts.sort(key=lambda x: x[3], reverse=True)
    labels = [x[0] for x in impacts]
    lows = [x[1] for x in impacts]
    highs = [x[2] for x in impacts]

    plt.figure(figsize=(9, 6))
    y = np.arange(len(labels))
    plt.barh(y, lows, alpha=0.85, label=f"-{int(delta*100)}%")
    plt.barh(y, highs, alpha=0.85, label=f"+{int(delta*100)}%")
    plt.axvline(0, color="k")
    plt.yticks(y, labels)
    plt.xlabel("% Change in Total Objective")
    plt.title("Tornado Sensitivity (Objective)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Fig6_Tornado_Sensitivity.png"))

def export_results_table(results: List[Dict[str, Any]]):
    rows = []
    for r in results:
        p = r["p"]
        rows.append({
            "ScenarioLabel": r["label"],
            "ScenarioMode": p.scenario,
            "Gamma": p.gamma,
            "SE_Disrupt": "ON" if p.se_failure_active else "OFF",
            "Rocket_Disrupt": "ON" if p.rocket_failure_active else "OFF",
            "M(T) [Mt]": float(r["M"][-1]) / 1e6 if r.get("feasible") else 0.0,
            "BuildTime t* (yr)": f"{r['t_build']:.2f}" if r.get("t_build") is not None else "NA",
            "Obj ($B)": f"{float(r.get('obj_raw',0.0))/1e9:.2f}",
            "OpsWater (tons/yr)": f"{float(r.get('ops_water_tons',0.0)):.0f}",
            "OpsFeasible": "YES" if r.get("ops_feasible", False) else "NO",
            "OpsUnmet (tons/yr)": f"{float(r.get('ops_unmet_tons',0.0)):.0f}",
            "OpsCost ($B)": f"{float(r.get('ops_cost_usd',0.0))/1e9:.3f}",
            "A_gate (t)": f"{p.A_gate:.0f}",
            "w_end": f"{p.w_end:.2g}",
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTDIR, "final_summary_unique.csv"), index=False)
    print(df.to_string(index=False))
    print(f"[Output] {OUTDIR}/final_summary_unique.csv")

def main():
    print(">>> Running UNIQUE-ONLY SE scenario solver...")

    results = []
    results.append(solve_haat(Params(gamma=1e-4, scenario="rocket_only"), "A_RocketOnly"))
    results.append(solve_haat(Params(gamma=1e-4, scenario="se_only"), "B_SEOnly"))
    results.append(solve_haat(Params(gamma=1e-4, scenario="mixed"), "C_Mixed"))
    results.append(solve_haat(Params(gamma=1e-4, scenario="mixed", se_failure_active=True), "D_SEFail"))
    results.append(solve_haat(Params(gamma=1e-4, scenario="mixed", se_failure_active=True, rocket_failure_active=True), "E_SE+RocketFail"))

    export_results_table(results)

    if RUN_TORNADO:
        run_tornado_sensitivity(Params(gamma=1e-4, scenario="mixed"), measure="obj_raw", delta=0.10)
        print(f"[Output] {OUTDIR}/Fig6_Tornado_Sensitivity.png")

    print(">>> Done.")

if __name__ == "__main__":
    main()