# -*- coding: utf-8 -*-
"""
MCM 2026 Problem B: O-Prize Submission Core (Core + Sensitivity)
---------------------------------------------------------------
PURPOSE (Run once for paper figures):
1) Dynamic Model + HS Collocation NLP
2) Robust Homotopy Solve
3) Sensitivity Heatmap (Phase-out Year)

Outputs -> mcm_final_outputs/
  - Fig1_OptimalSwitching.png
  - Fig2_Resilience.png
  - Fig4_Sensitivity_Heatmap.png
  - final_summary.csv
  - final_summary.tex
"""

import sys
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np

print(f"[Env] Python: {sys.executable}")

try:
    import casadi as ca
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    print("\n[CRITICAL ERROR] Missing libraries.")
    print("Please run: pip install casadi pandas matplotlib seaborn")
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


# ==========================================
# 1. Params
# ==========================================
@dataclass
class Params:
    # Mission
    M_goal: float = 1e8
    P_pop: float = 1e5
    w_ls: float = 1.5
    W_req: float = 0.0  # derived

    # SE
    K_max_total: float = 1.5e7
    alpha: float = 0.15
    r: float = 0.18
    delta_eff: float = 0.974

    # Failure
    failure_active: bool = False
    fail_t_start: float = 20.0
    fail_duration: float = 3.0
    fail_severity: float = 0.90

    # Rockets / Shuttles
    L_R: float = 125.0
    FE_max: float = 40000.0
    FA_max: float = 200000.0  # >= 25 Mt/yr

    # Buffer
    A_max: float = 5e5

    # Costs & Policy
    C_E: float = 1.0e8
    C_A: float = 1.0e7
    e_E: float = 1275.0
    gamma: float = 0.08  # Policy scenario uses larger gamma

    # Solver / Discretization
    k_schedule: Tuple[float, ...] = (1.0, 5.0, 20.0, 100.0)
    T: float = 50.0
    N: int = 100

    # Smoothing (rate penalization)
    w_smooth: float = 50.0

    # Phase-out
    phaseout_eps: float = 10.0
    phaseout_m: int = 4  # confirm window

    # IPOPT
    ipopt_max_iter: int = 4000
    ipopt_tol: float = 1e-6


def finalize_params(p: Params) -> Params:
    p.W_req = p.P_pop * p.w_ls * 365.0 / 1000.0
    return p


# ==========================================
# 2. Core math
# ==========================================
def sigma_k(z, k):
    return 1.0 / (1.0 + ca.exp(-ca.fmin(ca.fmax(k * z, -50), 50)))

def gk_relu(z, k):
    return z * sigma_k(z, k)

def capacity_logic(t, p: Params, is_numeric=False):
    exp_fn = np.exp if is_numeric else ca.exp
    K = p.K_max_total
    a = p.alpha
    base = K / (1.0 + ((1.0 - a) / a) * exp_fn(-p.r * t))
    base = base * p.delta_eff

    if p.failure_active:
        k_step = 10.0
        if is_numeric:
            def num_sig(x):
                return 1.0 / (1.0 + np.exp(-np.clip(k_step * x, -50, 50)))
            start = num_sig(t - p.fail_t_start)
            end = num_sig(t - (p.fail_t_start + p.fail_duration))
        else:
            start = sigma_k(t - p.fail_t_start, k_step)
            end = sigma_k(t - (p.fail_t_start + p.fail_duration), k_step)
        window = start - end
        health = 1.0 - p.fail_severity * window
        return base * health
    return base


# ==========================================
# 3. NLP build
# ==========================================
def build_hs_nlp(p: Params, k: float):
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
    g += [A[0], M[0]]; lbg += [0.0, 0.0]; ubg += [0.0, 0.0]

    def f(Ax, Mx, FEx, FAx, t):
        Phi_se = capacity_logic(t, p, is_numeric=False)
        inflow = Phi_se * sigma_k(p.A_max - Ax, k)
        outflow = p.L_R * FAx * sigma_k(Ax, k)
        dA = inflow - outflow

        Phi_arrive = p.L_R * (FEx + FAx)
        dM = gk_relu(Phi_arrive - p.W_req, k)
        return dA, dM

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
        lbg += [0.0, 0.0]; ubg += [0.0, 0.0]

    # meet goal
    g += [M[-1] - p.M_goal]; lbg += [0.0]; ubg += [ca.inf]

    # objective (rate smoothing)
    J = 0
    for i in range(N):
        FEavg = 0.5 * (FE[i] + FE[i + 1])
        FAavg = 0.5 * (FA[i] + FA[i + 1])

        cost = p.C_E * FEavg + p.C_A * FAavg
        env = p.gamma * (p.e_E * FEavg) ** 2

        dFE_dt = (FE[i + 1] - FE[i]) / h
        dFA_dt = (FA[i + 1] - FA[i]) / h
        smooth = p.w_smooth * (dFE_dt ** 2 + dFA_dt ** 2)

        J += h * (cost + env + smooth)

    J_scale = 1e9
    nlp = {"x": x, "f": J / J_scale, "g": ca.vertcat(*g)}
    meta = {"tgrid": tgrid, "h": h, "N": N, "k": k, "idx": idx, "J_scale": J_scale}
    return nlp, lbx, ubx, np.array(lbg), np.array(ubg), meta


# ==========================================
# 4. Solve + phaseout (m consecutive points)
# ==========================================
def _detect_phaseout(FE: np.ndarray, t: np.ndarray, eps: float, m: int) -> Dict[str, Any]:
    below = FE <= eps
    run = 0
    for i, ok in enumerate(below):
        run = run + 1 if ok else 0
        if run >= m:
            idx = i - m + 1
            return {"phaseout_idx": idx, "phaseout_t": float(t[idx]), "phaseout_confirm_window": m}
    return {"phaseout_idx": None, "phaseout_t": None, "phaseout_confirm_window": m}


def solve_haat(p: Params, label: str) -> Dict[str, Any]:
    p = finalize_params(p)
    warm_x = None
    last_res = None
    quiet = ("Sens" in label) or ("Heatmap" in label)

    if not quiet:
        print(f"Solving {label}...", end=" ")

    for k in p.k_schedule:
        nlp, lbx, ubx, lbg, ubg, meta = build_hs_nlp(p, k)

        opts = {
            "ipopt": {
                "print_level": 0,
                "max_iter": int(p.ipopt_max_iter),
                "tol": float(p.ipopt_tol),
                "mu_strategy": "adaptive",
            },
            "print_time": False
        }
        solver = ca.nlpsol("solver", "ipopt", nlp, opts)

        x0 = warm_x if warm_x is not None else np.zeros(nlp["x"].shape[0])
        if warm_x is None:
            idx = meta["idx"]
            x0[idx["FE"]] = p.FE_max * 0.5
            x0[idx["FA"]] = p.FA_max * 0.2

        x0 = np.clip(x0, lbx, ubx)

        try:
            sol = solver(x0=x0, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
        except Exception:
            # one retry (damped)
            x1 = np.clip(0.7 * x0, lbx, ubx)
            sol = solver(x0=x1, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

        warm_x = np.array(sol["x"]).flatten()
        last_res = {"sol": sol, "meta": meta}

        if not quiet:
            print(f"k={k}..", end=" ")

    if not quiet:
        print("Done.")

    if last_res is None:
        return {"label": label, "feasible": False, "p": p, "M": np.array([0.0]), "t": np.array([0.0]),
                "obj_raw": 0.0, "phaseout_t": None}

    x_opt = np.array(last_res["sol"]["x"]).flatten()
    idx = last_res["meta"]["idx"]

    res = {
        "label": label,
        "p": p,
        "t": last_res["meta"]["tgrid"],
        "A": x_opt[idx["A"]],
        "M": x_opt[idx["M"]],
        "FE": x_opt[idx["FE"]],
        "FA": x_opt[idx["FA"]],
        "obj_raw": float(last_res["sol"]["f"]) * last_res["meta"]["J_scale"],
        "feasible": float(last_res["sol"]["g"][-1]) >= -1e-2,
    }
    res.update(_detect_phaseout(res["FE"], res["t"], p.phaseout_eps, p.phaseout_m))
    return res


# ==========================================
# 5. Sensitivity Heatmap
# ==========================================
def run_3d_sensitivity_heatmap():
    print("\n>>> Running 3D Sensitivity Heatmap...")
    gammas = [0.001, 0.01, 0.05, 0.08, 0.1, 0.15]
    se_caps = [5e6, 1e7, 1.5e7, 2e7]

    data = []
    for k_cap in se_caps:
        row = []
        for g in gammas:
            p_test = Params(gamma=g, K_max_total=k_cap, failure_active=False)
            res = solve_haat(p_test, label=f"Heatmap_K{k_cap:.0f}_G{g}")
            t_out = res["phaseout_t"] if res["phaseout_t"] is not None else Params().T
            row.append(float(t_out))
        data.append(row)

    df = pd.DataFrame(data, index=[f"{x/1e6:.1f}Mt" for x in se_caps], columns=[str(g) for g in gammas])
    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn_r",
                cbar_kws={"label": f"Phase-out Year (lower is better; {Params().T:.0f}=Never)"})
    plt.title("Fig 4: Phase-out Heatmap (Policy γ vs Technology K)", fontsize=14)
    plt.xlabel("Environmental Penalty γ")
    plt.ylabel("Mature SE Capacity (Mt/yr)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Fig4_Sensitivity_Heatmap.png"))
    print(f"[Output] Heatmap -> {OUTDIR}/Fig4_Sensitivity_Heatmap.png")


# ==========================================
# 6. Reporting + Plots
# ==========================================
def export_results_table(results: List[Dict[str, Any]]):
    rows = []
    for r in results:
        rows.append({
            "Scenario": r["label"],
            "Gamma": r["p"].gamma,
            "Resilience": "ON" if r["p"].failure_active else "OFF",
            "M(T) [Mt]": float(r["M"][-1] / 1e6),
            "Phase-out (yr)": f"{r['phaseout_t']:.1f}" if r["phaseout_t"] is not None else "Never",
            "Phaseout m": r.get("phaseout_confirm_window", ""),
            "Feasible": "YES" if r["feasible"] else "NO",
            "Total Obj ($B)": f"{float(r['obj_raw'])/1e9:.2f}",
        })
    df = pd.DataFrame(rows)
    print("\n=== Experiment Summary ===")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(OUTDIR, "final_summary.csv"), index=False)
    with open(os.path.join(OUTDIR, "final_summary.tex"), "w") as f:
        f.write(df.to_latex(index=False, float_format="%.3g", caption="Comparison of Scenarios"))


def plot_combined_results(res_pol, res_fail):
    t = res_pol["t"]

    # Fig 1
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    se_cap_eq = [capacity_logic(ti, res_pol["p"], is_numeric=True) / res_pol["p"].L_R for ti in t]
    ax1.plot(t, res_pol["FE"], label=r"Earth Rockets $F_E$")
    ax1.plot(t, se_cap_eq, linestyle="-.", alpha=0.7, label="Elevator Capacity (eq)")
    pt = res_pol.get("phaseout_t", None)
    if pt is not None:
        ax1.axvline(pt, linestyle=":", alpha=0.7)
        ax1.annotate(f"Phase-out @ {pt:.1f}",
                     xy=(pt, 0),
                     xytext=(pt + 2, max(res_pol["FE"]) * 0.6),
                     arrowprops=dict(arrowstyle="->"),
                     bbox=dict(boxstyle="round", fc="white"))
    ax1.set_title("Fig 1: Optimal Mode Switching Strategy")
    ax1.set_xlabel("Years after 2050")
    ax1.set_ylabel("Launch Rate (1/yr)")
    ax1.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Fig1_OptimalSwitching.png"))

    # Fig 2
    fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax2a.plot(t, res_pol["FE"], linestyle="--", label="Baseline")
    ax2a.plot(t, res_fail["FE"], label="Resilience Response")
    p_f = res_fail["p"]
    ax2a.axvspan(p_f.fail_t_start, p_f.fail_t_start + p_f.fail_duration, alpha=0.15, label="Failure")
    ax2a.set_title("Fig 2: System Resilience Analysis")
    ax2a.set_ylabel("Launch Rate (1/yr)")
    ax2a.legend()

    ax2b.plot(t, res_pol["M"]/1e6, label="Baseline mass")
    ax2b.plot(t, res_fail["M"]/1e6, linestyle="--", label="Resilience mass")
    ax2b.axhline(100, linestyle=":", label="Goal")
    ax2b.set_ylabel("Mass (Mt)")
    ax2b.set_xlabel("Years after 2050")
    ax2b.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "Fig2_Resilience.png"))


def main():
    print(">>> Running CORE submission script (no MC / no Tornado)...")

    # Baseline (low gamma)
    resA = solve_haat(Params(gamma=1e-5, failure_active=False), "A_Baseline")
    # Policy
    resB = solve_haat(Params(gamma=0.08, failure_active=False), "B_Policy")
    # Resilience
    resC = solve_haat(Params(gamma=0.05, failure_active=True), "C_Resilience")

    run_3d_sensitivity_heatmap()
    export_results_table([resA, resB, resC])
    plot_combined_results(resB, resC)

    print(f"\n>>> Done. Outputs in {OUTDIR}")


if __name__ == "__main__":
    main()