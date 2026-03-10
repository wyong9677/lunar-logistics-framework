# -*- coding: utf-8 -*-
"""
robustness.py
---------------------
Reduced-model Monte Carlo robustness diagnostic for finite-horizon launch
phase-out behavior.

This script intentionally operates on the reduced mature-regime / paper-level
model layer only (PaperParams + solve_haat). It is suitable for:
- reduced-model robustness diagnostics around the adopted mature-regime baseline,
- machine-readable ledgers for accepted / infeasible / crashed samples,
- phase-out detection summaries when finite-horizon phase-out is actually
  observed,
- no-phaseout / absence-of-detection summaries when it is not.

It is NOT a campaign-scale robustness runner and must not be described as full
campaign-scale robustness evidence in the manuscript.

Primary outputs (default OUTDIR=paper_outputs):
- mc_log.csv
- mc_summary.csv
- mc_meta.json
- monte_carlo_robustness.png
- mc_reference_probe.csv   (only if no feasible reduced baseline can be found)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTDIR = Path(os.environ.get("PAPER_OUTDIR", str(ROOT / "paper_outputs")))
DEFAULT_MPLCONFIGDIR = ROOT / ".mplconfig"

os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))
DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)

try:
    import pandas as pd
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as e:
    raise ImportError("Need pandas + matplotlib. Install: pip install pandas matplotlib") from e

from revised_solver_core import (
    MODEL_LAYER_REDUCED,
    PaperParams as ReducedParams,
    classify_feasibility,
    finalize_params,
    solve_haat,
)

DEFAULT_PERTURB_KEYS = ["K_max_total", "L_R", "r", "C_E", "C_A"]
DEFAULT_GAMMA_CANDIDATES = [
    0.08,
    0.0,
    1e-8,
    1e-6,
    1e-4,
    1e-3,
    5e-3,
    1e-2,
    5e-2,
    0.10,
    0.20,
    0.50,
    1.0,
]


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def _clone_params(base: ReducedParams) -> ReducedParams:
    return copy.deepcopy(base)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _get_terminal_mass(res: Dict[str, Any]) -> float:
    M = res.get("M", [])
    try:
        return float(M[-1]) if len(M) else 0.0
    except Exception:
        return 0.0


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
    vals: List[float] = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals


def _validate_perturb_keys(base: ReducedParams, keys: List[str]) -> None:
    if not keys:
        raise ValueError("perturb_keys must be a non-empty list.")
    missing = [k for k in keys if not hasattr(base, k)]
    if missing:
        raise ValueError(f"Unknown perturbation keys for ReducedParams: {missing}")


# ---------------------------------------------------------------------
# Explicit reduced-model mature-regime baseline
# ---------------------------------------------------------------------
def build_reduced_mature_baseline(
    *,
    gamma: float = 0.08,
    failure_active: bool = False,
) -> ReducedParams:
    """
    Explicit reduced mature-regime baseline aligned with the manuscript.

    Notes
    -----
    - This is NOT the full campaign-scale model.
    - alpha=1.0 removes commissioning transients.
    - failure_active is off by default for reduced mature-regime diagnostics.
    """
    p = ReducedParams(
        # Mission / sustainment
        M_goal=1.0e8,
        P_pop=1.0e5,
        w_gross=3.75,
        rho_recovery=0.95,

        # Reduced mature-regime SE settings
        K_max_total=3.0 * 179000.0,
        alpha=1.0,
        r=0.18,
        delta_eff=0.8714,
        failure_active=bool(failure_active),
        fail_t_start=20.0,
        fail_duration=3.0,
        fail_severity=0.90,

        # Transport
        L_R=125.0,
        FE_max=100000.0,
        FA_max=200000.0,
        A_max=5.0e5,

        # Economics / policy
        C_E=1.0e8,
        C_A=2.0e7,
        c_SE=250.0,
        e_E=1275.0,
        gamma=float(gamma),

        # Reduced-model horizon / discretization
        T=50.0,
        N=100,
        k_schedule=(1.0, 5.0, 20.0, 100.0),

        # Smoothness / phase-out rule
        w_smooth=50.0,
        phaseout_eps=1.0,
        phaseout_window_yrs=1.0,
        phaseout_relax=1.0,

        # Solver
        ipopt_max_iter=6000,
        ipopt_tol=1e-6,
    )
    return finalize_params(p)


def _deliverable_upper_bound(p: ReducedParams) -> float:
    """
    Conservative physical deliverable upper bound over horizon T:
        T * (direct_cap + two_stage_cap)
    where
        direct_cap = L_R * FE_max
        two_stage_cap = min(K_eff, L_R * FA_max)
    """
    keff = float(p.delta_eff * p.K_max_total)
    direct_cap = float(p.L_R * p.FE_max)
    two_stage_cap = min(keff, float(p.L_R * p.FA_max))
    return float(p.T) * (direct_cap + two_stage_cap)


# ---------------------------------------------------------------------
# Reduced-model baseline feasibility helpers
# ---------------------------------------------------------------------
def _evaluate_reduced_baseline(p: ReducedParams, label: str) -> Dict[str, Any]:
    return solve_haat(p, label=label)


def _is_ok(res: Dict[str, Any]) -> bool:
    return classify_feasibility(res) == "ok"


def _write_reference_probe(
    outdir: Path,
    gamma_candidates: List[float],
    failure_active: bool,
    requested_gamma: float,
) -> None:
    rows: List[Dict[str, Any]] = []
    for g in gamma_candidates:
        p = build_reduced_mature_baseline(gamma=float(g), failure_active=bool(failure_active))
        res = _evaluate_reduced_baseline(p, label=f"mc_probe_g{g:g}")
        rows.append(
            {
                "model_layer": res.get("ModelLayer", MODEL_LAYER_REDUCED),
                "requested_gamma": float(requested_gamma),
                "gamma": float(g),
                "solver_status": res.get("solver_status", ""),
                "solver_return_status": res.get("solver_return_status", ""),
                "feasible": bool(res.get("feasible", False)),
                "class": classify_feasibility(res),
                "viol_max": _safe_float(res.get("viol_max", np.nan)),
                "terminal_gap": _safe_float(res.get("terminal_gap", np.nan)),
                "JE": _safe_float(res.get("JE", np.nan)),
                "phaseout_t": _safe_float(res.get("phaseout_t", np.nan)),
                "M_end": _get_terminal_mass(res),
                "goal": float(p.M_goal),
                "T": float(p.T),
                "deliverable_upper_bound": _deliverable_upper_bound(p),
                "K_max_total": float(p.K_max_total),
                "delta_eff": float(p.delta_eff),
                "FE_max": float(p.FE_max),
                "FA_max": float(p.FA_max),
                "A_max": float(p.A_max),
                "failure_active": bool(p.failure_active),
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "mc_reference_probe.csv", index=False)


def _find_feasible_mc_baseline(
    *,
    requested_gamma: float,
    failure_active: bool,
    gamma_candidates: Optional[List[float]] = None,
) -> Tuple[ReducedParams, Dict[str, Any], float, bool]:
    """
    Return:
        (baseline_params, baseline_result, actual_gamma_used, used_fallback)

    First test the requested gamma. If infeasible, search a fallback gamma list.
    """
    p_req = build_reduced_mature_baseline(
        gamma=float(requested_gamma),
        failure_active=bool(failure_active),
    )

    ub = _deliverable_upper_bound(p_req)
    if ub < 1.01 * p_req.M_goal:
        raise RuntimeError(
            f"Reduced baseline is capacity-infeasible: deliverable upper bound "
            f"{ub:.3e} < goal {p_req.M_goal:.3e}. "
            f"Increase T, FE_max, FA_max, or K_max_total."
        )

    res_req = _evaluate_reduced_baseline(p_req, label=f"mc_base_g{requested_gamma:g}")
    if _is_ok(res_req):
        return p_req, res_req, float(requested_gamma), False

    if gamma_candidates is None:
        gamma_candidates = list(DEFAULT_GAMMA_CANDIDATES)

    seen = set()
    gamma_candidates_unique: List[float] = []
    for g in [float(requested_gamma)] + [float(x) for x in gamma_candidates]:
        if g not in seen:
            gamma_candidates_unique.append(g)
            seen.add(g)

    for g in gamma_candidates_unique:
        p = build_reduced_mature_baseline(gamma=g, failure_active=bool(failure_active))
        res = _evaluate_reduced_baseline(p, label=f"mc_fallback_g{g:g}")
        if _is_ok(res):
            return p, res, g, (g != float(requested_gamma))

    raise RuntimeError("No feasible reduced-model baseline found for requested or fallback gamma values.")


# ---------------------------------------------------------------------
# Monte Carlo perturbation helpers
# ---------------------------------------------------------------------
def _sample_perturbed_params(
    base: ReducedParams,
    rng: np.random.Generator,
    jitter: float,
    perturb_keys: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Return:
        key -> {"baseline": ..., "factor": ..., "sampled": ...}
    """
    out: Dict[str, Dict[str, float]] = {}
    for key in perturb_keys:
        baseline = _safe_float(getattr(base, key))
        factor = float(rng.uniform(1.0 - jitter, 1.0 + jitter))
        sampled = baseline * factor
        out[key] = {
            "baseline": baseline,
            "factor": factor,
            "sampled": sampled,
        }
    return out


def _write_metadata(
    outdir: Path,
    *,
    base: ReducedParams,
    base_result: Dict[str, Any],
    n_sims: int,
    seed: int,
    jitter: float,
    goal_relax: float,
    perturb_keys: List[str],
    figure_name: str,
    requested_gamma: float,
    actual_gamma: float,
    gamma_fallback_used: bool,
    gamma_candidates: List[float],
) -> None:
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "robustness.py",
        "model_layer": base_result.get("ModelLayer", MODEL_LAYER_REDUCED),
        "interpretation_scope": base_result.get(
            "InterpretationScope",
            "Reduced-model outputs are mechanism-oriented diagnostics and not full campaign-scale robustness evidence.",
        ),
        "notes": [
            "This script evaluates reduced-model robustness of finite-horizon launch phase-out behavior.",
            "It should not be described as full campaign-scale robustness unless the full model is also sampled.",
            "The perturbed parameter set is explicitly recorded below.",
            "The actual gamma used may differ from the requested gamma if baseline fallback was needed.",
            "A feasible sample is defined here as solver-class ok plus terminal mass >= goal_relax * M_goal.",
            "Finite-horizon no-phaseout is reported separately and is not treated as solver failure.",
            "The figure is a robustness diagnostic; when no finite-horizon phase-out is detected, the panel reports absence of detection rather than a non-empty year distribution.",
        ],
        "n_sims": int(n_sims),
        "seed": int(seed),
        "jitter": float(jitter),
        "goal_relax": float(goal_relax),
        "perturb_keys": perturb_keys,
        "requested_gamma": float(requested_gamma),
        "actual_gamma_used": float(actual_gamma),
        "gamma_fallback_used": bool(gamma_fallback_used),
        "gamma_candidates": [float(x) for x in gamma_candidates],
        "baseline_deliverable_upper_bound": _deliverable_upper_bound(base),
        "base_params": vars(base),
        "baseline_solver": {
            "solver_status": base_result.get("solver_status", ""),
            "solver_return_status": base_result.get("solver_return_status", ""),
            "feasible": bool(base_result.get("feasible", False)),
            "viol_max": _safe_float(base_result.get("viol_max", np.nan)),
            "terminal_gap": _safe_float(base_result.get("terminal_gap", np.nan)),
            "acceptance_rule": base_result.get("ReducedAcceptanceRule", {}),
        },
        "figure": figure_name,
    }
    with open(outdir / "mc_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Main Monte Carlo routine
# ---------------------------------------------------------------------
def run_monte_carlo_robustness(
    n_sims: int = 25,
    seed: int = 42,
    jitter: float = 0.05,
    goal_relax: float = 0.99,
    perturb_keys: Optional[List[str]] = None,
    gamma: float = 0.08,
    gamma_candidates: Optional[List[float]] = None,
    failure_active: bool = False,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    if int(n_sims) < 1:
        raise ValueError(f"n_sims must be >= 1, got {n_sims}")
    if not (0.0 <= float(jitter) < 1.0):
        raise ValueError(f"jitter must satisfy 0 <= jitter < 1, got {jitter}")
    if not (0.0 < float(goal_relax) <= 1.0):
        raise ValueError(f"goal_relax must satisfy 0 < goal_relax <= 1, got {goal_relax}")

    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    gamma_candidates = list(DEFAULT_GAMMA_CANDIDATES) if gamma_candidates is None else list(gamma_candidates)

    # Find a feasible reduced-model baseline
    try:
        base, base_res, actual_gamma_used, gamma_fallback_used = _find_feasible_mc_baseline(
            requested_gamma=float(gamma),
            failure_active=bool(failure_active),
            gamma_candidates=gamma_candidates,
        )
    except RuntimeError as exc:
        _write_reference_probe(
            outdir=outdir,
            gamma_candidates=[float(gamma)] + [float(x) for x in gamma_candidates],
            failure_active=bool(failure_active),
            requested_gamma=float(gamma),
        )
        raise RuntimeError(
            "No feasible reduced-model baseline found. "
            f"A diagnostic probe table has been written to {outdir / 'mc_reference_probe.csv'}. "
            f"Original reason: {exc}"
        ) from exc

    if perturb_keys is None:
        perturb_keys = list(DEFAULT_PERTURB_KEYS)
    _validate_perturb_keys(base, perturb_keys)

    figure_name = "monte_carlo_robustness.png"
    _write_metadata(
        outdir=outdir,
        base=base,
        base_result=base_res,
        n_sims=n_sims,
        seed=seed,
        jitter=jitter,
        goal_relax=goal_relax,
        perturb_keys=perturb_keys,
        figure_name=figure_name,
        requested_gamma=float(gamma),
        actual_gamma=float(actual_gamma_used),
        gamma_fallback_used=bool(gamma_fallback_used),
        gamma_candidates=gamma_candidates,
    )

    outcomes = {
        "crashed": 0,
        "infeasible": 0,
        "low_mass": 0,
        "no_phaseout": 0,
        "phaseout_detected": 0,
    }

    solver_ok_count = 0
    mass_ok_count = 0

    rows: List[Dict[str, Any]] = []
    phaseouts: List[float] = []

    for i in range(int(n_sims)):
        p = _clone_params(base)
        sampled = _sample_perturbed_params(base, rng, jitter, perturb_keys)

        for key, info in sampled.items():
            setattr(p, key, info["sampled"])
        p = finalize_params(p)

        res = solve_haat(p, label=f"MC_{i}")
        solver_class = classify_feasibility(res)

        mT = _get_terminal_mass(res)
        ph = res.get("phaseout_t", None)

        solver_feasible = solver_class == "ok"
        terminal_mass_ok = bool(mT >= p.M_goal * goal_relax)
        feasible_sample = bool(solver_feasible and terminal_mass_ok)

        if solver_feasible:
            solver_ok_count += 1
        if terminal_mass_ok:
            mass_ok_count += 1

        if solver_class == "crashed":
            outcomes["crashed"] += 1
            outcome = "crashed"
        elif solver_class == "infeasible":
            outcomes["infeasible"] += 1
            outcome = "infeasible"
        elif not terminal_mass_ok:
            outcomes["low_mass"] += 1
            outcome = "low_mass"
        elif ph is None:
            outcomes["no_phaseout"] += 1
            outcome = "no_phaseout"
        else:
            outcomes["phaseout_detected"] += 1
            outcome = "phaseout_detected"
            phaseouts.append(float(ph))

        row: Dict[str, Any] = {
            "model_layer": res.get("ModelLayer", MODEL_LAYER_REDUCED),
            "interpretation_scope": res.get(
                "InterpretationScope",
                "Reduced-model outputs are mechanism-oriented diagnostics and not full campaign-scale robustness evidence.",
            ),
            "run": int(i),
            "outcome": outcome,
            "solver_class": solver_class,
            "solver_status": res.get("solver_status", ""),
            "solver_return_status": res.get("solver_return_status", ""),
            "accepted_by_ipopt": bool(res.get("accepted_by_ipopt", False)),
            "viol_max": _safe_float(res.get("viol_max", np.nan)),
            "terminal_gap": _safe_float(res.get("terminal_gap", np.nan)),
            "feasible_flag": bool(res.get("feasible", False)),
            "solver_feasible": bool(solver_feasible),
            "feasible_sample": bool(feasible_sample),
            "terminal_mass_ok": bool(terminal_mass_ok),
            "M_T": float(mT),
            "phaseout_t": np.nan if ph is None else float(ph),
            "phaseout_detected": bool(ph is not None and feasible_sample),
            "JE": _safe_float(res.get("JE", np.nan)),
            "N_EM": _safe_float(res.get("N_EM", np.nan)),
            "obj_raw": _safe_float(res.get("obj_raw", np.nan)),
            "requested_gamma": float(gamma),
            "actual_gamma_used": float(actual_gamma_used),
            "gamma_fallback_used": bool(gamma_fallback_used),
            "gamma": _safe_float(p.gamma),
            "delta_eff": _safe_float(p.delta_eff),
            "goal_relax": float(goal_relax),
            "seed": int(seed),
            "jitter": float(jitter),
        }

        for key in perturb_keys:
            row[f"{key}_base"] = sampled[key]["baseline"]
            row[f"{key}_factor"] = sampled[key]["factor"]
            row[f"{key}_sampled"] = sampled[key]["sampled"]

        rows.append(row)

    df_log = pd.DataFrame(rows)
    df_log.to_csv(outdir / "mc_log.csv", index=False)

    feasible_count = int(outcomes["phaseout_detected"] + outcomes["no_phaseout"])
    feasible_rate = feasible_count / max(int(n_sims), 1)

    phaseout_detected_count = int(outcomes["phaseout_detected"])
    no_phaseout_count = int(outcomes["no_phaseout"])

    phaseout_detected_rate_among_feasible = phaseout_detected_count / max(feasible_count, 1)
    no_phaseout_rate_among_feasible = no_phaseout_count / max(feasible_count, 1)

    summary_row: Dict[str, Any] = {
        "model_layer": MODEL_LAYER_REDUCED,
        "n_sims": int(n_sims),
        "solver_ok_count": int(solver_ok_count),
        "solver_ok_rate": float(solver_ok_count / max(int(n_sims), 1)),
        "mass_ok_count": int(mass_ok_count),
        "mass_ok_rate": float(mass_ok_count / max(int(n_sims), 1)),
        "feasible_count": feasible_count,
        "feasible_rate": float(feasible_rate),
        "phaseout_detected_count": phaseout_detected_count,
        "phaseout_detected_rate_among_feasible": float(phaseout_detected_rate_among_feasible),
        "no_phaseout_count": no_phaseout_count,
        "no_phaseout_rate_among_feasible": float(no_phaseout_rate_among_feasible),
        "crashed": int(outcomes["crashed"]),
        "infeasible": int(outcomes["infeasible"]),
        "low_mass": int(outcomes["low_mass"]),
        "seed": int(seed),
        "jitter": float(jitter),
        "goal_relax": float(goal_relax),
        "requested_gamma": float(gamma),
        "actual_gamma_used": float(actual_gamma_used),
        "gamma_fallback_used": bool(gamma_fallback_used),
        "failure_active": bool(failure_active),
        "perturb_keys": ",".join(perturb_keys),
        "baseline_deliverable_upper_bound": _deliverable_upper_bound(base),
    }

    if len(phaseouts) > 0:
        summary_row.update(
            {
                "phaseout_mean": float(np.mean(phaseouts)),
                "phaseout_median": float(np.median(phaseouts)),
                "phaseout_min": float(np.min(phaseouts)),
                "phaseout_max": float(np.max(phaseouts)),
            }
        )
    else:
        summary_row.update(
            {
                "phaseout_mean": np.nan,
                "phaseout_median": np.nan,
                "phaseout_min": np.nan,
                "phaseout_max": np.nan,
            }
        )

    df_sum = pd.DataFrame([summary_row])
    df_sum.to_csv(outdir / "mc_summary.csv", index=False)

    # Figure: histogram only for feasible samples with finite-horizon phase-out detected
    plt.figure(figsize=(7, 5))
    if len(phaseouts) > 0:
        plt.hist(phaseouts, bins=min(10, len(phaseouts)), edgecolor="black", alpha=0.7)
        mean_v = float(np.mean(phaseouts))
        med_v = float(np.median(phaseouts))
        plt.axvline(mean_v, linestyle="--", linewidth=1.2, label=f"Mean: {mean_v:.1f}")
        plt.axvline(med_v, linestyle=":", linewidth=1.2, label=f"Median: {med_v:.1f}")
        plt.legend()
        plt.xlabel("Phase-out year (years after 2050)")
        plt.ylabel("Frequency")
    else:
        plt.text(
            0.5,
            0.58,
            "No finite-horizon phase-out detected\namong feasible reduced-model samples",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
        )
        plt.text(
            0.5,
            0.33,
            f"feasible samples = {feasible_count}/{n_sims}\n"
            f"solver-ok = {solver_ok_count}/{n_sims}\n"
            f"mass-ok = {mass_ok_count}/{n_sims}",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=9,
        )
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")

    plt.title(
        f"Reduced-Model Robustness Diagnostic\n"
        f"(N={n_sims}, ±{int(jitter * 100)}% perturbation)"
    )
    plt.tight_layout()
    plt.savefig(outdir / figure_name, dpi=300)
    plt.close()

    print(df_sum.to_string(index=False))
    print(f"[OK] Outputs in: {outdir}/")
    print(f"[OK] Wrote: {outdir / 'mc_meta.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduced-model Monte Carlo robustness runner")
    parser.add_argument("--n-sims", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jitter", type=float, default=0.05)
    parser.add_argument("--goal-relax", type=float, default=0.99)
    parser.add_argument(
        "--perturb-keys",
        type=str,
        default=",".join(DEFAULT_PERTURB_KEYS),
        help="comma-separated ReducedParams fields to perturb, e.g. K_max_total,L_R,r,C_E,C_A",
    )
    parser.add_argument("--gamma", type=float, default=0.08)
    parser.add_argument(
        "--gamma-candidates",
        type=str,
        default=",".join(str(x) for x in DEFAULT_GAMMA_CANDIDATES),
        help="comma-separated fallback gamma candidates, e.g. 0.08,0,1e-8,1e-6,1e-4",
    )
    parser.add_argument("--failure-active", action="store_true")
    parser.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    args = parser.parse_args()

    perturb_keys = _parse_list(args.perturb_keys)
    gamma_candidates = _parse_float_list(args.gamma_candidates)

    run_monte_carlo_robustness(
        n_sims=int(args.n_sims),
        seed=int(args.seed),
        jitter=float(args.jitter),
        goal_relax=float(args.goal_relax),
        perturb_keys=perturb_keys,
        gamma=float(args.gamma),
        gamma_candidates=gamma_candidates,
        failure_active=bool(args.failure_active),
        outdir=Path(args.outdir),
    )


if __name__ == "__main__":
    main()
