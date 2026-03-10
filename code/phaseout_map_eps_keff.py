# -*- coding: utf-8 -*-
"""
phaseout_map_eps_keff.py
------------------------
Reduced mature-regime 2D policy-map sweep:
    (epsilon, K_SE_eff) -> phase-out year

IMPORTANT
---------
This script is intended for the reduced mature-regime / paper-level diagnostic model.
It is NOT the full campaign-scale baseline scenario runner.

ASSUMPTION
----------
This script assumes that revised_solver_core.py has already been patched so that
the reduced-model solver accepts numerically usable solutions via an appropriate
acceptance rule (e.g. Solve_Succeeded / Solved_To_Acceptable_Level together with
terminal-mass and constraint-violation checks). If the core solver still rejects
all reduced-model runs, this script will write diagnostic probe tables and stop.

Main features
-------------
1) Explicit reduced mature-regime baseline, aligned with the manuscript baseline.
2) Feasibility pre-check via a conservative deliverable upper bound.
3) Robust feasible-reference search starting from gamma = 0.
4) No hard-coded assumption that JE(gamma) is globally monotone.
5) Uses:
      - feasible reference search,
      - gamma-grid scan,
      - local sign-change bracketing,
      - safeguarded log-domain bisection refinement.
6) If no feasible reference is found, writes a probe CSV for debugging.
7) Writes metadata for reproducibility.

Outputs (default OUTDIR=paper_outputs):
- phaseout_map_eps_keff.csv
- phaseout_map_eps_keff_pivot.csv
- phaseout_map_eps_keff_meta.json
- phaseout_reference_probe.csv   (only if feasible reference cannot be found)
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from revised_solver_core import (
    PaperParams as ReducedParams,
    finalize_params,
    solve_haat,
    classify_feasibility,
)

DEFAULT_OUTDIR = "paper_outputs"
DEFAULT_REF_GAMMAS = [
    0.0, 1e-8, 1e-6, 1e-4, 1e-3,
    5e-3, 1e-2, 5e-2, 0.08, 0.10,
    0.20, 0.50, 1.0,
]


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------
def _is_ok(res: Dict[str, Any]) -> bool:
    return classify_feasibility(res) == "ok"


def _clone_params(base: ReducedParams) -> ReducedParams:
    return copy.deepcopy(base)


def _prepare_params(base: ReducedParams, **updates: Any) -> ReducedParams:
    p = _clone_params(base)
    for k, v in updates.items():
        setattr(p, k, v)
    return finalize_params(p)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _relerr(value: float, target: float) -> float:
    return abs(float(value) - float(target)) / max(abs(float(target)), 1e-12)


def _unique_float_list(vals: List[float]) -> List[float]:
    seen = set()
    out: List[float] = []
    for v in vals:
        vf = float(v)
        if vf not in seen:
            out.append(vf)
            seen.add(vf)
    return out


# ---------------------------------------------------------------------
# Explicit reduced-model mature-regime baseline
# ---------------------------------------------------------------------
def build_reduced_mature_baseline() -> ReducedParams:
    """
    Build an explicit reduced mature-regime baseline aligned with the
    campaign-scale manuscript baseline wherever possible.

    Notes
    -----
    - This is NOT the full campaign-scale model.
    - alpha is forced to 1.0 to remove commissioning transients.
    - failure_active is forced to False for mature-regime diagnostics.
    - The horizon T remains a reduced-model diagnostic horizon.
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
        failure_active=False,
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
        gamma=0.0,  # reference search starts from minimal launch aversion

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
    Conservative annual deliverable upper bound for the reduced model.

    direct_cap    = L_R * FE_max
    two_stage_cap = min(K_eff, L_R * FA_max)

    Over horizon T, the upper bound is:
        T * (direct_cap + two_stage_cap)
    """
    keff = float(p.delta_eff * p.K_max_total)
    direct_cap = float(p.L_R * p.FE_max)
    two_stage_cap = min(keff, float(p.L_R * p.FA_max))
    return float(p.T) * (direct_cap + two_stage_cap)


# ---------------------------------------------------------------------
# Reduced-model evaluations
# ---------------------------------------------------------------------
def _evaluate_gamma(
    base: ReducedParams,
    gamma: float,
    label: str,
) -> Tuple[Optional[Dict[str, Any]], float, str]:
    """
    Evaluate one gamma value and return:
        (result_or_none, JE_or_nan, status)
    """
    p = _prepare_params(base, gamma=float(gamma))
    res = solve_haat(p, label=label)

    if not _is_ok(res):
        return None, float("nan"), "infeasible"

    je = _safe_float(res.get("JE", np.nan))
    if not np.isfinite(je):
        return None, float("nan"), "nonfinite_JE"

    return res, je, "ok"


def _probe_row_from_result(
    p: ReducedParams,
    g: float,
    res: Dict[str, Any],
) -> Dict[str, Any]:
    m_end = np.nan
    if res.get("M") is not None:
        try:
            m_end = float(res["M"][-1])
        except Exception:
            m_end = np.nan

    return {
        "gamma": float(g),
        "solver_status": res.get("solver_status", ""),
        "feasible": bool(res.get("feasible", False)),
        "class": classify_feasibility(res),
        "JE": _safe_float(res.get("JE", np.nan)),
        "phaseout_t": _safe_float(res.get("phaseout_t", np.nan)),
        "M_end": m_end,
        "goal": float(p.M_goal),
        "T": float(p.T),
        "deliverable_upper_bound": _deliverable_upper_bound(p),
        "K_max_total": float(p.K_max_total),
        "delta_eff": float(p.delta_eff),
        "FE_max": float(p.FE_max),
        "FA_max": float(p.FA_max),
        "A_max": float(p.A_max),
    }


def find_feasible_reference(
    base: ReducedParams,
    gamma_candidates: Optional[List[float]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[float], str]:
    """
    Find a feasible reduced-model reference run to define JE_ref.
    We do NOT assume gamma -> 0 is feasible, but we test it explicitly.
    """
    if gamma_candidates is None:
        gamma_candidates = list(DEFAULT_REF_GAMMAS)

    gamma_candidates = _unique_float_list(gamma_candidates)

    for g in gamma_candidates:
        res, _, status = _evaluate_gamma(base, gamma=float(g), label=f"ref_g{g:g}")
        if res is not None and status == "ok":
            return res, float(g), "ok"

    return None, None, "no_feasible_ref"


def _write_reference_probe(
    base: ReducedParams,
    outdir: Path,
    gamma_candidates: List[float],
) -> None:
    """
    If no feasible reference is found, write a probe table so the user can inspect
    solver status, M_end, JE, and parameter values across candidate gamma values.
    """
    rows: List[Dict[str, Any]] = []
    for g in _unique_float_list(gamma_candidates):
        p = _prepare_params(base, gamma=float(g))
        res = solve_haat(p, label=f"probe_g{g:g}")
        rows.append(_probe_row_from_result(p, g, res))

    pd.DataFrame(rows).to_csv(outdir / "phaseout_reference_probe.csv", index=False)


# ---------------------------------------------------------------------
# JE(gamma) target matching
# ---------------------------------------------------------------------
def _find_best_grid_point(
    pts: List[Tuple[float, float, Dict[str, Any]]],
    eps_target: float,
) -> Tuple[Dict[str, Any], float, float, float]:
    """
    Return best grid point by relative JE error:
        (res_best, gamma_best, je_best, err_best)
    """
    best = min(pts, key=lambda x: _relerr(x[1], eps_target))
    g_best, je_best, res_best = best
    return res_best, g_best, je_best, _relerr(je_best, eps_target)


def _find_sign_change_bracket(
    pts: List[Tuple[float, float, Dict[str, Any]]],
    eps_target: float,
) -> Optional[Tuple[Tuple[float, float, Dict[str, Any]], Tuple[float, float, Dict[str, Any]]]]:
    """
    Find an adjacent gamma bracket with sign change in f(g)=JE(g)-eps_target.
    Among all adjacent sign-change brackets, choose the one with the best endpoint closeness.

    Returns:
        ((g_lo, je_lo, res_lo), (g_hi, je_hi, res_hi)) or None
    """
    if len(pts) < 2:
        return None

    pts_sorted = sorted(pts, key=lambda x: x[0])
    candidates = []

    for i in range(len(pts_sorted) - 1):
        a = pts_sorted[i]
        b = pts_sorted[i + 1]
        fa = a[1] - eps_target
        fb = b[1] - eps_target

        if fa == 0.0 or fb == 0.0:
            candidates.append((a, b, 0.0))
            continue

        if fa * fb < 0.0:
            score = max(_relerr(a[1], eps_target), _relerr(b[1], eps_target))
            candidates.append((a, b, score))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[2])
    return candidates[0][0], candidates[0][1]


def gamma_search_to_match_epsilon(
    base: ReducedParams,
    eps_target: float,
    rel_tol: float = 0.05,
    gamma_grid: Optional[List[float]] = None,
    max_bisect: int = 24,
) -> Tuple[Optional[Dict[str, Any]], float, float, float, str]:
    """
    Match JE(gamma) ≈ eps_target robustly.

    Procedure:
    1) evaluate on gamma grid,
    2) return best grid point if already within tolerance,
    3) otherwise search for an adjacent sign-change bracket in f(g)=JE(g)-eps_target,
    4) if such a bracket exists, refine by safeguarded bisection in log(gamma).

    This refinement assumes only a usable local sign-change bracket, not global monotonicity.
    """
    if gamma_grid is None:
        gamma_grid = list(np.logspace(-4, 0, 17))

    gamma_grid = _unique_float_list(gamma_grid)

    pts: List[Tuple[float, float, Dict[str, Any]]] = []
    for g in gamma_grid:
        res, je, status = _evaluate_gamma(base, gamma=float(g), label=f"grid_g{g:g}")
        if res is None or status != "ok":
            continue
        pts.append((float(g), float(je), res))

    if not pts:
        return None, np.nan, np.nan, np.nan, "no_feasible_gamma"

    res_best, g_best, je_best, e_best = _find_best_grid_point(pts, eps_target)
    if e_best <= rel_tol:
        return res_best, g_best, je_best, e_best, "ok_grid"

    bracket = _find_sign_change_bracket(pts, eps_target)
    if bracket is None:
        return res_best, g_best, je_best, e_best, "approx_grid_no_sign_change"

    left, right = bracket
    g_lo, je_lo, _ = left
    g_hi, je_hi, _ = right

    if g_lo > g_hi:
        g_lo, g_hi = g_hi, g_lo
        je_lo, je_hi = je_hi, je_lo

    f_lo = je_lo - eps_target
    f_hi = je_hi - eps_target

    if f_lo == 0.0:
        return left[2], g_lo, je_lo, 0.0, "ok_grid_exact_left"
    if f_hi == 0.0:
        return right[2], g_hi, je_hi, 0.0, "ok_grid_exact_right"

    best = (res_best, g_best, je_best, e_best)
    lo, hi = float(g_lo), float(g_hi)

    for _ in range(max_bisect):
        mid = float(np.sqrt(lo * hi))  # log-domain midpoint
        res_mid, je_mid, status_mid = _evaluate_gamma(base, gamma=mid, label=f"bisect_g{mid:.3e}")

        if res_mid is None or status_mid != "ok" or not np.isfinite(je_mid):
            return best[0], best[1], best[2], best[3], "approx_bisect_mid_invalid"

        e_mid = _relerr(je_mid, eps_target)
        if e_mid < best[3]:
            best = (res_mid, mid, je_mid, e_mid)

        if e_mid <= rel_tol:
            return res_mid, mid, je_mid, e_mid, "ok_bisect"

        f_mid = je_mid - eps_target

        if f_lo * f_mid < 0.0:
            hi = mid
            f_hi = f_mid
        elif f_mid * f_hi < 0.0:
            lo = mid
            f_lo = f_mid
        else:
            return best[0], best[1], best[2], best[3], "approx_bisect_no_subbracket"

    return best[0], best[1], best[2], best[3], "approx_bisect_maxiter"


# ---------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------
def _parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _write_metadata(
    outdir: Path,
    base_params: ReducedParams,
    *,
    Keff_ref: float,
    JE_ref: float,
    gamma_ref: Optional[float],
    ref_status: str,
    ref_gamma_candidates: List[float],
    keff_factors: List[float],
    eps_multipliers: List[float],
    gamma_grid: List[float],
    rel_tol: float,
    max_bisect: int,
    resume: bool,
) -> None:
    meta = {
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "script": "phaseout_map_eps_keff.py",
        "model_layer": "reduced_mature_regime",
        "notes": [
            "This sweep is not the full campaign-scale baseline model.",
            "alpha is forced to 1.0 for mature-regime mapping.",
            "failure_active is forced to False for mature-regime mapping.",
            "Keff is realized by varying K_max_total while keeping delta_eff fixed.",
            "epsilon targets are scaled from a feasible JE reference run.",
            "Successful execution assumes revised_solver_core.py uses an updated reduced-model acceptance rule.",
        ],
        "base_params_after_mature_regime_adjustment": vars(base_params),
        "base_deliverable_upper_bound": _deliverable_upper_bound(base_params),
        "Keff_ref_tpy": Keff_ref,
        "JE_ref": JE_ref,
        "gamma_ref": gamma_ref,
        "reference_status": ref_status,
        "reference_gamma_candidates": ref_gamma_candidates,
        "keff_factors": keff_factors,
        "eps_multipliers": eps_multipliers,
        "gamma_grid": gamma_grid,
        "rel_tol": rel_tol,
        "max_bisect": max_bisect,
        "resume": resume,
    }
    meta_path = outdir / "phaseout_map_eps_keff_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------
def run_phaseout_map(
    fast: bool = False,
    rel_tol: float = 0.05,
    max_bisect: int = 24,
    gamma_grid_size: int = 17,
    resume: bool = True,
    keff_factors: Optional[List[float]] = None,
    eps_multipliers: Optional[List[float]] = None,
    outdir: str = DEFAULT_OUTDIR,
) -> None:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # Explicit reduced mature-regime baseline
    p0 = build_reduced_mature_baseline()

    # Physical capacity sanity check
    ub = _deliverable_upper_bound(p0)
    if ub < 1.01 * p0.M_goal:
        raise RuntimeError(
            f"Reduced baseline is capacity-infeasible: deliverable upper bound "
            f"{ub:.3e} < goal {p0.M_goal:.3e}. "
            f"Increase T, FE_max, FA_max, or K_max_total."
        )

    # Reference effective throughput
    Keff_ref = float(p0.delta_eff * p0.K_max_total)

    # Keff grid
    kf_list = (
        keff_factors
        if keff_factors is not None
        else ([0.75, 1.00, 1.25] if fast else [0.50, 0.75, 1.00, 1.25, 1.50])
    )
    kf_list = list(map(float, kf_list))
    Keff_list = [Keff_ref * f for f in kf_list]

    # Reference search candidates
    ref_gamma_candidates = list(DEFAULT_REF_GAMMAS)

    # Feasible reference for epsilon scaling
    ref_res, gamma_ref, ref_status = find_feasible_reference(
        p0,
        gamma_candidates=ref_gamma_candidates,
    )
    if ref_res is None:
        _write_reference_probe(p0, outdir_path, ref_gamma_candidates)
        raise RuntimeError(
            "No feasible reduced-model reference found. "
            f"A diagnostic probe table has been written to "
            f"{outdir_path / 'phaseout_reference_probe.csv'}. "
            "This usually means the reduced-model core acceptance logic in revised_solver_core.py "
            "still rejects all numerically usable reduced-model solutions."
        )

    JE_ref = float(ref_res["JE"])
    print(f"[Ref] gamma={gamma_ref} feasible, JE_ref={JE_ref:.6e} ({ref_status})")

    # epsilon grid scaled from feasible reference
    em_list = (
        eps_multipliers
        if eps_multipliers is not None
        else ([0.20, 0.50, 1.00] if fast else [0.10, 0.20, 0.50, 1.00, 2.00])
    )
    em_list = list(map(float, em_list))
    eps_list = [float(m) * JE_ref for m in em_list]

    gamma_grid = list(np.logspace(-4, 0, max(int(gamma_grid_size), 5)))

    _write_metadata(
        outdir=outdir_path,
        base_params=p0,
        Keff_ref=Keff_ref,
        JE_ref=JE_ref,
        gamma_ref=gamma_ref,
        ref_status=ref_status,
        ref_gamma_candidates=ref_gamma_candidates,
        keff_factors=kf_list,
        eps_multipliers=em_list,
        gamma_grid=list(map(float, gamma_grid)),
        rel_tol=float(rel_tol),
        max_bisect=int(max_bisect),
        resume=bool(resume),
    )

    out_csv = outdir_path / "phaseout_map_eps_keff.csv"
    rows: List[Dict[str, Any]] = []
    done_keys = set()

    if resume and out_csv.exists():
        try:
            old = pd.read_csv(out_csv)
            old = old.drop_duplicates(subset=["iK", "iE"], keep="last")
            for _, rr in old.iterrows():
                done_keys.add((int(rr["iK"]), int(rr["iE"])))
            rows.extend(old.to_dict(orient="records"))
            print(f"[Resume] loaded {len(done_keys)} existing points from {out_csv}")
        except Exception as exc:
            print(f"[Resume] warning: failed to load existing CSV ({exc}); starting fresh.")

    for iK, Keff in enumerate(Keff_list):
        pK = _prepare_params(p0, K_max_total=float(Keff / max(p0.delta_eff, 1e-12)))

        for iE, eps_target in enumerate(eps_list):
            if (iK, iE) in done_keys:
                continue

            res, g_star, je_star, rel_err, status = gamma_search_to_match_epsilon(
                base=pK,
                eps_target=float(eps_target),
                rel_tol=float(rel_tol),
                gamma_grid=gamma_grid,
                max_bisect=int(max_bisect),
            )

            if res is None:
                rows.append({
                    "model_layer": "reduced_mature_regime",
                    "iK": iK,
                    "iE": iE,
                    "Keff_tpy": float(Keff),
                    "K_max_total": float(pK.K_max_total),
                    "delta_eff": float(pK.delta_eff),
                    "epsilon": float(eps_target),
                    "epsilon_mult": float(eps_target / max(JE_ref, 1e-12)),
                    "JE_ref": float(JE_ref),
                    "gamma_ref": float(gamma_ref) if gamma_ref is not None else np.nan,
                    "gamma_star": np.nan,
                    "JE": np.nan,
                    "JE_relerr": np.nan,
                    "phaseout_t": np.nan,
                    "phaseout_label": "Never",
                    "status": status,
                })
                pd.DataFrame(rows).drop_duplicates(subset=["iK", "iE"], keep="last").to_csv(out_csv, index=False)
                continue

            ph = res.get("phaseout_t", None)
            phaseout_t = np.nan if ph is None else float(ph)
            phaseout_label = "Never" if ph is None else f"{phaseout_t:.2f}"

            rows.append({
                "model_layer": "reduced_mature_regime",
                "iK": iK,
                "iE": iE,
                "Keff_tpy": float(Keff),
                "K_max_total": float(pK.K_max_total),
                "delta_eff": float(pK.delta_eff),
                "epsilon": float(eps_target),
                "epsilon_mult": float(eps_target / max(JE_ref, 1e-12)),
                "JE_ref": float(JE_ref),
                "gamma_ref": float(gamma_ref) if gamma_ref is not None else np.nan,
                "gamma_star": float(g_star),
                "JE": float(je_star),
                "JE_relerr": float(rel_err),
                "phaseout_t": float(phaseout_t) if np.isfinite(phaseout_t) else np.nan,
                "phaseout_label": phaseout_label,
                "status": status,
            })

            print(
                f"[Keff={Keff/1e6:.3f} Mt/yr] "
                f"eps={eps_target:.3e} -> JE={je_star:.3e}, "
                f"phaseout={phaseout_label}, gamma={g_star:.3e} ({status})"
            )

            pd.DataFrame(rows).drop_duplicates(subset=["iK", "iE"], keep="last").to_csv(out_csv, index=False)

    df = pd.DataFrame(rows).drop_duplicates(subset=["iK", "iE"], keep="last")
    df.to_csv(out_csv, index=False)

    pivot = df.pivot_table(
        index="Keff_tpy",
        columns="epsilon_mult",
        values="phaseout_t",
        aggfunc="first",
    )
    pivot_csv = outdir_path / "phaseout_map_eps_keff_pivot.csv"
    pivot.to_csv(pivot_csv)

    print(f"[OK] wrote: {out_csv}")
    print(f"[OK] wrote: {pivot_csv}")
    print(f"[OK] wrote: {outdir_path / 'phaseout_map_eps_keff_meta.json'}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reduced mature-regime 2D phase-out map sweep over (epsilon, K_SE_eff)"
    )
    parser.add_argument("--fast", action="store_true", help="use smaller grids for quick runs")
    parser.add_argument("--rel-tol", type=float, default=0.05, help="relative JE matching tolerance")
    parser.add_argument("--max-bisect", type=int, default=24, help="maximum safeguarded log-bisection steps")
    parser.add_argument("--gamma-grid-size", type=int, default=17, help="number of gamma grid points")
    parser.add_argument("--no-resume", action="store_true", help="disable resume from existing CSV")
    parser.add_argument("--keff-factors", type=str, default="", help="comma-separated list, e.g. 0.5,1.0,1.5")
    parser.add_argument("--eps-multipliers", type=str, default="", help="comma-separated list, e.g. 0.1,0.5,1.0")
    parser.add_argument("--outdir", type=str, default=DEFAULT_OUTDIR, help="output directory")
    args = parser.parse_args()

    kf = _parse_float_list(args.keff_factors) if args.keff_factors else None
    em = _parse_float_list(args.eps_multipliers) if args.eps_multipliers else None

    run_phaseout_map(
        fast=bool(args.fast),
        rel_tol=float(args.rel_tol),
        max_bisect=int(args.max_bisect),
        gamma_grid_size=int(args.gamma_grid_size),
        resume=not bool(args.no_resume),
        keff_factors=kf,
        eps_multipliers=em,
        outdir=str(args.outdir),
    )