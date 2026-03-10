# -*- coding: utf-8 -*-
"""
run_consistency_check.py
------------------------
Campaign-scale numerical consistency audit for the revised solver.

Purpose
-------
This script evaluates numerical stability under multiple randomized restarts:
- randomized initial guesses,
- perturbed initial horizon guesses,
- repeated solves of the same scenario-policy pair.

It is intended to support:
- Methods: numerical stability / verification,
- Supplementary Information: solver consistency audit,
- reproducibility package: machine-readable audit ledgers.

Model layer
-----------
campaign_scale

Outputs (default OUTDIR from revised_solver_core, unless overridden):
- consistency_report.csv
- consistency_summary.txt
- consistency_summary.json
- consistency_meta.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from revised_solver_core import OUTDIR, Params, preset_mode, solve_policy, tau_star_bisect


def rel_spread(x: np.ndarray) -> float:
    """
    Relative spread = (max - min) / max(|mean|, 1e-12)
    Returns NaN if input is empty.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan
    denom = max(abs(float(np.mean(x))), 1e-12)
    return float((np.max(x) - np.min(x)) / denom)


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _is_valid_tau(x: Any) -> bool:
    try:
        xf = float(x)
    except Exception:
        return False
    return math.isfinite(xf) and xf > 0.0


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    try:
        return float(obj)
    except Exception:
        return str(obj)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, ensure_ascii=False)


def _build_case(args: argparse.Namespace) -> Params:
    p = preset_mode(Params(failure_timing_mode=args.failure_timing), args.run_mode)
    p = replace(
        p,
        scenario=args.scenario,
        se_failure_active=bool(args.se_fail),
        rocket_failure_active=bool(args.rocket_fail),
    )
    return p


def _solve_one_restart(
    p: Params,
    *,
    policy: str,
    label: str,
    seed_i: int,
    tau0_i: float,
    init_noise: float,
    tau_star_ref: Optional[float],
    write_timeseries: bool,
) -> Dict[str, Any]:
    if policy == "tau_star":
        tau_star_i = tau_star_bisect(
            p,
            label=label,
            init_noise_scale=init_noise,
            init_seed_base=seed_i,
        )
        if not _is_valid_tau(tau_star_i):
            raise RuntimeError(f"tau_star_bisect failed for restart label={label}, got {tau_star_i}")
        res = solve_policy(
            p,
            label=label,
            policy="tau_star",
            tau_fixed=float(tau_star_i),
            tau0_override=tau0_i,
            init_noise_scale=init_noise,
            init_seed=seed_i,
            write_timeseries=bool(write_timeseries),
        )
        res["tau_star_bisect (yr)"] = float(tau_star_i)

    elif policy == "cost_opt":
        if tau_star_ref is None or not _is_valid_tau(tau_star_ref):
            raise RuntimeError("cost_opt consistency audit requires a valid tau_star_ref.")
        res = solve_policy(
            p,
            label=label,
            policy="cost_opt",
            tau_star=float(tau_star_ref),
            tau0_override=tau0_i,
            init_noise_scale=init_noise,
            init_seed=seed_i,
            write_timeseries=bool(write_timeseries),
        )
        res["tau_star_bisect (yr)"] = float(tau_star_ref)

    elif policy == "time_opt":
        res = solve_policy(
            p,
            label=label,
            policy="time_opt",
            tau0_override=tau0_i,
            init_noise_scale=init_noise,
            init_seed=seed_i,
            write_timeseries=bool(write_timeseries),
        )

    else:
        raise ValueError(f"Unsupported policy: {policy}")

    return res


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Campaign-scale numerical consistency audit under randomized restarts"
    )
    parser.add_argument("--run-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--failure-timing", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--scenario", choices=["mixed", "se_only", "rocket_only"], default="mixed")
    parser.add_argument("--se-fail", action="store_true")
    parser.add_argument("--rocket-fail", action="store_true")
    parser.add_argument("--policy", choices=["tau_star", "time_opt", "cost_opt"], default="cost_opt")

    parser.add_argument("--n-runs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--init-noise", type=float, default=0.10,
                        help="randomized initial-guess perturbation scale passed into the solver")
    parser.add_argument("--tau0-jitter", type=float, default=0.10,
                        help="Gaussian relative perturbation of the initial horizon guess tau0")
    parser.add_argument("--tol-rel-cost", type=float, default=0.03)
    parser.add_argument("--tol-rel-tau", type=float, default=0.03)

    parser.add_argument("--write-timeseries", action="store_true")
    parser.add_argument("--outdir", type=str, default=str(OUTDIR))
    parser.add_argument("--label-prefix", type=str, default="cons")
    parser.add_argument("--strict", action="store_true",
                        help="return nonzero exit code if CONSISTENCY_PASS is False")
    parser.add_argument("--print-table", action="store_true",
                        help="print the full restart report table to stdout")

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    p = _build_case(args)

    n_runs = max(int(args.n_runs), 2)
    init_noise = max(float(args.init_noise), 0.0)
    tau0_jitter = max(float(args.tau0_jitter), 0.0)

    tau_star_ref: Optional[float] = None
    if args.policy == "cost_opt":
        tau_star_ref = tau_star_bisect(p, label=f"{args.label_prefix}_ref")
        if not _is_valid_tau(tau_star_ref):
            raise RuntimeError(f"Reference tau_star_bisect failed for cost_opt audit: {tau_star_ref}")

    rng = np.random.default_rng(int(args.seed))
    base_tau0 = max(8.0, float(p.tau_min) * 2.0)

    rows: List[Dict[str, Any]] = []

    for i in range(n_runs):
        seed_i = int(args.seed + i)
        tau0_i = float(
            np.clip(
                base_tau0 * (1.0 + tau0_jitter * rng.standard_normal()),
                p.tau_min,
                p.tau_max,
            )
        )
        label = f"{args.label_prefix}_{i:02d}"

        res = _solve_one_restart(
            p,
            policy=args.policy,
            label=label,
            seed_i=seed_i,
            tau0_i=tau0_i,
            init_noise=init_noise,
            tau_star_ref=tau_star_ref,
            write_timeseries=bool(args.write_timeseries),
        )

        rows.append({
            "model_layer": "campaign_scale",
            "entrypoint": "run_consistency_check.py",
            "run_id": int(i),
            "seed": int(seed_i),
            "tau0_used": float(tau0_i),
            "init_noise": float(init_noise),
            "policy": args.policy,
            "scenario": args.scenario,
            "run_mode": args.run_mode,
            "failure_timing": args.failure_timing,
            "se_fail": bool(args.se_fail),
            "rocket_fail": bool(args.rocket_fail),
            "SolverSuccess": res.get("SolverSuccess", "NO"),
            "ConstraintOK": res.get("ConstraintOK", "NO"),
            "BuildOK": res.get("BuildOK", "NO"),
            "OpsOK": res.get("OpsOK", "NO"),
            "tau_star_bisect (yr)": _safe_float(res.get("tau_star_bisect (yr)", np.nan)),
            "tau_build (yr)": _safe_float(res.get("tau_build (yr)", np.nan)),
            "Time_total (yr)": _safe_float(res.get("Time_total (yr)", np.nan)),
            "Cost_total ($B)": _safe_float(res.get("Cost_total ($B)", np.nan)),
            "E_total (index)": _safe_float(res.get("E_total (index)", np.nan)),
            "Viol_scaled": _safe_float(res.get("Viol_scaled", np.nan)),
            "ReturnStatus": res.get("ReturnStatus", ""),
            "TimeseriesCSV": res.get("TimeseriesCSV", ""),
        })

    df = pd.DataFrame(rows)

    out_csv = outdir / "consistency_report.csv"
    df.to_csv(out_csv, index=False)

    ok_mask = (
        (df["SolverSuccess"] == "YES")
        & (df["ConstraintOK"] == "YES")
        & (df["BuildOK"] == "YES")
        & (df["OpsOK"] == "YES")
    )

    tau_rel = rel_spread(df.loc[ok_mask, "tau_build (yr)"].to_numpy())
    cost_rel = rel_spread(df.loc[ok_mask, "Cost_total ($B)"].to_numpy())

    all_feasible = bool(ok_mask.all())
    feasible_fraction = float(ok_mask.mean()) if len(ok_mask) > 0 else float("nan")

    pass_tau = bool(np.isfinite(tau_rel) and tau_rel <= float(args.tol_rel_tau))
    pass_cost = bool(np.isfinite(cost_rel) and cost_rel <= float(args.tol_rel_cost))
    passed = bool(all_feasible and pass_tau and pass_cost)

    summary = {
        "model_layer": "campaign_scale",
        "entrypoint": "run_consistency_check.py",
        "policy": args.policy,
        "scenario": args.scenario,
        "run_mode": args.run_mode,
        "failure_timing": args.failure_timing,
        "se_fail": bool(args.se_fail),
        "rocket_fail": bool(args.rocket_fail),
        "n_runs": int(n_runs),
        "seed": int(args.seed),
        "init_noise": float(init_noise),
        "tau0_jitter": float(tau0_jitter),
        "write_timeseries": bool(args.write_timeseries),
        "all_feasible": all_feasible,
        "feasible_fraction": feasible_fraction,
        "tau_rel_spread": float(tau_rel) if np.isfinite(tau_rel) else None,
        "cost_rel_spread": float(cost_rel) if np.isfinite(cost_rel) else None,
        "tol_rel_tau": float(args.tol_rel_tau),
        "tol_rel_cost": float(args.tol_rel_cost),
        "CONSISTENCY_PASS": passed,
        "report_csv": str(out_csv),
    }

    summary_lines = [
        f"policy={summary['policy']}",
        f"scenario={summary['scenario']}",
        f"run_mode={summary['run_mode']}",
        f"failure_timing={summary['failure_timing']}",
        f"n_runs={summary['n_runs']}",
        f"all_feasible={summary['all_feasible']}",
        f"feasible_fraction={summary['feasible_fraction']:.6f}",
        f"tau_rel_spread={summary['tau_rel_spread'] if summary['tau_rel_spread'] is not None else 'nan'}",
        f"cost_rel_spread={summary['cost_rel_spread'] if summary['cost_rel_spread'] is not None else 'nan'}",
        f"tol_rel_tau={summary['tol_rel_tau']:.6f}",
        f"tol_rel_cost={summary['tol_rel_cost']:.6f}",
        f"CONSISTENCY_PASS={summary['CONSISTENCY_PASS']}",
        f"report_csv={summary['report_csv']}",
    ]

    out_txt = outdir / "consistency_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")

    out_json = outdir / "consistency_summary.json"
    _write_json(out_json, summary)

    meta = {
        "base_params": asdict(p),
        "label_prefix": args.label_prefix,
        "base_tau0": float(base_tau0),
        "tau_star_ref": float(tau_star_ref) if tau_star_ref is not None else None,
        "notes": [
            "This script audits numerical consistency under randomized initial guesses and perturbed initial horizon guesses.",
            "The pass condition currently requires: all runs feasible, tau relative spread <= tol_rel_tau, and cost relative spread <= tol_rel_cost.",
            "This is a campaign-scale consistency audit and should not be confused with reduced-model robustness runs.",
        ],
    }
    out_meta = outdir / "consistency_meta.json"
    _write_json(out_meta, meta)

    if args.print_table:
        print(df.to_string(index=False))

    print("\n".join(summary_lines))
    print(f"[OK] wrote summary TXT:  {out_txt}")
    print(f"[OK] wrote summary JSON: {out_json}")
    print(f"[OK] wrote metadata JSON: {out_meta}")

    if args.strict and not passed:
        raise SystemExit(3)


if __name__ == "__main__":
    main()