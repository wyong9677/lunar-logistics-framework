# -*- coding: utf-8 -*-
"""
run_bridge_sensitivity.py
-------------------------
Campaign-scale bridge sensitivity sweep for the Scientific Reports manuscript.

Purpose
-------
This script fills the methodological gap between:
1) campaign-scale headline scenario results, and
2) reduced-model mechanism diagnostics.

It performs a small multi-level parameter sweep directly on the campaign-scale
full model, so that the manuscript can show whether the main baseline regime
classification (e.g. rocket-dominant fast-build behavior) is preserved under
structured perturbations of a few key parameters.

Typical use
-----------
- sweep K_max_total_tpy, delta_eff, L_R_ton_per_launch, gamma_env, C_E_dollars_per_launch
- evaluate tau_star and cost_opt (optionally time_opt)
- compare mixed_base and selected disruption cases
- report regime label from direct-vs-transshipment lunar-delivered route share

Outputs
-------
- bridge_sensitivity_summary.csv
- bridge_sensitivity_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from revised_solver_core import (
    OUTDIR,
    Params,
    preset_mode,
    build_cases,
    tau_star_bisect,
    solve_policy,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTDIR = Path(os.environ.get("PAPER_OUTDIR", str(ROOT / "paper_outputs")))


VALID_PARAMS = {
    "K_max_total_tpy",
    "alpha",
    "r",
    "delta_eff",
    "L_R_ton_per_launch",
    "C_E_dollars_per_launch",
    "C_A_dollars_per_launch",
    "C_SE_dollars_per_ton",
    "gamma_env",
}

VALID_POLICIES = {"tau_star", "cost_opt", "time_opt"}


def _parse_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_float_list(s: str) -> List[float]:
    vals: List[float] = []
    for x in s.split(","):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def _validate_params(params: List[str]) -> None:
    bad = [p for p in params if p not in VALID_PARAMS]
    if bad:
        raise ValueError(f"Unsupported parameters: {bad}\nValid choices: {sorted(VALID_PARAMS)}")


def _validate_policies(policies: List[str]) -> None:
    bad = [p for p in policies if p not in VALID_POLICIES]
    if bad:
        raise ValueError(f"Unsupported policies: {bad}\nValid choices: {sorted(VALID_POLICIES)}")


def _clip_param_value(param: str, value: float) -> float:
    """
    Keep bounded parameters in physically / numerically valid ranges.
    """
    if param == "alpha":
        return float(np.clip(value, 1e-4, 0.999))
    if param == "delta_eff":
        return float(np.clip(value, 1e-4, 0.999))
    if param == "gamma_env":
        return float(max(value, 0.0))
    return float(value)


def _route_share_and_regime(row: Dict) -> Tuple[float, str]:
    """
    Route-share proxy from lunar-delivered launch totals.

    Since both direct and Apex->Moon launches carry L_R payload to the Moon,
    the lunar-delivered mass share is proportional to total FE vs FA launches.
    """
    fe = _safe_float(row.get("Total_FE_Launches", np.nan))
    fa = _safe_float(row.get("Total_FA_Launches", np.nan))
    denom = fe + fa
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan"), "unknown"

    chi_r = fe / denom
    if chi_r >= 0.9:
        regime = "rocket-dominant"
    elif chi_r <= 0.1:
        regime = "SE-assisted"
    else:
        regime = "mixed"
    return float(chi_r), regime


def _build_case_map(base: Params) -> Dict[str, Params]:
    return {label: cfg for label, cfg in build_cases(base)}


def _solve_selected_policies(
    p: Params,
    label: str,
    policies: List[str],
    write_timeseries: bool = False,
) -> List[Dict]:
    rows: List[Dict] = []

    tau_star = np.nan
    if "tau_star" in policies or "cost_opt" in policies:
        tau_star = tau_star_bisect(p, label=label)
        if not np.isfinite(tau_star):
            raise RuntimeError(f"tau_star_bisect failed for case '{label}'")

    if "tau_star" in policies:
        r_tau = solve_policy(
            p,
            label,
            policy="tau_star",
            tau_fixed=float(tau_star),
            write_timeseries=write_timeseries,
        )
        r_tau["tau_star_bisect (yr)"] = float(tau_star)
        rows.append(r_tau)

    if "time_opt" in policies:
        r_time = solve_policy(
            p,
            label,
            policy="time_opt",
            write_timeseries=write_timeseries,
        )
        r_time["tau_star_bisect (yr)"] = float(tau_star) if np.isfinite(tau_star) else np.nan
        rows.append(r_time)

    if "cost_opt" in policies:
        r_cost = solve_policy(
            p,
            label,
            policy="cost_opt",
            tau_star=float(tau_star),
            write_timeseries=write_timeseries,
        )
        r_cost["tau_star_bisect (yr)"] = float(tau_star)
        rows.append(r_cost)

    return rows


# ---------------------------------------------------------------------
# Main bridge sweep
# ---------------------------------------------------------------------
def run_bridge_sensitivity(
    *,
    run_mode: str = "full",
    failure_timing: str = "relative",
    cases: List[str] | None = None,
    params: List[str] | None = None,
    factors: List[float] | None = None,
    policies: List[str] | None = None,
    write_timeseries: bool = False,
    outdir: Path = DEFAULT_OUTDIR,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    base = Params()
    base = preset_mode(base, run_mode)
    base.failure_timing_mode = str(failure_timing)

    case_map = _build_case_map(base)

    if cases is None:
        cases = ["mixed_base"]
    for c in cases:
        if c not in case_map:
            raise ValueError(f"Unknown case '{c}'. Available cases: {sorted(case_map.keys())}")

    if params is None:
        params = [
            "K_max_total_tpy",
            "delta_eff",
            "L_R_ton_per_launch",
            "gamma_env",
            "C_E_dollars_per_launch",
        ]
    _validate_params(params)

    if factors is None:
        factors = [0.6, 0.8, 1.0, 1.2, 1.5]
    factors = list(map(float, factors))

    if policies is None:
        policies = ["tau_star", "cost_opt"]
    _validate_policies(policies)

    rows: List[Dict] = []

    for case_label in cases:
        case_cfg = case_map[case_label]

        for param in params:
            baseline_value = getattr(case_cfg, param)

            for factor in factors:
                trial_value = _clip_param_value(param, baseline_value * factor)
                p2 = replace(case_cfg, **{param: trial_value})

                label = f"bridge_{case_label}_{param}_{factor:g}"

                try:
                    solved = _solve_selected_policies(
                        p2,
                        label=label,
                        policies=policies,
                        write_timeseries=write_timeseries,
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "model_layer": "campaign_scale",
                            "entrypoint": "run_bridge_sensitivity.py",
                            "case_label": case_label,
                            "param": param,
                            "baseline_value": baseline_value,
                            "factor": factor,
                            "trial_value": trial_value,
                            "policy": "ALL_REQUESTED",
                            "status": "FAILED",
                            "error": str(exc),
                        }
                    )
                    continue

                for rr in solved:
                    chi_r, regime = _route_share_and_regime(rr)

                    rows.append(
                        {
                            "model_layer": "campaign_scale",
                            "entrypoint": "run_bridge_sensitivity.py",
                            "case_label": case_label,
                            "ScenarioMode": rr.get("ScenarioMode", ""),
                            "FailureTiming": rr.get("FailureTiming", ""),
                            "SE_Disrupt": rr.get("SE_Disrupt", ""),
                            "Rocket_Disrupt": rr.get("Rocket_Disrupt", ""),
                            "param": param,
                            "baseline_value": baseline_value,
                            "factor": factor,
                            "trial_value": trial_value,
                            "policy": rr.get("Policy", ""),
                            "tau_star_bisect (yr)": _safe_float(rr.get("tau_star_bisect (yr)", np.nan)),
                            "tau_build (yr)": _safe_float(rr.get("tau_build (yr)", np.nan)),
                            "Time_total (yr)": _safe_float(rr.get("Time_total (yr)", np.nan)),
                            "TotalCost ($B)": _safe_float(rr.get("TotalCost ($B)", np.nan)),
                            "E_total (index)": _safe_float(rr.get("E_total (index)", np.nan)),
                            "Total_FE_Launches": _safe_float(rr.get("Total_FE_Launches", np.nan)),
                            "Total_FA_Launches": _safe_float(rr.get("Total_FA_Launches", np.nan)),
                            "Total_SE_Throughput (tons)": _safe_float(rr.get("Total_SE_Throughput (tons)", np.nan)),
                            "Apex_Inventory_Peak (tons)": _safe_float(rr.get("Apex_Inventory_Peak (tons)", np.nan)),
                            "route_share_direct": chi_r,
                            "regime_label": regime,
                            "SolverSuccess": rr.get("SolverSuccess", ""),
                            "ConstraintOK": rr.get("ConstraintOK", ""),
                            "BuildOK": rr.get("BuildOK", ""),
                            "OpsOK": rr.get("OpsOK", ""),
                            "ReturnStatus": rr.get("ReturnStatus", ""),
                            "status": "OK",
                        }
                    )

    df = pd.DataFrame(rows)
    out_csv = outdir / "bridge_sensitivity_summary.csv"
    df.to_csv(out_csv, index=False)

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": "run_bridge_sensitivity.py",
        "interpretation_scope": (
            "Campaign-scale bridge sensitivity intended to connect the main "
            "scenario results to reduced-model diagnostics. These outputs are "
            "campaign-scale full-model results, not reduced-model approximations."
        ),
        "run_mode": run_mode,
        "failure_timing": failure_timing,
        "cases": cases,
        "params": params,
        "factors": factors,
        "policies": policies,
        "write_timeseries": bool(write_timeseries),
        "output_csv": str(out_csv),
    }
    out_meta = outdir / "bridge_sensitivity_meta.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote summary CSV: {out_csv}")
    print(f"[OK] wrote metadata JSON: {out_meta}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Campaign-scale bridge sensitivity sweep for key parameters"
    )
    parser.add_argument("--run-mode", choices=["fast", "full"], default="full")
    parser.add_argument("--failure-timing", choices=["absolute", "relative"], default="relative")
    parser.add_argument(
        "--cases",
        type=str,
        default="mixed_base",
        help="comma-separated case labels, e.g. mixed_base,mixed_sefail,mixed_rocketfail,mixed_bothfail",
    )
    parser.add_argument(
        "--params",
        type=str,
        default="K_max_total_tpy,delta_eff,L_R_ton_per_launch,gamma_env,C_E_dollars_per_launch",
        help="comma-separated campaign-scale Params fields to sweep",
    )
    parser.add_argument(
        "--factors",
        type=str,
        default="0.6,0.8,1.0,1.2,1.5",
        help="comma-separated multiplicative factors",
    )
    parser.add_argument(
        "--policies",
        type=str,
        default="tau_star,cost_opt",
        help="comma-separated policies from {tau_star,time_opt,cost_opt}",
    )
    parser.add_argument(
        "--write-timeseries",
        action="store_true",
        help="also write timeseries CSVs for each solve (off by default)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help="output directory",
    )
    args = parser.parse_args()

    run_bridge_sensitivity(
        run_mode=str(args.run_mode),
        failure_timing=str(args.failure_timing),
        cases=_parse_list(args.cases),
        params=_parse_list(args.params),
        factors=_parse_float_list(args.factors),
        policies=_parse_list(args.policies),
        write_timeseries=bool(args.write_timeseries),
        outdir=Path(args.outdir),
    )


if __name__ == "__main__":
    main()