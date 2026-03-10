# -*- coding: utf-8 -*-
"""
run_single_case.py
------------------
Run one campaign-scale case + one policy only.

This script is intended for:
- reproducing an individual scenario/policy pair,
- checking a single table row or figure source case,
- providing a minimal command-line entry point in the README.

IMPORTANT
---------
This script uses the campaign-scale model:
    Params + preset_mode + tau_star_bisect + solve_policy

It is NOT the reduced paper-level / mature-regime runner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from revised_solver_core import Params, preset_mode, solve_policy, tau_star_bisect


DEFAULT_KEYS = [
    "ScenarioLabel",
    "Policy",
    "ScenarioMode",
    "FailureTiming",
    "SE_Disrupt",
    "Rocket_Disrupt",
    "SE_Triggered",
    "Rocket_Triggered",
    "SolverSuccess",
    "ConstraintOK",
    "BuildOK",
    "OpsOK",
    "tau_ub_eps (yr)",
    "tau_build (yr)",
    "Time_total (yr)",
    "Cost_build ($B)",
    "Cost_ops1yr ($B)",
    "Cost_total ($B)",
    "E_total (index)",
    "EnvPenalty",
    "TimeseriesCSV",
]


def _safe_jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable values."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_jsonable(v) for v in obj]
    try:
        return float(obj)
    except Exception:
        return str(obj)


def _print_selected(res: Dict[str, Any], keys: Iterable[str]) -> None:
    for k in keys:
        if k in res:
            print(f"{k}: {res[k]}")


def _is_valid_tau(x: Any) -> bool:
    try:
        xf = float(x)
    except Exception:
        return False
    return math.isfinite(xf) and xf > 0.0


def _flag_is_yes(x: Any) -> bool:
    """
    Normalize solver-style flags such as YES/NO, True/False, 1/0.
    """
    if isinstance(x, bool):
        return x
    if x is None:
        return False
    s = str(x).strip().upper()
    return s in {"YES", "TRUE", "1"}


def _build_params(args: argparse.Namespace) -> Params:
    p = preset_mode(Params(failure_timing_mode=args.failure_timing), args.run_mode)
    p = replace(
        p,
        scenario=args.scenario,
        se_failure_active=bool(args.se_fail),
        rocket_failure_active=bool(args.rocket_fail),
    )
    return p


def _solve_single_case(args: argparse.Namespace) -> Dict[str, Any]:
    p = _build_params(args)

    tau_star_val: Optional[float] = None
    tau_fixed_val: Optional[float] = None

    if args.tau_fixed is not None:
        tau_fixed_val = float(args.tau_fixed)
        if not _is_valid_tau(tau_fixed_val):
            raise ValueError("--tau-fixed must be a positive finite number.")

    if args.policy == "tau_star":
        tau_star_val = tau_star_bisect(p, label=args.label)
        if not _is_valid_tau(tau_star_val):
            raise RuntimeError("tau_star_bisect failed to return a valid positive finite horizon.")
        res = solve_policy(
            p,
            label=args.label,
            policy="tau_star",
            tau_fixed=float(tau_star_val),
            write_timeseries=bool(args.write_timeseries),
        )
        res["tau_star_bisect (yr)"] = float(tau_star_val)

    elif args.policy == "cost_opt":
        if tau_fixed_val is not None:
            res = solve_policy(
                p,
                label=args.label,
                policy="cost_opt",
                tau_star=float(tau_fixed_val),
                write_timeseries=bool(args.write_timeseries),
            )
            res["tau_input_for_cost_opt (yr)"] = float(tau_fixed_val)
        else:
            tau_star_val = tau_star_bisect(p, label=args.label)
            if not _is_valid_tau(tau_star_val):
                raise RuntimeError("tau_star_bisect failed to return a valid positive finite horizon.")
            res = solve_policy(
                p,
                label=args.label,
                policy="cost_opt",
                tau_star=float(tau_star_val),
                write_timeseries=bool(args.write_timeseries),
            )
            res["tau_star_bisect (yr)"] = float(tau_star_val)

    elif args.policy == "time_opt":
        if tau_fixed_val is not None:
            raise ValueError("--tau-fixed is not used with policy=time_opt.")
        res = solve_policy(
            p,
            label=args.label,
            policy="time_opt",
            write_timeseries=bool(args.write_timeseries),
        )

    else:
        raise ValueError(f"Unsupported policy: {args.policy}")

    # Explicit metadata for downstream reproducibility
    res["model_layer"] = "campaign_scale"
    res["entrypoint"] = "run_single_case.py"
    res["input_label"] = args.label
    res["input_policy"] = args.policy
    res["input_run_mode"] = args.run_mode
    res["input_failure_timing"] = args.failure_timing
    res["input_scenario"] = args.scenario
    res["input_se_fail"] = bool(args.se_fail)
    res["input_rocket_fail"] = bool(args.rocket_fail)
    res["input_write_timeseries"] = bool(args.write_timeseries)

    try:
        res["params_snapshot"] = asdict(p)
    except Exception:
        res["params_snapshot"] = str(p)

    strict_pass = (
        _flag_is_yes(res.get("SolverSuccess"))
        and _flag_is_yes(res.get("ConstraintOK"))
        and _flag_is_yes(res.get("BuildOK"))
        and _flag_is_yes(res.get("OpsOK"))
    )
    res["strict_pass"] = bool(strict_pass)

    return res


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one campaign-scale case and one policy"
    )
    parser.add_argument("--run-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--failure-timing", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--scenario", choices=["mixed", "se_only", "rocket_only"], default="mixed")
    parser.add_argument("--se-fail", action="store_true")
    parser.add_argument("--rocket-fail", action="store_true")
    parser.add_argument("--policy", choices=["tau_star", "time_opt", "cost_opt"], default="cost_opt")
    parser.add_argument(
        "--tau-fixed",
        type=float,
        default=None,
        help="optional fixed horizon for cost_opt; useful for row-level reproduction",
    )
    parser.add_argument("--label", default="single_case")
    parser.add_argument("--write-timeseries", action="store_true",
                        help="write timeseries output if supported by solve_policy")

    parser.add_argument("--print-json", action="store_true",
                        help="print the full result as JSON instead of the compact text summary")
    parser.add_argument("--summary-json", type=str, default="",
                        help="optional path to save the full result dictionary as JSON")
    parser.add_argument("--strict", action="store_true",
                        help="exit with nonzero status if solver/key feasibility flags indicate failure")

    args = parser.parse_args()

    try:
        res = _solve_single_case(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if args.print_json:
        print(json.dumps(_safe_jsonable(res), indent=2, ensure_ascii=False))
    else:
        _print_selected(res, DEFAULT_KEYS)
        for extra_key in [
            "tau_star_bisect (yr)",
            "tau_input_for_cost_opt (yr)",
            "model_layer",
            "entrypoint",
            "strict_pass",
        ]:
            if extra_key in res:
                print(f"{extra_key}: {res[extra_key]}")

    if args.summary_json:
        outpath = Path(args.summary_json)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(_safe_jsonable(res), f, indent=2, ensure_ascii=False)
        print(f"[OK] wrote summary JSON: {outpath}")

    if args.strict and not bool(res.get("strict_pass", False)):
        raise SystemExit(3)


if __name__ == "__main__":
    main()