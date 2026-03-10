# -*- coding: utf-8 -*-
"""
run_scenarios.py
----------------
Campaign-scale batch scenario runner (no tornado).

Purpose
-------
This script generates batch results for campaign-scale scenario comparisons,
including representative policy variants for each scenario case.

It is intended to support:
- baseline architecture comparison tables,
- representative scenario/policy summaries,
- machine-readable batch outputs for the paper and supplementary materials.

Model layer
-----------
campaign_scale

IMPORTANT
---------
This script uses:
    Params + preset_mode + build_cases + tau_star_bisect + solve_policy

It is NOT the reduced paper-level / mature-regime runner.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from revised_solver_core import (
    OUTDIR,
    Params,
    preset_mode,
    build_cases,
    solve_policy,
    tau_star_bisect,
)


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


def _is_valid_tau(x: Any) -> bool:
    try:
        xf = float(x)
    except Exception:
        return False
    return math.isfinite(xf) and xf > 0.0


def _attach_metadata(res: Dict[str, Any], *, label: str, run_mode: str, entrypoint: str) -> Dict[str, Any]:
    res["model_layer"] = "campaign_scale"
    res["entrypoint"] = entrypoint
    res["case_label_input"] = label
    res["run_mode_input"] = run_mode
    return res


def _solve_case_bundle(
    label: str,
    cfg: Params,
    *,
    run_mode: str,
    write_timeseries: bool,
) -> List[Dict[str, Any]]:
    tau_star = tau_star_bisect(cfg, label=label)
    if not _is_valid_tau(tau_star):
        raise RuntimeError(f"tau_star_bisect returned invalid value for case '{label}': {tau_star}")

    r_tau = solve_policy(
        cfg,
        label,
        policy="tau_star",
        tau_fixed=float(tau_star),
        write_timeseries=write_timeseries,
    )
    r_tau["tau_star_bisect (yr)"] = float(tau_star)
    _attach_metadata(r_tau, label=label, run_mode=run_mode, entrypoint="run_scenarios.py")

    r_time = solve_policy(
        cfg,
        label,
        policy="time_opt",
        write_timeseries=write_timeseries,
    )
    r_time["tau_star_bisect (yr)"] = float(tau_star)
    _attach_metadata(r_time, label=label, run_mode=run_mode, entrypoint="run_scenarios.py")

    r_cost = solve_policy(
        cfg,
        label,
        policy="cost_opt",
        tau_star=float(tau_star),
        write_timeseries=write_timeseries,
    )
    r_cost["tau_star_bisect (yr)"] = float(tau_star)
    _attach_metadata(r_cost, label=label, run_mode=run_mode, entrypoint="run_scenarios.py")

    return [r_tau, r_time, r_cost]


def _check_strict_pass(df: pd.DataFrame) -> bool:
    required_flags = ["SolverSuccess", "ConstraintOK", "BuildOK", "OpsOK"]
    for col in required_flags:
        if col not in df.columns:
            return False
        if not df[col].fillna(False).astype(str).eq("YES").all():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Campaign-scale batch scenario runner"
    )
    parser.add_argument("--run-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--failure-timing", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--only-case", default="", help="optional exact case label to run")
    parser.add_argument("--no-timeseries", action="store_true")
    parser.add_argument("--out-csv", type=str, default="", help="optional output CSV path")
    parser.add_argument("--meta-json", type=str, default="", help="optional metadata JSON path")
    parser.add_argument("--print-table", action="store_true", help="print the summary table to stdout")
    parser.add_argument("--strict", action="store_true",
                        help="return nonzero exit code if any row fails key feasibility flags")
    args = parser.parse_args()

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    base = preset_mode(Params(failure_timing_mode=args.failure_timing), args.run_mode)

    cases: List[Tuple[str, Params]] = build_cases(base)
    if args.only_case:
        cases = [x for x in cases if x[0] == args.only_case]
        if not cases:
            raise ValueError(f"case not found: {args.only_case}")

    rows: List[Dict[str, Any]] = []
    case_labels: List[str] = []

    for label, cfg in cases:
        print(f"--- Solving {label} ({args.run_mode}, {cfg.failure_timing_mode})")
        case_labels.append(label)
        rows.extend(
            _solve_case_bundle(
                label=label,
                cfg=cfg,
                run_mode=args.run_mode,
                write_timeseries=not args.no_timeseries,
            )
        )

    df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv) if args.out_csv else (outdir / "final_revised_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    meta = {
        "model_layer": "campaign_scale",
        "entrypoint": "run_scenarios.py",
        "notes": [
            "This script runs campaign-scale batch scenario comparisons only.",
            "It should not be confused with reduced-model robustness or reduced regime-map sweeps.",
            "Each case is solved for tau_star, time_opt, and cost_opt.",
        ],
        "run_mode": args.run_mode,
        "failure_timing": args.failure_timing,
        "only_case": args.only_case,
        "write_timeseries": not bool(args.no_timeseries),
        "n_cases": int(len(cases)),
        "case_labels": case_labels,
        "n_rows": int(len(df)),
        "base_params": asdict(base),
        "out_csv": str(out_csv),
    }

    out_json = Path(args.meta_json) if args.meta_json else (outdir / "scenario_batch_meta.json")
    _write_json(out_json, meta)

    print(f"[OK] wrote summary CSV: {out_csv}")
    print(f"[OK] wrote metadata JSON: {out_json}")

    if args.print_table:
        cols = [c for c in [
            "ScenarioLabel", "Policy", "ScenarioMode", "FailureTiming",
            "SolverSuccess", "ConstraintOK", "BuildOK", "OpsOK",
            "tau_star_bisect (yr)", "tau_build (yr)", "Time_total (yr)",
            "Cost_total ($B)", "E_total (index)", "TimeseriesCSV"
        ] if c in df.columns]
        if cols:
            print(df[cols].to_string(index=False))
        else:
            print(df.to_string(index=False))

    if args.strict and not _check_strict_pass(df):
        raise SystemExit(3)


if __name__ == "__main__":
    main()