# -*- coding: utf-8 -*-
"""
final_revised_solver.py
-----------------------
Compatibility entrypoint for the campaign-scale batch scenario runner.

IMPORTANT
---------
This script keeps the old command name for backward compatibility, but it is
not the core solver implementation. All numerical logic is delegated to
revised_solver_core.

Model layer:
- campaign_scale

Recommended usage going forward:
- use run_scenarios.py for the main batch scenario tables,
- use run_single_case.py for single-case reproduction,
- use dedicated scripts for tornado / consistency / reduced-model analyses.

This wrapper remains useful when older commands or external notes still refer
to 'final_revised_solver.py'.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from revised_solver_core import (
    OUTDIR,
    Params,
    build_cases,
    preset_mode,
    run_tornado,
    solve_policy,
    tau_star_bisect,
)


def _is_valid_tau(x: Any) -> bool:
    try:
        xf = float(x)
    except Exception:
        return False
    return math.isfinite(xf) and xf > 0.0


def _attach_metadata(res: Dict[str, Any], *, label: str, entrypoint: str) -> Dict[str, Any]:
    res["model_layer"] = "campaign_scale"
    res["entrypoint"] = entrypoint
    res["case_label_input"] = label
    return res


def _solve_case_bundle(
    label: str,
    cfg: Params,
    *,
    write_timeseries: bool,
    strict_tau: bool = True,
) -> List[Dict[str, Any]]:
    tau_star = tau_star_bisect(cfg, label=label)
    if strict_tau and not _is_valid_tau(tau_star):
        raise RuntimeError(f"tau_star_bisect returned invalid value for case '{label}': {tau_star}")

    r_tau = solve_policy(
        cfg,
        label,
        policy="tau_star",
        tau_fixed=float(tau_star),
        write_timeseries=write_timeseries,
    )
    r_tau["tau_star_bisect (yr)"] = float(tau_star)
    _attach_metadata(r_tau, label=label, entrypoint="final_revised_solver.py")

    r_time = solve_policy(
        cfg,
        label,
        policy="time_opt",
        write_timeseries=write_timeseries,
    )
    r_time["tau_star_bisect (yr)"] = float(tau_star)
    _attach_metadata(r_time, label=label, entrypoint="final_revised_solver.py")

    r_cost = solve_policy(
        cfg,
        label,
        policy="cost_opt",
        tau_star=float(tau_star),
        write_timeseries=write_timeseries,
    )
    r_cost["tau_star_bisect (yr)"] = float(tau_star)
    _attach_metadata(r_cost, label=label, entrypoint="final_revised_solver.py")

    return [r_tau, r_time, r_cost]


def _check_strict_pass(df: pd.DataFrame) -> bool:
    def _flag_is_true(v: Any) -> bool:
        if isinstance(v, str):
            s = v.strip().lower()
            if s in {"yes", "true", "1"}:
                return True
            if s in {"no", "false", "0", ""}:
                return False
        if pd.isna(v):
            return False
        return bool(v)

    required_flags = ["SolverSuccess", "ConstraintOK", "BuildOK", "OpsOK"]
    for col in required_flags:
        if col not in df.columns:
            return False
        if not df[col].map(_flag_is_true).all():
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compatibility batch entrypoint delegating to revised_solver_core"
    )
    parser.add_argument("--run-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--failure-timing", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--no-timeseries", action="store_true")
    parser.add_argument("--tornado", action="store_true", help="also run campaign-scale mixed-case tornado")
    parser.add_argument("--delta", type=float, default=0.10, help="relative perturbation size for tornado")
    parser.add_argument("--se-fail", action="store_true", help="enable SE failure in tornado base case")
    parser.add_argument("--rocket-fail", action="store_true", help="enable rocket failure in tornado base case")
    parser.add_argument("--out-csv", type=str, default="", help="optional summary CSV path")
    parser.add_argument("--strict", action="store_true",
                        help="return nonzero exit code if any solved row fails key feasibility flags")
    parser.add_argument("--no-summary-print", action="store_true",
                        help="suppress printing of the summary DataFrame head")
    args = parser.parse_args()

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    base = preset_mode(Params(failure_timing_mode=args.failure_timing), args.run_mode)

    rows: List[Dict[str, Any]] = []
    try:
        for label, cfg in build_cases(base):
            print(f"--- Solving {label} ({args.run_mode}, {cfg.failure_timing_mode})")
            rows.extend(
                _solve_case_bundle(
                    label=label,
                    cfg=cfg,
                    write_timeseries=not args.no_timeseries,
                    strict_tau=True,
                )
            )
    except Exception as exc:
        print(f"[ERROR] batch solve failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    df = pd.DataFrame(rows)

    out_csv = Path(args.out_csv) if args.out_csv else (outdir / "final_revised_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[OK] wrote summary CSV: {out_csv}")

    if not args.no_summary_print:
        cols = [c for c in [
            "ScenarioLabel", "Policy", "ScenarioMode", "SolverSuccess",
            "ConstraintOK", "BuildOK", "OpsOK", "tau_build (yr)",
            "Cost_total ($B)", "E_total (index)", "TimeseriesCSV"
        ] if c in df.columns]
        if cols:
            print(df[cols].to_string(index=False))
        else:
            print(df.head().to_string(index=False))

    if args.tornado:
        tbase = preset_mode(
            Params(
                failure_timing_mode=args.failure_timing,
                scenario="mixed",
                se_failure_active=bool(args.se_fail),
                rocket_failure_active=bool(args.rocket_fail),
            ),
            args.run_mode,
        )
        try:
            tdf = run_tornado(tbase, delta=float(args.delta))
            if "delta_cost" in tdf.columns:
                tdf["abs_delta_cost"] = tdf["delta_cost"].abs()
                tdf = tdf.sort_values(by="abs_delta_cost", ascending=False, na_position="last")

            tornado_csv = outdir / "final_revised_tornado.csv"
            tdf.to_csv(tornado_csv, index=False)
            print(f"[OK] wrote tornado CSV: {tornado_csv}")
            print(tdf.to_string(index=False))
        except Exception as exc:
            print(f"[ERROR] tornado run failed: {exc}", file=sys.stderr)
            raise SystemExit(4) from exc

    if args.strict:
        ok = _check_strict_pass(df)
        if not ok:
            print("[ERROR] strict mode failed: at least one summary row is not fully feasible/successful.", file=sys.stderr)
            raise SystemExit(3)


if __name__ == "__main__":
    main()
