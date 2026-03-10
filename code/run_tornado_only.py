# -*- coding: utf-8 -*-
"""
run_tornado_only.py
-------------------
Campaign-scale one-at-a-time (OAT) tornado sensitivity runner.

Purpose
-------
This script generates local OAT sensitivity outputs around a campaign-scale base case.
It is intended to support:
- the campaign-scale sensitivity section in the paper,
- supplementary reproducibility tables,
- machine-readable output for downstream plotting / reporting.

Model layer
-----------
campaign_scale

IMPORTANT
---------
This script uses:
    Params + preset_mode + run_tornado

It is NOT the reduced paper-level / mature-regime sensitivity runner.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from revised_solver_core import OUTDIR, Params, preset_mode, run_tornado


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


def _build_base_case(args: argparse.Namespace) -> Params:
    p = preset_mode(Params(failure_timing_mode=args.failure_timing), args.run_mode)
    p = replace(
        p,
        scenario=args.scenario,
        se_failure_active=bool(args.se_fail),
        rocket_failure_active=bool(args.rocket_fail),
    )
    return p


def _maybe_sort_tornado(df: pd.DataFrame, sort_by: str) -> pd.DataFrame:
    """
    Sort tornado output by a preferred delta column if present.
    Falls back gracefully if the requested column is missing.
    """
    if df.empty or sort_by == "none":
        return df

    candidate_col: Optional[str] = None

    if sort_by in df.columns:
        candidate_col = sort_by
    else:
        # Accept aliases without the leading 'delta_' if needed
        aliases = {
            "cost": "delta_cost",
            "time": "delta_tau",
            "tau": "delta_tau",
            "env": "delta_env",
            "JE": "delta_env",
        }
        if sort_by in aliases and aliases[sort_by] in df.columns:
            candidate_col = aliases[sort_by]

    if candidate_col is None:
        return df

    abs_col = f"abs_{candidate_col}"
    out = df.copy()
    out[abs_col] = out[candidate_col].abs()
    out = out.sort_values(by=abs_col, ascending=False, na_position="last")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Campaign-scale OAT tornado sensitivity runner"
    )
    parser.add_argument("--run-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--failure-timing", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--scenario", choices=["mixed", "se_only", "rocket_only"], default="mixed")
    parser.add_argument("--delta", type=float, default=0.10,
                        help="relative perturbation amplitude used by run_tornado")
    parser.add_argument("--se-fail", action="store_true",
                        help="enable SE failure in the tornado base case")
    parser.add_argument("--rocket-fail", action="store_true",
                        help="enable rocket failure in the tornado base case")

    parser.add_argument("--sort-by", type=str, default="delta_cost",
                        help="preferred sorting column (e.g. delta_cost, delta_tau, delta_env, none)")
    parser.add_argument("--out-csv", type=str, default="",
                        help="optional output CSV path; defaults to OUTDIR/tornado_report.csv")
    parser.add_argument("--summary-json", type=str, default="",
                        help="optional metadata JSON path; defaults to OUTDIR/tornado_meta.json")
    parser.add_argument("--print-table", action="store_true",
                        help="print the full tornado table to stdout")
    parser.add_argument("--strict", action="store_true",
                        help="return nonzero exit code if the tornado result is empty")

    args = parser.parse_args()

    outdir = Path(OUTDIR)
    outdir.mkdir(parents=True, exist_ok=True)

    base = _build_base_case(args)

    tdf = run_tornado(base, delta=float(args.delta))
    tdf = _maybe_sort_tornado(tdf, args.sort_by)

    out_csv = Path(args.out_csv) if args.out_csv else (outdir / "tornado_report.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tdf.to_csv(out_csv, index=False)

    meta = {
        "model_layer": "campaign_scale",
        "entrypoint": "run_tornado_only.py",
        "notes": [
            "This script runs campaign-scale one-at-a-time local sensitivity.",
            "It should not be confused with reduced-model robustness or reduced policy-map sweeps.",
            "The reported perturbation size is controlled by --delta.",
        ],
        "run_mode": args.run_mode,
        "failure_timing": args.failure_timing,
        "scenario": args.scenario,
        "se_fail": bool(args.se_fail),
        "rocket_fail": bool(args.rocket_fail),
        "delta": float(args.delta),
        "sort_by": args.sort_by,
        "n_rows": int(len(tdf)),
        "columns": list(map(str, tdf.columns)),
        "base_params": asdict(base),
        "out_csv": str(out_csv),
    }

    out_json = Path(args.summary_json) if args.summary_json else (outdir / "tornado_meta.json")
    _write_json(out_json, meta)

    print(f"[OK] wrote tornado CSV: {out_csv}")
    print(f"[OK] wrote tornado metadata JSON: {out_json}")

    if args.print_table:
        if tdf.empty:
            print("[WARN] Tornado DataFrame is empty.")
        else:
            print(tdf.to_string(index=False))

    if args.strict and tdf.empty:
        raise SystemExit(3)


if __name__ == "__main__":
    main()