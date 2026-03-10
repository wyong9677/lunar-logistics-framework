# -*- coding: utf-8 -*-
"""
paper_pipeline.py
-------------------------
Pipeline runner for the paper's computational artifacts.

Primary artifacts (default):
- campaign-scale scenario batch regeneration (main paper results)
- reduced-model Monte Carlo robustness regeneration (mechanism diagnostic)
- reduced mature-regime 2D phase-out / policy-map regeneration (mechanism diagnostic)

Optional extensions:
- campaign-scale OAT tornado regeneration
- campaign-scale numerical consistency audit regeneration

Design goals:
1) Safe subprocess execution (no shell=True)
2) Clear step-by-step logging
3) Reproducibility-friendly environment handling
4) Explicit distinction between campaign-scale headline artifacts and reduced-model diagnostics
5) Single configurable output directory with script-specific output filenames
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTDIR = ROOT / "paper_outputs"
DEFAULT_MPLCONFIGDIR = ROOT / ".mplconfig"


def _make_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))
    DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
    return env


def _script_path(name: str) -> Path:
    return ROOT / name


def _require_script(name: str) -> Path:
    path = _script_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Required script not found: {path}")
    return path


def _optional_script(name: str) -> Optional[Path]:
    path = _script_path(name)
    return path if path.exists() else None


def _run_mode_from_args(args: argparse.Namespace) -> str:
    return "fast" if args.fast else "full"


def _print_header(args: argparse.Namespace, outdir: Path, env: Dict[str, str]) -> None:
    print("Paper pipeline starting...")
    print(f"Root directory: {ROOT}")
    print(f"Output directory: {outdir}")
    print(f"MPLCONFIGDIR: {env['MPLCONFIGDIR']}")
    print(f"Mode: {_run_mode_from_args(args)}")
    print("Primary result layer: campaign-scale scenario batch")
    print("Diagnostic layers: reduced-model robustness + reduced mature-regime policy map")



def run_step(step_name: str, cmd: List[str], env: Dict[str, str]) -> None:
    print(f"\n=== [{step_name}] ===")
    print(">>>", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True, env=env, cwd=str(ROOT))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(
            f"\n[ERROR] Step '{step_name}' failed with exit code {exc.returncode}.\n"
            f"Command: {' '.join(cmd)}"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline runner for paper-level computational artifacts"
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="quick validation mode with smaller/sparser runs where supported",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(DEFAULT_OUTDIR),
        help="directory for pipeline outputs and script-level artifacts",
    )

    # Primary paper artifacts
    parser.add_argument(
        "--skip-scenarios",
        action="store_true",
        help="skip campaign-scale scenario batch regeneration",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="skip reduced-model Monte Carlo robustness regeneration",
    )
    parser.add_argument(
        "--skip-phaseout-map",
        action="store_true",
        help="skip reduced mature-regime 2D phase-out/policy-map regeneration",
    )

    # Optional additional artifacts
    parser.add_argument(
        "--include-tornado",
        action="store_true",
        help="also regenerate campaign-scale OAT tornado sensitivity",
    )
    parser.add_argument(
        "--include-consistency",
        action="store_true",
        help="also regenerate campaign-scale numerical consistency audit",
    )

    # Shared campaign-scale settings
    parser.add_argument(
        "--failure-timing",
        choices=["absolute", "relative"],
        default="relative",
        help="failure timing mode for campaign-scale scripts that expose this option",
    )
    parser.add_argument(
        "--scenario-only-case",
        type=str,
        default="",
        help="optional exact case label forwarded to run_scenarios.py",
    )
    parser.add_argument(
        "--no-timeseries",
        action="store_true",
        help="disable campaign-scale timeseries writing in run_scenarios.py",
    )

    # Robustness settings (reduced model)
    parser.add_argument("--n-sims", type=int, default=25, help="Monte Carlo sample count")
    parser.add_argument("--seed", type=int, default=42, help="base random seed")
    parser.add_argument("--jitter", type=float, default=0.05, help="relative perturbation amplitude")
    parser.add_argument("--goal-relax", type=float, default=0.99, help="goal relaxation factor")
    parser.add_argument("--gamma", type=float, default=0.08, help="requested gamma for reduced-model robustness")
    parser.add_argument(
        "--gamma-candidates",
        type=str,
        default="0.08,0,1e-8,1e-6,1e-4,1e-3,5e-3,1e-2,5e-2,0.10,0.20,0.50,1.0",
        help="comma-separated gamma fallback candidates forwarded to robustness.py",
    )
    parser.add_argument(
        "--failure-active",
        action="store_true",
        help="enable reduced-model failure-active mode in robustness.py",
    )

    # Reduced phase-out map settings
    parser.add_argument("--rel-tol", type=float, default=0.05, help="relative JE matching tolerance")
    parser.add_argument("--max-bisect", type=int, default=24, help="maximum safeguarded refinement steps")
    parser.add_argument("--gamma-grid-size", type=int, default=17, help="gamma grid size")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="disable resume behavior in phaseout_map_eps_keff.py",
    )
    parser.add_argument(
        "--keff-factors",
        type=str,
        default="",
        help="comma-separated list for phaseout_map_eps_keff.py, e.g. 0.5,1.0,1.5",
    )
    parser.add_argument(
        "--eps-multipliers",
        type=str,
        default="",
        help="comma-separated list for phaseout_map_eps_keff.py, e.g. 0.1,0.5,1.0",
    )

    # Optional tornado settings
    parser.add_argument(
        "--tornado-scenario",
        choices=["mixed", "se_only", "rocket_only"],
        default="mixed",
        help="campaign-scale base case for OAT tornado, used if --include-tornado",
    )
    parser.add_argument("--tornado-delta", type=float, default=0.10, help="relative perturbation amplitude for OAT tornado")
    parser.add_argument(
        "--tornado-sort-by",
        type=str,
        default="delta_cost",
        help="preferred sorting column for the tornado report",
    )
    parser.add_argument("--tornado-se-fail", action="store_true", help="enable SE failure in tornado base case")
    parser.add_argument("--tornado-rocket-fail", action="store_true", help="enable rocket failure in tornado base case")

    # Optional consistency settings
    parser.add_argument("--consistency-scenario", choices=["mixed", "se_only", "rocket_only"], default="mixed")
    parser.add_argument("--consistency-policy", choices=["tau_star", "time_opt", "cost_opt"], default="cost_opt")
    parser.add_argument("--consistency-restarts", type=int, default=8, help="number of restart runs")
    parser.add_argument("--consistency-init-noise", type=float, default=0.10, help="initial-guess perturbation scale")
    parser.add_argument("--consistency-tau0-jitter", type=float, default=0.10, help="relative Gaussian jitter of tau0")
    parser.add_argument("--consistency-seed", type=int, default=42, help="random seed for consistency audit")
    parser.add_argument("--consistency-se-fail", action="store_true", help="enable SE failure in consistency base case")
    parser.add_argument("--consistency-rocket-fail", action="store_true", help="enable rocket failure in consistency base case")

    args = parser.parse_args()

    py = sys.executable
    env = _make_env()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    run_mode = _run_mode_from_args(args)

    _print_header(args, outdir, env)

    # 1) Campaign-scale scenario batch (primary paper results)
    if not args.skip_scenarios:
        scenarios_script = _require_script("run_scenarios.py")
        cmd = [
            py,
            str(scenarios_script),
            "--run-mode", run_mode,
            "--failure-timing", args.failure_timing,
            "--out-csv", str(outdir / "scenario_batch_summary.csv"),
            "--meta-json", str(outdir / "scenario_batch_meta.json"),
        ]
        if args.scenario_only_case:
            cmd.extend(["--only-case", args.scenario_only_case])
        if args.no_timeseries:
            cmd.append("--no-timeseries")
        run_step("Campaign-scale scenario batch", cmd, env)

    # 2) Reduced-model Monte Carlo robustness (mechanism diagnostic)
    if not args.skip_robustness:
        robustness_script = _require_script("robustness.py")
        n_sims = 10 if args.fast else int(args.n_sims)
        cmd = [
            py,
            str(robustness_script),
            "--n-sims", str(n_sims),
            "--seed", str(int(args.seed)),
            "--jitter", str(float(args.jitter)),
            "--goal-relax", str(float(args.goal_relax)),
            "--gamma", str(float(args.gamma)),
            "--gamma-candidates", args.gamma_candidates,
            "--outdir", str(outdir),
        ]
        if args.failure_active:
            cmd.append("--failure-active")
        run_step("Reduced-model Monte Carlo robustness", cmd, env)

    # 3) Reduced mature-regime 2D phase-out / policy map (mechanism diagnostic)
    if not args.skip_phaseout_map:
        phaseout_script = _require_script("phaseout_map_eps_keff.py")
        cmd = [
            py,
            str(phaseout_script),
            "--rel-tol", str(float(args.rel_tol)),
            "--max-bisect", str(int(args.max_bisect)),
            "--gamma-grid-size", str(int(args.gamma_grid_size)),
            "--outdir", str(outdir),
        ]
        if args.fast:
            cmd.append("--fast")
        if args.no_resume:
            cmd.append("--no-resume")
        if args.keff_factors:
            cmd.extend(["--keff-factors", args.keff_factors])
        if args.eps_multipliers:
            cmd.extend(["--eps-multipliers", args.eps_multipliers])
        run_step("Reduced mature-regime phase-out map", cmd, env)

    # 4) Optional: campaign-scale OAT tornado
    if args.include_tornado:
        path = _optional_script("run_tornado_only.py")
        if path is None:
            print("\n[WARN] run_tornado_only.py not found; skipping tornado regeneration.")
        else:
            cmd = [
                py,
                str(path),
                "--run-mode", run_mode,
                "--failure-timing", args.failure_timing,
                "--scenario", args.tornado_scenario,
                "--delta", str(float(args.tornado_delta)),
                "--sort-by", args.tornado_sort_by,
                "--out-csv", str(outdir / "tornado_report.csv"),
                "--summary-json", str(outdir / "tornado_meta.json"),
            ]
            if args.tornado_se_fail:
                cmd.append("--se-fail")
            if args.tornado_rocket_fail:
                cmd.append("--rocket-fail")
            run_step("Campaign-scale OAT tornado sensitivity", cmd, env)

    # 5) Optional: campaign-scale numerical consistency audit
    if args.include_consistency:
        path = _optional_script("run_consistency_check.py")
        if path is None:
            print("\n[WARN] run_consistency_check.py not found; skipping consistency audit.")
        else:
            cmd = [
                py,
                str(path),
                "--run-mode", run_mode,
                "--failure-timing", args.failure_timing,
                "--scenario", args.consistency_scenario,
                "--policy", args.consistency_policy,
                "--n-runs", str(int(args.consistency_restarts)),
                "--seed", str(int(args.consistency_seed)),
                "--init-noise", str(float(args.consistency_init_noise)),
                "--tau0-jitter", str(float(args.consistency_tau0_jitter)),
                "--outdir", str(outdir),
                "--label-prefix", "consistency",
            ]
            if args.consistency_se_fail:
                cmd.append("--se-fail")
            if args.consistency_rocket_fail:
                cmd.append("--rocket-fail")
            run_step("Campaign-scale numerical consistency audit", cmd, env)

    print("\n[OK] paper pipeline complete.")
    print(f"[OK] Primary outputs are in: {outdir}")
    print("[OK] Interpretation guide:")
    print("     - scenario_batch_*: campaign-scale main results")
    print("     - tornado_* / consistency*: campaign-scale support artifacts")
    print("     - robustness* / phaseout_map*: reduced-model diagnostics")


if __name__ == "__main__":
    main()
