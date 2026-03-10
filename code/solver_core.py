# -*- coding: utf-8 -*-
"""
solver_core.py
--------------
Shared low-level utilities for the Scientific Reports code package.

Purpose
-------
This module provides:
1) global scaling conventions,
2) smooth step / softplus utilities for CasADi expressions,
3) numerically matched NumPy counterparts for post-processing,
4) shared transport-capacity and disruption-health functions.

Design note
-----------
The NumPy and CasADi implementations are designed to remain numerically
stable while preserving the intended asymptotic behavior of the smoothing
functions. In particular, the logistic step uses clipping for stability,
whereas the softplus implementation uses a stable exact reformulation
rather than hard clipping, so that large positive inputs retain the
expected linear tail.

Important implementation note
-----------------------------
For campaign-scale optimization with relative failure timing, some disruption
durations may be symbolic CasADi expressions (for example, a fixed fraction
of the build horizon). Therefore, symbolic-mode health windows must NOT
coerce all inputs through float()-based validation. Numeric validation is
performed only in explicitly numeric branches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np
import casadi as ca


# ---------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------
NumericLike = Union[float, int, np.ndarray]
SymbolicLike = Union[ca.SX, ca.MX, ca.DM]
AnyLike = Union[NumericLike, SymbolicLike]


# ---------------------------------------------------------------------
# Global scaling
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Scales:
    """
    Default global scaling:
    - mass: tons -> Mt
    - launch: launches -> k-launches
    - money: dollars -> billion dollars
    """
    mass: float = 1e6
    launch: float = 1e3
    money: float = 1e9


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------
def _validate_positive(name: str, value: float) -> float:
    """Require a strictly positive scalar parameter."""
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _validate_nonnegative(name: str, value: float) -> float:
    """Require a nonnegative scalar parameter."""
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value}.")
    return value


def _validate_unit_interval(name: str, value: float) -> float:
    """Require a scalar parameter in [0, 1]."""
    value = float(value)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in [0, 1], got {value}.")
    return value


def _clamp_alpha(alpha: float) -> float:
    """
    Clamp alpha away from 0 and 1 to avoid singular logistics.

    Logical requirement: 0 < alpha < 1.
    Numerical implementation: clip to [1e-6, 1-1e-6].
    """
    alpha = float(alpha)
    if alpha <= 0.0 or alpha >= 1.0:
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {alpha}.")
    return float(np.clip(alpha, 1e-6, 1.0 - 1e-6))


def _as_numeric_array(z: NumericLike) -> np.ndarray:
    """Convert numeric input to a float ndarray."""
    return np.asarray(z, dtype=float)


def _maybe_scalar(x: np.ndarray | np.floating | float) -> float | np.ndarray:
    """
    Return Python float for scalar-shaped numeric outputs,
    otherwise return ndarray.
    """
    arr = np.asarray(x)
    if arr.ndim == 0:
        return float(arr)
    return arr


def _is_symbolic(x: AnyLike) -> bool:
    """Return True if x is a CasADi symbolic/numeric object."""
    return isinstance(x, (ca.SX, ca.MX, ca.DM))


# ---------------------------------------------------------------------
# Smooth nonlinearities (CasADi)
# ---------------------------------------------------------------------
def sigma_k(z: AnyLike, k: float) -> AnyLike:
    """
    Smoothed step / logistic function for symbolic expressions.

    Uses clipping of k*z to [-50, 50] for numerical stability.
    """
    k = _validate_positive("k", k)
    kz = ca.fmin(ca.fmax(k * z, -50.0), 50.0)
    return 1.0 / (1.0 + ca.exp(-kz))


def softplus_k(z: AnyLike, k: float) -> AnyLike:
    """
    Stable softplus for symbolic expressions.

    This implementation avoids hard clipping of k*z, so the function
    preserves the correct linear asymptote for large positive inputs:
        softplus(z) ~ z  as z -> +infinity
    """
    k = _validate_positive("k", k)
    kz = k * z
    return (ca.fmax(kz, 0.0) + ca.log(1.0 + ca.exp(-ca.fabs(kz)))) / k


# ---------------------------------------------------------------------
# Smooth nonlinearities (NumPy)
# ---------------------------------------------------------------------
def sigma_num(z: NumericLike, k: float) -> float | np.ndarray:
    """
    NumPy counterpart of sigma_k with identical clipping.

    Returns a Python float for scalar input and ndarray for array input.
    """
    k = _validate_positive("k", k)
    kz = np.clip(k * _as_numeric_array(z), -50.0, 50.0)
    out = 1.0 / (1.0 + np.exp(-kz))
    return _maybe_scalar(out)


def softplus_num(z: NumericLike, k: float) -> float | np.ndarray:
    """
    Stable NumPy counterpart of softplus_k.

    This implementation preserves the correct linear asymptote for
    large positive inputs and avoids overflow without hard clipping.
    """
    k = _validate_positive("k", k)
    kz = k * _as_numeric_array(z)
    out = (np.maximum(kz, 0.0) + np.log1p(np.exp(-np.abs(kz)))) / k
    return _maybe_scalar(out)


# ---------------------------------------------------------------------
# Shared health / availability windows
# ---------------------------------------------------------------------
def window_health(
    t: AnyLike,
    t0: AnyLike,
    dur: AnyLike,
    severity: float,
    k_step: float = 10.0,
    is_numeric: bool = False,
) -> AnyLike:
    """
    Smooth multiplicative health factor over a finite time window.

    Parameters
    ----------
    t : numeric or symbolic time
    t0 : start time (may be symbolic in advanced constructions)
    dur : duration; may be symbolic when relative-to-build timing is used
    severity : reduction fraction in [0,1]
    k_step : smoothing steepness
    is_numeric : choose NumPy or CasADi backend

    Returns
    -------
    health factor in [1-severity, 1] (up to smoothing effects)

    Notes
    -----
    In numeric mode, dur is validated as a nonnegative scalar.
    In symbolic mode, dur may be a CasADi expression and must not be
    coerced through float().
    """
    k_step = _validate_positive("k_step", k_step)
    severity = _validate_unit_interval("severity", severity)

    if is_numeric:
        dur_num = _validate_nonnegative("dur", dur)
        w = sigma_num(t - t0, k_step) - sigma_num(t - (t0 + dur_num), k_step)
        return 1.0 - severity * w

    # Symbolic branch: permit symbolic t0/dur (e.g. relative failure windows).
    # If dur happens to be a plain scalar, still check nonnegativity.
    if not _is_symbolic(dur):
        _validate_nonnegative("dur", dur)

    w = sigma_k(t - t0, k_step) - sigma_k(t - (t0 + dur), k_step)
    return 1.0 - severity * w


# ---------------------------------------------------------------------
# Shared transport-capacity model
# ---------------------------------------------------------------------
def capacity_logistic(
    t: AnyLike,
    K_max: AnyLike,
    alpha: float,
    r: float,
    delta_eff: float,
    failure_active: bool = False,
    fail_t_start: AnyLike = 0.0,
    fail_duration: AnyLike = 0.0,
    fail_severity: float = 0.0,
    k_step: float = 10.0,
    is_numeric: bool = False,
) -> AnyLike:
    """
    Logistic commissioning curve with optional multiplicative disruption.

    Parameters
    ----------
    t : time
    K_max : nominal mature throughput
    alpha : initial commissioning fraction, must satisfy 0 < alpha < 1
    r : logistic growth rate, must be nonnegative
    delta_eff : mean effective availability multiplier in [0,1]
    failure_active : whether to apply a disruption window
    fail_t_start, fail_duration, fail_severity : disruption parameters
        In symbolic mode, fail_t_start and fail_duration may be CasADi
        expressions when relative timing is used.
    k_step : smoothing steepness for the window
    is_numeric : choose NumPy or CasADi backend

    Returns
    -------
    Effective throughput at time t.
    """
    alpha = _clamp_alpha(alpha)
    r = _validate_nonnegative("r", r)
    delta_eff = _validate_unit_interval("delta_eff", delta_eff)

    exp_fn = np.exp if is_numeric else ca.exp
    base = K_max / (1.0 + ((1.0 - alpha) / alpha) * exp_fn(-r * t))
    base = base * delta_eff

    if failure_active:
        return base * window_health(
            t=t,
            t0=fail_t_start,
            dur=fail_duration,
            severity=fail_severity,
            k_step=k_step,
            is_numeric=is_numeric,
        )
    return base


def rocket_health(
    t: AnyLike,
    failure_active: bool,
    fail_t_start: AnyLike,
    fail_duration: AnyLike,
    fail_severity: float,
    k_step: float = 10.0,
    is_numeric: bool = False,
) -> AnyLike:
    """
    Multiplicative rocket-side health factor.

    Returns 1 when no failure is active, otherwise returns the smoothed
    window-based health factor.

    In symbolic mode, fail_t_start and fail_duration may be CasADi
    expressions when relative timing is used.
    """
    if not failure_active:
        return 1.0

    return window_health(
        t=t,
        t0=fail_t_start,
        dur=fail_duration,
        severity=fail_severity,
        k_step=k_step,
        is_numeric=is_numeric,
    )


__all__ = [
    "Scales",
    "sigma_k",
    "softplus_k",
    "sigma_num",
    "softplus_num",
    "window_health",
    "capacity_logistic",
    "rocket_health",
]