"""Analytic and numerical theory for the eligibility kernels.

The linear Erlang representation is a linear time-invariant cascade.  Its impulse
response is Gamma shaped and its finite rectangular-pulse response is the
difference of two Erlang step responses.  The physical-headroom representation is
nonlinear and is *not* globally Gamma; only its low-occupancy, drive-off limit has
a Gamma-shaped downstream propagation envelope.

The helpers here keep those two statements separate and return plain arrays/dicts
so publication notebooks and regression tests consume the same implementation.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc, gammaln

from .device import CascadeEligibilityGate, LinearErlangEligibilityGate, tau_r
from .model_specs import LINEAR_MODEL_ID, PRIMARY_MODEL_ID, device_model_spec


KERNEL_THEORY_SCHEMA_VERSION = 1


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def linear_pole_rate(tau_r_s: float, tau_leak_s: float, k: int) -> float:
    """Return the repeated pole ``lambda = k/tau_r + 1/tau_leak``."""
    tau_r_s = _positive(tau_r_s, "tau_r_s")
    tau_leak_s = _positive(tau_leak_s, "tau_leak_s")
    if int(k) < 1:
        raise ValueError("k must be a positive integer")
    return int(k) / tau_r_s + 1.0 / tau_leak_s


def linear_impulse_peak(tau_r_s: float, tau_leak_s: float, k: int) -> float:
    """Peak lag of the continuous-time linear cascade's impulse response."""
    k = int(k)
    if k < 1:
        raise ValueError("k must be a positive integer")
    return 0.0 if k == 1 else (k - 1.0) / linear_pole_rate(
        tau_r_s, tau_leak_s, k
    )


def linear_impulse_response(time_s, tau_r_s: float, tau_leak_s: float,
                            k: int, *, normalise: bool = True) -> np.ndarray:
    """Gamma-shaped impulse response of ``linear_erlang_v1``.

    The unnormalised response includes the transfer gain ``alpha**k``.  At zero
    time it is zero for ``k > 1`` and finite for ``k == 1``.
    """
    time_s = np.asarray(time_s, dtype=float)
    if np.any(~np.isfinite(time_s)) or np.any(time_s < 0):
        raise ValueError("time_s must be finite and non-negative")
    k = int(k)
    if k < 1:
        raise ValueError("k must be a positive integer")
    tau_r_s = _positive(tau_r_s, "tau_r_s")
    rate = linear_pole_rate(tau_r_s, tau_leak_s, k)
    alpha = k / tau_r_s
    if k == 1:
        values = alpha * np.exp(-rate * time_s)
    else:
        values = np.zeros_like(time_s)
        positive = time_s > 0
        log_values = (
            k * math.log(alpha)
            + (k - 1) * np.log(time_s[positive])
            - rate * time_s[positive]
            - gammaln(k)
        )
        values[positive] = np.exp(log_values)
    if normalise:
        peak = float(np.max(values, initial=0.0))
        if peak > 0:
            values = values / peak
    return values


def linear_rectangular_response(time_s, tau_r_s: float, tau_leak_s: float,
                                k: int, duration_s: float,
                                *, normalise: bool = True) -> np.ndarray:
    """Exact response of the linear cascade to a unit rectangular deposit."""
    time_s = np.asarray(time_s, dtype=float)
    if np.any(~np.isfinite(time_s)) or np.any(time_s < 0):
        raise ValueError("time_s must be finite and non-negative")
    duration_s = _positive(duration_s, "duration_s")
    k = int(k)
    if k < 1:
        raise ValueError("k must be a positive integer")
    tau_r_s = _positive(tau_r_s, "tau_r_s")
    rate = linear_pole_rate(tau_r_s, tau_leak_s, k)
    alpha = k / tau_r_s
    gain = (alpha / rate) ** k
    values = gain * gammainc(k, rate * time_s)
    shifted = np.clip(time_s - duration_s, 0.0, None)
    values -= gain * gammainc(k, rate * shifted) * (time_s >= duration_s)
    if normalise:
        peak = float(np.max(values, initial=0.0))
        if peak > 0:
            values = values / peak
    return values


def linear_rectangular_peak(tau_r_s: float, tau_leak_s: float, k: int,
                            duration_s: float) -> float:
    """Peak of the finite-pulse response from a deterministic derivative root.

    For ``k > 1`` the derivative after pulse offset is the difference between two
    Gamma impulse responses.  Bracketing uses the corresponding closed-form root,
    while :func:`scipy.optimize.brentq` verifies the root numerically.
    """
    duration_s = _positive(duration_s, "duration_s")
    k = int(k)
    if k < 1:
        raise ValueError("k must be a positive integer")
    if k == 1:
        return duration_s
    rate = linear_pole_rate(tau_r_s, tau_leak_s, k)
    exact = duration_s / (1.0 - math.exp(-rate * duration_s / (k - 1.0)))

    def signed_log_difference(time: float) -> float:
        # The common positive constants cancel.  Scaling by the larger log term
        # avoids underflow for long traces.
        first = (k - 1.0) * math.log(time) - rate * time
        shifted = time - duration_s
        second = (k - 1.0) * math.log(shifted) - rate * shifted
        maximum = max(first, second)
        return math.exp(first - maximum) - math.exp(second - maximum)

    lower = duration_s * (1.0 + 1e-12)
    upper = max(exact * 1.5, duration_s + 10.0 / rate)
    return float(brentq(signed_log_difference, lower, upper, xtol=1e-13,
                        rtol=1e-13))


def physical_small_signal_peak(tau_leak_s: float, k: int) -> float:
    """Drive-off low-occupancy peak of the nonlinear headroom cascade.

    Once an impulse-like deposit has populated stage one, linearising the
    downstream headroom products around zero gives a Gamma envelope with leakage
    pole ``1/tau_leak``.  This is not the global driven response.
    """
    tau_leak_s = _positive(tau_leak_s, "tau_leak_s")
    k = int(k)
    if k < 1:
        raise ValueError("k must be a positive integer")
    return max(0.0, (k - 1.0) * tau_leak_s)


def simulate_gate_response(time_s, *, model_id: str, voltage_v: float,
                           tau_leak_s: float, k: int, duration_s: float,
                           drive_amplitude: float = 1.0, v_max: float = 1.0,
                           dt_s: float | None = None,
                           tau_r_override: float | None = None) -> dict:
    """Numerically simulate one finite deposit and retain occupancy diagnostics."""
    time_s = np.asarray(time_s, dtype=float)
    if time_s.ndim != 1 or time_s.size < 2 or np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must be a strictly increasing vector")
    if time_s[0] < 0 or np.any(~np.isfinite(time_s)):
        raise ValueError("time_s must be finite and non-negative")
    duration_s = _positive(duration_s, "duration_s")
    v_max = _positive(v_max, "v_max")
    if not math.isfinite(float(drive_amplitude)):
        raise ValueError("drive_amplitude must be finite")
    if model_id == PRIMARY_MODEL_ID:
        gate_class = CascadeEligibilityGate
    elif model_id == LINEAR_MODEL_ID:
        gate_class = LinearErlangEligibilityGate
    else:
        raise ValueError(f"unknown model_id {model_id!r}")
    dt = float(dt_s if dt_s is not None else np.min(np.diff(time_s)))
    gate = gate_class(V=voltage_v, tau_leak=tau_leak_s, k=int(k), dt=dt,
                      vnmax=v_max, beta_leak=1.0,
                      tau_r_override=tau_r_override)
    values = np.empty(time_s.size, dtype=float)
    occupancy = np.empty(time_s.size, dtype=float)
    next_time = 0.0
    current = 0.0
    for index, sample_time in enumerate(time_s):
        while next_time <= sample_time + 0.5 * dt:
            drive = float(drive_amplitude) if next_time < duration_s else 0.0
            current = float(gate.step(drive))
            next_time += dt
        values[index] = current
        occupancy[index] = float(np.max(np.abs(gate.vn)) / v_max)
    peak_index = int(np.argmax(values))
    return {
        "time_s": time_s,
        "response": values,
        "normalised_response": values / max(float(np.max(values)), 1e-30),
        "preferred_lag_s": float(time_s[peak_index]),
        "peak_occupancy_fraction": float(np.max(occupancy)),
        "occupancy_fraction": occupancy,
    }


def kernel_phase_map(*, voltages: Iterable[float] = (0.5, 0.7, 0.9, 1.2, 1.5),
                     tau_leaks_s: Iterable[float] = (1.0, 2.0, 5.0, 10.0, 20.0),
                     depths: Iterable[int] = (2, 3, 4),
                     durations_s: Iterable[float] = (0.05, 0.3),
                     v_max_values: Iterable[float] = (0.5, 1.0, 2.0),
                     drive_amplitude: float = 1.0, dt_s: float = 0.01) -> list[dict]:
    """Return a deterministic physical/linear crossover and headroom phase map."""
    rows: list[dict] = []
    v_max_values = tuple(map(float, v_max_values))
    if not v_max_values:
        raise ValueError("v_max_values must not be empty")
    reference_v_max = v_max_values[0]
    for voltage in map(float, voltages):
        rise = float(tau_r(abs(voltage)))
        for tau_leak_s in map(float, tau_leaks_s):
            for k in map(int, depths):
                rate = linear_pole_rate(rise, tau_leak_s, k)
                rho = (k / rise) * tau_leak_s
                for duration_s in map(float, durations_s):
                    horizon = max(6.0 * tau_leak_s, duration_s + 8.0 / rate,
                                  duration_s + 1.5 * physical_small_signal_peak(
                                      tau_leak_s, k), 2.0)
                    grid = np.arange(0.0, horizon + dt_s, dt_s)
                    linear_peak = linear_rectangular_peak(
                        rise, tau_leak_s, k, duration_s
                    )
                    # Both implementations use dimensionless upstream occupancy and
                    # return v_k/V_max, so normalized trajectories are analytically
                    # invariant to V_max.  Simulate once; parity at arbitrary scales
                    # is independently regression-tested.
                    physical = simulate_gate_response(
                        grid, model_id=PRIMARY_MODEL_ID, voltage_v=voltage,
                        tau_leak_s=tau_leak_s, k=k, duration_s=duration_s,
                        drive_amplitude=drive_amplitude, v_max=reference_v_max,
                        dt_s=dt_s,
                    )
                    linear = simulate_gate_response(
                        grid, model_id=LINEAR_MODEL_ID, voltage_v=voltage,
                        tau_leak_s=tau_leak_s, k=k, duration_s=duration_s,
                        drive_amplitude=drive_amplitude, v_max=reference_v_max,
                        dt_s=dt_s,
                    )
                    for v_max in v_max_values:
                        rows.append({
                            "voltage_v": voltage, "tau_r_s": rise,
                            "tau_leak_s": tau_leak_s, "cascade_depth": k,
                            "duration_s": duration_s, "v_max": v_max,
                            "drive_amplitude": float(drive_amplitude),
                            "rho_alpha_tau_leak": rho,
                            "linear_impulse_peak_s": linear_impulse_peak(
                                rise, tau_leak_s, k),
                            "linear_finite_pulse_peak_s": linear_peak,
                            "linear_numerical_peak_s": linear["preferred_lag_s"],
                            "linear_peak_relative_error": abs(
                                linear["preferred_lag_s"] - linear_peak
                            ) / max(linear_peak, dt_s),
                            "physical_small_signal_peak_s": physical_small_signal_peak(
                                tau_leak_s, k),
                            "physical_numerical_peak_s": physical["preferred_lag_s"],
                            "physical_peak_relative_departure": abs(
                                physical["preferred_lag_s"]
                                - physical_small_signal_peak(tau_leak_s, k)
                            ) / max(physical_small_signal_peak(tau_leak_s, k), dt_s),
                            "physical_peak_occupancy_fraction": physical[
                                "peak_occupancy_fraction"
                            ],
                            "v_max_shape_invariance": True,
                        })
    return rows


def kernel_theory_manifest(rows: list[dict], *, protocol: dict | None = None) -> dict:
    """Build a stable non-pickle manifest for theory and phase-map results."""
    payload = {
        "schema_version": KERNEL_THEORY_SCHEMA_VERSION,
        "analysis": "eligibility_kernel_small_signal_and_crossover",
        "protocol": dict(protocol or {}),
        "equations": {
            "linear_pole_rate": "lambda=k/tau_r+1/tau_leak",
            "linear_impulse": "e(t) proportional to t^(k-1) exp(-lambda*t)",
            "linear_impulse_peak": "(k-1)/lambda",
            "linear_finite_pulse": "step(t)-step(t-duration)",
            "physical_small_signal_limit": (
                "drive-off low-occupancy downstream Gamma envelope with pole "
                "1/tau_leak; not a global physical model"
            ),
        },
        "model_specifications": {
            model_id: device_model_spec(model_id)
            for model_id in (PRIMARY_MODEL_ID, LINEAR_MODEL_ID)
        },
        "claim_limits": [
            "Bias dependence is a frozen model prediction, not demonstrated programmability.",
            "The nonlinear physical-headroom model is not globally a Gamma kernel.",
            "Integer cascade depth is a representation parameter, not a microscopic count.",
        ],
        "rows": rows,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_digest_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


__all__ = [
    "KERNEL_THEORY_SCHEMA_VERSION", "linear_pole_rate",
    "linear_impulse_peak", "linear_impulse_response",
    "linear_rectangular_response", "linear_rectangular_peak",
    "physical_small_signal_peak", "simulate_gate_response", "kernel_phase_map",
    "kernel_theory_manifest",
]
