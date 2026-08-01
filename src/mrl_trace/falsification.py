"""Frozen, uncertainty-banded predictions for future physical tests."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .model_specs import PRIMARY_MODEL_ID, LINEAR_MODEL_ID, device_model_spec


FALSIFICATION_SCHEMA_VERSION = 2


def _simulate_physical_batch(time: np.ndarray, *, tau_r_s,
                             drive_amplitudes, tau_leak_s: float, k: int,
                             duration_s: float, dt_s: float) -> dict:
    """Vectorised Euler evaluation of the frozen physical-headroom equation."""
    time = np.asarray(time, dtype=float)
    tau_r_s, drive_amplitudes = np.broadcast_arrays(
        np.asarray(tau_r_s, dtype=float),
        np.asarray(drive_amplitudes, dtype=float),
    )
    flat_tau = tau_r_s.ravel()
    flat_drive = drive_amplitudes.ravel()
    state = np.zeros((flat_tau.size, int(k)), dtype=float)
    response = np.empty((flat_tau.size, time.size), dtype=float)
    peak_occupancy = np.zeros(flat_tau.size, dtype=float)
    alpha = int(k) / flat_tau
    for time_index, sample_time in enumerate(time):
        drive = flat_drive if sample_time < float(duration_s) else 0.0
        new = state.copy()
        previous_fraction = drive
        for stage in range(int(k)):
            value = state[:, stage]
            new[:, stage] = value + float(dt_s) * (
                alpha * previous_fraction * (1.0 - np.abs(value))
                - value / float(tau_leak_s)
            )
            previous_fraction = value
        state = np.clip(new, -1.0, 1.0)
        response[:, time_index] = state[:, -1]
        peak_occupancy = np.maximum(
            peak_occupancy, np.max(np.abs(state), axis=1)
        )
    peak = np.maximum(np.max(response, axis=1), 1e-30)
    normalised = response / peak[:, None]
    peak_index = np.argmax(response, axis=1)
    return {
        "normalised_response": normalised.reshape(
            tau_r_s.shape + (time.size,)
        ),
        "preferred_lag_s": time[peak_index].reshape(tau_r_s.shape),
        "peak_occupancy_fraction": peak_occupancy.reshape(tau_r_s.shape),
    }


def _simultaneous_band(curves: np.ndarray, *, level: float = 0.95) -> dict:
    """Pointwise percentile band plus a simultaneous max-deviation envelope."""
    curves = np.asarray(curves, dtype=float)
    centre = np.median(curves, axis=0)
    point_low, point_high = np.percentile(
        curves, (100 * (1 - level) / 2, 100 * (1 + level) / 2), axis=0
    )
    scale = np.maximum(
        (point_high - point_low) / 2.0,
        np.maximum(np.abs(centre), 1e-8) * 1e-6,
    )
    maximum = np.max(np.abs(curves - centre) / scale, axis=1)
    critical = float(np.quantile(maximum, level))
    return {
        "median": centre,
        "pointwise_low": point_low,
        "pointwise_high": point_high,
        "simultaneous_low": centre - critical * scale,
        "simultaneous_high": centre + critical * scale,
        "simultaneous_level": float(level),
        "studentised_max_deviation_critical": critical,
    }


def _parameter_draws(parameter_bootstrap: dict | None, *, samples: int,
                     seed: int) -> tuple[np.ndarray, np.ndarray]:
    if parameter_bootstrap and "parameter_draws" in parameter_bootstrap:
        draws = parameter_bootstrap["parameter_draws"]
        tr0 = np.asarray(draws["tau_r0_s"], dtype=float)
        cr = np.asarray(draws["rise_field_coefficient_per_v"], dtype=float)
        if tr0.size != cr.size or tr0.size == 0:
            raise ValueError("invalid KWW bootstrap parameter draws")
        rng = np.random.default_rng(int(seed))
        indices = rng.choice(tr0.size, size=int(samples), replace=tr0.size < samples)
        return tr0[indices], cr[indices]
    # Frozen-source fallback: a degenerate band is explicitly recorded.  It allows
    # smoke reproduction without raw Au data but is not publication uncertainty.
    return np.full(int(samples), 145.0), np.full(int(samples), 2.9)


def build_falsification_predictions(*, parameter_bootstrap: dict | None = None,
                                    samples: int = 400,
                                    seed: int = 20260804,
                                    voltage_grid=(0.5, 0.7, 0.9, 1.1, 1.3, 1.5),
                                    tau_leak_s: float = 10.0, k: int = 3,
                                    duration_s: float = 0.3,
                                    dt_s: float = 0.02) -> dict:
    """Freeze three predictions and their explicit future falsifiers."""
    tr0, cr = _parameter_draws(parameter_bootstrap, samples=samples, seed=seed)
    horizon = 6.0 * float(tau_leak_s)
    time = np.arange(0.0, horizon + dt_s, dt_s)

    tau_r_09 = tr0 * np.exp(-cr * 0.9)
    pulse_batch = _simulate_physical_batch(
        time, tau_r_s=tau_r_09, drive_amplitudes=np.ones_like(tau_r_09),
        tau_leak_s=tau_leak_s, k=k, duration_s=duration_s, dt_s=dt_s,
    )
    pulse_curves = pulse_batch["normalised_response"]
    voltage_values = np.asarray(list(map(float, voltage_grid)))
    rise_by_voltage = tr0[:, None] * np.exp(-cr[:, None] * voltage_values[None, :])
    voltage_batch = _simulate_physical_batch(
        time, tau_r_s=rise_by_voltage,
        drive_amplitudes=np.ones_like(rise_by_voltage),
        tau_leak_s=tau_leak_s, k=k, duration_s=duration_s, dt_s=dt_s,
    )
    peak_matrix = voltage_batch["preferred_lag_s"]

    drive_amplitudes = np.geomspace(1e-3, 1.0, 13)
    amplitude_grid = np.broadcast_to(
        drive_amplitudes[None, :], (len(tr0), len(drive_amplitudes))
    )
    nonlinear_batch = _simulate_physical_batch(
        time, tau_r_s=np.broadcast_to(tau_r_09[:, None], amplitude_grid.shape),
        drive_amplitudes=amplitude_grid, tau_leak_s=tau_leak_s, k=k,
        duration_s=duration_s, dt_s=dt_s,
    )
    nonlinear_curves = nonlinear_batch["normalised_response"]
    difference = nonlinear_curves - nonlinear_curves[:, :1, :]
    departure_draws = np.sqrt(np.mean(np.square(difference), axis=2))
    maximum_departure_draws = np.max(np.abs(difference), axis=2)
    occupancy_draws = nonlinear_batch["peak_occupancy_fraction"]
    nonlinear_peak_draws = nonlinear_batch["preferred_lag_s"]

    pulse_band = _simultaneous_band(np.asarray(pulse_curves))
    peak_band = _simultaneous_band(peak_matrix)
    peak_rows = []
    for index, voltage in enumerate(voltage_values):
        peak_rows.append({
            "voltage_v": voltage,
            "median_preferred_lag_s": float(peak_band["median"][index]),
            "bootstrap_95ci_s": [
                float(peak_band["pointwise_low"][index]),
                float(peak_band["pointwise_high"][index]),
            ],
            "simultaneous_95band_s": [
                float(peak_band["simultaneous_low"][index]),
                float(peak_band["simultaneous_high"][index]),
            ],
        })
    departure_band = _simultaneous_band(departure_draws)
    departure = []
    for index, amplitude in enumerate(drive_amplitudes):
        max_low, max_high = np.percentile(
            maximum_departure_draws[:, index], (2.5, 97.5)
        )
        occupancy_low, occupancy_high = np.percentile(
            occupancy_draws[:, index], (2.5, 97.5)
        )
        peak_low, peak_high = np.percentile(
            nonlinear_peak_draws[:, index], (2.5, 97.5)
        )
        departure.append({
            "drive_amplitude": float(amplitude),
            "median_peak_occupancy_fraction": float(
                np.median(occupancy_draws[:, index])
            ),
            "peak_occupancy_pointwise_95ci": [
                float(occupancy_low), float(occupancy_high)
            ],
            "median_preferred_lag_s": float(
                np.median(nonlinear_peak_draws[:, index])
            ),
            "preferred_lag_pointwise_95ci_s": [
                float(peak_low), float(peak_high)
            ],
            "median_rms_departure_from_low_occupancy": float(
                departure_band["median"][index]
            ),
            "rms_departure_pointwise_95ci": [
                float(departure_band["pointwise_low"][index]),
                float(departure_band["pointwise_high"][index]),
            ],
            "rms_departure_simultaneous_95band": [
                float(departure_band["simultaneous_low"][index]),
                float(departure_band["simultaneous_high"][index]),
            ],
            "median_maximum_departure_from_low_occupancy": float(
                np.median(maximum_departure_draws[:, index])
            ),
            "maximum_departure_pointwise_95ci": [
                float(max_low), float(max_high)
            ],
        })
    payload = {
        "schema_version": FALSIFICATION_SCHEMA_VERSION,
        "analysis": "frozen_future_physical_falsification_predictions",
        "status": "untested_prediction",
        "protocol": {
            "samples": int(samples), "seed": int(seed),
            "tau_leak_s": float(tau_leak_s), "cascade_depth": int(k),
            "deposit_duration_s": float(duration_s), "dt_s": float(dt_s),
            "voltage_grid_v": list(map(float, voltage_grid)),
            "kinetic_uncertainty_source": (
                "repeat-stratified KWW parameter refits"
                if parameter_bootstrap and "parameter_draws" in parameter_bootstrap
                else "degenerate frozen 145*exp(-2.9V) smoke fallback"
            ),
        },
        "model_specifications": {
            model_id: device_model_spec(model_id)
            for model_id in (PRIMARY_MODEL_ID, LINEAR_MODEL_ID)
        },
        "normalised_pulse_off": {
            "delay_s": time,
            **pulse_band,
        },
        "bias_conditioned_preferred_lag": peak_rows,
        "bias_conditioned_preferred_lag_band": {
            "simultaneous_level": peak_band["simultaneous_level"],
            "studentised_max_deviation_critical": peak_band[
                "studentised_max_deviation_critical"
            ],
        },
        "nonlinear_departure": {
            "reference": "lowest-drive numerical low-occupancy response",
            "simultaneous_level": departure_band["simultaneous_level"],
            "studentised_max_deviation_critical": departure_band[
                "studentised_max_deviation_critical"
            ],
            "rows": departure,
        },
        "numbered_falsifiers": [
            {
                "id": "F1",
                "prediction": "a resolved non-zero pulse-off eligibility peak",
                "falsifier": (
                    "a reproducible zero-lag maximum under the frozen deposit and "
                    "read protocol"
                ),
            },
            {
                "id": "F2",
                "prediction": "preferred lag shifts with bias inside the frozen band",
                "falsifier": (
                    "future confidence intervals miss the simultaneous model band at "
                    "two adjacent predeclared biases"
                ),
            },
            {
                "id": "F3",
                "prediction": (
                    "low occupancy approaches the frozen low-signal response and "
                    "headroom saturation produces the recorded departure"
                ),
                "falsifier": (
                    "normalised low-drive responses fail to converge or the departure "
                    "has the opposite amplitude trend"
                ),
            },
        ],
        "claim_limit": (
            "These are frozen compact-model predictions for a future experiment, "
            "not measurements or evidence of physical programmability."
        ),
    }
    digestable = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
        default=lambda value: np.asarray(value).tolist(),
    ).encode()
    payload["manifest_digest_sha256"] = hashlib.sha256(digestable).hexdigest()
    return payload


__all__ = ["FALSIFICATION_SCHEMA_VERSION", "build_falsification_predictions"]
