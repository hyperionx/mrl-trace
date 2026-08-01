"""Amplitude-controlled timing tasks for frozen eligibility kernels.

The task labels are defined on an absolute delay grid, not from a device-model
peak.  Kernel scaling is label free; only a scalar readout temperature/intercept is
fit on pilot trials and then frozen for disjoint evaluation trials.
"""
from __future__ import annotations

import hashlib
import json
import math

import numpy as np
from scipy.optimize import minimize
from scipy.integrate import trapezoid
from scipy.special import expit

from .device import decay_matched_exponential_tau, tau_r
from .kernel_theory import linear_rectangular_response, simulate_gate_response
from .model_specs import LINEAR_MODEL_ID, PRIMARY_MODEL_ID, device_model_spec


TIMING_BENCHMARK_SCHEMA_VERSION = 1
ABSTRACT_GAMMA_ID = "abstract_gamma_k3"
MATCHED_EXPONENTIAL_ID = "post_peak_decay_matched_exponential"
MATCHING_REGIMES = ("same_drive_raw", "unit_peak", "unit_l1", "pilot_rms")


def _logsumexp(values, axis=-1):
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.sum(np.exp(values - maximum), axis=axis)
    )


def _sigmoid(values):
    return expit(np.asarray(values, dtype=float))


def frozen_kernel_bank(time_s, *, voltage_v: float = 0.9,
                       tau_leak_s: float = 10.0, k: int = 3,
                       duration_s: float = 0.3, dt_s: float = 0.01) -> dict:
    """Evaluate the four predeclared same-drive kernel shapes on one grid."""
    time_s = np.asarray(time_s, dtype=float)
    if time_s.ndim != 1 or time_s.size < 2 or np.any(np.diff(time_s) <= 0):
        raise ValueError("time_s must be a strictly increasing vector")
    physical = simulate_gate_response(
        time_s, model_id=PRIMARY_MODEL_ID, voltage_v=voltage_v,
        tau_leak_s=tau_leak_s, k=k, duration_s=duration_s, dt_s=dt_s,
    )["response"]
    linear = simulate_gate_response(
        time_s, model_id=LINEAR_MODEL_ID, voltage_v=voltage_v,
        tau_leak_s=tau_leak_s, k=k, duration_s=duration_s, dt_s=dt_s,
    )["response"]
    gamma = linear_rectangular_response(
        time_s, tau_r(abs(float(voltage_v))), tau_leak_s, k, duration_s,
        normalise=False,
    )
    exp_tau = decay_matched_exponential_tau(
        tau_leak_s, V=voltage_v, k=k, beta_leak=1.0,
        coincidence_dur=duration_s, gate_model=PRIMARY_MODEL_ID,
    )
    exponential = np.exp(-time_s / exp_tau)
    return {
        "time_s": time_s,
        "values": {
            PRIMARY_MODEL_ID: np.asarray(physical),
            LINEAR_MODEL_ID: np.asarray(linear),
            ABSTRACT_GAMMA_ID: np.asarray(gamma),
            MATCHED_EXPONENTIAL_ID: np.asarray(exponential),
        },
        "configuration": {
            "voltage_v": float(voltage_v), "tau_leak_s": float(tau_leak_s),
            "cascade_depth": int(k), "duration_s": float(duration_s),
            "dt_s": float(dt_s), "matched_exponential_tau_s": float(exp_tau),
        },
    }


def matching_scales(bank: dict, *, pilot_delays_s=None) -> dict:
    """Return positive label-free scale factors for all four matching regimes."""
    time = np.asarray(bank["time_s"], dtype=float)
    pilot = np.asarray(
        np.linspace(time[0], time[-1], 257)
        if pilot_delays_s is None else pilot_delays_s,
        dtype=float,
    )
    scales = {}
    for name, raw in bank["values"].items():
        raw = np.asarray(raw, dtype=float)
        sampled = np.interp(pilot, time, raw)
        centred = sampled - float(np.mean(sampled))
        peak = max(float(np.max(np.abs(raw))), 1e-30)
        area = max(float(trapezoid(np.abs(raw), time)), 1e-30)
        rms = max(float(np.sqrt(np.mean(np.square(centred)))), 1e-30)
        scales[name] = {
            "same_drive_raw": 1.0,
            "unit_peak": 1.0 / peak,
            "unit_l1": 1.0 / area,
            "pilot_rms": 1.0 / rms,
            "raw_peak": peak, "raw_l1": area, "pilot_centred_rms": rms,
        }
    return {
        "pilot_delays_s": pilot,
        "label_free": True,
        "horizon_s": [float(time[0]), float(time[-1])],
        "by_kernel": scales,
    }


def build_interval_attribution_trials(*, split: str, seed: int,
                                      target_delays_s=(2.5, 7.5, 12.5, 17.5,
                                                       22.5, 27.5, 35.0, 45.0),
                                      horizon_s: float = 60.0,
                                      trials_per_block: int = 64,
                                      candidates: int = 4) -> dict:
    """Four-event attribution whose target centres are model independent."""
    if split not in {"pilot", "evaluation"}:
        raise ValueError("split must be pilot or evaluation")
    rng = np.random.default_rng(int(seed))
    lags, targets, blocks, centres = [], [], [], []
    for block, centre in enumerate(map(float, target_delays_s)):
        for _ in range(int(trials_per_block)):
            causal = float(np.clip(
                rng.normal(centre, max(0.15, 0.06 * centre)), 0.05, horizon_s
            ))
            distractors = []
            while len(distractors) < int(candidates) - 1:
                value = float(rng.uniform(0.05, horizon_s))
                if abs(value - causal) >= max(0.5, 0.12 * centre):
                    distractors.append(value)
            row = np.asarray([causal, *distractors])
            order = rng.permutation(int(candidates))
            lags.append(row[order])
            targets.append(int(np.flatnonzero(order == 0)[0]))
            blocks.append(block)
            centres.append(centre)
    return {
        "task": "four_event_interval_attribution", "split": split,
        "seed": int(seed), "lags_s": np.asarray(lags),
        "target": np.asarray(targets, dtype=int),
        "block_id": np.asarray(blocks, dtype=int),
        "target_delay_s": np.asarray(centres),
        "target_delay_grid_s": list(map(float, target_delays_s)),
        "label_definition": "event generated around predeclared absolute target delay",
    }


def build_temporal_order_trials(*, split: str, seed: int,
                                centres_s=(10.0, 20.0, 30.0, 40.0),
                                separations_s=(2.0, 6.0, 10.0),
                                trials_per_block: int = 64) -> dict:
    """Balanced two-cue order trials with matched mean event-to-reward duration."""
    if split not in {"pilot", "evaluation"}:
        raise ValueError("split must be pilot or evaluation")
    rng = np.random.default_rng(int(seed))
    lags, labels, blocks, centres, separations = [], [], [], [], []
    block = 0
    for centre in map(float, centres_s):
        for separation in map(float, separations_s):
            if separation >= 2.0 * centre:
                raise ValueError("separation must keep both lags positive")
            base = np.asarray([centre + separation / 2.0,
                               centre - separation / 2.0])
            for trial in range(int(trials_per_block)):
                earlier_is_a = bool((trial + int(rng.integers(0, 2))) % 2)
                noise = rng.normal(0.0, min(0.1, separation / 20.0), size=2)
                row = base + noise
                if earlier_is_a:
                    lags.append(row)
                    labels.append(1)
                else:
                    lags.append(row[::-1])
                    labels.append(0)
                blocks.append(block)
                centres.append(centre)
                separations.append(separation)
            block += 1
    return {
        "task": "two_cue_temporal_order", "split": split, "seed": int(seed),
        "lags_s": np.asarray(lags), "target": np.asarray(labels, dtype=int),
        "block_id": np.asarray(blocks, dtype=int),
        "centre_s": np.asarray(centres), "separation_s": np.asarray(separations),
        "label_definition": "one iff cue A occurred earlier than cue B",
    }


def _sample_kernel(bank: dict, lags, kernel: str, scale: float) -> np.ndarray:
    return float(scale) * np.interp(
        np.asarray(lags, dtype=float), bank["time_s"], bank["values"][kernel]
    )


def _fit_interval_readout(values, target) -> dict:
    values = np.asarray(values, dtype=float)
    target = np.asarray(target, dtype=int)
    numerical_scale = max(float(np.std(values)), 1e-30)
    scaled = values / numerical_scale

    def objective(parameters):
        logits = float(parameters[0]) * scaled
        return float(np.mean(
            _logsumexp(logits, axis=1)
            - logits[np.arange(target.size), target]
        ))

    result = minimize(objective, np.asarray([1.0]), method="L-BFGS-B",
                      bounds=[(-1e6, 1e6)],
                      options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000})
    finite = (
        np.isfinite(float(result.fun))
        and np.all(np.isfinite(np.asarray(result.x, dtype=float)))
    )
    # Recent SciPy L-BFGS-B releases can return ABNORMAL after a line-search
    # round-off stall even though the bounded convex fit is finite.  Preserve
    # that candidate; only a non-finite result makes the readout unusable.
    if not finite:
        raise RuntimeError(f"interval readout fit failed: {result.message}")
    return {"temperature": float(result.x[0] / numerical_scale), "intercept": 0.0,
            "pilot_log_loss": float(result.fun)}


def _fit_order_readout(values, target) -> dict:
    difference = np.asarray(values[:, 0] - values[:, 1], dtype=float)
    target = np.asarray(target, dtype=float)
    numerical_scale = max(float(np.std(difference)), 1e-30)
    scaled = difference / numerical_scale

    def objective(parameters):
        logits = parameters[0] * scaled + parameters[1]
        return float(np.mean(
            np.logaddexp(0.0, logits) - target * logits
        ))

    result = minimize(objective, np.asarray([1.0, 0.0]), method="L-BFGS-B",
                      bounds=[(-1e6, 1e6), (-20.0, 20.0)],
                      options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 1000})
    finite = (
        np.isfinite(float(result.fun))
        and np.all(np.isfinite(np.asarray(result.x, dtype=float)))
    )
    if not finite:
        raise RuntimeError(f"order readout fit failed: {result.message}")
    return {"temperature": float(result.x[0] / numerical_scale),
            "intercept": float(result.x[1]),
            "pilot_log_loss": float(result.fun)}


def _evaluate(task: dict, values: np.ndarray, readout: dict) -> list[dict]:
    target = np.asarray(task["target"], dtype=int)
    if task["task"] == "four_event_interval_attribution":
        logits = readout["temperature"] * values
        log_probability = logits - _logsumexp(logits, axis=1)[:, None]
        losses = -log_probability[np.arange(target.size), target]
        correct = np.argmax(logits, axis=1) == target
    else:
        logits = (
            readout["temperature"] * (values[:, 0] - values[:, 1])
            + readout["intercept"]
        )
        probability = _sigmoid(logits)
        losses = -(
            target * np.log(np.clip(probability, 1e-15, 1.0))
            + (1 - target) * np.log(np.clip(1.0 - probability, 1e-15, 1.0))
        )
        correct = (probability >= 0.5) == target
    rows = []
    for block in np.unique(task["block_id"]):
        mask = task["block_id"] == block
        row = {
            "task": task["task"], "block_id": int(block),
            "log_loss": float(np.mean(losses[mask])),
            "top1_accuracy": float(np.mean(correct[mask])),
            "trials": int(np.count_nonzero(mask)),
        }
        if task["task"] == "four_event_interval_attribution":
            row["target_delay_s"] = float(np.unique(task["target_delay_s"][mask])[0])
        else:
            row["centre_s"] = float(np.unique(task["centre_s"][mask])[0])
            row["separation_s"] = float(np.unique(task["separation_s"][mask])[0])
        rows.append(row)
    return rows


def _paired_bootstrap(rows: list[dict], *, physical=PRIMARY_MODEL_ID,
                      samples: int = 10_000, seed: int = 20260803) -> dict:
    def compare(selected_rows, comparison_seed):
        keys = sorted({(row["matching"], row["task"], row["block_id"])
                       for row in selected_rows if row["kernel"] == physical})
        lookup = {
            (row["kernel"], row["matching"], row["task"], row["block_id"]):
            row["log_loss"] for row in selected_rows
        }
        comparators = sorted({row["kernel"] for row in selected_rows
                              if row["kernel"] != physical})
        rng = np.random.default_rng(int(comparison_seed))
        result = {}
        for comparator in comparators:
            delta = np.asarray([
                lookup[(physical, *key)] - lookup[(comparator, *key)] for key in keys
            ])
            draw = rng.integers(0, delta.size, size=(int(samples), delta.size))
            means = delta[draw].mean(axis=1)
            low, high = np.percentile(means, (2.5, 97.5))
            result[comparator] = {
                "mean_physical_minus_comparator_log_loss": float(np.mean(delta)),
                "ci95": [float(low), float(high)],
                "negative_is_physical_better": True,
                "verdict": "physical_advantage" if high < 0 else (
                    "comparator_advantage" if low > 0 else "unresolved"
                ),
            }
        return result

    overall = compare(rows, seed)
    by_matching = {}
    by_task = {}
    offset = 1
    for matching in MATCHING_REGIMES:
        selected = [row for row in rows if row["matching"] == matching]
        by_matching[matching] = compare(selected, seed + offset)
        offset += 1
        by_task[matching] = {}
        for task in sorted({row["task"] for row in selected}):
            task_rows = [row for row in selected if row["task"] == task]
            by_task[matching][task] = compare(task_rows, seed + offset)
            offset += 1
    return {
        "overall": overall,
        "by_matching": by_matching,
        "by_matching_and_task": by_task,
        "samples": int(samples),
        "paired_unit": "evaluation scenario block",
    }


def run_matched_timing_benchmark(*, voltage_v: float = 0.9,
                                 tau_leak_s: float = 10.0, k: int = 3,
                                 duration_s: float = 0.3,
                                 pilot_seed: int = 3200,
                                 evaluation_seed: int = 9200,
                                 trials_per_block: int = 64,
                                 bootstrap_samples: int = 10_000) -> dict:
    """Run the compact matching matrix on disjoint timing-task scenarios."""
    horizon = 6.0 * float(tau_leak_s)
    dt = 0.01
    grid = np.arange(0.0, horizon + dt, dt)
    bank = frozen_kernel_bank(
        grid, voltage_v=voltage_v, tau_leak_s=tau_leak_s, k=k,
        duration_s=duration_s, dt_s=dt,
    )
    scales = matching_scales(bank, pilot_delays_s=np.linspace(0.0, horizon, 257))
    pilot_tasks = (
        build_interval_attribution_trials(
            split="pilot", seed=pilot_seed, horizon_s=horizon,
            trials_per_block=trials_per_block,
        ),
        build_temporal_order_trials(
            split="pilot", seed=pilot_seed + 1,
            trials_per_block=trials_per_block,
        ),
    )
    evaluation_tasks = (
        build_interval_attribution_trials(
            split="evaluation", seed=evaluation_seed, horizon_s=horizon,
            trials_per_block=trials_per_block,
        ),
        build_temporal_order_trials(
            split="evaluation", seed=evaluation_seed + 1,
            trials_per_block=trials_per_block,
        ),
    )
    rows, readouts = [], {}
    for matching in MATCHING_REGIMES:
        for kernel in bank["values"]:
            scale = scales["by_kernel"][kernel][matching]
            for pilot, evaluation in zip(pilot_tasks, evaluation_tasks):
                pilot_values = _sample_kernel(bank, pilot["lags_s"], kernel, scale)
                readout = (
                    _fit_interval_readout(pilot_values, pilot["target"])
                    if pilot["task"] == "four_event_interval_attribution"
                    else _fit_order_readout(pilot_values, pilot["target"])
                )
                key = f"{matching}/{kernel}/{pilot['task']}"
                readouts[key] = readout
                values = _sample_kernel(bank, evaluation["lags_s"], kernel, scale)
                for row in _evaluate(evaluation, values, readout):
                    rows.append({"matching": matching, "kernel": kernel, **row})
    paired = _paired_bootstrap(
        rows, samples=bootstrap_samples, seed=20260803
    )
    summary = {}
    for matching in MATCHING_REGIMES:
        summary[matching] = {}
        for kernel in bank["values"]:
            selected = [row for row in rows
                        if row["matching"] == matching and row["kernel"] == kernel]
            summary[matching][kernel] = {
                "log_loss": float(np.average(
                    [row["log_loss"] for row in selected],
                    weights=[row["trials"] for row in selected],
                )),
                "top1_accuracy": float(np.average(
                    [row["top1_accuracy"] for row in selected],
                    weights=[row["trials"] for row in selected],
                )),
            }
    payload = {
        "schema_version": TIMING_BENCHMARK_SCHEMA_VERSION,
        "analysis": "matched_kernel_model_independent_timing_battery",
        "protocol": {
            **bank["configuration"], "horizon_s": horizon,
            "matching_regimes": list(MATCHING_REGIMES),
            "pilot_seed": int(pilot_seed), "evaluation_seed": int(evaluation_seed),
            "trials_per_block": int(trials_per_block),
            "readout": "pilot-fitted scalar temperature/intercept only",
            "evaluation_used_for_fitting": False,
        },
        "model_specifications": {
            model_id: device_model_spec(model_id)
            for model_id in (PRIMARY_MODEL_ID, LINEAR_MODEL_ID)
        },
        "kernel_definitions": {
            ABSTRACT_GAMMA_ID: "analytic finite-pulse small-signal linear cascade",
            MATCHED_EXPONENTIAL_ID: "physical post-peak decay-matched exponential",
        },
        "scales": scales, "readouts": readouts,
        "per_block_metrics": rows, "summary": summary,
        "paired_bootstrap": paired,
        "scalar_aliasing_limit": (
            "Equal scalar values on the rising and falling limbs cannot encode "
            "temporal order; resolving them requires additional accessible state."
        ),
        "claim_limit": (
            "Synthetic fixed-feature timing analysis, not measured hardware learning "
            "or general reinforcement-learning superiority."
        ),
    }
    digest_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                default=lambda value: value.tolist()).encode()
    payload["manifest_digest_sha256"] = hashlib.sha256(digest_payload).hexdigest()
    return payload


__all__ = [
    "TIMING_BENCHMARK_SCHEMA_VERSION", "ABSTRACT_GAMMA_ID",
    "MATCHED_EXPONENTIAL_ID", "MATCHING_REGIMES", "frozen_kernel_bank",
    "matching_scales", "build_interval_attribution_trials",
    "build_temporal_order_trials", "run_matched_timing_benchmark",
]
