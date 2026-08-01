"""Resource-matched temporal-attribution and hardware-cost analysis.

This module deliberately separates three questions that were previously mixed in
the manuscript:

1. what temporal prior a frozen eligibility kernel expresses;
2. how that prior compares with an equally state-limited learned temporal basis; and
3. what state and arithmetic a digital implementation must provide if the material
   relaxation is not used directly.

The attribution benchmark is not a closed-loop learning task and is not presented as
one.  It asks which of four perfectly observed candidate events receives the largest
credit at reward time.  A positive result therefore establishes a timing inductive
bias, not general reinforcement-learning superiority.  Pilot and evaluation scenario
blocks are generated from disjoint seeds, and evaluation data never enter gain or
kernel fitting.

The hardware table is structural accounting, not an energy or area measurement.  It
reports stored state, coefficients and recurrence operations per synapse.  Absolute
energy can only be obtained after supplying a circuit, technology and peripheral-cost
model; the helper :func:`digital_cost_equivalent` keeps those assumptions explicit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy.optimize import minimize

from .device import (
    CascadeEligibilityGate,
    LinearErlangEligibilityGate,
    decay_matched_exponential_tau,
    tau_r,
)
from .model_specs import (
    LINEAR_MODEL_ID,
    PRIMARY_MODEL_ID,
    device_model_spec,
)

TEMPORAL_BENCHMARK_SCHEMA_VERSION = 1
LEARNED_BASIS_ID = "learned_signed_exponential_k3"
MATCHED_EXPONENTIAL_ID = "physical_decay_matched_exponential"
DEFAULT_RETENTION_DEFINITIONS = (
    ("held_bias_q10", 0.8448946663770562, "direct held-bias distribution"),
    ("held_bias_q25", 1.1669372374932345, "direct held-bias distribution"),
    ("held_bias_q50", 1.3896742156393977, "direct held-bias distribution"),
    ("held_bias_q75", 1.6179662155669912, "direct held-bias distribution"),
    ("near_zero_device_1", 4.312684718514085, "supplementary 1-mV read"),
    ("near_zero_device_2", 9.1966164265996, "supplementary 1-mV read"),
    ("synthetic_10s", 10.0, "declared synthetic operating point"),
)

TEMPORAL_CODESIGN_PROVENANCE = {
    "status": "resource_matched_model_analysis",
    "established_basis": [
        "eligibility-kernel temporal attribution",
        "state-space resource accounting",
        "held-out hyperparameter selection",
    ],
    "repository_adaptation": (
        "Frozen physical and linear kernels are compared with a pilot-fitted signed "
        "three-exponential basis on disjoint held-out timing blocks."
    ),
    "claim_limit": (
        "The benchmark measures temporal attribution, not closed-loop reward "
        "learning. Resource counts are topology-derived lower bounds and exclude "
        "uncharacterised analogue peripherals, routing, retention refresh and write "
        "energy."
    ),
}


def _canonical_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _gate_class(model_id: str):
    if model_id == PRIMARY_MODEL_ID:
        return CascadeEligibilityGate
    if model_id == LINEAR_MODEL_ID:
        return LinearErlangEligibilityGate
    raise ValueError(f"unknown model_id {model_id!r}")


def eligibility_kernel(
    lags,
    *,
    model_id: str = PRIMARY_MODEL_ID,
    voltage_v: float = 0.9,
    tau_leak_s: float = 10.0,
    cascade_depth: int = 3,
    beta_leak: float = 1.0,
    coincidence_duration_s: float = 0.3,
    dt_s: float = 0.01,
):
    """Evaluate one frozen device kernel at event-to-reward ``lags``.

    The response is simulated once on an absolute time grid and normalised by its
    peak.  ``lags`` may have any shape; the returned array has the same shape.
    """
    lags = np.asarray(lags, dtype=float)
    if np.any(~np.isfinite(lags)) or np.any(lags < 0):
        raise ValueError("lags must be finite and non-negative")
    if dt_s <= 0 or coincidence_duration_s <= 0:
        raise ValueError("time increments must be positive")
    fitted_rise = float(tau_r(abs(float(voltage_v))))
    horizon = max(
        float(np.max(lags, initial=0.0)) + 2.0 * coincidence_duration_s,
        10.0 * tau_leak_s,
        10.0 * fitted_rise,
    )
    grid = np.arange(0.0, horizon + dt_s, dt_s)
    gate = _gate_class(model_id)(
        V=voltage_v,
        tau_leak=tau_leak_s,
        k=cascade_depth,
        beta_leak=beta_leak,
        dt=dt_s,
    )
    response = gate.trace(
        grid,
        coincidence_at=0.0,
        coincidence_dur=coincidence_duration_s,
        normalise=True,
    )
    return np.interp(lags, grid, response, left=response[0], right=response[-1])


def preferred_lag(**kernel_kwargs) -> float:
    """Return the non-zero preferred lag of a frozen device kernel."""
    voltage = float(kernel_kwargs.get("voltage_v", 0.9))
    tau_leak_s = float(kernel_kwargs.get("tau_leak_s", 10.0))
    fitted_rise = float(tau_r(abs(voltage)))
    horizon = max(12.0 * fitted_rise, 10.0 * tau_leak_s, 2.0)
    dt_s = float(kernel_kwargs.get("dt_s", 0.01))
    grid = np.arange(0.0, horizon + dt_s, dt_s)
    values = eligibility_kernel(grid, **kernel_kwargs)
    return float(grid[int(np.argmax(values))])


def _scenario_centres(preferred: float, split: str) -> dict[str, tuple[float, ...]]:
    if split == "pilot":
        return {
            "physical_aligned": (0.80 * preferred, preferred, 1.20 * preferred),
            "early_delay": (4.0, 6.0, 8.0),
            "recency": (0.30, 0.45, 0.60),
            "broad_delay": (3.0, 0.55 * preferred, 1.35 * preferred),
        }
    if split == "evaluation":
        return {
            "physical_aligned": (0.70 * preferred, 0.90 * preferred,
                                 1.10 * preferred, 1.30 * preferred),
            "early_delay": (3.0, 5.0, 7.0, 9.0),
            "recency": (0.25, 0.40, 0.55, 0.70),
            "broad_delay": (2.0, 0.45 * preferred, 0.75 * preferred,
                            1.45 * preferred),
        }
    raise ValueError("split must be 'pilot' or 'evaluation'")


def build_temporal_attribution_trials(
    *,
    split: str,
    seed: int,
    preferred_delay_s: float,
    trials_per_block: int = 48,
    candidates_per_trial: int = 4,
) -> dict:
    """Build deterministic, blocked timing scenarios for pilot or evaluation.

    Four regimes are included so no single temporal prior is labelled universally
    correct: a physical-kernel-aligned delay, an early-delay regime, a recency regime
    and a broad-delay regime.  Candidate order is shuffled within every trial.
    """
    if trials_per_block < 2 or candidates_per_trial < 2:
        raise ValueError("each block and trial must contain at least two observations")
    rng = np.random.default_rng(int(seed))
    centres = _scenario_centres(float(preferred_delay_s), split)
    lag_rows = []
    causal_indices = []
    regimes = []
    block_ids = []
    block = 0
    horizon = max(40.0, 1.8 * preferred_delay_s)
    for regime, values in centres.items():
        for centre in values:
            for _ in range(int(trials_per_block)):
                causal = max(0.05, rng.normal(centre, max(0.03, 0.06 * centre)))
                if regime in {"physical_aligned", "early_delay"}:
                    distractors = [rng.uniform(0.15, 0.80)]
                    while len(distractors) < candidates_per_trial - 1:
                        value = rng.uniform(0.15, horizon)
                        if abs(value - causal) > max(0.35, 0.10 * causal):
                            distractors.append(value)
                elif regime == "recency":
                    distractors = [rng.normal(preferred_delay_s, 0.08 * preferred_delay_s)]
                    distractors.extend(
                        rng.uniform(2.0, horizon, candidates_per_trial - 2).tolist()
                    )
                else:
                    distractors = []
                    while len(distractors) < candidates_per_trial - 1:
                        value = rng.uniform(0.15, horizon)
                        if abs(value - causal) > max(0.35, 0.08 * causal):
                            distractors.append(value)
                row = np.asarray([causal, *distractors], dtype=float)
                order = rng.permutation(candidates_per_trial)
                lag_rows.append(row[order])
                causal_indices.append(int(np.flatnonzero(order == 0)[0]))
                regimes.append(regime)
                block_ids.append(block)
            block += 1
    return {
        "lags_s": np.asarray(lag_rows),
        "causal_index": np.asarray(causal_indices, dtype=int),
        "regime": np.asarray(regimes),
        "block_id": np.asarray(block_ids, dtype=int),
        "split": split,
        "seed": int(seed),
        "preferred_delay_s": float(preferred_delay_s),
    }


def _log_softmax(values: np.ndarray) -> np.ndarray:
    maximum = np.max(values, axis=1, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))


def _loss_from_values(values, causal_index, gain: float) -> float:
    log_probability = _log_softmax(float(gain) * np.asarray(values, dtype=float))
    return float(-np.mean(log_probability[np.arange(len(causal_index)), causal_index]))


def _rms_calibration_gain(values) -> float:
    """Return a label-free scale with unit RMS within-trial score spread."""
    values = np.asarray(values, dtype=float)
    centred = values - np.mean(values, axis=1, keepdims=True)
    rms = float(np.sqrt(np.mean(np.square(centred))))
    return 1.0 / max(rms, 1e-12)


def _signed_exponential_values(lags, parameters, k: int = 3):
    parameters = np.asarray(parameters, dtype=float)
    tau = np.exp(parameters[:k])
    weights = np.tanh(parameters[k:2 * k])
    norm = max(float(np.linalg.norm(weights)), 1e-12)
    weights = weights / norm
    basis = np.exp(-np.asarray(lags, dtype=float)[..., None] / tau)
    return np.einsum("...k,k->...", basis, weights)


def fit_learned_temporal_basis(trials: dict, *, k: int = 3,
                               multistarts: int = 12, seed: int = 4100) -> dict:
    """Fit a signed ``k``-exponential basis using pilot blocks only."""
    lags = np.asarray(trials["lags_s"], dtype=float)
    target = np.asarray(trials["causal_index"], dtype=int)
    rng = np.random.default_rng(int(seed))
    lower = np.r_[np.full(k, math.log(0.15)), np.full(k, -5.0)]
    upper = np.r_[np.full(k, math.log(120.0)), np.full(k, 5.0)]

    def objective(theta):
        values = _signed_exponential_values(lags, theta, k=k)
        gain = _rms_calibration_gain(values)
        return _loss_from_values(values, target, gain) + 1e-5 * float(
            np.sum(np.square(theta[k:2 * k]))
        )

    starts = []
    base_tau = np.geomspace(0.5, 40.0, k)
    starts.append(np.r_[np.log(base_tau), np.linspace(-1.0, 1.0, k)])
    for _ in range(max(0, int(multistarts) - 1)):
        starts.append(rng.uniform(lower, upper))
    results = [
        minimize(objective, start, method="L-BFGS-B", bounds=list(zip(lower, upper)))
        for start in starts
    ]
    successful = [result for result in results if result.success]
    if not successful:
        raise RuntimeError("all learned-basis optimization starts failed")
    best = min(successful, key=lambda result: (float(result.fun), tuple(result.x)))
    tau = np.exp(best.x[:k])
    weights = np.tanh(best.x[k:2 * k])
    weights /= max(float(np.linalg.norm(weights)), 1e-12)
    return {
        "state_count": int(k),
        "tau_s": tau,
        "readout_weights": weights,
        "gain": float(_rms_calibration_gain(
            _signed_exponential_values(lags, best.x, k=k)
        )),
        "pilot_objective": float(best.fun),
        "parameters": best.x,
        "multistarts": int(multistarts),
        "fit_seed": int(seed),
    }


def _method_values(trials: dict) -> dict[str, np.ndarray]:
    lags = trials["lags_s"]
    physical = eligibility_kernel(
        lags, model_id=PRIMARY_MODEL_ID, voltage_v=0.9, tau_leak_s=10.0,
        cascade_depth=3, beta_leak=1.0,
    )
    linear = eligibility_kernel(
        lags, model_id=LINEAR_MODEL_ID, voltage_v=0.9, tau_leak_s=10.0,
        cascade_depth=3, beta_leak=1.0,
    )
    matched_tau = decay_matched_exponential_tau(
        10.0, V=0.9, k=3, beta_leak=1.0, gate_model=PRIMARY_MODEL_ID
    )
    result = {
        PRIMARY_MODEL_ID: physical,
        LINEAR_MODEL_ID: linear,
        MATCHED_EXPONENTIAL_ID: np.exp(-lags / matched_tau),
    }
    return result


def _regime_trials(trials: dict, regime: str) -> dict:
    mask = np.asarray(trials["regime"]) == regime
    if not np.any(mask):
        raise ValueError(f"unknown or empty regime {regime!r}")
    return {
        key: (value[mask] if isinstance(value, np.ndarray)
              and value.shape[:1] == mask.shape else value)
        for key, value in trials.items()
    }


def _block_metrics(values, trials, gains: dict[str, float]) -> list[dict]:
    values = np.asarray(values, dtype=float)
    target = trials["causal_index"]
    trial_gains = np.asarray([gains[str(regime)] for regime in trials["regime"]])
    logs = _log_softmax(trial_gains[:, None] * values)
    losses = -logs[np.arange(target.size), target]
    correct = np.argmax(values, axis=1) == target
    rows = []
    for block in np.unique(trials["block_id"]):
        mask = trials["block_id"] == block
        regimes = np.unique(trials["regime"][mask])
        if regimes.size != 1:
            raise RuntimeError("a scenario block must have one regime")
        rows.append({
            "block_id": int(block),
            "regime": str(regimes[0]),
            "log_loss": float(np.mean(losses[mask])),
            "top1_accuracy": float(np.mean(correct[mask])),
            "trials": int(np.count_nonzero(mask)),
        })
    return rows


def _paired_block_bootstrap(a, b, *, seed: int, resamples: int = 10_000) -> dict:
    difference = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, difference.size, size=(int(resamples), difference.size))
    samples = difference[index].mean(axis=1)
    low, high = np.percentile(samples, (2.5, 97.5))
    return {
        "mean_physical_minus_comparator": float(np.mean(difference)),
        "ci95": [float(low), float(high)],
        "negative_is_better_for_physical": True,
        "resampling_unit": "held_out_scenario_block",
        "resamples": int(resamples),
    }


def run_temporal_attribution_benchmark(
    *,
    pilot_seed: int = 3100,
    evaluation_seed: int = 9100,
    trials_per_block: int = 48,
    learned_multistarts: int = 12,
    bootstrap_resamples: int = 10_000,
) -> dict:
    """Run the disjoint pilot/evaluation, resource-matched timing benchmark."""
    if int(pilot_seed) == int(evaluation_seed):
        raise ValueError("pilot and evaluation seeds must be disjoint")
    physical_peak = preferred_lag(
        model_id=PRIMARY_MODEL_ID, voltage_v=0.9, tau_leak_s=10.0,
        cascade_depth=3, beta_leak=1.0,
    )
    pilot = build_temporal_attribution_trials(
        split="pilot", seed=pilot_seed, preferred_delay_s=physical_peak,
        trials_per_block=trials_per_block,
    )
    evaluation = build_temporal_attribution_trials(
        split="evaluation", seed=evaluation_seed, preferred_delay_s=physical_peak,
        trials_per_block=trials_per_block,
    )
    regimes = sorted({str(value) for value in pilot["regime"]})
    learned_by_regime = {
        regime: fit_learned_temporal_basis(
            _regime_trials(pilot, regime), k=3,
            multistarts=learned_multistarts,
            seed=pilot_seed + 17 + index,
        )
        for index, regime in enumerate(regimes)
    }
    pilot_values = _method_values(pilot)
    evaluation_values = _method_values(evaluation)
    learned_pilot = np.empty_like(pilot["lags_s"], dtype=float)
    learned_evaluation = np.empty_like(evaluation["lags_s"], dtype=float)
    for regime, learned in learned_by_regime.items():
        pilot_mask = pilot["regime"] == regime
        evaluation_mask = evaluation["regime"] == regime
        learned_pilot[pilot_mask] = _signed_exponential_values(
            pilot["lags_s"][pilot_mask], learned["parameters"],
            k=int(learned["state_count"]),
        )
        learned_evaluation[evaluation_mask] = _signed_exponential_values(
            evaluation["lags_s"][evaluation_mask], learned["parameters"],
            k=int(learned["state_count"]),
        )
    pilot_values[LEARNED_BASIS_ID] = learned_pilot
    evaluation_values[LEARNED_BASIS_ID] = learned_evaluation
    gains: dict[str, dict[str, float]] = {}
    rows = []
    for method, values in pilot_values.items():
        gains[method] = {}
        for regime in regimes:
            mask = pilot["regime"] == regime
            gain = _rms_calibration_gain(values[mask])
            gains[method][regime] = float(gain)
        for row in _block_metrics(evaluation_values[method], evaluation, gains[method]):
            rows.append({"method": method, **row})
    summary = {}
    for method in evaluation_values:
        selected = [row for row in rows if row["method"] == method]
        summary[method] = {
            "log_loss": float(np.mean([row["log_loss"] for row in selected])),
            "top1_accuracy": float(np.mean([row["top1_accuracy"] for row in selected])),
            "by_regime": {
                regime: {
                    "log_loss": float(np.mean([
                        row["log_loss"] for row in selected if row["regime"] == regime
                    ])),
                    "top1_accuracy": float(np.mean([
                        row["top1_accuracy"] for row in selected if row["regime"] == regime
                    ])),
                }
                for regime in sorted({row["regime"] for row in selected})
            },
        }
    physical_rows = {
        row["block_id"]: row for row in rows if row["method"] == PRIMARY_MODEL_ID
    }
    paired = {}
    for index, comparator in enumerate(
        (LINEAR_MODEL_ID, MATCHED_EXPONENTIAL_ID, LEARNED_BASIS_ID)
    ):
        comparator_rows = {
            row["block_id"]: row for row in rows if row["method"] == comparator
        }
        blocks = sorted(physical_rows)
        paired[comparator] = _paired_block_bootstrap(
            [physical_rows[block]["log_loss"] for block in blocks],
            [comparator_rows[block]["log_loss"] for block in blocks],
            seed=7000 + index,
            resamples=bootstrap_resamples,
        )
    result = {
        "schema_version": TEMPORAL_BENCHMARK_SCHEMA_VERSION,
        "analysis": "resource_matched_temporal_attribution",
        "protocol": {
            "pilot_seed": int(pilot_seed),
            "evaluation_seed": int(evaluation_seed),
            "evaluation_data_used_for_fitting": False,
            "trials_per_scenario_block": int(trials_per_block),
            "candidate_events_per_trial": 4,
            "synthetic_operating_point": {
                "voltage_v": 0.9, "tau_leak_s": 10.0,
                "cascade_depth": 3, "beta_leak": 1.0,
            },
            "physical_preferred_lag_s": float(physical_peak),
            "score": "predeclared causal-event softmax log loss",
            "amplitude_control": (
                "label-free unit-RMS within-trial score normalization fitted on "
                "the corresponding pilot regime"
            ),
        },
        "model_specifications": {
            model_id: device_model_spec(model_id)
            for model_id in (PRIMARY_MODEL_ID, LINEAR_MODEL_ID)
        },
        "learned_basis_by_regime": {
            regime: {
                key: value for key, value in learned.items() if key != "parameters"
            }
            for regime, learned in learned_by_regime.items()
        },
        "pilot_fitted_gains": gains,
        "held_out_blocks": rows,
        "summary": summary,
        "paired_bootstrap": paired,
        "method_provenance": TEMPORAL_CODESIGN_PROVENANCE,
        "interpretation": (
            "Regime-specific timing attribution under equal dynamic-state count. "
            "All kernels receive label-free pilot RMS scaling; the signed three-state basis "
            "is pilot-fitted separately for each predeclared task family and is a "
            "tuned upper comparator, not a material model. No result is a "
            "closed-loop learning or hardware-energy claim."
        ),
    }
    result["artifact_digest_sha256"] = _canonical_digest(_jsonable(result))
    return result


def structural_resource_table(*, state_word_bits: int = 16) -> list[dict]:
    """Return transparent per-synapse state and recurrence lower bounds.

    ``external_state_bits`` is zero for the physical row only under the explicit
    assumption that the material relaxation is used in situ.  It does not imply zero
    cell, selector, sensing, reward-routing or programming cost.
    """
    bits = int(state_word_bits)
    if bits < 1:
        raise ValueError("state_word_bits must be positive")
    return [
        {
            "implementation": "physical_headroom_in_material",
            "dynamic_states": 3,
            "externally_stored_state_words": 0,
            "shared_coefficient_words": 0,
            "external_state_bits_per_synapse": 0,
            "recurrence_multiplies_per_tick": 0,
            "recurrence_adds_per_tick": 0,
            "readout_multiplies_per_reward": 0,
            "physical_cells_per_signed_synapse": 2,
            "claim_limit": (
                "Assumes the fitted internal relaxation is supplied by the two "
                "differential cells; excludes selector, sense, reward line and write."
            ),
        },
        {
            "implementation": MATCHED_EXPONENTIAL_ID,
            "dynamic_states": 1,
            "externally_stored_state_words": 1,
            "shared_coefficient_words": 1,
            "external_state_bits_per_synapse": bits,
            "recurrence_multiplies_per_tick": 1,
            "recurrence_adds_per_tick": 1,
            "readout_multiplies_per_reward": 0,
            "physical_cells_per_signed_synapse": 2,
            "claim_limit": "Digital-state lower bound; weight-cell and routing cost excluded.",
        },
        {
            "implementation": "digital_linear_erlang_k3",
            "dynamic_states": 3,
            "externally_stored_state_words": 3,
            "shared_coefficient_words": 2,
            "external_state_bits_per_synapse": bits * 3,
            "recurrence_multiplies_per_tick": 6,
            "recurrence_adds_per_tick": 6,
            "readout_multiplies_per_reward": 0,
            "physical_cells_per_signed_synapse": 2,
            "claim_limit": "Forward-Euler lower bound before memory traffic and control.",
        },
        {
            "implementation": LEARNED_BASIS_ID,
            "dynamic_states": 3,
            "externally_stored_state_words": 3,
            "shared_coefficient_words": 6,
            "external_state_bits_per_synapse": bits * 3,
            "recurrence_multiplies_per_tick": 3,
            "recurrence_adds_per_tick": 3,
            "readout_multiplies_per_reward": 3,
            "physical_cells_per_signed_synapse": 2,
            "claim_limit": (
                "Three decays plus signed readout; excludes coefficient-training and "
                "memory-traffic cost."
            ),
        },
    ]


def digital_cost_equivalent(resource_row: dict, *,
                            multiply_cost: float, add_cost: float,
                            state_access_cost: float,
                            reward_readout_cost: float = 0.0) -> float:
    """Evaluate one resource row under caller-supplied technology costs.

    Costs may be energy, area-weighted energy or any other consistent unit.  No
    technology-specific defaults are supplied because doing so would turn a structural
    comparison into an unsupported hardware claim.
    """
    values = (multiply_cost, add_cost, state_access_cost, reward_readout_cost)
    if any(not np.isfinite(value) or value < 0 for value in values):
        raise ValueError("all technology costs must be finite and non-negative")
    return float(
        resource_row["recurrence_multiplies_per_tick"] * multiply_cost
        + resource_row["recurrence_adds_per_tick"] * add_cost
        + resource_row["externally_stored_state_words"] * state_access_cost
        + resource_row["readout_multiplies_per_reward"] * reward_readout_cost
    )


def timing_phase_envelope(
    *,
    voltages=(0.8, 0.9, 1.1, 1.3, 1.5),
    retention_definitions=DEFAULT_RETENTION_DEFINITIONS,
    depths=(2, 3, 4, 5),
    model_ids=(PRIMARY_MODEL_ID, LINEAR_MODEL_ID),
    dt_s: float = 0.05,
) -> list[dict]:
    """Propagate representation, depth and retention uncertainty into ``t*``.

    The returned range is a supported-model envelope, not a confidence interval.
    Direct held-bias quantiles and near-zero supplementary values can be supplied
    alongside the deliberately synthetic 10-s operating point without conflating
    their retention definitions.
    """
    rows = []
    for model_id in model_ids:
        for depth in depths:
            for voltage in voltages:
                for retention_id, retention, retention_source in retention_definitions:
                    peak = preferred_lag(
                        model_id=model_id, voltage_v=float(voltage),
                        tau_leak_s=float(retention), cascade_depth=int(depth),
                        beta_leak=1.0, dt_s=float(dt_s),
                    )
                    rows.append({
                        "model_id": str(model_id),
                        "cascade_depth": int(depth),
                        "voltage_v": float(voltage),
                        "tau_leak_s": float(retention),
                        "retention_id": str(retention_id),
                        "retention_source": str(retention_source),
                        "tau_r_s": float(tau_r(abs(float(voltage)))),
                        "preferred_lag_s": float(peak),
                    })
    return rows


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_codesign_reference(*, predictive_linkage_manifest_path=None,
                             phase_kwargs=None, **benchmark_kwargs) -> dict:
    benchmark = run_temporal_attribution_benchmark(**benchmark_kwargs)
    if predictive_linkage_manifest_path is None:
        predictive_linkage_manifest_path = (
            Path(__file__).resolve().parents[2]
            / "data/results/reference/predictive_linkage_manifest.json"
        )
    linkage_path = Path(predictive_linkage_manifest_path)
    if linkage_path.exists():
        linkage = json.loads(linkage_path.read_text(encoding="utf-8"))
        supported_candidates = linkage.get("selection", {}).get(
            "supported_candidates", linkage.get("supported_candidates", [])
        )
        supported_depths = linkage.get("supported_depths")
        if supported_depths is None:
            supported_depths = sorted({
                int(match.group(1))
                for candidate in supported_candidates
                if (match := re.fullmatch(r"(?:physical|linear)_k(\d+)", candidate))
            })
        linkage_record = {
            "available": True,
            "file_sha256": _sha256_file(linkage_path),
            "manifest_digest_sha256": linkage.get("manifest_digest_sha256"),
            "supported_candidates": supported_candidates,
            "supported_depths": supported_depths,
            "used_for_benchmark_fitting": False,
            "role": "provenance and uncertainty context only",
        }
    else:
        linkage_record = {
            "available": False,
            "used_for_benchmark_fitting": False,
            "role": "optional provenance only; synthetic benchmark remains runnable",
        }
    payload = {
        "schema_version": 1,
        "analysis": "temporal_credit_codesign_reference",
        "benchmark": benchmark,
        "resource_accounting": {
            "state_word_bits": 16,
            "rows": structural_resource_table(state_word_bits=16),
            "interpretation": (
                "Topology-derived lower bounds only. No absolute energy, area or "
                "throughput advantage is claimed without a circuit and peripherals."
            ),
        },
        "timing_phase_envelope": timing_phase_envelope(**(phase_kwargs or {})),
        "predictive_linkage": linkage_record,
        "method_provenance": TEMPORAL_CODESIGN_PROVENANCE,
    }
    payload["artifact_digest_sha256"] = _canonical_digest(_jsonable(payload))
    return payload


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_codesign_artifacts(reference: dict, *, json_path, block_csv_path,
                             phase_csv_path) -> dict:
    json_path = Path(json_path)
    block_csv_path = Path(block_csv_path)
    phase_csv_path = Path(phase_csv_path)
    for path in (json_path, block_csv_path, phase_csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(_jsonable(reference), indent=2, sort_keys=True, allow_nan=False)
        + "\n", encoding="utf-8",
    )
    block_rows = reference["benchmark"]["held_out_blocks"]
    with block_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(block_rows[0]))
        writer.writeheader()
        writer.writerows(block_rows)
    phase_rows = reference["timing_phase_envelope"]
    with phase_csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(phase_rows[0]))
        writer.writeheader()
        writer.writerows(phase_rows)
    return {
        "json": str(json_path), "blocks": str(block_csv_path),
        "phase": str(phase_csv_path),
    }


def render_codesign_figure(reference: dict, path) -> str:
    import matplotlib.pyplot as plt

    summary = reference["benchmark"]["summary"]
    regimes = ("physical_aligned", "early_delay", "recency", "broad_delay")
    methods = (
        PRIMARY_MODEL_ID, LINEAR_MODEL_ID,
        MATCHED_EXPONENTIAL_ID, LEARNED_BASIS_ID,
    )
    labels = {
        PRIMARY_MODEL_ID: "physical",
        LINEAR_MODEL_ID: "linear",
        MATCHED_EXPONENTIAL_ID: "matched exp.",
        LEARNED_BASIS_ID: "learned 3-state",
    }
    colors = {
        PRIMARY_MODEL_ID: "#3aa07a", LINEAR_MODEL_ID: "#4c78a8",
        MATCHED_EXPONENTIAL_ID: "#2f4b8f", LEARNED_BASIS_ID: "#d8902f",
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.1))
    x = np.arange(len(regimes))
    width = 0.19
    for index, method in enumerate(methods):
        axes[0].bar(
            x + (index - 1.5) * width,
            [summary[method]["by_regime"][regime]["log_loss"] for regime in regimes],
            width=width, color=colors[method], label=labels[method],
        )
    axes[0].axhline(math.log(4.0), color="0.45", ls="--", lw=0.9)
    axes[0].set_xticks(x, ["physical\naligned", "early\ndelay", "recency", "broad\ndelay"])
    axes[0].set(ylabel="held-out causal-event log loss",
                title="(c) timing prior by regime")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    resources = reference["resource_accounting"]["rows"]
    resource_map = {row["implementation"]: row for row in resources}
    bits = [
        resource_map["physical_headroom_in_material"]["external_state_bits_per_synapse"],
        resource_map["digital_linear_erlang_k3"]["external_state_bits_per_synapse"],
        resource_map[MATCHED_EXPONENTIAL_ID]["external_state_bits_per_synapse"],
        resource_map[LEARNED_BASIS_ID]["external_state_bits_per_synapse"],
    ]
    losses = [summary[method]["log_loss"] for method in methods]
    for method, x_value, y_value in zip(methods, bits, losses):
        axes[1].scatter(x_value, y_value, s=55, color=colors[method], zorder=3)
        axes[1].annotate(labels[method], (x_value, y_value), xytext=(4, 4),
                         textcoords="offset points", fontsize=7)
    axes[1].set(xlabel="external dynamic-state bits / synapse",
                ylabel="overall held-out log loss",
                title="(d) structural state trade-off")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.88", lw=0.5)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/results/reference")
    parser.add_argument("--figure")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    kwargs = ({
        "trials_per_block": 4,
        "learned_multistarts": 2,
        "bootstrap_resamples": 100,
    } if args.smoke else {})
    reference = build_codesign_reference(**kwargs)
    output = Path(args.output_dir)
    write_codesign_artifacts(
        reference,
        json_path=output / "codesign_reference.json",
        block_csv_path=output / "codesign_evaluation.csv",
        phase_csv_path=output / "codesign_phase.csv",
    )
    if args.figure:
        render_codesign_figure(reference, args.figure)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "TEMPORAL_BENCHMARK_SCHEMA_VERSION", "LEARNED_BASIS_ID",
    "MATCHED_EXPONENTIAL_ID", "DEFAULT_RETENTION_DEFINITIONS",
    "TEMPORAL_CODESIGN_PROVENANCE",
    "eligibility_kernel", "preferred_lag", "build_temporal_attribution_trials",
    "fit_learned_temporal_basis", "run_temporal_attribution_benchmark",
    "structural_resource_table", "digital_cost_equivalent",
    "timing_phase_envelope", "build_codesign_reference",
    "write_codesign_artifacts", "render_codesign_figure",
]
