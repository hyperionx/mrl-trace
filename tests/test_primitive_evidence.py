from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mrl_trace.falsification import build_falsification_predictions
from mrl_trace.kernel_theory import (
    linear_impulse_peak,
    linear_pole_rate,
    linear_rectangular_peak,
    linear_rectangular_response,
    physical_small_signal_peak,
    simulate_gate_response,
)
from mrl_trace.model_specs import LINEAR_MODEL_ID, PRIMARY_MODEL_ID
from mrl_trace.predictive_linkage import bootstrap_held_out_support
from mrl_trace.timing_benchmarks import (
    MATCHING_REGIMES,
    frozen_kernel_bank,
    matching_scales,
    run_matched_timing_benchmark,
)
from mrl_trace.claim_ledger import validate_claim_ledger


def test_linear_impulse_peak_and_two_crossover_limits() -> None:
    k, tau_r_s = 3, 12.0
    alpha = k / tau_r_s
    leak_limited = linear_impulse_peak(tau_r_s, 0.01, k)
    fill_limited = linear_impulse_peak(tau_r_s, 1e6, k)
    assert np.isclose(leak_limited, (k - 1) * 0.01, rtol=3e-3)
    assert np.isclose(fill_limited, (k - 1) / alpha, rtol=1e-4)
    assert np.isclose(
        leak_limited, (k - 1) / linear_pole_rate(tau_r_s, 0.01, k)
    )


def test_finite_rectangular_peak_matches_sampled_analytic_response() -> None:
    tau_r_s, tau_leak_s, k, duration = 4.0, 10.0, 3, 0.3
    peak = linear_rectangular_peak(tau_r_s, tau_leak_s, k, duration)
    time = np.linspace(0.0, 15.0, 30_001)
    response = linear_rectangular_response(
        time, tau_r_s, tau_leak_s, k, duration
    )
    assert abs(time[int(np.argmax(response))] - peak) < 1e-3
    assert peak > linear_impulse_peak(tau_r_s, tau_leak_s, k)


def test_linear_gate_converges_to_continuous_finite_pulse_law() -> None:
    time = np.arange(0.0, 20.0 + 0.001, 0.001)
    simulated = simulate_gate_response(
        time, model_id=LINEAR_MODEL_ID, voltage_v=0.9,
        tau_leak_s=10.0, k=3, duration_s=0.3, dt_s=0.001,
        tau_r_override=4.0,
    )
    analytic = linear_rectangular_response(time, 4.0, 10.0, 3, 0.3)
    assert np.max(np.abs(simulated["normalised_response"] - analytic)) < 3e-3
    assert abs(simulated["preferred_lag_s"]
               - linear_rectangular_peak(4.0, 10.0, 3, 0.3)) < 0.01


def test_physical_low_occupancy_limit_and_vmax_normalisation() -> None:
    time = np.arange(0.0, 45.0, 0.005)
    low = simulate_gate_response(
        time, model_id=PRIMARY_MODEL_ID, voltage_v=0.9, tau_leak_s=10.0,
        k=3, duration_s=0.01, drive_amplitude=1e-4, v_max=1.0, dt_s=0.005,
    )
    scaled = simulate_gate_response(
        time, model_id=PRIMARY_MODEL_ID, voltage_v=0.9, tau_leak_s=10.0,
        k=3, duration_s=0.01, drive_amplitude=1e-4, v_max=7.0, dt_s=0.005,
    )
    expected = physical_small_signal_peak(10.0, 3)
    assert abs(low["preferred_lag_s"] - expected) < 0.2
    assert np.allclose(low["normalised_response"], scaled["normalised_response"],
                       atol=2e-5)


def test_repeat_stratified_support_bootstrap_is_deterministic() -> None:
    rows = []
    means = {}
    for candidate, base in (("linear_k2", 0.10), ("linear_k3", 0.101),
                            ("physical_k3", 0.20)):
        values = []
        for group in (0.8, 0.9, 1.1):
            for trial in (1, 2, 3):
                value = base + 0.001 * trial
                values.append(value)
                rows.append({"held_out_group": group, "trial": trial,
                             "candidate": candidate, "nrmse": value})
        means[candidate] = float(np.mean(values))
    comparison = {"rows": rows, "mean_nrmse": means}
    first = bootstrap_held_out_support(comparison, samples=500, seed=7)
    second = bootstrap_held_out_support(comparison, samples=500, seed=7)
    assert first == second
    assert first["full_refit"] is False
    assert first["cascade_depth_identified"] is False
    assert first["candidate_summary"]["physical_k3"]["within_5pct_frequency"] == 0


def test_matching_definitions_are_label_free_and_exact() -> None:
    bank = frozen_kernel_bank(np.linspace(0.0, 60.0, 3001))
    scales = matching_scales(bank, pilot_delays_s=np.linspace(0.0, 60.0, 257))
    assert scales["label_free"] is True
    for kernel, values in bank["values"].items():
        record = scales["by_kernel"][kernel]
        assert np.isclose(np.max(np.abs(values)) * record["unit_peak"], 1.0)
        assert all(record[regime] > 0 for regime in MATCHING_REGIMES)


def test_timing_scale_treatments_cannot_create_an_amplitude_win() -> None:
    result = run_matched_timing_benchmark(
        trials_per_block=8, bootstrap_samples=200,
    )
    assert result["protocol"]["pilot_seed"] != result["protocol"]["evaluation_seed"]
    for kernel in result["summary"]["pilot_rms"]:
        losses = [result["summary"][regime][kernel]["log_loss"]
                  for regime in MATCHING_REGIMES]
        assert max(losses) - min(losses) < 1e-3
    assert "two-terminal" not in result["scalar_aliasing_limit"]
    assert "additional accessible state" in result["scalar_aliasing_limit"]


def test_falsification_manifest_is_deterministic_and_untested() -> None:
    first = build_falsification_predictions(
        samples=4, voltage_grid=(0.5, 0.9), dt_s=0.05
    )
    second = build_falsification_predictions(
        samples=4, voltage_grid=(0.5, 0.9), dt_s=0.05
    )
    assert first["manifest_digest_sha256"] == second["manifest_digest_sha256"]
    assert first["status"] == "untested_prediction"
    assert len(first["numbered_falsifiers"]) == 3
    assert first["claim_limit"].startswith("These are frozen compact-model predictions")
    assert "simultaneous_95band_s" in first["bias_conditioned_preferred_lag"][0]
    assert "rms_departure_simultaneous_95band" in first[
        "nonlinear_departure"
    ]["rows"][0]


def test_falsification_batch_matches_the_frozen_scalar_gate() -> None:
    prediction = build_falsification_predictions(
        samples=1, voltage_grid=(0.9,), dt_s=0.05
    )
    time = np.asarray(prediction["normalised_pulse_off"]["delay_s"])
    scalar = simulate_gate_response(
        time, model_id=PRIMARY_MODEL_ID, voltage_v=0.9,
        tau_leak_s=10.0, k=3, duration_s=0.3, dt_s=0.05,
        tau_r_override=145.0 * np.exp(-2.9 * 0.9),
    )
    assert np.allclose(
        prediction["normalised_pulse_off"]["median"],
        scalar["normalised_response"], atol=1e-12, rtol=1e-12,
    )


def test_tracked_claim_ledger_resolves_every_artifact() -> None:
    root = Path(__file__).resolve().parents[1]
    result = validate_claim_ledger(
        root / "claims.yaml", repository_root=root,
    )
    assert result["claims_validated"] == 12
