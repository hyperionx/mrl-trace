from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def test_preferred_lag_exposes_distinct_physical_and_linear_priors() -> None:
    from mrl_trace.codesign import preferred_lag
    from mrl_trace.model_specs import LINEAR_MODEL_ID, PRIMARY_MODEL_ID

    common = dict(
        voltage_v=0.9, tau_leak_s=10.0, cascade_depth=3,
        beta_leak=1.0, dt_s=0.05,
    )
    physical = preferred_lag(model_id=PRIMARY_MODEL_ID, **common)
    linear = preferred_lag(model_id=LINEAR_MODEL_ID, **common)
    assert 15.0 < physical < 25.0
    assert 3.0 < linear < 8.0
    assert physical > 2.0 * linear


def test_benchmark_fits_pilot_only_and_is_deterministic() -> None:
    from mrl_trace.codesign import run_temporal_attribution_benchmark

    kwargs = dict(
        pilot_seed=3100, trials_per_block=4, learned_multistarts=1,
        bootstrap_resamples=100,
    )
    first = run_temporal_attribution_benchmark(evaluation_seed=9100, **kwargs)
    repeated = run_temporal_attribution_benchmark(evaluation_seed=9100, **kwargs)
    different_evaluation = run_temporal_attribution_benchmark(
        evaluation_seed=9200, **kwargs
    )

    assert first["artifact_digest_sha256"] == repeated["artifact_digest_sha256"]
    assert first["protocol"]["evaluation_data_used_for_fitting"] is False
    assert first["pilot_fitted_gains"] == different_evaluation["pilot_fitted_gains"]
    for regime, fitted in first["learned_basis_by_regime"].items():
        other = different_evaluation["learned_basis_by_regime"][regime]
        np.testing.assert_allclose(fitted["tau_s"], other["tau_s"])
        np.testing.assert_allclose(
            fitted["readout_weights"], other["readout_weights"]
        )
    assert first["artifact_digest_sha256"] != (
        different_evaluation["artifact_digest_sha256"]
    )


def test_structural_resources_keep_shared_coefficients_out_of_synapse_bits() -> None:
    from mrl_trace.codesign import (
        LEARNED_BASIS_ID,
        MATCHED_EXPONENTIAL_ID,
        digital_cost_equivalent,
        structural_resource_table,
    )

    rows = {row["implementation"]: row for row in structural_resource_table()}
    assert rows["physical_headroom_in_material"]["external_state_bits_per_synapse"] == 0
    assert rows[MATCHED_EXPONENTIAL_ID]["external_state_bits_per_synapse"] == 16
    assert rows["digital_linear_erlang_k3"]["external_state_bits_per_synapse"] == 48
    assert rows[LEARNED_BASIS_ID]["external_state_bits_per_synapse"] == 48
    assert rows[LEARNED_BASIS_ID]["shared_coefficient_words"] == 6
    cost = digital_cost_equivalent(
        rows[LEARNED_BASIS_ID], multiply_cost=2.0, add_cost=1.0,
        state_access_cost=3.0, reward_readout_cost=4.0,
    )
    assert cost == 30.0


def test_phase_envelope_labels_retention_definition() -> None:
    from mrl_trace.codesign import timing_phase_envelope
    from mrl_trace.model_specs import PRIMARY_MODEL_ID

    rows = timing_phase_envelope(
        voltages=(0.9,),
        retention_definitions=(("direct_test", 1.5, "test definition"),),
        depths=(3,), model_ids=(PRIMARY_MODEL_ID,), dt_s=0.1,
    )
    assert len(rows) == 1
    assert rows[0]["retention_id"] == "direct_test"
    assert rows[0]["retention_source"] == "test definition"
    assert rows[0]["preferred_lag_s"] > 0


def test_predictive_linkage_is_recorded_as_provenance_not_fit_input(
        tmp_path: Path) -> None:
    from mrl_trace.codesign import build_codesign_reference

    manifest = tmp_path / "predictive.json"
    manifest.write_text(json.dumps({
        "manifest_digest_sha256": "declared-digest",
        "supported_depths": [2, 3, 4, 5],
    }), encoding="utf-8")
    reference = build_codesign_reference(
        predictive_linkage_manifest_path=manifest,
        phase_kwargs={
            "voltages": (0.9,),
            "retention_definitions": (("direct_test", 1.5, "test"),),
            "depths": (3,),
        },
        trials_per_block=3, learned_multistarts=1, bootstrap_resamples=50,
    )
    linkage = reference["predictive_linkage"]
    assert linkage["available"] is True
    assert linkage["manifest_digest_sha256"] == "declared-digest"
    assert linkage["supported_depths"] == [2, 3, 4, 5]
    assert linkage["used_for_benchmark_fitting"] is False


def test_predictive_linkage_derives_depths_from_supported_candidates(
        tmp_path: Path) -> None:
    from mrl_trace.codesign import build_codesign_reference

    manifest = tmp_path / "predictive.json"
    manifest.write_text(json.dumps({
        "manifest_digest_sha256": "declared-digest",
        "selection": {
            "supported_candidates": [
                "kww", "linear_k2", "linear_k3", "linear_k4", "linear_k5",
            ],
        },
    }), encoding="utf-8")
    reference = build_codesign_reference(
        predictive_linkage_manifest_path=manifest,
        phase_kwargs={
            "voltages": (0.9,),
            "retention_definitions": (("direct_test", 1.5, "test"),),
            "depths": (3,),
        },
        trials_per_block=3, learned_multistarts=1, bootstrap_resamples=50,
    )
    linkage = reference["predictive_linkage"]
    assert linkage["supported_candidates"] == [
        "kww", "linear_k2", "linear_k3", "linear_k4", "linear_k5",
    ]
    assert linkage["supported_depths"] == [2, 3, 4, 5]
    assert linkage["used_for_benchmark_fitting"] is False
