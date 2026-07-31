from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_dopamine_sound_markers_form_rewarded_trial_pairs() -> None:
    from mrl_trace.dopamine import pair_cue_events

    reward = np.arange(1, 51, dtype=float) * 10.0
    sound = np.ravel(np.column_stack((reward - 2.0, reward - 1.0)))
    trials = pair_cue_events(sound, reward)
    assert trials.shape == (50, 3)
    assert np.array_equal(trials[:, 2], reward)
    assert np.all(trials[:, 0] < trials[:, 1])
    assert np.all(trials[:, 1] < trials[:, 2])


def test_dopamine_pairing_fails_closed_instead_of_inventing_omissions() -> None:
    from mrl_trace.dopamine import pair_cue_events

    with pytest.raises(ValueError, match="two sound markers per reward"):
        pair_cue_events([8.0, 9.0, 18.0], [10.0, 20.0])


def test_reward_aligned_epochs_have_one_row_per_valid_reward() -> None:
    from mrl_trace.dopamine import reward_aligned_epochs

    time = np.arange(0.0, 31.0, 0.05)
    reward = np.array([10.0, 20.0])
    session = {
        "dff_t": time,
        "dff": np.sin(time / 4.0),
        "sound_t": np.array([8.0, 9.0, 18.0, 19.0]),
        "reward_t": reward,
    }
    epochs, relative, events = reward_aligned_epochs(
        session, tmin=-2.0, tmax=2.0, baseline=(-1.5, -0.5)
    )
    assert epochs.shape == (2, len(relative))
    assert np.array_equal(events[:, 2], reward)


def test_eligibility_surrogate_has_no_unused_space_charge_state() -> None:
    from mrl_trace.bandit import GateBankBatched
    from mrl_trace.device import CascadeEligibilityGate, TransientGate
    from mrl_trace.distal_reward import SpikingGateBank

    gates = [
        CascadeEligibilityGate(),
        GateBankBatched(1, 2, 2),
        SpikingGateBank(2),
    ]
    assert all(not hasattr(gate, "vsc") for gate in gates)
    with pytest.warns(DeprecationWarning):
        legacy = TransientGate(tau_d_override=123.0)
    assert not hasattr(legacy, "vsc")


def test_stretched_discharge_is_explicit_and_beta_one_is_backward_compatible() -> None:
    from mrl_trace.device import CascadeEligibilityGate

    default = CascadeEligibilityGate(tau_leak=1.5, k=2, dt=0.01,
                                     tau_r_override=0.2)
    beta_one = CascadeEligibilityGate(tau_leak=1.5, k=2, dt=0.01,
                                      tau_r_override=0.2, beta_leak=1.0)
    stretched = CascadeEligibilityGate(tau_leak=1.5, k=2, dt=0.01,
                                       tau_r_override=0.2, beta_leak=0.6)
    drive = [1.0] * 10 + [0.0] * 100
    y_default = np.asarray([default.step(value) for value in drive])
    y_beta_one = np.asarray([beta_one.step(value) for value in drive])
    y_stretched = np.asarray([stretched.step(value) for value in drive])
    assert np.array_equal(y_default, y_beta_one)
    assert not np.allclose(y_default, y_stretched)
    assert stretched.beta_leak == 0.6


@pytest.mark.parametrize("k", [1, 3, 5])
def test_no_leak_unit_step_matches_identified_erlang_candidate(k: int) -> None:
    """The learning gate must implement the Erlang model used for identification."""
    from scipy.special import gammainc

    from mrl_trace.device import CascadeEligibilityGate

    dt = 2.0e-4
    tau_r = 0.8
    time = np.arange(dt, 3.0 * tau_r + dt / 2.0, dt)
    gate = CascadeEligibilityGate(
        tau_leak=1.0e12,
        k=k,
        dt=dt,
        tau_r_override=tau_r,
    )
    observed = np.asarray([gate.step(1.0) for _ in time])
    expected = gammainc(k, k * time / tau_r)

    # This is a forward-Euler implementation of the exact continuous-time cascade;
    # the tolerance is intentionally much tighter than plotting/fit resolution.
    np.testing.assert_allclose(observed, expected, rtol=2.0e-3, atol=4.0e-4)


def test_scalar_batched_and_spiking_cascades_share_identical_dynamics() -> None:
    from mrl_trace.bandit import GateBankBatched
    from mrl_trace.device import CascadeEligibilityGate
    from mrl_trace.distal_reward import SpikingGateBank

    kwargs = {
        "tau_leak": 1.7,
        "k": 3,
        "dt": 0.002,
        "tau_r_override": 0.3,
        "beta_leak": 0.65,
    }
    scalar = CascadeEligibilityGate(vnmax=1.3, **kwargs)
    batched = GateBankBatched(1, 1, 1, Vnmax=1.3, **kwargs)
    spiking = SpikingGateBank(1, Vnmax=1.3, **kwargs)
    drive = np.concatenate((
        np.ones(25),
        np.zeros(70),
        np.full(15, -0.4),
        np.zeros(90),
    ))

    y_scalar = np.asarray([scalar.step(value) for value in drive])
    y_batched = np.asarray([
        batched.step(np.asarray([[[value]]]))[0, 0, 0] for value in drive
    ])
    y_spiking = np.asarray([
        spiking.step(np.asarray([value]))[0] for value in drive
    ])

    np.testing.assert_allclose(y_batched, y_scalar, rtol=0.0, atol=1.0e-14)
    np.testing.assert_allclose(y_spiking, y_scalar, rtol=0.0, atol=1.0e-14)


def test_exponential_control_matches_full_post_peak_decay_not_nominal_tau() -> None:
    from mrl_trace.device import decay_matched_exponential_tau

    # For a one-stage beta=1 gate after the pulse, the decay rate is the sum of
    # the intrinsic cascade rate and the separate leakage rate.
    nominal_tau = 2.0
    rise_tau = 0.5
    expected = 1.0 / (1.0 / rise_tau + 1.0 / nominal_tau)
    matched = decay_matched_exponential_tau(
        nominal_tau, k=1, tau_r_override=rise_tau, beta_leak=1.0
    )
    assert matched == pytest.approx(expected, rel=0.01)
    assert matched != pytest.approx(nominal_tau)


def test_method_provenance_distinguishes_established_and_proposed() -> None:
    from mrl_trace.bandit import BANDIT_METHOD_PROVENANCE
    from mrl_trace.biosignal import EEG_METHOD_PROVENANCE
    from mrl_trace.deep import DEEP_METHOD_PROVENANCE
    from mrl_trace.device import CASCADE_METHOD_PROVENANCE
    from mrl_trace.distal_cue import DISTAL_CUE_METHOD_PROVENANCE
    from mrl_trace.distal_reward import DISTAL_METHOD_PROVENANCE
    from mrl_trace.extensions import (EXTENSIONS_METHOD_PROVENANCE,
                                      RETENTION_DELAY_METHOD_PROVENANCE)
    from mrl_trace.hybrid import HYBRID_METHOD_PROVENANCE
    from mrl_trace.learning import SIGNED_RULE_PROVENANCE, THREE_FACTOR_PROVENANCE
    from mrl_trace.probselect import PST_METHOD_PROVENANCE
    from mrl_trace.selectivity import SELECTIVITY_METHOD_PROVENANCE

    assert THREE_FACTOR_PROVENANCE["status"] == "established"
    assert SIGNED_RULE_PROVENANCE["status"] == "proposed"
    assert DEEP_METHOD_PROVENANCE["status"] == "proposed"
    assert CASCADE_METHOD_PROVENANCE["status"] == "adapted"
    assert PST_METHOD_PROVENANCE["status"] == "adapted"
    required = {"status", "established_basis", "repository_adaptation", "claim_limit"}
    for record in (
        BANDIT_METHOD_PROVENANCE,
        EEG_METHOD_PROVENANCE,
        DEEP_METHOD_PROVENANCE,
        CASCADE_METHOD_PROVENANCE,
        DISTAL_CUE_METHOD_PROVENANCE,
        DISTAL_METHOD_PROVENANCE,
        EXTENSIONS_METHOD_PROVENANCE,
        RETENTION_DELAY_METHOD_PROVENANCE,
        HYBRID_METHOD_PROVENANCE,
        SIGNED_RULE_PROVENANCE,
        THREE_FACTOR_PROVENANCE,
        PST_METHOD_PROVENANCE,
        SELECTIVITY_METHOD_PROVENANCE,
    ):
        assert set(record) == required


def test_signed_rule_ablations_are_numerically_distinct() -> None:
    from mrl_trace.learning import coincidence_drive

    pre = np.array([1.0, 1.0])
    post = np.array([True, False])
    signed = coincidence_drive(pre, post, mode="signed", ltd=0.6)
    unsigned = coincidence_drive(pre, post, mode="unsigned", ltd=0.6)
    no_negative = coincidence_drive(pre, post, mode="no_negative", ltd=0.6)
    assert np.array_equal(signed, [1.0, -0.6])
    assert np.array_equal(unsigned, [1.0, 0.6])
    assert np.array_equal(no_negative, [1.0, 0.0])


def test_fault_scope_names_omitted_nonidealities() -> None:
    from mrl_trace.device_faults import siox_fault_stack

    scope = siox_fault_stack(p_stuck=0.2, sigma_g=0.5, pf_on=False).describe()
    assert "stuck_off" in scope["included_nonidealities"]
    assert "sampled_device_to_device_lognormal" in scope["included_nonidealities"]
    assert "line_resistance" in scope["excluded_nonidealities"]
    assert "drift" in scope["excluded_nonidealities"]


def test_deep_trace_ablation_keeps_homeostasis_fixed(monkeypatch) -> None:
    import mrl_trace.deep as deep

    observed = {}

    def fake_train_deep(*, mode, B, trials, seed0, **kwargs):
        observed.update(mode=mode, homeo=kwargs.get("homeo"))
        return np.zeros((B, trials), dtype=float)

    monkeypatch.setattr(deep, "train_deep", fake_train_deep)
    deep._deep_local_one(
        "no_trace_homeo", seeds=2, trials=210, hp={}, homeo=0.1
    )
    assert observed == {"mode": "no_trace", "homeo": 0.1}
    assert ("no_trace_homeo_dist", "no_trace", deep.DEEP_DMS_HOMEO, True) in (
        deep.DEEP_DMS_CONDS
    )


def test_reversal_grid_normalizes_updates_and_limits_mechanistic_claim() -> None:
    from mrl_trace.bandit import run_reversal_grid

    result = run_reversal_grid(
        [1.0], B=2, trials=8, calibration_trajectories=4,
        calibration_trials=2, dt=0.02, cue_dur=0.04, D=0.02,
        device_k=2, tau_r_override=0.1, beta_leak=0.8,
        retention_definition="measured_held_bias_quantiles",
    )
    for record in result["calibration"]["records"].values():
        assert record["normalized_effective_update_rms"] in {0.0, 1.0}
        assert record["trace_reset_each_trial"] is True
    assert "not persistence" in result["interpretation"]
    assert result["retention_definition"] == "measured_held_bias_quantiles"


def test_reversal_summary_treats_unsolved_seeds_as_right_censored() -> None:
    from mrl_trace.bandit import kaplan_meier_recovery_summary

    summary = kaplan_meier_recovery_summary(
        times=[2, 4, 10, 10], censored=[False, False, True, True], horizon=10
    )
    assert summary["censored_fraction"] == 0.5
    assert summary["recovered_fraction"] == 0.5
    assert summary["kaplan_meier_median_trials"] == 4.0
    # RMST is explicitly restricted to follow-up; censored records remain non-events.
    assert summary["restricted_mean_trials"] == pytest.approx(6.5)
    assert summary["median_status"] == "observed"


def test_withdrawn_dopamine_learning_api_is_not_public() -> None:
    import mrl_trace.dopamine as dopamine

    for name in ("decode_reward", "build_reward_pools", "run_dopamine_shallow",
                 "run_dopamine_deep"):
        assert not hasattr(dopamine, name)


def test_withdrawn_dopamine_archive_cannot_be_loaded_or_saved() -> None:
    from mrl_trace import paths

    with pytest.raises(ValueError, match="scientifically invalid"):
        paths.load_result("exp11_dopamine_capstone.npy")
    with pytest.raises(ValueError, match="scientifically invalid"):
        paths.save_result("exp11_dopamine_capstone.npy", {"invalid": True})


def test_retention_archives_fail_closed_without_direct_definition(tmp_path) -> None:
    from mrl_trace.extensions import load_measured_tau

    legacy = tmp_path / "legacy_retention.npz"
    np.savez(legacy, tau=np.linspace(0.6, 2.0, 12))
    with pytest.raises(ValueError, match="legacy or transformed"):
        load_measured_tau(archive=legacy)


def test_direct_retention_archive_preserves_fitted_tau(tmp_path) -> None:
    from mrl_trace.extensions import load_measured_tau

    expected = np.linspace(0.6, 2.0, 12)
    archive = tmp_path / "direct_retention.npz"
    np.savez(
        archive,
        tau=expected,
        retention_definition=np.asarray("direct_held_bias_tau"),
        analysis_schema_version=np.asarray(2),
    )
    observed, source = load_measured_tau(archive=archive)
    assert np.array_equal(observed, expected)
    assert "direct held-bias" in source


def test_public_sources_do_not_use_transformed_retention() -> None:
    sources = [ROOT / "experiments" / "00_device_physics_and_trace.ipynb",
               ROOT / "experiments" / "06_nmi_predictive_linkage.ipynb"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert "tau_fill_corrected" not in text
    assert "fill_corrected_tau" not in text
    assert "tau / (1-rho)" not in text


def test_notebook_ito_fitter_treats_floor_as_offset_not_time_rescaling() -> None:
    import ast
    import json
    import math

    from scipy.optimize import least_squares

    notebook = json.loads(
        (ROOT / "experiments" / "06_nmi_predictive_linkage.ipynb").read_text(
            encoding="utf-8"
        )
    )
    definition = None
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]), filename=f"ito-cell-{index}")
        definition = next(
            (node for node in tree.body
             if isinstance(node, ast.FunctionDef) and node.name == "_fit_ito_trace"),
            definition,
        )
    assert definition is not None
    namespace = {"np": np, "math": math, "least_squares": least_squares}
    ast.fix_missing_locations(definition)
    exec(compile(ast.Module(body=[definition], type_ignores=[]),
                 "notebook-ito-fitter", "exec"), namespace)

    time = np.linspace(0.0, 15.0, 301)
    expected_tau = 2.3
    fitted = [
        namespace["_fit_ito_trace"](
            time, np.exp(-np.power(time / expected_tau, 0.7)) + floor
        )["tau_held_s"]
        for floor in (0.05, 0.40)
    ]
    np.testing.assert_allclose(fitted, expected_tau, rtol=0.01)
    assert np.isclose(fitted[0], fitted[1], rtol=0.01)


def test_notebook_separates_near_zero_reads_from_held_bias() -> None:
    import ast
    import json

    notebook = json.loads(
        (ROOT / "experiments" / "06_nmi_predictive_linkage.ipynb").read_text(
            encoding="utf-8"
        )
    )
    selected = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]), filename=f"ito-cell-{index}")
        for node in tree.body:
            is_threshold = (
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name)
                        and target.id == "NEAR_ZERO_MAX_V"
                        for target in node.targets)
            )
            if is_threshold or (
                isinstance(node, ast.FunctionDef)
                and node.name == "_measurement_regime"
            ):
                selected.append(node)
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
                 "notebook-measurement-regime", "exec"), namespace)

    classify = namespace["_measurement_regime"]
    assert classify(0.001) == "near_zero_read"
    assert classify(-0.002) == "near_zero_read"
    assert classify(0.0021) == "held_bias"
    assert classify(0.1) == "held_bias"
    source = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
    )
    assert 'row["measurement_regime"] == "held_bias"' in source
    assert '"near_zero_read"' in source
