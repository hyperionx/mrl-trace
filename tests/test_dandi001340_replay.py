from __future__ import annotations

import inspect
import json
import os
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from mrl_trace.dopamine import (
    DANDISET_VERSION,
    LoggedSession,
    airpls,
    asset_manifest,
    download_dandi001340,
    load_logged_session,
    prepare_dandi001340,
    preprocess_photometry,
    robust_reference_regression,
)
from mrl_trace.dopamine_replay import (
    ReplayParameters,
    evaluate_logged_replay_loso,
    run_logged_replay,
    trial_modulators,
)


def _session(
    mouse: str = "mouse-1",
    session: str = "session-1",
    *,
    quality: bool = False,
) -> LoggedSession:
    n_trials = 6
    action = np.asarray((-1, 1, -1, 1, 1, -1), dtype=np.int8)
    outcome = np.asarray((0, 1, 1, 0, 1, 0), dtype=np.int8)
    segment_time, segment_signal, offsets = [], [], [0]
    center_out = np.arange(n_trials, dtype=float) * 3.0
    outcome_time = center_out + 1.0
    for start in center_out:
        time = start + np.linspace(0.0, 2.0, 81)
        signal = np.square(time - start)
        segment_time.append(time)
        segment_signal.append(signal)
        offsets.append(offsets[-1] + len(time))
    trace_time = np.arange(0.0, n_trials * 3.0 + 2.0, 0.025)
    trace_signal = np.sin(trace_time / 2.0)
    return LoggedSession(
        mouse_id=mouse,
        session_id=session,
        trial_id=np.arange(1, n_trials + 1, dtype=np.int32),
        action=action,
        rewarded=outcome,
        center_in_s=center_out - 0.5,
        center_out_s=center_out,
        side_in_s=center_out + 0.8,
        outcome_s=outcome_time,
        waveform_offsets=np.asarray(offsets, dtype=np.int64),
        waveform_time_s=np.concatenate(segment_time),
        waveform_dlight_z=np.concatenate(segment_signal),
        trace_time_s=trace_time,
        trace_dlight_z=trace_signal,
        source_sha256="a" * 64,
        preprocessing={"detrend_window_s": 60.0},
        quality_pass=quality,
    )


def test_pinned_manifest_has_all_69_immutable_assets() -> None:
    manifest = asset_manifest()
    assert DANDISET_VERSION == "0.250221.0527"
    assert len(manifest) == 69
    assert len({row.path for row in manifest}) == 69
    assert len({row.asset_id for row in manifest}) == 69
    assert all(row.size > 0 and len(row.sha256) == 64 for row in manifest)


def test_downloads_are_explicitly_opt_in(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        download_dandi001340(tmp_path)


def test_airpls_and_huber_regression_are_deterministic() -> None:
    time = np.arange(0.0, 20.0, 0.05)
    baseline = 0.02 * np.square(time)
    signal = baseline + np.exp(-np.square(time - 8.0) / 0.2)
    first, first_meta = airpls(signal, time, window_s=5.0)
    second, second_meta = airpls(signal, time, window_s=5.0)
    np.testing.assert_array_equal(first, second)
    assert first_meta == second_meta
    expected_lambda = (2 * np.sin(np.pi * 0.05 / 5.0)) ** -4
    assert first_meta["lambda"] == pytest.approx(expected_lambda)
    assert first_meta["max_iter"] == 50
    assert first_meta["tolerance"] == pytest.approx(1e-3)

    reference = np.linspace(-2.0, 2.0, 101)
    observed = 1.2 + 0.7 * reference
    observed[50] += 50.0
    fitted, metadata = robust_reference_regression(reference, observed)
    assert metadata["huber_c"] == pytest.approx(1.345)
    assert metadata["max_iter"] == 50
    assert metadata["intercept"] == pytest.approx(1.2, abs=1e-5)
    assert metadata["slope"] == pytest.approx(0.7, abs=1e-5)
    assert fitted.shape == observed.shape


def test_photometry_alignment_uses_actual_470_timestamps() -> None:
    time_470 = np.arange(0.02, 12.0, 0.05)
    time_415 = np.arange(0.0, 12.02, 0.047)
    reference = np.sin(time_415 / 3.0)
    signal = 0.4 * np.interp(time_470, time_415, reference)
    signal += np.exp(-np.square(time_470 - 6.0) / 0.3)
    aligned_time, dlight, metadata = preprocess_photometry(
        signal,
        time_470,
        reference,
        time_415,
        detrend_window_s=4.0,
    )
    np.testing.assert_array_equal(aligned_time, time_470)
    assert np.mean(dlight) == pytest.approx(0.0, abs=1e-12)
    assert np.std(dlight) == pytest.approx(1.0, abs=1e-12)
    assert metadata["implementation"] == "mrl_trace_independent_approximation"


def test_logged_session_round_trip_is_non_pickle_and_state_free(tmp_path: Path) -> None:
    session = _session()
    path = session.save(tmp_path / "logged.npz")
    with np.load(path, allow_pickle=False) as archive:
        assert not any(array.dtype.hasobject for array in archive.values())
        assert not any("state" in name.lower() for name in archive.files)
        metadata = json.loads(str(archive["preprocessing_json"]))
        assert "state" not in json.dumps(metadata).lower()
    restored = load_logged_session(path)
    assert restored.mouse_id == session.mouse_id
    assert restored.session_id == session.session_id
    np.testing.assert_array_equal(restored.action, session.action)
    np.testing.assert_array_equal(
        restored.waveform_dlight_z, session.waveform_dlight_z
    )
    assert not any(field.name == "state" for field in fields(LoggedSession))
    with pytest.raises(ValueError, match="hidden state"):
        replace(session, preprocessing={"state": "right"})


@pytest.mark.parametrize("gate_model", ["physical_headroom_v1", "linear_erlang_v1"])
def test_complete_waveform_overlap_and_kernel_area_matching(gate_model: str) -> None:
    session = _session()
    plain = trial_modulators(session, "plain_dlight")
    # Integral from center-out 0 through outcome+1 at 2 of (t^2 - value_at_outcome).
    np.testing.assert_allclose(plain, 2.0 / 3.0, atol=2.1e-4)

    from mrl_trace.dopamine_replay import _frozen_kernel_tables

    time, device, exponential = _frozen_kernel_tables(gate_model)
    np.testing.assert_allclose(
        np.trapezoid(device, time),
        np.trapezoid(exponential, time),
        rtol=1e-12,
    )
    assert np.all(device >= 0)
    assert np.all(exponential >= 0)


def test_shuffle_and_shift_controls_are_deterministic() -> None:
    base = _session()
    varied = base.waveform_dlight_z.copy()
    for index in range(base.n_trials):
        start, stop = base.waveform_offsets[index:index + 2]
        varied[start:stop] *= index + 1
    session = replace(base, waveform_dlight_z=varied)
    device = trial_modulators(session, "device")
    shuffled_1 = trial_modulators(session, "shuffled_device", seed=12)
    shuffled_2 = trial_modulators(session, "shuffled_device", seed=12)
    np.testing.assert_array_equal(shuffled_1, shuffled_2)
    np.testing.assert_allclose(np.sort(shuffled_1), np.sort(device))
    shifted_1 = trial_modulators(session, "shifted_device")
    shifted_2 = trial_modulators(session, "shifted_device")
    np.testing.assert_array_equal(shifted_1, shifted_2)


def test_prediction_precedes_current_outcome_and_waveform() -> None:
    session = _session()
    params = ReplayParameters(bias=0.2, kappa=-0.3, eta=0.4)
    index = 2

    changed_outcome = session.rewarded.copy()
    changed_outcome[index] = 1 - changed_outcome[index]
    outcome_session = replace(session, rewarded=changed_outcome)
    original_outcome = run_logged_replay(session, "outcome_rl", params)
    altered_outcome = run_logged_replay(outcome_session, "outcome_rl", params)
    assert altered_outcome["probability_right"][index] == pytest.approx(
        original_outcome["probability_right"][index]
    )

    changed_waveform = session.waveform_dlight_z.copy()
    start, stop = session.waveform_offsets[index:index + 2]
    changed_waveform[start:stop] *= 3.0
    waveform_session = replace(session, waveform_dlight_z=changed_waveform)
    original_waveform = run_logged_replay(session, "device", params)
    altered_waveform = run_logged_replay(waveform_session, "device", params)
    assert altered_waveform["probability_right"][index] == pytest.approx(
        original_waveform["probability_right"][index]
    )


def test_teacher_forcing_follows_recorded_action_not_model_prediction() -> None:
    session = _session()
    modulators = np.zeros(session.n_trials)
    modulators[0] = 2.0
    result = run_logged_replay(
        session,
        "device",
        ReplayParameters(bias=20.0, kappa=0.0, eta=0.5),
        modulators=modulators,
    )
    assert result["probability_right"][0] > 0.999
    assert session.action[0] == -1
    np.testing.assert_allclose(result["weights_before"][1], (1.0, 0.0))
    assert not result["score_mask"][0]


def test_loso_parameter_fit_never_receives_held_out_mouse(monkeypatch) -> None:
    import mrl_trace.dopamine_replay as replay

    sessions = [_session(f"mouse-{index}", f"session-{index}") for index in range(3)]
    observed_training_sets = []

    def fake_fit(training, condition, **kwargs):
        observed_training_sets.append(frozenset(item.mouse_id for item in training))
        return ReplayParameters()

    monkeypatch.setattr(replay, "_fit_parameters", fake_fit)
    evaluate_logged_replay_loso(
        sessions,
        conditions=("previous_choice",),
        n_starts=1,
        n_boot=20,
        include_quality_sensitivity=False,
    )
    assert set(observed_training_sets) == {
        frozenset(("mouse-1", "mouse-2")),
        frozenset(("mouse-0", "mouse-2")),
        frozenset(("mouse-0", "mouse-1")),
    }


def test_predictive_linkage_manifest_is_metadata_only() -> None:
    sessions = [_session("mouse-a", "session-a"), _session("mouse-b", "session-b")]
    first = evaluate_logged_replay_loso(
        sessions, conditions=("previous_choice",), n_starts=1, n_boot=20,
        include_quality_sensitivity=False,
        predictive_linkage_manifest={"candidate_scores": {"physical_k3": 1.0}},
    )
    second = evaluate_logged_replay_loso(
        sessions, conditions=("previous_choice",), n_starts=1, n_boot=20,
        include_quality_sensitivity=False,
        predictive_linkage_manifest={"candidate_scores": {"physical_k3": 999.0}},
    )
    assert first["predictions"] == second["predictions"]
    first_meta = first["summary"]["configuration"]["predictive_linkage_manifest"]
    second_meta = second["summary"]["configuration"]["predictive_linkage_manifest"]
    assert first_meta["used_for_computation"] is False
    assert first_meta["sha256"] != second_meta["sha256"]


def test_default_replay_metadata_finds_tracked_predictive_manifest() -> None:
    from mrl_trace.dopamine_replay import _predictive_linkage_metadata

    metadata = _predictive_linkage_metadata()
    assert metadata["available"] is True
    assert metadata["used_for_computation"] is False
    assert metadata["path"].endswith(
        "data\\results\\reference\\predictive_linkage_manifest.json"
    ) or metadata["path"].endswith(
        "data/results/reference/predictive_linkage_manifest.json"
    )
    assert len(metadata["sha256"]) == 64


def test_no_learning_signature_or_public_api_retains_reward_pools() -> None:
    import mrl_trace
    from mrl_trace.bandit import train
    from mrl_trace.deep import train_deep
    from mrl_trace.probselect import train_pst

    for function in (train, train_deep, train_pst):
        assert "reward_pools" not in inspect.signature(function).parameters
        assert "reward_pools" not in inspect.getsource(function)
    for withdrawn in (
        "build_reward_pools",
        "run_biosignal_reward",
        "run_eeg_capstone",
    ):
        assert not hasattr(mrl_trace, withdrawn)


def test_nwb_disposition_validation(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    from mrl_trace.dopamine import (
        _behavior_arrays,
        _photometry_status,
        _valid_behavior_mask,
    )

    path = tmp_path / "sub-test_ses-unit.nwb"
    with h5py.File(path, "w") as nwb:
        trials = nwb.create_group("intervals/trials")
        text = h5py.string_dtype("utf-8")
        trials.create_dataset("trial", data=np.arange(3))
        trials.create_dataset("action", data=np.asarray(("left", "right", "left"), dtype=text))
        trials.create_dataset("state", data=np.asarray(("left", "left", "right"), dtype=text))
        trials.create_dataset("rewarded", data=(1, 0, 0))
        trials.create_dataset("center_in", data=(1.0, 2.0, 3.0))
        trials.create_dataset("center_out", data=(1.1, 2.1, 3.1))
        trials.create_dataset("side_in", data=(1.2, 2.2, 3.2))
        trials.create_dataset("outcome", data=(1.3, 2.3, 3.3))
        trials.create_dataset("last_side_out", data=(1.4, 2.4, np.nan))
        values = _behavior_arrays(nwb)
        assert _valid_behavior_mask(values).tolist() == [True, True, True]
        assert _photometry_status(nwb, values)[0] == "behavior_only"
        acquisition = nwb.create_group("acquisition")
        for channel in ("fp_series_415nm", "fp_series_470nm"):
            series = acquisition.create_group(channel)
            series.create_dataset("data", data=np.ones(20))
            series.create_dataset("timestamps", data=np.arange(20.0))
        assert _photometry_status(nwb, values)[0] == "truncated"


@pytest.mark.external
def test_dandi001340_external_reproduction_targets(tmp_path: Path) -> None:
    if os.getenv("MRL_RUN_EXTERNAL_DATA", "0") != "1":
        pytest.skip("set MRL_RUN_EXTERNAL_DATA=1 to invoke pinned DANDI acceptance")
    raw = os.getenv("MRL_DANDI001340_RAW_DIR")
    if not raw:
        pytest.fail("MRL_DANDI001340_RAW_DIR is required for invoked DANDI acceptance")
    manifest = prepare_dandi001340(raw, tmp_path / "cache", tmp_path / "output")
    primary = manifest["reports"]["60"]
    assert primary["n_valid_behavior_trials"] == 58_590
    assert primary["n_behavior_only_sessions"] == 21
    assert primary["n_truncated_sessions"] == 2
    assert primary["n_substantive_sessions"] == 46
    assert primary["n_aligned_trials"] == 39_020
    assert primary["n_quality_sessions"] == 37
    assert primary["n_quality_trials"] == 32_373
    assert primary["device_overlap_reward_omission_auc"] == pytest.approx(
        0.731, abs=0.01
    )
    assert primary["device_exponential_overlap_correlation"] == pytest.approx(
        0.94, abs=0.01
    )

    sessions = [
        load_logged_session(path)
        for path in sorted((tmp_path / "cache" / "window_60s").glob("*.npz"))
    ]
    replay = evaluate_logged_replay_loso(sessions)
    assert {"device", "matched_exponential", "linear_device",
            "linear_matched_exponential"}.issubset(
        replay["summary"]["pooled_log_loss"]
    )
    assert set(replay["summary"]["configuration"]["model_specifications"]) == {
        "physical_headroom_v1", "linear_erlang_v1"
    }
    expected = {
        "previous_choice": 0.5096,
        "plain_dlight": 0.4976,
        "device": 0.5048,
        "shuffled_device": 0.5093,
        "outcome_rl": 0.3999,
    }
    for condition, target in expected.items():
        assert replay["summary"]["pooled_log_loss"][condition] == pytest.approx(
            target, abs=0.01
        )
    interval = replay["summary"]["device_minus_shuffled_bootstrap"]["ci95"]
    assert interval[0] == pytest.approx(-0.0132, abs=0.005)
    assert interval[1] == pytest.approx(0.0071, abs=0.005)
    for report in manifest["reports"].values():
        assert report["verdict"] == "Conditional Go"
        assert report["n_substantive_sessions"] == 46
        assert report["n_aligned_trials"] == 39_020
        assert report["n_quality_sessions"] == 37
        assert report["n_quality_trials"] == 32_373
        assert len(report["quality_mice"]) == 5
        assert report["device_overlap_reward_omission_auc"] > 0.5
