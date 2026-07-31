"""Scientific-integrity regressions for delayed-reward task claims."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from mrl_trace.maze import (
    ActionSequenceTrack,
    DelayedCuedChoice,
    EpropTrace,
    ShallowEpropPolicyTrace,
    TMaze,
    _dmax_assemble,
    calibrate_comparator_scales,
    run_action_sequence,
)


def _roll_constant_policy(action: int) -> bool:
    env = ActionSequenceTrack()
    pos = env.start(1)
    for _ in range(env.L):
        pos, reached, done = env.step(pos, np.array([action]))
        if done[0]:
            return bool(reached[0])
    return False


def test_action_sequence_requires_four_consequential_decisions() -> None:
    env = ActionSequenceTrack()
    assert env.required_actions == (0, 1, 1, 0)
    pos = env.start(1)
    for index, expected in enumerate(env.required_actions):
        wrong = (expected + 1) % env.n_actions
        _, reached_wrong, done_wrong = env.step(pos, np.array([wrong]))
        assert done_wrong[0] and not reached_wrong[0]

        pos, reached, done = env.step(pos, np.array([expected]))
        assert bool(reached[0]) is (index == env.L - 1)
        assert bool(done[0]) is (index == env.L - 1)


def test_no_constant_action_policy_solves_default_sequence() -> None:
    assert not _roll_constant_policy(0)
    assert not _roll_constant_policy(1)


def test_historical_names_warn_and_preserve_types() -> None:
    with pytest.warns(DeprecationWarning, match="one-choice delayed contextual"):
        historical_task = TMaze()
    assert isinstance(historical_task, DelayedCuedChoice)

    with pytest.warns(DeprecationWarning, match="custom shallow"):
        historical_trace = EpropTrace(1, 1, 2, dt=0.01)
    assert isinstance(historical_trace, ShallowEpropPolicyTrace)


def test_balanced_calibration_records_scale_and_provenance() -> None:
    calibration = calibrate_comparator_scales(
        trajectories=4,
        tau_leak=1.0,
        D=0.02,
        dt=0.02,
        step_dur=0.02,
        include_rule_ablations=False,
        tau_r_override=0.1,
    )
    assert calibration["rewarded"] == calibration["unrewarded"] == 2
    assert calibration["required_actions"] == [0, 1, 1, 0]
    assert set(calibration["records"]) == {
        "device", "exponential", "conventional_rstdp", "shallow_eprop", "no_trace",
    }
    for name, record in calibration["records"].items():
        assert record["eligibility_normalizer"] > 0
        assert set(record["method_provenance"]) == {
            "status", "established_basis", "repository_adaptation", "claim_limit",
        }
        if name != "no_trace":
            assert np.isfinite(record["raw_effective_update_rms"])


def test_fixed_calibration_actions_do_not_shift_comparator_rng(monkeypatch) -> None:
    import mrl_trace.maze as maze

    observed = {}

    def run(label, **condition):
        draws = []

        def fake_lif(v, charge, dt, rng, **kwargs):
            draws.append((charge.copy(), rng.random(v.shape)))
            zeros = np.zeros_like(v)
            if kwargs.get("return_pre"):
                return v, zeros, v.copy()
            return v, zeros

        monkeypatch.setattr(maze, "lif_step_batched", fake_lif)
        env = ActionSequenceTrack()
        forced = maze._balanced_calibration_actions(env, 4, env.L)
        maze.train_sequential(
            env, B=4, episodes=1, max_steps=env.L, tau_leak=1.0,
            D=0.02, dt=0.02, step_dur=0.02, eta=0.0, seed0=123,
            forced_actions=forced, tau_r_override=0.1, **condition,
        )
        observed[label] = draws

    run("device")
    run("shallow_eprop", eprop=True)
    assert len(observed["device"]) == len(observed["shallow_eprop"])
    for device, eprop in zip(observed["device"], observed["shallow_eprop"]):
        assert np.array_equal(device[0], eprop[0])
        assert np.array_equal(device[1], eprop[1])


def test_tuning_and_evaluation_seed_blocks_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="overlap"):
        run_action_sequence(tuning_seeds=(1000,), evaluation_seeds=(1000,))


def test_exponential_comparator_receives_frozen_decay_match(monkeypatch) -> None:
    import mrl_trace.maze as maze
    from mrl_trace.bandit import AbstractTrace
    from mrl_trace.device import decay_matched_exponential_tau

    observed = []

    class RecordingTrace(AbstractTrace):
        def __init__(self, *args, tau_elig=10.0, **kwargs):
            observed.append(float(tau_elig))
            super().__init__(*args, tau_elig=tau_elig, **kwargs)

    monkeypatch.setattr(maze, "AbstractTrace", RecordingTrace)
    maze.train_sequential(
        ActionSequenceTrack(),
        B=1,
        episodes=1,
        tau_leak=2.5,
        D=0.02,
        dt=0.02,
        step_dur=0.02,
        abstract=True,
    )
    expected = decay_matched_exponential_tau(2.5, V=1.5)
    assert observed == [pytest.approx(expected)]
    assert observed[0] != pytest.approx(2.5)


def test_controlled_runner_exposes_comparator_diagnostics() -> None:
    result = run_action_sequence(
        episodes=1,
        tuning_episodes=1,
        calibration_trajectories=4,
        learning_rates=(0.1,),
        tuning_seeds=(1000,),
        evaluation_seeds=(2000,),
        tau_leak=1.0,
        D=0.02,
        dt=0.02,
        step_dur=0.02,
        include_rule_ablations=False,
        tau_r_override=0.1,
    )
    assert result["seed_partition"] == {
        "tuning": [1000], "evaluation": [2000], "disjoint": True,
    }
    expected = {
        "raw_trace_peak", "raw_trace_area", "eligibility_normalizer",
        "selected_eta", "raw_effective_update_rms",
        "normalized_effective_update_rms", "matched_exponential_tau_s",
    }
    for diagnostic in result["comparator_diagnostics"].values():
        assert set(diagnostic) == expected


def test_inline_parallel_mapping_matches_serial_controlled_runner() -> None:
    class InlinePool:
        def map(self, function, jobs, chunksize=1):
            return list(map(function, jobs))

    kwargs = dict(
        episodes=2, tuning_episodes=2, calibration_trajectories=4,
        learning_rates=(0.1,), tuning_seeds=(1000,),
        evaluation_seeds=(2000,), tau_leak=1.0, D=0.02, dt=0.02,
        step_dur=0.02, include_rule_ablations=False, tau_r_override=0.1,
    )
    serial = run_action_sequence(**kwargs)
    parallel_interface = run_action_sequence(
        **kwargs, workers=2, pool=InlinePool()
    )
    for name in serial["curves"]:
        assert np.array_equal(serial["curves"][name], parallel_interface["curves"][name])
    assert {
        name: serial["tuning"]["conditions"][name]["selected_eta"]
        for name in serial["tuning"]["conditions"]
    } == {
        name: parallel_interface["tuning"]["conditions"][name]["selected_eta"]
        for name in parallel_interface["tuning"]["conditions"]
    }


@pytest.mark.skipif(sys.platform != "win32", reason="Windows spawn regression")
def test_real_windows_spawn_runs_controlled_action_sequence() -> None:
    result = run_action_sequence(
        episodes=1, tuning_episodes=1, calibration_trajectories=4,
        learning_rates=(0.1,), tuning_seeds=(1000,),
        evaluation_seeds=(2000,), tau_leak=1.0, D=0.02, dt=0.02,
        step_dur=0.02, include_rule_ablations=False, tau_r_override=0.1,
        workers=2,
    )
    assert result["seed_partition"]["disjoint"] is True
    assert all(values.shape == (1, 1) for values in result["curves"].values())


def test_retention_delay_payload_is_explicitly_simulation_only() -> None:
    results = [
        (1.0, 1.0, 0.9, 0.8, 1.0, np.array([0.9])),
        (1.0, 2.0, 0.6, 0.5, 0.7, np.array([0.6])),
        (2.0, 1.0, 0.95, 0.9, 1.0, np.array([0.95])),
        (2.0, 2.0, 0.8, 0.7, 0.9, np.array([0.8])),
    ]
    payload = _dmax_assemble(
        results, taus=[1.0, 2.0], delays=[1.0, 2.0], crit=0.75,
        seeds=1, episodes=1, V=1.5,
    )
    assert payload["simulation_only"] is True
    assert payload["independent_physical_validation"] is False
    assert payload["threshold_delay"] == payload["dmax"]
    assert payload["method_provenance"]["status"] == "proposed"
