from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


def _minimal_gold_export(root: Path) -> Path:
    directory = root / "gold_export"
    directory.mkdir()
    for index in range(24):
        (directory / f"trace_fixture_{index:02d}.csv").touch()
    (directory / "dataset.csv").write_text("fixture\n", encoding="utf-8")
    (directory / "manifest.csv").write_text("fixture\n", encoding="utf-8")
    return directory


def test_ito_source_cohort_is_date_frozen_and_includes_declared_supplements(
        tmp_path: Path) -> None:
    from mrl_trace.paths import select_ito_source_cohort

    for run in range(91):
        date = "01-15-2025" if run < 45 else "01-16-2025"
        (tmp_path / f"instrument voltage bias#1 Run{run} {date}.xls").touch()
    # Use the exact instrument names because the supplemental acquisition is frozen
    # by identity rather than accepted by date or directory order.
    supplements = [
        tmp_path / "vcv Site@1 Subsite 2-wire-resistor voltage bias#1 Run471 06-26-2026.xls",
        tmp_path / "vcv Site@1 Subsite 2-wire-resistor voltage bias#1 Run498 06-26-2026.xls",
    ]
    for path in supplements:
        path.touch()
    extra = tmp_path / "instrument voltage bias#1 Run999 07-01-2026.xls"
    extra.touch()

    cohort = select_ito_source_cohort(tmp_path)
    assert len(cohort["primary_workbooks"]) == 91
    assert list(cohort["supplemental_workbooks"]) == supplements
    assert list(cohort["additional_workbooks"]) == [extra]
    assert len(cohort["workbooks"]) == 93
    assert cohort["primary_acquisition_dates"] == ("01-15-2025", "01-16-2025")


def test_physical_scalar_batched_spiking_parity_and_arbitrary_vmax() -> None:
    from mrl_trace.bandit import GateBankBatched
    from mrl_trace.device import CascadeEligibilityGate
    from mrl_trace.distal_reward import SpikingGateBank

    drive = np.r_[np.ones(20), np.zeros(80), -np.ones(10), np.zeros(30)]
    outputs = []
    for vmax in (0.4, 2.7):
        scalar = CascadeEligibilityGate(
            tau_leak=1.7, k=3, dt=0.01, vnmax=vmax, tau_r_override=0.6
        )
        batched = GateBankBatched(
            1, 1, 1, tau_leak=1.7, k=3, dt=0.01, Vnmax=vmax,
            tau_r_override=0.6,
        )
        spiking = SpikingGateBank(
            1, tau_leak=1.7, k=3, dt=0.01, Vnmax=vmax,
            tau_r_override=0.6,
        )
        rows = []
        for value in drive:
            rows.append((
                float(scalar.step(value)),
                float(batched.step(np.asarray([[[value]]]))[0, 0, 0]),
                float(spiking.step(np.asarray([value]))[0]),
            ))
        rows = np.asarray(rows)
        np.testing.assert_allclose(rows[:, 0], rows[:, 1], atol=1e-14)
        np.testing.assert_allclose(rows[:, 0], rows[:, 2], atol=1e-14)
        outputs.append(rows[:, 0])
    # Normalized upstream transfer makes V_max a state scale, not a hidden gain.
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-14)


def test_physical_single_state_drive_off_is_exponential() -> None:
    from mrl_trace.device import CascadeEligibilityGate

    dt, tau = 1e-3, 0.8
    gate = CascadeEligibilityGate(
        tau_leak=tau, k=1, dt=dt, tau_r_override=0.2
    )
    for _ in range(100):
        gate.step(1.0)
    start = float(gate.step(0.0))
    observed = np.asarray([gate.step(0.0) for _ in range(500)])
    time = dt * np.arange(1, observed.size + 1)
    np.testing.assert_allclose(
        observed / start, np.exp(-time / tau), rtol=1e-3, atol=1e-4
    )


def test_linear_erlang_sensitivity_retains_analytic_limit() -> None:
    from scipy.special import gammainc
    from mrl_trace.device import LinearErlangEligibilityGate

    dt, tau_r, k = 2e-4, 0.8, 3
    time = np.arange(dt, 2 * tau_r, dt)
    gate = LinearErlangEligibilityGate(
        tau_leak=1e12, k=k, dt=dt, tau_r_override=tau_r
    )
    observed = np.asarray([gate.step(1.0) for _ in time])
    np.testing.assert_allclose(
        observed, gammainc(k, k * time / tau_r), rtol=2e-3, atol=4e-4
    )


def test_model_specification_identity_and_manifest_are_deterministic() -> None:
    from mrl_trace.model_specs import (
        PRIMARY_MODEL_ID, LINEAR_MODEL_ID, build_predictive_linkage_manifest,
        device_model_spec, select_supported_state_space,
    )

    physical = device_model_spec(PRIMARY_MODEL_ID)
    linear = device_model_spec(LINEAR_MODEL_ID)
    assert physical["spec_digest_sha256"] != linear["spec_digest_sha256"]
    physical["default_voltage_v"] = -1
    assert device_model_spec(PRIMARY_MODEL_ID)["default_voltage_v"] == 0.9
    assert len(physical["tau_r_law"]["source_sha256"]) == 64
    source = Path(__file__).resolve().parents[1] / "data/device_model/kww_final.json"
    source_text = source.read_text(encoding="utf-8").replace("\r\n", "\n")
    canonical_source = source_text.replace("\r", "\n").replace("\n", "\r\n")
    assert physical["tau_r_law"]["source_sha256"] == hashlib.sha256(
        canonical_source.encode("utf-8")
    ).hexdigest()
    selection = select_supported_state_space({
        "kww": 1.0, "physical_k3": 1.04, "physical_k4": 1.06,
        "linear_k3": 1.02,
    })
    assert selection["supported_candidates"] == [
        "kww", "linear_k3", "physical_k3"
    ]
    assert selection["physical_within_tolerance"] is True
    kwargs = {
        "candidate_scores": {"linear_k3": 1.02, "physical_k3": 1.04},
        "source_digests": {"b": "2", "a": "1"},
        "parameter_estimates": {"tau": 2.0},
        "retention_definition": "direct_held_bias_tau",
        "software_versions": {"numpy": "2"},
    }
    assert build_predictive_linkage_manifest(**kwargs) == (
        build_predictive_linkage_manifest(**kwargs)
    )


def test_shared_candidate_response_linear_and_physical_are_separate() -> None:
    from scipy.special import gammainc
    from mrl_trace.model_specs import PRIMARY_MODEL_ID, LINEAR_MODEL_ID
    from mrl_trace.predictive_linkage import candidate_response

    time = np.linspace(0, 2, 41)
    theta = np.asarray([np.log(1.0), 0.0, np.log(1e12), 0.0])
    linear = candidate_response(time, 0.9, theta, LINEAR_MODEL_ID, k=3)
    physical = candidate_response(time, 0.9, theta, PRIMARY_MODEL_ID, k=3)
    np.testing.assert_allclose(linear, gammainc(3, 3 * time), rtol=1e-10)
    assert np.max(np.abs(physical - linear)) > 0.01


def test_empirical_fit_residual_dimension_is_stable_across_amplitude_sign() -> None:
    from mrl_trace.model_specs import LINEAR_MODEL_ID
    from mrl_trace.predictive_linkage import fit_candidate

    time = np.linspace(0.0, 2.0, 41)
    # Opposite-polarity traces force the affine amplitude through different signs
    # during optimisation; the penalty vector must retain a fixed length.
    traces = [
        {"time": time, "current": sign * np.exp(-time), "bias": 0.9,
         "trial": index}
        for index, sign in enumerate((1.0, -1.0))
    ]
    theta = fit_candidate(traces, LINEAR_MODEL_ID, k=3, max_nfev=20)
    assert np.isfinite(theta).all()


def test_aulc_tuning_expands_only_pilot_grid(monkeypatch) -> None:
    import mrl_trace.maze as maze

    conditions = maze._comparator_conditions(False)
    calibration = {
        "records": {
            name: {"eligibility_normalizer": 1.0} for name in conditions
        }
    }

    def fake_job(job):
        name, _, _, eta, seed, *_ = job
        # Initial edge eta=1 wins; expanded eta=10 is worse, so eta=1 is frozen.
        score = 1.0 if eta == 1.0 else (0.5 if eta == 10.0 else 0.2)
        return name, float(eta), int(seed), score, score

    monkeypatch.setattr(maze, "_action_tuning_job", fake_job)
    result = maze.tune_comparator_learning_rates(
        calibration=calibration, learning_rates=(0.1, 1.0),
        tuning_seeds=(1000, 1001), episodes=3, include_rule_ablations=False,
        max_boundary_expansions=2,
    )
    assert result["grid_expansions"] == [{
        "reason": "pilot_optimum_on_boundary", "added_rates": [10.0]
    }]
    assert result["evaluation_data_used"] is False
    assert all(
        record["selected_eta"] == 1.0
        for name, record in result["conditions"].items() if name != "no_trace"
    )
    assert result["conditions"]["no_trace"]["selected_eta"] == 0.0


def test_all_device_notebooks_record_shared_model_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    for index in range(7):
        notebook = next((root / "experiments").glob(f"{index:02d}_*.ipynb"))
        source = notebook.read_text(encoding="utf-8")
        assert "device_model_spec" in source, notebook.name
        assert "PRIMARY_MODEL_ID" in source, notebook.name


def test_restored_gold_manifest_has_24_digest_addressed_traces() -> None:
    root = Path(__file__).resolve().parents[1]
    directory = root / "data" / "device_model" / "gold_export"
    traces = sorted(directory.glob("trace_*.csv"))
    if len(traces) != 24:
        pytest.skip("raw Au traces are intentionally excluded from Git")
    assert len(traces) == 24
    manifest = (directory / "manifest.csv").read_text(encoding="utf-8")
    for path in traces:
        assert path.name in manifest
        assert hashlib.sha256(path.read_bytes()).hexdigest()


def test_restored_gold_traces_reproduce_published_kww_laws() -> None:
    from mrl_trace.device import fit_kww_laws

    root = Path(__file__).resolve().parents[1]
    if len(list((root / "data/device_model/gold_export").glob("trace_*.csv"))) != 24:
        pytest.skip("raw Au traces are intentionally excluded from Git")
    expected = json.loads(
        (root / "data/device_model/kww_final.json").read_text(encoding="utf-8")
    )["laws"]
    observed = fit_kww_laws()["laws"]
    for name in ("tr0", "cr", "td0", "cd", "beta"):
        assert observed[name] == pytest.approx(expected[name], rel=1e-6, abs=1e-10)


def test_publication_preflight_rejects_absent_and_unverified_ito(tmp_path: Path) -> None:
    from mrl_trace.paths import publication_device_preflight

    gold = _minimal_gold_export(tmp_path)
    missing = tmp_path / "missing.npz"
    with pytest.raises(FileNotFoundError, match="source-verified schema-2"):
        publication_device_preflight(gold_raw_dir=gold, ito_archive=missing)

    unverified = tmp_path / "unverified.npz"
    np.savez(
        unverified,
        retention_definition="direct_held_bias_tau",
        analysis_schema_version=2,
        source_verified=False,
        source_manifest_sha256="a" * 64,
    )
    with pytest.raises(FileNotFoundError, match="source-verified schema-2"):
        publication_device_preflight(gold_raw_dir=gold, ito_archive=unverified)


def test_publication_preflight_accepts_only_source_verified_schema2(tmp_path: Path) -> None:
    from mrl_trace.paths import publication_device_preflight

    gold = _minimal_gold_export(tmp_path)
    archive = tmp_path / "verified.npz"
    np.savez(
        archive,
        retention_definition="direct_held_bias_tau",
        analysis_schema_version=2,
        source_verified=True,
        source_manifest_sha256="A" * 64,
    )
    result = publication_device_preflight(gold_raw_dir=gold, ito_archive=archive)
    assert result == {
        "gold_traces": 24,
        "ito_evidence": "source_verified_schema_2_archive",
        "source_manifest_sha256": "a" * 64,
    }


def test_tracked_reference_artifacts_and_tex_macros_agree() -> None:
    root = Path(__file__).resolve().parents[1]
    reference = root / "data" / "results" / "reference"
    sequential = json.loads((reference / "sequential_reference.json").read_text())
    dms = json.loads((reference / "dms_reference.json").read_text())
    interval = json.loads((reference / "interval_reference.json").read_text())
    macros = (reference / "benchmark_macros.tex").read_text(encoding="utf-8")

    assert sequential["protocol"]["seed_partition"]["tuning"] == list(range(1000, 1020))
    assert sequential["protocol"]["seed_partition"]["evaluation"] == list(range(2000, 2020))
    assert sequential["calibration"]["trajectories"] == 256
    assert dms["protocol"]["trials"] == 2500
    assert dms["calibration"]["trials"] == 256
    assert interval["protocol"]["eta_values"] == [0.01, 0.1, 0.5]
    for model in interval["summary"].values():
        assert model["direction_holds_all_eta"] is True
        for values in model["means"].values():
            assert values["device"] > 1.0
            assert values["matched_exponential"] < 1.0
    assert sequential["model_specifications"]["physical_headroom_v1"][
        "spec_digest_sha256"
    ]
    assert "\\newcommand{\\SequentialDeviceAULC}{0.445}" in macros
    assert "\\newcommand{\\DMSDeviceAULC}{0.996}" in macros
    assert "\\newcommand{\\IntervalPhysicalDeviceEtaLow}{3.98}" in macros


@pytest.mark.external
@pytest.mark.publication
def test_invoked_publication_device_preflight_fails_closed() -> None:
    import os
    from mrl_trace.paths import publication_device_preflight

    if os.getenv("MRL_RUN_EXTERNAL_DATA", "0") != "1":
        pytest.skip("set MRL_RUN_EXTERNAL_DATA=1 to invoke publication preflight")
    publication_device_preflight(
        gold_raw_dir=os.getenv("MRL_TRACE_GOLD_DIR"),
        ito_raw_dir=os.getenv("MRL_TRACE_ITO_DIR"),
        ito_archive=os.getenv("MRL_TRACE_ITO_ARCHIVE"),
    )
