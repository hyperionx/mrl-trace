from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from mrl_trace.coddington import (
    CODDINGTON_EXPECTED_MD5,
    CODDINGTON_GROUPS,
    CoddingtonSubject,
    _mouse_bootstrap,
    _source_style_two_way,
    evaluate_coddington_causal_learning,
    load_coddington_dataset,
    write_coddington_tex_macros,
)


def _subject() -> CoddingtonSubject:
    values = np.arange(8, dtype=float)
    return CoddingtonSubject(
        mouse_id=3,
        group="stimLick-",
        session_id=np.ones(8),
        lick_state=np.asarray([0, 1] * 4, dtype=float),
        stimulation=np.asarray([1, 0] * 4, dtype=float),
        reward_collection_latency_ms=np.linspace(100, 800, 8),
        reward_dopamine_z=np.linspace(0, 1, 8),
        preparatory_lick_hz=values + 0.5,
        baseline_lick_hz=values,
    )


def test_frozen_group_map_is_complete_and_disjoint() -> None:
    identifiers = [mouse for group in CODDINGTON_GROUPS.values() for mouse in group]
    assert sorted(identifiers) == list(range(1, 25))
    assert len(set(identifiers)) == 24
    assert len(CODDINGTON_GROUPS["control"]) == 9
    assert len(CODDINGTON_GROUPS["stimLick-"]) == 6
    assert len(CODDINGTON_GROUPS["stimLick+"]) == 5
    assert len(CODDINGTON_EXPECTED_MD5) == 32


def test_subject_baseline_correction_is_trial_aligned() -> None:
    subject = _subject()
    np.testing.assert_array_equal(
        subject.baseline_corrected_preparatory_lick(),
        np.full(subject.n_trials, 0.5),
    )


def test_source_style_two_way_detects_frozen_contingency_direction() -> None:
    block = np.arange(8, dtype=float)
    minus = np.stack([0.25 * block + offset for offset in np.linspace(-0.2, 0.2, 6)])
    plus = np.stack([0.25 * block - 0.8 + offset for offset in np.linspace(-0.2, 0.2, 5)])
    result = _source_style_two_way(minus, plus)
    assert result["denominator_df"] == 72
    assert result["contrast_stimLick_minus_minus_plus_hz"] == pytest.approx(0.8)
    assert result["f_value"] > 100
    assert result["p_value"] < 1e-10


def test_mouse_bootstrap_is_deterministic_and_uses_mouse_units() -> None:
    minus = np.arange(48, dtype=float).reshape(6, 8) / 10
    plus = np.arange(40, dtype=float).reshape(5, 8) / 10
    first = _mouse_bootstrap(minus, plus, n_resamples=500, seed=19)
    second = _mouse_bootstrap(minus, plus, n_resamples=500, seed=19)
    assert first == second
    assert first["n_resamples"] == 500


def test_coddington_tex_macros_are_generated_from_reference(tmp_path) -> None:
    result = {
        "source_style_inference": {
            "contrast_stimLick_minus_minus_plus_hz": 0.7266,
            "f_value": 12.0306,
            "denominator_df": 72,
            "p_value": 0.0008873,
        },
        "mouse_resampling_sensitivity": {
            "bootstrap_probability_positive": 0.9663,
            "ci95_hz": [-0.0611, 1.5195],
        },
        "dopamine_manipulation_check": {
            "stimLick-": {"mean_z": 1.146},
            "stimLick+": {"mean_z": 1.237},
        },
        "manifest_digest_sha256": "90f9ea3b205bc58005e436f0be1c81754",
    }
    target = write_coddington_tex_macros(result, tmp_path / "macros.tex")
    text = target.read_text(encoding="utf-8")
    assert "\\newcommand{\\CoddingtonFValue}{12.03}" in text
    assert "\\newcommand{\\CoddingtonBootstrapProbability}{96.6\\%}" in text
    assert "90f9ea3b205b" in text


@pytest.mark.external
def test_public_coddington_v1_reproduces_causal_signature() -> None:
    source = os.environ.get("CODDINGTON_DATA")
    if not source:
        pytest.skip("set CODDINGTON_DATA to the external seshMerge.mat")
    subjects = load_coddington_dataset(Path(source))
    assert len(subjects) == 24
    result = evaluate_coddington_causal_learning(source)
    inference = result["source_style_inference"]
    assert result["verdict"] == "positive_action_contingent_causal_learning"
    assert result["all_block_contrasts_positive"]
    assert inference["contrast_stimLick_minus_minus_plus_hz"] == pytest.approx(
        0.7265877447, abs=1e-9
    )
    assert inference["f_value"] == pytest.approx(12.03055076, abs=1e-7)
    assert inference["p_value"] == pytest.approx(0.00088733235, abs=1e-10)
    assert result["mouse_resampling_sensitivity"]["ci95_hz"][0] < 0
    for group in ("stimLick-", "stimLick+"):
        assert result["dopamine_manipulation_check"][group]["ci95_z"][0] > 0
