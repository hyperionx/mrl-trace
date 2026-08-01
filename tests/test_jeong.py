from __future__ import annotations

import numpy as np

from mrl_trace.jeong import (
    JEONG_CONTROL_SUBJECTS,
    JEONG_INHIBITED_SUBJECTS,
    _independent_bootstrap,
    _subject_and_day,
)


def test_jeong_figure6_cohort_matches_public_source() -> None:
    assert len(JEONG_CONTROL_SUBJECTS) == 7
    assert len(JEONG_INHIBITED_SUBJECTS) == 6
    assert "HJ-FP-datHT-stGtACR-M8" not in JEONG_INHIBITED_SUBJECTS
    assert len(set(JEONG_CONTROL_SUBJECTS + JEONG_INHIBITED_SUBJECTS)) == 13


def test_jeong_subject_day_parser_excludes_random_reward_and_other_cohorts() -> None:
    path = (
        "sub-HJ-FP-datHT-stGtACR-M4/"
        "sub-HJ-FP-datHT-stGtACR-M4_ses-Day7-Pavlovian.nwb"
    )
    assert _subject_and_day(path) == ("HJ-FP-datHT-stGtACR-M4", 7)
    assert _subject_and_day(path.replace("Pavlovian", "RandomRewards")) is None
    assert _subject_and_day(path.replace("Day7-Pavlovian", "Day1")) is None
    assert _subject_and_day(path.replace("M4", "M8")) is None


def test_jeong_mouse_bootstrap_is_deterministic_and_directional() -> None:
    control = np.asarray([3.0, 4.0, 5.0, 4.5])
    inhibited = np.asarray([0.5, 1.0, 1.5])
    first = _independent_bootstrap(
        control, inhibited, n_resamples=1000, seed=23
    )
    second = _independent_bootstrap(
        control, inhibited, n_resamples=1000, seed=23
    )
    assert first == second
    assert first["estimate_control_minus_inhibited"] > 0
    assert first["ci95"][0] > 0
