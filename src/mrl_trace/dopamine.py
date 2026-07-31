r"""Validated descriptive analysis of the DANDI 000351 dopamine recordings.

The reduced NWB event stream contains two ``Sound 1`` markers before each reward.
They are paired cue-onset/cue-offset markers from one rewarded trial; they are not
separate rewarded and omitted trials.  This module therefore exposes reward-aligned
descriptive epochs only.  It deliberately contains no reward/omission decoder,
learning-reward pools, or dopamine-gated reinforcement-learning experiment.

The Jeong et al. recording establishes the provenance of the measured photometry.
The cue-pair interpretation and the reward-aligned processing below are repository
adaptations, not procedures attributed to that paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "load_session",
    "pair_cue_events",
    "reward_aligned_epochs",
    "reject_legacy_capstone",
    "METHOD_PROVENANCE",
]

METHOD_PROVENANCE = {
    "status": "adapted",
    "established_basis": ["reward-aligned fibre-photometry averaging"],
    "repository_adaptation": (
        "Two Sound 1 markers are validated as onset/offset markers for each rewarded "
        "trial and epochs are aligned to the recorded solenoid reward."
    ),
    "claim_limit": (
        "The available event stream contains no verified omission class and cannot "
        "support reward-versus-omission decoding or dopamine-gated learning."
    ),
}


def load_session(path: str | Path) -> dict:
    """Load one analysis cache and validate its required arrays."""
    value = np.load(Path(path), allow_pickle=True).item()
    required = {"dff", "dff_t", "sound_t", "reward_t"}
    missing = sorted(required.difference(value))
    if missing:
        raise ValueError(f"dopamine cache is missing required fields: {missing}")
    return value


def pair_cue_events(sound_t, reward_t) -> np.ndarray:
    """Return one ``(cue_onset, cue_offset, reward)`` row per rewarded trial.

    Events are assigned to the interval after the previous reward and through the
    current reward.  Every interval must contain exactly two sound markers.  The
    function fails closed on any other event structure and never converts an
    unmatched sound marker into an omission.
    """
    sound = np.sort(np.asarray(sound_t, dtype=float))
    reward = np.sort(np.asarray(reward_t, dtype=float))
    if sound.ndim != 1 or reward.ndim != 1 or not len(reward):
        raise ValueError("sound_t and reward_t must be non-empty one-dimensional arrays")
    if not np.isfinite(sound).all() or not np.isfinite(reward).all():
        raise ValueError("event timestamps must be finite")
    if len(sound) != 2 * len(reward):
        raise ValueError(
            f"expected exactly two sound markers per reward; found {len(sound)} "
            f"sounds and {len(reward)} rewards"
        )

    rows = []
    previous_reward = -np.inf
    consumed = np.zeros(len(sound), dtype=bool)
    for reward_time in reward:
        indices = np.flatnonzero(
            (~consumed) & (sound > previous_reward) & (sound < reward_time)
        )
        if len(indices) != 2:
            raise ValueError(
                "each inter-reward interval must contain exactly two cue markers; "
                f"reward at {reward_time:g}s has {len(indices)}"
            )
        onset, offset = sound[indices]
        if not onset < offset < reward_time:
            raise ValueError("cue markers must satisfy onset < offset < reward")
        consumed[indices] = True
        rows.append((float(onset), float(offset), float(reward_time)))
        previous_reward = reward_time
    if not consumed.all():
        raise ValueError("unpaired sound markers remain after reward matching")
    return np.asarray(rows, dtype=float)


def reward_aligned_epochs(session: dict, *, tmin: float = -2.0,
                          tmax: float = 4.0, baseline=(-1.5, -0.5)):
    """Return baseline-corrected epochs aligned to recorded rewards.

    Returns ``(epochs, relative_time, trial_events)``.  Trials whose complete epoch
    lies outside the photometry record are excluded after event validation.
    """
    dff = np.asarray(session["dff"], dtype=float)
    dff_t = np.asarray(session["dff_t"], dtype=float)
    if dff.ndim != 1 or dff_t.ndim != 1 or dff.shape != dff_t.shape:
        raise ValueError("dff and dff_t must be matching one-dimensional arrays")
    if len(dff_t) < 2 or np.any(np.diff(dff_t) <= 0):
        raise ValueError("dff_t must be strictly increasing")

    events = pair_cue_events(session["sound_t"], session["reward_t"])
    dt = float(np.median(np.diff(dff_t)))
    relative = np.arange(tmin, tmax + 0.5 * dt, dt)
    keep_epochs, keep_events = [], []
    baseline_mask = (relative >= baseline[0]) & (relative <= baseline[1])
    if not baseline_mask.any():
        raise ValueError("baseline interval does not overlap the requested epoch")
    for event in events:
        target = event[2] + relative
        if target[0] < dff_t[0] or target[-1] > dff_t[-1]:
            continue
        epoch = np.interp(target, dff_t, dff)
        epoch = epoch - float(epoch[baseline_mask].mean())
        keep_epochs.append(epoch)
        keep_events.append(event)
    if not keep_epochs:
        raise ValueError("no reward has a complete photometry epoch")
    return np.asarray(keep_epochs), relative, np.asarray(keep_events)


def reject_legacy_capstone(path: str | Path) -> None:
    """Reject an invalid historical dopamine-capstone result explicitly."""
    raise ValueError(
        f"{Path(path).name} is invalid: cue onset/offset markers were previously "
        "misclassified as rewarded and omitted trials. Recompute descriptive "
        "reward-aligned photometry with reward_aligned_epochs()."
    )


def main(argv=None):
    """Fail closed instead of reproducing the withdrawn dopamine capstone."""
    raise SystemExit(
        "The dopamine learning capstone has been withdrawn because DANDI 000351's "
        "reduced event stream contains cue onset/offset markers, not omissions."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
