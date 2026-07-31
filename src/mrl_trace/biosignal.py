"""Descriptive feedback-locked EEG helpers.

These functions load, epoch, and decode the OpenNeuro ds003474 probabilistic-
selection EEG. Recorded EEG is deliberately not resampled into any learning
loop: trial-wise biological validation now uses the action-contingent DANDI
001340 logged-replay API in :mod:`mrl_trace.dopamine_replay`.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import scipy.io as sio

__all__ = [
    "load_subject",
    "epoch_feedback",
    "rewp_features",
    "decode_reward",
    "FRONTOCENTRAL",
    "EEG_METHOD_PROVENANCE",
]

EEG_METHOD_PROVENANCE = {
    "status": "descriptive_only",
    "established_basis": [
        "feedback-locked EEG analysis",
        "cross-validated logistic decoding",
    ],
    "repository_adaptation": (
        "External EEG may be loaded, epoched, and decoded descriptively; decoder "
        "outputs never enter a reinforcement-learning update."
    ),
    "claim_limit": (
        "No online brain-in-the-loop learning, action-contingent replay, or "
        "biological-learning claim follows from these helpers."
    ),
}

FRONTOCENTRAL = ("FZ", "FC1", "FCZ", "FC2", "CZ", "C1", "C2")


def load_subject(eeg_set, eeg_fdt, events_tsv):
    """Load one subject and its Correct/Incorrect feedback events."""
    m = sio.loadmat(eeg_set, squeeze_me=True, struct_as_record=False)["EEG"]
    srate = float(m.srate)
    nbchan, pnts = int(m.nbchan), int(m.pnts)
    raw = np.fromfile(eeg_fdt, dtype=np.float32)
    if raw.size != nbchan * pnts:
        if raw.size % pnts:
            raise ValueError(
                f"{os.path.basename(eeg_fdt)}: .fdt size {raw.size} "
                f"not divisible by pnts {pnts}; channel layout unclear"
            )
        nbchan = raw.size // pnts
    data = raw.reshape((pnts, nbchan)).T
    labels = [str(c.labels).upper() for c in m.chanlocs]
    labels = (labels + [f"X{i}" for i in range(nbchan)])[:nbchan]

    onsets, valence = [], []
    with open(events_tsv, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            trial_type = row.get("trial_type", "")
            if trial_type == "Feedback: Correct":
                onsets.append(float(row["onset"]))
                valence.append(1)
            elif trial_type == "Feedback: Incorrect":
                onsets.append(float(row["onset"]))
                valence.append(0)
    return (
        data,
        srate,
        labels,
        np.asarray(onsets, dtype=float),
        np.asarray(valence, dtype=np.int8),
    )


def epoch_feedback(
    data,
    srate,
    labels,
    onsets,
    *,
    t0=-0.1,
    t1=0.6,
    channels=FRONTOCENTRAL,
    pnts=None,
):
    """Return baseline-corrected feedback epochs, times, and a kept-trial mask."""
    data = np.asarray(data)
    if pnts is None:
        pnts = data.shape[1]
    channel_indices = [i for i, label in enumerate(labels) if label in channels]
    start_offset, stop_offset = int(t0 * srate), int(t1 * srate)
    baseline_samples = int(-t0 * srate)
    epochs, kept = [], []
    for onset in np.asarray(onsets, dtype=float):
        start = int(onset * srate) + start_offset
        stop = int(onset * srate) + stop_offset
        valid = 0 <= start and stop < pnts
        kept.append(valid)
        if valid:
            segment = data[channel_indices, start:stop].copy()
            segment -= segment[:, :baseline_samples].mean(axis=1, keepdims=True)
            epochs.append(segment)
    times = np.arange(start_offset, stop_offset) / srate
    return np.asarray(epochs), times, np.asarray(kept, dtype=bool)


def rewp_features(epochs, times):
    """Mean frontocentral amplitudes in predeclared RewP/FRN time bins."""
    epochs, times = np.asarray(epochs), np.asarray(times)
    bins = ((0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.20, 0.40))
    return np.concatenate(
        [epochs[:, :, (times >= lo) & (times < hi)].mean(axis=2) for lo, hi in bins],
        axis=1,
    )


def decode_reward(features, valence, *, n_splits=5, seed=0, C=0.1):
    """Cross-validated descriptive feedback-valence decoder."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    features, valence = np.asarray(features), np.asarray(valence)
    proba = np.zeros(len(valence))
    prediction = np.zeros(len(valence), dtype=int)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train, test in folds.split(features, valence):
        scaler = StandardScaler().fit(features[train])
        classifier = LogisticRegression(
            C=C, class_weight="balanced", max_iter=1000
        ).fit(scaler.transform(features[train]), valence[train])
        transformed = scaler.transform(features[test])
        proba[test] = classifier.predict_proba(transformed)[:, 1]
        prediction[test] = classifier.predict(transformed)
    return {
        "acc": float(np.mean(prediction == valence)),
        "bal_acc": float(balanced_accuracy_score(valence, prediction)),
        "auc": (
            float(roc_auc_score(valence, proba))
            if len(np.unique(valence)) > 1
            else float("nan")
        ),
        "proba": proba,
        "pred": prediction,
        "valence": valence,
        "method_provenance": EEG_METHOD_PROVENANCE,
    }
