"""Real-EEG reward signal for the biosignal reinforcement-learning experiment.

Loads the OpenNeuro ds003474 probabilistic-selection EEG (Cavanagh; CC0), epochs the
signal around feedback events, and trains a single-trial decoder of the feedback
valence (Correct vs Incorrect) from the frontocentral reward-positivity / feedback-
related negativity. The decoder's out-of-fold per-trial output is used as a recorded
EEG-derived reward gate (R - b) of the three-factor rule, in place of a synthetic reward.

No MNE dependency: EEGLAB ``.set`` is read with scipy.io.loadmat and the ``.fdt`` as a
float32 (pnts x nbchan) array; feedback events come from the BIDS ``events.tsv`` (clean
"Feedback: Correct"/"Feedback: Incorrect" labels with onsets), not the .set event
struct (which carries only numeric trigger codes).

Honesty: this supplies the THIRD FACTOR (a real, noisy reward-prediction-error signal)
to the device-trace bandit; it is a robustness/biorealism test, not a new device
capability, and it is an offline recorded-signal study, not a live brain in the loop.
"""
from __future__ import annotations

import csv
import os

import numpy as np
import scipy.io as sio

__all__ = ["load_subject", "epoch_feedback", "rewp_features", "decode_reward",
           "FRONTOCENTRAL", "build_reward_pools", "run_biosignal_reward",
           "run_eeg_capstone", "EEG_DATA_DEFAULT", "EEG_POOLS_CACHE",
           "EEG_METHOD_PROVENANCE"]

EEG_METHOD_PROVENANCE = {
    "status": "adapted",
    "established_basis": [
        "feedback-locked EEG analysis",
        "cross-validated logistic decoding",
        "three-factor reward modulation",
    ],
    "repository_adaptation": (
        "Out-of-fold feedback-valence predictions from an external EEG dataset are "
        "resampled as the scalar gate in simulated learning."
    ),
    "claim_limit": (
        "This is not online brain-in-the-loop learning, per-trial replay of recorded "
        "behaviour, or evidence that the simulated agent is a biological learner."
    ),
}

#: Frontocentral channels carrying the reward-positivity / FRN (uppercased labels).
FRONTOCENTRAL = ("FZ", "FC1", "FCZ", "FC2", "CZ", "C1", "C2")


def load_subject(eeg_set, eeg_fdt, events_tsv):
    """Load one subject: returns (data[nbchan, pnts], srate, labels, fb_onsets,
    fb_labels) where fb_labels is 1 for Correct (reward) and 0 for Incorrect."""
    m = sio.loadmat(eeg_set, squeeze_me=True, struct_as_record=False)["EEG"]
    srate = float(m.srate)
    nbchan, pnts = int(m.nbchan), int(m.pnts)
    raw = np.fromfile(eeg_fdt, dtype=np.float32)
    # The .set header nbchan/pnts can disagree with the .fdt (some ds003474 subjects
    # have interpolated/extra channels; README flags this). Reconcile against the file
    # size: trust pnts, recover the true channel count, and fail loudly if it does not
    # divide evenly so such subjects are skipped rather than silently mis-shaped.
    if raw.size != nbchan * pnts:
        if raw.size % pnts == 0:
            nbchan = raw.size // pnts
        else:
            raise ValueError(f"{os.path.basename(eeg_fdt)}: .fdt size {raw.size} "
                             f"not divisible by pnts {pnts}; channel layout unclear")
    data = raw.reshape((pnts, nbchan)).T
    labels = [str(c.labels).upper() for c in m.chanlocs]
    # if labels list is shorter/longer than recovered channels, pad/trim defensively so
    # frontocentral selection (by name) still works on the channels that ARE labelled
    if len(labels) < nbchan:
        labels = labels + [f"X{i}" for i in range(nbchan - len(labels))]
    elif len(labels) > nbchan:
        labels = labels[:nbchan]
    onsets, valence = [], []
    with open(events_tsv) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            tt = row.get("trial_type", "")
            if tt == "Feedback: Correct":
                onsets.append(float(row["onset"])); valence.append(1)
            elif tt == "Feedback: Incorrect":
                onsets.append(float(row["onset"])); valence.append(0)
    return data, srate, labels, np.array(onsets), np.array(valence)


def epoch_feedback(data, srate, labels, onsets, *, t0=-0.1, t1=0.6,
                   channels=FRONTOCENTRAL, pnts=None):
    """Feedback-locked, baseline-corrected frontocentral epochs.

    Returns ``epochs[n_trials, n_chan, n_time]`` (baseline -100..0 ms removed) and the
    time vector. Trials whose window falls outside the recording are dropped (and the
    caller must drop the matching labels via the returned ``keep`` mask)."""
    if pnts is None:
        pnts = data.shape[1]
    ch = [i for i, l in enumerate(labels) if l in channels]
    a, b = int(t0 * srate), int(t1 * srate)
    base = int(-t0 * srate)
    segs, keep = [], []
    for o in onsets:
        s, e = int(o * srate) + a, int(o * srate) + b
        if 0 <= s and e < pnts:
            seg = data[ch, s:e]
            seg = seg - seg[:, :base].mean(1, keepdims=True)
            segs.append(seg); keep.append(True)
        else:
            keep.append(False)
    t = np.arange(a, b) / srate
    return np.array(segs), t, np.array(keep, dtype=bool)


def rewp_features(epochs, t):
    """Per-trial features for valence decoding: mean amplitude in successive 50 ms bins
    over 150-500 ms post-feedback (the RewP/FRN window), per frontocentral channel,
    flattened. Robust, interpretable, and avoids overfitting the full time course."""
    bins = [(0.15, 0.25), (0.25, 0.35), (0.35, 0.45), (0.20, 0.40)]
    feats = []
    for lo, hi in bins:
        w = (t >= lo) & (t < hi)
        feats.append(epochs[:, :, w].mean(2))      # (n_trials, n_chan)
    return np.concatenate(feats, axis=1)            # (n_trials, n_chan*len(bins))


def decode_reward(features, valence, *, n_splits=5, seed=0, C=0.1):
    """Cross-validated single-trial valence decoder (logistic regression, standardised).

    Returns ``dict(acc, bal_acc, proba, pred, auc)`` where ``proba``/``pred`` are
    OUT-OF-FOLD per-trial predictions (so the reward signal handed to the bandit is never
    trained on its own trial). Honest single-trial ErrP/RewP decoding is ~0.6-0.8
    balanced accuracy; whatever it is, it is reported as-is.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    X, y = np.asarray(features), np.asarray(valence)
    proba = np.zeros(len(y)); pred = np.zeros(len(y), dtype=int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=C, class_weight="balanced", max_iter=1000)
        clf.fit(sc.transform(X[tr]), y[tr])
        proba[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
        pred[te] = clf.predict(sc.transform(X[te]))
    return {
        "acc": float((pred == y).mean()),
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float("nan"),
        "proba": proba, "pred": pred, "valence": y,
        "method_provenance": EEG_METHOD_PROVENANCE,
    }


# =============================================================================
# Experiment cores (the biosignal-reward RL studies that compose ``bandit.train``
# and ``deep.train_deep``)
#
# Each ``run_*`` returns the result grid as a plain dict -- no file I/O, no
# plotting, no stdout.  Notebooks call these at a small (quick) seed/trial count
# and render the figures inline; ``main()`` (below) calls them at the published
# 20-seed scale and writes the grid under ``data/results/`` for the notebooks to
# render.
#
# The raw OpenNeuro ds003474 EEG is NOT bundled: it is a several-GB external
# download (Cavanagh; CC0), so these cores DEGRADE GRACEFULLY -- when the EEG is
# absent :func:`build_reward_pools` finds no usable subject, returns empty pools,
# and each core returns ``{"skipped": True, ...}`` with a printed note (the
# original drivers behaved the same way).  Provenance for the reported numbers:
#   - exp7 biosignal   : experiments/05_biological_grounding/biosignal_reward.py
#   - exp9 EEG capstone: experiments/05_biological_grounding/eeg_capstone.py
# =============================================================================

#: Default location of the (unbundled) OpenNeuro ds003474 EEG download, overridable
#: via ``SIOX_EEG_DIR`` or the ``--data`` CLI flag.
EEG_DATA_DEFAULT = os.environ.get("SIOX_EEG_DIR", "/tmp/ds003474")

#: Optional cache of the decoded reward pools (the capstone driver's ``POOLS_CACHE``).
EEG_POOLS_CACHE = os.environ.get("SIOX_EEG_POOLS_CACHE", "/tmp/eeg_pools.npy")


def build_reward_pools(data_dir=EEG_DATA_DEFAULT, *, seed=0, min_trials=40,
                       cache=None):
    """Decode per-trial reward from every usable ds003474 subject and pool the
    out-of-fold predictions by TRUE valence.

    Returns ``(pools, meta)`` where ``pools = {1: array, 0: array}`` are the decoded
    reward VALUES (the decoder's 0/1 prediction) on true-correct and true-incorrect
    trials respectively -- the exact ``reward_pools`` interface ``bandit.train`` /
    ``deep.train_deep`` accept -- and ``meta`` records the decode accuracy actually
    achieved (the realistic reward-noise level).  Subjects with a broken channel
    layout, a single valence class, or fewer than ``min_trials`` feedback events are
    skipped (not silently mis-shaped).  Identical to the two drivers' local
    ``build_reward_pools``.

    Graceful degradation: if ``data_dir`` holds no usable subject (e.g. the several-GB
    EEG is not downloaded here), both pools come back EMPTY and ``meta == {}``; callers
    treat that as "skip this experiment" rather than erroring.

    ``cache`` (default ``None``) may be a path to a ``.npy`` reward-pool cache; if it
    exists it is loaded verbatim (the capstone driver's ``POOLS_CACHE`` behaviour),
    otherwise a freshly decoded pool is written to it.
    """
    if cache and os.path.exists(cache):
        d = np.load(cache, allow_pickle=True).item()
        meta = d.get("_meta", {})
        pools = {1: np.asarray(d[1], float), 0: np.asarray(d[0], float)}
        return pools, meta
    import glob
    subs = sorted(glob.glob(os.path.join(data_dir, "sub-*/eeg/"
                  "sub-*_task-ProbabilisticSelection_eeg.fdt")))
    pool1, pool0, baccs = [], [], []
    used = 0
    for fdt in subs:
        pre = fdt[:-len("_eeg.fdt")]
        try:
            data, srate, labels, onsets, val = load_subject(
                pre + "_eeg.set", pre + "_eeg.fdt", pre + "_events.tsv")
        except ValueError:
            continue                                   # skip channel-layout-broken subj
        ep, t, keep = epoch_feedback(data, srate, labels, onsets, pnts=data.shape[1])
        y = val[keep]
        if len(np.unique(y)) < 2 or len(y) < min_trials:
            continue
        r = decode_reward(rewp_features(ep, t), y, seed=seed)
        baccs.append(r["bal_acc"]); used += 1
        # out-of-fold predicted reward, split by the TRUE outcome valence
        pool1.extend(r["pred"][r["valence"] == 1].tolist())   # decoded R | true=correct
        pool0.extend(r["pred"][r["valence"] == 0].tolist())   # decoded R | true=incorrect
    pools = {1: np.array(pool1, dtype=float), 0: np.array(pool0, dtype=float)}
    if used == 0:
        return pools, {}                               # graceful: nothing usable here
    meta = {"n_subj": used, "mean_bal_acc": float(np.mean(baccs)),
            "sd_bal_acc": float(np.std(baccs)),
            "P_R1_given_correct": float(pools[1].mean()),
            "P_R1_given_incorrect": float(pools[0].mean())}
    if cache:
        np.save(cache, {1: pools[1], 0: pools[0], "_meta": meta}, allow_pickle=True)
    return pools, meta


def _running(r1d, window=50):
    """Running-mean reward curve (window trials), matching the exp7 driver."""
    c = np.cumsum(np.insert(np.asarray(r1d, float), 0, 0.0))
    return (c[window:] - c[:-window]) / window


def run_biosignal_reward(pools, meta, *, seeds=20, trials=600, S=2, A=2,
                         tau_leak=10.0, D=2.0, seed0=0):
    """Experiment 7 -- close the RL loop with a REAL biosignal reward (Arm: biorealism).

    The contextual spiking bandit's synthetic reward ``R in {0,1}`` is replaced by a
    per-trial reward-prediction-error DECODED FROM REAL EEG (ds003474 reward positivity /
    FRN).  The device supplies the eligibility trace; biology supplies the third factor.
    Reported reward rate is TASK PERFORMANCE (true outcome); the EEG-decoded value only
    gates the ``dw = eta (R - b) e`` update.

    Conditions (device gate, same task/seeds/budget):
      synthetic : clean ``R in {0,1}`` (reference);
      biosignal : per-trial reward sampled from the EEG decoder's out-of-fold predictions,
                  keyed by the true outcome valence (realistic decode noise);
      shuffled  : the SAME decoded values with valence permuted (a noise-matched control:
                  same reward statistics, no genuine outcome information).

    Retrospectively recorded criteria B1-B4 (BIOSIGNAL_RL_notes.md):
      B1 decoder beats chance (bal-acc > 0.55);
      B2 biosignal reaches criterion (>= 0.5*(1+1/A));
      B3 biosignal beats shuffled by > 0.10;
      B4 biosignal graceful vs synthetic (>= synthetic - 0.25).

    ``pools``/``meta`` come from :func:`build_reward_pools`.  If ``pools`` is empty (EEG
    absent) the experiment is skipped and ``{"skipped": True}`` is returned.  Serial: no
    multiprocessing, no file I/O -- notebook-callable.
    """
    from .bandit import train, reward_rate
    from .stats import bootstrap_ci
    if len(pools.get(1, [])) == 0 or len(pools.get(0, [])) == 0:
        return {"skipped": True, "reason": "no usable ds003474 EEG subjects found",
                "method_provenance": EEG_METHOD_PROVENANCE}
    chance = 1.0 / A
    crit = 0.5 * (1 + chance)

    # shuffled control: identical pooled reward values, valence association destroyed
    allvals = np.concatenate([pools[1], pools[0]])
    rng = np.random.default_rng(0)
    shuf = {1: rng.permutation(allvals), 0: rng.permutation(allvals)}

    conds = {
        "synthetic": dict(reward_pools=None),
        "biosignal": dict(reward_pools=pools),
        "shuffled":  dict(reward_pools=shuf),
    }
    finals, curves = {}, {}
    for name, kw in conds.items():
        r = train(S, A, B=seeds, tau_leak=tau_leak, D=D, trials=trials, seed0=seed0, **kw)
        finals[name] = reward_rate(r, window=100)
        curves[name] = _running(r.mean(0))

    b1 = meta.get("mean_bal_acc", float("nan")) > 0.55
    b2 = finals["biosignal"].mean() >= crit
    b3 = finals["biosignal"].mean() > finals["shuffled"].mean() + 0.1
    b4 = finals["biosignal"].mean() >= finals["synthetic"].mean() - 0.25
    return {
        "finals": finals, "curves": curves, "meta": meta,
        "chance": chance, "crit": crit, "seeds": seeds, "trials": trials,
        "S": S, "A": A, "tau_leak": tau_leak, "D": D,
        "retention_definition": "deliberately_swept",
        "method_provenance": EEG_METHOD_PROVENANCE,
        "criteria": {"B1": bool(b1), "B2": bool(b2), "B3": bool(b3), "B4": bool(b4)},
    }


# --- capstone (Experiment 9, Arm E): EEG reward x deep all-local RL + homeostasis ---
# Frozen Arm-D operating point recorded in legacy retrospective analysis notes.
_CAP_HP = dict(H=16, tau_leak=10.0, D=5.0, eta=0.2, eta_hidden=3.0,
               fb_scale=2.0, bias_o=0.3, V=1.5, sigma0=0.15, sigma1=0.05)
_CAP_HOMEO = 0.1
_CAP_RULES = [("dfa", 0.0), ("dfa_homeo", _CAP_HOMEO)]
_CAP_REW = ["synthetic", "eeg", "shuffled"]


def _capstone_worker(spec, trials, pools, shuf, seeds):
    """Train one (rule, reward-source) cell of the capstone grid; returns
    ``((rule, rname), finals, curve)``.  Module-level so it is picklable for the
    ``multiprocessing.Pool`` in :func:`main`."""
    from .deep import train_deep, reward_rate
    rule, rname = spec
    mode, homeo = ("dfa", 0.0) if rule == "dfa" else ("dfa", _CAP_HOMEO)
    rp = {"synthetic": None, "eeg": {0: pools[0], 1: pools[1]}, "shuffled": shuf}[rname]
    rew = train_deep(mode=mode, B=seeds, trials=trials, seed0=0, homeo=homeo,
                     reward_pools=rp, **_CAP_HP)
    finals = reward_rate(rew, window=200)
    win = 100
    csum = np.cumsum(rew, axis=1)
    curve = ((csum[:, win:] - csum[:, :-win]) / win).mean(0)
    return (rule, rname), finals, curve


def _summarize_capstone(res, meta, *, seeds, trials, chance, crit, homeo):
    """Package raw per-cell ``_capstone_worker`` output into the saved grid dict
    (str-keyed ``finals``/``curves``/``ci`` as the driver saved them, plus the
    retrospectively recorded E2-E4 criteria)."""
    from .stats import bootstrap_ci
    finals = {k: f for k, f, _ in res}
    curves = {k: c for k, _, c in res}
    ci = {k: bootstrap_ci(finals[k]) for k in finals}

    d_eeg, h_eeg = ci[("dfa", "eeg")], ci[("dfa_homeo", "eeg")]
    h_eeg_m = finals[("dfa_homeo", "eeg")].mean()
    h_shuf = ci[("dfa_homeo", "shuffled")]
    h_syn_m = finals[("dfa_homeo", "synthetic")].mean()
    e2 = bool(h_eeg_m >= crit and h_eeg[0] > d_eeg[1])
    e3 = bool(h_eeg[0] > h_shuf[1])
    e4 = bool(h_eeg_m >= h_syn_m - 0.25)
    return {
        "finals": {str(k): v for k, v in finals.items()},
        "curves": {str(k): v for k, v in curves.items()},
        "ci": {str(k): v for k, v in ci.items()}, "meta": meta,
        "HP": _CAP_HP, "homeo": homeo, "seeds": seeds, "trials": trials,
        "chance": chance, "crit": crit,
        "retention_definition": "deliberately_swept",
        "method_provenance": EEG_METHOD_PROVENANCE,
        "hyperparameter_provenance": "pilot_tuned_then_frozen",
        "criteria": {"E2": e2, "E3": e3, "E4": e4},
    }


def run_eeg_capstone(pools, meta, *, seeds=20, trials=3000, chance=0.5, crit=0.75):
    """Experiment 9 (Arm E) -- capstone: EEG reward x deep all-local RL + homeostasis.

    Composes the three biorealistic ingredients into one experiment:
      - temporal credit : the physical device eligibility trace (on both layers);
      - spatial credit  : a deep two-layer policy trained fully locally by direct
                          feedback alignment (no backprop, no weight transport);
      - the third factor: a REAL human reward-prediction-error decoded from EEG
                          (ds003474), the same biosignal reward as
                          :func:`run_biosignal_reward`.

    Tests a hypothesis recorded retrospectively in the legacy analysis notes: the
    local homeostatic stabiliser -- which prevents the policy collapse behind deep-DFA's
    unreliability (Arm D) -- ALSO confers robustness to a noisy biological reward, so
    DFA+homeostasis learns under the EEG reward where DFA-alone does not.

    Design: 2 rules {DFA, DFA+homeostasis} x 3 reward sources {synthetic, EEG, shuffled}
    on deep XOR at the frozen Arm-D operating point, 20 seeds, bootstrap 95% CI.  Reported
    reward rate is TRUE task performance; the EEG-decoded value only gates the update.

    Retrospectively recorded criteria:
      E2 homeostasis robustness (EEG homeo >= crit AND its CI-lo > DFA CI-hi);
      E3 genuine reward info (EEG homeo CI-lo > shuffled CI-hi, disjoint);
      E4 graceful vs clean (EEG homeo >= synthetic homeo - 0.25).

    SERIAL (notebook-callable): iterates the 6 grid cells in-process, no ``Pool``, no file
    I/O.  ``main()`` runs the same cells across a ``multiprocessing.Pool``.  If ``pools`` is
    empty (EEG absent) the experiment is skipped and ``{"skipped": True}`` is returned.
    """
    if len(pools.get(1, [])) == 0 or len(pools.get(0, [])) == 0:
        return {"skipped": True, "reason": "no usable ds003474 EEG subjects found",
                "method_provenance": EEG_METHOD_PROVENANCE}
    rng = np.random.default_rng(0)
    allv = np.concatenate([pools[1], pools[0]])
    shuf = {1: rng.permutation(allv), 0: rng.permutation(allv)}
    specs = [(rule, rname) for rule, _ in _CAP_RULES for rname in _CAP_REW]
    res = [_capstone_worker(s, trials, pools, shuf, seeds) for s in specs]
    return _summarize_capstone(res, meta, seeds=seeds, trials=trials,
                               chance=chance, crit=crit, homeo=_CAP_HOMEO)


def main(argv=None):
    """Full-scale reproduction CLI for the biosignal-reward RL grids (writes ``data/results``).

    ``python -m mrl_trace.biosignal [--biosignal] [--capstone] [--full|--quick]
    [--data DIR]``

    With no experiment flag, runs both.  ``--full`` = 20 seeds (published); ``--quick`` =
    a fast few-seed smoke run.  ``--data DIR`` points at the (unbundled) OpenNeuro ds003474
    download (default ``$SIOX_EEG_DIR`` or ``/tmp/ds003474``).  When the EEG is absent the
    reward pools come back empty and each experiment is SKIPPED with a printed note (exit
    0) -- the graceful-degradation path, since the several-GB EEG is not bundled.
    """
    import argparse
    from . import paths
    ap = argparse.ArgumentParser(description="Biosignal-reward RL reproductions")
    ap.add_argument("--biosignal", action="store_true",
                    help="exp7 device bandit under real-EEG reward -> exp7_biosignal.npy")
    ap.add_argument("--capstone", action="store_true",
                    help="exp9 EEG x deep DFA + homeostasis -> exp9_capstone.npy")
    ap.add_argument("--quick", action="store_true", help="fast few-seed smoke run")
    ap.add_argument("--full", action="store_true", help="published 20-seed run (default)")
    ap.add_argument("--data", default=EEG_DATA_DEFAULT,
                    help="OpenNeuro ds003474 EEG directory (default $SIOX_EEG_DIR or /tmp/ds003474)")
    a = ap.parse_args(argv)
    run_all = not (a.biosignal or a.capstone)

    print("=== decoding real-EEG reward pools from ds003474 ===")
    pools, meta = build_reward_pools(a.data)
    if not meta:
        print(f"  no usable ds003474 EEG subjects under {a.data!r} -- "
              f"biosignal experiments SKIPPED (EEG is a several-GB external download; "
              f"set --data / $SIOX_EEG_DIR to reproduce).")
        return
    print(f"  {meta['n_subj']} subjects | decoder bal-acc {meta['mean_bal_acc']:.3f} "
          f"+/- {meta.get('sd_bal_acc', float('nan')):.3f}")
    print(f"  P(R=1 | correct)={pools[1].mean():.3f}  "
          f"P(R=1 | incorrect)={pools[0].mean():.3f}  (clean would be 1.00 / 0.00)")

    if a.biosignal or run_all:
        seeds = 6 if a.quick else 20
        trials = 400 if a.quick else 600
        print(f"\n=== exp7 biosignal-reward bandit ({seeds} seeds, {trials} trials) ===")
        grid = run_biosignal_reward(pools, meta, seeds=seeds, trials=trials)
        if grid.get("skipped"):
            print(f"  skipped: {grid['reason']}")
        else:
            f = grid["finals"]
            from .stats import bootstrap_ci
            for name in ("synthetic", "biosignal", "shuffled"):
                lo, hi = bootstrap_ci(f[name])
                print(f"  {name:10s} reward rate = {f[name].mean():.3f} [{lo:.3f}, {hi:.3f}]")
            paths.save_result("exp7_biosignal.npy", grid)
            print(f"  wrote exp7_biosignal.npy  criteria={grid['criteria']}")

    if a.capstone or run_all:
        import os as _os
        from functools import partial
        from multiprocessing import Pool
        seeds = 6 if a.quick else 20
        trials = 600 if a.quick else 3000
        chance, crit = 0.5, 0.75
        print(f"\n=== exp9 capstone: EEG x deep all-local + homeostasis "
              f"({seeds} seeds, {trials} trials) ===")
        rng = np.random.default_rng(0)
        allv = np.concatenate([pools[1], pools[0]])
        shuf = {1: rng.permutation(allv), 0: rng.permutation(allv)}
        specs = [(rule, rname) for rule, _ in _CAP_RULES for rname in _CAP_REW]
        # Pool over the 6 (rule x reward-source) cells -- this is a real __main__.
        with Pool(processes=min(len(specs), (_os.cpu_count() or 4) - 1)) as pool:
            res = pool.map(partial(_capstone_worker, trials=trials, pools=pools,
                                   shuf=shuf, seeds=seeds), specs)
        grid = _summarize_capstone(res, meta, seeds=seeds, trials=trials,
                                   chance=chance, crit=crit, homeo=_CAP_HOMEO)
        fin = {eval(k): v for k, v in grid["finals"].items()}
        cis = {eval(k): v for k, v in grid["ci"].items()}
        print(f"  {'rule':18s} {'reward':10s} {'mean':>6} {'95% CI':>16}")
        for rule, _ in _CAP_RULES:
            for rname in _CAP_REW:
                k = (rule, rname); lo, hi = cis[k]
                print(f"  {rule:18s} {rname:10s} {fin[k].mean():6.3f} [{lo:5.3f},{hi:5.3f}]")
        paths.save_result("exp9_capstone.npy", grid)
        print(f"  wrote exp9_capstone.npy  criteria={grid['criteria']}")


if __name__ == "__main__":
    main()
