r"""Real measured-dopamine reward signal for the biosignal reinforcement-learning
experiment -- the direct-neuromodulator analogue of :mod:`biosignal` (EEG).

Where :mod:`biosignal` decodes a reward-prediction error from the human EEG reward
positivity (a *cortical correlate* of dopaminergic RPE), this module decodes it from a
**directly recorded dopamine transient**: NAcc dLight1.3b fiber photometry during
Pavlovian conditioning (Jeong et al. 2022, Science; DANDI 000351). Dopamine is the literal
third factor the three-factor theory names, so this is a strictly stronger biological
grounding of the reward term.

Pipeline, mirroring :mod:`biosignal` exactly so the learning code is unchanged:
  1. ``load_session`` reads a cached session (dF/F trace @100 Hz + cue/reward event times).
  2. ``epoch_outcome`` epochs the dF/F around the expected-outcome time of each cue and
     labels each cue reward (CS+, a solenoid followed within the window) or omission (CS-).
  3. ``rewp_features`` -> ``decode_reward`` train a cross-validated single-trial decoder of
     reward vs omission; its OUT-OF-FOLD per-trial predictions are the reward gate (R-b).

Honesty: this supplies the THIRD FACTOR (a real, noisy dopaminergic reward-prediction
error). It uses an offline recorded animal dopamine signal -- the right kind of signal
(the actual neuromodulator), cross-species, not a live brain in the loop. It is a
biorealism/robustness test, not a new device capability. Whatever the single-trial decode
accuracy is, it is reported as-is.

The cache files under ``DA_CACHE`` are produced by the extractor that streams the dF/F
trace and eventlog out of the (large) NWB files via HTTP range reads; only the small
arrays are kept. See ``experiments/`` for the extractor and provenance.
"""
from __future__ import annotations

import glob
import os

import numpy as np

__all__ = ["load_session", "epoch_outcome", "rewp_features", "decode_reward",
           "build_reward_pools", "DA_CACHE",
           "make_shuffled_pools", "run_dopamine_shallow", "run_dopamine_deep",
           "HP", "HOMEO", "REW"]

#: Default cache directory of extracted per-session dopamine arrays.
DA_CACHE = os.environ.get("DA_CACHE", "/tmp/da_cache")

#: Reward window after a cue: a solenoid in (lo, hi) s marks the cue as rewarded (CS+).
REWARD_WIN = (0.2, 5.0)


def load_session(path):
    """Load one cached session.

    Returns ``dict(dff, dff_t, sound_t, reward_t, fs, sub)`` where ``dff`` is the
    analysis-ready dF/F dopamine trace, ``dff_t`` its timestamps (s), ``sound_t`` the cue
    onset times (s) and ``reward_t`` the solenoid (reward) times (s)."""
    return np.load(path, allow_pickle=True).item()


def _label_cues(sound_t, reward_t, win=REWARD_WIN):
    """Pair each reward to its nearest preceding cue within ``win``; label cues CS+/CS-.

    Returns (rewarded[bool, n_cue], delay) where ``delay`` is the median cue->reward lag
    over CS+ trials (the expected-outcome lag used to centre the omission epochs)."""
    lo, hi = win
    rewarded = np.zeros(len(sound_t), dtype=bool)
    delays = []
    for r in reward_t:
        cand = np.where((sound_t < r - lo) & (sound_t > r - hi))[0]
        if len(cand):
            j = cand[-1]
            if not rewarded[j]:
                rewarded[j] = True
                delays.append(r - sound_t[j])
    delay = float(np.median(delays)) if delays else 0.5 * (lo + hi)
    return rewarded, delay


def epoch_outcome(sess, *, pre=1.0, post=3.0):
    """Outcome-locked, baseline-corrected dF/F epochs for one session.

    Each cue is epoched over ``[-pre, +post]`` s around its expected-outcome time
    (cue + median CS+ delay), baseline-subtracted on ``[-pre, 0]``. Returns
    ``(epochs[n_cue, n_time], t, valence[n_cue])`` with valence 1 = reward (CS+),
    0 = omission (CS-). Cues whose window falls outside the recording are dropped."""
    dff = np.asarray(sess["dff"], float)
    ts = np.asarray(sess["dff_t"], float)
    fs = float(sess["fs"])
    rewarded, delay = _label_cues(sess["sound_t"], sess["reward_t"])
    npre, npost = int(pre * fs), int(post * fs)
    n = npre + npost
    t = np.arange(-npre, npost) / fs
    segs, val = [], []
    for s, rw in zip(sess["sound_t"], rewarded):
        c = s + delay
        i0 = int(np.searchsorted(ts, c)) - npre
        i1 = i0 + n
        if i0 < 0 or i1 > len(dff):
            continue
        seg = dff[i0:i1] - dff[i0:i0 + npre].mean()
        segs.append(seg)
        val.append(int(rw))
    return np.array(segs), t, np.array(val, dtype=int)


def rewp_features(epochs, t):
    """Per-trial features for reward/omission decoding: mean dF/F in successive bins over
    the post-outcome dopamine-response window (0-2 s), parallel to the EEG RewP features.

    Coarse, interpretable bins (avoids overfitting the full time course); the dopamine
    reward response is a transient peak in the first ~1.5 s after a rewarded outcome."""
    bins = [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (0.0, 1.5)]
    feats = []
    for lo, hi in bins:
        w = (t >= lo) & (t < hi)
        feats.append(epochs[:, w].mean(1, keepdims=True))
    return np.concatenate(feats, axis=1)            # (n_trials, len(bins))


def decode_reward(features, valence, *, n_splits=5, seed=0, C=0.1):
    """Cross-validated single-trial reward-vs-omission decoder (logistic regression).

    Returns ``dict(acc, bal_acc, auc, proba, pred, valence)`` with ``proba``/``pred`` the
    OUT-OF-FOLD per-trial predictions, so the reward signal handed to the agent is never
    trained on its own trial. Identical contract to :func:`biosignal.decode_reward`."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score

    X, y = np.asarray(features, float), np.asarray(valence, int)
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
    }


def build_reward_pools(cache_dir=DA_CACHE, *, seed=0, min_trials=40):
    """Decode every usable cached session and pool OUT-OF-FOLD predictions by TRUE valence.

    Returns ``(pools, meta)`` where ``pools = {1: array, 0: array}`` are the decoded reward
    VALUES on true-reward and true-omission trials respectively -- the exact
    ``reward_pools`` interface ``bandit.train`` / ``deep.train_deep`` accept. ``meta``
    carries the subject count and decode statistics. Sessions with one valence class or
    too few trials are skipped (not silently mis-shaped)."""
    files = sorted(glob.glob(os.path.join(cache_dir, "*.npy")))
    pool1, pool0, baccs, aucs, subs = [], [], [], [], []
    for fn in files:
        sess = load_session(fn)
        ep, t, val = epoch_outcome(sess)
        if len(np.unique(val)) < 2 or len(val) < min_trials:
            continue
        r = decode_reward(rewp_features(ep, t), val, seed=seed)
        baccs.append(r["bal_acc"]); aucs.append(r["auc"]); subs.append(sess["sub"])
        pool1.extend(r["pred"][r["valence"] == 1].tolist())
        pool0.extend(r["pred"][r["valence"] == 0].tolist())
    pools = {1: np.array(pool1, float), 0: np.array(pool0, float)}
    meta = {"n_subj": len(baccs), "subjects": subs,
            "mean_bal_acc": float(np.mean(baccs)) if baccs else float("nan"),
            "sd_bal_acc": float(np.std(baccs)) if baccs else float("nan"),
            "mean_auc": float(np.mean(aucs)) if aucs else float("nan"),
            "P_R1_given_reward": float(pools[1].mean()) if len(pool1) else float("nan"),
            "P_R1_given_omission": float(pools[0].mean()) if len(pool0) else float("nan")}
    return pools, meta


# =============================================================================
# Experiment 11 / Arm G -- the capstone with a REAL MEASURED DOPAMINE reward
#
# The direct-neuromodulator analogue of the EEG capstone (Arm E / exp9). The third
# factor that gates the device eligibility trace is a per-trial reward-prediction
# error decoded (above) from a *directly recorded dopamine transient* -- NAcc
# dLight1.3b fiber photometry during Pavlovian conditioning (Jeong et al. 2022,
# Science; DANDI 000351) -- rather than the human EEG reward-positivity *correlate*
# of that signal. Dopamine is the literal third factor the three-factor theory names,
# so this is the strongest biological grounding of the reward term in the paper.
#
# Two arms, both at frozen operating points (nothing retuned for the dopamine reward):
#   G-shallow : single-layer contextual bandit (mirrors exp7_biosignal), reward
#               sources {synthetic, dopamine, shuffled}.
#   G-deep    : deep XOR, rules {DFA, DFA+homeostasis} x reward {synthetic, dopamine,
#               shuffled} (mirrors exp9 capstone, frozen Arm-D operating point).
#
# Reported reward rate = TRUE task performance; the decoded dopamine value only gates
# the update. Retrospective criteria G1-G4 / KILL live in
# Retrospective dopamine protocol file. Negative results reported as-is; no goalpost
# moving.
#
# The two ``run_*`` cores below are SERIAL and return plain result dicts (no file I/O,
# no plotting, no stdout) so a notebook can call them at a small seed/trial count.
# ``main()`` runs the published 20-seed scale, parallelises the deep grid over specs,
# computes the criteria and writes the grid via ``paths.save_result``.
# =============================================================================

#: Frozen deep operating point -- identical to the exp9 capstone (Arm D/E). Nothing retuned.
HP = dict(H=32, tau_leak=10.0, D=5.0, eta=0.2, eta_hidden=3.0,
          fb_scale=2.0, bias_o=0.3, V=1.5, sigma0=0.15, sigma1=0.05)
#: Homeostasis strength for the ``dfa_homeo`` rule (moderate; the Arm-D value).
HOMEO = 0.1
#: Reward sources, in reporting order.
REW = ["synthetic", "dopamine", "shuffled"]

_CHANCE = 0.5
_CRIT = 0.75


def make_shuffled_pools(pools, *, seed=0):
    """Shuffled-reward control: destroy the valence->reward mapping while preserving the
    marginal distribution of decoded values.

    Concatenates both true-valence pools and returns ``{1: perm, 0: perm}`` -- two
    independent permutations of the SAME pooled values, so a trial's reward no longer
    carries any information about its true outcome. This is the exact ``shuf`` construction
    the original driver used to bound how much of the learning is genuine reward information
    versus the reward's marginal statistics."""
    rng = np.random.default_rng(seed)
    allv = np.concatenate([pools[1], pools[0]])
    return {1: rng.permutation(allv), 0: rng.permutation(allv)}


def _running(r1d, window=50):
    """Running-``window`` mean of a 1-D per-trial reward sequence (the learning curve)."""
    c = np.cumsum(np.insert(np.asarray(r1d, float), 0, 0.0))
    return (c[window:] - c[:-window]) / window


def run_dopamine_shallow(pools, shuf, *, seeds=20, trials=600, S=2, A=2,
                         tau_leak=10.0, D=2.0, seed0=0):
    """G-shallow (Experiment 11): single-layer contextual bandit under the measured
    dopamine reward.

    Trains the 2x2 bandit (:func:`mrl_trace.bandit.train`) with three reward
    sources -- ``synthetic`` (clean {0,1}), ``dopamine`` (the decoded ``pools`` gating the
    three-factor update), and ``shuffled`` (``shuf``, the valence-scrambled control) -- at
    the frozen operating point (nothing retuned for the dopamine reward). The reported
    reward rate is TRUE task performance; the decoded dopamine value only gates the update.

    Serial, returns a plain dict ``{finals, curves, ci}`` (per-seed final reward rate,
    mean running curve, bootstrap CI per source); no file I/O, no plotting, no stdout."""
    from .bandit import train as bandit_train, reward_rate as bandit_rate
    from .stats import bootstrap_ci

    finals, curves = {}, {}
    for name, rp in [("synthetic", None), ("dopamine", pools), ("shuffled", shuf)]:
        r = bandit_train(S, A, B=seeds, tau_leak=tau_leak, D=D, trials=trials,
                         seed0=seed0, reward_pools=rp)
        finals[name] = bandit_rate(r, window=100)
        curves[name] = _running(r.mean(0))
    ci = {n: bootstrap_ci(finals[n]) for n in finals}
    return {"finals": finals, "curves": curves, "ci": ci}


def _deep_worker(spec, trials, nseed, pools, shuf):
    """One deep-XOR condition (rule x reward source) -- module-level so ``main()`` can
    dispatch it over a :class:`multiprocessing.Pool`. Returns
    ``((rule, rname), finals (nseed,), curve)``."""
    from .deep import train_deep, reward_rate as deep_rate
    rule, rname = spec
    homeo = 0.0 if rule == "dfa" else HOMEO
    rp = {"synthetic": None, "dopamine": pools, "shuffled": shuf}[rname]
    rew = train_deep(mode="dfa", B=nseed, trials=trials, seed0=0, homeo=homeo,
                     reward_pools=rp, **HP)
    finals = deep_rate(rew, window=200)
    win = 100
    csum = np.cumsum(rew, axis=1)
    curve = ((csum[:, win:] - csum[:, :-win]) / win).mean(0)
    return (rule, rname), finals, curve


def run_dopamine_deep(pools, shuf, *, seeds=20, trials=3000, pool=None):
    """G-deep (Experiment 11): deep XOR at the frozen Arm-D operating point under the
    measured dopamine reward.

    Runs the full rule x reward grid: rules ``{dfa, dfa_homeo}`` x reward sources
    ``{synthetic, dopamine, shuffled}`` (six conditions). The temporal factor (the device
    trace) and every hyper-parameter (:data:`HP`, :data:`HOMEO`) are frozen at the Arm-D
    point; only the SPATIAL-credit rule (homeostasis on/off) and the reward SOURCE vary.

    Serial by default (a notebook can call it in-kernel); pass an open
    :class:`multiprocessing.Pool` as ``pool`` to dispatch the six conditions in parallel
    (used by ``main()``). Returns a plain dict ``{finals, curves, ci}`` keyed by the
    ``(rule, rname)`` tuple; no file I/O, no plotting, no stdout."""
    from .stats import bootstrap_ci
    specs = [(rule, rname) for rule in ("dfa", "dfa_homeo") for rname in REW]
    args = (specs, [trials] * len(specs), [seeds] * len(specs),
            [pools] * len(specs), [shuf] * len(specs))
    if pool is None:
        res = [_deep_worker(*a) for a in zip(*args)]
    else:
        res = pool.starmap(_deep_worker, list(zip(*args)))
    finals = {k: f for k, f, _ in res}
    curves = {k: c for k, _, c in res}
    ci = {k: bootstrap_ci(finals[k]) for k in finals}
    return {"finals": finals, "curves": curves, "ci": ci}


def _dopamine_criteria(shallow, deep, chance=_CHANCE):
    """Evaluate the retrospective criteria G1-G3 from the shallow + deep result dicts.

    G1  shallow learns from real dopamine (CI-lo > max(chance, shuffled CI-hi)).
    G2  deep genuine reward info (homeo dopamine CI-lo > its own shuffled CI-hi, disjoint).
    G3  homeostasis robustness (homeo dopamine CI-lo > DFA-alone dopamine CI-hi).
    (G4 is descriptive-only: the dopamine ceiling vs the EEG capstone; reported in
    ``main()``.) Returns ``{"G1": bool, "G2": bool, "G3": bool}``."""
    s_ci = shallow["ci"]
    d_ci = deep["ci"]
    g1 = s_ci["dopamine"][0] > max(chance, s_ci["shuffled"][1])
    hda = d_ci[("dfa_homeo", "dopamine")]
    hsh = d_ci[("dfa_homeo", "shuffled")]
    g2 = hda[0] > hsh[1]
    dda = d_ci[("dfa", "dopamine")]
    g3 = hda[0] > dda[1]
    return {"G1": bool(g1), "G2": bool(g2), "G3": bool(g3)}


def _build_grid(shallow, deep, meta, *, seeds, trials_deep, trials_shallow):
    """Package the shallow + deep result dicts into the exact saved-grid schema the
    original ``exp11_dopamine_capstone.npy`` used (tuple deep-keys stringified for the
    pickle, criteria pre-computed)."""
    finals = shallow["finals"]
    d_finals, d_curves, d_ci = deep["finals"], deep["curves"], deep["ci"]
    return {
        "shallow": {"finals": finals, "curves": shallow["curves"], "ci": shallow["ci"]},
        "deep": {"finals": {str(k): v for k, v in d_finals.items()},
                 "curves": {str(k): v for k, v in d_curves.items()},
                 "ci": {str(k): v for k, v in d_ci.items()}},
        "meta": meta, "HP": HP, "homeo": HOMEO, "seeds": seeds,
        "trials_deep": trials_deep, "trials_shallow": trials_shallow,
        "chance": _CHANCE, "crit": _CRIT,
        "criteria": _dopamine_criteria(shallow, deep),
    }


def _replot_from_cache():
    """Graceful ``--figonly`` path: load the cached grid via ``paths`` and report its
    contents WITHOUT rerunning training (or requiring the DANDI cache to be present).

    The figure itself is rendered by the topic notebook from this same grid; this entry
    point just confirms the cached grid is readable and prints its headline numbers.
    Returns the loaded grid dict, or ``None`` if no cache has been written yet."""
    from . import paths
    fn = "exp11_dopamine_capstone.npy"
    path = paths.results_dir() / fn
    if not path.exists():
        print(f"  [figonly] no cached grid at {path}; run without --figonly first")
        return None
    d = paths.load_result(fn)
    meta = d.get("meta", {})
    print(f"  [figonly] loaded {fn}: {d.get('seeds')} seeds, "
          f"DA bal-acc {meta.get('mean_bal_acc', float('nan')):.3f}, "
          f"criteria={d.get('criteria')}")
    return d


def main(argv=None):
    """Full-scale reproduction CLI for the Arm-G dopamine capstone (writes ``data/results``).

    ``python -m mrl_trace.dopamine [--exp11] [--figonly] [--full|--quick]``

    Decodes the real dopamine reward pools from the DANDI 000351 cache
    (:func:`build_reward_pools`), runs the frozen-operating-point shallow bandit and deep
    XOR grids, computes the retrospective G1-G4 criteria and writes
    ``exp11_dopamine_capstone.npy`` (the exact filename the original driver produced) via
    ``paths.save_result``.

    Graceful degradation (matching the original): with ``--figonly`` it only re-reports
    the cached grid; and if the DANDI cache under ``DA_CACHE`` is empty/absent (as it is
    wherever the multi-GB raw photometry has not been streamed), it prints a note and
    skips the run without error rather than crashing."""
    import argparse
    import time
    from . import paths

    ap = argparse.ArgumentParser(description="Arm-G dopamine capstone (Experiment 11)")
    ap.add_argument("--exp11", action="store_true",
                    help="dopamine capstone -> exp11_dopamine_capstone.npy (the default)")
    ap.add_argument("--figonly", action="store_true",
                    help="re-report the cached grid without rerunning training")
    ap.add_argument("--quick", action="store_true", help="fast few-seed smoke run")
    ap.add_argument("--full", action="store_true", help="published 20-seed run (default)")
    a = ap.parse_args(argv)

    if a.figonly:
        _replot_from_cache()
        return

    t0 = time.time()
    td = 600 if a.quick else 3000
    ts = 300 if a.quick else 600
    nseed = 6 if a.quick else 20

    print("Dopamine capstone (Arm G): REAL measured-dopamine reward x device trace")
    print(f"  decoding dopamine reward pools from DANDI 000351 (Jeong 2022) [DA_CACHE={DA_CACHE}] ...")
    pools, meta = build_reward_pools()
    if meta["n_subj"] == 0 or pools[1].size == 0 or pools[0].size == 0:
        # Graceful skip: the DANDI cache is empty/absent here (the multi-GB raw photometry
        # has not been streamed via the guarded DANDI extraction workflow). Report and exit cleanly --
        # do NOT write a degenerate grid over any existing published one.
        print(f"  [skip] no usable dopamine sessions under DA_CACHE={DA_CACHE!r} "
              f"(n_subj={meta['n_subj']}); run the guarded DANDI extraction workflow first. "
              f"Nothing written.")
        return
    print(f"  {meta['n_subj']} mice | single-trial decoder bal-acc {meta['mean_bal_acc']:.3f}"
          f" +/- {meta['sd_bal_acc']:.3f}, AUC {meta['mean_auc']:.3f}")
    print(f"  P(R=1 | reward)={meta['P_R1_given_reward']:.3f}  "
          f"P(R=1 | omission)={meta['P_R1_given_omission']:.3f}  (clean would be 1.00/0.00)")

    shuf = make_shuffled_pools(pools, seed=0)

    print(f"\n[G-shallow] single-layer contextual bandit | {nseed} seeds, {ts} trials")
    shallow = run_dopamine_shallow(pools, shuf, seeds=nseed, trials=ts)
    for name in REW:
        lo, hi = shallow["ci"][name]
        print(f"  {name:10s} reward rate = {shallow['finals'][name].mean():.3f} [{lo:.3f}, {hi:.3f}]")

    print(f"\n[G-deep] deep XOR, frozen Arm-D point | {nseed} seeds, {td} trials")
    import multiprocessing as mp
    specs = [(rule, rname) for rule in ("dfa", "dfa_homeo") for rname in REW]
    nproc = min(len(specs), (os.cpu_count() or 4) - 1)
    with mp.Pool(processes=max(1, nproc)) as pool:
        deep = run_dopamine_deep(pools, shuf, seeds=nseed, trials=td, pool=pool)
    d_finals, d_ci = deep["finals"], deep["ci"]
    print(f"  {'rule':18s} {'reward':12s} {'mean':>6} {'95% CI':>16} {'solved':>8}")
    for rule in ("dfa", "dfa_homeo"):
        for rname in REW:
            k = (rule, rname); lo, hi = d_ci[k]
            print(f"  {rule:18s} {rname:12s} {d_finals[k].mean():6.3f} "
                  f"[{lo:5.3f},{hi:5.3f}]  {int((d_finals[k] >= 0.9).sum()):>3}/{nseed}")

    crit = _dopamine_criteria(shallow, deep)
    da_deep = d_finals[("dfa_homeo", "dopamine")].mean()
    da_shal = shallow["finals"]["dopamine"].mean()
    print("\n=== recorded retrospective criteria (dopamine protocol) ===")
    print(f"  G1 shallow learns from dopamine (>chance & >shuffled, disjoint): "
          f"{'PASS' if crit['G1'] else 'FAIL'} "
          f"(dopamine {da_shal:.3f}{shallow['ci']['dopamine']} vs "
          f"shuf {shallow['finals']['shuffled'].mean():.3f}{shallow['ci']['shuffled']})")
    print(f"  G2 deep genuine reward info (homeo dopamine > shuffled, disjoint): "
          f"{'PASS' if crit['G2'] else 'FAIL'} "
          f"(homeo {da_deep:.3f}{d_ci[('dfa_homeo', 'dopamine')]} vs "
          f"shuf {d_finals[('dfa_homeo', 'shuffled')].mean():.3f}{d_ci[('dfa_homeo', 'shuffled')]})")
    print(f"  G3 homeostasis robustness (homeo dopamine CI-lo > DFA-alone CI-hi): "
          f"{'PASS' if crit['G3'] else 'FAIL'} "
          f"(homeo {da_deep:.3f}{d_ci[('dfa_homeo', 'dopamine')]} vs "
          f"DFA {d_finals[('dfa', 'dopamine')].mean():.3f}{d_ci[('dfa', 'dopamine')]})")
    print(f"  G4 ceiling vs EEG (descriptive): dopamine deep {da_deep:.3f} / shallow {da_shal:.3f}"
          f" | EEG was 0.62 / 0.66 ; DA decoder {meta['mean_bal_acc']:.2f} vs EEG 0.69")

    grid = _build_grid(shallow, deep, meta, seeds=nseed,
                       trials_deep=td, trials_shallow=ts)
    paths.save_result("exp11_dopamine_capstone.npy", grid)
    print(f"\n  wrote exp11_dopamine_capstone.npy  criteria={grid['criteria']} "
          f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
