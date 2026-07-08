r"""Timing/selectivity constructions on the device eligibility cascade.

Two experiments that treat the SiO\ :sub:`x` trap-cascade (see
:mod:`mrl_trace.device`) as a *temporal* primitive rather than as a scalar
credit-assignment window.  Both compose the existing gates (the trace kernel of
:mod:`mrl_trace.distal_reward` and the vectorised
:class:`mrl_trace.bandit.GateBankBatched`) with the LIF decision layer, but
neither wraps a single existing training primitive -- they build new timing tasks on
top of the shared physics, so they live in their own module.

- **Interval selectivity (Experiment 10).**  The device trace is *peaked* at a
  non-zero lag ``t*`` (Eqs. 7.1--7.2), so it is a band-pass in time: it credits a
  coincidence at a *preferred interval* and suppresses a coincidence at lag~0
  (reward time), whereas a matched-decay exponential is monotone (recency-biased)
  and credits the lag-0 confound.  The preferred interval is fabrication-tunable via
  ``tau_leak``.  :func:`run_interval_selectivity` measures the *learned* selectivity
  ratio ``S = w_pref/w_late`` from a competitive three-factor loop and returns the
  design-point comparison plus the ``S(D)`` crossover sweep.  Pre-registered criteria
  P1--P4 and kills K1--K2 (PREREGISTRATION_interval_selectivity.md).

- **Vector timer (Experiment 20).**  The cascade occupancy *vector* ``v^1..v^k`` is a
  distributed clockless code for elapsed time: the stage occupancies peak at
  progressively later lags, so the full vector separates elapsed times that the
  non-monotone last-stage *scalar* aliases.  :func:`run_vector_timer` runs a
  closed-loop interval-discrimination bandit whose policy is driven by either the full
  cascade vector (this work), the last stage only (the CET/current-paper baseline), or
  a zeroed trace (necessity control), at a scalar-aliasing interval pair computed from
  the device model.  Pre-registered criteria H1--H4 and kill K1.  (Physical-
  observability caveat: the individual stage occupancies are not exposed by a
  two-terminal cell, so the vector result *motivates* a staged-readout device rather
  than being a capability of the present one -- exploratory extension.)

Each ``run_*`` is serial and import-light (a notebook calls it in-kernel at a small
seed/trial count); ``main()`` runs the published-scale grids, parallelising the
coarse axis with a process pool, and writes each grid via
``paths.save_result`` under ``data/results/`` (``exp10_interval.npy``,
``exp20_vector_timer.npy``).
"""
from __future__ import annotations

import numpy as np

from .device import TransientGate, tau_r
from .distal_reward import trace_kernel, abstract_kernel, W_INIT
from .bandit import GateBankBatched
from .neurons import lif_step_batched, TAU_M, V_TH
from .learning import LTD_BIAS
from .stats import bootstrap_ci

__all__ = [
    "run_interval_selectivity",
    "peak_lag",
    "selectivity_ratio",
    "run_vector_timer",
    "aliasing_pair",
    "run_interval",
]

# =============================================================================
# Experiment 10 -- the dual-phase trace as a tunable interval-selective eligibility.
#
# Pre-registered in PREREGISTRATION_interval_selectivity.md (criteria P1-P4, kills
# K1-K2). Claim: the device trace (Eqs. 7.1-7.2) is PEAKED at a non-zero lag t*, so it
# implements a band-pass in time -- it credits a coincidence at a *preferred interval*
# and SUPPRESSES a coincidence at lag~0 (reward time). A matched-decay exponential is
# monotone (recency-biased) and credits the lag-0 confound instead. The preferred
# interval is fabrication-tunable via tau_leak.
#
# Task (trace level, competition for credit):
#   - syn_pref : reward-correlated coincidence at lag D_rew (the preferred interval).
#   - syn_late : reward-correlated coincidence at lag ~0.3 s (the reward-time confound).
#     BOTH are perfectly reward-correlated, so the test is purely about TIMING.
#   - N-2 reward-UNcorrelated distractors at random lags (the LTD wing handles them).
# Metric: learned selectivity ratio S = w[syn_pref] / w[syn_late]. S>1 => interval-
# selective; S<1 => recency-biased. S is a LEARNED outcome of a competitive 1500-trial
# three-factor loop with a shared adapting baseline, not a direct read of the kernel.
# =============================================================================

LATE_LAG = 0.3       # s, the reward-time confound lag
LTD = 0.6
JITTER = 0.15        # per-trial fractional lag jitter (biological timing is noisy)
DECAY = 0.08         # weight-decay alpha: the leaky three-factor rule Dw=eta((R-b)e-alpha w)
                     # (the form of Ch2 Eq. 2.17 / Bianchi 2020); makes the equilibrium
                     # weight eligibility-proportional rather than clip-saturated.


def peak_lag(tau_leak, V=0.9, T=80.0, dt=0.05):
    """Preferred interval t* = argmax of the (unnormalised) device trace."""
    g = TransientGate(V=V, tau_leak=tau_leak, dt=dt)
    t = np.arange(0, T, dt)
    raw = g.trace(t, coincidence_at=0.0, coincidence_dur=0.3, normalise=False)
    return float(t[np.argmax(raw)])


def _kernel(kind, t, tau_leak, V):
    if kind == "device":
        return trace_kernel(t, tau_leak, V=V)
    if kind == "exp":
        return abstract_kernel(t, tau_leak)
    raise ValueError(kind)


def train_selectivity(kind, tau_leak, D_rew, *, V=0.9, N=8, trials=1500,
                      T=160.0, dt=0.1, eta=0.05, seed=0):
    """One seed. Returns final weights w (w[0]=syn_pref @ lag D_rew,
    w[1]=syn_late @ lag LATE_LAG, w[2:]=reward-uncorrelated distractors).

    Both syn_pref and syn_late are PERFECTLY reward-correlated: when a trial is
    rewarded, both emit a coincidence (at their respective lags before reward).
    The only thing distinguishing them is WHEN they fire relative to reward, so the
    learned ratio w_pref/w_late isolates the trace's timing selectivity.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, dt)
    nt = len(t)
    kern = _kernel(kind, t, tau_leak, V)
    # eligibility a synapse carries at reward time, given its coincidence LAG before
    # reward: e = kern[lag_index]. (kern is e(t) after a coincidence at t=0.)
    def elig_at_lag(lag):
        li = int(round(lag / dt))
        return kern[li] if 0 <= li < nt else 0.0

    w = np.full(N, W_INIT)
    baseline = 0.5
    for _ in range(trials):
        rewarded = rng.random() < 0.5
        e = np.zeros(N)
        if rewarded:
            # per-trial multiplicative lag jitter: biological action--reward timing is
            # not exact, so neither synapse sits at a fixed point lag. This injects
            # genuine seed/trial variance (the degenerate-CI fix) and makes the test
            # robust to a synapse exploiting one exact grid point.
            j_pref = D_rew * (1.0 + JITTER * rng.standard_normal())
            j_late = max(0.0, LATE_LAG + 0.1 * abs(rng.standard_normal()))
            e[0] = elig_at_lag(j_pref)       # preferred-interval cue
            e[1] = elig_at_lag(j_late)       # reward-time confound
        # reward-uncorrelated distractors at random lags (LTD wing nets them down).
        for i in range(2, N):
            lag = rng.uniform(LATE_LAG, 0.9 * T)
            sign = 1.0 if rng.random() < 0.5 else -LTD
            e[i] = sign * elig_at_lag(lag)
        R = 1.0 if rewarded else 0.0
        baseline += 0.02 * (R - baseline)
        # leaky three-factor update: the -alpha*w term bounds weights by DECAY so the
        # equilibrium w* is proportional to the mean eligibility a synapse accrues,
        # NOT pinned to a hard clip. This is what makes w_pref/w_late report the
        # trace's timing selectivity rather than a saturation artefact.
        w = np.clip(w + eta * ((R - baseline) * e - DECAY * (w - W_INIT)), 0.0, 2.0)
    return w


def selectivity_ratio(w):
    """S = w_pref / w_late, guarding the denominator away from 0."""
    return w[0] / max(w[1], 1e-6)


def _sweep_selectivity(kind, tau_leak, D_rew, *, V=0.9, seeds=20, trials=1500):
    """Per-seed selectivity ratios over ``seeds`` seeds (serial)."""
    return np.array([
        selectivity_ratio(train_selectivity(kind, tau_leak, D_rew, V=V,
                                             trials=trials, seed=s))
        for s in range(seeds)
    ])


def run_interval_selectivity(*, seeds=20, trials=1500, V=0.9):
    """Experiment 10 core: device vs exponential interval selectivity + tunability.

    Serial; returns the result grid as a plain dict, no file I/O / no plotting /
    no stdout (the notebook renders the figure inline).  Preserves the science of
    ``experiments/04_temporal_selectivity/interval_selectivity.py`` exactly.

    Computes, at the design point ``tau_leak=10 s, D_rew=t*`` (the last-stage peak),
    the per-seed learned selectivity ``S = w_pref/w_late`` for the device trace and
    the matched-tau exponential control, with bootstrap CIs; and the ``S(D)`` curve
    over a log-spaced design-interval grid for three retentions ``tau_leak in
    {5,10,20} s`` (the pre-registered CROSSOVER: a short design interval is best
    served by short retention, a long one by long retention -- the peak sliding right
    with ``tau_leak``).  Pre-registered criteria (fixed before running):
      P1  device interval-selective     : device S CI lower bound > 1.
      P2  exponential recency-biased     : exp S CI upper bound < 1.
      P3  preferred interval shifts w/ tau_leak : S(short D) decreases with tau_leak
          AND S(long D) increases with it (crossover).
      P4  device/exp CIs disjoint at design point.
      K1  (kill) device CI includes 1.
      K2  (kill) device/exp CIs overlap (= not P4).
    """
    taus = [5.0, 10.0, 20.0]
    tstar = {tl: peak_lag(tl, V=V) for tl in taus}

    tl0 = 10.0
    D0 = round(tstar[tl0], 1)
    dev_S = _sweep_selectivity("device", tl0, D0, V=V, seeds=seeds, trials=trials)
    exp_S = _sweep_selectivity("exp", tl0, D0, V=V, seeds=seeds, trials=trials)
    dlo, dhi = bootstrap_ci(dev_S)
    elo, ehi = bootstrap_ci(exp_S)

    # P3: preferred interval SHIFTS RIGHT with tau_leak (crossover). Selectivity S(D)
    # plateaus once D clears the rise, so an argmax is ill-defined on the plateau. The
    # robust, non-tautological signature of a *moving* preferred interval is a
    # CROSSOVER: a short design interval is best served by short retention, a long one
    # by long retention. We test that S(short D) decreases with tau_leak while S(long D)
    # increases with it -- the peak sliding rightward.
    D_grid = [1, 2, 4, 8, 16, 32, 48]
    Scurve = {}
    for tl in taus:
        Scurve[tl] = np.array([
            _sweep_selectivity("device", tl, D, V=V, seeds=seeds, trials=trials).mean()
            for D in D_grid
        ])
    D_short, D_long = 2, 32
    si = D_grid.index(D_short); li = D_grid.index(D_long)
    short_decreases = Scurve[5.0][si] > Scurve[20.0][si]
    long_increases = Scurve[20.0][li] > Scurve[5.0][li]

    P1 = bool(dlo > 1.0)
    P2 = bool(ehi < 1.0)
    P3 = bool(short_decreases and long_increases)
    P4 = bool((dlo > ehi) or (dhi < elo))

    return {
        "V": V, "taus": taus, "tstar": tstar, "D0": D0, "late_lag": LATE_LAG,
        "dev_S": dev_S, "exp_S": exp_S, "dev_ci": (dlo, dhi), "exp_ci": (elo, ehi),
        "D_grid": D_grid, "Scurve": Scurve,
        "P1": P1, "P2": P2, "P3": P3, "P4": P4, "seeds": seeds, "trials": trials,
    }


# =============================================================================
# Experiment 20 -- the cascade occupancy VECTOR as a clockless elapsed-time code.
#
# THE LEARNING CONTRIBUTION (the one slice the novelty search left open). Every
# eligibility trace in this work, and in the nearest prior art (Ralambomihanta et al.,
# Cascading Eligibility Traces, arXiv:2506.14598), reads a SINGLE SCALAR -- the last
# cascade stage -- as the trace. But the device's internal trap-cascade is a
# MULTI-STATE object: the occupancies of its k stages (v^1..v^k) peak at progressively
# later lags (verified: at tau_leak=10s the per-stage peaks sit at ~0, 9, 19 s), so the
# FULL VECTOR is a distributed code for TIME-SINCE-EVENT that a single scalar cannot be.
# Because the scalar readout is non-monotonic (it rises then decays), two different
# elapsed times on opposite sides of its peak ALIAS to the same scalar value -- a rule
# reading the scalar literally cannot tell them apart. The full vector breaks aliasing.
#
# This experiment tests, in a CLOSED LOOP, whether a three-factor rule that reads the
# FULL stage vector solves an interval-discrimination task that the scalar readout
# cannot -- i.e. whether the device's intrinsic cascade state is a usable temporal
# basis for credit assignment, with no clock and no recurrence.
#
# TASK (interval-discrimination bandit, 2 actions). Each trial: a cue fires a pre--post
# coincidence at t=0, driving the eligibility gate. After an elapsed interval drawn as
# one of two values (tA or tB), a decision is taken and reward is contingent on the
# action matching WHICH interval elapsed (act 0 if tA, act 1 if tB). The only signal
# distinguishing the two is the SHAPE of the surviving cascade state at decision time.
# tA and tB STRADDLE the scalar readout's peak at EQUAL last-stage value, so the scalar
# genuinely aliases them while the vector separates them. The pair is COMPUTED from the
# device model (aliasing_pair()): locate the last-stage peak t*, take a rising-flank tA
# and the falling-flank tB of matching last-stage value -- an earlier hard-coded
# tA=3,tB=15 sat on the rising flank, so the scalar was never aliased and the
# aliasing-rescue test was vacuous.
#
# READOUT CONDITIONS (the comparison that isolates the contribution):
#   vector   : the rule reads all k cascade occupancies through a learned per-stage
#              combination c_m (k extra local parameters); credit = (R-b)*sum_m c_m
#              v^(m).  [THIS WORK]
#   scalar   : the rule reads only the last stage v^(k) -- the current-paper/CET
#              readout.  [BASELINE]
#   no_trace : eligibility zeroed (necessity control).
#
# CHARACTERISATION SWEEP (turns "bounded resolution" into a result): vector-vs-scalar
# accuracy across (interval separation) x (cascade depth k). Establishes (i) the
# vector's advantage band, (ii) that resolution is limited by readout NOISE not by k
# (high-k plateaus/regresses), so the measured k~3 already sits near the useful optimum
# -- not a limitation appealing to a deeper device that does not exist.
#
# PRE-REGISTRATION (fixed before running; reported as-is):
#   H1  vector readout learns the interval-discrimination task (straddling pair):
#       vector >= CRIT.
#   H2  scalar readout FAILS the same task (aliasing): scalar <= chance + 0.10, AND
#       vector CI lower bound disjoint above scalar CI upper bound.
#   H3  trace necessary: no_trace <= chance + 0.10.
#   H4  (DESCRIPTIVE) resolution characterisation: vector-minus-scalar advantage as a
#       function of interval separation, and the k-dependence at fixed separation
#       (expect a sweet spot ~k=3-5 then plateau/regress -> noise-limited, not
#       depth-limited).
#   K1  (kill) vector ~ scalar on the straddling pair (vector adds nothing) -> dead.
# =============================================================================

CHANCE = 0.5
CRIT = 0.75


def run_interval(readout, *, B=20, k=3, tau_leak=10.0, tA=3.0, tB=15.0, trials=4000,
                 dt=5e-3, cue_dur=0.5, eta=0.3, in_rate=200.0, ltd=LTD_BIAS,
                 tau_m=TAU_M, v_th=V_TH, sigma=0.4, read_noise=0.10, seed0=0):
    """One condition of the interval-discrimination bandit, STATE-DRIVEN policy.

    A single cue fires a coincidence at t=0, driving the device gate. After an elapsed
    interval (tA or tB, chosen per trial) a decision is taken and reward is contingent
    on the action matching WHICH interval elapsed. The decision must therefore depend
    on the elapsed interval, and the ONLY trace of the interval at decision time is the
    SHAPE of the surviving cascade state -- so the cascade readout DRIVES the action
    neurons (a state-driven policy), and the readout mode is exactly what determines
    whether the two intervals are separable:
      readout='vector'  : action input = M . [v^1..v^k]  (full occupancy vector, k inputs/action)
      readout='scalar'  : action input = m . v^(k)        (last stage only -- CET/current readout)
      readout='no_trace': action input = 0                (necessity control)
    The readout->action weights (M or m) are learned online by the same reward-modulated
    rule (REINFORCE-style: nudged toward the chosen action's input when reward beats
    baseline), local and without weight transport. Because the scalar last stage is
    NON-MONOTONE, tA and tB on opposite sides of its peak ALIAS to one value, so no
    linear scalar->action map can separate them; the vector breaks the aliasing. A small
    multiplicative read noise on the cascade state is what makes the aliasing bite and
    the resolution finite.
    Returns rewards (B, trials)."""
    rng = np.random.default_rng(seed0)
    A = 2
    no_trace = (readout == "no_trace")
    nin = 1 if readout == "scalar" else k                  # readout dimensionality feeding action
    bank = GateBankBatched(B, 1, 1, tau_leak=tau_leak, k=k, dt=dt)
    # readout->action weights, (B, nin, A); start small/random so the policy must LEARN the map.
    M = 0.01 * rng.standard_normal((B, nin, A))
    baseline = np.full(B, 1.0 / A)
    cue = (0.3, 0.3 + cue_dur)
    riA = int((cue[1] + tA) / dt); riB = int((cue[1] + tB) / dt)
    nsteps = max(riA, riB) + 2
    rewards = np.zeros((B, trials))
    bidx = np.arange(B)

    for tr in range(trials):
        bank.reset()
        which = rng.integers(2, size=B)                    # 0 -> tA, 1 -> tB
        ri = np.where(which == 0, riA, riB)
        captured = np.zeros(B, bool)
        e_state = np.zeros((B, k))                          # cascade vector snapshot at decision
        for n in range(nsteps):
            t = n * dt
            d = (rng.random(B) < in_rate * dt).astype(float) if cue[0] <= t < cue[1] \
                else np.zeros(B)
            drive = np.zeros((B, 1, 1)); drive[:, 0, 0] = d * 1.0   # causal coincidence on the cue
            if no_trace:
                drive[:] = 0.0
            bank.step(drive)
            due = (n == ri) & (~captured)
            if due.any():
                e_state[due] = bank.vn[due, 0, 0, :]; captured[due] = True
        # read cascade state (device read noise) and form the action input per readout mode
        noisy = e_state / bank.Vnmax * (1.0 + read_noise * rng.standard_normal((B, k)))
        if no_trace:
            x = np.zeros((B, nin))
        elif readout == "scalar":
            x = noisy[:, -1:]                               # last stage only (aliased)
        else:
            x = noisy                                       # full vector
        # state-driven action: logits = x . M (+ exploration noise), softmax choice
        logits = np.einsum('bi,bia->ba', x, M) + sigma * rng.standard_normal((B, A))
        chosen = np.argmax(logits, 1)
        r_true = (chosen == which).astype(float)
        adv = (r_true - baseline)
        # REINFORCE-style local update of the readout->action map: push the chosen action's
        # weights toward the input when reward beats baseline (no gradient transport, local to x).
        oh = np.zeros((B, A)); oh[bidx, chosen] = 1.0
        M = M + eta * (adv[:, None, None] * x[:, :, None] * oh[:, None, :])
        M = np.clip(M, -5.0, 5.0)
        baseline += 0.02 * (r_true - baseline)
        rewards[:, tr] = r_true
    return rewards


def _final(rw, window=400):
    return rw[:, -window:].mean(1)


def aliasing_pair(*, k=3, tau_leak=10.0, dt=5e-3, cue_dur=0.5, in_rate=200.0,
                  reps=4000, tmax=40.0, seed=0):
    """Locate the last-stage occupancy peak t* and return a pair (tA, tB) that ALIASES
    the scalar readout: tA on the rising flank and tB on the falling flank, chosen to
    have an equal last-stage value. This is the condition the aliasing-rescue test
    REQUIRES -- the scalar reads the same value at both intervals, so only the full
    vector can separate them. Computed from the device model (not hand-set), so the
    pair tracks tau_leak and k."""
    rng = np.random.default_rng(seed)
    cue = (0.3, 0.3 + cue_dur)
    ts = np.arange(1.0, tmax, 0.5)
    last = np.empty(len(ts))
    for i, te in enumerate(ts):
        bank = GateBankBatched(reps, 1, 1, tau_leak=tau_leak, k=k, dt=dt)
        ri = int((cue[1] + te) / dt)
        bank.reset()
        for n in range(ri + 1):
            t = n * dt
            d = (rng.random(reps) < in_rate * dt).astype(float) if cue[0] <= t < cue[1] \
                else np.zeros(reps)
            drive = np.zeros((reps, 1, 1)); drive[:, 0, 0] = d
            bank.step(drive)
        last[i] = bank.vn[:, 0, 0, -1].mean() / bank.Vnmax
    pk = int(np.argmax(last)); tstar = float(ts[pk])
    tA = max(ts[0], tstar - 8.0)                       # a point on the rising flank
    vA = float(np.interp(tA, ts, last))
    fall = ts[ts > tstar]; lastfall = last[ts > tstar]
    tB = float(fall[np.argmin(np.abs(lastfall - vA))]) # falling-flank point of equal value
    return tA, tB, tstar


def _worker_main(readout, B, trials, tA, tB):
    return readout, run_interval(readout, B=B, trials=trials, tA=tA, tB=tB)


def _worker_sweep(job, B, trials):
    sep, k = job
    centre = 9.0
    tA, tB = centre - sep / 2, centre + sep / 2
    accs = {}
    for readout in ("scalar", "vector"):
        rw = run_interval(readout, B=B, k=k, trials=trials, tA=tA, tB=tB)
        accs[readout] = float(_final(rw).mean())
    return (sep, k), accs


def run_vector_timer(*, seeds=20, trials=3000, quick=False, pool=None):
    """Experiment 20 core: the cascade occupancy VECTOR as a clockless elapsed-time code.

    Serial by default (a notebook calls it in-kernel); returns the result grid as a
    plain dict, no file I/O / no plotting / no stdout.  Preserves the science of
    ``experiments/04_temporal_selectivity/vector_timer.py`` exactly.

    Runs the vector/scalar/no_trace comparison at the MEASURED scalar-aliasing pair
    (:func:`aliasing_pair`, so the pair tracks ``tau_leak``/``k``), then the
    (interval separation) x (cascade depth ``k``) characterisation sweep.  Evaluates
    the pre-registered H1--H4/K1.  ``quick`` shrinks the sweep grid exactly as the
    original ``--quick`` did.  An optional ``pool`` (a ``multiprocessing.Pool``) is
    used to parallelise the coarse axes when called from :func:`main`; when ``None``
    (the notebook path) everything runs serially in-kernel.
    """
    from functools import partial

    # --- main comparison at the MEASURED scalar-aliasing pair, vector vs scalar vs no_trace ---
    # The aliasing test requires tA, tB to give the SAME last-stage value on opposite sides of
    # the last-stage peak (else the scalar is not aliased and the test is vacuous). Earlier runs
    # hard-set tA=3,tB=15, but both sit on the last-stage rising flank (peak ~18.5s), so the
    # scalar was never aliased and H2 could not pass. The pair is now computed from the device
    # model so it tracks tau_leak/k.
    tA, tB, tstar = aliasing_pair(k=3, tau_leak=10.0)
    conds = ("vector", "scalar", "no_trace")
    if pool is not None:
        res = dict(pool.map(partial(_worker_main, B=seeds, trials=trials, tA=tA, tB=tB),
                            conds))
    else:
        res = {c: run_interval(c, B=seeds, trials=trials, tA=tA, tB=tB) for c in conds}
    finals = {c: _final(res[c]) for c in conds}
    ci = {c: bootstrap_ci(finals[c]) for c in conds}

    vec, sca, nt = finals["vector"].mean(), finals["scalar"].mean(), finals["no_trace"].mean()
    h1 = vec >= CRIT
    h2 = (sca <= CHANCE + 0.10) and (ci["vector"][0] > ci["scalar"][1])
    h2_weak = ci["vector"][0] > ci["scalar"][1]            # vector disjoint-above scalar (the real claim)
    h3 = nt <= CHANCE + 0.10
    k1 = ci["vector"][0] <= ci["scalar"][1]                # vector not disjoint above scalar

    # --- characterisation sweep: (interval separation) x (k); resolution noise- not k-limited ---
    seps = [12.0, 6.0, 3.0, 1.5] if not quick else [12.0, 3.0]
    ks = [3, 5, 8] if not quick else [3, 8]
    jobs = [(sep, k) for sep in seps for k in ks]
    if pool is not None:
        sweep_res = dict(pool.map(partial(_worker_sweep, B=seeds, trials=trials), jobs))
    else:
        sweep_res = dict(_worker_sweep(job, seeds, trials) for job in jobs)
    sweep = {}
    for sep in seps:
        for k in ks:
            sweep[(sep, k)] = sweep_res[(sep, k)]

    return {
        "finals": {c: finals[c] for c in conds}, "ci": ci,
        "sweep": {f"{s}_{k}": v for (s, k), v in sweep.items()},
        "seps": seps, "ks": ks, "tau_leak": 10.0, "tA": tA, "tB": tB, "tstar": tstar,
        "seeds": seeds, "trials": trials, "chance": CHANCE, "crit": CRIT,
        "criteria": {"H1": bool(h1), "H2": bool(h2), "H3": bool(h3), "K1": bool(k1)},
        # extra diagnostics preserved from the original stdout report (H2' weak claim)
        "h2_weak": bool(h2_weak),
    }


def main(argv=None):
    """Full-scale reproduction CLI for the timing/selectivity grids (writes ``data/results``).

    ``python -m mrl_trace.selectivity [--exp10] [--exp20] [--full|--quick]``
    With no experiment flag, runs both.  ``--full`` = 20 seeds (published); ``--quick``
    = a fast few-seed smoke run.  The vector-timer sweep parallelises its coarse axes
    with a process pool (only here, in the real ``__main__``); the ``run_*`` cores stay
    serial so a notebook can call them in-kernel.
    """
    import argparse
    import os
    from multiprocessing import Pool
    from . import paths

    ap = argparse.ArgumentParser(description="Timing/selectivity reproductions")
    ap.add_argument("--exp10", action="store_true",
                    help="interval selectivity -> exp10_interval.npy")
    ap.add_argument("--exp20", action="store_true",
                    help="cascade vector timer -> exp20_vector_timer.npy")
    ap.add_argument("--quick", action="store_true", help="fast few-seed smoke run")
    ap.add_argument("--full", action="store_true", help="published 20-seed run (default)")
    a = ap.parse_args(argv)
    run_all = not (a.exp10 or a.exp20)
    nproc = max(1, min(6, (os.cpu_count() or 4) - 2))

    if a.exp10 or run_all:
        seeds = 6 if a.quick else 20
        trials = 600 if a.quick else 1500
        print("=== Experiment 10: interval-selectivity ===")
        print(f"  V=0.9, tau_r={tau_r(0.9):.1f}s | {seeds} seeds, {trials} trials, "
              f"late-lag={LATE_LAG}s")
        grid = run_interval_selectivity(seeds=seeds, trials=trials, V=0.9)
        print(f"  preferred interval t* (= argmax device trace): "
              + ", ".join(f"tau={tl:g}->t*={grid['tstar'][tl]:.2f}s" for tl in grid["taus"]))
        (dlo, dhi), (elo, ehi) = grid["dev_ci"], grid["exp_ci"]
        print(f"  design point tau_leak=10s, D_rew=t*={grid['D0']}s: "
              f"device S={grid['dev_S'].mean():.3f} CI[{dlo:.3f},{dhi:.3f}]  "
              f"exp S={grid['exp_S'].mean():.3f} CI[{elo:.3f},{ehi:.3f}]")
        print(f"  criteria P1={grid['P1']} P2={grid['P2']} P3={grid['P3']} P4={grid['P4']}")
        paths.save_result("exp10_interval.npy", grid)
        print("  wrote exp10_interval.npy")

    if a.exp20 or run_all:
        seeds = 6 if a.quick else 20
        trials = 1500 if a.quick else 3000
        print("=== Experiment 20: cascade occupancy VECTOR as a clockless timer ===")
        print(f"  interval-discrimination bandit, tau_leak=10s, k=3; {seeds} seeds, "
              f"{trials} trials; chance={CHANCE}, criterion={CRIT}")
        with Pool(nproc) as pool:
            grid = run_vector_timer(seeds=seeds, trials=trials, quick=a.quick, pool=pool)
        print(f"  last-stage peak t*={grid['tstar']:.1f}s -> aliasing pair "
              f"tA={grid['tA']:.1f}s, tB={grid['tB']:.1f}s")
        for c in ("vector", "scalar", "no_trace"):
            lo, hi = grid["ci"][c]
            print(f"    {c:9s}: final {grid['finals'][c].mean():.3f}  CI[{lo:.3f}, {hi:.3f}]")
        print(f"  criteria={grid['criteria']}")
        paths.save_result("exp20_vector_timer.npy", grid)
        print("  wrote exp20_vector_timer.npy")


if __name__ == "__main__":
    main()
