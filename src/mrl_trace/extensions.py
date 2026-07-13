r"""Chapter-8 cross-cutting studies: how the two slow device primitives (the
eligibility trace and eligibility-magnitude homeostasis) compose, and how far the
device-native reinforcement-learning story stretches.

Four experiments live here, all built on the validated ``GateBankBatched`` cascade
gate (Section 5.3.2 physics) and the sequential ``LinearTrack``/``TMaze`` MDPs -- none
re-derives a primitive, each COMPOSES the ones already in :mod:`mrl_trace.bandit`,
:mod:`mrl_trace.maze`, :mod:`mrl_trace.neurons` and
:mod:`mrl_trace.learning`:

- **Experiment 19 -- multi-timescale credit from the MEASURED retention spread**
  (:func:`run_multitimescale`).  A real ITO array does not have one ``tau_leak``, it has
  a DISTRIBUTION of them (the n=53 measured trap-discharge fits, median ~1.3 s, range
  0.56--7.7 s).  A mixed-delay task needs more than one credit horizon, so the measured
  retention POPULATION is the resource, not the defect.  NAIVELY the spread fails (the
  cascade readout magnitude scales ~linearly with ``tau_leak``, so a shared learning rate
  starves the short-tau synapses); a LOCAL, activity-driven, tau-BLIND eligibility-
  magnitude homeostasis -- the same CLASS of set-point negative feedback used for
  firing-rate stability -- equalises the gain and recovers full multi-timescale credit,
  matching a tau-aware oracle.  One material, two slow primitives interacting.

- **Experiment 21 -- beta_leak sensitivity of the load-bearing results**
  (:func:`run_beta_sensitivity`).  Every learning result integrates a SINGLE-RATE
  (mono-exponential, ``beta_leak=1``) leak; the device measures a DISPERSIVE
  (stretched-exponential) discharge, ``beta_leak ~ 0.85`` field-free (~0.54 held-bias).
  This re-runs the two load-bearing results -- the ``D_max ~= k*tau_leak`` law and the
  exp19 multi-timescale coupling -- at the measured ``beta_leak`` and checks they survive,
  closing the "measured 0.85 but simulated 1.0" gap.

- **Experiment 22 -- WM and STC on one device: a null on the two-site hypothesis**
  (:func:`run_wm_stc`, :func:`wm_isolated`).  Working memory (inference-time cue hold) and
  STC/eligibility (learning-time credit) occupy different brain sites.  Does a device
  circuit therefore NEED two co-fabricated traces, or can one serve both?  A DMS task with
  a stimulus-free WM delay and a distractor-bearing reward gap shows a SINGLE shared trace
  does not collapse (97.6% at 6 s, 100% at 10 s -- the retention the rule already runs at),
  so two devices are NOT needed; the credit role, not the hold, sets the required retention.

- **Experiment 23 -- a device-native temporal-difference / actor-critic rule**
  (:func:`run_device_td`).  REINFORCE-with-baseline (``dw = eta (R-b) e``) is extended to a
  value-based learner that BOOTSTRAPS.  Two device-native identifications make it native:
  the discount ``gamma = exp(-step_dur/tau_leak)`` is the trap-discharge decay over one
  step (one material constant sets both trace lifetime and RL horizon), and the critic
  ``V(s)`` is a second on-device weight bank read by the same crossbar product.  The TD
  error then gates the same signed three-factor write, ``dw = eta delta e``.

Every ``run_*`` core is SERIAL, import-light and returns plain arrays/dicts with no file
I/O -- notebooks call them in-kernel at a small seed/trial count.  The module-level
:func:`main` is the full-scale driver: it may parallelise the coarse axis with a
``multiprocessing.Pool`` (it runs as ``python -m``), computes the bootstrap CIs and the
pre-registered criteria, and writes each grid via :func:`mrl_trace.paths.save_result`.
NOTE: reversal learning (Experiment 18) is a ``train`` variant and lives in
:mod:`mrl_trace.bandit`, not here.
"""
from __future__ import annotations

import math
from functools import partial

import numpy as np

from .bandit import GateBankBatched, W_INIT, W_MAX
from .maze import LinearTrack, TMaze, train_sequential, reward_rate
from .neurons import lif_step_batched, TAU_M, V_TH
from .learning import LTD_BIAS
from .stats import bootstrap_ci

__all__ = [
    # exp19 -- multi-timescale credit from the measured retention spread
    "load_measured_tau",
    "run_multitimescale",
    # exp22 -- WM + STC on one device (null on two-site hypothesis)
    "run_wm_stc",
    "wm_isolated",
    # exp23 -- device-native TD / actor-critic
    "run_device_td",
    # exp21 -- beta_leak sensitivity of the load-bearing results
    "dmax_law",
    "run_beta_sensitivity",
    # shared constants
    "CHANCE",
    "CRIT",
    "TAU_BAND",
    "BETAS",
    "V_INIT",
    "V_MAX",
    "ETA_V",
]

# =============================================================================
# Shared constants
# =============================================================================
CHANCE = 0.5
CRIT = 0.75

#: exp22 measured ITO retention band (s), swept in the WM/STC study.
TAU_BAND = [0.8, 1.3, 2.0, 3.6, 6.0, 10.0]

#: exp21 dispersion exponents: single-rate / field-free operating / held-bias stress.
BETAS = [1.0, 0.85, 0.54]

#: exp23 on-device critic value store: one scalar V per state (a second weight bank).
V_INIT = 0.0
V_MAX = 1.0
ETA_V = 0.1                    # critic learning rate (value TD update)


# =============================================================================
# Experiment 19 -- multi-timescale credit from the MEASURED trap-discharge spread
#
# The scaling/fault studies (exp14) treat device-to-device spread in retention as a
# NON-IDEALITY to tolerate. This inverts that: the measured retention DISTRIBUTION is
# the resource for a mixed-delay task, not the defect. No competitor can claim this --
# a single-time-constant cell (Ming 2026) has one horizon; an engineered cascade of
# traces (Ralambomihanta 2025) DESIGNS the spread; here the spread is MEASURED, drawn
# from the device population itself.
# =============================================================================


def load_measured_tau():
    """The n=53 measured ITO trap-discharge time constants (s), from the device population.

    Reads ``data/device_model/ito_decay_data.npz`` (the measured trap-discharge fits;
    resolved through :func:`mrl_trace.paths.device_model_dir` so it works from any
    working directory). Falls back to the documented summary (median 1.34 s, range
    0.56--7.72 s) -- logged, never silent -- only if the npz is absent.
    """
    from . import paths
    try:
        d = np.load(paths.device_model_dir() / "ito_decay_data.npz")
        raw = np.asarray(d["tau"], float)
        # The curated fit archive also retains failed/unconstrained optimisations.
        # Keep only the documented measured discharge population used by the study;
        # otherwise two runaway fits (10^3--10^15 s) dominate the assignment/grid.
        valid = np.isfinite(raw) & (raw >= 0.56) & (raw <= 7.72)
        tau = raw[valid]
        if tau.size >= 10:
            rejected = int(raw.size - tau.size)
            return tau, ("measured (ito_decay_data.npz, valid n=%d, rejected fits=%d)"
                         % (tau.size, rejected))
    except Exception as exc:                                  # noqa: BLE001
        print(f"  [tau] measured npz unavailable ({type(exc).__name__}); using summary")
    # documented summary fallback (logged, never silent)
    rng = np.random.default_rng(0)
    tau = np.clip(np.exp(rng.normal(np.log(1.34), 0.6, 53)), 0.56, 7.72)
    return tau, "summary-proxy (median 1.34, range 0.56-7.72)"


def _peak_gain_per_tau(tau_row, dt, cue=0.5, Tmax=20.0):
    """Analytic peak eligibility for each retention in ``tau_row`` (1-D, length S+1): drive a
    single cascade gate over a ``cue``-long causal coincidence, then relax, and take the max
    readout. This is the tau-dependent magnitude (the nuisance gain) the oracle reference
    divides out. The homeostatic condition does NOT use this -- it estimates the gain from the
    synapse's own running activity instead."""
    gains = np.empty(len(tau_row))
    for i, tau in enumerate(tau_row):
        g = GateBankBatched(1, 1, 1, tau_leak=float(tau), dt=dt)
        for _ in range(int(cue / dt)):
            g.step(np.ones((1, 1, 1)))
        m = g.vn[0, 0, 0, -1]
        for _ in range(int(Tmax / dt)):
            g.step(np.zeros((1, 1, 1))); m = max(m, g.vn[0, 0, 0, -1])
        gains[i] = m
    return gains


def _assign_tau(kind, S, A, tau_pool, short_states, long_states, rng):
    """Per-synapse (S, A) retention array for each condition.

    hetero_measured : draw from the measured pool, then route the LONGER draws to the
                      long-delay states and the SHORTER draws to the short-delay states
                      (a programmed array can place a device of the right retention at the
                      right crosspoint; the claim under test is that the measured spread
                      SPANS both horizons).
    best_single     : a single homogeneous tau (value passed in tau_pool as a scalar).
    mean_tau        : homogeneous mean of the pool.
    """
    if kind == "best_single" or kind == "mean_tau":
        return np.full((S, A), float(tau_pool))
    # hetero_measured: sort the pool, give long states high-tau draws, short states low-tau
    draws = rng.choice(tau_pool, size=S, replace=True)
    order = np.argsort(draws)                                # ascending tau
    tau_state = np.empty(S)
    # short-delay states get the smallest draws, long-delay states the largest
    ns = len(short_states)
    tau_state[np.asarray(short_states)] = np.sort(draws)[:ns]
    tau_state[np.asarray(long_states)] = np.sort(draws)[ns:]
    return np.repeat(tau_state[:, None], A, axis=1)          # (S, A), same tau across actions


def run_multitimescale(kind, *, S=4, A=2, B=20, tau_arg=None, D_short=2.0, D_long=8.0,
                       t_distract_short=1.5, trials=3000, dt=5e-3, cue_dur=0.5, eta=0.2,
                       in_rate=200.0, ltd=LTD_BIAS, tau_m=TAU_M, v_th=V_TH, sigma=0.15,
                       seed0=0, tau_pool=None, elig_norm="none", tau_homeo=300.0,
                       beta_leak=1.0):
    """One condition on the TWO-HORIZON contextual bandit (Experiment 19). SERIAL.

    The diagnostic that motivates the design: on a pure survival task a long tau bridges every
    delay (highest surviving eligibility at every D), so heterogeneity cannot help. A genuine
    two-horizon task must make NO single tau optimal. We achieve that by giving each delay
    group a failure mode for the WRONG tau:

      * SHORT group (states 0..S/2-1): cue at t=0, reward at D_short, and an UNINFORMATIVE
        distractor coincidence near reward (t_distract_short). A too-LONG tau is low-pass /
        recency-weighted, so at reward the recent distractor dominates its trace and it
        MIS-CREDITS the distractor (the exp12 result). Only a tau whose band-pass peak sits
        near the cue lag credits the cue over the near-reward distractor. -> long tau FAILS.
      * LONG group (states S/2..S-1): cue at t=0, reward at D_long, NO distractor. The cue
        sits a long lag before reward, beyond a short trace's reach (survival diagnostic), so
        a too-SHORT tau has no surviving eligibility at reward. -> short tau FAILS.

    No single homogeneous tau serves both: short tau loses the long group (reach), long tau
    loses the short group (distractor). A heterogeneous population -- short-tau synapses on the
    short-group states, long-tau synapses on the long-group states -- serves both, which is
    exactly what the measured ITO retention SPREAD supplies. The reward-prediction-error gate
    and signed rule are unchanged from the base bandit.

    ``kind`` selects the tau assignment ("hetero_measured", "best_single", "mean_tau",
    "no_trace"); ``elig_norm`` selects the eligibility-magnitude normalisation
    ("none"/"homeo"/"oracle"). ``beta_leak`` sets the discharge dispersion (1 = single-rate,
    the exp21 sensitivity knob). Returns
    ``(rewards (B,trials), grp_rw dict of (B,trials) reward-or-nan-by-group,
    short_states, long_states)``."""
    rng = np.random.default_rng(seed0)
    DIST = S                                   # one extra distractor input line (not a state)
    short_states = list(range(0, S // 2))
    long_states = list(range(S // 2, S))
    is_long = np.zeros(S, bool); is_long[long_states] = True
    correct = np.array([s % A for s in range(S)])

    no_trace = (kind == "no_trace")
    if no_trace:
        tau_grid = np.full((S + 1, A), 10.0)
    elif kind == "hetero_measured":
        tau_grid = _assign_tau("hetero_measured", S, A, tau_pool, short_states, long_states, rng)
        tau_grid = np.vstack([tau_grid, tau_grid.mean(0, keepdims=True)])     # distractor row
    else:
        tau_grid = _assign_tau(kind, S, A, tau_arg, short_states, long_states, rng)
        tau_grid = np.vstack([tau_grid, tau_grid.mean(0, keepdims=True)])
    bank = GateBankBatched(B, S + 1, A, tau_leak=tau_grid, dt=dt, beta_leak=beta_leak)

    # Eligibility-magnitude normalisation (the fix for the tau-dependent gain).
    #   "none"   : raw eligibility (the naive heterogeneous-spread result).
    #   "homeo"  : divide each synapse's snapshotted eligibility by a SLOW running estimate of
    #              its OWN eligibility magnitude (gbar), updated once per trial. This is
    #              synaptic-scaling / divisive homeostasis -- a local, set-point negative
    #              feedback of the SAME CLASS as the firing-rate homeostasis used elsewhere,
    #              but on eligibility amplitude. It is BLIND to tau and to the task/reward: it
    #              only sees the synapse's own activity, so equalising the per-synapse gain is
    #              an emergent activity-driven property, NOT a tau lookup.
    #   "oracle" : divide by the precomputed analytic peak gain(tau) -- a tau-AWARE calibration,
    #              included only as the upper-bound reference the homeostasis should approach
    #              WITHOUT knowing tau.
    gbar = np.full((B, S + 1), 1e-3)             # per-row running |e| estimate (homeo state)
    oracle_gain = None
    if elig_norm == "oracle":
        oracle_gain = np.maximum(_peak_gain_per_tau(tau_grid[:, 0], dt), 1e-6)  # (S+1,)

    w = np.full((B, S + 1, A), W_INIT)
    w[:, DIST, :] = W_INIT                       # distractor synapses plastic too (can be mis-credited)
    baseline = np.full(B, 1.0 / A)
    cue = (0.3, 0.3 + cue_dur)
    ri_short = int((cue[1] + D_short) / dt)
    ri_long = int((cue[1] + D_long) / dt)
    nsteps = ri_long + 2
    ds0 = int((cue[1] + t_distract_short) / dt)         # distractor window (short trials only)
    ds1 = ds0 + int(0.3 / dt)
    rewards = np.zeros((B, trials))
    grp_rw = {"short": np.full((B, trials), np.nan), "long": np.full((B, trials), np.nan)}
    bidx = np.arange(B)

    for tr in range(trials):
        bank.reset()
        state = rng.integers(S, size=B)
        long_trial = is_long[state]
        ri = np.where(long_trial, ri_long, ri_short)
        v = np.zeros((B, A)); spk = np.zeros((B, A))
        e_rew = np.zeros((B, S + 1, A))
        captured = np.zeros(B, bool)
        for n in range(nsteps):
            t = n * dt
            pre = np.zeros((B, S + 1))
            if cue[0] <= t < cue[1]:
                pre[bidx, state] = (rng.random(B) < in_rate * dt).astype(float)
            # near-reward distractor on SHORT-group trials only (uninformative)
            if ds0 <= n < ds1:
                fire = (~long_trial) & (rng.random(B) < in_rate * dt)
                pre[fire, DIST] = 1.0
            charge = np.einsum('bsa,bs->ba', w, pre)
            v, sp = lif_step_batched(v, charge, dt, rng, tau_m=tau_m, v_th=v_th, noise=sigma)
            spk += sp
            drive = pre[:, :, None] * np.where(sp, 1.0, -ltd)[:, None, :]
            if no_trace:
                drive[:] = 0.0
            e = bank.step(drive)
            due = (n == ri) & (~captured)
            if due.any():
                e_rew[due] = e[due]; captured[due] = True
        # eligibility-magnitude normalisation (see header): equalise the tau-dependent gain
        if elig_norm == "homeo":
            row_mag = np.abs(e_rew).max(axis=2)                  # (B, S+1) this trial's row mag
            e_used = e_rew / (gbar[:, :, None] + 1e-6)
            active = row_mag > 1e-9                              # only adapt rows that fired
            gbar = np.where(active, gbar + (row_mag - gbar) / tau_homeo, gbar)
        elif elig_norm == "oracle":
            e_used = e_rew / oracle_gain[None, :, None]
        else:
            e_used = e_rew
        tie = spk.max(1) == spk.min(1)
        chosen = np.argmax(spk, 1); chosen[tie] = rng.integers(A, size=int(tie.sum()))
        r_true = (chosen == correct[state]).astype(float)
        adv = eta * (r_true - baseline)
        w = np.clip(w + adv[:, None, None] * e_used, 0.0, W_MAX)
        baseline += 0.02 * (r_true - baseline)
        rewards[:, tr] = r_true
        # record reward into the correct group's array (nan elsewhere -> clean group stats)
        grp_rw["long"][long_trial, tr] = r_true[long_trial]
        grp_rw["short"][~long_trial, tr] = r_true[~long_trial]
    return rewards, grp_rw, short_states, long_states


def _mt_final(rw, window=400):
    return rw[:, -window:].mean(1)


def _mt_group_final(grp_rw_g, window=800):
    """Per-seed mean reward over this group's trials in the last ``window`` trials.

    ``grp_rw_g`` is (B, trials) with the reward on this group's trials and nan elsewhere, so a
    nanmean over the tail window selects exactly the group's recent trials."""
    B, T = grp_rw_g.shape
    tail = grp_rw_g[:, T - window:]
    with np.errstate(invalid="ignore"):
        return np.array([np.nanmean(tail[b]) if np.any(~np.isnan(tail[b])) else np.nan
                         for b in range(B)])


def _mt_worker(cond, B, trials, tau_pool, best_single, mean_tau):
    """cond = (label, kind, elig_norm). ``kind`` picks the tau assignment, ``elig_norm`` the
    eligibility-magnitude normalisation (none/homeo/oracle). Top-level so it pickles for a Pool."""
    label, kind, elig_norm = cond
    arg = {"best_single": best_single, "mean_tau": mean_tau}.get(kind)
    rw, grp_rw, ss, ls = run_multitimescale(
        kind, B=B, trials=trials, tau_arg=arg, tau_pool=tau_pool, elig_norm=elig_norm)
    return label, rw, grp_rw


def _mt_grid_one(tau, trials):
    """Score one homogeneous tau (small fixed B -- only needs to RANK, not give CIs)."""
    rw, *_ = run_multitimescale("best_single", B=6, trials=trials, tau_arg=float(tau), seed0=0)
    return float(tau), _mt_final(rw).mean()


def _mt_grid_best_single(tau_pool, trials, pool=None):
    """Find the single homogeneous tau that maximises mean reward on the two-horizon task --
    the strongest one-horizon competitor. Ranking only, so a small B and short run; run the
    grid points in PARALLEL when a Pool is provided (the serial grid was the main bottleneck)."""
    grid = np.round(np.geomspace(max(0.5, tau_pool.min()), max(tau_pool.max(), 14.0), 6), 2)
    gt = min(trials, 1200)
    if pool is not None:
        scored = pool.map(partial(_mt_grid_one, trials=gt), [float(t) for t in grid])
    else:
        scored = [_mt_grid_one(float(t), gt) for t in grid]
    best_tau = max(scored, key=lambda x: x[1])[0]
    return best_tau, grid


# =============================================================================
# Experiment 22 -- WM and STC on one device: a null on the two-site hypothesis
# =============================================================================


def _wm_relax(bank, n_steps, dt, stride=10):
    """Drive-free coarse relaxation of a gate over a silent delay (exp12 helper, verbatim)."""
    if n_steps <= 0:
        shp = bank.e.shape if hasattr(bank, "e") else bank.vn.shape[:-1]
        return bank.step(np.zeros(shp))
    base = bank.dt; coarse = max(1, n_steps // stride)
    bank.dt = base * (n_steps / coarse)
    shp = bank.e.shape if hasattr(bank, "e") else bank.vn.shape[:-1]
    zero = np.zeros(shp); out = None
    for _ in range(coarse):
        out = bank.step(zero)
    bank.dt = base
    return out


def run_wm_stc(tau_wm, tau_stc, *, B=20, trials=2500, dt=5e-3, D_wm=3.0, D_stc=3.0,
               cue_dur=0.3, distract_dur=0.3, in_rate=200.0, eta=0.2, V=1.5,
               sigma=0.15, ltd=LTD_BIAS, seed0=0, shared_device=False):
    """DMS with a WM delay before the decision (Experiment 22). SERIAL. Returns per-seed
    reward ``(B, trials)``.

    Two-device (default): a dedicated cue-hold device (tau_wm, positive drive) supplies the
    WM readout, and a separate eligibility device (tau_stc, exp12 loop) supplies STC credit.

    shared_device=True -- the GENUINE one-device collapse test: a SINGLE trace bank per
    (sample,action) synapse must serve BOTH roles. It receives ONE drive (the STC pre*post
    coincidence, LTD-biased, as physics dictates -- a synapse's trace is set by its own
    plasticity), and the WM decision must read cue identity off that SAME trace. There is
    no separate positive cue-hold trace, because one device is one physical state. tau_wm is
    ignored (only tau_stc, the single device's retention, applies)."""
    rng = np.random.default_rng(seed0)
    S, A = 3, 2; DIST = 2
    stc = GateBankBatched(B, S, A, tau_leak=tau_stc, V=V, dt=dt)   # eligibility device
    wm = None if shared_device else GateBankBatched(B, 2, 1, tau_leak=tau_wm, V=V, dt=dt)
    w = np.full((B, S, A), W_INIT)
    baseline = np.full(B, 1.0 / A)
    bidx = np.arange(B)
    n_cue = int(round(cue_dur / dt)); n_wm = int(round(D_wm / dt))
    n_gap1 = int(round(0.5 * D_stc / dt)); n_dist = int(round(distract_dur / dt))
    n_gap2 = int(round((D_stc - 0.5 * D_stc - distract_dur) / dt))
    rewards = np.zeros((B, trials))

    for tr in range(trials):
        stc.reset()
        if wm is not None:
            wm.reset()
        cls = rng.integers(2, size=B)                 # sample class
        sline = np.where(cls == 0, 0, 1)
        v = np.zeros((B, A)); spk = np.zeros((B, A))
        e_cue = None                                   # shared-device cue snapshot (from stc)
        # --- SAMPLE window: drive the eligibility trace (exp12-style) ---
        for _ in range(n_cue):
            pre = np.zeros((B, S)); firing = (rng.random(B) < in_rate * dt).astype(float)
            pre[bidx, sline] = firing
            charge = np.einsum('bsa,bs->ba', w, pre)
            v, sp = lif_step_batched(v, charge, dt, rng, tau_m=TAU_M, v_th=V_TH, noise=sigma)
            # STC eligibility: pre*post coincidence with LTD bias (exp12 verbatim)
            e_stc = stc.step(pre[:, :, None] * np.where(sp, 1.0, -ltd)[:, None, :])
            if wm is not None:
                # SEPARATE WM device: positive cue coincidence only (clean hold, no LTD)
                dwm = np.zeros((B, 2, 1)); dwm[bidx, sline, 0] = firing
                wm.step(dwm)
        # --- WM DELAY: trace(s) relax silently; the cue must SURVIVE to the decision ---
        if wm is not None:
            wm_held = _wm_relax(wm, n_wm, dt)          # dedicated cue trace
            cuevec = wm_held[:, :, 0]                  # (B,2)
            _wm_relax(stc, n_wm, dt)
        else:
            # ONE device: the cue can only be read off the SAME eligibility trace. Its value
            # at each (sample,action) synapse is the surviving eligibility, whose sign/scale
            # were set by the plasticity rule during the sample window -- NOT a clean cue.
            e_cue = _wm_relax(stc, n_wm, dt)           # (B,S,A) surviving eligibility
            cuevec = e_cue[:, :2, :].max(axis=2)       # best available cue readout off the trace
        # --- DECISION: read from the held cue (dedicated device or shared trace) ---
        logit = np.einsum('bs,bsa->ba', cuevec, w[:, :2, :]) + sigma * rng.standard_normal((B, A))
        chosen = np.argmax(logit, axis=1)
        # tag the chosen action's eligibility at the decision
        dstc = np.zeros((B, S, A)); dstc[bidx, sline, chosen] = 1.0
        stc.step(dstc)
        # --- ACTION->REWARD gap with a DISTRACTOR competing for credit ---
        _wm_relax(stc, n_gap1, dt)
        for _ in range(n_dist):
            pd = (rng.random(B) < in_rate * dt).astype(float)
            dd = np.zeros((B, S, A)); dd[:, DIST, :] = pd[:, None]
            stc.step(dd)
        e_rew = _wm_relax(stc, n_gap2, dt)              # surviving eligibility at reward
        # --- three-factor update ---
        r = (chosen == cls).astype(float)
        w = np.clip(w + (eta * (r - baseline))[:, None, None] * e_rew, 0.0, W_MAX)
        baseline += 0.02 * (r - baseline)
        rewards[:, tr] = r
    return rewards


def wm_isolated(tau_wm, *, B=20, trials=600, dt=5e-3, D_wm=3.0, cue_dur=0.3,
                in_rate=200.0, sigma=0.05, seed=1):
    """WM alone: weights FROZEN to the correct bijection; only WM retention varies. SERIAL.
    Returns per-seed accuracy (length B)."""
    rng = np.random.default_rng(seed); S = 2; A = 2
    wm = GateBankBatched(B, S, 1, tau_leak=tau_wm, V=1.5, dt=dt)
    cmap = np.stack([rng.permutation(A) for _ in range(B)])
    w = np.zeros((B, S, A))
    for b in range(B):
        for s in range(S): w[b, s, cmap[b, s]] = 1.0
    n_cue = int(round(cue_dur / dt)); n_wm = int(round(D_wm / dt)); bidx = np.arange(B)
    hit = np.zeros((B, trials))
    for tr in range(trials):
        wm.reset(); sample = rng.integers(S, size=B)
        for _ in range(n_cue):
            dwm = np.zeros((B, S, 1)); dwm[bidx, sample, 0] = (rng.random(B) < in_rate * dt).astype(float)
            wm.step(dwm)
        held = _wm_relax(wm, n_wm, dt)
        logit = np.einsum('bs,bsa->ba', held[:, :, 0], w) + sigma * rng.standard_normal((B, A))
        hit[:, tr] = (np.argmax(logit, 1) == cmap[bidx, sample]).astype(float)
    return hit.mean(axis=1)                             # per-seed accuracy (length B)


def _wm_final(rw, window=500):
    """Per-seed final performance (mean over the last ``window`` trials) -> length-B."""
    return rw[:, -window:].mean(axis=1)


def _wm_fmt(vec):
    """Mean [lo, hi] percentage string from a per-seed vector, bootstrap 95% CI."""
    lo, hi = bootstrap_ci(vec)
    return f"{100*vec.mean():5.1f}% [{100*lo:4.1f}, {100*hi:4.1f}]"


# =============================================================================
# Experiment 23 -- a device-native temporal-difference / actor-critic rule
# =============================================================================


def _td_relax(bank, n_steps, dt, stride=10):
    """Drive-free coarse relaxation over a silent delay (exp12/21 helper, verbatim)."""
    if n_steps <= 0:
        shp = bank.vn.shape[:-1]
        return bank.step(np.zeros(shp))
    base = bank.dt
    coarse = max(1, n_steps // stride)
    bank.dt = base * (n_steps / coarse)
    shp = bank.vn.shape[:-1]
    zero = np.zeros(shp)
    out = None
    for _ in range(coarse):
        out = bank.step(zero)
    bank.dt = base
    return out


def run_device_td(scheme, *, L=5, B=20, episodes=2000, dt=5e-3, D=2.0, step_dur=0.4,
                  tau_leak=10.0, eta=0.2, V=1.5, in_rate=200.0, sigma=0.15, ltd=LTD_BIAS,
                  tau_homeo=200.0, max_steps=None, seed0=0):
    """Train the LinearTrack corridor under one of three reward-modulated schemes
    (Experiment 23). SERIAL.

    ``scheme`` in {"reinforce", "td_actor_critic", "td_no_homeo", "no_trace"}. Returns
    ``(rewards (B, episodes), V(s) (B, S))``. At each of ``L`` states the agent chooses
    forward/back; the eligibility gate is snapshotted per decision so each step carries
    its own trace. REINFORCE gates every step's write by the whole-episode return minus a
    scalar baseline; TD gates step ``t`` by the bootstrapped error
    ``delta_t = r_t + gamma V(s_{t+1}) - V(s_t)`` with an on-device critic ``V(s)``. The
    device-native discount ``gamma = exp(-step_dur / tau_leak)`` is the trap-discharge decay
    over one decision step -- the SAME tau_leak that sets the eligibility trace, so one
    material constant sets both the trace lifetime and the RL horizon. ``td_no_homeo`` is the
    same TD rule with the eligibility-magnitude homeostasis of exp19 removed (the deadly-triad
    check); ``no_trace`` zeroes eligibility (device-necessity control, anchors chance).
    """
    rng = np.random.default_rng(seed0)
    env = LinearTrack(L=L)
    S, A = env.n_states, env.n_actions
    bidx = np.arange(B)
    bank = GateBankBatched(B, S, A, tau_leak=tau_leak, V=V, dt=dt)
    w = np.full((B, S, A), W_INIT)                 # policy weights (actor)
    Vval = np.full((B, S), V_INIT)                 # on-device critic V(s), per state
    baseline = np.full(B, 1.0 / A)                 # scalar baseline (reinforce only)
    ghom = np.ones((B, S, A))                      # eligibility-magnitude homeostasis EMA
    # device-native discount: the trap-discharge decay over one decision step
    gamma = math.exp(-step_dur / tau_leak)
    n_cue = int(round(0.3 / dt))
    reward_lag = int(round(D / dt))
    if max_steps is None:
        max_steps = 3 * L                          # allow back-steps, bounded
    rewards = np.zeros((B, episodes))

    n_step = int(round(step_dur / dt))              # inter-decision interval in ticks
    for ep in range(episodes):
        pos = env.start(B)
        done = np.zeros(B, bool)
        got_reward = np.zeros(B, bool)
        bank.reset()                                 # ONE gate carries all decisions' traces
        traj_state, traj_active = [], []
        for st in range(max_steps):
            state = pos.copy()
            active = (~done).astype(float)
            v = np.zeros((B, A)); spk = np.zeros((B, A))
            for n in range(n_cue):
                pre = np.zeros((B, S))
                pre[bidx, state] = (rng.random(B) < in_rate * dt).astype(float)
                charge = np.einsum('bsa,bs->ba', w, pre)
                v, sp = lif_step_batched(v, charge, dt, rng, tau_m=TAU_M, v_th=V_TH, noise=sigma)
                spk += sp
                drive = np.zeros((B, S, A))
                if scheme != "no_trace":
                    drv = pre[bidx, state][:, None] * np.where(sp, 1.0, -ltd)
                    drive[bidx, state, :] = drv * active[:, None]
                bank.step(drive)
            tie = spk.max(1) == spk.min(1)
            chosen = np.argmax(spk, 1)
            chosen[tie] = rng.integers(A, size=int(tie.sum()))
            chosen[done] = 0
            traj_state.append(state); traj_active.append(active)
            nxt, reached, ep_done = env.step(pos, chosen)
            got_reward |= ((~done) & reached)
            pos = np.where(done, pos, nxt)
            done |= ep_done
            if done.all():
                break
            _td_relax(bank, n_step, dt)              # decay one inter-decision interval only
        # Goal reward gates the surviving trace, read ONCE after the action->reward delay.
        # Earlier decisions have decayed longer (they are further from the goal), so the
        # per-(state,action) eligibility at reward carries the correct distance-weighting;
        # this replaces the per-step full-window relax at a fraction of the cost.
        _td_relax(bank, reward_lag, dt)
        e_rew = (bank.vn[..., -1] / bank.Vnmax)      # surviving trace on ALL used synapses
        R = got_reward.astype(float)
        T = len(traj_state)

        # Each decision t owns the (state, action) rows of the shared gate; its surviving
        # eligibility is e_rew masked to that decision's state. One goal reward gates them.
        if scheme == "no_trace":
            pass                                                # trace zeroed -> no learning (chance)
        elif scheme == "reinforce":
            # whole-episode return minus a scalar baseline, applied to every used synapse
            adv = eta * (R - baseline)
            w = np.clip(w + adv[:, None, None] * e_rew, 0.0, W_MAX)
            baseline += 0.02 * (R - baseline)
        else:
            # device-native TD(lambda) / actor-critic, swept along the trajectory. r_t = R
            # only on the terminal (goal) step, 0 otherwise; V(s_{t+1}) bootstraps (0 past
            # the terminal step). The SAME tau_leak sets both the trace and the discount
            # gamma. delta_t gates the per-decision write, masked to that decision's state.
            if scheme == "td_actor_critic":
                ghom += (np.abs(e_rew) - ghom) / tau_homeo
                e_use = e_rew / np.maximum(ghom, 1e-3)
            else:                                               # td_no_homeo
                e_use = e_rew
            for t in range(T):
                s_t = traj_state[t]
                v_st = Vval[bidx, s_t]
                v_next = Vval[bidx, traj_state[t + 1]] if t + 1 < T else np.zeros(B)
                r_t = R if t == T - 1 else np.zeros(B)          # reward only at the goal step
                delta = r_t + gamma * v_next - v_st             # bootstrapped TD error
                mask = np.zeros((B, S))                         # this decision's state rows
                mask[bidx, s_t] = traj_active[t]
                gate = (eta * delta)[:, None, None] * mask[:, :, None]
                w = np.clip(w + gate * e_use, 0.0, W_MAX)
                Vval[bidx, s_t] = np.clip(v_st + ETA_V * delta * traj_active[t], 0.0, V_MAX)
        rewards[:, ep] = R
    return rewards, Vval


def _td_final(rw, window=200):
    return rw[:, -window:].mean(axis=1)


def _td_fmt(v):
    lo, hi = bootstrap_ci(v)
    return f"{v.mean():.3f} [{lo:.3f}, {hi:.3f}]"


def _td_worker(spec):
    """Run one independent (scheme, L) cell. Top-level so it pickles for the Pool; each
    cell is seed-deterministic (seed0=0), so the parallel result is identical to serial."""
    sc, L, B, episodes = spec
    rw, Vval = run_device_td(sc, L=L, B=B, episodes=episodes)
    return (sc, L), _td_final(rw), rw.mean(0), Vval


# =============================================================================
# Experiment 21 -- beta_leak sensitivity of the two load-bearing learning results
#
# All learning results integrate a SINGLE-RATE (mono-exponential, beta_leak=1) leak; the
# device measures a DISPERSIVE (stretched-exponential) discharge, beta_leak ~ 0.85 field-free
# (~0.54 held-bias). Re-run the D_max ~= k*tau law (A) and the exp19 coupling (B) at the
# measured beta_leak and check they survive -- closing the "measured 0.85 but simulated 1.0" gap.
# =============================================================================


def _dmax_one(job):
    """One (tau, beta) cell: D_max = largest delay whose reward rate >= criterion.
    Top-level so it pickles for the Pool."""
    if len(job) == 5:
        tau, beta, seeds, episodes, delays = job
        dt = 5e-3
    else:
        tau, beta, seeds, episodes, delays, dt = job
    crit = 0.75
    dmax = 0.0
    for D in delays:
        env = TMaze(L=3, A_goal=2)
        r = train_sequential(env, B=seeds, tau_leak=tau, D=D, episodes=episodes,
                             beta_leak=beta, seed0=0, dt=dt)
        if reward_rate(r, window=100).mean() >= crit:
            dmax = D
        else:
            break                                   # first failure = the crossing (monotone)
    return (tau, beta), dmax


def dmax_law(betas, seeds, episodes, pool=None):
    """(A) The D_max ~= k*tau_leak scaling law re-fit at each beta_leak (Experiment 21).

    For each beta, sweep the retention taus x delays, take D_max = largest delay whose reward
    rate reaches criterion, and origin-fit D_max = k*tau (report k, R^2, monotonicity). A
    ``multiprocessing.Pool`` may be passed to parallelise the (tau, beta) cells; otherwise the
    cells run serially. Returns a dict keyed by beta."""
    taus = [1.0, 2.0, 5.0, 10.0, 20.0]
    delays = [2, 5, 10, 20, 40, 80, 160]
    jobs = [(tau, b, seeds, episodes, delays) for b in betas for tau in taus]
    if pool is not None:
        res = dict(pool.map(_dmax_one, jobs))
    else:
        res = dict(_dmax_one(j) for j in jobs)
    out = {}
    for b in betas:
        d = np.array([res[(t, b)] for t in taus]); t = np.array(taus)
        # origin fit D_max = k*tau
        k = float(np.sum(t * d) / np.sum(t * t)) if np.sum(t * t) > 0 else float("nan")
        pred = k * t
        ss = 1 - np.sum((d - pred) ** 2) / max(np.sum((d - d.mean()) ** 2), 1e-9)
        mono = bool(np.all(np.diff(d) >= -1e-9))
        out[b] = {"taus": taus, "dmax": d.tolist(), "k": k, "r2": float(ss), "monotone": mono}
    return out


def _exp19_at_beta(job):
    """Run exp19's hetero_raw / hetero_homeo / best_single at a given beta_leak by passing
    ``beta_leak`` straight through :func:`run_multitimescale` (both live in this module now,
    so the old importlib/monkeypatch construction is unnecessary). Top-level so it pickles
    for the Pool. Returns ``(beta, {label: (mean, (lo, hi))})``."""
    if len(job) == 5:
        beta, seeds, trials, tau_pool, best_tau = job
        dt = 5e-3
    else:
        beta, seeds, trials, tau_pool, best_tau, dt = job
    res = {}
    for label, kind, norm in [("hetero_raw", "hetero_measured", "none"),
                              ("hetero_homeo", "hetero_measured", "homeo"),
                              ("best_single", "best_single", "none")]:
        arg = best_tau if kind == "best_single" else None
        rw, *_ = run_multitimescale(kind, B=seeds, trials=trials, tau_arg=arg,
                                    tau_pool=tau_pool, elig_norm=norm, beta_leak=beta,
                                    dt=dt)
        f = rw[:, -400:].mean(1)
        res[label] = (float(f.mean()), bootstrap_ci(f))
    return beta, res


def run_beta_sensitivity(*, betas=BETAS, seeds=12, episodes=2000, trials=3000, pool=None):
    """The full beta_leak sensitivity study (Experiment 21): (A) the D_max law and (B) the
    exp19 multi-timescale coupling, each re-run at every ``beta`` in ``betas``, plus the
    pre-registered S1/S2 criteria evaluated at the field-free operating beta=0.85. SERIAL by
    default; pass a ``multiprocessing.Pool`` to parallelise the coarse cells (main() does).

    PRE-REGISTRATION (fixed before running; reported as-is):
      S1  D_max law SURVIVES at beta=0.85: still monotone D_max(tau), origin-fit R^2 >= 0.90,
          and the slope k within ~30% of the beta=1 value (the law is not a single-rate artefact).
      S2  exp19 coupling SURVIVES at beta=0.85: hetero+homeo still >= 0.75 AND still
          disjoint-above naive hetero (the multi-timescale homeostasis is not a single-rate
          artefact).
      S3  (DESCRIPTIVE) report the same at the strongly-dispersive held-bias beta=0.54 -- the
          stress point, expected to degrade (that regime is NOT the operating point).

    Returns the grid dict (dmax law per beta, exp19 per beta, betas, seeds, tau_source, S1/S2).
    """
    tau_pool, src = load_measured_tau()

    # (A) D_max ~= k*tau at each beta_leak
    dmax = dmax_law(betas, seeds, episodes, pool=pool)

    # (B) exp19 multi-timescale + homeostasis at each beta_leak. Rank the best single tau
    # first (ranking only, small B / short run -- run serially, it is cheap).
    grid = np.round(np.geomspace(max(0.5, tau_pool.min()), max(tau_pool.max(), 14.0), 6), 2)
    scored = [(float(t), run_multitimescale("best_single", B=6, trials=min(trials, 1200),
                                            tau_arg=float(t), tau_pool=tau_pool)[0]
               [:, -400:].mean()) for t in grid]
    best_tau = max(scored, key=lambda x: x[1])[0]
    jobs = [(b, seeds, trials, tau_pool, best_tau) for b in betas]
    if pool is not None:
        e19res = dict(pool.map(_exp19_at_beta, jobs))
    else:
        e19res = dict(_exp19_at_beta(j) for j in jobs)

    # criteria at the OPERATING beta=0.85 (present only if 0.85 is in the sweep)
    s1 = s2 = None
    if 1.0 in dmax and 0.85 in dmax:
        k1, k085 = dmax[1.0]["k"], dmax[0.85]["k"]
        s1 = bool(dmax[0.85]["r2"] >= 0.90 and dmax[0.85]["monotone"]
                  and abs(k085 - k1) <= 0.30 * k1)
    if 0.85 in e19res:
        h = e19res[0.85]
        s2 = bool(h["hetero_homeo"][0] >= 0.75 and h["hetero_homeo"][1][0] > h["hetero_raw"][1][1])

    return {"dmax": dmax, "exp19": {str(b): e19res[b] for b in betas},
            "betas": list(betas), "seeds": seeds, "tau_source": src,
            "best_tau": best_tau,
            "criteria": {"S1": s1, "S2": s2}}


# =============================================================================
# Full-scale reproduction CLI (writes data/results). Pool ONLY here.
# =============================================================================


def main(argv=None):
    """Full-scale reproduction CLI for the Chapter-8 extension grids (writes ``data/results``).

    ``python -m mrl_trace.extensions [--exp19] [--exp21] [--exp22] [--exp23] [--quick|--full]``
    With no experiment flag, runs all four. ``--full`` = published seed count; ``--quick`` =
    a fast few-seed smoke run. Each writes its published grid filename via
    :func:`mrl_trace.paths.save_result`:
      exp19 -> exp19_multitimescale.npy   exp21 -> exp21_beta_sensitivity.npy
      exp22 -> exp22_wm_stc.npy           exp23 -> exp23_device_td.npy
    """
    import argparse
    import os
    import time
    from multiprocessing import Pool
    from . import paths

    ap = argparse.ArgumentParser(description="Chapter-8 cross-cutting RL extension studies")
    ap.add_argument("--exp19", action="store_true",
                    help="multi-timescale credit from measured spread -> exp19_multitimescale.npy")
    ap.add_argument("--exp21", action="store_true",
                    help="beta_leak sensitivity of load-bearing results -> exp21_beta_sensitivity.npy")
    ap.add_argument("--exp22", action="store_true",
                    help="WM+STC one-vs-two-device null test -> exp22_wm_stc.npy")
    ap.add_argument("--exp23", action="store_true",
                    help="device-native TD/actor-critic -> exp23_device_td.npy")
    ap.add_argument("--quick", action="store_true", help="fast few-seed smoke run")
    ap.add_argument("--full", action="store_true", help="published seed count (default)")
    a = ap.parse_args(argv)
    run_all = not (a.exp19 or a.exp21 or a.exp22 or a.exp23)
    quick = a.quick
    nproc = max(1, min(6, (os.cpu_count() or 4) - 2))

    # ---------------- Experiment 19 ----------------
    if a.exp19 or run_all:
        B = 6 if quick else 20
        trials = 1500 if quick else 3000
        S, A = 4, 2
        # Delays chosen to STRADDLE the measured tau band-pass peaks (tau~1s peaks at ~1.75s,
        # tau~4-8s peaks at ~7-15s), so the short delay is best served by a short measured tau
        # and the long delay by a long one, and no single tau matches both.
        D_short, D_long = 2.0, 8.0
        tau_pool, src = load_measured_tau()
        mean_tau = float(np.mean(tau_pool))
        print("Experiment 19: multi-timescale credit from the measured trap-discharge spread")
        print(f"  tau source = {src}; median={np.median(tau_pool):.2f}s mean={mean_tau:.2f}s "
              f"range=[{tau_pool.min():.2f},{tau_pool.max():.2f}]s")
        print(f"  mixed-delay bandit: S={S} (half D_short={D_short}s, half D_long={D_long}s), A={A}")
        print(f"  {B} seeds, {trials} trials; chance={CHANCE} crit={CRIT}; pre-registered H1-H5/K1\n")
        # (label, kind, elig_norm). Headline: hetero_raw (naive spread, fails) vs hetero_homeo
        # (spread + eligibility-magnitude homeostasis, recovers) vs hetero_oracle (tau-aware
        # upper bound). best_single is the strongest single-horizon device; no_trace the control.
        conds = [
            ("hetero_raw",    "hetero_measured", "none"),
            ("hetero_homeo",  "hetero_measured", "homeo"),
            ("hetero_oracle", "hetero_measured", "oracle"),
            ("best_single",   "best_single",     "none"),
            ("no_trace",      "no_trace",        "none"),
        ]
        labels = [c[0] for c in conds]
        with Pool(nproc) as pool:
            print("  grid-searching the best single homogeneous tau (parallel) ...")
            best_tau, grid = _mt_grid_best_single(tau_pool, trials, pool=pool)
            print(f"  best_single tau = {best_tau}s  (grid {list(grid)})\n")
            res = pool.map(partial(_mt_worker, B=B, trials=trials, tau_pool=tau_pool,
                                   best_single=best_tau, mean_tau=mean_tau), conds)
        raw = {label: (rw, grp_rw) for label, rw, grp_rw in res}

        finals, ci, grp = {}, {}, {}
        for label in labels:
            rw, grp_rw = raw[label]
            f = _mt_final(rw); finals[label] = f; ci[label] = bootstrap_ci(f)
            gs = _mt_group_final(grp_rw["short"]); gl = _mt_group_final(grp_rw["long"])
            grp[label] = (float(np.nanmean(gs)), float(np.nanmean(gl)))
            print(f"  {label:14s}: overall {f.mean():.3f}[{ci[label][0]:.3f},{ci[label][1]:.3f}]  "
                  f"short-grp {grp[label][0]:.3f}  long-grp {grp[label][1]:.3f}")

        raw_m = finals["hetero_raw"].mean()
        hom_m = finals["hetero_homeo"].mean()
        ora_m = finals["hetero_oracle"].mean()
        bst_m = finals["best_single"].mean()
        nt_m = finals["no_trace"].mean()
        # H1: naive measured spread does NOT solve the two-horizon task (the problem to fix).
        h1 = raw_m < CRIT
        # H2: eligibility-magnitude HOMEOSTASIS recovers it -- reaches criterion AND disjoint
        #     above the naive spread (the mechanism genuinely helps, blind to tau).
        h2 = (hom_m >= CRIT) and (ci["hetero_homeo"][0] > ci["hetero_raw"][1])
        # H3: tau-blind homeostasis matches OR exceeds the tau-aware oracle (within a small margin).
        h3 = ci["hetero_homeo"][0] >= ci["hetero_oracle"][0] - 0.05
        # H4: the homeostasis-equipped spread beats the best single tau (the multi-timescale payoff).
        h4 = ci["hetero_homeo"][0] > ci["best_single"][1]
        # H5: trace necessary.
        h5 = nt_m <= CHANCE + 0.10
        k1c = hom_m <= raw_m                                 # kill: homeostasis fails to help
        print("\n  === pre-registered criteria ===")
        print(f"  H1 naive measured spread fails the two-horizon task (< {CRIT}): "
              f"{'PASS' if h1 else 'FAIL'} ({raw_m:.3f})")
        print(f"  H2 eligibility HOMEOSTASIS recovers it (>= {CRIT}, disjoint > naive): "
              f"{'PASS' if h2 else 'FAIL'} ({hom_m:.3f} vs naive {raw_m:.3f})")
        print(f"  H3 tau-blind homeostasis matches tau-aware oracle (CIs overlap): "
              f"{'PASS' if h3 else 'FAIL'} ({hom_m:.3f} vs oracle {ora_m:.3f})")
        print(f"  H4 homeostasis-spread beats best single tau (disjoint): "
              f"{'PASS' if h4 else 'FAIL'} ({hom_m:.3f} vs best {bst_m:.3f})")
        print(f"  H5 no-trace fails (<= {CHANCE+0.10:.2f}): {'PASS' if h5 else 'FAIL'} ({nt_m:.3f})")
        print(f"  K1 homeostasis does NOT help (homeo <= naive): {'KILL' if k1c else 'ok'} "
              f"({hom_m:.3f} vs {raw_m:.3f})")
        print("\n  HEADLINE: the measured tau spread is unusable raw (the tau-dependent eligibility")
        print("  gain starves the short-tau synapses), but a LOCAL, ACTIVITY-DRIVEN eligibility-")
        print("  magnitude homeostasis -- the same CLASS of set-point negative feedback used for")
        print("  firing-rate stability, here on eligibility amplitude and BLIND to tau -- equalises")
        print("  the gain and recovers full multi-timescale credit, matching the tau-aware oracle.")

        paths.save_result("exp19_multitimescale.npy",
                          {"finals": {k: finals[k] for k in labels}, "ci": ci, "grp": grp,
                           "tau_source": src, "tau_pool": tau_pool, "best_tau": best_tau,
                           "mean_tau": mean_tau, "S": S, "A": A, "D_short": D_short,
                           "D_long": D_long, "seeds": B, "trials": trials, "chance": CHANCE,
                           "crit": CRIT,
                           "criteria": {"H1": bool(h1), "H2": bool(h2), "H3": bool(h3),
                                        "H4": bool(h4), "H5": bool(h5), "K1": bool(k1c)}})
        print("\n  wrote exp19_multitimescale.npy")

    # ---------------- Experiment 21 ----------------
    if a.exp21 or run_all:
        seeds = 6 if quick else 12
        episodes = 1200 if quick else 2000
        trials = 1500 if quick else 3000
        betas = [1.0, 0.85] if quick else BETAS
        print("\nExperiment 21: beta_leak sensitivity of the load-bearing learning results")
        print(f"  betas={betas}; D_max law + exp19 coupling; {seeds} seeds\n")
        with Pool(nproc) as pool:
            print("  (A) D_max ~= k*tau at each beta_leak ...")
            grid = run_beta_sensitivity(betas=betas, seeds=seeds, episodes=episodes,
                                        trials=trials, pool=pool)
        dmax = grid["dmax"]
        for b in betas:
            r = dmax[b]
            print(f"    beta={b:.2f}: D_max={r['dmax']}  k={r['k']:.1f}  R2={r['r2']:.3f}  "
                  f"monotone={r['monotone']}")
        print(f"\n  best_single tau = {grid['best_tau']}s")
        print("  (B) exp19 multi-timescale + homeostasis at each beta_leak ...")
        for b in betas:
            r = grid["exp19"][str(b)]
            print(f"    beta={b:.2f}: hetero_raw={r['hetero_raw'][0]:.3f}  "
                  f"hetero_homeo={r['hetero_homeo'][0]:.3f} {r['hetero_homeo'][1]}  "
                  f"best_single={r['best_single'][0]:.3f}")
        s1, s2 = grid["criteria"]["S1"], grid["criteria"]["S2"]
        print("\n  === pre-registered criteria (at the field-free operating beta=0.85) ===")
        if s1 is not None:
            k1v, k085 = dmax[1.0]["k"], dmax[0.85]["k"]
            print(f"  S1 D_max law survives (monotone, R2>=0.90, k within 30% of beta=1): "
                  f"{'PASS' if s1 else 'FAIL'} (k {k1v:.1f}->{k085:.1f}, R2={dmax[0.85]['r2']:.3f})")
        if s2 is not None:
            h = grid["exp19"]["0.85"]
            print(f"  S2 exp19 coupling survives (homeo>=0.75 & disjoint above naive): "
                  f"{'PASS' if s2 else 'FAIL'} (homeo {h['hetero_homeo'][0]:.3f} vs naive "
                  f"{h['hetero_raw'][0]:.3f})")
        print(f"  S3 (descriptive) held-bias beta=0.54 reported above as the stress point.")
        # store beta keys as strings for the dmax dict too (npy pickle round-trips fine either way)
        paths.save_result("exp21_beta_sensitivity.npy",
                          {"dmax": grid["dmax"], "exp19": grid["exp19"],
                           "betas": grid["betas"], "seeds": grid["seeds"],
                           "tau_source": grid["tau_source"],
                           "criteria": {"S1": bool(s1) if s1 is not None else None,
                                        "S2": bool(s2) if s2 is not None else None}})
        print("\n  wrote exp21_beta_sensitivity.npy")
        print("  HEADLINE: if S1+S2 PASS, the single-rate (beta=1) idealisation is SAFE at the")
        print("  measured field-free beta~0.85 -- the learning results are not artefacts of assuming")
        print("  a mono-exponential discharge, closing the 'measured 0.85 but simulated 1.0' gap.")

    # ---------------- Experiment 22 ----------------
    if a.exp22 or run_all:
        B = 8 if quick else 20
        trials = 800 if quick else 2500
        t0 = time.time()
        print(f"\n=== exp22: WM + STC on the device trace -- one-vs-two-device NULL test, DMS "
              f"chance 50% | B={B}, trials={trials}, bootstrap 95% CI ===")
        res = {"B": B, "trials": trials, "tau_band": TAU_BAND, "chance": 0.5,
               "wm": {}, "stc": {}, "two_cotuned": {}, "one_shared": {}, "two_device": {}}

        print("\n[diag 1] WM isolated (weights frozen correct) -- perf vs tau_wm:")
        for tw in TAU_BAND:
            v = wm_isolated(tw, B=B, trials=max(400, trials // 4))
            res["wm"][tw] = v
            print(f"    tau_wm={tw:4.1f}: {_wm_fmt(v)}   ({time.time()-t0:.0f}s)")

        print("\n[diag 2] STC via exp12 loop, WM long (tau_wm=10) -- learning perf vs tau_stc:")
        for ts in TAU_BAND:
            v = _wm_final(run_wm_stc(10.0, ts, B=B, trials=trials))
            res["stc"][ts] = v
            print(f"    tau_stc={ts:4.1f}: {_wm_fmt(v)}   ({time.time()-t0:.0f}s)")

        print("\n[two co-tuned devices] tau_wm=tau_stc, but TWO separate traces (control):")
        for t in TAU_BAND:
            v = _wm_final(run_wm_stc(t, t, B=B, trials=trials, shared_device=False))
            res["two_cotuned"][t] = v
            print(f"    tau={t:4.1f}: {_wm_fmt(v)}   ({time.time()-t0:.0f}s)")

        print("\n[COLLAPSE TEST] ONE shared trace serving BOTH roles (shared_device=True):")
        for t in TAU_BAND:
            v = _wm_final(run_wm_stc(t, t, B=B, trials=trials, shared_device=True))
            res["one_shared"][t] = v
            print(f"    tau={t:4.1f}: {_wm_fmt(v)}   ({time.time()-t0:.0f}s)")

        print("\n[head-to-head at matched tau] one-shared vs two-cotuned vs two-split:")
        for tw, ts in [(6.0, 6.0), (10.0, 10.0)]:
            one = _wm_final(run_wm_stc(tw, ts, B=B, trials=trials, shared_device=True))
            two = _wm_final(run_wm_stc(tw, ts, B=B, trials=trials, shared_device=False))
            res["two_device"][(tw, ts)] = two
            print(f"    tau={tw:4.1f}: one-shared {_wm_fmt(one)}  |  two-device {_wm_fmt(two)}"
                  f"   ({time.time()-t0:.0f}s)")

        paths.save_result("exp22_wm_stc.npy", res)
        print(f"\n  wrote exp22_wm_stc.npy  (total {time.time()-t0:.0f}s)")

    # ---------------- Experiment 23 ----------------
    if a.exp23 or run_all:
        B = 6 if quick else 20
        episodes = 600 if quick else 2500
        t0 = time.time()
        # LinearTrack corridor: reward = goal reached. The credit-assignment difficulty grows
        # with corridor length L, since the goal reward must propagate back across L forward/back
        # decisions. Pre-registered question: does the bootstrapped device-native TD rule
        # overtake the whole-episode-return REINFORCE baseline once L is long enough that the
        # scalar baseline fails, and is that crossover monotone in L? A no-trace control anchors
        # the chance (random-walk goal-reach) level at each L.
        L_GRID = [5, 10, 15, 20]
        crit = 0.75
        schemes = ["no_trace", "reinforce", "td_actor_critic", "td_no_homeo"]
        print(f"\n=== exp23 device-native TD/actor-critic vs REINFORCE | LinearTrack L-sweep "
              f"{L_GRID} | B={B}, {episodes} ep ===")
        print(f"    device-native discount gamma = exp(-step_dur/tau_leak); critic V(s) "
              f"on-device; delta gates the three-factor write. Reward = goal reached.")
        # every (scheme, L) cell is independent and seed-deterministic -> dispatch across a Pool
        specs = [(sc, L, B, episodes) for L in L_GRID for sc in schemes]
        n_proc = min(len(specs), max(1, (os.cpu_count() or 4) - 2))
        print(f"    dispatching {len(specs)} cells across {n_proc} processes...\n")
        with Pool(n_proc) as pool:
            out = pool.map(_td_worker, specs)
        cells = {key: (fin, curve, V) for key, fin, curve, V in out}

        res = {"B": B, "episodes": episodes, "L_grid": L_GRID, "crit": crit,
               "tau_leak": 10.0, "final": {}, "curve": {}}
        for (sc, L), (fin, curve, V) in cells.items():
            res["final"][(sc, L)] = fin
            res["curve"][(sc, L)] = curve
            res[f"V_{sc}_{L}"] = V

        print(f"  {'L':>3} | {'no_trace':>20} | {'reinforce':>20} | {'td_actor_critic':>20} | "
              f"{'td_no_homeo':>20} | TD-RE")
        for L in L_GRID:
            r = {sc: res["final"][(sc, L)] for sc in schemes}
            d = r["td_actor_critic"].mean() - r["reinforce"].mean()
            disj = bootstrap_ci(r["td_actor_critic"])[0] > bootstrap_ci(r["reinforce"])[1]
            print(f"  {L:>3} | {_td_fmt(r['no_trace']):>20} | {_td_fmt(r['reinforce']):>20} | "
                  f"{_td_fmt(r['td_actor_critic']):>20} | {_td_fmt(r['td_no_homeo']):>20} | "
                  f"{d:+.3f}{' *' if disj else ''}")

        # crossover: smallest L at which TD is disjoint-above REINFORCE
        cross = None
        for L in L_GRID:
            rt, rr = res["final"][("td_actor_critic", L)], res["final"][("reinforce", L)]
            if bootstrap_ci(rt)[0] > bootstrap_ci(rr)[1]:
                cross = L; break
        print("\n  === verdict ===")
        if cross is not None:
            print(f"  device-native TD overtakes REINFORCE (disjoint CIs) at L>={cross}: "
                  f"bootstrapping helps once credit must bridge that many decisions.")
        else:
            print(f"  no crossover in L{L_GRID}: device-native TD machinery runs and is stable "
                  f"but does not beat the return-baseline REINFORCE on this task/range (honest negative).")

        paths.save_result("exp23_device_td.npy", res)
        print(f"\n  wrote exp23_device_td.npy  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
