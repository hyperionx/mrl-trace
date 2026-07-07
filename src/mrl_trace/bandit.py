"""Contextual spiking bandit -- the closed-loop reinforcement-learning task.

This is genuine RL, not a credit-assignment microbenchmark: the network's *action*
determines the reward (closed loop), it must learn a state->action *policy*, and
success is the *reward rate* rising over trials.

Task (``S`` states x ``A`` actions):
  - ``S`` stimulus inputs encode the state; exactly one is active per trial.
  - ``A`` leaky integrate-and-fire action neurons compete; the chosen action is the
    one that spikes most over the cue epoch (membrane noise gives exploration).
  - ``S*A`` plastic device synapses ``w[state, action]``.
  - Rewarded mapping: state ``s`` -> action ``s mod A``.
  - Reward ``R in {0,1}`` delivered after delay ``D``, *contingent* on the chosen
    action matching the rewarded action for the presented state.
  - Per synapse, a signed leak-dominant pre x post coincidence drives the device
    eligibility gate; at reward, ``dw = eta (R - b) e``.

``train`` runs ``B`` seeds as a vectorised batch (leading axis ``B``) and returns a
``(B, trials)`` reward array.  ``reward_rate`` and ``trials_to_criterion`` reduce it.
The same call, with ``no_trace=True``, zeroes the eligibility (the device-necessity
control), and with ``abstract=True`` swaps the device gate for a plain leaky
integrator (the "is the device the bottleneck?" control of the scaling study).
"""
from __future__ import annotations

import numpy as np

from .device import tau_r, tau_d, K_STAGES
from .neurons import lif_step_batched, TAU_M, V_TH
from .learning import LTD_BIAS

__all__ = ["GateBankBatched", "AbstractTrace", "train", "reward_rate",
           "trials_to_criterion", "W_INIT", "W_MAX"]

W_INIT, W_MAX = 0.5, 1.5


class GateBankBatched:
    """Device eligibility gates for ``B`` parallel ``(S, A)`` synapse grids.

    Vectorised form of :class:`siox_eligibility.device.TransientGate` (Section 5.3.2
    physics): ``k`` cascade trap nodes plus one space-charge node per synapse, signed
    leak-dominant drive, trace bounded in ``[-Vnmax, Vnmax]`` (the LTD wing).
    """

    def __init__(self, B, S, A, tau_leak=10.0, V=0.9, k=K_STAGES, dt=5e-3, Vnmax=1.0,
                 beta_leak=1.0):
        self.B, self.S, self.A, self.k, self.dt, self.Vnmax = B, S, A, k, dt, Vnmax
        self.alpha = k / tau_r(V)
        self.tau_d = tau_d(V)
        # Dispersion of the trap-DISCHARGE. beta_leak=1 is the single-rate (mono-exponential)
        # leak used throughout the learning results; beta_leak<1 is the measured dispersive
        # (stretched-exponential) discharge, instantaneous rate (beta/tau)(t/tau)^(beta-1) where
        # t is time-since-last-drive (per synapse). beta=1 recovers 1/tau exactly (bit-identical),
        # so this is a backward-compatible sensitivity knob: the device study measures
        # beta_leak~0.85 field-free, and this lets the learning results be re-run at that value
        # to confirm the single-rate idealisation does not change them.
        self.beta_leak = float(beta_leak)
        # per-synapse discharge clock, shape (B,S,A,1) to compose with vn (B,S,A,k) and a
        # per-synapse tau_leak (1,S,A,1); only allocated when the dispersive discharge is used.
        self._t_since = np.full((B, S, A, 1), dt) if beta_leak != 1.0 else None
        # ``tau_leak`` may be a scalar (one retention for every synapse, the default) or a
        # per-synapse array of shape (S, A) -- a heterogeneous retention population, e.g.
        # sampled from the measured ITO trap-discharge distribution (exp19). An (S, A) array
        # is reshaped to (1, S, A, 1) so it broadcasts against the cascade state vn of shape
        # (B, S, A, k) in step(); a scalar broadcasts trivially. Backward-compatible: a
        # scalar reproduces the original behaviour bit-for-bit.
        tl = np.asarray(tau_leak, dtype=float)
        self.tau_leak = tl.reshape(1, S, A, 1) if tl.ndim == 2 else tl
        self.vn = np.zeros((B, S, A, k))
        self.vsc = np.zeros((B, S, A))

    def reset(self):
        self.vn[:] = 0.0
        self.vsc[:] = 0.0
        if self._t_since is not None:
            self._t_since[:] = self.dt

    def step(self, drive):                 # drive: (B, S, A) signed
        # Vectorised over the k cascade stages. Each stage j advances from the PREVIOUS
        # stage's OLD value (the input ``drive`` for j=0, ``vn[...,j-1]`` for j>0), so all
        # stages update from ``self.vn`` in one fused forward-Euler expression rather than a
        # Python loop with a per-stage write into a full-array copy. Verified bit-for-bit
        # identical to the staged loop (and against the gate behaviour tests). NOTE: this is
        # a readability/structure win, not a speed one -- profiling shows the cost is the
        # memory-bandwidth-bound elementwise math over the (B,S,A,k) array, which is
        # irreducible at fixed dt and k; the step still dominates runtime.
        dt, a, Vm = self.dt, self.alpha, self.Vnmax
        vn = self.vn
        prev = np.empty_like(vn)
        prev[..., 0] = drive                       # stage 0 driven by the input coincidence
        prev[..., 1:] = vn[..., :-1]               # stage j>0 driven by its old upstream node
        if self.beta_leak == 1.0:
            leak_rate = 1.0 / self.tau_leak                       # single-rate (default)
        else:
            # dispersive stretched-exponential discharge: per-synapse instantaneous rate
            # (beta/tau)(t_since/tau)^(beta-1), broadcast over the k stages. A coincidence on a
            # synapse (|drive|>0) resets that synapse's discharge clock to dt. Work in the
            # (B,S,A,1) shape so it composes with both a scalar tau_leak and a per-synapse
            # (1,S,A,1) tau_leak array.
            self._t_since = np.where(np.abs(drive)[..., None] > 1e-9, dt, self._t_since + dt)
            tau = self.tau_leak                                   # scalar or (1,S,A,1)
            leak_rate = (self.beta_leak / tau) * np.power(
                np.clip(self._t_since / tau, 1e-6, None), self.beta_leak - 1.0)  # (B,S,A,1)
        new = vn + dt * (a * prev * (Vm - np.abs(vn)) - vn * leak_rate)
        np.clip(new, -Vm, Vm, out=new)
        self.vsc += dt * (a * drive * (Vm - np.abs(self.vsc)) - self.vsc / self.tau_d)
        self.vn = new
        return new[..., -1] / Vm


class AbstractTrace:
    """Control: an abstract exponential eligibility trace (the hand-set kernel of
    prior algorithmic work), matched in decay timescale to the device ``tau_leak``.
    Same signed drive and readout interface as :class:`GateBankBatched`; only the
    dynamics differ (a single leaky integrator, no sigmoidal rise)."""

    def __init__(self, B, S, A, tau_elig=10.0, dt=5e-3, **_ignored):
        self.e = np.zeros((B, S, A))
        self.dt = dt
        self.tau = tau_elig

    def reset(self):
        self.e[:] = 0.0

    def step(self, drive):
        self.e += self.dt * (drive - self.e / self.tau)
        return np.clip(self.e, -1.0, 1.0)


def train(S, A, *, B=6, tau_leak=10.0, D=2.0, trials=1500, dt=5e-3, cue_dur=1.0,
          eta=0.2, in_rate=200.0, ltd=LTD_BIAS, tau_m=TAU_M, v_th=V_TH,
          sigma0=0.15, sigma1=None, abstract=False, no_trace=False, seed0=0,
          reward_pools=None):
    """Train the contextual bandit on ``B`` parallel seeds; return rewards ``(B, trials)``.

    Parameters mirror the manuscript's experiments:
      ``D``         action->reward delay (s);
      ``tau_leak``  device retention / credit-assignment window (s);
      ``sigma0``    membrane-noise (exploration) level; if ``sigma1`` is given the
                    noise is annealed linearly ``sigma0 -> sigma1`` over training;
      ``abstract``  use :class:`AbstractTrace` instead of the device gate;
      ``no_trace``  zero the eligibility every step (device-necessity control).
      ``reward_pools`` biosignal reward: if given, a dict {1: array, 0: array} of
                    per-trial reward VALUES decoded from real EEG (out-of-fold
                    predictions, keyed by the TRUE outcome valence). On each trial the
                    reward actually delivered to the three-factor rule is sampled from
                    the pool matching the true outcome, so R is a real, noisy,
                    biologically-measured reward-prediction-error rather than the clean
                    synthetic {0,1}. The reported reward rate still uses the TRUE outcome
                    (task performance), not the noisy gate. Default None = synthetic.

    Two reward streams are tracked: ``r_true`` (did the action match the rewarded
    action -- task performance, what is RETURNED) and ``r_gate`` (what actually drives
    the weight update -- synthetic = r_true, or the EEG-decoded value when
    ``reward_pools`` is given).
    """
    rng = np.random.default_rng(seed0)
    Bank = AbstractTrace if abstract else GateBankBatched
    bank = Bank(B, S, A, tau_leak=tau_leak, dt=dt)
    w = np.full((B, S, A), W_INIT)
    correct = np.array([s % A for s in range(S)])      # rewarded action per state
    baseline = np.full(B, 1.0 / A)                     # baseline tracks chance for this A
    cue = (0.3, 0.3 + cue_dur)
    ri = int((cue[1] + D) / dt)
    nsteps = ri + 2
    rewards = np.zeros((B, trials))
    bidx = np.arange(B)
    for tr in range(trials):
        sigma = sigma0 if sigma1 is None else sigma0 + (sigma1 - sigma0) * tr / trials
        bank.reset()
        state = rng.integers(S, size=B)                # per-seed state this trial
        v = np.zeros((B, A))
        spk = np.zeros((B, A))
        e_rew = np.zeros((B, S, A))
        for n in range(nsteps):
            t = n * dt
            pre = np.zeros((B, S))
            if cue[0] <= t < cue[1]:
                pre[bidx, state] = (rng.random(B) < in_rate * dt).astype(float)
            charge = np.einsum('bsa,bs->ba', w, pre)   # bitline currents = G^T V
            v, sp = lif_step_batched(v, charge, dt, rng, tau_m=tau_m, v_th=v_th, noise=sigma)
            spk += sp
            drive = np.zeros((B, S, A))
            drive[bidx, state, :] = pre[bidx, state][:, None] * np.where(sp, 1.0, -ltd)
            if no_trace:
                drive[:] = 0.0
            e = bank.step(drive)
            if n == ri:
                e_rew = e.copy()
        # action selection: argmax spikes; ties (incl. all-zero) -> random
        tie = spk.max(axis=1) == spk.min(axis=1)
        chosen = np.argmax(spk, axis=1)
        chosen[tie] = rng.integers(A, size=tie.sum())
        r_true = (chosen == correct[state]).astype(float)   # task performance (returned)
        if reward_pools is None:
            r_gate = r_true                                  # synthetic clean reward
        else:
            # biosignal reward: for each seed, draw a decoded reward VALUE from the EEG
            # pool matching the TRUE outcome valence -- a real, noisy reward-prediction-
            # error gates the update, while r_true still measures task performance.
            r_gate = np.empty(B)
            for b in range(B):
                pool = reward_pools[int(r_true[b])]
                r_gate[b] = pool[rng.integers(len(pool))]
        adv = eta * (r_gate - baseline)                # signed three-factor update
        w = np.clip(w + adv[:, None, None] * e_rew, 0.0, W_MAX)
        baseline += 0.02 * (r_gate - baseline)
        rewards[:, tr] = r_true                         # report TRUE performance
    return rewards


def reward_rate(rewards, window=100):
    """Mean reward over the last ``window`` trials, per seed (axis -1)."""
    rewards = np.asarray(rewards)
    if rewards.shape[-1] < window:
        return rewards.mean(axis=-1)
    return rewards[..., -window:].mean(axis=-1)


def trials_to_criterion(rew_1d, crit, window=100):
    """First trial index at which the running ``window``-reward rate reaches ``crit``
    for a single seed's reward sequence, or ``None`` if it never does."""
    rew_1d = np.asarray(rew_1d)
    if len(rew_1d) < window:
        return None
    csum = np.cumsum(rew_1d)
    rr = (csum[window:] - csum[:-window]) / window
    if not np.any(rr >= crit):
        return None
    return int(np.argmax(rr >= crit) + window)


# =============================================================================
# Experiment cores (the closed-loop RL studies that compose ``train``)
#
# Each ``run_*`` returns the result grid as a plain dict -- no file I/O, no
# plotting, no stdout.  Notebooks call these at a small (quick) seed/trial count
# and render the figures inline; ``main()`` (below) calls them at the published
# 20-seed scale and writes the grid under ``data/results/`` for the notebooks to
# replay.  Reversal (Experiment 18) is a ``train`` variant, so it lives here.
# =============================================================================


def _mapping(S, A, shift):
    """Rewarded action per state, cyclically shifted by ``shift`` (0 = identity)."""
    return np.array([(s + shift) % A for s in range(S)])


def run_reversal(cond, *, S=2, A=2, B=20, tau_leak=10.0, D=2.0, trials=3000,
                 n_phases=2, dt=5e-3, cue_dur=1.0, eta=0.2, in_rate=200.0,
                 ltd=LTD_BIAS, tau_m=TAU_M, v_th=V_TH, sigma=0.15, seed0=0):
    """Reversal learning (Experiment 18): the signed rule must UNLEARN a stale
    contingency.  Mirrors :func:`train` exactly, but the rewarded mapping is
    cyclically shifted by one at each of ``n_phases`` equal-length phase boundaries,
    so the agent must repeatedly unlearn the stale contingency and acquire the new one.

    ``cond`` is ``"device"`` (trap-discharge gate), ``"abstract"`` (matched-tau single
    exponential, the recency-trace baseline) or ``"no_trace"`` (eligibility zeroed,
    the device-necessity control).

    Returns ``(rewards (B, trials), flips list[int])`` where ``rewards[:, t]`` is the
    TRUE task performance under the contingency in force at trial ``t``.
    """
    rng = np.random.default_rng(seed0)
    if cond == "abstract":
        bank = AbstractTrace(B, S, A, tau_elig=tau_leak, dt=dt)
    else:
        bank = GateBankBatched(B, S, A, tau_leak=tau_leak, dt=dt)
    no_trace = (cond == "no_trace")

    w = np.full((B, S, A), W_INIT)
    baseline = np.full(B, 1.0 / A)
    cue = (0.3, 0.3 + cue_dur)
    ri = int((cue[1] + D) / dt)
    nsteps = ri + 2
    rewards = np.zeros((B, trials))
    bidx = np.arange(B)

    phase_len = trials // n_phases
    flips = [phase_len * p for p in range(1, n_phases)]

    for tr in range(trials):
        phase = min(tr // phase_len, n_phases - 1)
        correct = _mapping(S, A, phase)              # cyclic-shifted mapping this phase
        bank.reset()
        state = rng.integers(S, size=B)
        v = np.zeros((B, A)); spk = np.zeros((B, A))
        e_rew = np.zeros((B, S, A))
        for n in range(nsteps):
            t = n * dt
            pre = np.zeros((B, S))
            if cue[0] <= t < cue[1]:
                pre[bidx, state] = (rng.random(B) < in_rate * dt).astype(float)
            charge = np.einsum('bsa,bs->ba', w, pre)
            v, sp = lif_step_batched(v, charge, dt, rng, tau_m=tau_m, v_th=v_th, noise=sigma)
            spk += sp
            drive = np.zeros((B, S, A))
            drive[bidx, state, :] = pre[bidx, state][:, None] * np.where(sp, 1.0, -ltd)
            if no_trace:
                drive[:] = 0.0
            e = bank.step(drive)
            if n == ri:
                e_rew = e.copy()
        tie = spk.max(axis=1) == spk.min(axis=1)
        chosen = np.argmax(spk, axis=1)
        chosen[tie] = rng.integers(A, size=int(tie.sum()))
        r_true = (chosen == correct[state]).astype(float)
        adv = eta * (r_true - baseline)
        w = np.clip(w + adv[:, None, None] * e_rew, 0.0, W_MAX)
        baseline += 0.02 * (r_true - baseline)
        rewards[:, tr] = r_true
    return rewards, flips


def reversal_phase_final(rw, flips, trials, window=200):
    """Per-seed reward rate in the last ``window`` trials of phase 1 (pre-reversal)
    and of the final phase (post-reversal)."""
    first_flip = flips[0]
    pre = rw[:, max(0, first_flip - window):first_flip].mean(1)
    post = rw[:, trials - window:].mean(1)
    return pre, post


def reversal_recriterion(rw_1d, flip, crit, window=100):
    """Trials AFTER the flip until the running reward rate first reaches ``crit`` again
    (post-flip re-acquisition speed); ``None`` if never re-acquired within the phase."""
    seg = rw_1d[flip:]
    if len(seg) < window:
        return None
    csum = np.cumsum(seg)
    rr = (csum[window:] - csum[:-window]) / window
    hit = np.argmax(rr >= crit) if np.any(rr >= crit) else None
    return int(hit + window) if hit is not None else None


def run_learning_and_window(*, seeds=20, delays=(1, 2, 5, 10, 20),
                            tau_leaks=(10.0, 2.0, 0.5), D0=5.0, trials=600):
    """Fig 6 grid (tier3): 2-state bandit learning curve + delay x retention table.

    Returns the result dict (per-seed learning curves, delay x retention reward-rate
    table with bootstrap CIs, max learnable delay per tau); no file I/O.
    """
    from .stats import bootstrap_ci
    S, A = 2, 2
    dev = train(S, A, B=seeds, tau_leak=10.0, D=D0, trials=trials)
    nt = train(S, A, B=seeds, tau_leak=10.0, D=D0, trials=trials, no_trace=True)
    dev_fr, nt_fr = reward_rate(dev), reward_rate(nt)

    rates, rates_ci = {}, {}
    for tl in tau_leaks:
        row, row_ci = [], []
        for D in delays:
            fr = reward_rate(train(S, A, B=seeds, tau_leak=tl, D=D, trials=trials))
            row.append(float(fr.mean()))
            row_ci.append(bootstrap_ci(fr))
        rates[tl] = row
        rates_ci[tl] = row_ci

    def _maxd(tl):
        ds = [d for d, v in zip(delays, rates[tl]) if v >= 0.75]
        return max(ds) if ds else 0

    def _ci(arr):
        a = np.asarray(arr, float)
        lo, hi = bootstrap_ci(a)
        return (float(a.mean()), float(a.std()), lo, hi)

    return {
        "delays": list(delays), "reward_rate": rates, "reward_rate_ci": rates_ci,
        "max_learn": {tl: _maxd(tl) for tl in rates}, "D0": D0, "n_seeds": seeds,
        "curve_device": dev, "curve_notrace": nt,
        "device_final": _ci(dev_fr), "notrace_final": _ci(nt_fr),
    }


def run_scaling(*, seeds=20, grid=((2, 2), (4, 2), (4, 4), (8, 4), (8, 8), (12, 8)),
                D=2.0):
    """Fig 8a grid (tier4): does the policy still converge as ``S x A`` grows?

    Criterion (pre-registered, no goalpost moving): reward rate >= 0.5*(1 + 1/A).
    Returns a dict keyed by ``(S, A)`` with device/no-trace means + bootstrap CIs,
    trials-to-criterion, and pass/fail; no file I/O.
    """
    from .stats import bootstrap_ci

    def _ci(arr):
        a = np.asarray(arr, float)
        lo, hi = bootstrap_ci(a)
        return (float(a.mean()), float(a.std()), lo, hi)

    results = {}
    for S, A in grid:
        chance = 1.0 / A
        crit = 0.5 * (1 + chance)
        dev = train(S, A, B=seeds, D=D)
        nt = train(S, A, B=seeds, D=D, no_trace=True)
        dev_fr, nt_fr = reward_rate(dev), reward_rate(nt)
        ttc = [trials_to_criterion(dev[b], crit) for b in range(seeds)]
        ok = [t for t in ttc if t is not None]
        results[(S, A)] = dict(
            chance=chance, crit=crit, device=_ci(dev_fr), no_trace=_ci(nt_fr),
            trials_to_crit=int(np.median(ok)) if ok else None,
            n_converged=len(ok), passed=bool(dev_fr.mean() >= crit), n_seeds=seeds)
    return results


def run_remedies(*, seeds=20):
    """Fig 8b (tier5): five remedies on the failing 8x8 case.

    Returns a dict keyed by remedy label with final reward-rate mean + bootstrap CI
    and pass/fail against the 8x8 criterion; no file I/O.
    """
    from .stats import bootstrap_ci
    S, A = 8, 8
    crit = 0.5 * (1 + 1.0 / A)
    rows = [
        ("R0 baseline (device, undirected, 1500)", dict(trials=1500)),
        ("R1 more budget (device, undirected, 6000)", dict(trials=6000)),
        ("R2 directed expl (device, sigma 0.45->0.05, 1500)",
         dict(trials=1500, sigma0=0.45, sigma1=0.05)),
        ("R3 abstract trace (NOT device, undirected, 1500)", dict(trials=1500, abstract=True)),
        ("R4 directed+budget (device, sigma anneal, 6000)",
         dict(trials=6000, sigma0=0.45, sigma1=0.05)),
    ]
    out = {}
    for name, kw in rows:
        fr = reward_rate(train(S, A, B=seeds, D=2.0, **kw))
        a = np.asarray(fr, float); lo, hi = bootstrap_ci(a)
        out[name] = dict(final=(float(a.mean()), float(a.std()), lo, hi),
                         passed=bool(a.mean() >= crit), n_seeds=seeds)
    return out


def _summarize_reversal(raw, conds, trials, crit, chance):
    """Package raw per-condition ``run_reversal`` output into the saved grid dict
    (mean curves, pre/post finals + CIs, re-acquisition cost, recovery bins,
    pre-registered H1-H4/K1)."""
    from .stats import bootstrap_ci
    flips = raw[conds[0]][1]
    B = raw[conds[0]][0].shape[0]
    curves, pre_f, post_f, ci_pre, ci_post, recrit, recovery = {}, {}, {}, {}, {}, {}, {}
    for c in conds:
        rw, fl = raw[c]
        curves[c] = rw.mean(0)
        pre, post = reversal_phase_final(rw, fl, trials)
        pre_f[c], post_f[c] = pre, post
        ci_pre[c], ci_post[c] = bootstrap_ci(pre), bootstrap_ci(post)
        tt = [reversal_recriterion(rw[b], fl[0], crit) for b in range(B)]
        solved = [x for x in tt if x is not None]
        recrit[c] = (float(np.median(solved)) if solved else None, len(solved), B)
        seg = rw[:, fl[0]:].mean(0)
        recovery[c] = [float(seg[i:i + 500].mean()) for i in range(0, len(seg), 500)]
    dev_pre, dev_post = pre_f["device"].mean(), post_f["device"].mean()
    nt_post = post_f["no_trace"].mean()
    dev_rc, abs_rc = recrit["device"][0], recrit["abstract"][0]
    ratio = (dev_rc / abs_rc) if (dev_rc and abs_rc) else float("nan")
    criteria = {"H1": bool(dev_pre >= crit), "H2": bool(dev_post >= crit),
                "H3": bool(nt_post <= chance + 0.10), "K1": bool(dev_post <= chance + 0.10)}
    return {"curves": curves, "pre": pre_f, "post": post_f, "recovery": recovery,
            "ci_pre": ci_pre, "ci_post": ci_post, "recrit": recrit, "flips": flips,
            "seeds": B, "trials": trials, "n_phases": len(flips) + 1, "S": 2, "A": 2,
            "tau_leak": 10.0, "chance": chance, "crit": crit,
            "reacq_cost": {"device": dev_rc, "abstract": abs_rc, "ratio": ratio},
            "criteria": criteria}


def main(argv=None):
    """Full-scale reproduction CLI for the bandit-family grids (writes ``data/results``).

    ``python -m siox_eligibility.bandit [--exp18] [--bandit] [--full|--quick]``
    With no experiment flag, runs all. ``--full`` = 20 seeds (published); ``--quick``
    = a fast few-seed smoke run.
    """
    import argparse
    from . import paths
    ap = argparse.ArgumentParser(description="Bandit-family RL reproductions")
    ap.add_argument("--exp18", action="store_true", help="reversal learning -> exp18_reversal.npy")
    ap.add_argument("--bandit", action="store_true",
                    help="fig6/8a/8b -> tier3/tier4/tier5_results.npy")
    ap.add_argument("--quick", action="store_true", help="fast few-seed smoke run")
    ap.add_argument("--full", action="store_true", help="published 20-seed run (default)")
    a = ap.parse_args(argv)
    run_all = not (a.exp18 or a.bandit)
    seeds = 6 if a.quick else 20

    if a.bandit or run_all:
        print(f"=== bandit reruns (N={seeds}, bootstrap 95% CIs) ===")
        paths.save_result("tier3_results.npy", run_learning_and_window(seeds=seeds))
        paths.save_result("tier4_results.npy", run_scaling(seeds=seeds))
        paths.save_result("tier5_results.npy", run_remedies(seeds=seeds))
        print("  wrote tier3/tier4/tier5_results.npy")

    if a.exp18 or run_all:
        B = 6 if a.quick else 20
        trials = 4000 if a.quick else 8000
        conds = ("device", "abstract", "no_trace")
        chance, crit = 0.5, 0.75
        print(f"=== reversal (Experiment 18): {B} seeds, {trials} trials ===")
        raw = {c: run_reversal(c, B=B, trials=trials, n_phases=2, tau_leak=10.0)
               for c in conds}
        grid = _summarize_reversal(raw, conds, trials, crit, chance)
        paths.save_result("exp18_reversal.npy", grid)
        print(f"  wrote exp18_reversal.npy  criteria={grid['criteria']}")


if __name__ == "__main__":
    main()
