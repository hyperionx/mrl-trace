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

from .device import decay_matched_exponential_tau, tau_r, K_STAGES
from .model_specs import LINEAR_MODEL_ID, PRIMARY_MODEL_ID
from .neurons import lif_step_batched, TAU_M, V_TH
from .learning import (LTD_BIAS, coincidence_drive, SIGNED_RULE_PROVENANCE,
                       THREE_FACTOR_PROVENANCE)

__all__ = ["GateBankBatched", "LinearErlangGateBankBatched", "gate_bank_class",
           "AbstractTrace",
           "train", "reward_rate",
           "trials_to_criterion", "W_INIT", "W_MAX", "run_signed_rule_ablations",
           "run_reversal", "run_reversal_grid", "calibrate_reversal_scales",
           "reversal_phase_final", "reversal_recriterion",
           "kaplan_meier_recovery_summary",
           "BANDIT_METHOD_PROVENANCE"]

W_INIT, W_MAX = 0.5, 1.5

BANDIT_METHOD_PROVENANCE = {
    "status": "proposed",
    "established_basis": [
        "contextual bandit evaluation",
        "leaky integrate-and-fire dynamics",
        "three-factor reward modulation",
    ],
    "repository_adaptation": (
        "The repository's signed coincidence drive is integrated by either the "
        "cascade eligibility surrogate or an explicit exponential control."
    ),
    "claim_limit": (
        "Results test this repository-specific rule and task implementation; they do "
        "not establish superiority over complete published learning algorithms."
    ),
}


class GateBankBatched:
    """Primary nonlinear headroom gate over ``B x S x A`` synapses."""

    model_id = PRIMARY_MODEL_ID

    def __init__(self, B, S, A, tau_leak=10.0, V=0.9, k=K_STAGES, dt=5e-3,
                 Vnmax=1.0, beta_leak=1.0, tau_r_override=None):
        tl = np.asarray(tau_leak, dtype=float)
        if (not np.isfinite(beta_leak) or beta_leak <= 0
                or not np.all(np.isfinite(tl)) or np.any(tl <= 0)
                or not np.isfinite(Vnmax) or Vnmax <= 0 or int(k) < 1):
            raise ValueError("gate scales, depth and leakage parameters must be positive")
        self.B, self.S, self.A = int(B), int(S), int(A)
        self.k, self.dt, self.Vnmax = int(k), float(dt), float(Vnmax)
        fitted_tau_r = tau_r(V) if tau_r_override is None else float(tau_r_override)
        self.alpha = self.k / fitted_tau_r
        self.beta_leak = float(beta_leak)
        self.tau_leak = tl.reshape(1, S, A, 1) if tl.ndim == 2 else tl
        self._t_since = (
            np.full((B, S, A, 1), self.dt) if beta_leak != 1.0 else None
        )
        self.vn = np.zeros((B, S, A, self.k))

    def reset(self):
        self.vn[:] = 0.0
        if self._t_since is not None:
            self._t_since[:] = self.dt

    def step(self, drive):
        drive = np.asarray(drive, dtype=float)
        if drive.shape != (self.B, self.S, self.A):
            raise ValueError(
                f"drive must have shape {(self.B, self.S, self.A)}, got {drive.shape}"
            )
        vn = self.vn
        previous_fraction = np.empty_like(vn)
        previous_fraction[..., 0] = drive
        previous_fraction[..., 1:] = vn[..., :-1] / self.Vnmax
        if self.beta_leak == 1.0:
            leak_rate = 1.0 / self.tau_leak
        else:
            self._t_since = np.where(
                np.abs(drive)[..., None] > 1e-9,
                self.dt,
                self._t_since + self.dt,
            )
            tau = self.tau_leak
            leak_rate = (self.beta_leak / tau) * np.power(
                np.clip(self._t_since / tau, 1e-6, None), self.beta_leak - 1.0
            )
        new = vn + self.dt * (
            self.alpha * previous_fraction * (self.Vnmax - np.abs(vn))
            - vn * leak_rate
        )
        self.vn = np.clip(new, -self.Vnmax, self.Vnmax)
        return self.vn[..., -1] / self.Vnmax


class LinearErlangGateBankBatched:
    """Linear Erlang-exact sensitivity over ``B x S x A`` synapses.

    Vectorised form of :class:`mrl_trace.device.LinearErlangEligibilityGate`: ``k``
    linear low-pass nodes with signed leak-dominant drive, bounded in
    ``[-Vnmax, Vnmax]``.  Without leakage, its unit-step response is the Erlang
    candidate used by model identification (up to forward-Euler error). It is a
    computational surrogate, not the complete product-current model used to fit the
    measured Au transient.
    """

    model_id = LINEAR_MODEL_ID

    def __init__(self, B, S, A, tau_leak=10.0, V=0.9, k=K_STAGES, dt=5e-3, Vnmax=1.0,
                 beta_leak=1.0, tau_r_override=None):
        self.B, self.S, self.A, self.k, self.dt, self.Vnmax = B, S, A, k, dt, Vnmax
        fitted_tau_r = tau_r(V) if tau_r_override is None else float(tau_r_override)
        self.alpha = k / fitted_tau_r
        # Dispersion of the trap discharge. beta_leak=1 is the historical single-rate
        # approximation; beta_leak<1 uses the instantaneous stretched-exponential hazard
        # (beta/tau)(t/tau)^(beta-1), with t measured since the most recent drive on each
        # synapse. This is a reset-clock, non-Markov surrogate rather than an exact KWW
        # state-space realization. Empirical-linked runs pass and record the direct
        # held-bias fit; beta=1 remains a separately labelled sensitivity.
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
        if (not np.isfinite(self.beta_leak) or self.beta_leak <= 0
                or not np.all(np.isfinite(tl)) or np.any(tl <= 0)):
            raise ValueError("beta_leak and all tau_leak values must be finite and positive")
        self.tau_leak = tl.reshape(1, S, A, 1) if tl.ndim == 2 else tl
        self.vn = np.zeros((B, S, A, k))

    def reset(self):
        self.vn[:] = 0.0
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
        # A unit input has the same scale as the internal state.  With no leakage,
        # dv_1/dt = alpha*(Vnmax*drive-v_1) and
        # dv_m/dt = alpha*(v_{m-1}-v_m), whose unit-step final stage is the Erlang CDF
        # gammainc(k, alpha*t). This is the same candidate evaluated in device fitting.
        prev[..., 0] = Vm * drive                  # stage 0 driven by input coincidence
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
        new = vn + dt * (a * (prev - vn) - vn * leak_rate)
        np.clip(new, -Vm, Vm, out=new)
        self.vn = new
        return new[..., -1] / Vm


def gate_bank_class(gate_model=PRIMARY_MODEL_ID):
    """Resolve a frozen model identity to its batched gate implementation."""
    if gate_model == PRIMARY_MODEL_ID:
        return GateBankBatched
    if gate_model == LINEAR_MODEL_ID:
        return LinearErlangGateBankBatched
    raise ValueError(f"unknown gate model {gate_model!r}")


class AbstractTrace:
    """Control: an abstract exponential eligibility trace (the hand-set kernel of
    prior algorithmic work). Publication comparators supply the time constant from
    :func:`mrl_trace.device.decay_matched_exponential_tau`.
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
          device_k=K_STAGES, tau_r_override=None,
          coincidence_mode="signed", beta_leak=1.0,
          gate_model=PRIMARY_MODEL_ID):
    """Train the contextual bandit on ``B`` parallel seeds; return rewards ``(B, trials)``.

    Parameters mirror the manuscript's experiments:
      ``D``         action->reward delay (s);
      ``tau_leak``  device retention / credit-assignment window (s);
      ``sigma0``    membrane-noise (exploration) level; if ``sigma1`` is given the
                    noise is annealed linearly ``sigma0 -> sigma1`` over training;
      ``abstract``  use :class:`AbstractTrace` instead of the device gate;
      ``no_trace``  zero the eligibility every step (device-necessity control).
      ``coincidence_mode`` selects the proposed signed drive or an unsigned/no-negative
                    ablation; it is not attributed to the cited R-STDP algorithms.

    This is a wholly synthetic closed-loop task. Recorded biosignals are evaluated
    separately by the state-free logged-replay workflow in
    :mod:`mrl_trace.dopamine_replay`; outcome-keyed resampling is intentionally not
    supported.
    """
    rng = np.random.default_rng(seed0)
    if abstract:
        matched_tau = decay_matched_exponential_tau(
            tau_leak, V=0.9, k=device_k, tau_r_override=tau_r_override,
            beta_leak=beta_leak, gate_model=gate_model,
        )
        bank = AbstractTrace(B, S, A, tau_elig=matched_tau, dt=dt)
    else:
        bank = gate_bank_class(gate_model)(
            B, S, A, tau_leak=tau_leak, dt=dt, k=device_k,
            tau_r_override=tau_r_override, beta_leak=beta_leak,
        )
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
            drive[bidx, state, :] = coincidence_drive(
                pre[bidx, state][:, None], sp, mode=coincidence_mode, ltd=ltd)
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
        adv = eta * (r_true - baseline)                # signed three-factor update
        w = np.clip(w + adv[:, None, None] * e_rew, 0.0, W_MAX)
        baseline += 0.02 * (r_true - baseline)
        rewards[:, tr] = r_true                         # report TRUE performance
    return rewards


def run_signed_rule_ablations(*, seeds=6, trials=600, **kwargs):
    """Run the proposed signed drive beside its explicit no-negative ablation."""
    modes = ("signed", "unsigned", "no_negative")
    rewards = {
        mode: train(2, 2, B=seeds, trials=trials, coincidence_mode=mode, **kwargs)
        for mode in modes
    }
    return {
        "rewards": rewards,
        "finals": {mode: reward_rate(value, window=min(100, trials))
                   for mode, value in rewards.items()},
        "retention_definition": "deliberately_swept",
        "method_provenance": BANDIT_METHOD_PROVENANCE,
        "component_provenance": {
            "three_factor": THREE_FACTOR_PROVENANCE,
            "signed_coincidence": SIGNED_RULE_PROVENANCE,
            "unsigned_control": {
                "status": "proposed",
                "established_basis": ["ablation analysis"],
                "repository_adaptation": (
                    "The magnitude of the signed drive is retained while its sign is removed."
                ),
                "claim_limit": "Ablation only; not a published R-STDP implementation.",
            },
            "no_negative_control": {
                "status": "proposed",
                "established_basis": ["coincidence-only eligibility"],
                "repository_adaptation": "Presynaptic-only negative events are set to zero.",
                "claim_limit": "Ablation only; not a published R-STDP implementation.",
            },
        },
    }


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
    csum = np.r_[0.0, np.cumsum(rew_1d)]
    rr = (csum[window:] - csum[:-window]) / window
    if not np.any(rr >= crit):
        return None
    return int(np.argmax(rr >= crit) + window)


# =============================================================================
# Experiment cores (the closed-loop RL studies that compose ``train``)
#
# Each ``run_*`` returns the result grid as a plain dict -- no file I/O, no
# plotting, no stdout. Notebooks select reduced or publication budgets and render
# figures inline; any large per-seed archive is an explicit external output rather
# than bundled evidence. Reversal is a ``train`` variant, so it lives here.
# =============================================================================


def _mapping(S, A, shift):
    """Rewarded action per state, cyclically shifted by ``shift`` (0 = identity)."""
    return np.array([(s + shift) % A for s in range(S)])


def run_reversal(cond, *, S=2, A=2, B=20, tau_leak=10.0, D=2.0, trials=3000,
                 n_phases=2, dt=5e-3, cue_dur=1.0, eta=0.2, in_rate=200.0,
                 ltd=LTD_BIAS, tau_m=TAU_M, v_th=V_TH, sigma=0.15, seed0=0,
                 device_k=K_STAGES, tau_r_override=None, beta_leak=1.0,
                 eligibility_normalizer=1.0, return_diagnostics=False,
                 gate_model=PRIMARY_MODEL_ID):
    """Reversal sensitivity with a trial-local eligibility state.

    The rewarded mapping is
    cyclically shifted by one at each of ``n_phases`` equal-length phase boundaries,
    so the agent must repeatedly unlearn the stale contingency and acquire the new one.

    ``cond`` is ``"device"`` (trap-discharge gate), ``"abstract"`` (post-peak
    decay-matched single exponential, the recency-trace baseline) or ``"no_trace"`` (eligibility zeroed,
    the device-necessity control).

    The eligibility bank is reset at the beginning of every trial. Therefore a
    retention-dependent recovery difference reflects within-trial filtering and
    effective plasticity, not eligibility carried across the reversal boundary.
    ``eligibility_normalizer`` is a frozen reward-time update RMS from a separate
    common calibration stream.

    Returns ``(rewards, flips)`` by default, or ``(rewards, flips, diagnostics)``.
    """
    if cond not in {"device", "abstract", "no_trace"}:
        raise ValueError("cond must be 'device', 'abstract', or 'no_trace'")
    eligibility_normalizer = float(eligibility_normalizer)
    if not np.isfinite(eligibility_normalizer) or eligibility_normalizer <= 0:
        raise ValueError("eligibility_normalizer must be finite and positive")
    rng = np.random.default_rng(seed0)
    matched_tau = None
    if cond == "abstract":
        matched_tau = decay_matched_exponential_tau(
            tau_leak, V=0.9, k=device_k, tau_r_override=tau_r_override,
            beta_leak=beta_leak, gate_model=gate_model,
        )
        bank = AbstractTrace(B, S, A, tau_elig=matched_tau, dt=dt)
    else:
        bank = gate_bank_class(gate_model)(
            B, S, A, tau_leak=tau_leak, dt=dt, k=device_k,
            tau_r_override=tau_r_override, beta_leak=beta_leak,
        )
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
    update_sq_sum = 0.0
    update_count = 0
    trace_peak = 0.0

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
        raw_update = (r_true - baseline)[:, None, None] * e_rew
        update_sq_sum += float(np.sum(raw_update ** 2))
        update_count += int(raw_update.size)
        trace_peak = max(trace_peak, float(np.max(np.abs(e_rew))))
        w = np.clip(
            w + eta * raw_update / eligibility_normalizer, 0.0, W_MAX
        )
        baseline += 0.02 * (r_true - baseline)
        rewards[:, tr] = r_true
    if not return_diagnostics:
        return rewards, flips
    raw_rms = float(np.sqrt(update_sq_sum / max(update_count, 1)))
    return rewards, flips, {
        "raw_trace_peak": trace_peak,
        "raw_effective_update_rms": raw_rms,
        "eligibility_normalizer": eligibility_normalizer,
        "normalized_effective_update_rms": raw_rms / eligibility_normalizer,
        "trace_reset_each_trial": True,
        "beta_leak": float(beta_leak),
        "matched_exponential_tau_s": (
            None if matched_tau is None else float(matched_tau)
        ),
        "exponential_matching_protocol": (
            "normalized device-surrogate 80--10% post-peak decay after a "
            "0.3-s standard coincidence"
        ) if cond == "abstract" else None,
        "claim_limit": (
            "Within-trial retention sensitivity; eligibility does not persist across "
            "the reversal boundary."
        ),
    }


def _reversal_job(job):
    key, kwargs = job
    return key, run_reversal(**kwargs)


def _map_reversal_jobs(jobs, workers=1, pool=None):
    if pool is not None:
        return pool.map(_reversal_job, jobs, chunksize=1)
    if int(workers) <= 1:
        return list(map(_reversal_job, jobs))
    from multiprocessing import get_context
    with get_context("spawn").Pool(min(int(workers), len(jobs))) as own_pool:
        return own_pool.map(_reversal_job, jobs, chunksize=1)


def calibrate_reversal_scales(taus, *, conditions=("device", "abstract"),
                              trajectories=256, calibration_trials=8,
                              workers=1, pool=None, **shared):
    """Freeze reward-time update RMS on one common untrained random stream.

    This scale control removes the most direct update-magnitude confound across
    retention/model conditions. It does not make the filters or algorithms
    equivalent and does not turn reversal into a cross-trial trace-memory test.
    """
    jobs = []
    for tau in (float(value) for value in taus):
        for condition in conditions:
            kwargs = dict(shared)
            kwargs.update(
                cond=condition, B=int(trajectories), tau_leak=tau,
                trials=int(calibration_trials), n_phases=1, eta=0.0,
                eligibility_normalizer=1.0, return_diagnostics=True,
            )
            jobs.append(((tau, condition), kwargs))
    mapped = _map_reversal_jobs(jobs, workers=workers, pool=pool)
    records = {}
    for key, (_, _, diagnostics) in mapped:
        raw = diagnostics["raw_effective_update_rms"]
        normalizer = raw if np.isfinite(raw) and raw > np.finfo(float).eps else 1.0
        records[key] = {
            **diagnostics,
            "eligibility_normalizer": float(normalizer),
            "normalized_effective_update_rms": float(raw / normalizer),
        }
    return {
        "protocol": "common_untrained_reward_time_update_rms",
        "trajectories": int(trajectories),
        "trials_per_trajectory": int(calibration_trials),
        "records": records,
        "claim_limit": (
            "Scale calibration only; filters remain different and eligibility is "
            "reset on every trial."
        ),
    }


def run_reversal_grid(taus, *, conditions=("device", "abstract"), workers=1,
                      pool=None, calibration_trajectories=256,
                      calibration_trials=8,
                      retention_definition="deliberately_swept", **shared):
    """Calibrate and run independent reversal cells across retention/model pairs."""
    conditions = tuple(conditions)
    taus = tuple(float(value) for value in taus)
    allowed_retention = {
        "measured_held_bias", "measured_held_bias_quantiles",
        "measured_near_zero_field", "extrapolated", "deliberately_swept",
    }
    retention_definition = str(retention_definition)
    if retention_definition not in allowed_retention:
        raise ValueError(
            f"retention_definition must be one of {sorted(allowed_retention)}, "
            f"got {retention_definition!r}"
        )
    calibration = calibrate_reversal_scales(
        taus, conditions=conditions, trajectories=calibration_trajectories,
        calibration_trials=calibration_trials, workers=workers, pool=pool,
        **shared,
    )
    jobs = []
    for tau in taus:
        for condition in conditions:
            kwargs = dict(shared)
            kwargs.update(
                cond=condition, tau_leak=tau,
                eligibility_normalizer=calibration["records"][(tau, condition)][
                    "eligibility_normalizer"
                ],
                return_diagnostics=True,
            )
            jobs.append(((tau, condition), kwargs))
    mapped = _map_reversal_jobs(jobs, workers=workers, pool=pool)
    return {
        "taus": list(taus),
        "conditions": list(conditions),
        "calibration": calibration,
        "cells": dict(mapped),
        "retention_definition": retention_definition,
        "interpretation": (
            "Within-trial retention sensitivity after effective-update RMS control; "
            "not persistence of eligibility across reversal."
        ),
        "method_provenance": {
            "status": "adapted",
            "established_basis": ["reversal learning", "eligibility traces"],
            "repository_adaptation": (
                "Trial-local device and post-peak decay-matched filters are compared "
                "after reward-time update-scale calibration."
            ),
            "claim_limit": (
                "Descriptive simulated recovery; the trace resets each trial and a "
                "null result must be retained."
            ),
        },
    }


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
    csum = np.r_[0.0, np.cumsum(seg)]
    rr = (csum[window:] - csum[:-window]) / window
    hit = np.argmax(rr >= crit) if np.any(rr >= crit) else None
    return int(hit + window) if hit is not None else None


def kaplan_meier_recovery_summary(times, censored, horizon):
    """Summarise recovery times without treating right-censoring as an event.

    ``times`` contains the observed recovery time or the last observation time for
    each seed. ``censored`` is true when recovery was not observed. The returned
    restricted mean survival time (RMST) integrates the Kaplan--Meier curve only to
    the common follow-up ``horizon``; the median remains unavailable when survival
    never falls to 0.5. This is preferable to averaging censoring times as though
    every unsolved seed recovered at the end of the phase.
    """
    times = np.asarray(times, dtype=float).ravel()
    censored = np.asarray(censored, dtype=bool).ravel()
    horizon = float(horizon)
    if times.size == 0 or times.shape != censored.shape:
        raise ValueError("times and censored must be non-empty arrays of equal shape")
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be finite and positive")
    if np.any(~np.isfinite(times)) or np.any(times < 0) or np.any(times > horizon):
        raise ValueError("all observed or censoring times must lie within the horizon")

    survival = 1.0
    previous = 0.0
    rmst = 0.0
    median = None
    for time in np.unique(times):
        rmst += survival * (float(time) - previous)
        at_risk = int(np.count_nonzero(times >= time))
        events = int(np.count_nonzero((times == time) & ~censored))
        if events:
            survival *= 1.0 - events / at_risk
            if median is None and survival <= 0.5:
                median = float(time)
        previous = float(time)
    rmst += survival * max(0.0, horizon - previous)
    return {
        "kaplan_meier_median_trials": median,
        "median_status": "observed" if median is not None else "not_reached",
        "restricted_mean_trials": float(rmst),
        "restriction_horizon_trials": horizon,
        "recovered_fraction": float(np.mean(~censored)),
        "censored_fraction": float(np.mean(censored)),
        "n_seeds": int(times.size),
    }


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
        "retention_definition": "deliberately_swept",
        "method_provenance": BANDIT_METHOD_PROVENANCE,
    }


def _scaling_cell(job):
    """One spawn-safe array-size cell for :func:`run_scaling`."""
    from .stats import bootstrap_ci
    S, A, seeds, D, trials = job
    chance = 1.0 / A
    crit = 0.5 * (1 + chance)
    dev = train(S, A, B=seeds, D=D, trials=trials)
    nt = train(S, A, B=seeds, D=D, trials=trials, no_trace=True)
    dev_fr, nt_fr = reward_rate(dev), reward_rate(nt)
    ttc = [trials_to_criterion(dev[b], crit) for b in range(seeds)]
    ok = [t for t in ttc if t is not None]

    def ci(arr):
        a = np.asarray(arr, float)
        lo, hi = bootstrap_ci(a)
        return (float(a.mean()), float(a.std()), lo, hi)

    return (S, A), dict(
        chance=chance, crit=crit, device=ci(dev_fr), no_trace=ci(nt_fr),
        trials_to_crit=int(np.median(ok)) if ok else None,
        n_converged=len(ok), passed=bool(dev_fr.mean() >= crit), n_seeds=seeds,
        trials=trials, retention_definition="deliberately_swept",
        method_provenance=BANDIT_METHOD_PROVENANCE)


def run_scaling(*, seeds=20, grid=((2, 2), (4, 2), (4, 4), (8, 4), (8, 8), (12, 8)),
                D=2.0, trials=1500, workers=1):
    """Fig 8a grid (tier4): does the policy still converge as ``S x A`` grows?

    Retrospectively recorded criterion: reward rate >= 0.5*(1 + 1/A).
    Returns a dict keyed by ``(S, A)`` with device/no-trace means + bootstrap CIs,
    trials-to-criterion, and pass/fail; no file I/O.
    """
    jobs = [(S, A, seeds, D, trials) for S, A in grid]
    if int(workers) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(int(workers), len(jobs))) as pool:
            rows = pool.map(_scaling_cell, jobs, chunksize=1)
    else:
        rows = map(_scaling_cell, jobs)
    return dict(rows)


def _remedy_cell(job):
    """One spawn-safe 8x8 remedy cell for :func:`run_remedies`."""
    from .stats import bootstrap_ci
    name, kw, seeds, crit = job
    fr = reward_rate(train(8, 8, B=seeds, D=2.0, **kw))
    a = np.asarray(fr, float); lo, hi = bootstrap_ci(a)
    return name, dict(final=(float(a.mean()), float(a.std()), lo, hi),
                      passed=bool(a.mean() >= crit), n_seeds=seeds,
                      trials=kw["trials"], retention_definition="deliberately_swept",
                      method_provenance=BANDIT_METHOD_PROVENANCE)


def run_remedies(*, seeds=20, trials=1500, budget_multiplier=4, workers=1):
    """Fig 8b (tier5): five remedies on the failing 8x8 case.

    Returns a dict keyed by remedy label with final reward-rate mean + bootstrap CI
    and pass/fail against the 8x8 criterion; no file I/O.
    """
    A = 8
    crit = 0.5 * (1 + 1.0 / A)
    long_trials = int(trials * budget_multiplier)
    rows = [
        (f"R0 baseline (device, undirected, {trials})", dict(trials=trials)),
        (f"R1 more budget (device, undirected, {long_trials})", dict(trials=long_trials)),
        (f"R2 directed expl (device, sigma 0.45->0.05, {trials})",
         dict(trials=trials, sigma0=0.45, sigma1=0.05)),
        (f"R3 abstract trace (NOT device, undirected, {trials})",
         dict(trials=trials, abstract=True)),
        (f"R4 directed+budget (device, sigma anneal, {long_trials})",
         dict(trials=long_trials, sigma0=0.45, sigma1=0.05)),
    ]
    jobs = [(name, kw, seeds, crit) for name, kw in rows]
    if int(workers) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(int(workers), len(jobs))) as pool:
            return dict(pool.map(_remedy_cell, jobs, chunksize=1))
    return dict(map(_remedy_cell, jobs))


def _summarize_reversal(raw, conds, trials, crit, chance):
    """Package raw per-condition ``run_reversal`` output into the saved grid dict
    (mean curves, pre/post finals + CIs, re-acquisition cost, recovery bins,
    retrospectively recorded H1-H4/K1)."""
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
            "criteria": criteria, "retention_definition": "deliberately_swept",
            "method_provenance": BANDIT_METHOD_PROVENANCE}


def main(argv=None):
    """Full-scale reproduction CLI for the bandit-family grids (writes ``data/results``).

    ``python -m mrl_trace.bandit [--exp18] [--bandit] [--full|--quick]``
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
