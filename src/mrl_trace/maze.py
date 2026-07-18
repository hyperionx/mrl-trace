"""Sequential distal-reward navigation -- a multi-step MDP credit-assignment task.

The contextual bandit of :mod:`mrl_trace.bandit` is *one step*: a single
state, a single action, an immediate (delayed) reward. That isolates the
credit-assignment mechanism but invites the criticism that it is an
association lookup rather than reinforcement learning of a *policy over a
trajectory*. This module adds the missing piece: a sequential Markov decision
process in which an action must be chosen at each of several states, reward is
delivered only at the goal after the whole trajectory, and credit must propagate
back across the intermediate steps. That backward propagation across a trajectory
is exactly what an eligibility trace exists for, so it is the natural task on which
to test a device-supplied trace and to benchmark it against an algorithmic one.

Two environments, sharing one training harness:

- :class:`LinearTrack` -- an ``L``-state corridor; at each state the agent chooses
  FORWARD (towards the goal) or BACK. Only a sustained forward run reaches the goal
  and is rewarded, after an action--reward delay ``D`` measured from the goal-entry
  action. A greedy one-step rule cannot short-circuit it: every step must carry the
  right action, so the policy spans the trajectory.
- :class:`TMaze` -- an ``L``-state stem leading to a junction with ``A_goal`` arms,
  exactly one rewarded. The agent advances along the stem, then selects an arm; the
  reward (after delay ``D``) is contingent on the arm matching the rewarded arm for
  the episode's cue. This makes the learned object a state->action *policy* whose
  final decision sits a whole trajectory away from the reward.

DESIGN NOTE (anti-"relabelled bandit"). The eligibility trace is accumulated
*online across the steps of an episode* on the synapses actually used at each state,
and a single delayed goal reward gates the surviving trace into all of them at once.
Because the device relaxes over seconds while steps are taken over the same
timescale, the trace from EARLY (distal) steps has decayed more than from LATE steps
when the reward lands -- so bridging the full trajectory genuinely requires a
retention long enough to span it. Short retention learns only the steps near the
goal; this is the sequential analogue of the bandit's delay window and the lever the
benchmark and the D_max-vs-tau prediction both exercise.

Conventions follow the rest of the package: ``B`` seeds run as a vectorised batch,
device gate via :class:`GateBankBatched` (``abstract=True`` swaps the exponential
R-STDP kernel, ``no_trace=True`` zeroes eligibility), signed leak-dominant
coincidence, three-factor update ``dw = eta (R - b) e``, ``dt = 5e-3`` s.
"""
from __future__ import annotations

import numpy as np

from .bandit import GateBankBatched, AbstractTrace, W_INIT, W_MAX
from .device import K_STAGES
from .neurons import lif_step_batched, TAU_M, V_TH
from .learning import LTD_BIAS

__all__ = [
    "LinearTrack", "TMaze", "train_sequential",
    "reward_rate", "trials_to_criterion", "policy_correct",
    # experiment cores (composed of TMaze + train_sequential)
    "run_sequential", "run_dmax_law", "run_long_horizon", "run_long_horizon_faults",
    # analysis helpers
    "running_rate", "interp_dmax",
]


# ----------------------------------------------------------------------------
# Environments
# ----------------------------------------------------------------------------
class LinearTrack:
    """``L``-state corridor MDP. Actions: 0 = forward, 1 = back.

    State index advances on a forward action and retreats on a back action; the
    episode ends with reward when the agent steps forward out of the last state
    (reaching the goal), or unrewarded if it runs out of steps. ``n_states`` is the
    state-encoding width and ``n_actions = 2``.
    """

    def __init__(self, L=3):
        self.L = L
        self.n_states = L
        self.n_actions = 2
        self.goal_action = 0          # forward at the last state reaches the goal

    def start(self, B):
        return np.zeros(B, dtype=int)     # all seeds start at state 0

    def step(self, pos, action):
        """Vectorised transition. ``pos`` (B,), ``action`` (B,) in {0,1}.

        Returns ``(next_pos, reached_goal, done)`` boolean/int arrays of shape (B,).
        """
        forward = action == 0
        nxt = np.where(forward, pos + 1, np.maximum(pos - 1, 0))
        reached = nxt >= self.L          # stepped forward out of the last state
        nxt = np.where(reached, self.L - 1, nxt)   # clamp (episode ends anyway)
        return nxt, reached, reached

    def correct_action(self, pos):
        """The action that makes progress towards the goal from ``pos`` (always
        forward here); used only to score policy correctness, never during learning."""
        return np.zeros_like(pos)


class TMaze:
    """``L``-state stem then a ``A_goal``-arm junction; one arm rewarded per cue.

    States ``0..L-1`` are the stem (advance with action 0). At the junction state
    ``L-1`` the agent's action selects an arm ``0..A_goal-1``; reward is contingent on
    the arm matching the episode's rewarded arm (set by a cue presented at the start).
    ``n_actions = max(2, A_goal)`` so the same action neurons serve stem-advance and
    arm-choice.
    """

    def __init__(self, L=3, A_goal=2):
        self.L = L
        self.A_goal = A_goal
        self.n_states = L + A_goal       # stem states + one state per cue context
        self.n_actions = max(2, A_goal)
        self.goal_action = None          # depends on the cued arm (set per episode)

    def start(self, B, rng):
        self.cue = rng.integers(self.A_goal, size=B)   # rewarded arm this episode
        return np.zeros(B, dtype=int)

    def encode(self, pos):
        """State encoding: stem position, or (at the junction) a cue-specific index so
        the policy can route to the cued arm. Returns an index in ``[0, n_states)``."""
        at_junction = pos >= self.L - 1
        return np.where(at_junction, self.L + self.cue, pos)

    def step(self, pos, action):
        at_junction = pos >= self.L - 1
        # The stem is a one-way corridor: the agent AUTO-ADVANCES through it regardless
        # of action (navigation, not a learned decision). The only learned choice -- and
        # the only credit-assignment problem -- is the arm selection at the junction.
        # This keeps the reward DISTAL (cue at start, decision a trajectory later) while
        # removing the artefact of stem-advance reinforcing one action everywhere.
        nxt = np.where(at_junction, pos, pos + 1)
        done = at_junction
        reached = at_junction & (action == self.cue)    # correct arm -> reward
        return nxt, reached, done

    def correct_action(self, pos):
        at_junction = pos >= self.L - 1
        return np.where(at_junction, self.cue, 0)


# ----------------------------------------------------------------------------
# Training harness (shared across device / R-STDP / no-trace)
# ----------------------------------------------------------------------------
class EpropTrace:
    """Reward-based e-prop eligibility (Bellec et al. 2020), the third benchmark.

    Unlike the device gate (a physical relaxation of a signed coincidence) and R-STDP
    (a hand-set exponential filter of that coincidence), e-prop *computes* its
    eligibility from the network's own dynamics. For a LIF action neuron j fed by
    state-line i, the per-synapse eligibility trace is

        e_ij(t) = LP_kappa[ psi_j(t) * zbar_i(t) ],

    where ``zbar_i`` is the low-pass-filtered presynaptic spike train (membrane filter,
    tau_m), ``psi_j`` is the pseudo-derivative of the spike nonlinearity evaluated on the
    PRE-RESET membrane (the surrogate gradient, a triangular/Gaussian bump at threshold),
    and ``LP_kappa`` is a slow output filter (the eligibility time constant, played here
    by ``tau_leak`` so the three methods are compared at a matched retention). The signed
    LEARNING SIGNAL (R - b) supplies credit/blame at reward; e-prop does NOT use a signed
    coincidence (that is the R-STDP/device mechanism), so its drive is the unsigned
    product psi*zbar. This is the shallow, single-decision-layer reduction of reward-based
    e-prop: faithful to the per-synapse eligibility, but it does not exercise e-prop's
    recurrent-credit machinery (a property of the feedforward task, stated plainly).
    """

    def __init__(self, B, S, A, tau_leak=10.0, dt=5e-3, tau_m=TAU_M, v_th=V_TH,
                 psi_width=0.3, dampening=0.3, **_ignored):
        self.B, self.S, self.A, self.dt = B, S, A, dt
        self.tau_m, self.v_th = tau_m, v_th
        self.psi_width = psi_width          # surrogate-gradient width
        self.gamma = dampening              # surrogate-gradient peak (Bellec's gamma)
        self.tau = tau_leak                 # output (eligibility) filter time constant
        self.Vnmax = 1.0                    # interface parity with the device gate
        self.zbar = np.zeros((B, S))        # filtered presynaptic trace per state-line
        self.elig = np.zeros((B, S, A))     # filtered eligibility per synapse
        self.vn = np.zeros((B, S, A, 1))    # mirror of device .vn[...,-1] for readout

    def reset(self):
        self.zbar[:] = 0.0
        self.elig[:] = 0.0
        self.vn[:] = 0.0

    def step_eprop(self, pre, v_pre, active):
        """Advance one dt-tick. ``pre`` (B,S) presynaptic spikes this tick, ``v_pre``
        (B,A) pre-reset membrane, ``active`` (B,) episode mask. Returns nothing; the
        eligibility is read from ``self.vn[...,-1]`` for parity with the device gate."""
        dt = self.dt
        # presynaptic membrane-filtered trace zbar_i
        self.zbar += (-self.zbar / self.tau_m + pre) * dt
        # pseudo-derivative psi_j on the pre-reset membrane (triangular surrogate):
        # gamma * max(0, 1 - |v - v_th| / width)
        psi = self.gamma * np.maximum(0.0, 1.0 - np.abs(v_pre - self.v_th) / self.psi_width)
        # per-synapse instantaneous eligibility = psi_j * zbar_i  (outer over S x A)
        inst = self.zbar[:, :, None] * psi[:, None, :]          # (B,S,A)
        # slow output filter -> the eligibility trace (time constant tau)
        self.elig += (-self.elig / self.tau + inst) * dt
        self.elig *= active[:, None, None]
        self.vn[..., 0] = self.elig


def _relax(bank, zero_drive, n, stride, abstract):
    """Advance an undriven gate for ``n`` blocks of ``stride`` dt-ticks each, using a
    temporarily coarsened timestep for speed. With zero drive the cascade/leaky
    dynamics are smooth, so a coarse forward-Euler step is accurate provided
    stride*dt/tau << 1 (checked: stride=10, dt=5e-3, tau>=2 s -> step ratio <=0.025).
    Restores the gate's original dt afterwards."""
    if n <= 0:
        return
    dt0 = bank.dt
    bank.dt = dt0 * stride
    try:
        for _ in range(n):
            bank.step(zero_drive)
    finally:
        bank.dt = dt0


def _relax_eprop(bank, B, S, A, n, stride):
    """Coarse-stepped zero-input relaxation of the e-prop eligibility filters (mirror
    of :func:`_relax` for the EpropTrace, whose decay is equally smooth with no input)."""
    if n <= 0:
        return
    zp, zv, act = np.zeros((B, S)), np.zeros((B, A)), np.ones(B)
    dt0 = bank.dt
    bank.dt = dt0 * stride
    try:
        for _ in range(n):
            bank.step_eprop(zp, zv, act)
    finally:
        bank.dt = dt0

def train_sequential(env, *, B=20, tau_leak=10.0, D=2.0, episodes=1500,
                     max_steps=None, dt=5e-3, step_dur=0.4, eta=0.2, V=1.5,
                     in_rate=200.0, ltd=LTD_BIAS, tau_m=TAU_M, v_th=V_TH,
                     sigma0=0.15, sigma1=None, abstract=False, no_trace=False,
                     eprop=False, beta_pol=1.0, beta_leak=1.0, weight_fault=None, seed0=0,
                     return_weights=False, device_k=K_STAGES, tau_r_override=None):
    """Train the sequential policy on ``env`` for ``B`` parallel seeds.

    The network is a state x action grid of device synapses ``w[state, action]``.
    Each EPISODE runs a trajectory: at every visited state the active state-line
    drives the ``A`` action neurons (Kirchhoff bitline sum), the spiking winner is the
    action, and the signed pre--post coincidence on the chosen (state, action) synapse
    drives the eligibility gate. The gate integrates CONTINUOUSLY across the whole
    trajectory (it is not reset between steps), so eligibility from early steps decays
    while later steps are taken. A single goal reward, delivered ``D`` seconds after
    the goal-entry action, then gates the surviving trace into every synapse:
    ``dw = eta (R - b) e``. Returns rewards ``(B, episodes)``.

    ``abstract`` swaps the device gate for the exponential R-STDP kernel;
    ``no_trace`` zeroes eligibility (necessity control). ``weight_fault`` applies the
    device-fault prior at READ time (once per episode, since weights change only per
    episode): the array reads a faulted conductance while learning still targets the
    clean weight -- a fixed physical realisation, not a learnable parameter, exactly as
    in :func:`mrl_trace.deep.train_deep`. The maze weights are NON-NEGATIVE
    (``[0, W_MAX]``) single conductances, so the signed-pair SiO_x stack cannot be used
    verbatim; pass a fault built for the non-negative weight space (see
    ``device_faults.maze_fault_stack``). ``step_dur`` is the wall-clock time the agent
    dwells in each state (so a trajectory of ``L`` steps spans
    ``L*step_dur`` seconds, against which ``tau_leak`` must be long enough).
    """
    rng = np.random.default_rng(seed0)
    S, A = env.n_states, env.n_actions
    if max_steps is None:
        max_steps = 3 * env.L + 2
    if eprop:
        bank = EpropTrace(B, S, A, tau_leak=tau_leak, dt=dt, tau_m=tau_m, v_th=v_th)
    elif abstract:
        bank = AbstractTrace(B, S, A, tau_leak=tau_leak, dt=dt)
    else:
        bank = GateBankBatched(B, S, A, tau_leak=tau_leak, dt=dt, V=V,
                               beta_leak=beta_leak, k=device_k,
                               tau_r_override=tau_r_override)
    w = np.full((B, S, A), W_INIT)
    baseline = np.full(B, 1.0 / A)
    bidx = np.arange(B)
    steps_per_state = max(1, int(round(step_dur / dt)))
    reward_lag = int(round(D / dt))
    rewards = np.zeros((B, episodes))

    for ep in range(episodes):
        sigma = sigma0 if sigma1 is None else sigma0 + (sigma1 - sigma0) * ep / episodes
        # Read-time device faults applied ONCE per episode: the array reads the faulted
        # conductance (wr) while learning updates the clean weight (w). Computing this per
        # episode (not per dt-tick) is what keeps the run tractable.
        wr = weight_fault(w) if weight_fault is not None else w
        bank.reset()
        if isinstance(env, TMaze):
            pos = env.start(B, rng)
        else:
            pos = env.start(B)
        v = np.zeros((B, A))
        done = np.zeros(B, dtype=bool)
        got_reward = np.zeros(B, dtype=bool)
        # accumulated eligibility snapshot, captured at each seed's reward instant
        e_rew = np.zeros((B, S, A))
        reward_due = np.full(B, -1)         # step index at which reward gates in
        chosen_final = np.zeros(B, dtype=int)   # decision-point action per seed (e-prop)
        pol_final = np.full((B, A), 1.0 / A)     # decision-point policy pi (e-prop)
        # --- run the trajectory ---
        for st in range(max_steps):
            state = env.encode(pos) if isinstance(env, TMaze) else pos
            chosen = np.zeros(B, dtype=int)
            # dwell in this state for steps_per_state dt-ticks: integrate spikes + trace
            spk = np.zeros((B, A))
            for _ in range(steps_per_state):
                pre = np.zeros((B, S))
                active = ~done
                pre[bidx[active], state[active]] = (
                    rng.random(active.sum()) < in_rate * dt).astype(float)
                charge = np.einsum('bsa,bs->ba', wr, pre)
                v, sp, v_pre = lif_step_batched(v, charge, dt, rng, tau_m=tau_m,
                                                v_th=v_th, noise=sigma, return_pre=True)
                spk += sp * active[:, None]
                if eprop:
                    # e-prop computes eligibility from network dynamics: the pre-reset
                    # membrane (pseudo-derivative) and the presynaptic trace. No signed
                    # coincidence -- credit/blame comes from (R - b) at reward.
                    bank.step_eprop(pre, v_pre, active)
                else:
                    # signed leak-dominant coincidence on the CURRENT state's lines
                    # (device gate and R-STDP share this drive)
                    drive = np.zeros((B, S, A))
                    drv = pre[bidx, state][:, None] * np.where(sp, 1.0, -ltd)
                    drive[bidx, state, :] = drv * active[:, None]
                    if no_trace:
                        drive[:] = 0.0
                    bank.step(drive)
            if eprop:
                # Reward-based e-prop is a policy-gradient method: the action is SAMPLED
                # from a softmax policy over the action-neuron spike counts (this sampling
                # is the exploration), and the policy pi is stored so the per-action
                # learning signal (1[a=chosen] - pi_a) can be formed at reward.
                logits = beta_pol * (spk - spk.mean(1, keepdims=True))
                pol_p = np.exp(logits - logits.max(1, keepdims=True))
                pol_p /= pol_p.sum(1, keepdims=True)
                cdf = np.cumsum(pol_p, axis=1)
                u = rng.random((B, 1))
                chosen = (u > cdf).sum(1)
                chosen = np.clip(chosen, 0, A - 1)
            else:
                # action = spiking winner over the dwell (ties -> random)
                tie = spk.max(1) == spk.min(1)
                chosen = np.argmax(spk, 1)
                chosen[tie] = rng.integers(A, size=int(tie.sum()))
            chosen[done] = 0
            nxt, reached, ep_done = env.step(pos, chosen)
            pos = np.where(done, pos, nxt)
            newly = (~done) & ep_done
            # record the decision-point action and policy (the step that ended the
            # episode) for the e-prop per-action learning signal (1[a=chosen] - pi_a)
            chosen_final = np.where(newly, chosen, chosen_final)
            if eprop:
                pol_final = np.where(newly[:, None], pol_p, pol_final)
            got_reward |= (newly & reached)
            # schedule reward gating reward_lag ticks after goal entry (approx: capture
            # the trace reward_lag dt-ticks later by letting the gate relax that long)
            reward_due = np.where(newly & (reward_due < 0), st, reward_due)
            done |= ep_done
            if done.all():
                break
        # Let the trace relax for the action->reward delay, then snapshot eligibility.
        # During relaxation there is NO drive and no spikes, so the dynamics are smooth;
        # integrate with a coarser effective step (stride x dt) to keep long delays
        # tractable. Equivalent to bank.step(0) repeated reward_lag times, to first
        # order, but ~stride x cheaper -- the relaxation is numerically smooth so this
        # introduces negligible error at stride=10 (verified against the tick loop).
        zero = np.zeros((B, S, A))
        stride = 10 if reward_lag > 200 else 1
        n_relax = reward_lag // stride
        rem = reward_lag - n_relax * stride
        if eprop:
            # e-prop's slow output filter keeps decaying with no input; relax it the
            # same way (coarse-stepped), reading eligibility from vn[...,-1].
            _relax_eprop(bank, B, S, A, n_relax, stride)
            for _ in range(rem):
                bank.step_eprop(np.zeros((B, S)), np.zeros((B, A)), np.ones(B))
            e_rew = bank.vn[..., -1]
        elif no_trace:
            e_rew = np.zeros((B, S, A))
        elif abstract:
            for _ in range(reward_lag):
                bank.step(zero)
            e_rew = bank.e
        else:
            _relax(bank, zero, n_relax, stride, abstract)
            for _ in range(rem):
                bank.step(zero)
            e_rew = bank.vn[..., -1] / bank.Vnmax
        R = got_reward.astype(float)
        adv = eta * (R - baseline)
        if eprop:
            # Reward-based e-prop = policy gradient with an e-prop eligibility. The
            # per-action learning signal is the REINFORCE log-policy gradient
            # L_j = (R - b) * (1[j = chosen] - pi_j), which supplies the arm selectivity
            # that e-prop's unsigned eligibility lacks: the chosen action is pushed up by
            # (1 - pi) and the others down by (-pi), scaled by the advantage. Delta w_ij
            # = eta * (R-b) * (1[j=chosen]-pi_j) * e_ij. eta absorbs the small eligibility
            # magnitude (e-prop eligibility ~1e-4 here; see eta scaling).
            onehot = np.zeros((B, A))
            onehot[bidx, chosen_final] = 1.0
            Lsig = (R - baseline)[:, None] * (onehot - pol_final)     # (B, A)
            w = np.clip(w + eta * Lsig[:, None, :] * e_rew, 0.0, W_MAX)
        else:
            w = np.clip(w + adv[:, None, None] * e_rew, 0.0, W_MAX)
        baseline += 0.02 * (R - baseline)
        rewards[:, ep] = R
    if return_weights:
        return rewards, w          # w: (B, S, A) final learned weights
    return rewards


# ----------------------------------------------------------------------------
# Reductions (reuse the bandit's conventions)
# ----------------------------------------------------------------------------
def reward_rate(rewards, window=100):
    rewards = np.asarray(rewards)
    if rewards.shape[-1] < window:
        return rewards.mean(axis=-1)
    return rewards[..., -window:].mean(axis=-1)


def trials_to_criterion(rew_1d, crit, window=100):
    rew_1d = np.asarray(rew_1d)
    if len(rew_1d) < window:
        return None
    csum = np.cumsum(rew_1d)
    rr = (csum[window:] - csum[:-window]) / window
    if not np.any(rr >= crit):
        return None
    return int(np.argmax(rr >= crit) + window)


def policy_correct(env, w):
    """Fraction of states whose argmax-weight action equals the correct action.

    A policy-quality probe independent of stochastic spiking: for each state, the
    action with the largest learned weight should be the progress/arm-correct action.
    ``w`` is a single seed's ``(S, A)`` weight grid.
    """
    S = env.n_states
    correct = 0
    total = 0
    for s in range(S):
        # correct action per state, derived directly from the state index (no episode
        # state needed): stem states want FORWARD (action 0); a junction-context state
        # s = L + arm wants that arm.
        if isinstance(env, TMaze):
            ca = 0 if s < env.L else (s - env.L)
        else:
            ca = 0                          # LinearTrack: always forward
        if w[s].max() == w[s].min():
            continue                       # untrained / tied row -> skip
        if int(np.argmax(w[s])) == int(ca):
            correct += 1
        total += 1
    return correct / total if total else 0.0


# =============================================================================
# Experiment cores (the sequential-MDP credit-assignment studies that compose
# ``TMaze`` + ``train_sequential``)
#
# Each ``run_*`` returns the result grid as a plain dict -- no file I/O, no
# plotting, no stdout.  Notebooks call these at a small (quick) seed/episode
# count and render the figures inline; ``main()`` (below) calls them at the
# published scale, parallelising the coarse axis over a process pool, and writes
# each grid under ``data/results/`` via ``paths.save_result`` for full-cache mode.
# The original driver scripts each computed one grid:
#   * exp6_sequential.npy            -- sequential T-maze + R-STDP/e-prop/no-trace
#                                       benchmark and the retention x delay grid;
#   * exp8_dmax_law.npy              -- densely-sampled D_max(tau_leak) scaling law;
#   * exp15_long_horizon.npy         -- longer-horizon (trajectory length L) credit
#                                       feasibility vs the established traces;
#   * exp15_long_horizon_faults.npy  -- the same under the SiO_x device-fault prior.
# =============================================================================


# -----------------------------------------------------------------------------
# Experiment 6 -- sequential distal-reward task + R-STDP / e-prop benchmark.
#
# Extends the one-step contextual bandit to a multi-step MDP (T-maze), where reward
# is delivered only at the goal after a trajectory and credit must propagate back
# across intermediate steps -- the regime an eligibility trace exists for. Four
# conditions share ONE harness for a fair comparison:
#   device   : the SiOx trap-cascade eligibility gate (this work);
#   rstdp    : an abstract exponential eligibility trace at matched timescale
#              (the hand-set kernel of Florian 2007 / Izhikevich 2007 / Legenstein 2008);
#   eprop    : network-computed eligibility (reward-modulated policy gradient, Bellec 2020);
#   no_trace : eligibility zeroed (device-necessity control).
# Operating point V=1.5 (rise time tau_r=1.9 s) so the eligibility snapshot reflects
# RETENTION (tau_leak), not the sigmoidal rise. Retrospective criteria C1-C4 / kill K1
# are evaluated in the summary.
# -----------------------------------------------------------------------------
EPROP_ETA = 2000.0        # e-prop eligibility magnitude ~1e-4 << device/R-STDP; eta scaled up


def running_rate(r1d, window=50):
    """Running ``window``-episode reward rate of a single mean-over-seeds curve."""
    c = np.cumsum(np.insert(r1d, 0, 0.0))
    return (c[window:] - c[:-window]) / window


def run_sequential(*, seeds=20, episodes=800, V=1.5, D0=4.0, TAU0=10.0,
                   taus=(2.0, 5.0, 20.0), delays=(2, 4, 8, 16, 32, 64), seed0=0,
                   workers=1, device_k=K_STAGES, tau_r_override=None):
    """Experiment 6: sequential T-maze learning curves + retention x delay grid.

    Runs the four conditions (device / rstdp / eprop / no_trace) as learning curves
    at the reference ``(D0, TAU0)`` operating point, a device policy-correctness probe
    from the ACTUAL learned weights, and a ``taus x delays`` device reward-rate grid
    from which ``D_max(tau_leak)`` (the prediction-arm datapoints) is read off.

    Every cell is an independent, seed-deterministic ``train_sequential`` call
    (``B=seeds``, ``seed0=0``), so the result is identical whether the cells are run
    serially or dispatched across a spawn-safe process pool with ``workers > 1``.
    Returns the result
    dict (per-condition per-seed finals + running curves, policy correctness, the
    retention x delay grid with bootstrap CIs, ``D_max`` per tau, and the retrospective
    criteria); no file I/O.
    """
    from functools import partial
    taus = list(taus); delays = list(delays)
    conds = {"device": dict(), "rstdp": dict(abstract=True),
             "eprop": dict(eprop=True, eta=EPROP_ETA),
             "no_trace": dict(no_trace=True)}
    specs = []
    for name, kw in conds.items():
        specs.append((("curve", name), TAU0, D0, kw, False))
    specs.append((("weights", "device"), TAU0, D0, dict(), True))
    for tau in taus:
        for D in delays:
            specs.append((("grid", tau, D), tau, D, dict(), False))
    worker = partial(_exp6_worker_seeded, seeds=seeds, episodes=episodes, V=V,
                     seed0=seed0, device_k=device_k,
                     tau_r_override=tau_r_override)
    if int(workers) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(int(workers), len(specs))) as pool:
            res = dict(pool.map(worker, specs, chunksize=1))
    else:
        res = dict(map(worker, specs))
    return _exp6_assemble(res, conds, taus, delays, seeds, episodes, V, D0, TAU0)


# -----------------------------------------------------------------------------
# Experiment 8 -- densely-sampled retention->delay scaling, D_max(tau_leak).
#
# Sharpens the D_max ~= k*tau relation from the 3-point version (exp6) to ~13 retention
# values on a fine delay grid, so the FUNCTIONAL FORM can be read off rather than assumed:
# is it linear in the trace-dominated regime, and where does it saturate (the inverted-U
# the conditioning literature predicts)? D_max is interpolated as the largest delay whose
# reward rate is >= criterion (threshold crossing), not just the coarsest passing grid
# point. Device condition only (the law is a device-physics claim).
#
# HONEST SCOPE: this improves the resolution of the FORM. It does NOT remove the
# in-simulation near-tautology (tau_leak sets the trace decay, which by construction
# bounds the bridgeable delay). The non-tautological tests are the hardware tau_leak sweep
# and the biological arm (future work). This is the "proposed scaling relation, sampled"
# -- not a claim of a tested cross-domain law.
# -----------------------------------------------------------------------------
# Retention values spanning the measured band (~1-40 s). The trace-dominated regime
# (tau<=8 s) is densified with interpolating points so the linear fit rests on 9 points
# rather than 5; the longer tau (>=12 s) are kept for context but remain right-censored at
# the delay-grid ceiling.
DMAX_TAUS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.5, 8.0, 12.0, 20.0, 30.0, 40.0]
# Delay grid extended (adding 420, 560, 700) so the tau=30 and 40 s points can resolve
# their D_max crossings; with k~14 the tau=40 point is expected near 560 s.
DMAX_DELAYS = [1, 2, 3, 5, 8, 12, 20, 32, 48, 72, 100, 140, 180, 240, 300, 420, 560, 700]
DMAX_PLATEAU_TAU = 20.0


def interp_dmax(delays, rates, crit):
    """Largest delay at which the (monotone-from-peak) reward rate is >= crit, linearly
    interpolated in log-delay between the last passing and first failing grid points.
    Returns 0 if never reached, or the max delay if always above."""
    d = np.asarray(delays, float); r = np.asarray(rates, float)
    above = r >= crit
    if not above.any():
        return 0.0
    # take the falling edge AFTER the peak (D_max is where it drops below crit going up
    # in delay), so scan from the peak rightwards for the last contiguous >=crit delay
    pk = int(np.argmax(r))
    last = pk
    for j in range(pk, len(d)):
        if r[j] >= crit:
            last = j
        else:
            break
    if last == len(d) - 1:
        return float(d[last])              # still above at the largest delay tested
    # interpolate in log-delay between d[last] (>=crit) and d[last+1] (<crit)
    x0, x1 = np.log(d[last]), np.log(d[last + 1])
    y0, y1 = r[last], r[last + 1]
    frac = (y0 - crit) / (y0 - y1) if y0 != y1 else 0.0
    return float(np.exp(x0 + frac * (x1 - x0)))


def _dmax_origin_fit(taus, dvals, mask):
    """Origin-forced linear fit ``D_max = k*tau`` over the masked points -> ``(k, R^2)``."""
    if mask.sum() < 2:
        return float("nan"), float("nan")
    kk = float(np.sum(taus[mask] * dvals[mask]) / np.sum(taus[mask] ** 2))
    res = dvals[mask] - kk * taus[mask]
    sst = float(np.sum((dvals[mask] - dvals[mask].mean()) ** 2))
    return kk, (1 - float(np.sum(res ** 2)) / sst if sst > 0 else float("nan"))


def _dmax_cell(spec, episodes, V, seeds, device_k=K_STAGES, tau_r_override=None):
    """One device cell, retaining the per-seed final reward rates."""
    from .stats import bootstrap_ci
    tau, D = spec
    env = TMaze(L=3, A_goal=2)
    r = train_sequential(env, B=seeds, tau_leak=tau, D=D, episodes=episodes, V=V,
                         seed0=0, device_k=device_k,
                         tau_r_override=tau_r_override)
    f = reward_rate(r, 100)
    lo, hi = bootstrap_ci(f)
    return (tau, D, float(f.mean()), lo, hi, np.asarray(f, float))


def _dmax_assemble(results, taus, delays, crit, seeds, episodes, V):
    """Package raw ``(tau, D, mean, lo, hi)`` cells into the exp8 grid dict (per-tau
    reward-rate rows, interpolated ``D_max``, the origin-forced headline fit + the
    asymptotic plateau slope, and the right-censoring flags)."""
    grid = {tau: {} for tau in taus}
    seed_grid = {tau: {} for tau in taus}
    for tau, D, m, lo, hi, seed_values in results:
        grid[tau][D] = (m, lo, hi)
        seed_grid[tau][D] = seed_values
    dmax = {tau: interp_dmax(delays, [grid[tau][D][0] for D in delays], crit)
            for tau in taus}
    taus_a = np.array(taus); dvals = np.array([dmax[t] for t in taus])
    censored = dvals >= max(delays) - 1e-6
    fitmask = (~censored) & (dvals > 0)
    k, r2 = _dmax_origin_fit(taus_a, dvals, fitmask)                     # all resolved points
    platmask = fitmask & (taus_a >= DMAX_PLATEAU_TAU)
    k_plateau, r2_plateau = _dmax_origin_fit(taus_a, dvals, platmask)    # asymptotic slope
    return {"taus": list(taus), "delays": list(delays), "grid": grid,
            "seed_grid": seed_grid, "dmax": dmax,
            "k": k, "r2": r2, "k_plateau": k_plateau, "r2_plateau": r2_plateau,
            "plateau_tau": DMAX_PLATEAU_TAU, "crit": crit,
            "censored": censored.tolist(), "seeds": seeds, "episodes": episodes,
            "V_op": V}


def run_dmax_law(*, seeds=20, episodes=800, V=1.5, taus=None, delays=None, crit=0.75,
                 workers=1, device_k=K_STAGES, tau_r_override=None):
    """Experiment 8: dense D_max(tau_leak) scaling law (device condition only).

    Runs every ``(tau, D)`` device cell serially and reads off the interpolated
    ``D_max`` per retention (threshold crossing of reward rate >= ``crit``), then
    fits ``D_max = k*tau`` through the origin over the resolved (non-censored) points
    and separately over the trace-dominated plateau (``tau >= DMAX_PLATEAU_TAU``).
    Each cell is deterministic (``seed0=0``); serial here, pooled in ``main``. Returns
    the result dict; no file I/O.
    """
    from functools import partial
    taus = DMAX_TAUS if taus is None else list(taus)
    delays = DMAX_DELAYS if delays is None else list(delays)
    specs = [(tau, D) for tau in taus for D in delays]
    worker = partial(_dmax_cell, episodes=episodes, V=V, seeds=seeds,
                     device_k=device_k, tau_r_override=tau_r_override)
    if int(workers) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(int(workers), len(specs))) as pool:
            results = pool.map(worker, specs, chunksize=1)
    else:
        results = list(map(worker, specs))
    return _dmax_assemble(results, taus, delays, crit, seeds, episodes, V)


def run_dmax_adaptive(*, seeds=4, episodes=300, V=1.5, taus=None,
                      delay_ratios=(4, 8, 12, 16, 20), crit=0.75, workers=1,
                      device_k=K_STAGES, tau_r_override=None):
    """Reduced live validation of the dense retention--delay law.

    The publication sweep uses one 18-value absolute-delay grid for every retention.
    A lightweight notebook need not spend most of its time far from the falling edge:
    this runner evaluates the same device task at fixed ``D / tau_leak`` ratios, then
    interpolates the criterion crossing independently for each retention.  No model
    outputs or fitted values are substituted; every displayed point is obtained from
    a live ``train_sequential`` cell.  The published-scale :func:`run_dmax_law` remains
    the exact-grid route.
    """
    from functools import partial
    taus = DMAX_TAUS if taus is None else [float(t) for t in taus]
    sampled_delays = {
        tau: sorted(set(float(np.clip(round(tau * r, 3), 1.0, 700.0))
                        for r in delay_ratios))
        for tau in taus
    }
    specs = [(tau, D) for tau in taus for D in sampled_delays[tau]]
    worker = partial(_dmax_cell, episodes=episodes, V=V, seeds=seeds,
                     device_k=device_k, tau_r_override=tau_r_override)
    if int(workers) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(int(workers), len(specs))) as pool:
            results = pool.map(worker, specs, chunksize=1)
    else:
        results = list(map(worker, specs))

    sampled = {tau: {} for tau in taus}
    seed_grid = {tau: {} for tau in taus}
    for tau, D, mean, lo, hi, seed_values in results:
        sampled[tau][D] = (mean, lo, hi)
        seed_grid[tau][D] = seed_values
    dmax = {}
    censored = []
    for tau in taus:
        ds = sampled_delays[tau]
        rates = [sampled[tau][D][0] for D in ds]
        dmax[tau] = interp_dmax(ds, rates, crit)
        censored.append(bool(rates[-1] >= crit))
    taus_a = np.asarray(taus, float)
    dvals = np.asarray([dmax[t] for t in taus], float)
    censored_a = np.asarray(censored, bool)
    fitmask = (~censored_a) & (dvals > 0)
    k, r2 = _dmax_origin_fit(taus_a, dvals, fitmask)
    plateau_tau = DMAX_PLATEAU_TAU
    k_plateau, r2_plateau = _dmax_origin_fit(
        taus_a, dvals, fitmask & (taus_a >= plateau_tau))
    return {"taus": list(taus), "sampled_delays": sampled_delays, "sampled": sampled,
            "seed_grid": seed_grid,
            "dmax": dmax, "k": k, "r2": r2, "k_plateau": k_plateau,
            "r2_plateau": r2_plateau, "plateau_tau": plateau_tau, "crit": crit,
            "censored": censored, "seeds": seeds, "episodes": episodes, "V_op": V,
            "mode": "adaptive-live"}


# -----------------------------------------------------------------------------
# Experiment 15 -- longer-horizon temporal-credit feasibility (Tier 2).
#
# Answers the *task-depth* objection ("your task is too easy") on the axis the device
# primitive is actually for: temporal credit assignment across a multi-step trajectory
# with a delayed goal reward. NOT a perception/benchmark study -- the input stays
# low-dimensional; what grows is the temporal DEPTH (trajectory length L) of the credit
# problem. Shallow state x action policy (no homeostasis). Per cell the device trace is
# compared against the established baselines on the SAME task/network/seeds/budget:
#   device / eprop / abstract / no_trace.
# Scaling axes: trajectory length L (T-maze stem) and arm count A_goal (chance = 1/A).
# tau_leak is set generously so retention can span the longest trajectory tested.
# -----------------------------------------------------------------------------
LONG_HORIZON_CONDS = {
    "device":   dict(),
    "eprop":    dict(eprop=True),
    "abstract": dict(abstract=True),
    "no_trace": dict(no_trace=True),
}


def _lh_final_rate(r):
    """Final-window (100-episode) reward rate averaged over the seed batch."""
    return float(np.mean(reward_rate(r, window=100)))


def _lh_run(job):
    """One (L, A, cond, seed) long-horizon cell -> (L, A, cond, seed, final rate).
    Top-level for multiprocessing; ``B=1`` and ``seed0=seed`` so seeds are independent."""
    if len(job) == 7:
        L, A, cond, seed, episodes, D, tau_leak = job
        dt = 5e-3
    else:
        L, A, cond, seed, episodes, D, tau_leak, dt = job
    env = TMaze(L=L, A_goal=A)
    r = train_sequential(env, B=1, tau_leak=tau_leak, D=D, episodes=episodes, dt=dt,
                         seed0=seed, **LONG_HORIZON_CONDS[cond])
    return (L, A, cond, seed, _lh_final_rate(r))


def _lh_boot_ci(vals, seed=0, n_boot=10000):
    """Percentile bootstrap CI over the per-seed final rates (matches the driver's own
    resampler; ``bootstrap_ci`` in ``stats`` uses the same convention)."""
    rng = np.random.default_rng(seed)
    b = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _lh_summarize(results, L_grid, A_grid, episodes, D, tau_leak, seeds):
    """Package raw long-horizon cells into the exp15 grid dict (per-(L, A, cond) mean +
    bootstrap CI)."""
    acc = {}
    for L, A, cond, s, val in results:
        acc.setdefault((L, A, cond), []).append(val)
    summary = {}
    for L in L_grid:
        for A in A_grid:
            for cond in LONG_HORIZON_CONDS:
                v = np.array(acc[(L, A, cond)])
                lo, hi = _lh_boot_ci(v, seed=L * 100 + A * 10)
                summary[(L, A, cond)] = (float(v.mean()), lo, hi)
    return {"L": L_grid, "A": A_grid, "conds": list(LONG_HORIZON_CONDS),
            "summary": summary, "episodes": episodes, "D": D,
            "tau_leak": tau_leak, "seeds": seeds}


def run_long_horizon(*, L_grid=(3, 5, 8, 12), A_grid=(2, 3), seeds=12,
                     episodes=2500, D=2.0, tau_leak=10.0, dt=5e-3):
    """Experiment 15: longer-horizon temporal-credit feasibility, serial.

    Sweeps trajectory length ``L`` x arm count ``A_goal`` x condition x seed and reports
    the final reward rate (mean + bootstrap CI over seeds) for the device trace against
    the eprop / abstract / no_trace baselines. Each seed is an independent ``B=1``,
    ``seed0=seed`` run, so the serial sweep here matches the pooled sweep in ``main``.
    Returns the result dict; no file I/O.
    """
    L_grid = list(L_grid); A_grid = list(A_grid)
    jobs = [(L, A, cond, s, episodes, D, tau_leak, dt)
            for L in L_grid for A in A_grid
            for cond in LONG_HORIZON_CONDS for s in range(seeds)]
    results = [_lh_run(j) for j in jobs]
    return _lh_summarize(results, L_grid, A_grid, episodes, D, tau_leak, seeds)


# -----------------------------------------------------------------------------
# Experiment 15-faults -- longer-horizon temporal credit UNDER the SiO_x device-fault
# prior. The fault-robustness companion to ``run_long_horizon``.
#
# The maze weights are a SINGLE NON-NEGATIVE conductance per synapse in ``[0, W_MAX]``
# (not the signed differential pair of the deep net), so ``device_faults.maze_fault_stack``
# applies the SAME fault classes (stuck-off, lognormal D2D; Poole-Frenkel I-V excluded as a
# read-nonlinearity out of scope for a programmed-weight learning claim) through the correct
# ``[0, W_MAX] <-> [G_off, G_on]`` mapping. Per (L, p) cell we report the device trace's
# final reward rate (mean + bootstrap CI) under faults beside the clean no_trace control;
# the headline question is graceful degradation as the stuck fraction p rises.
# -----------------------------------------------------------------------------
LONG_HORIZON_SIGMA_G = 0.5                    # lognormal D2D width (matches exp14)


def _lhf_run(job):
    """One (L, p, seed, trace) fault cell -> (L, p, seed, trace, final rate). Top-level
    for multiprocessing; ``B=1`` and ``seed0=seed`` so seeds are independent. ``trace``
    True = device trace under the SiO_x programmed-conductance fault prior (stuck-off +
    D2D, read-time; PF off); False = no-trace control (faults moot with no trace)."""
    from .device_faults import maze_fault_stack
    L, p, seed, trace, A, episodes, D, tau_leak = job
    env = TMaze(L=L, A_goal=A)
    if trace:
        fault = maze_fault_stack(p_stuck=p, sigma_g=LONG_HORIZON_SIGMA_G, pf_on=False,
                                 stuck_kind="off", w_max=W_MAX,
                                 seed=1000 * seed + int(100 * p))
        r = train_sequential(env, B=1, tau_leak=tau_leak, D=D, episodes=episodes,
                             seed0=seed, weight_fault=fault)
    else:
        r = train_sequential(env, B=1, tau_leak=tau_leak, D=D, episodes=episodes,
                             seed0=seed, no_trace=True)
    return (L, p, seed, trace, _lh_final_rate(r))


def _lhf_summarize(results, L_grid, p_grid, A, episodes, D, tau_leak, seeds):
    """Package raw fault cells into the exp15-faults grid dict (device mean + bootstrap
    CI grid, clean no-trace control grid)."""
    dev = {(L, p): [] for L in L_grid for p in p_grid}
    ctl = {(L, p): [] for L in L_grid for p in p_grid}
    for L, p, s, trace, val in results:
        (dev if trace else ctl)[(L, p)].append(val)
    chance = 1.0 / A
    grid = np.zeros((len(L_grid), len(p_grid), 3))
    cgrid = np.zeros((len(L_grid), len(p_grid)))
    for i, L in enumerate(L_grid):
        for j, p in enumerate(p_grid):
            d = np.array(dev[(L, p)]); c = np.array(ctl[(L, p)])
            lo, hi = _lh_boot_ci(d, seed=i * 10 + j)
            grid[i, j] = (d.mean(), lo, hi); cgrid[i, j] = c.mean()
    return {"L": L_grid, "p": p_grid, "A": A, "grid": grid, "ctrl": cgrid,
            "chance": chance,
            "faults": "stuck-off + D2D(0.5) [PF excluded] (maze mapping)",
            "episodes": episodes, "D": D, "tau_leak": tau_leak, "seeds": seeds}


def run_long_horizon_faults(*, L_grid=(3, 5, 8, 12), p_grid=(0, 0.1, 0.2, 0.5),
                            A=2, seeds=12, episodes=2500, D=2.0, tau_leak=12.0):
    """Experiment 15-faults: longer-horizon temporal credit under the SiO_x fault prior.

    Sweeps trajectory length ``L`` x stuck-device fraction ``p``, comparing the device
    eligibility trace under the SiO_x programmed-conductance fault prior (stuck-off +
    lognormal D2D at read time; Poole-Frenkel excluded) against the clean no-trace
    control. Each seed is an independent ``B=1``, ``seed0=seed`` run; serial here, pooled
    in ``main``. Returns the result dict; no file I/O.
    """
    L_grid = list(L_grid); p_grid = list(p_grid)
    jobs = [(L, p, s, trace, A, episodes, D, tau_leak)
            for L in L_grid for p in p_grid
            for s in range(seeds) for trace in (True, False)]
    results = [_lhf_run(j) for j in jobs]
    return _lhf_summarize(results, L_grid, p_grid, A, episodes, D, tau_leak, seeds)


def main(argv=None):
    """Full-scale reproduction CLI for the sequential-maze grids (writes ``data/results``).

    ``python -m mrl_trace.maze [--exp6] [--exp8] [--exp15] [--exp15-faults] [--full|--quick]``

    With no experiment flag, runs all four. ``--full`` = the published scale (exp6/exp8:
    20 seeds; exp15/exp15-faults: 12 seeds); ``--quick`` = a fast few-seed smoke run.
    ``main`` parallelises the coarse axis over a process pool (it runs as ``python -m``,
    a real ``__main__``); the ``run_*`` cores themselves stay serial for in-notebook use.
    """
    import argparse
    import os
    from functools import partial
    from multiprocessing import Pool
    from . import paths

    ap = argparse.ArgumentParser(description="Sequential-maze credit-assignment reproductions")
    ap.add_argument("--exp6", action="store_true",
                    help="sequential T-maze + benchmark grid -> exp6_sequential.npy")
    ap.add_argument("--exp8", action="store_true",
                    help="dense D_max(tau_leak) law -> exp8_dmax_law.npy")
    ap.add_argument("--exp15", action="store_true",
                    help="longer-horizon feasibility -> exp15_long_horizon.npy")
    ap.add_argument("--exp15-faults", dest="exp15_faults", action="store_true",
                    help="longer-horizon under the SiOx fault prior -> exp15_long_horizon_faults.npy")
    ap.add_argument("--quick", action="store_true", help="fast few-seed smoke run")
    ap.add_argument("--full", action="store_true", help="published-scale run (default)")
    a = ap.parse_args(argv)
    run_all = not (a.exp6 or a.exp8 or a.exp15 or a.exp15_faults)

    n_proc = max(1, (os.cpu_count() or 4) - 2)

    # ---- Experiment 6: sequential T-maze + benchmark grid ----
    if a.exp6 or run_all:
        seeds = 6 if a.quick else 20
        episodes = 400 if a.quick else 800
        V, D0, TAU0 = 1.5, 4.0, 10.0
        taus = [2.0, 5.0, 20.0]
        delays = [2, 4, 8, 16, 32, 64]
        print(f"=== exp6 sequential T-maze (N={seeds}, {episodes} ep, V={V}) ===", flush=True)
        conds = {"device": dict(), "rstdp": dict(abstract=True),
                 "eprop": dict(eprop=True, eta=EPROP_ETA), "no_trace": dict(no_trace=True)}
        specs = []
        for name, kw in conds.items():                       # learning-curve runs
            specs.append((("curve", name), TAU0, D0, kw, False))
        specs.append((("weights", "device"), TAU0, D0, dict(), True))   # policy run
        for tau in taus:                                     # grid cells
            for D in delays:
                specs.append((("grid", tau, D), tau, D, dict(), False))
        with Pool(min(len(specs), n_proc)) as pool:
            res = dict(pool.map(partial(_exp6_worker, seeds=seeds, episodes=episodes, V=V),
                                specs))
        grid = _exp6_assemble(res, conds, taus, delays, seeds, episodes, V, D0, TAU0)
        paths.save_result("exp6_sequential.npy", grid)
        print(f"  wrote exp6_sequential.npy  criteria={grid['criteria']}", flush=True)

    # ---- Experiment 8: dense D_max(tau_leak) law ----
    if a.exp8 or run_all:
        seeds = 6 if a.quick else 20
        episodes = 400 if a.quick else 800
        V, crit = 1.5, 0.75
        taus, delays = DMAX_TAUS, DMAX_DELAYS
        print(f"=== exp8 dense D_max(tau_leak): {len(taus)} tau x {len(delays)} delays, "
              f"{seeds} seeds, {episodes} ep, V={V}, crit={crit} ===", flush=True)
        specs = [(tau, D) for tau in taus for D in delays]
        with Pool(min(len(specs), n_proc)) as pool:
            results = pool.map(partial(_dmax_cell, episodes=episodes, V=V, seeds=seeds),
                               specs)
        grid = _dmax_assemble(results, taus, delays, crit, seeds, episodes, V)
        paths.save_result("exp8_dmax_law.npy", grid)
        print(f"  wrote exp8_dmax_law.npy  D_max = {grid['k']:.2f} tau (R^2={grid['r2']:.3f}), "
              f"plateau slope ~{grid['k_plateau']:.2f}", flush=True)

    # ---- Experiment 15: longer-horizon feasibility ----
    if a.exp15 or run_all:
        seeds = 4 if a.quick else 12
        episodes = 400 if a.quick else 2500
        L_grid = [3, 5] if a.quick else [3, 5, 8, 12]
        A_grid = [2] if a.quick else [2, 3]
        D, tau_leak = 2.0, 10.0
        jobs = [(L, A, cond, s, episodes, D, tau_leak)
                for L in L_grid for A in A_grid
                for cond in LONG_HORIZON_CONDS for s in range(seeds)]
        print(f"=== exp15 long-horizon: {len(jobs)} runs; L={L_grid} A={A_grid} "
              f"conds={list(LONG_HORIZON_CONDS)} seeds={seeds} episodes={episodes} "
              f"D={D}s tau_leak={tau_leak}s ===", flush=True)
        with Pool(min(len(jobs), n_proc)) as pool:
            results = pool.map(_lh_run, jobs, chunksize=1)
        grid = _lh_summarize(results, L_grid, A_grid, episodes, D, tau_leak, seeds)
        paths.save_result("exp15_long_horizon.npy", grid)
        print("  wrote exp15_long_horizon.npy", flush=True)

    # ---- Experiment 15-faults: longer-horizon under the SiOx fault prior ----
    if a.exp15_faults or run_all:
        seeds = 4 if a.quick else 12
        episodes = 400 if a.quick else 2500
        L_grid = [3, 5] if a.quick else [3, 5, 8, 12]
        p_grid = [0, 0.2] if a.quick else [0, 0.1, 0.2, 0.5]
        A, D, tau_leak = 2, 2.0, 12.0
        jobs = [(L, p, s, trace, A, episodes, D, tau_leak)
                for L in L_grid for p in p_grid
                for s in range(seeds) for trace in (True, False)]
        print(f"=== exp15-faults long-horizon under SiOx fault prior: {len(jobs)} runs; "
              f"L={L_grid} p={p_grid} A={A} seeds={seeds} episodes={episodes} D={D}s "
              f"tau_leak={tau_leak}s (stuck-off + D2D, non-negative maze mapping) ===",
              flush=True)
        with Pool(min(len(jobs), n_proc)) as pool:
            results = pool.map(_lhf_run, jobs, chunksize=1)
        grid = _lhf_summarize(results, L_grid, p_grid, A, episodes, D, tau_leak, seeds)
        paths.save_result("exp15_long_horizon_faults.npy", grid)
        print("  wrote exp15_long_horizon_faults.npy", flush=True)


# --- exp6 pooled workers (module-level so they pickle for the process pool) ---
def _exp6_worker(spec, seeds, episodes, V):
    """Run one independent exp6 cell for the pool. ``spec`` = (key, tau_leak, D, kwargs,
    want_weights). Returns ``(key, rewards[, weights])``. Deterministic given seed0=0."""
    key, tau_leak, D, kw, want_w = spec
    env = TMaze(L=3, A_goal=2)
    out = train_sequential(env, B=seeds, tau_leak=tau_leak, D=D, episodes=episodes,
                           V=V, seed0=0, return_weights=want_w, **kw)
    return (key, out)


def _exp6_worker_seeded(spec, seeds, episodes, V, seed0=0, device_k=K_STAGES,
                        tau_r_override=None):
    """Spawn-safe exp6 worker that retains the public runner's ``seed0`` control."""
    key, tau_leak, D, kw, want_w = spec
    env = TMaze(L=3, A_goal=2)
    out = train_sequential(env, B=seeds, tau_leak=tau_leak, D=D, episodes=episodes,
                           V=V, seed0=seed0, return_weights=want_w,
                           device_k=device_k, tau_r_override=tau_r_override, **kw)
    return (key, out)


def _exp6_assemble(res, conds, taus, delays, seeds, episodes, V, D0, TAU0):
    """Package pooled exp6 results (keyed dict) into the exp6 grid dict -- identical
    structure to ``run_sequential`` output, built from the parallel cells."""
    from .stats import bootstrap_ci
    A = 2
    chance = 1.0 / A
    crit = 0.5 * (1 + chance)
    curves, finals, finals_ci = {}, {}, {}
    for name in conds:
        r = res[("curve", name)]
        finals[name] = reward_rate(r, window=100)
        curves[name] = running_rate(r.mean(0), window=50)
        finals_ci[name] = bootstrap_ci(finals[name])
    _, w_dev = res[("weights", "device")]
    env = TMaze(L=3, A_goal=2)
    pol = np.array([policy_correct(env, w_dev[b]) for b in range(seeds)])
    grid = np.zeros((len(taus), len(delays)))
    grid_ci = np.zeros((len(taus), len(delays), 2))
    for i, tau in enumerate(taus):
        for j, D in enumerate(delays):
            f = reward_rate(res[("grid", tau, D)], 100)
            grid[i, j] = f.mean()
            grid_ci[i, j] = bootstrap_ci(f)
    dmax = {}
    for i, tau in enumerate(taus):
        ok = [D for j, D in enumerate(delays) if grid[i, j] >= crit]
        dmax[tau] = max(ok) if ok else 0
    c1 = bool(finals["no_trace"].mean() <= chance + 0.10)
    c2 = bool(finals["device"].mean() >= crit)
    c3 = bool(pol.mean() >= crit)
    c4 = bool((dmax[20.0] >= dmax[5.0] >= dmax[2.0]) and (dmax[20.0] > dmax[2.0])) \
        if set((2.0, 5.0, 20.0)).issubset(dmax) else None
    dev_beats_nt = bool(any(grid[i, j] > finals["no_trace"].mean() + 0.1
                            for i in range(len(taus)) for j in range(len(delays))))
    criteria = {"C1_notrace_at_chance": c1, "C2_device_criterion": c2,
                "C3_device_policy": c3, "C4_dmax_grows_with_tau": c4,
                "K1_device_exceeds_notrace": dev_beats_nt}
    return {"finals": {k: v for k, v in finals.items()}, "finals_ci": finals_ci,
            "curves": curves, "policy": pol, "policy_ci": bootstrap_ci(pol),
            "taus": taus, "delays": delays, "grid": grid, "grid_ci": grid_ci,
            "dmax": dmax, "chance": chance, "crit": crit,
            "D0": D0, "TAU0": TAU0, "V": V, "seeds": seeds, "episodes": episodes,
            "criteria": criteria}


if __name__ == "__main__":
    main()
