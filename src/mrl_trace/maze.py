"""Delayed-reward tasks and explicitly scoped comparator adaptations.

The primary environment in this module is :class:`ActionSequenceTrack`: four
observable states require the action sequence ``(0, 1, 1, 0)`` and reward is
delivered only after every decision is correct.  Thus every transition is a learned
decision and a constant-action policy cannot solve the task.

The historical ``TMaze`` implementation is retained as
:class:`DelayedCuedChoice`.  Its stem advances automatically and only its junction
choice is learned, so it is a delayed contextual decision, not trajectory-level
policy learning.  ``TMaze`` remains a deprecated compatibility name.

The device and exponential conditions use this repository's proposed signed
coincidence drive.  :class:`ShallowEpropPolicyTrace` is a custom feed-forward
policy-gradient adaptation containing an e-prop-style eligibility term; it is not a
reproduction of the complete recurrent e-prop algorithm.  Result dictionaries carry
method-provenance and calibration metadata so component citations are not presented
as validation of these repository-specific combinations.
"""
from __future__ import annotations

import warnings

import numpy as np

from .bandit import (
    GateBankBatched, LinearErlangGateBankBatched, AbstractTrace, W_INIT, W_MAX,
)
from .device import K_STAGES, decay_matched_exponential_tau
from .neurons import lif_step_batched, TAU_M, V_TH
from .learning import LTD_BIAS
from .model_specs import (
    PRIMARY_MODEL_ID, LINEAR_MODEL_ID, device_model_spec,
)

__all__ = [
    "LinearTrack", "ActionSequenceTrack", "DelayedCuedChoice", "TMaze",
    "ConventionalRstdpTrace", "ShallowEpropPolicyTrace", "EpropTrace",
    "train_sequential",
    "reward_rate", "trials_to_criterion", "policy_correct",
    "calibrate_comparator_scales", "tune_comparator_learning_rates",
    "run_action_sequence", "run_delayed_cued_choice", "run_sequential",
    "run_retention_delay_curve", "run_dmax_law",
    "run_retention_delay_curve_adaptive", "run_dmax_adaptive",
    "run_long_horizon", "run_long_horizon_faults",
    # analysis helpers
    "running_rate", "interp_dmax",
]


PROVENANCE_STATUS = frozenset(
    {"established", "adapted", "proposed", "empirical_fit", "extrapolated"}
)
RETENTION_DEFINITIONS = frozenset({
    "measured_held_bias",
    "measured_held_bias_quantiles",
    "measured_near_zero_field",
    "extrapolated",
    "deliberately_swept",
})


def _provenance(status, established_basis, repository_adaptation, claim_limit):
    """Return a fresh, schema-stable method-provenance record."""
    if status not in PROVENANCE_STATUS:
        raise ValueError(f"unknown provenance status: {status!r}")
    return {
        "status": status,
        "established_basis": list(established_basis),
        "repository_adaptation": str(repository_adaptation),
        "claim_limit": str(claim_limit),
    }


def _retention_definition(value):
    value = str(value)
    if value not in RETENTION_DEFINITIONS:
        raise ValueError(
            f"retention_definition must be one of {sorted(RETENTION_DEFINITIONS)}, "
            f"got {value!r}"
        )
    return value


METHOD_PROVENANCE = {
    "device": _provenance(
        "proposed",
        ["three-factor reward modulation", "local eligibility traces"],
        "Repository-specific signed coincidence filtered by an approximate cascade gate.",
        "A computational device-trace condition, not an exact cited R-STDP rule or a "
        "microscopically identified trap cascade.",
    ),
    "linear_device": _provenance(
        "adapted",
        ["linear Erlang cascade", "three-factor reward modulation"],
        "Linear Erlang-exact sensitivity using the same signed coincidence drive.",
        "Computational sensitivity only; it omits nonlinear physical headroom.",
    ),
    "exponential": _provenance(
        "adapted",
        ["exponential eligibility traces", "three-factor reward modulation"],
        "The same repository-specific signed coincidence drive with an exponential filter.",
        "Its single time constant is fitted to the cascade surrogate's predefined "
        "post-peak decay band; it is not a verbatim reproduction of a cited R-STDP "
        "method and cannot reproduce the full stretched discharge shape.",
    ),
    "shallow_eprop": _provenance(
        "adapted",
        ["e-prop-style per-synapse eligibility", "policy-gradient learning signal"],
        "Feed-forward action layer with a custom reward-modulated policy-gradient readout.",
        "Does not reproduce recurrent e-prop, its full learning-signal machinery, or its benchmarks.",
    ),
    "conventional_rstdp": _provenance(
        "adapted",
        ["pair-based STDP", "exponential eligibility", "reward modulation"],
        "Conventional pre/post pair traces feed a decaying eligibility that is gated by reward.",
        "Reference three-factor R-STDP implementation; not a reproduction of every "
        "network or protocol detail in any one cited paper.",
    ),
    "no_trace": _provenance(
        "established",
        ["necessity ablation"],
        "Eligibility is set identically to zero in the shared task harness.",
        "Tests dependence on a trace only; it is not a competitive learning algorithm.",
    ),
    "signed_coincidence": _provenance(
        "proposed",
        ["local pre/post coincidence", "reward-modulated plasticity"],
        "Depression-biased signed drive: coincident postsynaptic spikes are positive and "
        "presynaptic-only events are negative.",
        "Custom Eq. 6 rule; the component literature does not establish this exact update.",
    ),
}


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

    def start(self, B, rng=None):
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


class ActionSequenceTrack:
    """Multi-decision track with terminal reward after a required action sequence.

    By default the four observable states require actions ``(0, 1, 1, 0)``.  A
    correct action advances one state; any wrong action terminates the episode
    unrewarded.  The final correct action reaches the goal, after which the shared
    harness applies the configured action--reward delay.  Since both actions occur
    in the target sequence, neither constant policy can solve the default task.
    """

    def __init__(self, required_actions=(0, 1, 1, 0), n_actions=None):
        actions = tuple(int(a) for a in required_actions)
        if not actions:
            raise ValueError("required_actions must contain at least one action")
        if min(actions) < 0:
            raise ValueError("required actions must be non-negative integers")
        if len(set(actions)) < 2:
            raise ValueError("the action sequence must require at least two actions")
        self.required_actions = actions
        self.L = len(actions)
        self.n_states = self.L
        minimum_actions = max(2, max(actions) + 1)
        self.n_actions = minimum_actions if n_actions is None else int(n_actions)
        if self.n_actions < minimum_actions:
            raise ValueError(
                f"n_actions must be at least {minimum_actions} for this sequence"
            )
        self.goal_action = actions[-1]

    def start(self, B, rng=None):
        return np.zeros(B, dtype=int)

    def step(self, pos, action):
        pos = np.asarray(pos, dtype=int)
        action = np.asarray(action, dtype=int)
        if pos.shape != action.shape:
            raise ValueError("pos and action must have matching shapes")
        expected = np.asarray(self.required_actions, dtype=int)[pos]
        correct = action == expected
        final = pos == self.L - 1
        reached = correct & final
        done = (~correct) | reached
        nxt = np.where(correct & ~final, pos + 1, pos)
        return nxt, reached, done

    def correct_action(self, pos):
        pos = np.asarray(pos, dtype=int)
        return np.asarray(self.required_actions, dtype=int)[pos]

    def correct_action_for_state(self, state):
        return self.required_actions[int(state)]


class DelayedCuedChoice:
    """Auto-advanced stem followed by one learned, cue-dependent junction choice.

    States ``0..L-1`` are the stem (advance with action 0). At the junction state
    ``L-1`` the agent's action selects an arm ``0..A_goal-1``; reward is contingent on
    the arm matching the episode's rewarded arm (set by a cue presented at the start).
    ``n_actions = max(2, A_goal)`` so the same action neurons serve stem-advance and
    arm-choice.  Stem actions do not affect transitions and the cue is exposed only
    in the junction-state encoding.  This is therefore a delayed contextual choice,
    not a multi-decision T-maze navigation task.
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

    def correct_action_for_state(self, state):
        state = int(state)
        return 0 if state < self.L else state - self.L


class TMaze(DelayedCuedChoice):
    """Deprecated compatibility name for :class:`DelayedCuedChoice`."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TMaze was a one-choice delayed contextual task, not a sequential "
            "T-maze. Use DelayedCuedChoice for the historical control or "
            "ActionSequenceTrack for multi-decision credit assignment.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# ----------------------------------------------------------------------------
# Training harness shared across device and explicit comparator implementations.
# ----------------------------------------------------------------------------
class ShallowEpropPolicyTrace:
    """Custom shallow policy trace containing an e-prop-style eligibility term.

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
    product psi*zbar. This repository-specific feed-forward adaptation does not
    reproduce the recurrent network, learning-signal machinery, or complete method of
    Bellec et al.; it is therefore an e-prop-style comparator rather than "e-prop".
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


class EpropTrace(ShallowEpropPolicyTrace):
    """Deprecated compatibility name for :class:`ShallowEpropPolicyTrace`."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "EpropTrace is a custom shallow policy-gradient adaptation, not the "
            "complete e-prop method; use ShallowEpropPolicyTrace.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


class ConventionalRstdpTrace:
    """Pair-based reward-modulated STDP eligibility with exponential decay.

    Presynaptic events depress in proportion to the recent postsynaptic trace and
    postsynaptic events potentiate in proportion to the recent presynaptic trace.
    The resulting signed pair term feeds a slower eligibility trace.  Reward gates
    that trace in :func:`train_sequential`.  This supplies a conventional comparator
    distinct from the repository's proposed signed-coincidence construction.
    """

    def __init__(self, B, S, A, tau_leak=10.0, dt=5e-3, tau_pair=20e-3,
                 a_plus=1.0, a_minus=1.05):
        if tau_leak <= 0 or tau_pair <= 0:
            raise ValueError("trace time constants must be positive")
        self.B, self.S, self.A, self.dt = B, S, A, float(dt)
        self.tau = float(tau_leak)
        self.tau_pair = float(tau_pair)
        self.a_plus, self.a_minus = float(a_plus), float(a_minus)
        self.pre_trace = np.zeros((B, S))
        self.post_trace = np.zeros((B, A))
        self.elig = np.zeros((B, S, A))
        self.vn = np.zeros((B, S, A, 1))
        self.Vnmax = 1.0

    def reset(self):
        self.pre_trace[:] = 0.0
        self.post_trace[:] = 0.0
        self.elig[:] = 0.0
        self.vn[:] = 0.0

    def step_rstdp(self, pre, post, active):
        pair_decay = np.exp(-self.dt / self.tau_pair)
        elig_decay = np.exp(-self.dt / self.tau)
        self.pre_trace *= pair_decay
        self.post_trace *= pair_decay
        # Use traces from preceding events for the pair terms, then register the
        # current events. Simultaneous events therefore do not self-pair.
        pair = (
            self.a_plus * self.pre_trace[:, :, None] * post[:, None, :]
            - self.a_minus * pre[:, :, None] * self.post_trace[:, None, :]
        )
        self.pre_trace += pre
        self.post_trace += post
        self.elig = elig_decay * self.elig + pair
        self.pre_trace *= active[:, None]
        self.post_trace *= active[:, None]
        self.elig *= active[:, None, None]
        self.vn[..., 0] = self.elig

    def relax(self, duration):
        duration = max(0.0, float(duration))
        self.pre_trace *= np.exp(-duration / self.tau_pair)
        self.post_trace *= np.exp(-duration / self.tau_pair)
        self.elig *= np.exp(-duration / self.tau)
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
    of :func:`_relax` for :class:`ShallowEpropPolicyTrace`, whose decay is equally
    smooth with no input)."""
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
                     eprop=False, rstdp=False, beta_pol=1.0, beta_leak=1.0,
                     weight_fault=None, seed0=0,
                     return_weights=False, device_k=K_STAGES, tau_r_override=None,
                     coincidence_mode="signed", eligibility_normalizer=1.0,
                     forced_actions=None, return_diagnostics=False,
                     gate_model=PRIMARY_MODEL_ID):
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

    ``abstract`` swaps the device gate for an exponential filter of the same custom
    signed drive. ``rstdp`` instead uses conventional pre/post pair traces and a
    decaying reward-gated eligibility, independently of the custom coincidence rule.
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

    ``coincidence_mode`` makes the custom-rule ablations explicit: ``"signed"`` is
    the proposed depression-biased drive, ``"unsigned"`` takes its absolute value,
    and ``"no_negative"`` retains positive pre/post coincidences only.
    ``eligibility_normalizer`` is a frozen positive scale obtained from the separate
    calibration batch; it is never estimated from evaluation trials.
    """
    if coincidence_mode not in {"signed", "unsigned", "no_negative"}:
        raise ValueError(
            "coincidence_mode must be 'signed', 'unsigned', or 'no_negative'"
        )
    eligibility_normalizer = float(eligibility_normalizer)
    if not np.isfinite(eligibility_normalizer) or eligibility_normalizer <= 0:
        raise ValueError("eligibility_normalizer must be finite and positive")
    if sum(bool(x) for x in (eprop, rstdp, abstract)) > 1:
        raise ValueError("eprop, rstdp, and abstract are mutually exclusive")
    rng = np.random.default_rng(seed0)
    S, A = env.n_states, env.n_actions
    matched_tau = None
    if max_steps is None:
        max_steps = 3 * env.L + 2
    if eprop:
        bank = ShallowEpropPolicyTrace(
            B, S, A, tau_leak=tau_leak, dt=dt, tau_m=tau_m, v_th=v_th
        )
    elif rstdp:
        bank = ConventionalRstdpTrace(B, S, A, tau_leak=tau_leak, dt=dt)
    elif abstract:
        matched_tau = decay_matched_exponential_tau(
            tau_leak, V=V, k=device_k, tau_r_override=tau_r_override,
            beta_leak=beta_leak, gate_model=gate_model,
        )
        bank = AbstractTrace(B, S, A, tau_elig=matched_tau, dt=dt)
    else:
        bank_cls = {
            PRIMARY_MODEL_ID: GateBankBatched,
            LINEAR_MODEL_ID: LinearErlangGateBankBatched,
        }.get(gate_model)
        if bank_cls is None:
            raise ValueError(f"unknown gate_model: {gate_model!r}")
        bank = bank_cls(B, S, A, tau_leak=tau_leak, dt=dt, V=V,
                        beta_leak=beta_leak, k=device_k,
                        tau_r_override=tau_r_override)
    w = np.full((B, S, A), W_INIT)
    baseline = np.full(B, 1.0 / A)
    bidx = np.arange(B)
    steps_per_state = max(1, int(round(step_dur / dt)))
    reward_lag = int(round(D / dt))
    rewards = np.zeros((B, episodes))
    if forced_actions is not None:
        forced_actions = np.asarray(forced_actions, dtype=int)
        if forced_actions.shape != (episodes, B, max_steps):
            raise ValueError(
                "forced_actions must have shape "
                f"(episodes, B, max_steps)={(episodes, B, max_steps)}, got "
                f"{forced_actions.shape}"
            )
        if np.any((forced_actions < 0) | (forced_actions >= A)):
            raise ValueError("forced_actions contains an action outside the environment")
    diag_peak = []
    diag_area = []
    diag_reward_rms = []

    def _current_eligibility():
        if no_trace:
            return np.zeros((B, S, A))
        if eprop:
            return bank.vn[..., -1]
        if rstdp:
            return bank.elig
        if abstract:
            return bank.e
        return bank.vn[..., -1] / bank.Vnmax

    for ep in range(episodes):
        sigma = sigma0 if sigma1 is None else sigma0 + (sigma1 - sigma0) * ep / episodes
        # Read-time device faults applied ONCE per episode: the array reads the faulted
        # conductance (wr) while learning updates the clean weight (w). Computing this per
        # episode (not per dt-tick) is what keeps the run tractable.
        wr = weight_fault(w) if weight_fault is not None else w
        bank.reset()
        pos = env.start(B, rng)
        v = np.zeros((B, A))
        done = np.zeros(B, dtype=bool)
        got_reward = np.zeros(B, dtype=bool)
        # accumulated eligibility snapshot, captured at each seed's reward instant
        e_rew = np.zeros((B, S, A))
        reward_due = np.full(B, -1)         # step index at which reward gates in
        # Per-state log-policy scores for the custom shallow e-prop-style comparator.
        # Unlike the historical terminal-only implementation, every consequential
        # decision contributes its own (1[a] - pi) factor.
        policy_score = np.zeros((B, S, A))
        trace_area = 0.0
        # --- run the trajectory ---
        for st in range(max_steps):
            state = env.encode(pos) if hasattr(env, "encode") else pos
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
                elif rstdp:
                    bank.step_rstdp(pre, sp, active)
                else:
                    # signed leak-dominant coincidence on the CURRENT state's lines
                    # (device and exponential controls share this custom drive)
                    drive = np.zeros((B, S, A))
                    if coincidence_mode == "signed":
                        coincidence = np.where(sp, 1.0, -ltd)
                    elif coincidence_mode == "unsigned":
                        coincidence = np.where(sp, 1.0, ltd)
                    else:  # remove the negative/depression term
                        coincidence = sp.astype(float)
                    drv = pre[bidx, state][:, None] * coincidence
                    drive[bidx, state, :] = drv * active[:, None]
                    if no_trace:
                        drive[:] = 0.0
                    bank.step(drive)
                if return_diagnostics:
                    trace_area += float(np.mean(np.abs(_current_eligibility()))) * dt
            if eprop:
                # Reward-based e-prop is a policy-gradient method: the action is SAMPLED
                # from a softmax policy over the action-neuron spike counts (this sampling
                # is the exploration), and the policy pi is stored so the per-action
                # learning signal (1[a=chosen] - pi_a) can be formed at reward.
                logits = beta_pol * (spk - spk.mean(1, keepdims=True))
                pol_p = np.exp(logits - logits.max(1, keepdims=True))
                pol_p /= pol_p.sum(1, keepdims=True)
                if forced_actions is None:
                    cdf = np.cumsum(pol_p, axis=1)
                    u = rng.random((B, 1))
                    chosen = (u > cdf).sum(1)
                    chosen = np.clip(chosen, 0, A - 1)
                else:
                    # Fixed calibration actions are chosen before any comparator
                    # runs. Do not consume policy-sampling draws and overwrite them,
                    # because that would shift later stimulus/noise RNG draws.
                    chosen = forced_actions[ep, :, st].copy()
            else:
                if forced_actions is None:
                    # action = spiking winner over the dwell (ties -> random)
                    tie = spk.max(1) == spk.min(1)
                    chosen = np.argmax(spk, 1)
                    chosen[tie] = rng.integers(A, size=int(tie.sum()))
                else:
                    # Likewise bypass random tie-breaking in calibration.
                    chosen = forced_actions[ep, :, st].copy()
            chosen[done] = 0
            if eprop:
                active_decision = ~done
                onehot_step = np.zeros((B, A))
                onehot_step[bidx, chosen] = 1.0
                policy_score[
                    bidx[active_decision], state[active_decision], :
                ] += (onehot_step - pol_p)[active_decision]
            nxt, reached, ep_done = env.step(pos, chosen)
            pos = np.where(done, pos, nxt)
            newly = (~done) & ep_done
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
            if return_diagnostics and n_relax:
                # Smooth decay is coarsened by ``stride`` exactly as in the state
                # update; integrate the diagnostic using its elapsed wall time.
                trace_area += float(np.mean(np.abs(_current_eligibility()))) * dt * stride * n_relax
            for _ in range(rem):
                bank.step_eprop(np.zeros((B, S)), np.zeros((B, A)), np.ones(B))
                if return_diagnostics:
                    trace_area += float(np.mean(np.abs(_current_eligibility()))) * dt
            e_rew = bank.vn[..., -1]
        elif rstdp:
            bank.relax(reward_lag * dt)
            if return_diagnostics:
                trace_area += (
                    float(np.mean(np.abs(_current_eligibility()))) * reward_lag * dt
                )
            e_rew = bank.elig
        elif no_trace:
            e_rew = np.zeros((B, S, A))
        elif abstract:
            for _ in range(reward_lag):
                bank.step(zero)
                if return_diagnostics:
                    trace_area += float(np.mean(np.abs(_current_eligibility()))) * dt
            e_rew = bank.e
        else:
            _relax(bank, zero, n_relax, stride, abstract)
            if return_diagnostics and n_relax:
                trace_area += float(np.mean(np.abs(_current_eligibility()))) * dt * stride * n_relax
            for _ in range(rem):
                bank.step(zero)
                if return_diagnostics:
                    trace_area += float(np.mean(np.abs(_current_eligibility()))) * dt
            e_rew = bank.vn[..., -1] / bank.Vnmax
        R = got_reward.astype(float)
        if eprop:
            # Reward-based e-prop = policy gradient with an e-prop eligibility. The
            # The custom reward learning signal is the per-state REINFORCE
            # log-policy score recorded at each decision. This is a repository
            # adaptation, not Bellec et al.'s recurrent learning-signal machinery.
            raw_update = (
                (R - baseline)[:, None, None] * policy_score * e_rew
            )
            w = np.clip(
                w + eta * raw_update / eligibility_normalizer, 0.0, W_MAX
            )
        else:
            raw_update = (R - baseline)[:, None, None] * e_rew
            w = np.clip(
                w + eta * raw_update / eligibility_normalizer, 0.0, W_MAX
            )
        if return_diagnostics:
            diag_peak.append(float(np.max(np.abs(e_rew))))
            diag_area.append(float(trace_area))
            diag_reward_rms.append(float(np.sqrt(np.mean(raw_update ** 2))))
        baseline += 0.02 * (R - baseline)
        rewards[:, ep] = R
    diagnostics = None
    if return_diagnostics:
        raw_rms = float(np.sqrt(np.mean(np.square(diag_reward_rms))))
        diagnostics = {
            "raw_trace_peak": float(np.mean(diag_peak)),
            "raw_trace_area": float(np.mean(diag_area)),
            "raw_effective_update_rms": raw_rms,
            "eligibility_normalizer": eligibility_normalizer,
            "normalized_effective_update_rms": raw_rms / eligibility_normalizer,
            "coincidence_mode": coincidence_mode,
            "gate_model": gate_model if not (eprop or rstdp or no_trace) else None,
            "matched_exponential_tau_s": (
                None if matched_tau is None else float(matched_tau)
            ),
        }
    if return_weights and return_diagnostics:
        return rewards, w, diagnostics
    if return_weights:
        return rewards, w          # w: (B, S, A) final learned weights
    if return_diagnostics:
        return rewards, diagnostics
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
    csum = np.r_[0.0, np.cumsum(rew_1d)]
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
        if hasattr(env, "correct_action_for_state"):
            ca = env.correct_action_for_state(s)
        else:
            ca = 0                          # LinearTrack: always forward
        if w[s].max() == w[s].min():
            continue                       # untrained / tied row -> skip
        if int(np.argmax(w[s])) == int(ca):
            correct += 1
        total += 1
    return correct / total if total else 0.0


# =============================================================================
# Experiment cores for the corrected action-sequence task and historical controls.
#
# Each ``run_*`` returns the result grid as a plain dict -- no file I/O, no
# plotting, no stdout.  Notebooks call these at a small (quick) seed/episode
# count and render the figures inline; ``main()`` (below) calls them at the
# published scale, parallelising the coarse axis over a process pool, and writes
# each grid under ``data/results/`` via ``paths.save_result`` for full-cache mode.
# The original driver scripts each computed one grid:
#   * exp6_action_sequence.npy       -- calibrated multi-decision benchmark;
#   * exp8_retention_delay_curve.npy -- simulated retention x delay design curve;
#   * exp15_long_horizon.npy         -- longer-horizon action-sequence credit
#                                       feasibility vs the established traces;
#   * exp15_long_horizon_faults.npy  -- the same under the SiO_x device-fault prior.
# =============================================================================


# -----------------------------------------------------------------------------
# Historical experiment 6 -- one-choice delayed contextual control. Four conditions
# share one harness, but their old learning-rate scales were not calibrated:
#   device   : the SiOx trap-cascade eligibility gate (this work);
#   exponential: exponential filter of the same custom signed drive;
#   shallow_eprop: custom feed-forward policy trace with an e-prop-style term;
#   no_trace : eligibility zeroed (device-necessity control).
# This historical runner is preserved only to reproduce the delayed-choice control.
# Its comparator scales were not calibrated, so its between-method differences must
# not be used as algorithm-superiority evidence.  ``run_action_sequence`` is the
# corrected, calibrated multi-decision benchmark.
# -----------------------------------------------------------------------------
EPROP_ETA = 2000.0        # legacy uncalibrated value; retained for archive compatibility


def running_rate(r1d, window=50):
    """Running ``window``-episode reward rate of a single mean-over-seeds curve."""
    c = np.cumsum(np.insert(r1d, 0, 0.0))
    return (c[window:] - c[:-window]) / window


LEARNING_RATE_GRID = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
TUNING_SEEDS = tuple(range(1000, 1020))
EVALUATION_SEEDS = tuple(range(2000, 2020))
CALIBRATION_TRAJECTORIES = 256


def _copy_provenance(record):
    return {
        "status": record["status"],
        "established_basis": list(record["established_basis"]),
        "repository_adaptation": record["repository_adaptation"],
        "claim_limit": record["claim_limit"],
    }


def _comparator_conditions(include_rule_ablations=True):
    conditions = {
        "device": {"gate_model": PRIMARY_MODEL_ID},
        "linear_device": {"gate_model": LINEAR_MODEL_ID},
        "exponential": {"abstract": True, "gate_model": PRIMARY_MODEL_ID},
        "conventional_rstdp": {"rstdp": True},
        "shallow_eprop": {"eprop": True},
        "no_trace": {"no_trace": True},
    }
    if include_rule_ablations:
        conditions.update({
            "device_unsigned": {"coincidence_mode": "unsigned"},
            "device_no_negative": {"coincidence_mode": "no_negative"},
        })
    return conditions


def _condition_provenance(name):
    name = {"rstdp": "exponential", "eprop": "shallow_eprop"}.get(name, name)
    if name in {"device_unsigned", "device_no_negative"}:
        mode = "absolute signed drive" if name.endswith("unsigned") else "positive coincidences only"
        return _provenance(
            "proposed",
            ["local coincidence", "ablation analysis"],
            f"Ablation of the custom signed rule using {mode}.",
            "Attributes behaviour to the repository-specific negative term; not an established method.",
        )
    return _copy_provenance(METHOD_PROVENANCE[name])


def _balanced_calibration_actions(env, trajectories, max_steps):
    """Fixed 50/50 successful/unsuccessful action schedules for scale calibration."""
    trajectories = int(trajectories)
    if trajectories < 2 or trajectories % 2:
        raise ValueError("calibration trajectories must be a positive even integer")
    actions = np.zeros((1, trajectories, max_steps), dtype=int)
    seq = np.asarray(env.required_actions, dtype=int)
    actions[0, :, :len(seq)] = seq
    # Every trajectory reaches the final state.  Half then take the other action,
    # yielding a balanced reward signal without using evaluation outcomes.
    actions[0, trajectories // 2:, len(seq) - 1] = (
        seq[-1] + 1
    ) % env.n_actions
    return actions


def calibrate_comparator_scales(*, trajectories=CALIBRATION_TRAJECTORIES,
                                tau_leak=10.0, D=12.0, V=1.5, dt=5e-3,
                                step_dur=0.4, seed=271828,
                                include_rule_ablations=True, device_k=K_STAGES,
                                tau_r_override=None, beta_leak=1.0):
    """Measure frozen reward-time update scales on 256 balanced trajectories.

    The action schedules are fixed before fitting: half complete ``(0,1,1,0)`` and
    half differ only at the terminal decision.  Each comparator sees the same state
    sequence, inputs, outcomes, delay and seed.  The returned RMS is used only as a
    multiplicative normalizer; learning rates are selected later on disjoint tuning
    seeds.  Evaluation trials never affect either value.
    """
    env = ActionSequenceTrack()
    max_steps = env.L
    forced = _balanced_calibration_actions(env, trajectories, max_steps)
    records = {}
    for name, kwargs in _comparator_conditions(include_rule_ablations).items():
        _, diagnostics = train_sequential(
            env,
            B=int(trajectories),
            tau_leak=tau_leak,
            D=D,
            episodes=1,
            max_steps=max_steps,
            dt=dt,
            step_dur=step_dur,
            eta=0.0,
            V=V,
            seed0=int(seed),
            forced_actions=forced,
            return_diagnostics=True,
            device_k=device_k,
            tau_r_override=tau_r_override,
            beta_leak=beta_leak,
            **kwargs,
        )
        raw = diagnostics["raw_effective_update_rms"]
        normalizer = raw if np.isfinite(raw) and raw > np.finfo(float).eps else 1.0
        records[name] = {
            **diagnostics,
            "eligibility_normalizer": float(normalizer),
            "normalized_effective_update_rms": float(raw / normalizer),
            "method_provenance": _condition_provenance(name),
        }
    return {
        "protocol": "fixed_balanced_action_sequences",
        "trajectories": int(trajectories),
        "rewarded": int(trajectories) // 2,
        "unrewarded": int(trajectories) // 2,
        "required_actions": list(env.required_actions),
        "seed": int(seed),
        "tau_leak_s": float(tau_leak),
        "beta_leak": float(beta_leak),
        "model_specifications": {
            model_id: device_model_spec(model_id)
            for model_id in (PRIMARY_MODEL_ID, LINEAR_MODEL_ID)
        },
        "records": records,
        "method_provenance": _provenance(
            "adapted",
            ["effective-update scale calibration", "balanced experimental design"],
            "All comparators are measured on the same fixed rewarded/unrewarded trajectories.",
            "Calibration equalises update opportunity; it does not make the algorithms equivalent.",
        ),
    }


def _action_tuning_job(job):
    """One independent comparator/rate/tuning-seed cell (spawn-safe)."""
    (name, kwargs, episodes, eta, seed, tau_leak, D, V, dt, step_dur,
     normalizer, device_k, tau_r_override, beta_leak) = job
    rewards = train_sequential(
        ActionSequenceTrack(), B=1, tau_leak=tau_leak, D=D,
        episodes=episodes, dt=dt, step_dur=step_dur, eta=eta, V=V,
        seed0=seed, eligibility_normalizer=normalizer,
        device_k=device_k, tau_r_override=tau_r_override,
        beta_leak=beta_leak, **kwargs,
    )
    curve = np.asarray(rewards[0], dtype=float)
    final = float(reward_rate(curve, min(100, episodes)))
    aulc = float(curve.mean())
    return name, float(eta), int(seed), aulc, final


def _action_evaluation_job(job):
    """One untouched evaluation-seed curve (spawn-safe)."""
    (name, kwargs, episodes, eta, seed, tau_leak, D, V, dt, step_dur,
     normalizer, device_k, tau_r_override, beta_leak) = job
    curve = train_sequential(
        ActionSequenceTrack(), B=1, tau_leak=tau_leak, D=D,
        episodes=episodes, dt=dt, step_dur=step_dur, eta=eta, V=V,
        seed0=seed, eligibility_normalizer=normalizer,
        device_k=device_k, tau_r_override=tau_r_override,
        beta_leak=beta_leak, **kwargs,
    )[0]
    return name, int(seed), np.asarray(curve, dtype=float)


def _map_spawn_safe(function, jobs, *, workers=1, pool=None):
    """Map independent deterministic jobs, owning a spawn pool only when needed."""
    if pool is not None:
        chunksize = max(1, len(jobs) // max(1, 4 * int(workers)))
        return pool.map(function, jobs, chunksize=chunksize)
    if int(workers) <= 1:
        return list(map(function, jobs))
    from multiprocessing import get_context
    with get_context("spawn").Pool(min(int(workers), len(jobs))) as own_pool:
        chunksize = max(1, len(jobs) // max(1, 4 * int(workers)))
        return own_pool.map(function, jobs, chunksize=chunksize)


def tune_comparator_learning_rates(*, calibration, episodes=300,
                                   learning_rates=LEARNING_RATE_GRID,
                                   tuning_seeds=TUNING_SEEDS, tau_leak=10.0,
                                   D=12.0, V=1.5, dt=5e-3, step_dur=0.4,
                                   include_rule_ablations=True,
                                   device_k=K_STAGES, tau_r_override=None,
                                   beta_leak=1.0, workers=1, pool=None,
                                   max_boundary_expansions=3):
    """Select each condition's rate by mean unsmoothed AULC on pilot seeds.

    The logarithmic grid is shared by all learned conditions.  If any optimum is
    on an edge, that edge is expanded by one decade and *pilot seeds only* are run
    there.  Ties are resolved toward the smaller rate.  The no-trace necessity
    control is assigned ``eta=0``.
    """
    rates = sorted(set(float(x) for x in learning_rates))
    seeds = tuple(int(x) for x in tuning_seeds)
    if not rates or any((not np.isfinite(x) or x <= 0) for x in rates):
        raise ValueError("learning_rates must be finite positive values")
    if not seeds:
        raise ValueError("tuning_seeds must not be empty")
    conditions = _comparator_conditions(include_rule_ablations)
    learned = tuple(name for name in conditions if name != "no_trace")
    raw = {name: {} for name in learned}
    expansion_log = []

    def evaluate_missing(candidate_rates):
        jobs = []
        for name in learned:
            kwargs = conditions[name]
            normalizer = calibration["records"][name]["eligibility_normalizer"]
            for eta in candidate_rates:
                if eta in raw[name]:
                    continue
                for seed in seeds:
                    jobs.append((
                        name, kwargs, int(episodes), eta, seed, float(tau_leak),
                        float(D), float(V), float(dt), float(step_dur),
                        float(normalizer), int(device_k), tau_r_override,
                        float(beta_leak),
                    ))
                raw[name][eta] = {}
        mapped = _map_spawn_safe(
            _action_tuning_job, jobs, workers=workers, pool=pool
        ) if jobs else []
        for name, eta, seed, aulc, final in mapped:
            raw[name][eta][seed] = {"aulc": aulc, "final_reward": final}

    evaluate_missing(rates)
    for _ in range(int(max_boundary_expansions)):
        boundary = set()
        for name in learned:
            selected = min(
                rates,
                key=lambda eta: (
                    -np.mean([raw[name][eta][seed]["aulc"] for seed in seeds]),
                    eta,
                ),
            )
            if selected == rates[0]:
                boundary.add(rates[0] / 10.0)
            if selected == rates[-1]:
                boundary.add(rates[-1] * 10.0)
        new_rates = sorted(boundary.difference(rates))
        if not new_rates:
            break
        expansion_log.append({
            "reason": "pilot_optimum_on_boundary",
            "added_rates": list(new_rates),
        })
        rates = sorted(set(rates).union(new_rates))
        evaluate_missing(new_rates)

    tuning = {}
    for name in conditions:
        if name == "no_trace":
            tuning[name] = {
                "selected_eta": 0.0,
                "scores": {0.0: None},
                "boundary_after_expansion": False,
            }
            continue
        scores = {}
        for eta in rates:
            aulcs = np.asarray(
                [raw[name][eta][seed]["aulc"] for seed in seeds], dtype=float
            )
            finals = np.asarray(
                [raw[name][eta][seed]["final_reward"] for seed in seeds],
                dtype=float,
            )
            scores[eta] = {
                "mean_aulc": float(aulcs.mean()),
                "mean_final_reward": float(finals.mean()),
                "per_seed_aulc": aulcs,
                "per_seed_final_reward": finals,
            }
        selected = min(rates, key=lambda x: (-scores[x]["mean_aulc"], x))
        tuning[name] = {
            "selected_eta": float(selected),
            "scores": scores,
            "boundary_after_expansion": selected in {rates[0], rates[-1]},
        }
    return {
        "learning_rate_grid": list(rates),
        "initial_learning_rate_grid": sorted(set(float(x) for x in learning_rates)),
        "grid_expansions": expansion_log,
        "max_boundary_expansions": int(max_boundary_expansions),
        "tuning_seeds": list(seeds),
        "episodes": int(episodes),
        "beta_leak": float(beta_leak),
        "workers": int(workers),
        "selection_rule": "highest_mean_unsmoothed_per_seed_aulc_then_smallest_eta",
        "evaluation_data_used": False,
        "conditions": tuning,
        "method_provenance": _provenance(
            "adapted",
            ["held-out hyperparameter selection"],
            "Learning rates are selected by pilot-seed AULC after scale calibration.",
            "Selected rates apply only to these repository implementations and task budgets.",
        ),
    }


def run_action_sequence(*, episodes=300, tuning_episodes=300,
                        calibration_trajectories=CALIBRATION_TRAJECTORIES,
                        learning_rates=LEARNING_RATE_GRID,
                        tuning_seeds=TUNING_SEEDS,
                        evaluation_seeds=EVALUATION_SEEDS,
                        tau_leak=10.0, D=12.0, V=1.5, dt=5e-3,
                        step_dur=0.4, include_rule_ablations=True,
                        device_k=K_STAGES, tau_r_override=None,
                        beta_leak=1.0,
                        retention_definition="deliberately_swept",
                        workers=1, pool=None, criterion=0.8,
                        bootstrap_resamples=10000,
                        max_boundary_expansions=3):
    """Controlled multi-decision benchmark with calibration, tuning and evaluation.

    Calibration uses fixed balanced trajectories; learning rates use only seeds
    1000--1019 by default; reported curves use untouched seeds 2000--2019.  Callers
    may pass strict prefixes for smoke runs while preserving the disjoint blocks.
    """
    tune_seeds = tuple(int(x) for x in tuning_seeds)
    eval_seeds = tuple(int(x) for x in evaluation_seeds)
    if not eval_seeds:
        raise ValueError("evaluation_seeds must not be empty")
    overlap = set(tune_seeds).intersection(eval_seeds)
    if overlap:
        raise ValueError(f"tuning and evaluation seeds overlap: {sorted(overlap)}")
    retention_definition = _retention_definition(retention_definition)
    calibration = calibrate_comparator_scales(
        trajectories=calibration_trajectories,
        tau_leak=tau_leak,
        D=D,
        V=V,
        dt=dt,
        step_dur=step_dur,
        include_rule_ablations=include_rule_ablations,
        device_k=device_k,
        tau_r_override=tau_r_override,
        beta_leak=beta_leak,
    )
    own_pool = None
    if pool is None and int(workers) > 1:
        from multiprocessing import get_context
        own_pool = get_context("spawn").Pool(int(workers))
        pool = own_pool
    try:
        tuning = tune_comparator_learning_rates(
            calibration=calibration,
            episodes=tuning_episodes,
            learning_rates=learning_rates,
            tuning_seeds=tune_seeds,
            tau_leak=tau_leak,
            D=D,
            V=V,
            dt=dt,
            step_dur=step_dur,
            include_rule_ablations=include_rule_ablations,
            device_k=device_k,
            tau_r_override=tau_r_override,
            beta_leak=beta_leak,
            workers=workers,
            pool=pool,
            max_boundary_expansions=max_boundary_expansions,
        )
        conditions = _comparator_conditions(include_rule_ablations)
        jobs = []
        for name, kwargs in conditions.items():
            eta = tuning["conditions"][name]["selected_eta"]
            normalizer = calibration["records"][name]["eligibility_normalizer"]
            for seed in eval_seeds:
                jobs.append((
                    name, kwargs, int(episodes), eta, seed, float(tau_leak),
                    float(D), float(V), float(dt), float(step_dur),
                    float(normalizer), int(device_k), tau_r_override,
                    float(beta_leak),
                ))
        mapped = _map_spawn_safe(
            _action_evaluation_job, jobs, workers=workers, pool=pool
        )
    finally:
        if own_pool is not None:
            own_pool.close()
            own_pool.join()
    gathered = {name: {} for name in conditions}
    for name, seed, curve in mapped:
        gathered[name][seed] = curve
    curves = {
        name: np.asarray([gathered[name][seed] for seed in eval_seeds], dtype=float)
        for name in conditions
    }
    finals = {
        name: reward_rate(values, min(100, int(episodes)))
        for name, values in curves.items()
    }
    from .stats import bootstrap_ci
    per_seed_metrics = {}
    for name, values in curves.items():
        criterion_raw = [
            trials_to_criterion(row, float(criterion), window=min(100, int(episodes)))
            for row in values
        ]
        per_seed_metrics[name] = {
            "aulc": values.mean(axis=1),
            "final_reward": finals[name],
            "criterion_time": np.asarray([
                int(episodes) + 1 if value is None else int(value)
                for value in criterion_raw
            ], dtype=int),
            "right_censored": np.asarray(
                [value is None for value in criterion_raw], dtype=bool
            ),
        }
    paired_bootstrap = {}
    device_aulc = per_seed_metrics["device"]["aulc"]
    for index, name in enumerate(conditions):
        if name == "device":
            continue
        difference = device_aulc - per_seed_metrics[name]["aulc"]
        lo, hi = bootstrap_ci(
            difference, n_boot=int(bootstrap_resamples), seed=314159 + index
        )
        paired_bootstrap[name] = {
            "metric": "device_minus_comparator_aulc",
            "mean": float(difference.mean()),
            "ci95": [lo, hi],
            "n_resamples": int(bootstrap_resamples),
            "verdict": "device_advantage" if lo > 0 else "unresolved",
        }
    comparator_diagnostics = {}
    for name in _comparator_conditions(include_rule_ablations):
        calibrated = calibration["records"][name]
        comparator_diagnostics[name] = {
            "raw_trace_peak": calibrated["raw_trace_peak"],
            "raw_trace_area": calibrated["raw_trace_area"],
            "eligibility_normalizer": calibrated["eligibility_normalizer"],
            "selected_eta": tuning["conditions"][name]["selected_eta"],
            "raw_effective_update_rms": calibrated["raw_effective_update_rms"],
            "normalized_effective_update_rms": calibrated[
                "normalized_effective_update_rms"
            ],
            "matched_exponential_tau_s": calibrated.get(
                "matched_exponential_tau_s"
            ),
        }
    return {
        "task": "ActionSequenceTrack",
        "task_classification": "multi_decision_terminal_reward",
        "required_actions": list(calibration["required_actions"]),
        "curves": curves,
        "finals": finals,
        "per_seed_metrics": per_seed_metrics,
        "paired_bootstrap": paired_bootstrap,
        "calibration": calibration,
        "tuning": tuning,
        "comparator_diagnostics": comparator_diagnostics,
        "evaluation_seeds": list(eval_seeds),
        "seed_partition": {
            "tuning": list(tune_seeds),
            "evaluation": list(eval_seeds),
            "disjoint": True,
        },
        "episodes": int(episodes),
        "tau_leak": float(tau_leak),
        "beta_leak": float(beta_leak),
        "retention_definition": retention_definition,
        "workers": int(workers),
        "delay": float(D),
        "criterion": float(criterion),
        "model_specifications": {
            model_id: device_model_spec(model_id)
            for model_id in (PRIMARY_MODEL_ID, LINEAR_MODEL_ID)
        },
        "method_provenance": _provenance(
            "proposed",
            ["multi-decision reinforcement-learning evaluation"],
            "Calibrated repository implementations are evaluated on ActionSequenceTrack.",
            "Descriptive comparison only; not superiority over complete cited methods.",
        ),
        "condition_method_provenance": {
            name: _condition_provenance(name)
            for name in _comparator_conditions(include_rule_ablations)
        },
        "claim_limit": (
            "Descriptive comparison of calibrated repository implementations; "
            "not algorithmic superiority over complete cited methods."
        ),
    }


def run_delayed_cued_choice(*, seeds=20, episodes=800, V=1.5, D0=4.0, TAU0=10.0,
                            taus=(2.0, 5.0, 20.0),
                            delays=(2, 4, 8, 16, 32, 64), seed0=0,
                            workers=1, device_k=K_STAGES,
                            tau_r_override=None):
    """Historical one-choice delayed-context curves and retention grid.

    The stem auto-advances and only the cue-dependent junction decision is learned.
    This function exists for archive compatibility and control analyses; it does not
    provide trajectory-level policy evidence.  Its historical e-prop-style learning
    rate is uncalibrated, so between-method rankings are not valid superiority tests.

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
    conds = {"device": dict(), "exponential": dict(abstract=True),
             "shallow_eprop": dict(eprop=True, eta=EPROP_ETA),
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


def run_sequential(*args, **kwargs):
    """Deprecated wrapper for :func:`run_delayed_cued_choice`."""
    warnings.warn(
        "run_sequential used a one-choice delayed contextual task. Use "
        "run_action_sequence for the calibrated multi-decision benchmark or "
        "run_delayed_cued_choice for the historical control.",
        DeprecationWarning,
        stacklevel=2,
    )
    result = run_delayed_cued_choice(*args, **kwargs)
    aliases = {"rstdp": "exponential", "eprop": "shallow_eprop"}
    # Preserve the historical dictionary interface for one deprecation release.
    # The explicit metadata prevents those labels from being mistaken for faithful
    # reproductions of the cited complete methods.
    for field in ("finals", "finals_ci", "curves"):
        for legacy, corrected in aliases.items():
            result[field][legacy] = result[field][corrected]
    for legacy, corrected in aliases.items():
        result["condition_method_provenance"][legacy] = (
            result["condition_method_provenance"][corrected]
        )
    result["legacy_condition_aliases"] = aliases
    return result


# -----------------------------------------------------------------------------
# Retention--delay simulation sensitivity. ``tau_leak`` directly defines trace decay,
# so threshold crossings, censoring and curvature are reported as a task-specific
# design curve, not as an independently established physical law.
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


def _dmax_cell(spec, episodes, V, seeds, device_k=K_STAGES,
               tau_r_override=None, beta_leak=1.0):
    """One device cell, retaining the per-seed final reward rates."""
    from .stats import bootstrap_ci
    tau, D = spec
    env = ActionSequenceTrack()
    r = train_sequential(env, B=seeds, tau_leak=tau, D=D, episodes=episodes, V=V,
                         seed0=0, device_k=device_k,
                         tau_r_override=tau_r_override, beta_leak=beta_leak)
    f = reward_rate(r, 100)
    lo, hi = bootstrap_ci(f)
    return (tau, D, float(f.mean()), lo, hi, np.asarray(f, float))


def _dmax_assemble(results, taus, delays, crit, seeds, episodes, V,
                   retention_definition="deliberately_swept", beta_leak=1.0):
    """Package a simulated retention--delay threshold-crossing design curve."""
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
    local_slope = np.full_like(dvals, np.nan, dtype=float)
    local_curvature = np.full_like(dvals, np.nan, dtype=float)
    resolved = np.flatnonzero(fitmask)
    if len(resolved) >= 2:
        rr = resolved
        local_slope[rr] = np.gradient(dvals[rr], taus_a[rr])
        if len(rr) >= 3:
            local_curvature[rr] = np.gradient(local_slope[rr], taus_a[rr])
    return {"taus": list(taus), "delays": list(delays), "grid": grid,
            "seed_grid": seed_grid, "dmax": dmax,
            "k": k, "r2": r2, "k_plateau": k_plateau, "r2_plateau": r2_plateau,
            "plateau_tau": DMAX_PLATEAU_TAU, "crit": crit,
            "censored": censored.tolist(), "seeds": seeds, "episodes": episodes,
            "V_op": V,
            "beta_leak": float(beta_leak),
            "retention_definition": _retention_definition(retention_definition),
            "threshold_delay": dict(dmax),
            "local_slope": local_slope.tolist(),
            "local_curvature": local_curvature.tolist(),
            "simulation_only": True,
            "independent_physical_validation": False,
            "interpretation": (
                "Internally generated retention--delay sensitivity: tau_leak is the "
                "simulated decay control, so this is not an independently tested physical law."
            ),
            "legacy_origin_fit": {"slope": k, "r2": r2},
            "method_provenance": _provenance(
                "proposed",
                ["eligibility-trace decay", "threshold-crossing analysis"],
                "Simulation sweep in which tau_leak directly controls trace decay.",
                "A task-specific design curve, not literature-derived or independently validated physics.",
            )}


def run_retention_delay_curve(*, seeds=20, episodes=800, V=1.5, taus=None,
                              delays=None, crit=0.75, workers=1,
                              device_k=K_STAGES, tau_r_override=None,
                              beta_leak=1.0,
                              retention_definition="deliberately_swept"):
    """Simulated retention--delay design curve for the action-sequence task.

    The threshold delay, censoring and local curvature are descriptive outputs.  Since
    ``tau_leak`` directly defines the simulated decay, the curve is not an independent
    physical-law test.  Legacy origin-fit fields remain nested for archive inspection,
    not as a headline model.
    """
    from functools import partial
    taus = DMAX_TAUS if taus is None else list(taus)
    delays = DMAX_DELAYS if delays is None else list(delays)
    specs = [(tau, D) for tau in taus for D in delays]
    worker = partial(_dmax_cell, episodes=episodes, V=V, seeds=seeds,
                     device_k=device_k, tau_r_override=tau_r_override,
                     beta_leak=beta_leak)
    if int(workers) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(int(workers), len(specs))) as pool:
            results = pool.map(worker, specs, chunksize=1)
    else:
        results = list(map(worker, specs))
    return _dmax_assemble(
        results, taus, delays, crit, seeds, episodes, V,
        retention_definition=retention_definition, beta_leak=beta_leak,
    )


def run_dmax_law(*args, **kwargs):
    """Deprecated wrapper for :func:`run_retention_delay_curve`."""
    warnings.warn(
        "run_dmax_law generated an internally constructed simulation sensitivity, "
        "not an independently tested physical law. Use run_retention_delay_curve.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_retention_delay_curve(*args, **kwargs)


def run_retention_delay_curve_adaptive(*, seeds=4, episodes=300, V=1.5,
                                       taus=None,
                                       delay_ratios=(4, 8, 12, 16, 20),
                                       crit=0.75, workers=1,
                                       device_k=K_STAGES,
                                       tau_r_override=None,
                                       beta_leak=1.0,
                                       retention_definition="deliberately_swept"):
    """Reduced live sampling of the simulated retention--delay design curve.

    The publication sweep uses one 18-value absolute-delay grid for every retention.
    A lightweight notebook need not spend most of its time far from the falling edge:
    this runner evaluates the same device task at fixed ``D / tau_leak`` ratios, then
    interpolates the criterion crossing independently for each retention.  No model
    outputs or fitted values are substituted; every displayed point is obtained from
    a live ``train_sequential`` cell.  This remains an internally constructed
    simulation sensitivity rather than independent physical validation.
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
                     device_k=device_k, tau_r_override=tau_r_override,
                     beta_leak=beta_leak)
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
            "beta_leak": float(beta_leak),
            "retention_definition": _retention_definition(retention_definition),
            "mode": "adaptive-live", "threshold_delay": dict(dmax),
            "simulation_only": True, "independent_physical_validation": False,
            "interpretation": (
                "Internally generated retention--delay sensitivity; tau_leak directly "
                "controls the simulated trace decay."
            ),
            "method_provenance": _provenance(
                "proposed",
                ["eligibility-trace decay", "adaptive threshold sampling"],
                "Adaptive simulation grid at fixed delay-to-retention ratios.",
                "Task-specific sensitivity only; not an independent physical law.",
            )}


def run_dmax_adaptive(*args, **kwargs):
    """Deprecated wrapper for :func:`run_retention_delay_curve_adaptive`."""
    warnings.warn(
        "run_dmax_adaptive samples a simulation design curve, not a physical law. "
        "Use run_retention_delay_curve_adaptive.",
        DeprecationWarning,
        stacklevel=2,
    )
    return run_retention_delay_curve_adaptive(*args, **kwargs)


# -----------------------------------------------------------------------------
# Experiment 15 -- longer-horizon temporal-credit feasibility (Tier 2).
#
# Answers the *task-depth* objection ("your task is too easy") on the axis the device
# primitive is actually for: temporal credit assignment across a multi-step trajectory
# with a delayed goal reward. NOT a perception/benchmark study -- the input stays
# low-dimensional; what grows is the temporal DEPTH (trajectory length L) of the credit
# problem. Shallow state x action policy (no homeostasis). Per cell the device trace is
# compared against repository controls on the same task/network/seeds/budget:
#   device / shallow_eprop / exponential / no_trace.
# Scaling axes: required action-sequence length L and action alphabet A.
# tau_leak is set generously so retention can span the longest trajectory tested.
# -----------------------------------------------------------------------------
LONG_HORIZON_CONDS = {
    "device":   dict(),
    "shallow_eprop": dict(eprop=True),
    "exponential": dict(abstract=True),
    "no_trace": dict(no_trace=True),
}


def _required_actions_for_length(L, A=2):
    """Deterministic non-constant sequence; L=4, A=2 gives (0, 1, 1, 0)."""
    L, A = int(L), int(A)
    if L < 2 or A < 2:
        raise ValueError("long-horizon sequences require L >= 2 and A >= 2")
    if A == 2:
        return tuple((i * (i + 1) // 2) % 2 for i in range(L))
    return tuple(i % A for i in range(L))


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
    env = ActionSequenceTrack(_required_actions_for_length(L, A), n_actions=A)
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
            "tau_leak": tau_leak, "seeds": seeds,
            "retention_definition": "deliberately_swept",
            "task": "ActionSequenceTrack",
            "task_classification": "multi_decision_terminal_reward",
            "method_provenance": _provenance(
                "proposed",
                ["multi-decision terminal-reward evaluation"],
                "Action-sequence length and action alphabet are swept in the shared harness.",
                "Legacy feasibility sweep with uncalibrated comparator rates.",
            ),
            "condition_method_provenance": {
                name: _condition_provenance(name) for name in LONG_HORIZON_CONDS
            },
            "claim_limit": (
                "Task-depth feasibility for repository implementations; comparator "
                "rates are not calibrated by this legacy sweep."
            )}


def run_long_horizon(*, L_grid=(3, 5, 8, 12), A_grid=(2, 3), seeds=12,
                     episodes=2500, D=2.0, tau_leak=10.0, dt=5e-3):
    """Experiment 15: longer-horizon temporal-credit feasibility, serial.

    Sweeps action-sequence length ``L`` x action alphabet ``A`` x condition x seed and reports
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
    env = ActionSequenceTrack(_required_actions_for_length(L, A), n_actions=A)
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
    chance_by_length = {int(L): float(A ** (-int(L))) for L in L_grid}
    grid = np.zeros((len(L_grid), len(p_grid), 3))
    cgrid = np.zeros((len(L_grid), len(p_grid)))
    for i, L in enumerate(L_grid):
        for j, p in enumerate(p_grid):
            d = np.array(dev[(L, p)]); c = np.array(ctl[(L, p)])
            lo, hi = _lh_boot_ci(d, seed=i * 10 + j)
            grid[i, j] = (d.mean(), lo, hi); cgrid[i, j] = c.mean()
    return {"L": L_grid, "p": p_grid, "A": A, "grid": grid, "ctrl": cgrid,
            "chance_by_length": chance_by_length,
            "faults": "stuck-off + D2D(0.5) [PF excluded] (maze mapping)",
            "faults_included": ["stuck_off", "sampled_lognormal_D2D"],
            "faults_excluded": [
                "line_resistance", "read_noise", "temporal_noise", "drift",
                "programming_noise", "Poole_Frenkel_read_nonlinearity",
            ],
            "episodes": episodes, "D": D, "tau_leak": tau_leak, "seeds": seeds,
            "retention_definition": "deliberately_swept",
            "task": "ActionSequenceTrack",
            "method_provenance": _provenance(
                "proposed",
                ["multi-decision terminal-reward evaluation", "fault stress testing"],
                "The action-sequence task is evaluated under the enumerated fault classes.",
                "The omitted nonidealities remain outside this simulation's scope.",
            ),
            "condition_method_provenance": {
                "device": _condition_provenance("device"),
                "no_trace": _condition_provenance("no_trace"),
            }}


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
    """Reproduction CLI for action-sequence and retention-delay grids.

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

    ap = argparse.ArgumentParser(description="Action-sequence credit-assignment reproductions")
    ap.add_argument("--exp6", action="store_true",
                    help="calibrated action-sequence benchmark -> exp6_action_sequence.npy")
    ap.add_argument("--exp8", action="store_true",
                    help="simulated retention-delay curve -> exp8_retention_delay_curve.npy")
    ap.add_argument("--exp15", action="store_true",
                    help="longer-horizon feasibility -> exp15_long_horizon.npy")
    ap.add_argument("--exp15-faults", dest="exp15_faults", action="store_true",
                    help="longer-horizon under the SiOx fault prior -> exp15_long_horizon_faults.npy")
    ap.add_argument("--quick", action="store_true", help="fast few-seed smoke run")
    ap.add_argument("--full", action="store_true", help="published-scale run (default)")
    a = ap.parse_args(argv)
    run_all = not (a.exp6 or a.exp8 or a.exp15 or a.exp15_faults)

    n_proc = max(1, (os.cpu_count() or 4) - 2)

    # ---- Experiment 6: calibrated multi-decision action sequence ----
    if a.exp6 or run_all:
        if a.quick:
            run_kw = dict(
                episodes=40, tuning_episodes=20, calibration_trajectories=8,
                learning_rates=(0.1, 1.0), tuning_seeds=(1000,),
                evaluation_seeds=(2000,), dt=0.01, step_dur=0.05, D=0.1,
            )
        else:
            run_kw = {}
        print("=== exp6 calibrated ActionSequenceTrack ===", flush=True)
        grid = run_action_sequence(**run_kw)
        paths.save_result("exp6_action_sequence.npy", grid)
        print("  wrote exp6_action_sequence.npy", flush=True)

    # ---- Experiment 8: simulated retention--delay design curve ----
    if a.exp8 or run_all:
        seeds = 6 if a.quick else 20
        episodes = 400 if a.quick else 800
        V, crit = 1.5, 0.75
        taus, delays = DMAX_TAUS, DMAX_DELAYS
        print(f"=== exp8 retention-delay curve: {len(taus)} tau x {len(delays)} delays, "
              f"{seeds} seeds, {episodes} ep, V={V}, crit={crit} ===", flush=True)
        specs = [(tau, D) for tau in taus for D in delays]
        with Pool(min(len(specs), n_proc)) as pool:
            results = pool.map(partial(_dmax_cell, episodes=episodes, V=V, seeds=seeds),
                               specs)
        grid = _dmax_assemble(results, taus, delays, crit, seeds, episodes, V)
        paths.save_result("exp8_retention_delay_curve.npy", grid)
        print("  wrote exp8_retention_delay_curve.npy (simulation-only)", flush=True)

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
    env = DelayedCuedChoice(L=3, A_goal=2)
    out = train_sequential(env, B=seeds, tau_leak=tau_leak, D=D, episodes=episodes,
                           V=V, seed0=0, return_weights=want_w, **kw)
    return (key, out)


def _exp6_worker_seeded(spec, seeds, episodes, V, seed0=0, device_k=K_STAGES,
                        tau_r_override=None):
    """Spawn-safe exp6 worker that retains the public runner's ``seed0`` control."""
    key, tau_leak, D, kw, want_w = spec
    env = DelayedCuedChoice(L=3, A_goal=2)
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
    env = DelayedCuedChoice(L=3, A_goal=2)
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
            "retention_definition": "deliberately_swept",
            "criteria": criteria,
            "task": "DelayedCuedChoice",
            "task_classification": "single_learned_contextual_decision_after_fixed_wait",
            "comparator_calibrated": False,
            "method_provenance": _provenance(
                "adapted",
                ["delayed contextual decision evaluation"],
                "The historical auto-advancing task is retained as DelayedCuedChoice.",
                "Not trajectory-level credit assignment or a calibrated superiority test.",
            ),
            "condition_method_provenance": {
                name: _condition_provenance(name) for name in conds
            },
            "claim_limit": (
                "Historical delayed-choice control only; not trajectory-level policy "
                "learning and not a calibrated algorithm-superiority comparison."
            )}


if __name__ == "__main__":
    main()
