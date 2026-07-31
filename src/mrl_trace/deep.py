"""Deep (two-layer) all-local spiking RL with a physical eligibility trace -- Arm D.

The bandit (:mod:`mrl_trace.bandit`) and the shallow delayed-reward tasks
(:mod:`mrl_trace.maze`) train a SINGLE layer of device synapses with a
single global reward scalar ``(R - b)``. That is enough when the policy is linearly
separable in the state lines, but it has two known limits:

1. A single trained layer cannot represent a non-linearly-separable policy (XOR).
2. A single global scalar gives a *hidden* layer no per-neuron credit -- every hidden
   synapse sees the same third factor, so reward-modulated Hebbian learning at depth
   cannot resolve which hidden unit helped (the structural-credit problem; the depth
   analogue of the earlier scaling wall).

This module asks whether a device-supplied physical eligibility trace can serve as the
TEMPORAL factor of a DEEP spiking policy that is trained FULLY LOCALLY -- no
backpropagation, no weight transport -- on a task a single trained layer cannot solve
(XOR). The temporal factor (the device trace) is held fixed across conditions; only the
SPATIAL-credit pathway varies, so the comparison isolates structural credit:

- ``shallow`` : one trained layer F->A (proves the task needs depth -- should FAIL XOR).
- ``elm``     : deep, hidden layer FIXED RANDOM, only the output trained (the
                random-features shortcut; bounds how much depth-without-learning buys).
- ``global``  : deep, BOTH layers trained by the pure global scalar ``(R - b)``
                (reward-modulated Hebb at the hidden layer; the structural-credit
                failure mode).
- ``dfa``     : deep, BOTH layers trained by DIRECT FEEDBACK ALIGNMENT -- a per-neuron
                learning signal broadcast to the hidden layer through a FIXED RANDOM
                feedback matrix (no W2 transpose, no gradient transport). All-local.
- ``no_trace``: deep DFA with eligibility zeroed; homeostasis is an independent
  argument and must be held fixed for a trace-only necessity control.

Every condition uses the same :class:`GateBankBatched` device physics for eligibility,
the same signed leak-dominant coincidence drive, and ``dw = eta * L * e`` updates,
where ``L`` is the (global or per-layer) learning signal. The criteria retained in
legacy analysis notes were retrospective protocol records.
"""
from __future__ import annotations

import numpy as np

from .bandit import GateBankBatched, W_INIT, W_MAX
from .device import K_STAGES, decay_matched_exponential_tau
from .neurons import lif_step_batched, TAU_M, V_TH
from .learning import LTD_BIAS

__all__ = ["xor_inputs", "train_deep", "reward_rate",
           "run_dms", "final_rate", "run_dms_all",
           "run_deep_local", "run_deep_dms", "run_array_scale", "main",
           "DEEP_METHOD_PROVENANCE"]


def _relax_gate(bank, n_relax, stride, rem):
    """Advance an undriven :class:`GateBankBatched` for ``n_relax*stride + rem`` ticks,
    using a coarsened step (``stride*dt``) for the bulk and single ``dt`` steps for the
    remainder. The drive is zero, so the dynamics are smooth and the coarse step is
    accurate (see the inline note at the call site). Restores the gate's ``dt``."""
    B, S, A = bank.B, bank.S, bank.A
    z = np.zeros((B, S, A))
    if n_relax > 0:
        dt0 = bank.dt
        bank.dt = dt0 * stride
        try:
            for _ in range(n_relax):
                bank.step(z)
        finally:
            bank.dt = dt0
    for _ in range(rem):
        bank.step(z)


# ----------------------------------------------------------------------------
# Task: XOR contextual bandit (depth genuinely required)
# ----------------------------------------------------------------------------
def xor_inputs(rng, B):
    """Draw ``B`` XOR trials.

    Two binary features ``b0, b1`` per seed; the correct action is ``XOR(b0, b1)``.
    The features are one-hot-per-feature encoded onto ``F = 4`` input lines
    ``[b0=0, b0=1, b1=0, b1=1]`` so that EXACTLY TWO lines are active every trial --
    a constant total input drive across all four states, which removes any input-rate
    confound between states. XOR is not linearly separable in these 4 lines, so a single
    trained linear layer F->A cannot solve it (this is what forces depth).

    Returns ``(feat_lines (B,4) in {0,1}, correct_action (B,) in {0,1})``.
    """
    b0 = rng.integers(2, size=B)
    b1 = rng.integers(2, size=B)
    lines = np.zeros((B, 4))
    bidx = np.arange(B)
    lines[bidx, 0 + b0] = 1.0          # b0=0 -> line 0, b0=1 -> line 1
    lines[bidx, 2 + b1] = 1.0          # b1=0 -> line 2, b1=1 -> line 3
    correct = (b0 ^ b1).astype(int)
    return lines, correct


# ----------------------------------------------------------------------------
# Training harness
# ----------------------------------------------------------------------------
def train_deep(*, mode="dfa", B=20, H=8, tau_leak=10.0, D=5.0, trials=2000,
               dt=5e-3, cue_dur=1.0, eta=0.2, eta_hidden=None, in_rate=200.0,
               ltd=LTD_BIAS, tau_m=TAU_M, v_th=V_TH, V=1.5, sigma0=0.15, sigma1=0.05,
               fb_scale=1.0, w_scale1=0.6, w_scale2=0.35, w_max=W_MAX,
               bias_o=0.35, homeo=0.0, homeo_target=0.35, homeo_tau=200.0,
               t_distract=None, distract_dur=0.3, weight_fault=None, early_stop=None,
               reward_pools=None, state_sampler=None, n_features=None, n_actions=None,
               seed0=0, return_weights=False, log_align=False,
               device_k=K_STAGES, tau_r_override=None):
    """Train the XOR contextual bandit with a two-layer spiking device-synapse network.

    Architecture: ``F=4`` input lines -> ``H`` hidden LIF neurons -> ``A=2`` action LIF
    neurons, with device-synapse weight matrices ``W1 (F,H)`` and ``W2 (H,A)``. Each
    matrix has its own :class:`GateBankBatched` eligibility (same trap-cascade physics,
    retention ``tau_leak``, signed leak-dominant drive). The action is the
    higher-spiking output neuron over the cue epoch (membrane noise = exploration); the
    reward ``R in {0,1}`` is delivered after delay ``D``, contingent on the chosen action
    matching ``XOR``.

    ``mode`` selects the SPATIAL-credit pathway (the eligibility/temporal factor is the
    device trace in all modes); see the module docstring and recorded criteria.

    ``state_sampler`` swaps the input STATE without touching the learning rule. By default
    (``None``) the state is the built-in XOR cue (``xor_inputs``, ``F=4``, ``A=2``). Pass a
    callable ``state_sampler(rng, B) -> (lines (B,F) >=0, correct (B,) in [0,A))`` to drive
    the SAME deep all-local stack from an arbitrary state source -- e.g. the low-dimensional
    readout of a frozen perception front-end (the exp16 hybrid). ``lines`` are interpreted as
    per-line input rates exactly as the one-hot cue is, so real-valued (noisy, overlapping)
    readout vectors slot in unchanged; only the policy the network must learn changes, never
    the rule. ``n_features`` and ``n_actions`` are then required (the ``F``/``A`` the sampler
    emits). The eligibility, DFA, homeostasis, fault and reward machinery are all identical,
    so a feasibility conclusion drawn here transfers to the built-in cue and vice versa.

    Returns rewards ``(B, trials)`` (and final weights if ``return_weights``).
    """
    if mode not in {"shallow", "elm", "global", "dfa", "no_trace"}:
        raise ValueError(f"unknown mode {mode!r}")
    rng = np.random.default_rng(seed0)
    if state_sampler is None:
        F, A = 4, 2
        sampler = xor_inputs
    else:
        if n_features is None or n_actions is None:
            raise ValueError("state_sampler requires n_features and n_actions")
        F, A = int(n_features), int(n_actions)
        sampler = state_sampler
    eta_h = eta if eta_hidden is None else eta_hidden
    deep = mode != "shallow"
    no_trace = mode == "no_trace"
    train_w1 = deep and mode != "elm"     # ELM keeps the hidden layer fixed random

    # --- weights ---
    # SIGNED weights in [-w_max, w_max]: each synapse is a differential conductance pair
    # w = G+ - G- (the manuscript's crossbar realisation), so a synapse can be excitatory
    # or inhibitory. This is required, not cosmetic: with the one-hot-per-feature XOR
    # encoding the four patterns carry equal total positive charge, so a positive-only
    # readout cannot separate them -- a signed readout (subtracting the AND-like feature)
    # can. Weights are RANDOM, not uniform: an identical init makes every hidden neuron
    # receive the same drive, collapsing the hidden layer to one effective unit (zero
    # capacity); independent random init breaks that symmetry. Per-layer scale (w_scale1,
    # w_scale2) sets the summed drive near the LIF threshold rather than saturating it
    # (output fan-in ~H/2 active hidden neurons -> smaller w_scale2).
    def _init(shape, scale):
        return np.clip(scale * rng.standard_normal(shape), -w_max, w_max)
    if deep:
        if mode == "elm":
            # fixed random hidden projection (ELM / random features); never updated.
            W1 = _init((B, F, H), w_scale1)
        else:
            W1 = _init((B, F, H), w_scale1)
        W2 = _init((B, H, A), w_scale2)
    else:
        # shallow: a single trained layer F->A (the hidden layer is absent)
        W2 = _init((B, F, A), w_scale1)
        W1 = None

    # --- device eligibility gates (one bank per trained matrix) ---
    gate_kw = dict(tau_leak=tau_leak, dt=dt, V=V, k=device_k,
                   tau_r_override=tau_r_override)
    g1 = GateBankBatched(B, F, H, **gate_kw) if deep else None
    g_out = GateBankBatched(B, (H if deep else F), A, **gate_kw)

    # --- DFA feedback matrix: fixed random, drawn ONCE, independent of W2 (no transport) ---
    B_fix = fb_scale * rng.standard_normal((A, H)) if deep else None

    baseline = np.full(B, 1.0 / A)
    bidx = np.arange(B)
    cue = (0.3, 0.3 + cue_dur)
    reward_lag = int(round(D / dt))
    nsteps = int(round(cue[1] / dt)) + 2
    rewards = np.zeros((B, trials))

    # Feedback-alignment diagnostic. The credit DFA delivers to the hidden layer is
    # B_fix @ e_out; the credit backprop WOULD deliver is W2 @ e_out. Both are linear in
    # the output error, so per seed the angle between the matrices W2 (H,A) and B_fix.T
    # (H,A) measures alignment: FA theory predicts learning succeeds when W2 evolves to
    # align with B_fix.T (angle -> 0) and stalls when they stay orthogonal (~90 deg).
    # Logged every `align_every` trials when requested; never affects the dynamics.
    align_log = None
    if log_align and deep:
        align_every = max(1, trials // 60)
        align_t, align_a = [], []

    def _align_angle():
        # per-seed angle (deg) between flattened W2 and B_fix.T
        bt = B_fix.T[None]                       # (1,H,A)
        a = W2.reshape(B, -1); b = np.broadcast_to(bt, (B, H, A)).reshape(B, -1)
        num = (a * b).sum(1)
        den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12
        return np.degrees(np.arccos(np.clip(num / den, -1.0, 1.0)))

    # Homeostatic activity regulation (optional). Each hidden neuron keeps a slow running
    # estimate of its own firing fraction and is nudged toward a shared target rate. This
    # is LOCAL (a neuron sees only its own activity) and hardware-plausible (an activity
    # integrator per row). It opposes the winner-take-all policy collapse diagnosed as the
    # cause of the DFA bimodality WITHOUT fighting the XOR solution, whose correct policy
    # has balanced marginal activity across actions. Off (homeo=0) reproduces the baseline.
    act_hidden = np.full((B, H), homeo_target) if (deep and homeo > 0) else None

    for tr in range(trials):
        sigma = sigma0 + (sigma1 - sigma0) * tr / trials
        # Read-time device faults applied ONCE per trial (weights change only per trial):
        # the array reads the faulted conductance while learning targets the clean weight,
        # so faults are a fixed physical realisation, not a learnable parameter. Computing
        # this here (not per timestep) is the difference between a tractable run and not.
        W1r = weight_fault(W1) if (weight_fault is not None and deep) else W1
        W2r = weight_fault(W2) if (weight_fault is not None) else W2
        if log_align and deep and (tr % align_every == 0):
            align_t.append(tr); align_a.append(_align_angle())
        if deep:
            g1.reset()
        g_out.reset()
        lines, correct = sampler(rng, B)
        vh = np.zeros((B, H))
        vo = np.zeros((B, A))
        spk_o = np.zeros((B, A))
        spk_h = np.zeros((B, H))

        for n in range(nsteps):
            t = n * dt
            on = cue[0] <= t < cue[1]
            pre_in = ((rng.random((B, F)) < (in_rate * dt) * lines).astype(float)
                      if on else np.zeros((B, F)))

            if deep:
                # layer 1: inputs -> hidden LIF
                ch_h = np.einsum('bfh,bf->bh', W1r, pre_in)
                vh, sp_h = lif_step_batched(vh, ch_h, dt, rng, tau_m=tau_m,
                                            v_th=v_th, noise=sigma)
                spk_h += sp_h
                # layer 2: hidden spikes -> action LIF, with a tonic bias drive during
                # the cue. The bias is the LIF analogue of a bias unit: it keeps both
                # action neurons in a firing regime even on patterns whose correct
                # response relies on hidden INHIBITION (the XOR=1 patterns, which fire no
                # excitatory detector), so the output is never fully silent and the
                # spike-count contrast that selects the action can form. It is a fixed
                # constant, not learned, identical for both actions (no class bias).
                ch_o = np.einsum('bha,bh->ba', W2r, sp_h.astype(float))
                if on:
                    ch_o = ch_o + bias_o
                vo, sp_o = lif_step_batched(vo, ch_o, dt, rng, tau_m=tau_m,
                                            v_th=v_th, noise=sigma)
                spk_o += sp_o
                # signed leak-dominant coincidence drives BOTH eligibility banks
                if no_trace:
                    g1.step(np.zeros((B, F, H)))
                    g_out.step(np.zeros((B, H, A)))
                else:
                    drive1 = pre_in[:, :, None] * np.where(sp_h, 1.0, -ltd)[:, None, :]
                    g1.step(drive1)
                    drive_o = (sp_h.astype(float)[:, :, None]
                               * np.where(sp_o, 1.0, -ltd)[:, None, :])
                    g_out.step(drive_o)
            else:
                # shallow: inputs -> action LIF directly (single trained layer F->A)
                ch_o = np.einsum('bfa,bf->ba', W2r, pre_in)
                vo, sp_o = lif_step_batched(vo, ch_o, dt, rng, tau_m=tau_m,
                                            v_th=v_th, noise=sigma)
                spk_o += sp_o
                drive_o = pre_in[:, :, None] * np.where(sp_o, 1.0, -ltd)[:, None, :]
                g_out.step(drive_o)

        # --- snapshot eligibility after the action->reward delay (undriven relaxation) ---
        # The relaxation has NO drive and no spikes, so the cascade/leaky dynamics are
        # numerically smooth: a coarsened forward-Euler step (stride x dt) is accurate
        # while stride*dt/tau << 1 (stride=10, dt=5e-3, tau>=2 s -> ratio <= 0.025), and
        # is ~stride x cheaper than ticking every dt. This is the same trick as
        # maze._relax, and it dominates runtime (reward_lag ~ D/dt undriven ticks/trial).
        #
        # TEMPORAL DISTRACTOR (when t_distract is set): partway through the delay an
        # UNINFORMATIVE coincidence fires on the eligibility gates -- a signed drive,
        # uncorrelated with the XOR label, on the same lines that carried the cue. It adds
        # fresh eligibility late in the delay, so a recency-weighted trace mis-credits it,
        # while the device's interval-peaked band-pass trace (whose surviving eligibility
        # at reward still favours the EARLIER cue) does not. This makes the deep task hard
        # in TIME (eligibility must bridge the cue->reward gap past interference) AND in
        # DEPTH (DFA + homeostasis must still resolve hidden-layer credit) at once.
        if not no_trace:
            stride = 10 if reward_lag > 200 else 1
            if t_distract is None:
                n_relax, rem = divmod(reward_lag, stride)
                _relax_gate(g_out, n_relax, stride, rem)
                if deep:
                    _relax_gate(g1, n_relax, stride, rem)
            else:
                # relax to the distractor, inject it (single dt ticks), relax to reward
                i_d0 = int(round(t_distract / dt))
                i_d1 = i_d0 + int(round(distract_dur / dt))
                i_d0 = min(i_d0, reward_lag); i_d1 = min(i_d1, reward_lag)
                pre_d, post_d = i_d0, reward_lag - i_d1
                n0, r0 = divmod(pre_d, stride)
                _relax_gate(g_out, n0, stride, r0)
                if deep:
                    _relax_gate(g1, n0, stride, r0)
                for _ in range(i_d1 - i_d0):
                    # distractor: random coincidence on the cue lines, label-independent
                    dfire = (rng.random((B, F)) < in_rate * dt).astype(float)
                    if deep:
                        hfire = (rng.random((B, H)) < in_rate * dt).astype(float)
                        g1.step(dfire[:, :, None] * np.where(hfire, 1.0, -ltd)[:, None, :])
                        ofire = (rng.random((B, A)) < in_rate * dt).astype(float)
                        g_out.step(hfire[:, :, None] * np.where(ofire, 1.0, -ltd)[:, None, :])
                    else:
                        ofire = (rng.random((B, A)) < in_rate * dt).astype(float)
                        g_out.step(dfire[:, :, None] * np.where(ofire, 1.0, -ltd)[:, None, :])
                n1, r1 = divmod(post_d, stride)
                _relax_gate(g_out, n1, stride, r1)
                if deep:
                    _relax_gate(g1, n1, stride, r1)
            eo_rew = g_out.vn[..., -1] / g_out.Vnmax
            e1_rew = (g1.vn[..., -1] / g1.Vnmax) if deep else None
        else:
            eo_rew = np.zeros_like(g_out.vn[..., -1])
            e1_rew = np.zeros((B, F, H)) if deep else None

        # --- action selection (argmax output spikes; ties -> random) ---
        tie = spk_o.max(1) == spk_o.min(1)
        chosen = np.argmax(spk_o, 1)
        chosen[tie] = rng.integers(A, size=int(tie.sum()))
        R_true = (chosen == correct).astype(float)        # task performance (returned)
        if reward_pools is None:
            R = R_true                                     # synthetic clean reward
        else:
            # biosignal reward: per seed draw a decoded reward VALUE from the EEG pool
            # matching the TRUE outcome valence (exactly as bandit.train) -- a real, noisy
            # reward-prediction-error gates the update, while R_true scores performance.
            R = np.array([reward_pools[int(R_true[b])][
                rng.integers(len(reward_pools[int(R_true[b])]))] for b in range(B)])

        # --- learning signals ---
        adv = (R - baseline)
        # OUTPUT layer: the proven three-factor scalar update dw = eta (R-b) e (the
        # contextual-bandit rule). Per-action selectivity comes from the SIGNED
        # eligibility itself -- the action neuron that fired has e>0, the others e<0 --
        # so the scalar advantage suffices and no softmax factor is needed (an explicit
        # (onehot-pol) factor at the output double-counts the action and destabilises the
        # update sign). This is identical across all modes; only the HIDDEN credit differs.
        Lo = adv[:, None, None]                                   # (B,1,1) global scalar

        if deep:
            if mode == "global":
                # pure global scalar to the hidden layer too: every hidden synapse sees
                # the SAME third factor, so the rule cannot tell which hidden unit helped
                # (the structural-credit failure mode this experiment is built to expose).
                Lh = adv[:, None, None]                           # (B,1,1)
            else:
                # DFA: a per-action error e_a = (R-b)(1[a=chosen]-pi_a) projected to the
                # hidden layer through a FIXED RANDOM feedback matrix B_fix (no W2^T, no
                # gradient transport). This gives each hidden unit a DIFFERENT, action-
                # dependent learning signal -- the per-neuron credit the global scalar
                # lacks -- while remaining fully local. pi is a softmax over the output
                # spike counts (used only to form the feedback signal, not the output update).
                logits = spk_o - spk_o.mean(1, keepdims=True)
                pol = np.exp(logits - logits.max(1, keepdims=True))
                pol /= pol.sum(1, keepdims=True)
                onehot = np.zeros((B, A)); onehot[bidx, chosen] = 1.0
                L_a = adv[:, None] * (onehot - pol)               # (B,A)
                Lh = np.einsum('ah,ba->bh', B_fix, L_a)[:, None, :]   # (B,1,H)

        W2 = np.clip(W2 + eta * Lo * eo_rew, -w_max, w_max)
        if train_w1:
            dW1 = eta_h * Lh * e1_rew
            if homeo > 0 and act_hidden is not None:
                # Local homeostatic regulation: update each hidden neuron's slow activity
                # estimate from this trial's firing fraction, then nudge its INCOMING
                # weights to push that activity toward the shared target. A neuron firing
                # above target has its drive scaled down, below target scaled up -- which
                # breaks the winner-take-all lock-in (an over-firing unit is reined in)
                # without touching the per-pattern selectivity the DFA signal learns.
                rate = spk_h / nsteps                              # (B,H) this-trial rate
                act_hidden += (rate - act_hidden) / homeo_tau      # slow running estimate
                # multiplicative scaling of the incoming weights toward balanced activity
                scale = 1.0 + homeo * (homeo_target - act_hidden)  # (B,H)
                dW1 = dW1 + (scale[:, None, :] - 1.0) * W1
            W1 = np.clip(W1 + dW1, -w_max, w_max)

        baseline += 0.02 * (R - baseline)        # baseline tracks the gating reward
        rewards[:, tr] = R_true                   # report TRUE task performance

        # Early stop: once the batch-mean reward over a trailing window has plateaued
        # above the criterion (or the run is clearly stuck near chance late on), the
        # remaining trials add no information -- pad with the converged level and break.
        # Cheap O(1) check; large speed-up on the many runs that converge well before
        # ``trials``. Disabled by ``early_stop=None``.
        if early_stop is not None and tr >= early_stop["min_trials"] and \
                (tr % early_stop["check_every"] == 0):
            w = early_stop["window"]
            recent = rewards[:, tr - w + 1:tr + 1].mean()
            prev = rewards[:, tr - 2 * w + 1:tr - w + 1].mean()
            converged = recent >= early_stop["criterion"] and abs(recent - prev) < early_stop["tol"]
            stuck = tr >= early_stop["stuck_after"] and recent < early_stop["chance"] + early_stop["tol"]
            if converged or stuck:
                rewards[:, tr + 1:] = recent      # pad remaining trials at current level
                break

    if log_align and deep:
        # align_log: dict with trial indices and per-seed angle (deg), shape (n_log, B)
        align_log = {"trials": np.array(align_t), "angle_deg": np.array(align_a)}

    if return_weights and log_align:
        return rewards, (W1, W2), align_log
    if return_weights:
        return rewards, (W1, W2)
    if log_align:
        return rewards, align_log
    return rewards


def reward_rate(rewards, window=100):
    rewards = np.asarray(rewards)
    if rewards.shape[-1] < window:
        return rewards.mean(axis=-1)
    return rewards[..., -window:].mean(axis=-1)


# =============================================================================
# Experiment cores (the deep all-local RL studies that compose ``train_deep``)
#
# Each ``run_*`` is SERIAL and returns the result grid as a plain dict -- no file
# I/O, no plotting, no stdout. Notebooks call these at a small (quick) seed/trial
# count and render the figures inline; ``main()`` (below) calls them at the
# published scale (and MAY use a multiprocessing.Pool over the coarse axis), then
# writes each grid under ``data/results/`` via ``paths.save_result``. The DMS
# temporal-distractor task (Experiment 12) is a shallow ``GateBankBatched`` variant
# whose difficulty is purely temporal, so it lives HERE alongside the deep stack
# rather than in the selectivity module.
# =============================================================================

# Frozen operating points (pilot-tuned, then frozen). Kept at
# module scope so both the ``run_*`` cores and ``main()`` share one source of truth.
DEEP_LOCAL_HP = dict(H=32, tau_leak=10.0, D=5.0, eta=0.2, eta_hidden=3.0,
                     fb_scale=2.0, bias_o=0.3, V=1.5, sigma0=0.15, sigma1=0.05)
DEEP_LOCAL_HOMEO = 0.1                 # frozen homeostatic strength (pilot-tuned)
DEEP_LOCAL_MODES = [
    "shallow", "elm", "global", "dfa", "no_trace", "no_trace_homeo", "dfa_homeo",
]

DEEP_METHOD_PROVENANCE = {
    "status": "proposed",
    "established_basis": [
        "direct feedback alignment", "firing-rate homeostasis",
        "three-factor reward modulation",
    ],
    "repository_adaptation": (
        "Fixed random feedback, the cascade eligibility surrogate and multiplicative "
        "homeostasis are combined in one local reinforcement-learning update."
    ),
    "claim_limit": (
        "The cited component methods do not establish this composite update; "
        "homeostasis parameters were pilot-tuned and then frozen."
    ),
}

# Experiment 13 operating point at the VERIFIED homeostasis regime (H=32 clean:
# homeo=0.1 -> 1.00 [1.00,1.00] vs homeo=0 -> 0.58 [0.47,0.69], gap +0.42, disjoint;
# device faults are exp14's separate axis, not confounded into the homeostasis claim).
DEEP_DMS_HP = dict(H=32, tau_leak=10.0, D=3.0, eta=0.2, eta_hidden=3.0,
                   fb_scale=2.0, bias_o=0.3, V=1.5, sigma0=0.15, sigma1=0.05)
DEEP_DMS_HOMEO = 0.1
DEEP_DMS_T_DISTRACT = 2.0              # distractor 2 s into the 3 s delay (1 s before reward)
DEEP_DMS_DISTRACT_DUR = 0.3
# (label, train_mode, homeo, distractor?)
DEEP_DMS_CONDS = [
    ("dfa_homeo_nodist", "dfa", DEEP_DMS_HOMEO, False),
    ("dfa_homeo_dist",   "dfa", DEEP_DMS_HOMEO, True),
    ("dfa_dist",         "dfa", 0.0,   True),
    ("no_trace_homeo_dist", "no_trace", DEEP_DMS_HOMEO, True),
    ("no_trace_dist",    "no_trace", 0.0, True),
]

# Experiment 14 array-scale early-stop (converged/stuck runs pad the rest at the
# reached level, leaving the final-200 metric unchanged; large speed-up at ceiling).
ARRAY_SCALE_EARLY_STOP = {
    "min_trials": 400, "check_every": 100, "window": 150,
    "criterion": 0.78, "tol": 0.03, "chance": 0.5, "stuck_after": 1200,
}


# ----------------------------------------------------------------------------
# Experiment 12: delayed-match-to-sample with a temporal distractor.
#
# The harder credit-assignment task identified as the single best fit for a SLOW
# (seconds-scale) eligibility trace: a delayed-association task whose difficulty is
# purely TEMPORAL (a seconds-scale gap and an interfering distractor), not a larger
# policy. It stresses the trace itself, sits in the device's 1-20 s regime, stays at
# 2 actions, and is the canonical "fails without an eligibility trace" regime (Bellec
# et al. e-prop, Nat. Commun. 2020, 10.1038/s41467-020-17236-y; DMS origin Miller,
# Erickson & Desimone, J. Neurosci. 1996, 10.1523/JNEUROSCI.16-16-05154.1996).
#
# The device band-pass trace (sigmoidal rise + tau_leak decay) is peaked at a
# non-zero lag, so with tau_leak tuned to the sample->reward interval it credits the
# sample over the later distractor -- the closed-loop, standard-task form of the
# interval-selectivity construction. The abstract single-exponential trace
# (matched tau) is monotone/recency-weighted (the informative
# comparison); the no-trace control must fail.
# ----------------------------------------------------------------------------
def _relax(bank, n_steps, dt, stride=10):
    """Fast-forward the (drive-free) eligibility gate by ``n_steps`` dt-ticks.

    During the silent delay there is no presynaptic drive and no spikes, so the gate
    just relaxes -- a smooth trajectory faithfully reproduced at a coarser effective
    step ``stride*dt`` (the same long-delay shortcut as ``train_sequential``;
    ``stride*dt`` stays well below ``tau_leak``/``tau_r``, so the cascade is unchanged).
    Returns the readout at the end of the interval.
    """
    if n_steps <= 0:
        return bank.step(np.zeros((bank.e.shape if hasattr(bank, "e") else
                                   bank.vn.shape[:-1])) * 0.0)
    zero = None
    base = bank.dt
    coarse = max(1, n_steps // stride)
    bank.dt = base * (n_steps / coarse)               # exact-coverage coarse step
    out = None
    for _ in range(coarse):
        if zero is None:
            # build a correctly-shaped zero drive once (shape (B,S,A))
            shp = bank.e.shape if hasattr(bank, "e") else bank.vn.shape[:-1]
            zero = np.zeros(shp)
        out = bank.step(zero)
    bank.dt = base
    return out


def run_dms(cond, *, B=20, tau_leak=1.5, G=5.0, t_distract=4.0, trials=2500,
            dt=5e-3, cue_dur=0.3, distract_dur=0.3, eta=0.2, V=1.5, in_rate=200.0,
            ltd=LTD_BIAS, tau_m=TAU_M, v_th=V_TH, sigma=0.15, seed0=0,
            device_k=K_STAGES, tau_r_override=None):
    """One DMS-with-distractor condition (Experiment 12). SERIAL, returns rewards
    ``(B, trials)``; no file I/O.

    State lines ``[sample0, sample1, distractor]`` -> 2 action neurons. A trial:
    sample coincidence at ``t in [0, cue_dur)``; distractor coincidence at
    ``t in [t_distract, t_distract+distract_dur)``; decision + reward at ``t = G``.
    The eligibility gate integrates continuously across the whole trial; the surviving
    trace at the reward instant (``t=G``) is what the three-factor rule commits.

    ``cond`` is ``"device"`` (historical single-rate cascade
    :class:`GateBankBatched`) or ``"abstract"`` (post-peak decay-matched single
    exponential :class:`AbstractTrace`, the recency-trace baseline)
    or ``"no_trace"`` (eligibility zeroed, the necessity control).

    Optimised by stepping the LIF + gate only during the two ACTIVE windows (sample,
    distractor) and fast-forwarding the gate through the two SILENT gaps with a coarse
    stride (:func:`_relax`); ~88% of dt-ticks are silent, so this is ~6x fewer
    iterations with no change to the dynamics that matter.
    """
    from .bandit import GateBankBatched, AbstractTrace, W_INIT, W_MAX
    rng = np.random.default_rng(seed0)
    S, A = 3, 2                       # [sample0, sample1, distractor] x {A, B}
    DIST_LINE = 2
    if cond == "abstract":
        matched_tau = decay_matched_exponential_tau(
            tau_leak, V=V, k=device_k, tau_r_override=tau_r_override,
        )
        bank = AbstractTrace(B, S, A, tau_elig=matched_tau, dt=dt)
    else:
        bank = GateBankBatched(B, S, A, tau_leak=tau_leak, V=V, dt=dt,
                               k=device_k, tau_r_override=tau_r_override)
    no_trace = (cond == "no_trace")

    w = np.full((B, S, A), W_INIT)
    baseline = np.full(B, 1.0 / A)
    bidx = np.arange(B)
    n_cue = int(round(cue_dur / dt))
    n_gap1 = int(round((t_distract - cue_dur) / dt))
    n_dist = int(round(distract_dur / dt))
    n_gap2 = int(round((G - t_distract - distract_dur) / dt))
    rewards = np.zeros((B, trials))

    def active_window(v, spk, sample_line, count_spikes, distractor):
        """Step LIF + gate for one active window; return (v, e_last)."""
        e = None
        nsteps = n_cue if not distractor else n_dist
        for _ in range(nsteps):
            pre = np.zeros((B, S))
            if not distractor:
                pre[bidx, sample_line] = (rng.random(B) < in_rate * dt).astype(float)
            else:
                pre[:, DIST_LINE] = (rng.random(B) < in_rate * dt).astype(float)
            charge = np.einsum('bsa,bs->ba', w, pre)
            v, sp = lif_step_batched(v, charge, dt, rng, tau_m=tau_m, v_th=v_th,
                                     noise=sigma)
            if count_spikes:
                spk += sp
            if no_trace:
                e = bank.step(np.zeros((B, S, A)))
            else:
                drive = pre[:, :, None] * np.where(sp, 1.0, -ltd)[:, None, :]
                e = bank.step(drive)
        return v, e

    for tr in range(trials):
        bank.reset()
        cls = rng.integers(2, size=B)                  # sample class this trial
        sample_line = np.where(cls == 0, 0, 1)
        v = np.zeros((B, A))
        spk = np.zeros((B, A))                          # decision-window spike count
        # SAMPLE window (informative; reads the decision)
        v, _ = active_window(v, spk, sample_line, count_spikes=True, distractor=False)
        # GAP 1 -- silent relaxation
        _relax(bank, n_gap1, dt)
        # DISTRACTOR window (uninformative; spikes not counted)
        v, _ = active_window(v, spk, sample_line, count_spikes=False, distractor=True)
        # GAP 2 -- silent relaxation to the reward instant; snapshot surviving trace
        e_rew = _relax(bank, n_gap2, dt)
        # action = spiking winner over the sample window (ties -> random); membrane
        # noise provides exploration
        tie = spk.max(1) == spk.min(1)
        chosen = np.argmax(spk, 1)
        chosen[tie] = rng.integers(A, size=int(tie.sum()))
        r = (chosen == cls).astype(float)              # match-to-sample reward
        w = np.clip(w + (eta * (r - baseline))[:, None, None] * e_rew, 0.0, W_MAX)
        baseline += 0.02 * (r - baseline)
        rewards[:, tr] = r
    return rewards


def final_rate(rw, window=300):
    """Per-seed reward rate over the last ``window`` trials (DMS final metric)."""
    return rw[:, -window:].mean(1)


def _summarize_dms(raw, conds, *, trials, chance, crit, tau_leak=1.5, G=5.0,
                   t_distract=4.0):
    """Package raw per-condition :func:`run_dms` output into the exp12 grid dict
    (seed-mean curves, per-seed finals + bootstrap CIs, retrospective H1-H3/K1).

    Retrospectively recorded:
      H1 device learns DMS-with-distractor:   device final >= crit
      H2 trace is necessary:                   no-trace <= chance + 0.10
      H3 band-pass beats recency at crediting the sample over the distractor:
         device CI lower bound > abstract CI upper bound (genuine separation)
      K1 (kill) device <= chance+0.10 -> the device-trace claim fails on this task
    """
    from .stats import bootstrap_ci
    curves, finals, ci = {}, {}, {}
    for cond in conds:
        rw = raw[cond]
        f = final_rate(rw)
        curves[cond] = rw.mean(0)
        finals[cond] = f
        ci[cond] = bootstrap_ci(f)
    dev = finals["device"].mean()
    nt = finals["no_trace"].mean()
    h1 = bool(dev >= crit)
    h2 = bool(nt <= chance + 0.10)
    h3 = bool(ci["device"][0] > ci["abstract"][1])       # genuine separation (CIs disjoint)
    k1 = bool(dev <= chance + 0.10)
    B = raw[conds[0]].shape[0]
    return {"curves": curves, "raw": {k: raw[k] for k in conds},
            "finals": {k: v for k, v in finals.items()},
            "ci": ci, "seeds": B, "trials": trials, "chance": chance, "crit": crit,
            "tau_leak": tau_leak, "G": G, "t_distract": t_distract,
            "retention_definition": "deliberately_swept",
            "method_provenance": DEEP_METHOD_PROVENANCE,
            "hyperparameter_provenance": "pilot_tuned_then_frozen",
            "criteria": {"H1": h1, "H2": h2, "H3": h3, "K1": k1}}


def _dms_one_job(job):
    """Spawn-safe coarse worker for one shallow-DMS condition."""
    cond, seeds, trials, device_k, tau_r_override = job
    return cond, run_dms(cond, B=seeds, trials=trials, device_k=device_k,
                         tau_r_override=tau_r_override)


def run_dms_all(*, seeds=20, trials=2500, conds=("device", "abstract", "no_trace"),
                pool=None, workers=1, device_k=K_STAGES, tau_r_override=None):
    """Experiment 12 core: DMS-with-distractor over all conditions.

    Serial by default; pass a spawn-safe process ``pool`` to distribute the coarse
    condition axis. Returns the packaged grid dict and performs no file I/O.
    """
    jobs = [(c, seeds, trials, device_k, tau_r_override) for c in conds]
    own_pool = None
    if pool is None and int(workers) > 1:
        from multiprocessing import get_context
        own_pool = get_context("spawn").Pool(min(int(workers), len(jobs)))
        pool = own_pool
    try:
        pairs = (pool.map(_dms_one_job, jobs, chunksize=1) if pool is not None
                 else map(_dms_one_job, jobs))
        raw = dict(pairs)
    finally:
        if own_pool is not None:
            own_pool.close()
            own_pool.join()
    return _summarize_dms(raw, conds, trials=trials, chance=0.5, crit=0.75)


# ----------------------------------------------------------------------------
# Experiment 7: deep all-local RL with a physical eligibility trace (XOR).
#
# Conditions isolate the spatial-credit pathway and include both a trace-only ablation
# (``no_trace_homeo`` keeps homeostasis fixed) and the historical double ablation
# (``no_trace`` also omits homeostasis). ``dfa_homeo`` adds the local homeostatic
# regulator. Legacy criteria were recorded retrospectively.
# ----------------------------------------------------------------------------
def _deep_local_one(mode, *, seeds, trials, hp, homeo):
    """Run one deep-local condition over all seeds. Returns (mode, finals[seeds],
    curve[trials-100]). The ``*_homeo`` modes keep the same frozen regulator."""
    kw = dict(hp)
    train_mode = mode
    if mode in {"dfa_homeo", "no_trace_homeo"}:
        train_mode = "dfa" if mode == "dfa_homeo" else "no_trace"
        kw["homeo"] = homeo
    rew = train_deep(mode=train_mode, B=seeds, trials=trials, seed0=0, **kw)
    finals = reward_rate(rew, window=200)             # per-seed final reward rate
    # running-mean learning curve (window 100), averaged over seeds
    win = 100
    csum = np.cumsum(rew, axis=1)
    run = (csum[:, win:] - csum[:, :-win]) / win      # (B, trials-win)
    curve = run.mean(0)
    return mode, finals, curve


def _summarize_deep_local(res, *, seeds, trials, hp, homeo, chance, crit,
                          modes=None):
    """Package raw ``(mode, finals, curve)`` triples into the exp7 grid dict with
    bootstrap CIs and the retrospective criteria C1-C4/C6/K4.

    C1 shallow fails (<= chance+0.10) -> depth is genuinely required
    C2 no-trace-with-homeostasis fails -> eligibility necessity
    C3 DFA learns (>= crit)
    C4 DFA > global (CIs disjoint)    -> per-neuron credit beats the global scalar
    C6 homeostasis fix (>= crit and DFA+homeo CI-above DFA)
    K4 (kill/bound) ELM shortcut solves XOR? (bounds the depth-without-learning claim)
    """
    from .stats import bootstrap_ci
    modes = list(modes) if modes is not None else DEEP_LOCAL_MODES
    finals = {m: f for m, f, _ in res}
    curves = {m: c for m, _, c in res}
    ci = {m: bootstrap_ci(finals[m]) for m in modes}
    dfa, homeo_f = finals["dfa"], finals["dfa_homeo"]
    c1 = bool(finals["shallow"].mean() <= chance + 0.10)
    c2 = bool(finals["no_trace_homeo"].mean() <= chance + 0.10)
    c3 = bool(dfa.mean() >= crit)
    c4 = bool(ci["dfa"][0] > ci["global"][1])            # CI gap excludes 0
    k4 = bool(finals["elm"].mean() >= crit)              # ELM shortcut solves it?
    c6 = bool(homeo_f.mean() >= crit and ci["dfa_homeo"][0] > ci["dfa"][1])
    return {"finals": finals, "curves": curves, "ci": ci, "HP": hp, "homeo": homeo,
            "seeds": seeds, "trials": trials, "chance": chance, "crit": crit,
            "retention_definition": "deliberately_swept",
            "method_provenance": DEEP_METHOD_PROVENANCE,
            "hyperparameter_provenance": "pilot_tuned_then_frozen",
            "criteria": {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C6": c6, "K4": k4}}


def _deep_local_job(job):
    """Spawn-safe coarse worker for one deep-local condition."""
    mode, seeds, trials, hp, homeo = job
    return _deep_local_one(mode, seeds=seeds, trials=trials, hp=hp, homeo=homeo)


def run_deep_local(*, seeds=20, trials=3000, hp=None, homeo=DEEP_LOCAL_HOMEO,
                   modes=None, chance=0.5, crit=0.75, pool=None, workers=1):
    """Experiment 7 core: deep all-local XOR over all conditions.

    Serial by default; pass a spawn-safe process ``pool`` to distribute the coarse
    condition axis. Returns the packaged grid dict and performs no file I/O.
    """
    hp = DEEP_LOCAL_HP if hp is None else hp
    modes = list(modes) if modes is not None else DEEP_LOCAL_MODES
    jobs = [(m, seeds, trials, hp, homeo) for m in modes]
    own_pool = None
    if pool is None and int(workers) > 1:
        from multiprocessing import get_context
        own_pool = get_context("spawn").Pool(min(int(workers), len(jobs)))
        pool = own_pool
    try:
        res = (pool.map(_deep_local_job, jobs, chunksize=1) if pool is not None
               else list(map(_deep_local_job, jobs)))
    finally:
        if own_pool is not None:
            own_pool.close()
            own_pool.join()
    return _summarize_deep_local(res, seeds=seeds, trials=trials, hp=hp, homeo=homeo,
                                 chance=chance, crit=crit, modes=modes)


# ----------------------------------------------------------------------------
# Experiment 13: the convergence test -- deep XOR with a temporal distractor.
#
# One task where the device eligibility trace must do TEMPORAL credit assignment
# (bridge a seconds-scale cue->reward delay past an interfering distractor) AND DFA +
# local homeostasis must do SPATIAL credit assignment through a hidden layer (XOR).
# ----------------------------------------------------------------------------
def _deep_dms_one(spec, *, seeds, trials, hp, t_distract, distract_dur):
    """Run one exp13 condition over all seeds. ``spec`` is (label, mode, homeo, dist).
    Returns (label, rewards (B, trials))."""
    label, mode, homeo, dist = spec
    kw = dict(hp)
    kw["homeo"] = homeo
    if dist:
        kw["t_distract"] = t_distract
        kw["distract_dur"] = distract_dur
    rew = train_deep(mode=mode, B=seeds, trials=trials, seed0=0, **kw)
    return label, rew                       # full (B, trials) for curve + bar


def _summarize_deep_dms(raw, conds, *, seeds, trials, hp, homeo, t_distract,
                        chance, crit):
    """Package raw exp13 output into the grid dict with per-seed finals, bootstrap
    CIs, seeds-solved robustness, and the retrospective H1-H3/K1.

    H1 full stack survives the distractor:   dfa_homeo+dist >= crit
    H2 eligibility necessary:                no_trace+homeostasis+dist <= chance + 0.10
    H3 homeostasis still helps under interference: dfa_homeo+dist > dfa+dist (CIs)
    K1 (kill) dfa_homeo+dist collapses to chance -> the convergence claim fails
    """
    from .stats import bootstrap_ci
    finals, ci, curves, seeds_solved = {}, {}, {}, {}
    for label, *_ in conds:
        rw = raw[label]
        f = reward_rate(rw, window=200)
        finals[label] = f
        ci[label] = bootstrap_ci(f)
        curves[label] = rw.mean(0)                      # seed-mean per trial
        seeds_solved[label] = int((f >= crit).sum())    # per-seed robustness
    fh = finals["dfa_homeo_dist"].mean()
    nt = finals["no_trace_homeo_dist"].mean()
    h1 = bool(fh >= crit)
    h2 = bool(nt <= chance + 0.10)
    h3 = bool(ci["dfa_homeo_dist"][0] > ci["dfa_dist"][1])   # homeostasis helps (CIs disjoint)
    k1 = bool(fh <= chance + 0.10)
    return {"finals": {k: v for k, v in finals.items()}, "ci": ci,
            "curves": curves, "seeds_solved": seeds_solved,
            "HP": hp, "homeo": homeo, "seeds": seeds, "trials": trials,
            "t_distract": t_distract, "chance": chance, "crit": crit,
            "retention_definition": "deliberately_swept",
            "method_provenance": DEEP_METHOD_PROVENANCE,
            "hyperparameter_provenance": "pilot_tuned_then_frozen",
            "criteria": {"H1": h1, "H2": h2, "H3": h3, "K1": k1}}


def _deep_dms_job(job):
    """Spawn-safe coarse worker for one deep temporal-distractor condition."""
    spec, seeds, trials, hp, t_distract, distract_dur = job
    return _deep_dms_one(spec, seeds=seeds, trials=trials, hp=hp,
                         t_distract=t_distract, distract_dur=distract_dur)


def run_deep_dms(*, seeds=20, trials=3000, hp=None, homeo=DEEP_DMS_HOMEO,
                 conds=None, t_distract=DEEP_DMS_T_DISTRACT,
                 distract_dur=DEEP_DMS_DISTRACT_DUR, chance=0.5, crit=0.75,
                 pool=None, workers=1):
    """Experiment 13 core: deep XOR plus temporal distractor over all conditions.

    Serial by default; pass a spawn-safe process ``pool`` to distribute the coarse
    condition axis. Returns the packaged grid dict and performs no file I/O.
    """
    hp = DEEP_DMS_HP if hp is None else hp
    conds = DEEP_DMS_CONDS if conds is None else conds
    jobs = [(spec, seeds, trials, hp, t_distract, distract_dur) for spec in conds]
    own_pool = None
    if pool is None and int(workers) > 1:
        from multiprocessing import get_context
        own_pool = get_context("spawn").Pool(min(int(workers), len(jobs)))
        pool = own_pool
    try:
        pairs = (pool.map(_deep_dms_job, jobs, chunksize=1) if pool is not None
                 else map(_deep_dms_job, jobs))
        raw = dict(pairs)
    finally:
        if own_pool is not None:
            own_pool.close()
            own_pool.join()
    return _summarize_deep_dms(raw, conds, seeds=seeds, trials=trials, hp=hp,
                               homeo=homeo, t_distract=t_distract,
                               chance=chance, crit=crit)


# ----------------------------------------------------------------------------
# Experiment 14: array-scale feasibility under specified simulated nonidealities.
#
# Does the device-eligibility three-factor rule (DFA + homeostasis, the deep all-local
# stack) keep learning as the network grows (hidden width H) AND under the specified
# SiO_x-inspired stress models (stuck-off + lognormal D2D + optionally the
# Poole-Frenkel I-V nonlinearity)? PASS = beats its own no-trace control
# with disjoint CIs AND > 0.75. KILL: K1 large-H fails at p=0 (no width composition);
# K2 high-p collapses everywhere; K3 control learns (artefact).
#
# The two provenance grids differ only in the fault prior:
#   exp14_array_scale       : pf_on=True,  sigma_g=0.5 symmetric  (the "full PF" grid);
#   exp14_array_scale_sweep : pf_on=False, sigma_g=0.5/sigma_g_on=0.05 asymmetric
#     (PF is a READ-path nonideality, excluded when crediting the PROGRAMMED weight).
# ``run_array_scale`` is the single core; the two are selected by its fault kwargs.
# ----------------------------------------------------------------------------
def _array_scale_run(job):
    """One independent training run -> (H, p, seed, trace, final reward rate).
    Picklable module-level worker for the ``main()`` Pool. ``job`` carries the fault-
    stack parameters so children need no shared config.
    """
    from .device_faults import siox_fault_stack
    H, p, seed, trace, fkw = job
    fault = siox_fault_stack(p_stuck=p, sigma_g=fkw["sigma_g"],
                             sigma_g_on=fkw["sigma_g_on"], pf_on=fkw["pf_on"],
                             stuck_kind="off", seed=1000 * seed + int(100 * p))
    rewards = train_deep(
        mode="dfa" if trace else "no_trace",
        H=H, tau_leak=10.0, D=5.0, t_distract=fkw["t_distract"], distract_dur=0.3,
        homeo=1.0 if trace else 0.0, trials=fkw["trials"], seed0=seed,
        weight_fault=fault, early_stop=ARRAY_SCALE_EARLY_STOP,
    )
    return (H, p, seed, trace, float(rewards[:, -200:].mean()))


def _array_scale_reduce(results, H_grid, p_grid, *, seeds, sigma_g, sigma_g_on,
                        pf_on, trials):
    """Reduce a list of ``_array_scale_run`` outputs into the exp14 grid dict
    (device mean/lo/hi per (H,p), control mean, fault label)."""
    from .stats import bootstrap_ci
    dev = {(H, p): [] for H in H_grid for p in p_grid}
    ctl = {(H, p): [] for H in H_grid for p in p_grid}
    for H, p, s, trace, val in results:
        (dev if trace else ctl)[(H, p)].append(val)
    grid = np.zeros((len(H_grid), len(p_grid), 3))       # mean, lo, hi (device)
    cgrid = np.zeros((len(H_grid), len(p_grid)))         # control mean
    passes = {}
    for i, H in enumerate(H_grid):
        for j, p in enumerate(p_grid):
            d = np.array(dev[(H, p)]); c = np.array(ctl[(H, p)])
            lo, hi = bootstrap_ci(d, seed=i * 10 + j)
            cmean = float(c.mean()) if c.size else np.nan
            grid[i, j] = (d.mean(), lo, hi); cgrid[i, j] = cmean
            passes[(H, p)] = bool(lo > cmean and d.mean() > 0.75) if c.size else None
    if pf_on:
        faults = "stuck-off + D2D(0.5) + PF-nonlinearity"
    else:
        faults = ("stuck-off + D2D({:g},{:g}) [PF excluded: read-path nonideality]"
                  .format(sigma_g_on if sigma_g_on is not None else sigma_g, sigma_g))
    included = ["stuck_off", "sampled_device_to_device_lognormal"]
    if pf_on:
        included.append("poole_frenkel_iv_nonlinearity")
    return {"H": list(H_grid), "p": list(p_grid), "grid": grid, "ctrl": cgrid,
            "faults": faults, "seeds": seeds, "trials": trials, "passes": passes,
            "retention_definition": "deliberately_swept",
            "method_provenance": DEEP_METHOD_PROVENANCE,
            "fault_scope": {
                "status": "adapted",
                "included_nonidealities": included,
                "excluded_nonidealities": ["line_resistance", "read_noise",
                    "temporal_noise", "drift", "programming_update_noise"],
                "claim_limit": "Simulation stress model, not a comprehensive measured fault prior.",
            }}


def run_array_scale(*, H_grid=(8, 32, 128, 512), p_grid=(0.0, 0.05, 0.20, 0.50),
                    seeds=12, trials=2000, sigma_g=0.5, sigma_g_on=None,
                    pf_on=True, pool=None, include_control=True):
    """Experiment 14 core: array-scale fault-tolerance feasibility (SERIAL by default,
    returns the packaged grid dict; no file I/O). Unifies ``array_scale_faults`` (the
    ``pf_on=True`` full-PF grid, ``exp14_array_scale.npy``) and ``entry_array_scale``
    (the ``pf_on=False`` asymmetric-D2D sweep, ``exp14_array_scale_sweep.npy``) -- the
    fault prior is selected by the ``sigma_g``/``sigma_g_on``/``pf_on`` kwargs.

    Every (H, p, seed) x {device, control} cell is an independent :func:`train_deep`
    call. In-notebook this runs serially; ``main()`` passes a ``multiprocessing.Pool``
    via ``pool`` to fan the jobs out across the coarse (cell x seed) axis. Set
    ``include_control=False`` when reproducing the Appendix heatmap, which plots only
    the device arm; the returned ``ctrl`` grid is then NaN and ``passes`` is unset.
    """
    H_grid = list(H_grid); p_grid = list(p_grid)
    fkw = dict(sigma_g=sigma_g, sigma_g_on=sigma_g_on, pf_on=pf_on,
               t_distract=3.0, trials=trials)
    traces = (True, False) if include_control else (True,)
    jobs = [(H, p, s, trace, fkw)
            for H in H_grid for p in p_grid for s in range(seeds)
            for trace in traces]
    if pool is None:
        results = [_array_scale_run(j) for j in jobs]
    else:
        results = pool.map(_array_scale_run, jobs, chunksize=1)
    return _array_scale_reduce(results, H_grid, p_grid, seeds=seeds, sigma_g=sigma_g,
                               sigma_g_on=sigma_g_on, pf_on=pf_on, trials=trials)


def main(argv=None):
    """Full-scale reproduction CLI for the deep all-local + DMS grids (writes ``data/results``).

    ``python -m mrl_trace.deep [--exp7] [--exp12] [--exp13] [--exp14]
    [--exp14-sweep] [--full|--quick]``

    With no experiment flag, runs all of exp7/exp12/exp13/exp14 (the sweep variant is
    opt-in via ``--exp14-sweep`` since it shares exp14's compute). ``--full`` = the
    published seed/trial scale; ``--quick`` = a fast few-seed smoke run. The exp14 grids
    fan their independent cell x seed runs across a ``multiprocessing.Pool``.
    """
    import argparse
    import os
    import multiprocessing as mp
    from . import paths
    ap = argparse.ArgumentParser(description="Deep all-local + DMS RL reproductions")
    ap.add_argument("--exp7", action="store_true",
                    help="deep all-local XOR -> exp7_deep_local.npy")
    ap.add_argument("--exp12", action="store_true",
                    help="DMS + temporal distractor -> exp12_dms.npy")
    ap.add_argument("--exp13", action="store_true",
                    help="deep XOR + temporal distractor (convergence) -> exp13_deep_dms.npy")
    ap.add_argument("--exp14", action="store_true",
                    help="array-scale faults (full PF) -> exp14_array_scale.npy")
    ap.add_argument("--exp14-sweep", action="store_true",
                    help="array-scale faults (asymmetric D2D, no PF) -> exp14_array_scale_sweep.npy")
    ap.add_argument("--quick", action="store_true", help="fast few-seed smoke run")
    ap.add_argument("--full", action="store_true", help="published-scale run (default)")
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 4),
                    help="Pool workers for the exp14 grid(s)")
    a = ap.parse_args(argv)
    any_flag = a.exp7 or a.exp12 or a.exp13 or a.exp14 or a.exp14_sweep
    run_default = not any_flag           # no flag -> exp7/12/13/14 (sweep opt-in)

    if a.exp7 or run_default:
        seeds = 6 if a.quick else 20
        trials = 600 if a.quick else 3000
        print(f"=== exp7 deep all-local XOR (N={seeds}, {trials} trials, H={DEEP_LOCAL_HP['H']}) ===")
        grid = run_deep_local(seeds=seeds, trials=trials)
        paths.save_result("exp7_deep_local.npy", grid)
        print(f"  wrote exp7_deep_local.npy  criteria={grid['criteria']}")

    if a.exp12 or run_default:
        seeds = 6 if a.quick else 20
        trials = 800 if a.quick else 2500
        print(f"=== exp12 DMS + distractor (N={seeds}, {trials} trials) ===")
        grid = run_dms_all(seeds=seeds, trials=trials)
        paths.save_result("exp12_dms.npy", grid)
        print(f"  wrote exp12_dms.npy  criteria={grid['criteria']}")

    if a.exp13 or run_default:
        seeds = 6 if a.quick else 20
        trials = 600 if a.quick else 3000
        print(f"=== exp13 deep XOR + distractor (N={seeds}, {trials} trials, "
              f"distractor@{DEEP_DMS_T_DISTRACT}s) ===")
        grid = run_deep_dms(seeds=seeds, trials=trials)
        paths.save_result("exp13_deep_dms.npy", grid)
        print(f"  wrote exp13_deep_dms.npy  criteria={grid['criteria']}")

    if a.exp14 or a.exp14_sweep or run_default:
        try:
            ctx = mp.get_context("fork")     # pure-NumPy single-threaded runs; fork is safe/fast
        except ValueError:                   # pragma: no cover (non-fork OS)
            ctx = mp.get_context()
        if a.quick:
            H_grid, p_grid, seeds, trials = (32, 512), (0.0, 0.5), 4, 400
        else:
            H_grid, p_grid, seeds, trials = (8, 32, 128, 512), (0.0, 0.05, 0.20, 0.50), 12, 2000
        n_cells = len(H_grid) * len(p_grid) * seeds * 2
        with ctx.Pool(min(a.workers, max(1, n_cells))) as pool:
            if a.exp14 or run_default:
                print(f"=== exp14 array-scale faults [full PF] (N={seeds}, H={list(H_grid)}, "
                      f"p={list(p_grid)}) ===")
                grid = run_array_scale(H_grid=H_grid, p_grid=p_grid, seeds=seeds,
                                       trials=trials, sigma_g=0.5, pf_on=True, pool=pool)
                paths.save_result("exp14_array_scale.npy", grid)
                print(f"  wrote exp14_array_scale.npy  faults={grid['faults']!r}")
            if a.exp14_sweep:
                print(f"=== exp14 array-scale faults [asymmetric D2D, no PF] (N={seeds}) ===")
                grid = run_array_scale(H_grid=H_grid, p_grid=p_grid, seeds=seeds,
                                       trials=trials, sigma_g=0.5, sigma_g_on=0.05,
                                       pf_on=False, pool=pool)
                paths.save_result("exp14_array_scale_sweep.npy", grid)
                print(f"  wrote exp14_array_scale_sweep.npy  faults={grid['faults']!r}")


if __name__ == "__main__":
    main()
