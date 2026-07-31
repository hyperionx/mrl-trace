"""Deep delayed contextual choice with a cascade eligibility surrogate.

This historical experiment composes a hidden LIF policy, Direct Feedback Alignment
(DFA), firing-rate homeostasis and the repository's custom signed eligibility rule.
The composite reinforcement-learning update is proposed here; citations for DFA,
homeostasis and three-factor learning support the components, not this exact method.

At episode start a two-bit cue is drawn and the rewarded terminal action is its XOR.
The environment then auto-advances through ``L`` position-labelled presentation windows:
actions during those windows do not affect transitions.  Only the final arm choice is
consequential.  Accordingly this is a deep delayed contextual decision with a temporal
input sequence, not a multi-decision T-maze or evidence of trajectory-level policy credit.
The cue remains present throughout, and a delayed reward gates the eligibility surrogate
after the final choice.

``GateBankBatched`` is an approximate computational cascade, not the empirical Au current
model itself.  Homeostasis settings were pilot-tuned and then frozen.  Returned reward
arrays therefore report behaviour of this repository-specific simulation only.
"""
from __future__ import annotations

import numpy as np

from .bandit import GateBankBatched, W_MAX
from .neurons import lif_step_batched, TAU_M, V_TH
from .learning import LTD_BIAS
from .maze import reward_rate

__all__ = ["train_deep_sequential", "reward_rate", "DEEP_DELAYED_CHOICE_PROVENANCE"]


DEEP_DELAYED_CHOICE_PROVENANCE = {
    "status": "proposed",
    "established_basis": [
        "LIF dynamics",
        "Direct Feedback Alignment",
        "firing-rate homeostasis",
        "three-factor reward modulation",
    ],
    "repository_adaptation": (
        "A deep delayed XOR choice combining the cascade eligibility surrogate, "
        "custom signed coincidence, DFA and multiplicative homeostasis."
    ),
    "claim_limit": (
        "The stem auto-advances and only the terminal choice affects reward; this is a "
        "delayed contextual decision, not multi-decision trajectory credit assignment."
    ),
}


def _relax_gate(bank, n_relax, stride, rem):
    """Advance an undriven :class:`GateBankBatched` for ``n_relax*stride + rem`` ticks,
    using a coarsened step (``stride*dt``) for the bulk and single ``dt`` steps for the
    remainder. Identical to :func:`mrl_trace.deep._relax_gate`: with zero drive
    the dynamics are smooth, so the coarse step is accurate while ``stride*dt/tau << 1``.
    Restores the gate's ``dt``."""
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


def _cue(rng, B):
    """Draw ``B`` XOR cues. Two binary features ``(b0, b1)``; correct arm = ``b0 ^ b1``.
    The features are one-hot-per-feature encoded onto ``Fc = 4`` cue lines
    ``[b0=0, b0=1, b1=0, b1=1]`` so exactly two cue lines are active every episode (a
    constant cue drive across all four states; no input-rate confound). XOR is not
    linearly separable in these lines, so a single trained layer cannot solve it -- this
    is what forces the hidden layer. Returns ``(cue_lines (B,4), correct (B,))``."""
    b0 = rng.integers(2, size=B)
    b1 = rng.integers(2, size=B)
    lines = np.zeros((B, 4))
    bidx = np.arange(B)
    lines[bidx, 0 + b0] = 1.0
    lines[bidx, 2 + b1] = 1.0
    correct = (b0 ^ b1).astype(int)
    return lines, correct


def train_deep_sequential(*, mode="dfa", B=20, H=16, L=4, A=2, tau_leak=20.0, D=3.0,
                          trials=2000, dt=5e-3, step_dur=0.4, dec_dur=1.0, eta=0.5,
                          eta_hidden=0.4,
                          in_rate=200.0, ltd=LTD_BIAS, tau_m=TAU_M, v_th=V_TH, V=1.5,
                          sigma0=0.15, sigma1=0.05, fb_scale=1.0, w_scale1=0.6,
                          w_scale2=0.3, w_max=W_MAX, bias_o=0.35, homeo=1.0,
                          homeo_target=0.35, homeo_tau=200.0, weight_fault=0.0,
                          early_stop=None, seed0=0, return_weights=False):
    """Deep delayed XOR choice, ``B`` parallel seeds; returns rewards ``(B, trials)``.

    Architecture (extends :func:`train_deep` with a stem-position input block):
      ``F = 4 + L`` input lines -> ``H`` hidden LIF neurons -> ``A`` action LIF neurons,
      with device-synapse matrices ``W1 (F,H)`` and ``W2 (H,A)``. The first 4 lines carry
      the XOR cue (persistent context, active on every step so the junction can decide on
      it); the next ``L`` lines are a one-hot of the current stem position (so each step
      is a distinct presentation window). Each weight matrix has its own
      :class:`GateBankBatched` eligibility surrogate (retention ``tau_leak``, custom signed
      drive) that integrates CONTINUOUSLY across all ``L`` stem steps -- never reset
      between steps.

    Delayed contextual sequence. The agent auto-advances through ``L`` presentation
    windows regardless of its actions, so the only consequential decision is the arm
    chosen at the final window. Reward
    ``R in {0,1}``, contingent on the junction action matching ``XOR(b0, b1)``, is
    delivered after the action->reward delay ``D``. The surviving eligibility then gates
    into both layers.

    ``mode`` selects the spatial-credit pathway (eligibility uses the cascade surrogate):
      ``shallow``  one trained layer F->A (should FAIL XOR -- depth-necessity control);
      ``elm``      hidden layer fixed random, only the output trained;
      ``global``   both layers trained by the pure global scalar (R-b) (structural-credit
                   failure mode);
      ``dfa``      both layers trained by the repository's DFA composite;
      ``no_trace`` deep DFA with eligibility zeroed (trace-necessity control).

    ``homeo > 0`` enables local homeostatic regulation of hidden-unit firing; its
    parameters were pilot-tuned and then frozen.
    ``weight_fault`` randomly stuck-at-zeroes that fraction of W1/W2 synapses (a device
    yield stressor; 0 disables). ``early_stop`` (a reward-rate threshold) stops a seed's
    updates once its trailing reward rate exceeds it (the policy is then run frozen);
    None disables. The signature is batch-sweep compatible.
    """
    if mode not in {"shallow", "elm", "global", "dfa", "no_trace"}:
        raise ValueError(f"unknown mode {mode!r}")
    rng = np.random.default_rng(seed0)
    Fc = 4
    F = Fc + L
    eta_h = eta if eta_hidden is None else eta_hidden
    deep = mode != "shallow"
    no_trace = mode == "no_trace"
    train_w1 = deep and mode != "elm"

    def _init(shape, scale):
        return np.clip(scale * rng.standard_normal(shape), -w_max, w_max)

    if deep:
        W1 = _init((B, F, H), w_scale1)
        W2 = _init((B, H, A), w_scale2)
    else:
        W2 = _init((B, F, A), w_scale1)
        W1 = None

    # --- weight-fault mask: stuck-at-zero synapses (device yield stressor) ---
    m1 = m2 = None
    if weight_fault > 0:
        if deep:
            m1 = (rng.random((B, F, H)) >= weight_fault).astype(float)
            W1 *= m1
        m2 = (rng.random(W2.shape) >= weight_fault).astype(float)
        W2 *= m2

    # --- device eligibility gates (one bank per trained matrix) ---
    g1 = GateBankBatched(B, F, H, tau_leak=tau_leak, dt=dt, V=V) if deep else None
    g_out = GateBankBatched(B, (H if deep else F), A, tau_leak=tau_leak, dt=dt, V=V)

    # --- DFA feedback matrix: fixed random, drawn ONCE, independent of W2 (no transport) ---
    B_fix = fb_scale * rng.standard_normal((A, H)) if deep else None

    baseline = np.full(B, 1.0 / A)
    bidx = np.arange(B)
    steps_per_state = max(1, int(round(step_dur / dt)))
    dec_dwell = max(1, int(round(dec_dur / dt)))     # longer dwell at the junction
    reward_lag = int(round(D / dt))
    rewards = np.zeros((B, trials))

    # Local homeostatic activity estimate per hidden neuron (see train_deep).
    act_hidden = np.full((B, H), homeo_target) if (deep and homeo > 0) else None
    frozen = np.zeros(B, dtype=bool)        # early-stop: seeds whose updates are frozen

    for tr in range(trials):
        sigma = sigma0 + (sigma1 - sigma0) * tr / trials
        if deep:
            g1.reset()
        g_out.reset()
        cue_lines, correct = _cue(rng, B)
        vh = np.zeros((B, H))
        vo = np.zeros((B, A))
        # junction-decision accumulators (the LAST stem step is the decision point)
        spk_o_dec = np.zeros((B, A))
        spk_h_dec = np.zeros((B, H))

        # --- auto-advance through the L presentation windows ---
        # The cue lines are presented on every step (persistent context); the position
        # line for the current window is active too. Actions before the last window have no
        # transition consequence, so this remains a delayed contextual decision. The
        # eligibility gates integrate across ALL steps (never reset). Only the LAST step
        # is the junction where the arm choice is read out (auto-advance corridor).
        for st in range(L):
            decision_step = (st == L - 1)
            lines = np.zeros((B, F))
            lines[:, :Fc] = cue_lines
            lines[:, Fc + st] = 1.0                      # one-hot stem position
            # The agent dwells LONGER at the junction (the decision point) than at the
            # auto-advanced stem states: more spike accumulation -> a cleaner argmax arm
            # choice and a stronger (less noisy) output-layer coincidence to credit. The
            # extra dwell also lengthens the cue->reward gap, which the device must bridge.
            dwell = dec_dwell if decision_step else steps_per_state
            for _ in range(dwell):
                pre_in = (rng.random((B, F)) < (in_rate * dt) * lines).astype(float)
                if deep:
                    ch_h = np.einsum('bfh,bf->bh', W1, pre_in)
                    vh, sp_h = lif_step_batched(vh, ch_h, dt, rng, tau_m=tau_m,
                                                v_th=v_th, noise=sigma)
                    ch_o = np.einsum('bha,bh->ba', W2, sp_h.astype(float)) + bias_o
                    vo, sp_o = lif_step_batched(vo, ch_o, dt, rng, tau_m=tau_m,
                                                v_th=v_th, noise=sigma)
                    if decision_step:
                        spk_h_dec += sp_h
                        spk_o_dec += sp_o
                    # Signed leak-dominant coincidence. The INPUT->HIDDEN bank g1 is
                    # driven on EVERY step: its cue+position rows differ per step, so it
                    # accumulates eligibility across presentation windows (early activity
                    # decays more than late activity by reward time).
                    # The HIDDEN->OUTPUT bank g_out shares the same (H,A) synapses across
                    # steps, so a stem-step output coincidence (made by an as-yet random
                    # arm decision) would POLLUTE the junction credit; the arm decision --
                    # and the only credit-relevant output coincidence -- happens at the
                    # junction, so g_out is driven only on the decision step (the maze
                    # analogue: each step's coincidence lands on its own state row, the
                    # junction decision on its own row).
                    if no_trace:
                        g1.step(np.zeros((B, F, H)))
                        g_out.step(np.zeros((B, H, A)))
                    else:
                        drive1 = pre_in[:, :, None] * np.where(sp_h, 1.0, -ltd)[:, None, :]
                        g1.step(drive1)
                        if decision_step:
                            drive_o = (sp_h.astype(float)[:, :, None]
                                       * np.where(sp_o, 1.0, -ltd)[:, None, :])
                            g_out.step(drive_o)
                        else:
                            g_out.step(np.zeros((B, H, A)))
                else:
                    ch_o = np.einsum('bfa,bf->ba', W2, pre_in)
                    vo, sp_o = lif_step_batched(vo, ch_o, dt, rng, tau_m=tau_m,
                                                v_th=v_th, noise=sigma)
                    if decision_step:
                        spk_o_dec += sp_o
                    drive_o = pre_in[:, :, None] * np.where(sp_o, 1.0, -ltd)[:, None, :]
                    g_out.step(drive_o)

        # --- relax for the action->reward delay, then snapshot eligibility ---
        if not no_trace:
            stride = 10 if reward_lag > 200 else 1
            n_relax, rem = divmod(reward_lag, stride)
            _relax_gate(g_out, n_relax, stride, rem)
            if deep:
                _relax_gate(g1, n_relax, stride, rem)
            eo_rew = g_out.vn[..., -1] / g_out.Vnmax
            e1_rew = (g1.vn[..., -1] / g1.Vnmax) if deep else None
        else:
            eo_rew = np.zeros_like(g_out.vn[..., -1])
            e1_rew = np.zeros((B, F, H)) if deep else None

        # --- arm selection at the junction (argmax decision-step spikes; ties -> rand) ---
        tie = spk_o_dec.max(1) == spk_o_dec.min(1)
        chosen = np.argmax(spk_o_dec, 1)
        chosen[tie] = rng.integers(A, size=int(tie.sum()))
        R = (chosen == correct).astype(float)

        # --- learning signals (identical math/signs to train_deep) ---
        adv = (R - baseline)
        Lo = adv[:, None, None]                                   # output: global scalar
        if deep:
            if mode == "global":
                Lh = adv[:, None, None]
            else:
                logits = spk_o_dec - spk_o_dec.mean(1, keepdims=True)
                pol = np.exp(logits - logits.max(1, keepdims=True))
                pol /= pol.sum(1, keepdims=True)
                onehot = np.zeros((B, A)); onehot[bidx, chosen] = 1.0
                L_a = adv[:, None] * (onehot - pol)               # (B,A)
                Lh = np.einsum('ah,ba->bh', B_fix, L_a)[:, None, :]   # (B,1,H)

        upd = (~frozen).astype(float)[:, None, None]              # freeze early-stopped seeds
        W2 = np.clip(W2 + upd * eta * Lo * eo_rew, -w_max, w_max)
        if train_w1:
            dW1 = eta_h * Lh * e1_rew
            if homeo > 0 and act_hidden is not None:
                rate = spk_h_dec / max(1, dec_dwell)              # decision-step firing rate
                act_hidden += (rate - act_hidden) / homeo_tau
                scale = 1.0 + homeo * (homeo_target - act_hidden)  # (B,H)
                dW1 = dW1 + (scale[:, None, :] - 1.0) * W1
            W1 = np.clip(W1 + upd * dW1, -w_max, w_max)
        if m1 is not None:
            W1 *= m1
        if m2 is not None:
            W2 *= m2

        baseline += 0.02 * (R - baseline)
        rewards[:, tr] = R

        if early_stop is not None and tr >= 150:
            rr = rewards[:, max(0, tr - 149):tr + 1].mean(axis=1)
            frozen |= (rr >= early_stop)

    if return_weights:
        return rewards, (W1, W2)
    return rewards
