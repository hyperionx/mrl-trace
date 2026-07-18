"""Distal-cue single-deposit task -- retention as the credit-assignment asset.

This module replaces the two temporally-weak experiments on the package's temporal
axis (the shallow T-maze of :func:`mrl_trace.maze.train_sequential` and the deep
T-maze of :func:`mrl_trace.deep_sequential.train_deep_sequential`) with a task
that is *designed to stress the eligibility trace* so that long device retention is a
measurable asset rather than a decoration.

WHY THE OLD TASKS WERE WEAK. In both T-mazes the informative cue is re-presented on
*every* stem step (it is a persistent context line), so the cue->action eligibility is
effectively *refreshed* the whole way down the corridor. A short-retention device then
bridges any gap (verified: TMaze device tau=1.5 s scores ~0.99 even at a 12 s delay),
because the trace it must carry is topped up continuously and never has to *survive* on
its own. The flat device-vs-length curve therefore reads as "the task never stressed the
trace", not "long retention bridges a gap a short trace cannot".

THE FIX -- A SINGLE DEPOSIT. Here the informative cue drives the credited synapses'
eligibility coincidence in ONE early window only. After that window the cue is never
re-presented and the credited eligibility is never driven again: it can only DECAY
(relax) while the agent traverses the rest of the trajectory / waits out the delay. A
single delayed reward then gates whatever eligibility has SURVIVED on those synapses.
As the trajectory-time / delay gap ``T_gap`` grows past the device retention
``tau_leak``, the surviving deposit on a SHORT-tau device collapses toward zero (no
credit, chance behaviour) while a LONG-tau device still carries it (credit assigned,
above-chance behaviour). That crossover -- device-LONG beating device-SHORT only once
the gap exceeds the short retention -- is the sequential form of the ``D_max ~ k*tau``
law: the learnable credit horizon scales with the device retention, a fabrication /
defect-engineering knob.

THE CONTROL THAT MUST FAIL IS A SHORT-RETENTION DEVICE, NOT AN ABSTRACT TRACE. A linear
exponential eligibility decaying over the gap rescales *every* credited synapse by the
SAME factor, so it preserves the relative credit pattern and still learns the policy
(abstract-short stays high even at a large gap). Only the device-long vs device-SHORT
contrast isolates retention as the asset, because the trap-cascade gate's *nonlinear*
rise+leak means a short ``tau_leak`` deposit decays below the noise floor and loses the
*pattern*, not merely its scale. ``abstract`` is reported here only as a baseline; the
headline contrast is device-long vs device-short (the intra-device STC fabrication-knob
story).

Two variants share the design:

- :func:`train_distal_shallow` -- the SHALLOW replacement for the shallow T-maze: a single
  trained ``context x action`` device-synapse layer; cue deposited once, trajectory/delay,
  then a single gated reward.
- :func:`train_distal_deep` -- the DEEP strengthening of
  :func:`mrl_trace.deep_sequential.train_deep_sequential`: it reuses that module's
  hidden-layer + DFA + homeostasis machinery but makes the XOR cue a SINGLE early deposit
  (the cue lines are silent after the deposit window) so the gap genuinely stresses the
  trace.

Conventions follow the package: pure NumPy, ``B`` seeds as a vectorised batch (leading
axis ``B``), device gate via :class:`mrl_trace.bandit.GateBankBatched` (``tau_leak``
the retention), signed leak-dominant coincidence, three-factor update
``dw = eta (R - b) e``, ``dt = 5e-3`` s, coarsened undriven relaxation for the gap.
"""
from __future__ import annotations

import numpy as np

from .bandit import GateBankBatched, AbstractTrace, W_INIT, W_MAX
from .neurons import lif_step_batched, TAU_M, V_TH
from .learning import LTD_BIAS
from .maze import reward_rate, trials_to_criterion
from . import deep_sequential

__all__ = ["train_distal_shallow", "train_distal_deep", "reward_rate",
           "trials_to_criterion", "run_distal_cue", "main"]


# ----------------------------------------------------------------------------
# Shared: coarsened undriven relaxation of a GateBankBatched / AbstractTrace
# ----------------------------------------------------------------------------
def _relax(bank, reward_lag, S, A):
    """Advance an undriven eligibility bank for ``reward_lag`` dt-ticks.

    With zero drive the cascade / leaky dynamics are smooth, so the bulk is integrated
    with a coarsened step (``stride*dt``) for tractability on long gaps and the
    remainder with single ``dt`` steps (the same scheme as
    :func:`mrl_trace.maze._relax`). ``bank`` may be a :class:`GateBankBatched`
    or an :class:`AbstractTrace`; both expose ``step`` and ``dt``. Restores ``dt``.
    """
    if reward_lag <= 0:
        return
    z = np.zeros((bank.B if hasattr(bank, "B") else bank.e.shape[0], S, A))
    stride = 10 if reward_lag > 200 else 1
    n_relax, rem = divmod(reward_lag, stride)
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


def _read_elig(bank, abstract):
    """Read the per-synapse eligibility snapshot (device cascade tip, or abstract e)."""
    if abstract:
        return np.clip(bank.e, -1.0, 1.0)
    return bank.vn[..., -1] / bank.Vnmax


# ----------------------------------------------------------------------------
# SHALLOW variant -- replacement for the shallow T-maze
# ----------------------------------------------------------------------------
def train_distal_shallow(*, B=20, tau_leak=20.0, T_gap=4.0, C=4, A=2, trials=1500,
                         dt=5e-3, cue_dur=1.0, dec_dur=1.0, eta=0.2, V=0.9,
                         in_rate=200.0, ltd=LTD_BIAS, tau_m=TAU_M, v_th=V_TH,
                         sigma0=0.15, sigma1=None, abstract=False, no_trace=False,
                         weight_fault=None, seed0=0, return_weights=False):
    """Shallow distal-cue single-deposit task; ``B`` seeds; returns rewards ``(B, trials)``.

    Architecture: ``C`` context lines -> ``A`` action LIF neurons through a single trained
    device-synapse grid ``w (C, A)``. Rewarded mapping: context ``c`` -> action ``c % A``.

    Each trial is a trajectory with three phases:

      1. DEPOSIT (``cue_dur`` s). The context line ``c`` for this trial fires; the action
         neurons spike; the signed leak-dominant pre x post coincidence drives the
         eligibility gate ONLY on the active context row. This is the single, literal
         deposit -- the only informative drive onto the credited synapses.
      2. GAP (``T_gap`` s). The cue is REMOVED and the credited eligibility is never driven
         again -- it only relaxes. (Implemented as a coarsened undriven relaxation, the
         trajectory-time / delay the trace must survive.)
      3. DECISION (``dec_dur`` s). The context line is re-presented PURELY to read out an
         action (Kirchhoff bitline sum -> spiking argmax); the eligibility is NOT driven
         here, so the credit gated by the reward comes only from the SURVIVING early
         deposit, not from any decision-time refresh.

    A single reward ``R in {0,1}`` (action == ``c % A``) then gates the surviving deposit:
    ``dw = eta (R - b) e``. As ``T_gap`` grows past ``tau_leak`` a short-tau device's
    deposit decays away (chance) while a long-tau device's survives (above chance).

    ``abstract`` swaps the device gate for the exponential trace (baseline); ``no_trace``
    zeroes the eligibility (device-necessity control). Signature is batch-sweep
    compatible (B, tau_leak, T_gap, trials, seed0, plus the sweep params).
    """
    rng = np.random.default_rng(seed0)
    bank = (AbstractTrace(B, C, A, tau_elig=tau_leak, dt=dt) if abstract
            else GateBankBatched(B, C, A, tau_leak=tau_leak, dt=dt, V=V))
    w = np.full((B, C, A), W_INIT)
    correct = np.array([c % A for c in range(C)])
    baseline = np.full(B, 1.0 / A)
    bidx = np.arange(B)
    dep_ticks = max(1, int(round(cue_dur / dt)))
    dec_ticks = max(1, int(round(dec_dur / dt)))
    gap_lag = int(round(T_gap / dt))
    rewards = np.zeros((B, trials))

    for tr in range(trials):
        sigma = sigma0 if sigma1 is None else sigma0 + (sigma1 - sigma0) * tr / trials
        bank.reset()
        context = rng.integers(C, size=B)
        v = np.zeros((B, A))
        # Read-time device faults applied ONCE per trial (weights change only per trial):
        # the array reads the faulted conductance wr while learning targets the clean w --
        # a fixed physical realisation, not a learnable parameter (as in deep.py/maze.py).
        # The maze weights are non-negative, so weight_fault should be a maze_fault_stack.
        wr = weight_fault(w) if weight_fault is not None else w

        # --- 1. DEPOSIT: cue fires, single signed coincidence onto context row ---
        for _ in range(dep_ticks):
            pre = np.zeros((B, C))
            pre[bidx, context] = (rng.random(B) < in_rate * dt).astype(float)
            charge = np.einsum('bca,bc->ba', wr, pre)
            v, sp = lif_step_batched(v, charge, dt, rng, tau_m=tau_m, v_th=v_th,
                                     noise=sigma)
            drive = np.zeros((B, C, A))
            drive[bidx, context, :] = pre[bidx, context][:, None] * np.where(sp, 1.0, -ltd)
            if no_trace:
                drive[:] = 0.0
            bank.step(drive)

        # --- 2. GAP: cue removed; credited eligibility only relaxes (single deposit) ---
        if not no_trace:
            _relax(bank, gap_lag, C, A)

        # snapshot the SURVIVING deposit -- this is what the reward will gate
        e_rew = np.zeros((B, C, A)) if no_trace else _read_elig(bank, abstract)

        # --- 3. DECISION: re-present cue to READ OUT an action (no eligibility drive) ---
        v[:] = 0.0
        spk = np.zeros((B, A))
        for _ in range(dec_ticks):
            pre = np.zeros((B, C))
            pre[bidx, context] = (rng.random(B) < in_rate * dt).astype(float)
            charge = np.einsum('bca,bc->ba', wr, pre)
            v, sp = lif_step_batched(v, charge, dt, rng, tau_m=tau_m, v_th=v_th,
                                     noise=sigma)
            spk += sp
            # NOTE: bank is NOT stepped here -- the decision must not refresh the deposit.

        tie = spk.max(1) == spk.min(1)
        chosen = np.argmax(spk, 1)
        chosen[tie] = rng.integers(A, size=int(tie.sum()))
        R = (chosen == correct[context]).astype(float)

        adv = eta * (R - baseline)
        w = np.clip(w + adv[:, None, None] * e_rew, 0.0, W_MAX)
        baseline += 0.02 * (R - baseline)
        rewards[:, tr] = R

    if return_weights:
        return rewards, w
    return rewards


# ----------------------------------------------------------------------------
# DEEP variant -- strengthening of deep_sequential.train_deep_sequential
# ----------------------------------------------------------------------------
def train_distal_deep(*, mode="dfa", B=20, H=16, L=4, A=2, tau_leak=20.0, T_gap=4.0,
                      trials=2000, dt=5e-3, step_dur=0.4, dec_dur=1.0, eta=0.5,
                      eta_hidden=0.4, in_rate=200.0, ltd=LTD_BIAS, tau_m=TAU_M,
                      v_th=V_TH, V=1.5, sigma0=0.15, sigma1=0.05, fb_scale=1.0,
                      w_scale1=0.6, w_scale2=0.3, w_max=W_MAX, bias_o=0.35, homeo=1.0,
                      homeo_target=0.35, homeo_tau=200.0, weight_fault=None, seed0=0,
                      return_weights=False):
    """Deep distal-cue single-deposit XOR task; ``B`` seeds; returns rewards ``(B, trials)``.

    This reuses the hidden-layer + DFA + homeostasis machinery of
    :func:`mrl_trace.deep_sequential.train_deep_sequential` (``F = 4 + L`` input
    lines -> ``H`` hidden LIF (DFA + homeostasis) -> ``A`` action LIF; device gates
    ``g1, g_out``; junction-only output coincidence) but fixes the temporal weakness:
    the XOR CUE IS A SINGLE EARLY DEPOSIT.

    Difference from ``train_deep_sequential``: the 4 cue lines are active ONLY during the
    first stem step (the deposit). For the remaining ``L-1`` stem steps and the junction
    the cue lines are SILENT -- only the one-hot stem-position line for the current step
    fires. The input->hidden eligibility ``g1`` therefore receives its informative
    cue->hidden coincidence only at the deposit; thereafter the cue rows of ``g1`` only
    decay across the trajectory and the ``T_gap`` action->reward delay. The surviving cue
    deposit -- not a re-presentation at the junction -- must carry the XOR association into
    the credited synapses. The position lines still make each step a distinct state (a real
    trajectory), and the junction-step output coincidence still credits the arm choice, so
    the deep all-local credit path is unchanged; only the cue's *refresh* is removed.

    As ``T_gap`` grows past ``tau_leak`` the surviving cue deposit on a short-tau device
    collapses (the XOR association is lost; chance) while a long-tau device retains it
    (above chance). ``mode`` (``shallow``/``elm``/``global``/``dfa``/``no_trace``),
    ``homeo``, and all other kwargs mirror ``train_deep_sequential``; the signature is
    batch-sweep compatible (B, H, tau_leak, T_gap, trials, seed0, mode, homeo, ...).
    """
    if mode not in {"shallow", "elm", "global", "dfa", "no_trace"}:
        raise ValueError(f"unknown mode {mode!r}")
    rng = np.random.default_rng(seed0)
    _cue = deep_sequential._cue
    _relax_gate = deep_sequential._relax_gate
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

    g1 = GateBankBatched(B, F, H, tau_leak=tau_leak, dt=dt, V=V) if deep else None
    g_out = GateBankBatched(B, (H if deep else F), A, tau_leak=tau_leak, dt=dt, V=V)
    B_fix = fb_scale * rng.standard_normal((A, H)) if deep else None

    baseline = np.full(B, 1.0 / A)
    bidx = np.arange(B)
    steps_per_state = max(1, int(round(step_dur / dt)))
    dec_dwell = max(1, int(round(dec_dur / dt)))
    gap_lag = int(round(T_gap / dt))
    rewards = np.zeros((B, trials))

    act_hidden = np.full((B, H), homeo_target) if (deep and homeo > 0) else None

    for tr in range(trials):
        sigma = sigma0 + (sigma1 - sigma0) * tr / trials
        if deep:
            g1.reset()
        g_out.reset()
        cue_lines, correct = _cue(rng, B)
        # Read-time device faults applied ONCE per trial: the array reads faulted weights
        # (W1r/W2r) while learning targets the clean W1/W2 (as in deep.py). Signed weights,
        # so weight_fault should be a siox_fault_stack.
        W1r = weight_fault(W1) if (weight_fault is not None and deep) else W1
        W2r = weight_fault(W2) if weight_fault is not None else W2
        vh = np.zeros((B, H))
        vo = np.zeros((B, A))
        spk_o_dec = np.zeros((B, A))
        spk_h_dec = np.zeros((B, H))

        # --- traverse the L-step stem; cue ELIGIBILITY is deposited ONLY on step 0 ---
        # Cue->hidden eligibility row mask: 1 during the deposit step, 0 afterwards. The
        # cue lines stay present in the FORWARD pass on every step (so the junction can
        # READ OUT the XOR -- the decision must be able to depend on the cue), but the
        # eligibility coincidence onto the cue rows of g1 is driven ONLY at the deposit.
        # The credit the reward later gates onto the cue->hidden synapses therefore comes
        # solely from the SURVIVING early deposit, never a junction-time refresh. The
        # position rows drive g1 every step (a genuine per-step trajectory state).
        for st in range(L):
            decision_step = (st == L - 1)
            deposit_step = (st == 0)
            lines = np.zeros((B, F))
            lines[:, :Fc] = cue_lines            # cue present every step (forward readout)
            lines[:, Fc + st] = 1.0              # one-hot stem position (every step)
            elig_row = np.ones(F)                # gate cue ROWS of the g1 drive
            if not deposit_step:
                elig_row[:Fc] = 0.0              # single deposit: cue eligibility early only
            dwell = dec_dwell if decision_step else steps_per_state
            for _ in range(dwell):
                pre_in = (rng.random((B, F)) < (in_rate * dt) * lines).astype(float)
                if deep:
                    ch_h = np.einsum('bfh,bf->bh', W1r, pre_in)
                    vh, sp_h = lif_step_batched(vh, ch_h, dt, rng, tau_m=tau_m,
                                                v_th=v_th, noise=sigma)
                    ch_o = np.einsum('bha,bh->ba', W2r, sp_h.astype(float)) + bias_o
                    vo, sp_o = lif_step_batched(vo, ch_o, dt, rng, tau_m=tau_m,
                                                v_th=v_th, noise=sigma)
                    if decision_step:
                        spk_h_dec += sp_h
                        spk_o_dec += sp_o
                    # g1: position rows driven every step; cue rows driven only at deposit
                    # (elig_row masks them off afterwards) -> a single cue->hidden deposit
                    # that then only decays. g_out is driven only at the arm decision.
                    if no_trace:
                        g1.step(np.zeros((B, F, H)))
                        g_out.step(np.zeros((B, H, A)))
                    else:
                        drive1 = ((pre_in * elig_row)[:, :, None]
                                  * np.where(sp_h, 1.0, -ltd)[:, None, :])
                        g1.step(drive1)
                        if decision_step:
                            drive_o = (sp_h.astype(float)[:, :, None]
                                       * np.where(sp_o, 1.0, -ltd)[:, None, :])
                            g_out.step(drive_o)
                        else:
                            g_out.step(np.zeros((B, H, A)))
                else:
                    ch_o = np.einsum('bfa,bf->ba', W2r, pre_in)
                    vo, sp_o = lif_step_batched(vo, ch_o, dt, rng, tau_m=tau_m,
                                                v_th=v_th, noise=sigma)
                    if decision_step:
                        spk_o_dec += sp_o
                    drive_o = pre_in[:, :, None] * np.where(sp_o, 1.0, -ltd)[:, None, :]
                    g_out.step(drive_o)

        # --- relax for the action->reward delay (the gap), then snapshot eligibility ---
        if not no_trace:
            stride = 10 if gap_lag > 200 else 1
            n_relax, rem = divmod(gap_lag, stride)
            _relax_gate(g_out, n_relax, stride, rem)
            if deep:
                _relax_gate(g1, n_relax, stride, rem)
            eo_rew = g_out.vn[..., -1] / g_out.Vnmax
            e1_rew = (g1.vn[..., -1] / g1.Vnmax) if deep else None
        else:
            eo_rew = np.zeros_like(g_out.vn[..., -1])
            e1_rew = np.zeros((B, F, H)) if deep else None

        tie = spk_o_dec.max(1) == spk_o_dec.min(1)
        chosen = np.argmax(spk_o_dec, 1)
        chosen[tie] = rng.integers(A, size=int(tie.sum()))
        R = (chosen == correct).astype(float)

        adv = (R - baseline)
        Lo = adv[:, None, None]
        if deep:
            if mode == "global":
                Lh = adv[:, None, None]
            else:
                logits = spk_o_dec - spk_o_dec.mean(1, keepdims=True)
                pol = np.exp(logits - logits.max(1, keepdims=True))
                pol /= pol.sum(1, keepdims=True)
                onehot = np.zeros((B, A)); onehot[bidx, chosen] = 1.0
                L_a = adv[:, None] * (onehot - pol)
                Lh = np.einsum('ah,ba->bh', B_fix, L_a)[:, None, :]

        W2 = np.clip(W2 + eta * Lo * eo_rew, -w_max, w_max)
        if train_w1:
            dW1 = eta_h * Lh * e1_rew
            if homeo > 0 and act_hidden is not None:
                rate = spk_h_dec / max(1, dec_dwell)
                act_hidden += (rate - act_hidden) / homeo_tau
                scale = 1.0 + homeo * (homeo_target - act_hidden)
                dW1 = dW1 + (scale[:, None, :] - 1.0) * W1
            W1 = np.clip(W1 + dW1, -w_max, w_max)

        baseline += 0.02 * (R - baseline)
        rewards[:, tr] = R

    if return_weights:
        return rewards, (W1, W2)
    return rewards


# =============================================================================
# Experiment core + full-scale driver -- the headline TEMPORAL result.
#
# Absorbed from experiments/sweeps/entry_distal_cue.py (the strengthened exp15
# shallow + exp17 deep temporal experiments).  ``run_distal_cue`` is the SERIAL,
# I/O-free core a notebook calls in quick mode; ``main()`` is the full-scale
# python -m driver that fans the (gap x arm x seed x p) grid over a Pool, bootstraps
# CIs, and writes the grid via ``paths.save_result`` (preserving the original
# ``distal_cue_<variant>.npy`` filename).
#
# THE SCIENCE (preserved verbatim from the driver's provenance). The distal-cue
# task deposits the informative cue into the eligibility trace ONCE, early, then
# never refreshes it: the agent traverses a trajectory / waits out a gap ``T_gap``
# and a single delayed reward gates whatever eligibility SURVIVED. As ``T_gap`` grows
# past the device retention ``tau_leak``, a SHORT-retention device's deposit decays
# away (chance) while a LONG-retention device still carries it (above chance). That
# crossover -- device-LONG beating device-SHORT only once the gap exceeds the short
# retention, while TIED at small gaps -- is the sequential form of the ``D_max ~
# k*tau`` law: the learnable credit horizon scales with retention, a fabrication /
# defect-engineering knob.
#
# CONTROL: the arm that must fail is a SHORT-tau DEVICE (same GateBankBatched
# physics, small tau_leak), NOT the abstract exponential trace -- a linear trace
# rescales every synapse equally and still learns. device-long vs device-short
# isolates retention as the asset.
#
# FAULTS (optional ``p`` > 0): the SiO_x programmed-conductance fault prior (stuck-at
# + realistic asymmetric D2D) is applied at read time -- maze_fault_stack for the
# non-negative shallow weights, siox_fault_stack for the signed deep weights. The
# Poole-Frenkel I-V term is excluded (a read/inference nonideality, out of scope for
# a programmed-weight rule). p=0 is the clean crossover; p>0 asks whether the
# retention asset survives device faults.
# =============================================================================

# Bootstrap resamples and the device-to-device (D2D) spreads for the SiO_x fault
# prior, plus the two retentions the crossover contrasts -- all preserved exactly
# from the original driver.
_DISTAL_N_BOOT = 10000
_DISTAL_SIGMA_G = 0.5        # D2D R_off-end log-std
_DISTAL_SIGMA_G_ON = 0.05    # D2D R_on-end log-std (realistic asymmetric spread)
_DISTAL_TAU_LONG = 20.0
_DISTAL_TAU_SHORT = 1.5

# Full-scale driver config, mutated inside ``main()`` before the fork Pool spawns
# (fork inherits it, matching the original entry point's module-global ``_CFG``).
# ``run_distal_cue`` does NOT read this -- it takes every parameter explicitly so a
# notebook can call it standalone; only the Pool worker consults it.
_DISTAL_CFG = {"trials": 2000, "variant": "deep", "window": 150}


def _distal_final(r, window):
    """Mean-over-seeds final reward rate of a ``(B, trials)`` reward array."""
    return float(np.mean(reward_rate(r, window=window)))


def _distal_make_fault(p, variant, seed):
    """SiO_x read-time programmed-conductance fault stack for a run, or ``None`` at p=0.

    ``maze_fault_stack`` for the non-negative shallow single-conductance weights,
    ``siox_fault_stack`` for the signed deep weights; the Poole-Frenkel I-V term is
    excluded (a read/inference nonideality, out of scope for a programmed-weight rule).
    """
    from .device_faults import siox_fault_stack, maze_fault_stack
    if p <= 0:
        return None
    if variant == "shallow":   # non-negative single-conductance weights
        return maze_fault_stack(p_stuck=p, sigma_g=_DISTAL_SIGMA_G, pf_on=False,
                                stuck_kind="off", w_max=W_MAX, seed=seed)
    return siox_fault_stack(p_stuck=p, sigma_g=_DISTAL_SIGMA_G,
                            sigma_g_on=_DISTAL_SIGMA_G_ON, pf_on=False,
                            stuck_kind="off", seed=seed)


def _distal_one_run(job):
    """Single (gap, p, arm, seed) run; returns ``(gap, p, arm, seed, final_reward)``.

    A module-level function (not a closure) so it is picklable under both spawn and
    fork. The job carries variant/trials/window explicitly.
    """
    # Carry the execution configuration in the job instead of relying on mutable
    # module globals.  This preserves the original numerical worker while making
    # it correct under Windows' spawn start method as well as POSIX fork.
    gap, p, arm, seed, variant, trials, window = job
    tau = _DISTAL_TAU_LONG if arm in ("long", "abstract") else _DISTAL_TAU_SHORT
    abstract = (arm == "abstract")
    no_trace = (arm == "no_trace")
    fault = _distal_make_fault(p, variant,
                               seed=1000 * seed + int(100 * p) + int(10 * gap))
    if variant == "shallow":
        r = train_distal_shallow(B=1, tau_leak=tau, T_gap=gap, trials=trials,
                                 seed0=seed, abstract=abstract, no_trace=no_trace,
                                 weight_fault=fault)
    else:
        mode = "no_trace" if no_trace else "dfa"
        # the deep variant has no abstract gate; skip abstract arm for deep (handled in main)
        r = train_distal_deep(mode=mode, B=1, H=16, tau_leak=tau, T_gap=gap,
                              trials=trials, seed0=seed, homeo=1.0,
                              weight_fault=fault)
    return (gap, p, arm, seed, _distal_final(r, window))


def _distal_batch_run(job):
    """Spawn-safe reduced-run worker that vectorises independent seeds in one batch.

    The published driver keeps one process job per seed.  For an examiner-facing
    reduced notebook, the same trainers can evolve the seed axis together (their
    native ``B`` dimension), which avoids repeating the Python time-step loop.
    """
    gap, p, arm, seeds, variant, trials, window = job
    tau = _DISTAL_TAU_LONG if arm in ("long", "abstract") else _DISTAL_TAU_SHORT
    abstract = arm == "abstract"
    no_trace = arm == "no_trace"
    fault = _distal_make_fault(p, variant, seed=int(100 * p) + int(10 * gap))
    if variant == "shallow":
        r = train_distal_shallow(B=seeds, tau_leak=tau, T_gap=gap, trials=trials,
                                 seed0=0, abstract=abstract, no_trace=no_trace,
                                 weight_fault=fault)
    else:
        mode = "no_trace" if no_trace else "dfa"
        r = train_distal_deep(mode=mode, B=seeds, H=16, tau_leak=tau, T_gap=gap,
                              trials=trials, seed0=0, homeo=1.0,
                              weight_fault=fault)
    vals = np.asarray(reward_rate(r, window=window), float).reshape(-1)
    return gap, p, arm, vals


def _distal_boot_ci(vals, seed=0):
    """Percentile bootstrap 95% CI for the mean of ``vals`` (matches the driver's own
    resample count so reproduced numbers are bit-identical)."""
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(seed)
    b = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(_DISTAL_N_BOOT)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def run_distal_cue(*, variant="deep", gaps=(1.0, 4.0, 10.0), p_grid=(0.0,), seeds=12,
                   trials=2000, window=150, workers=1, pool=None,
                   batch_seeds=False):
    """Distal-cue retention-as-asset grid; returns the result dict (no file I/O, no plots).

    Serial by default (``workers=1``). Pass an existing spawn-safe ``pool`` to distribute
    conditions without nesting pools. ``batch_seeds=True`` evolves the native seed batch
    together inside each condition, reducing Python-loop overhead for notebook runs;
    full-scale execution retains its one-job-per-seed path.

    Sweeps ``gaps x arms x p_grid x seeds`` where the arms are device-LONG
    (``tau=20 s``), device-SHORT (``tau=1.5 s``) and ``no_trace`` always, plus an
    ``abstract`` (matched-tau linear-trace) baseline for the SHALLOW variant only (the
    deep variant has no abstract eligibility gate). Reports each arm's final reward rate
    (mean + bootstrap 95% CI) per (gap, p), and the device-long minus device-short gap.

    Returns the grid dict with the SAME keys the original driver saved:
    ``{"variant", "gaps", "p", "arms", "tau_long", "tau_short", "summary", "faults",
    "seeds", "trials"}`` where ``summary[(gap, p, arm)] = (mean, lo, hi)``.
    """
    gaps = [float(x) for x in gaps]
    p_grid = [float(x) for x in p_grid]
    _DISTAL_CFG.update(trials=trials, variant=variant, window=window)

    # arms: device-long, device-short, no_trace always; abstract-long only for the
    # shallow variant (the deep variant has no abstract eligibility gate).
    arms = ["long", "short", "no_trace"] + (["abstract"] if variant == "shallow" else [])
    if batch_seeds:
        jobs = [(g, p, arm, seeds, variant, trials, window)
                for g in gaps for p in p_grid for arm in arms]
        worker = _distal_batch_run
    else:
        jobs = [(g, p, arm, s, variant, trials, window)
                for g in gaps for p in p_grid for arm in arms for s in range(seeds)]
        worker = _distal_one_run

    if pool is not None:
        results = pool.map(worker, jobs, chunksize=1)
    elif workers and workers > 1:
        import multiprocessing as mp
        try:
            ctx = mp.get_context("fork")
        except ValueError:                                   # pragma: no cover
            ctx = mp.get_context()
        with ctx.Pool(workers) as pool:
            results = pool.map(worker, jobs, chunksize=1)
    else:
        results = [worker(job) for job in jobs]

    acc = {}
    if batch_seeds:
        for g, p, arm, vals in results:
            acc.setdefault((g, p, arm), []).extend(np.asarray(vals, float).tolist())
    else:
        for g, p, arm, s, val in results:
            acc.setdefault((g, p, arm), []).append(val)

    summary = {}
    for g in gaps:
        for p in p_grid:
            for arm in arms:
                v = np.array(acc[(g, p, arm)])
                lo, hi = _distal_boot_ci(v, seed=int(10 * g) + int(100 * p))
                summary[(g, p, arm)] = (float(v.mean()), lo, hi)

    return {"variant": variant, "gaps": gaps, "p": p_grid, "arms": arms,
            "tau_long": _DISTAL_TAU_LONG, "tau_short": _DISTAL_TAU_SHORT,
            "summary": summary,
            "faults": "none" if p_grid == [0.0] else "stuck-off + D2D(0.05,0.5) [no PF]",
            "seeds": seeds, "trials": trials}


def _distal_figure(grid):
    """Optional convenience figure: crossover (device-long vs device-short vs T_gap) at
    the clean ``p``. Written under ``paths.results_dir()``; skipped silently if
    matplotlib is unavailable. Returns the saved path or ``None``."""
    from . import paths
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:                                   # plotting is optional
        print(f"[distal/{grid['variant']}] figure skipped "
              f"({type(e).__name__}: {e})", flush=True)
        return None
    gaps, arms, summary = grid["gaps"], grid["arms"], grid["summary"]
    tau_long, tau_short = grid["tau_long"], grid["tau_short"]
    p0 = grid["p"][0]
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    col = {"long": "#3aa07a", "short": "#c0392b", "no_trace": "#9aa6b2",
           "abstract": "#2f4b8f"}
    for arm in arms:
        m = [summary[(g, p0, arm)][0] for g in gaps]
        lo = [summary[(g, p0, arm)][1] for g in gaps]
        hi = [summary[(g, p0, arm)][2] for g in gaps]
        lbl = {"long": f"device tau={tau_long:g}s", "short": f"device tau={tau_short:g}s",
               "no_trace": "no trace", "abstract": f"abstract tau={tau_long:g}s"}[arm]
        ax.plot(gaps, m, "o-", color=col[arm], label=lbl, lw=1.9, ms=4)
        ax.fill_between(gaps, lo, hi, color=col[arm], alpha=0.15)
    ax.axhline(0.5, color="0.6", ls=":", lw=1.0, label="chance")
    ax.set_xlabel("cue->reward gap $T_{gap}$ (s)")
    ax.set_ylabel("final reward rate")
    ax.set_title(f"Retention is the asset ({grid['variant']}, p={p0:g})", fontsize=10)
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    out = paths.results_dir() / f"fig_distal_cue_{grid['variant']}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"[distal/{grid['variant']}] wrote {out.name}", flush=True)
    return out


def main(argv=None):
    """Full-scale reproduction CLI for the distal-cue temporal grid.

    ``python -m mrl_trace.distal_cue [--variant deep|shallow] [--gaps 1,4,10]``
    ``[-p 0] [--seeds N] [--trials N] [--workers N] [--quick|--full]``

    Fans the (gap x arm x seed x p) grid over a fork Pool (``__main__``-only), bootstraps
    95% CIs per (gap, p, arm), prints the per-row table + device-long-minus-short gap, and
    writes the grid via ``paths.save_result("distal_cue_<variant>.npy", obj)`` -- the SAME
    filename the original ``entry_distal_cue.py`` wrote. ``--full`` = the published 12-seed
    / 2000-trial run; ``--quick`` = a fast few-seed / few-trial smoke run.
    """
    import argparse
    import os
    from . import paths
    ap = argparse.ArgumentParser(description="Distal-cue retention-as-asset temporal grid")
    ap.add_argument("--variant", choices=["shallow", "deep"], default="deep")
    ap.add_argument("--gaps", default="1,4,10", help="T_gap values (s) to sweep")
    ap.add_argument("-p", "--p", default="0",
                    help="stuck-fault fractions (0 = clean crossover)")
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--workers", type=int, default=int(os.cpu_count() or 4))
    ap.add_argument("--quick", action="store_true", help="fast few-seed smoke run")
    ap.add_argument("--full", action="store_true",
                    help="published 12-seed / 2000-trial run (default)")
    a = ap.parse_args(argv)

    seeds = a.seeds if a.seeds is not None else (3 if a.quick else 12)
    trials = a.trials if a.trials is not None else (200 if a.quick else 2000)
    gaps = [float(x) for x in a.gaps.split(",")]
    p_grid = [float(x) for x in a.p.split(",")]
    window = _DISTAL_CFG["window"]

    arms = ["long", "short", "no_trace"] + (["abstract"] if a.variant == "shallow" else [])
    n_jobs = len(gaps) * len(p_grid) * len(arms) * seeds
    print(f"[distal/{a.variant}] {n_jobs} runs over {a.workers} workers; "
          f"gaps={gaps} p={p_grid} arms={arms} seeds={seeds} trials={trials}; "
          f"tau_long={_DISTAL_TAU_LONG} tau_short={_DISTAL_TAU_SHORT}; "
          f"{'clean' if p_grid == [0.0] else 'stuck+realistic-D2D (no PF)'}", flush=True)

    grid = run_distal_cue(variant=a.variant, gaps=gaps, p_grid=p_grid, seeds=seeds,
                          trials=trials, window=window, workers=a.workers)

    summary = grid["summary"]
    print(f"  {'gap':>5} {'p':>5} " + " ".join(f"{arm:>10}" for arm in arms) +
          "   long-short", flush=True)
    for g in gaps:
        for p in p_grid:
            line = f"  {g:5.1f} {p:5.2f} "
            means = {}
            for arm in arms:
                m, lo, hi = summary[(g, p, arm)]
                means[arm] = m
                line += f" {m:.2f}[{lo:.2f},{hi:.2f}]"
            line += f"   {means['long'] - means['short']:+.2f}"
            print(line, flush=True)

    out = paths.save_result(f"distal_cue_{a.variant}.npy", grid)
    print(f"[distal/{a.variant}] wrote {out.name} to {out.parent}", flush=True)

    _distal_figure(grid)


if __name__ == "__main__":
    main()
