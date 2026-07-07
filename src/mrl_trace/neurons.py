"""Leaky integrate-and-fire (LIF) neuron updates used by the spiking decision layer.

These are deliberately minimal, current-impulse LIF steps -- the same convention
used throughout the manuscript's RL experiments: synaptic input delivered this step
is added directly to the membrane, which then leaks.  A scalar form (one output
neuron, Tier 2 distal-reward) and a batched form (``(B, A)`` action neurons over
``B`` parallel seeds, Tiers 3--6) are provided.
"""
from __future__ import annotations

import numpy as np

__all__ = ["lif_step", "lif_step_batched", "TAU_M", "V_TH"]

#: Default membrane time constant (s).
TAU_M = 20e-3
#: Default firing threshold.
V_TH = 1.0


def lif_step(v, charge, dt, tau_m=TAU_M, v_th=V_TH, v_reset=0.0):
    """Single LIF neuron, one Euler step (no membrane noise).

    ``charge`` is the instantaneous synaptic input delivered this step (sum of the
    weights of inputs that spiked).  Returns ``(v_next, spike)``.
    """
    v = v + charge - dt * v / tau_m
    spike = v >= v_th
    v = v_reset if spike else v
    return v, spike


def lif_step_batched(v, charge, dt, rng, *, tau_m=TAU_M, v_th=V_TH,
                     noise=0.15, v_reset=0.0, return_pre=False):
    """Vectorised LIF step over an array of action neurons with membrane noise.

    ``v`` and ``charge`` are arrays of identical shape (e.g. ``(B, A)``).  The
    Gaussian membrane noise provides *exploration*: with several action neurons
    receiving near-identical input early in learning, stochastic spiking lets their
    spike counts differ trial-to-trial so the reward can differentiate the actions
    and a policy can form -- the standard exploration ingredient of reward-modulated
    spiking bandits.  Returns ``(v_next, spike)`` with ``spike`` a boolean array.

    With ``return_pre=True`` also returns the membrane potential *before* the spike
    reset, ``(v_next, spike, v_pre)``.  The pre-reset membrane is what the e-prop
    pseudo-derivative is evaluated on (the surrogate gradient of the spike
    nonlinearity at the threshold crossing), so the e-prop eligibility needs it.
    """
    v_pre = v + charge - dt * v / tau_m + noise * rng.standard_normal(v.shape)
    spike = v_pre >= v_th
    v = np.where(spike, v_reset, v_pre)
    if return_pre:
        return v, spike, v_pre
    return v, spike
