"""Three-factor (neo-Hebbian, reward-modulated) plasticity primitives.

The learning rule everywhere in this work is the canonical three-factor update

    Delta w_ij = eta * (R - b) * e_ij(t_R)

where ``e_ij`` is the *device* eligibility trace (see :mod:`mrl_trace.device`),
``R`` is the (delayed, global) reward, ``b`` is a slowly-tracked reward baseline, and
``eta`` is the learning rate.  Only two ingredients are local to a synapse -- its
eligibility state and the broadcast scalar ``(R - b)`` -- so the rule is
crossbar-native: no weight transport, no per-synapse gradient.

This module also provides the *signed, leak-dominant coincidence* used to drive the
gate: a causal pre--post coincidence contributes ``+1``; an acausal one (pre without
post) contributes ``-ltd``.  The device low-pass-filters this signed drive, so
reward-uncorrelated synapses net-depress (the LTD-dominant eligibility of Izhikevich
2007 / Fremaux--Gerstner 2016) rather than drifting up.
"""
from __future__ import annotations

import numpy as np

__all__ = ["signed_coincidence", "three_factor_update", "RewardBaseline", "LTD_BIAS"]

#: Default acausal (LTD-wing) weight of the signed coincidence kernel.
LTD_BIAS = 0.6


def signed_coincidence(pre, post, ltd=LTD_BIAS):
    """Signed leak-dominant pre--post coincidence drive.

    ``pre`` is the presynaptic activity (0/1 or float) and ``post`` whether the
    postsynaptic neuron spiked this step.  Returns ``pre * (+1 if post else -ltd)``,
    broadcasting over arrays.  This is the (signed) input handed to the device gate's
    ``step``.
    """
    pre = np.asarray(pre, dtype=float)
    sign = np.where(np.asarray(post, dtype=bool), 1.0, -ltd)
    return pre * sign


def three_factor_update(w, eligibility, reward, baseline, eta, w_min=0.0, w_max=1.5):
    """Apply ``Delta w = eta (R - b) e`` and clip to ``[w_min, w_max]``.

    Works for scalar or array ``w``/``eligibility``; ``reward`` and ``baseline`` are
    scalars (or per-seed arrays broadcasting against ``w``).  The signed eligibility
    does the competition: with ``R - b > 0`` causal synapses (``e > 0``) potentiate
    and acausal ones (``e < 0``) depress.
    """
    return np.clip(w + eta * (reward - baseline) * eligibility, w_min, w_max)


class RewardBaseline:
    """Exponentially-tracked reward baseline ``b`` (the subtracted predictor in the
    reward-prediction-error / covariance form of the rule).

    ``b <- b + rate * (R - b)`` after each trial.  Initialise at chance (``1/A`` for
    an ``A``-action bandit) so the advantage ``R - b`` is centred from the start.
    """

    def __init__(self, init=0.5, rate=0.02):
        self.b = float(init)
        self.rate = rate

    def update(self, reward):
        self.b += self.rate * (reward - self.b)
        return self.b
