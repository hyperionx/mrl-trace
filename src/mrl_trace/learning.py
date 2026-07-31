"""Three-factor (neo-Hebbian, reward-modulated) plasticity primitives.

The learning rule everywhere in this work is the canonical three-factor update

    Delta w_ij = eta * (R - b) * e_ij(t_R)

where ``e_ij`` is the *device* eligibility trace (see :mod:`mrl_trace.device`),
``R`` is the (delayed, global) reward, ``b`` is a slowly-tracked reward baseline, and
``eta`` is the learning rate.  Only two ingredients are local to a synapse -- its
eligibility state and the broadcast scalar ``(R - b)`` -- so the rule is
crossbar-native: no weight transport, no per-synapse gradient.

This module also provides the repository's *proposed signed, leak-dominant
coincidence* drive: a causal pre--post coincidence contributes ``+1`` and a
presynaptic event without a postsynaptic spike contributes ``-ltd``.  The cited
three-factor/R-STDP literature motivates reward-modulated eligibility, but does not
establish this exact discrete drive.
"""
from __future__ import annotations

import numpy as np

__all__ = ["signed_coincidence", "unsigned_coincidence", "coincidence_drive",
           "three_factor_update", "RewardBaseline", "LTD_BIAS",
           "SIGNED_RULE_PROVENANCE", "THREE_FACTOR_PROVENANCE"]

THREE_FACTOR_PROVENANCE = {
    "status": "established",
    "established_basis": ["three-factor reward-modulated plasticity"],
    "repository_adaptation": "A delayed scalar reward gates a local eligibility state.",
    "claim_limit": "This provenance does not establish the repository's coincidence drive.",
}

SIGNED_RULE_PROVENANCE = {
    "status": "proposed",
    "established_basis": ["three-factor plasticity", "potentiation/depression competition"],
    "repository_adaptation": (
        "pre * (+1 when post spikes, otherwise -ltd) is used as the gate input"
    ),
    "claim_limit": "This is not the exact update rule of the cited R-STDP papers.",
}

#: Default acausal (LTD-wing) weight of the signed coincidence kernel.
LTD_BIAS = 0.6


def signed_coincidence(pre, post, ltd=LTD_BIAS):
    """Repository-proposed signed leak-dominant coincidence drive.

    ``pre`` is the presynaptic activity (0/1 or float) and ``post`` whether the
    postsynaptic neuron spiked this step.  Returns ``pre * (+1 if post else -ltd)``,
    broadcasting over arrays.  This is the (signed) input handed to the device gate's
    ``step``.
    """
    pre = np.asarray(pre, dtype=float)
    sign = np.where(np.asarray(post, dtype=bool), 1.0, -ltd)
    return pre * sign


def unsigned_coincidence(pre, post, ltd=LTD_BIAS):
    """Magnitude-only ablation of the proposed signed drive.

    Positive coincidences retain magnitude one and presynaptic-only events retain
    magnitude ``ltd`` but lose their negative sign.  This is distinct from removing
    the negative term, which leaves only pre/post coincidences.
    """
    return np.abs(signed_coincidence(pre, post, ltd=ltd))


def coincidence_drive(pre, post, *, mode="signed", ltd=LTD_BIAS):
    """Select the proposed signed drive or its two explicit ablations.

    ``unsigned`` retains the magnitude of both wings but removes their sign;
    ``no_negative`` sets presynaptic-only events to zero.  They therefore attribute
    separately the roles of sign and of the depression wing itself.
    """
    if mode == "signed":
        return signed_coincidence(pre, post, ltd=ltd)
    if mode == "unsigned":
        return unsigned_coincidence(pre, post, ltd=ltd)
    if mode == "no_negative":
        return np.asarray(pre, dtype=float) * np.asarray(post, dtype=bool)
    raise ValueError(f"unknown coincidence mode {mode!r}")


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
