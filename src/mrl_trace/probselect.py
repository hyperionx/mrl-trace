"""Frank probabilistic-selection task (PST) -- a canonical RL paradigm for the device trace.

The XOR bandit of :mod:`mrl_trace.deep` is the minimal test of depth, but a toy.
This module replaces it with the probabilistic-selection task of Frank, Seeberger &
O'Reilly (2004) -- the SAME task the ds003474 EEG subjects performed -- so the device-trace
agent learns a real, canonical reinforcement-learning paradigm whose behaviour can be
compared to human data from the same dataset.

Task. Six stimuli A--F in three TRAIN pairs with fixed reinforcement probabilities
  AB: A 80% / B 20%;  CD: C 70% / D 30%;  EF: E 60% / F 40%.
Each trial shows one pair (the two stimuli in randomised left/right slots); the agent picks
a slot; the chosen slot's stimulus is rewarded stochastically at its probability. Learning
the better stimulus of each pair requires integrating reward OVER TRIALS -- the regime an
eligibility trace exists for. After training, a no-feedback TEST phase shows novel
recombinations; the two canonical Frank measures are
  choose-A : tendency to pick A (the most-positive stimulus) against C,D,E,F -- learning
             from POSITIVE feedback;
  avoid-B  : tendency to avoid B (the most-negative stimulus) against C,D,E,F -- learning
             from NEGATIVE feedback.

Each stimulus is a fixed random sparse binary code; the cue presented to the network is the
two codes placed in two position-tagged input blocks. The mapping from cue to the
better-slot action is a comparison of learned stimulus values, made non-linear by using
low-dimensional codes, so a single trained layer cannot in general solve every pair (a
``shallow`` control checks this empirically). The network, device eligibility, signed
coincidence, DFA feedback, and homeostatic stabiliser are exactly those of
:mod:`mrl_trace.deep`; only the task wrapper differs.
"""
from __future__ import annotations

import numpy as np

from .bandit import GateBankBatched, W_MAX
from .neurons import lif_step_batched, TAU_M, V_TH
from .learning import LTD_BIAS
from .deep import _relax_gate

__all__ = ["PST_PROBS", "make_codes", "train_pst", "reward_rate"]

#: Reinforcement probability of each stimulus (Frank PST).
PST_PROBS = {"A": 0.8, "B": 0.2, "C": 0.7, "D": 0.3, "E": 0.6, "F": 0.4}
#: The three training pairs.
TRAIN_PAIRS = [("A", "B"), ("C", "D"), ("E", "F")]
STIM = ["A", "B", "C", "D", "E", "F"]


def make_codes(rng, n_bits=4, n_active=2):
    """Six fixed random sparse binary codes (one per stimulus) over ``n_bits`` lines.

    Low ``n_bits`` (default 4 for six stimuli) makes the six learned values not in general
    a linear function of the code, so a hidden layer is needed; the ``shallow`` control
    verifies this empirically. Returns ``codes`` of shape ``(6, n_bits)``."""
    codes = np.zeros((6, n_bits))
    for s in range(6):
        on = rng.choice(n_bits, size=n_active, replace=False)
        codes[s, on] = 1.0
    return codes


def _present(rng, B, codes, n_bits):
    """Sample one TRAIN trial per seed. Returns the input field (B, 2*n_bits) with the two
    stimuli in position-tagged slot blocks, the per-seed (slot0_stim, slot1_stim) ids, and
    the better slot (the higher-probability stimulus' slot) for accuracy scoring."""
    pair_idx = rng.integers(len(TRAIN_PAIRS), size=B)
    swap = rng.integers(2, size=B).astype(bool)              # randomise left/right
    s0 = np.empty(B, int); s1 = np.empty(B, int)
    for b in range(B):
        a, c = TRAIN_PAIRS[pair_idx[b]]
        ia, ic = STIM.index(a), STIM.index(c)
        s0[b], s1[b] = (ic, ia) if swap[b] else (ia, ic)     # slot0, slot1 stimulus ids
    field = np.zeros((B, 2 * n_bits))
    field[:, :n_bits] = codes[s0]
    field[:, n_bits:] = codes[s1]
    probs = np.array([PST_PROBS[STIM[i]] for i in range(6)])
    better = (probs[s1] > probs[s0]).astype(int)             # better slot (0 or 1)
    return field, s0, s1, better, probs


def train_pst(*, mode="dfa_homeo", B=20, H=16, n_bits=4, tau_leak=10.0, D=5.0,
              trials=4000, dt=5e-3, cue_dur=1.0, eta=0.2, eta_hidden=3.0, in_rate=200.0,
              ltd=LTD_BIAS, tau_m=TAU_M, v_th=V_TH, V=1.5, sigma0=0.15, sigma1=0.05,
              fb_scale=2.0, w_scale1=0.6, w_scale2=0.35, w_max=W_MAX, bias_o=0.3,
              homeo=0.1, homeo_target=0.35, homeo_tau=200.0, reward_pools=None,
              seed0=0, return_test=False):
    """Train the PST policy. ``mode`` in {shallow, dfa, dfa_homeo, no_trace}; semantics as
    in :func:`mrl_trace.deep.train_deep`. Device eligibility on every plastic layer.

    Returns per-trial TRAIN correctness ``(B, trials)`` (chose the higher-probability
    stimulus). With ``return_test`` also returns ``(choose_A, avoid_B)`` arrays of shape
    ``(B,)`` from a no-feedback test phase over the novel recombinations.
    """
    if mode == "dfa_homeo":
        mode_deep, hm = "dfa", homeo
    elif mode == "no_trace":
        mode_deep, hm = "no_trace", homeo
    else:
        mode_deep, hm = mode, 0.0
    rng = np.random.default_rng(seed0)
    codes = make_codes(rng, n_bits=n_bits)
    F, A = 2 * n_bits, 2
    deep = mode_deep != "shallow"
    no_trace = mode_deep == "no_trace"
    train_w1 = deep

    def _init(shape, scale):
        return np.clip(scale * rng.standard_normal(shape), -w_max, w_max)
    if deep:
        W1 = _init((B, F, H), w_scale1)
        W2 = _init((B, H, A), w_scale2)
    else:
        W2 = _init((B, F, A), w_scale1); W1 = None

    g1 = GateBankBatched(B, F, H, tau_leak=tau_leak, dt=dt, V=V) if deep else None
    g_out = GateBankBatched(B, (H if deep else F), A, tau_leak=tau_leak, dt=dt, V=V)
    B_fix = fb_scale * rng.standard_normal((A, H)) if deep else None
    baseline = np.full(B, 0.5)
    bidx = np.arange(B)
    cue = (0.3, 0.3 + cue_dur)
    reward_lag = int(round(D / dt))
    nsteps = int(round(cue[1] / dt)) + 2
    rewards = np.zeros((B, trials))
    act_hidden = np.full((B, H), homeo_target) if (deep and hm > 0) else None

    def _run_cue(field, learn=True, rng_local=None):
        """Run one cue presentation; return chosen slot (B,) and (for learning) cached
        spike/eligibility state. If learn, integrates eligibility and is used in the loop."""
        r = rng if rng_local is None else rng_local
        vh = np.zeros((B, H)); vo = np.zeros((B, A))
        spk_o = np.zeros((B, A)); spk_h = np.zeros((B, H))
        if deep and learn:
            g1.reset()
        if learn:
            g_out.reset()
        for n in range(nsteps):
            on = cue[0] <= n * dt < cue[1]
            pre_in = ((r.random((B, F)) < (in_rate * dt) * field).astype(float)
                      if on else np.zeros((B, F)))
            if deep:
                ch_h = np.einsum('bfh,bf->bh', W1, pre_in)
                vh, sp_h = lif_step_batched(vh, ch_h, dt, r, tau_m=tau_m, v_th=v_th, noise=sigma)
                spk_h += sp_h
                ch_o = np.einsum('bha,bh->ba', W2, sp_h.astype(float))
                if on:
                    ch_o = ch_o + bias_o
                vo, sp_o = lif_step_batched(vo, ch_o, dt, r, tau_m=tau_m, v_th=v_th, noise=sigma)
                spk_o += sp_o
                if learn:
                    if no_trace:
                        g1.step(np.zeros((B, F, H))); g_out.step(np.zeros((B, H, A)))
                    else:
                        g1.step(pre_in[:, :, None] * np.where(sp_h, 1.0, -ltd)[:, None, :])
                        g_out.step(sp_h.astype(float)[:, :, None]
                                   * np.where(sp_o, 1.0, -ltd)[:, None, :])
            else:
                ch_o = np.einsum('bfa,bf->ba', W2, pre_in)
                if on:
                    ch_o = ch_o + bias_o
                vo, sp_o = lif_step_batched(vo, ch_o, dt, r, tau_m=tau_m, v_th=v_th, noise=sigma)
                spk_o += sp_o
                if learn:
                    g_out.step(pre_in[:, :, None] * np.where(sp_o, 1.0, -ltd)[:, None, :])
        tie = spk_o.max(1) == spk_o.min(1)
        chosen = np.argmax(spk_o, 1)
        chosen[tie] = r.integers(A, size=int(tie.sum()))
        return chosen, spk_h, spk_o

    for tr in range(trials):
        sigma = sigma0 + (sigma1 - sigma0) * tr / trials
        field, s0, s1, better, probs = _present(rng, B, codes, n_bits)
        chosen, spk_h, spk_o = _run_cue(field, learn=True)
        # stochastic reward at the chosen stimulus' reinforcement probability
        chosen_stim = np.where(chosen == 0, s0, s1)
        p_chosen = probs[chosen_stim]
        R_true = (chosen == better).astype(float)                # chose higher-prob stim
        outcome = (rng.random(B) < p_chosen).astype(int)         # actual stochastic reward
        if reward_pools is None:
            R = outcome.astype(float)
        else:
            R = np.array([reward_pools[int(outcome[b])][
                rng.integers(len(reward_pools[int(outcome[b])]))] for b in range(B)])

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

        adv = (R - baseline)
        Lo = adv[:, None, None]
        if deep:
            logits = spk_o - spk_o.mean(1, keepdims=True)
            pol = np.exp(logits - logits.max(1, keepdims=True)); pol /= pol.sum(1, keepdims=True)
            onehot = np.zeros((B, A)); onehot[bidx, chosen] = 1.0
            L_a = adv[:, None] * (onehot - pol)
            Lh = np.einsum('ah,ba->bh', B_fix, L_a)[:, None, :]
        W2 = np.clip(W2 + eta * Lo * eo_rew, -w_max, w_max)
        if train_w1:
            dW1 = eta_hidden * Lh * e1_rew
            if hm > 0 and act_hidden is not None:
                rate = spk_h / nsteps
                act_hidden += (rate - act_hidden) / homeo_tau
                scale = 1.0 + hm * (homeo_target - act_hidden)
                dW1 = dW1 + (scale[:, None, :] - 1.0) * W1
            W1 = np.clip(W1 + dW1, -w_max, w_max)
        baseline += 0.02 * (R - baseline)
        rewards[:, tr] = R_true

    if not return_test:
        return rewards

    # --- TEST phase: novel recombinations, no learning, greedy (low-noise) choice ---
    rng_t = np.random.default_rng(seed0 + 99991)
    saved = sigma
    def _choice_prob(slot0_stim, slot1_stim):
        """For each seed, fraction of repeats it picks slot 1 (the second listed)."""
        field = np.zeros((B, F))
        field[:, :n_bits] = codes[slot0_stim]; field[:, n_bits:] = codes[slot1_stim]
        picks = np.zeros(B)
        reps = 21
        nonlocal sigma
        sigma = 0.05                                          # near-greedy at test
        for _ in range(reps):
            chosen, _, _ = _run_cue(field, learn=False, rng_local=rng_t)
            picks += chosen
        sigma = saved
        return picks / reps                                   # P(choose slot1) per seed
    iA, iB = STIM.index("A"), STIM.index("B")
    others = [STIM.index(s) for s in ("C", "D", "E", "F")]
    # choose-A: A vs each other; pick rate of A (A placed in slot0 -> P(choose A)=1-P(slot1))
    cA = np.zeros(B)
    for o in others:
        cA += 1.0 - _choice_prob(np.full(B, iA), np.full(B, o))
    cA /= len(others)
    # avoid-B: B vs each other; avoid rate of B (B in slot0 -> avoid = P(choose slot1))
    aB = np.zeros(B)
    for o in others:
        aB += _choice_prob(np.full(B, iB), np.full(B, o))
    aB /= len(others)
    return rewards, cA, aB


def reward_rate(rewards, window=200):
    rewards = np.asarray(rewards)
    if rewards.shape[-1] < window:
        return rewards.mean(axis=-1)
    return rewards[..., -window:].mean(axis=-1)


# =============================================================================
# Experiment core (Experiment 10, Arm F -- the probabilistic-selection study)
#
# ``run_probselect`` returns one condition's result as a plain dict -- no file I/O,
# no plotting, no stdout -- so a notebook can call it in-kernel at a small (quick)
# trial count.  ``main()`` (below) runs the published 20-seed grid across a process
# Pool over conditions, computes bootstrap CIs + the pre-registered F1--F5 criteria,
# and writes ``exp10_probselect.npy`` under ``data/results/`` for full-cache mode.
# Provenance: experiments/05_biological_grounding/probabilistic_selection.py
# (Arm F), pre-registered criteria F1--F5 in PREREGISTRATION_probselect.md.
# =============================================================================

#: Fixed hyper-parameters of the published Arm-F grid (H, code width, retention,
#: reward lag, learning rates, DFA feedback scale, output bias) -- the shape shared
#: by every condition so only mode/reward-source varies.
PST_HP = dict(H=16, n_bits=12, tau_leak=10.0, D=5.0, eta=0.2, eta_hidden=3.0,
              fb_scale=2.0, bias_o=0.3)
#: Homeostatic-stabiliser strength used by the ``dfa_homeo`` family.
PST_HOMEO = 0.1
#: Published scale + task references.
PST_SEEDS = 20
PST_TRIALS = 5000
PST_CHANCE = 0.5
PST_CRIT = 0.75


def run_probselect(cond, *, trials=PST_TRIALS, seeds=PST_SEEDS, pools=None, shuf=None,
                   hp=None, homeo=PST_HOMEO, seed0=0):
    """One independent Arm-F cell: train the PST for a single condition ``cond`` on a
    vectorised ``seeds``-seed batch and return its behavioural summary as a plain dict.

    ``cond`` selects mode + reward source (semantics as in the Arm-F driver):
      ``shallow``          one trained device-synapse layer (the linearly-separable
                           control; solves the task -- headline Frank signature);
      ``dfa``              deep, DFA feedback, no homeostasis;
      ``dfa_homeo``        deep, DFA feedback + homeostatic stabiliser (the device agent);
      ``no_trace``         eligibility zeroed (device-necessity control);
      ``dfa_homeo_eeg``    ``dfa_homeo`` gated by the subjects' OWN decoded real-EEG
                           reward (requires ``pools``; the human non-invasive RPE);
      ``dfa_homeo_shuf``   ``dfa_homeo`` gated by the shuffled-reward control
                           (requires ``shuf``).

    The test phase (choose-A / avoid-B) is run only for the conditions the Frank
    figure reports (``shallow``, ``dfa``, ``dfa_homeo``).  Returns
    ``dict(cond, finals (seeds,), curve (trials-200,), chooseA, avoidB)`` where
    ``finals`` is the last-300-trial train accuracy per seed, ``curve`` is the
    seed-mean 200-trial running P(better stimulus), and ``chooseA``/``avoidB`` are the
    test-phase Frank measures per seed (``None`` for the non-test conditions).
    """
    hp = PST_HP if hp is None else hp
    rp = None
    mode = cond
    if cond == "dfa_homeo_eeg":
        mode, rp = "dfa_homeo", {0: pools[0], 1: pools[1]}
    elif cond == "dfa_homeo_shuf":
        mode, rp = "dfa_homeo", shuf
    want_test = cond in ("dfa_homeo", "shallow", "dfa")
    out = train_pst(mode=mode, B=seeds, trials=trials, seed0=seed0, homeo=homeo,
                    reward_pools=rp, return_test=want_test, **hp)
    if want_test:
        rew, cA, aB = out
    else:
        rew, cA, aB = out, None, None
    finals = reward_rate(rew, window=300)
    win = 200
    csum = np.cumsum(rew, axis=1)
    curve = ((csum[:, win:] - csum[:, :-win]) / win).mean(0)
    return {"cond": cond, "finals": finals, "curve": curve, "chooseA": cA, "avoidB": aB}


def _load_eeg_pools(cache="/tmp/eeg_pools.npy", seed=0):
    """Load the cached EEG out-of-fold reward pools (built by the EEG-capstone driver
    from ds003474) and derive the shuffled-reward control, degrading gracefully.

    Returns ``(pools, shuf, meta)``; ``(None, None, {})`` when the cache is absent (the
    subjects' raw EEG is not bundled), so the EEG conditions are skipped exactly as the
    original Arm-F driver does when ``pools`` is ``None``."""
    import os
    if not os.path.exists(cache):
        return None, None, {}
    pools = np.load(cache, allow_pickle=True).item()
    meta = pools.get("_meta", {})
    rng = np.random.default_rng(seed)
    allv = np.concatenate([pools[1], pools[0]])
    shuf = {1: rng.permutation(allv), 0: rng.permutation(allv)}
    return pools, shuf, meta


def _summarize_probselect(res_by_cond, conds, *, seeds, trials, meta, hp,
                          chance=PST_CHANCE, crit=PST_CRIT):
    """Package the per-condition :func:`run_probselect` output into the saved grid dict
    (the exact schema the Arm-F driver wrote to ``exp10_probselect.npy``) plus the
    pre-registered criteria F1--F5.

    Criteria (no goalpost moving; from PREREGISTRATION_probselect.md):
      F1  device learns the train phase             (dfa_homeo mean >= ``crit``);
      F2  depth is needed                            (homeo CI lower > shallow CI upper);
      F3  eligibility is necessary                   (no-trace mean <= 0.60);
      F4  Frank asymmetry present                    (choose-A and avoid-B CI lower > 0.5);
      F5  the real-EEG loop beats its shuffled control (EEG CI lower > shuf CI upper) --
          only evaluated when the EEG pools are available.
    """
    from .stats import bootstrap_ci
    finals = {c: res_by_cond[c]["finals"] for c in conds}
    curves = {c: res_by_cond[c]["curve"] for c in conds}
    tests = {c: (res_by_cond[c]["chooseA"], res_by_cond[c]["avoidB"])
             for c in conds if res_by_cond[c]["chooseA"] is not None}
    ci = {c: bootstrap_ci(finals[c]) for c in finals}

    dh = finals["dfa_homeo"]
    cA, aB = tests["dfa_homeo"]
    criteria = {
        "F1": bool(dh.mean() >= crit),
        "F2": bool(ci["dfa_homeo"][0] > ci["shallow"][1]),
        "F3": bool(finals["no_trace"].mean() <= 0.60),
        "F4": bool(bootstrap_ci(cA)[0] > 0.5 and bootstrap_ci(aB)[0] > 0.5),
    }
    if "dfa_homeo_eeg" in finals and "dfa_homeo_shuf" in finals:
        criteria["F5"] = bool(ci["dfa_homeo_eeg"][0] > ci["dfa_homeo_shuf"][1])

    return {
        "finals": finals, "curves": curves,
        "tests": {k: {"chooseA": v[0], "avoidB": v[1]} for k, v in tests.items()},
        "ci": {k: ci[k] for k in ci}, "meta": meta, "HP": hp,
        "seeds": seeds, "trials": trials, "chance": chance, "crit": crit,
        "criteria": criteria,
    }


def main(argv=None):
    """Full-scale reproduction CLI for the probabilistic-selection grid (Experiment 10,
    Arm F): the Frank PST -- the SAME task the ds003474 EEG subjects performed --
    learned by the deep all-local device-trace agent, lifting the RL result off the XOR toy.

    ``python -m mrl_trace.probselect [--probselect] [--full|--quick]``
    ``--full`` = 20 seeds x 5000 trials (published); ``--quick`` = 6 seeds x 1000 trials.
    Each condition (mode/reward source) is an independent cell run across a process Pool.
    The synthetic conditions always run; the two real-EEG conditions run only when the
    cached EEG reward pools (``/tmp/eeg_pools.npy``, built by the EEG-capstone driver from
    ds003474) are present -- absent, they are skipped with a printed note, exactly as the
    original Arm-F driver degrades.  Writes ``exp10_probselect.npy`` under ``data/results``.
    """
    import argparse
    import os
    from functools import partial
    from multiprocessing import Pool
    from . import paths
    from .stats import bootstrap_ci

    ap = argparse.ArgumentParser(description="Frank probabilistic-selection RL reproduction")
    ap.add_argument("--probselect", action="store_true",
                    help="run the Arm-F grid -> exp10_probselect.npy (default)")
    ap.add_argument("--quick", action="store_true", help="fast few-seed smoke run")
    ap.add_argument("--full", action="store_true", help="published 20-seed run (default)")
    a = ap.parse_args(argv)

    seeds = 6 if a.quick else PST_SEEDS
    trials = 1000 if a.quick else PST_TRIALS

    pools, shuf, meta = _load_eeg_pools()
    conds = ["shallow", "dfa", "dfa_homeo", "no_trace"]
    if pools is not None:
        conds += ["dfa_homeo_eeg", "dfa_homeo_shuf"]
    else:
        print("  (no EEG reward pools at /tmp/eeg_pools.npy -- skipping the real-EEG "
              "conditions; run the EEG-capstone driver first to include them)")

    print(f"Probabilistic-selection task | {seeds} seeds, {trials} trials, "
          f"H={PST_HP['H']}, n_bits={PST_HP['n_bits']}")

    worker = partial(run_probselect, trials=trials, seeds=seeds, pools=pools, shuf=shuf)
    with Pool(processes=min(len(conds), (os.cpu_count() or 4) - 1)) as pool:
        res = pool.map(worker, conds)
    res_by_cond = {r["cond"]: r for r in res}

    grid = _summarize_probselect(res_by_cond, conds, seeds=seeds, trials=trials,
                                 meta=meta, hp=PST_HP)

    print("\n=== final train accuracy (chose higher-prob stimulus, last 300 trials) ===")
    for c in conds:
        lo, hi = grid["ci"][c]
        print(f"  {c:22s} {grid['finals'][c].mean():.3f} [{lo:.3f}, {hi:.3f}]")

    print("\n=== test-phase Frank signature (choose-A / avoid-B) ===")
    for c, t in grid["tests"].items():
        cA, aB = t["chooseA"], t["avoidB"]
        clo, chi = bootstrap_ci(cA); alo, ahi = bootstrap_ci(aB)
        print(f"  {c:22s} choose-A {cA.mean():.3f} [{clo:.2f},{chi:.2f}]  "
              f"avoid-B {aB.mean():.3f} [{alo:.2f},{ahi:.2f}]")

    print(f"\n=== pre-registered criteria ===  {grid['criteria']}")
    paths.save_result("exp10_probselect.npy", grid)
    print("  wrote exp10_probselect.npy")


if __name__ == "__main__":
    main()
