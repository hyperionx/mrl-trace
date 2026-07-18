r"""Hybrid architecture -- a BPTT perception front-end + the device-eligibility decision stack.

This module answers the *high-dimensional input* objection to the device eligibility
trace ("real RL has rich perceptual input"), honestly and by DIVISION OF LABOUR: the
slow device trace is NOT a perception engine, so a conventional (gradient-trained,
frozen) perception stage supplies a low-dimensional state, and the all-local device
learning stack learns the policy behind it. The claim is *composition*, never that the
device does perception.

Two complementary studies live here, sharing the same perception front-end:

  * **exp5** (:func:`run_hybrid_decision`): a conv-SNN perception
    stage (snntorch ``Leaky`` + surrogate gradient, rate-coded input) classifies noisy
    oriented gratings and is frozen; a contextual spiking bandit then reads either its
    ``C``-dim frozen readout (*hybrid*), the ``P`` raw pixels (*raw*, no front-end), or
    the hybrid wiring with the eligibility zeroed (*no_trace*). Plastic device synapses
    carry the :class:`~mrl_trace.bandit.GateBankBatched` eligibility and are
    updated by the signed three-factor rule at EQUAL RL budget. Retrospective criteria: C1
    front-end acc >= 0.85; C2 hybrid >= criterion and no-trace at chance; C3 hybrid -
    raw >= 0.15. Reference (MPS): acc 0.87, hybrid 0.62, raw 0.27, no-trace 0.27 -- the
    front-end is the remedy. Writes ``tier6_results.npy``.

  * **exp16** (:func:`run_hybrid_scale`, Tier 3 batch sweep): a WIDTH-SCALING +
    FAULT-TOLERANCE robustness study. STAGE 1 (perception, once, offline, optional GPU)
    trains the SAME conv-SNN and caches its low-dimensional readouts to ``readouts.npz``
    (:func:`prep_readouts`). STAGE 2 (decision, pure NumPy) feeds those cached readouts
    as the STATE of the DEEP all-local agent (:func:`~mrl_trace.deep.train_deep`:
    trace + DFA + homeostasis) via its ``state_sampler`` hook, and sweeps hidden width
    ``H`` x SiO_x stuck-fault fraction ``p`` under the measured device-fault prior. If
    ``readouts.npz`` is ABSENT the sweep falls back to an HONEST proxy front-end (clearly
    logged): class-conditional noisy embeddings whose clusters overlap and whose label is
    a non-linearly-separable XOR of two latent bits, so depth is genuinely required. This
    is a stand-in for the readout *statistics*, not a silent fake of perception -- the
    headline result should use the real cached readouts. Writes ``exp16_hybrid_scale.npy``.

SCOPE (read before citing exp16). exp16 is scoped to width-scaling + fault-tolerance
ONLY: does the all-local device stack, behind a real frozen perception front-end, keep
learning the policy as the hidden layer widens and the stuck-fault fraction rises? It is
NOT a homeostasis experiment and carries NO homeostasis ablation. Homeostasis is
hidden-CAPACITY-dependent (it opposes a winner-take-all hidden-unit collapse), so at this
study's wide hidden layers (H up to 512, overcomplete) it is irrelevant at any strength --
a verified regime search found hybrid-full ~ no-homeo here (gap ~0). The homeostasis
CONTRIBUTION is large and CI-clean only at MODERATE width (H=32) under temporal
interference + device faults, which is exp13's job (gap +0.37 to +0.49, disjoint CIs);
citing exp16's homeostasis-neutrality as evidence about homeostasis would be a category
error, which is why the no-homeo arm is omitted rather than reported as a null.

ACCURACY SCOPE (exp16). On the real A=4 readouts the device policy CONVERGES to ~0.51
reward (flat from 2000 to 4000 trials), well above chance (0.25) and far above the
no-trace control (~0.27), but BELOW the frozen front-end's own classification ceiling
(~0.87). The claim is that the all-local stack learns a SUBSTANTIAL policy behind a real
perception front-end and that this DEGRADES GRACEFULLY with stuck faults across width --
NOT that it matches the front-end's accuracy.

The perception front-end (torch + snntorch) is the OPTIONAL ``hybrid`` extra. It is
imported LAZILY inside the functions that need it, so this module -- and the exp16
Stage-2 sweep on cached/proxy readouts -- imports and runs with NO torch installed. Only
:func:`run_frontend`, :func:`run_hybrid_decision` and :func:`prep_readouts` require torch;
:func:`run_hybrid_scale` never does (it consumes the cached ``.npz`` or the proxy).
Install the extra with ``pip install -e ".[hybrid]"``.
"""
from __future__ import annotations

import numpy as np

from .bandit import GateBankBatched, W_INIT, W_MAX
from .deep import train_deep
from .device_faults import siox_fault_stack
from .stats import bootstrap_ci

__all__ = [
    # exp5 perception front-end + hybrid decision (Fig 9)
    "IMG", "C", "NOISE",
    "make_grating", "make_batch",
    "run_frontend", "run_readout_pool", "run_hybrid_decision",
    # exp16 stage-1 caching + stage-2 sweep
    "prep_readouts", "run_hybrid_scale",
    # analysis helpers
    "final_rate",
]

# =============================================================================
# exp5 -- perception task geometry (noisy oriented gratings) and front-end
# =============================================================================
IMG = 16          # image side (P = 256 pixels)
C = 4             # orientation classes (0, 45, 90, 135 deg)
NOISE = 0.9       # additive noise std (high -> raw-pixel RL is genuinely hard)
_N_SEEDS = 20     # exp5 decision-layer seeds (comp-neuro-grade; batched, so ~free)


def _torch():
    """Lazily import the optional ``hybrid`` extra (torch + snntorch), raising the same
    actionable message as the original driver if it is not installed. Keeping the import
    inside the functions that need it lets this module -- and the pure-NumPy exp16 Stage-2
    sweep -- load and run with no torch present."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        import snntorch as snn
        from snntorch import surrogate
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "The hybrid perception front-end needs torch + snntorch. Install the "
            'optional extra:\n    pip install -e ".[hybrid]"'
        ) from exc
    dev = torch.device("cuda" if torch.cuda.is_available()
                       else "mps" if torch.backends.mps.is_available() else "cpu")
    return torch, nn, F, snn, surrogate, dev


def make_grating(theta, rng, img=IMG):
    """A single noisy oriented sinusoidal grating at orientation ``theta`` (radians)."""
    x = np.linspace(-1, 1, img)
    xx, yy = np.meshgrid(x, x)
    proj = xx * np.cos(theta) + yy * np.sin(theta)
    g = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * proj)
    g = g + NOISE * rng.standard_normal((img, img))
    g = (g - g.min()) / (g.max() - g.min() + 1e-9)
    return g.astype(np.float32)


def make_batch(n, rng):
    """A batch of ``n`` gratings with class labels (one of ``C`` orientations each)."""
    thetas = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    y = rng.integers(C, size=n)
    X = np.stack([make_grating(thetas[c], rng) for c in y])
    return X[:, None, :, :], y


def _build_convsnn(num_steps=20, beta=0.9):
    """Construct the compact convolutional spiking net (two conv+LIF stages + a readout
    LIF), trained by BPTT with a surrogate gradient on rate-coded input. Defined inside a
    factory so the ``nn.Module`` subclass is created only when torch is present (lazy)."""
    _torch_, nn, F, snn, surrogate, _dev = _torch()

    class ConvSNN(nn.Module):
        def __init__(self, num_steps=num_steps, beta=beta):
            super().__init__()
            sg = surrogate.fast_sigmoid()
            self.num_steps = num_steps
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.lif1 = snn.Leaky(beta=beta, spike_grad=sg)
            self.lif2 = snn.Leaky(beta=beta, spike_grad=sg)
            self.lif3 = snn.Leaky(beta=beta, spike_grad=sg)
            h = IMG // 2 // 2
            self.fc = nn.Linear(16 * h * h, C)

        def forward(self, x):
            m1 = self.lif1.init_leaky()
            m2 = self.lif2.init_leaky()
            m3 = self.lif3.init_leaky()
            memsum = 0.0
            spksum = 0.0
            for _ in range(self.num_steps):
                xt = (_torch_.rand_like(x) < x).float()
                c1 = F.max_pool2d(self.conv1(xt), 2); s1, m1 = self.lif1(c1, m1)
                c2 = F.max_pool2d(self.conv2(s1), 2); s2, m2 = self.lif2(c2, m2)
                c3 = self.fc(s2.flatten(1)); s3, m3 = self.lif3(c3, m3)
                memsum = memsum + m3
                spksum = spksum + s3
            return memsum / self.num_steps, spksum

    return ConvSNN()


def run_frontend(steps=600, bs=128, lr=2e-3, seed=0):
    """Train the perception front-end (conv-SNN, BPTT surrogate gradient) and freeze it.

    Returns ``(net, acc)`` where ``net`` is the frozen ``torch`` module and ``acc`` is its
    held-out classification accuracy on 2000 fresh gratings. Needs the ``hybrid`` extra.
    """
    torch, nn, F, snn, surrogate, dev = _torch()
    torch.manual_seed(seed)
    if getattr(dev, "type", str(dev)) == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    rng = np.random.default_rng(seed)
    net = _build_convsnn().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    net.train()
    for _ in range(steps):
        X, y = make_batch(bs, rng)
        mem, _ = net(torch.tensor(X, device=dev))
        loss = lossf(mem, torch.tensor(y, device=dev))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        Xv, yv = make_batch(2000, rng)
        mv, _ = net(torch.tensor(Xv, device=dev))
        acc = (mv.argmax(1).cpu().numpy() == yv).mean()
    return net, float(acc)


def run_readout_pool(net, per_class=500, seed=1):
    """Extract the frozen front-end's per-class spike-rate readouts + raw-pixel pools.

    Returns ``(feats, raws)`` -- dicts ``class -> (per_class, C)`` normalised readouts and
    ``class -> (per_class, P)`` raw pixels respectively. Needs the ``hybrid`` extra.
    """
    torch, nn, F, snn, surrogate, dev = _torch()
    rng = np.random.default_rng(seed)
    net.eval()
    feats, raws = {}, {}
    thetas = np.array([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4])
    with torch.no_grad():
        for c in range(C):
            X = np.stack([make_grating(thetas[c], rng) for _ in range(per_class)])
            _, spk = net(torch.tensor(X[:, None], device=dev))
            out = np.clip(spk.cpu().numpy(), 0, None)
            out = out / (out.max(1, keepdims=True) + 1e-9)
            feats[c] = out
            raws[c] = X.reshape(per_class, -1)
    return feats, raws


def _decision(state_pool, S, A, B=4, trials=1500, dt=5e-3, cue_dur=1.0, D=2.0,
              eta=0.2, in_rate=200.0, ltd=0.6, tau_m=20e-3, v_th=1.0, noise=0.15,
              tau_leak=10.0, no_trace=False, seed0=0):
    """Contextual bandit over ``C`` classes; ``state_pool`` is a dict ``class -> (n, S)``
    activations (``S=C`` for the hybrid readout, ``S=P`` for raw pixels). Uses the
    package's :class:`~mrl_trace.bandit.GateBankBatched` for the device eligibility
    and returns rewards ``(B, trials)``. Pure NumPy; no torch. This is the exp5 decision
    layer -- its inline current-impulse LIF matches ``neurons.lif_step_batched`` exactly
    (v_reset=0), kept inline here to preserve the RNG call ordering bit-for-bit."""
    rng = np.random.default_rng(seed0)
    bank = GateBankBatched(B, S, A, tau_leak=tau_leak, dt=dt)
    w = np.full((B, S, A), W_INIT)
    correct = np.arange(C) % A
    baseline = np.full(B, 1.0 / A)
    cue = (0.3, 0.3 + cue_dur)
    ri = int((cue[1] + D) / dt)
    nsteps = ri + 2
    rewards = np.zeros((B, trials))
    pools = {c: np.asarray(state_pool[c]) for c in range(C)}
    for tr in range(trials):
        bank.reset()
        cls = rng.integers(C, size=B)
        act = np.zeros((B, S))
        for b in range(B):
            p = pools[cls[b]]
            act[b] = p[rng.integers(len(p))]
        v = np.zeros((B, A)); spk = np.zeros((B, A)); e_rew = np.zeros((B, S, A))
        for n in range(nsteps):
            t = n * dt
            pre = np.zeros((B, S))
            if cue[0] <= t < cue[1]:
                pre = (rng.random((B, S)) < in_rate * dt * act).astype(float)
            charge = np.einsum('bsa,bs->ba', w, pre)
            v = v + charge - dt * v / tau_m + noise * rng.standard_normal((B, A))
            sp = v >= v_th; v = np.where(sp, 0.0, v); spk += sp
            drive = pre[:, :, None] * np.where(sp, 1.0, -ltd)[:, None, :]
            if no_trace:
                drive[:] = 0.0
            e = bank.step(drive)
            if n == ri:
                e_rew = e.copy()
        tie = spk.max(1) == spk.min(1)
        chosen = np.argmax(spk, 1); chosen[tie] = rng.integers(A, size=tie.sum())
        r = (chosen == correct[cls]).astype(float)
        w = np.clip(w + (eta * (r - baseline))[:, None, None] * e_rew, 0.0, W_MAX)
        baseline += 0.02 * (r - baseline)
        rewards[:, tr] = r
    return rewards


def final_rate(rw, window=100):
    """Per-seed reward rate over the last ``window`` trials (axis -1)."""
    rw = np.asarray(rw)
    return rw[:, -window:].mean(1)


def _hybrid_decision_cell(job):
    """One spawn-safe decision condition after the frozen front end is extracted."""
    name, pool, S, A, seeds, trials, D, no_trace = job
    vals = final_rate(_decision(pool, S=S, A=A, B=seeds, trials=trials, D=D,
                                no_trace=no_trace))
    return name, vals


def run_hybrid_decision(*, seeds=_N_SEEDS, front_steps=600, front_seed=0,
                        per_class=500, trials=1500, D=2.0, workers=1):
    """exp5 (Fig 9): train the perception front-end, then run the decision-layer RL at
    ``seeds`` seeds in three conditions -- *hybrid* (reads the C-dim readout), *raw*
    (reads P pixels, no front-end) and *no_trace* (hybrid wiring, eligibility zeroed) --
    at equal RL budget.

    Retrospectively recorded: C1 front-end acc >= 0.85; C2 hybrid >= criterion AND
    no-trace at chance (<= chance + 0.10); C3 the front-end is the remedy (hybrid - raw
    >= 0.15). ``criterion = 0.5*(1 + 1/A)``, ``A = C``. Reference (MPS): acc 0.87, hybrid
    0.62, raw 0.27, no-trace 0.27.

    Returns the result grid as a plain dict (no file I/O, no plotting, no stdout); needs
    the ``hybrid`` extra for the front-end. ``main()`` writes it to ``tier6_results.npy``.
    """
    A = C
    chance = 1.0 / A
    crit = 0.5 * (1 + chance)
    net, acc = run_frontend(steps=front_steps, seed=front_seed)
    feats, raws = run_readout_pool(net, per_class=per_class, seed=front_seed + 1)

    jobs = [
        ("hybrid", feats, C, A, seeds, trials, D, False),
        ("raw", raws, IMG * IMG, A, seeds, trials, D, False),
        ("no_trace", feats, C, A, seeds, trials, D, True),
    ]
    if int(workers) > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(min(int(workers), len(jobs))) as pool:
            per_seed = dict(pool.map(_hybrid_decision_cell, jobs, chunksize=1))
    else:
        per_seed = dict(map(_hybrid_decision_cell, jobs))
    res = {}
    for k, v in per_seed.items():
        lo, hi = bootstrap_ci(v)
        res[k] = (float(v.mean()), float(v.std()), lo, hi)

    hyb, raw, nt = res["hybrid"][0], res["raw"][0], res["no_trace"][0]
    c2 = (hyb >= crit) and (nt <= chance + 0.10)
    c3 = (hyb >= crit) and (hyb - raw >= 0.15)
    return dict(front_acc=acc, chance=chance, crit=crit, results=res,
                C1=bool(acc >= 0.85), C2=bool(c2), C3=bool(c3),
                C=C, A=A, P=IMG * IMG, noise=NOISE, n_seeds=seeds)


# =============================================================================
# exp16 Stage 1 -- cache the frozen front-end readouts (needs the hybrid extra)
# =============================================================================
def prep_readouts(per_class=800, steps=600, seed=0, out_name="readouts.npz"):
    """exp16 Stage 1: train the SAME conv-SNN front-end ONCE and cache its readouts to
    ``data/results/<out_name>`` in the schema Stage 2 expects::

        readouts.npz : X (N, F) float32 readout vectors, y (N,) int class labels, acc () float

    where ``N = per_class * C``. Rows are shuffled so a sequential sampler sees mixed
    classes. This is the offline perception step that upgrades the exp16 sweep from
    "proxy statistics" to the real frozen-front-end result; it is NOT part of the pure-NumPy
    Stage-2 batch job. Needs the ``hybrid`` extra (torch + snntorch); returns the written
    path. Writes through :mod:`mrl_trace.paths` (never to ``experiments/``).
    """
    from . import paths
    net, acc = run_frontend(steps=steps, seed=seed)
    feats, _raws = run_readout_pool(net, per_class=per_class, seed=seed + 1)
    Xs, ys = [], []
    for c in sorted(feats):
        Xs.append(np.asarray(feats[c], dtype=np.float32))
        ys.append(np.full(len(feats[c]), c, dtype=np.int64))
    X = np.concatenate(Xs, 0)
    y = np.concatenate(ys, 0)
    rng = np.random.default_rng(seed + 2)
    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]
    out = paths.results_dir() / out_name
    np.savez_compressed(out, X=X, y=y, acc=np.float32(acc))
    return out


# =============================================================================
# exp16 Stage 2 -- pure-NumPy width x fault sweep behind the readouts (or proxy)
# =============================================================================
N_BOOT = 10000
# Realistic ASYMMETRIC lognormal D2D (R_on_log_std=0.05, R_off_log_std=0.5), matching the
# awarememristor reference's realistic SiO_x device; symmetric (0.5,0.5) is its "high-D2D"
# stress regime. With the I-V nonlinearity now correctly active, symmetric 0.5 drives the net
# to chance while the realistic spread stays robust -- so the realistic spread is the headline.
SIGMA_G = 0.5            # R_off-end log-std
SIGMA_G_ON = 0.05        # R_on-end log-std
# Early-stop template; the chance/criterion are set per A in run_hybrid_scale (chance = 1/A,
# converged criterion scaled above chance) so multi-way readout tasks (A>2) are not falsely
# declared "stuck" at a still-improving rate.
EARLY_STOP = {"min_trials": 400, "check_every": 100, "window": 150,
              "criterion": 0.78, "tol": 0.03, "chance": 0.5, "stuck_after": 1200}
# Proxy front-end geometry (used only when no real readouts.npz is supplied).
PROXY_F = 8                    # readout dimensionality (front-end embedding width)
PROXY_NOISE = 0.35             # per-dim Gaussian noise -> overlapping class clusters

# exp16 is a WIDTH + FAULT-TOLERANCE robustness study (the all-local stack behind a real frozen
# front-end), NOT a homeostasis experiment. The no_homeo ablation is DELIBERATELY OMITTED: at
# this study's wide hidden layers (H=128/512, overcomplete) homeostasis is irrelevant at any
# strength (verified regime search: hybrid_full ~ no_homeo, gap ~0), while its contribution is
# large and CI-clean only at MODERATE H (=32) under interference+faults, which is exp13's job.
# Conditions kept: the device stack vs the eligibility-necessity control.
CONDS = {
    "hybrid_full": dict(mode="dfa", homeo=0.1),   # full stack (homeo at its validated strength)
    "no_trace":    dict(mode="no_trace", homeo=0.1),  # eligibility-necessity control
}


def _early_stop_for(A):
    """A-aware early-stop: chance = 1/A, and 'converged' is a high bar above chance so the
    stuck-padding never freezes a run that is still climbing on a >2-way task."""
    chance = 1.0 / A
    es = dict(EARLY_STOP)
    es["chance"] = chance
    es["criterion"] = round(chance + 0.55 * (1.0 - chance), 3)   # A=2 -> 0.78; A=4 -> 0.66
    return es


def _load_readouts(path):
    """Load cached frozen-front-end readouts. Returns ``(X (N,F) float in [0,1], y (N,) int,
    A)``. Readouts are min-max normalised per dimension so they are interpretable as input
    rates (the same [0,1] line-rate convention the built-in one-hot cue uses)."""
    d = np.load(path)
    X = np.asarray(d["X"], dtype=float)
    y = np.asarray(d["y"], dtype=int)
    lo = X.min(0, keepdims=True); hi = X.max(0, keepdims=True)
    X = (X - lo) / (hi - lo + 1e-9)
    return X, y, int(y.max()) + 1


def _make_sampler(ro, F):
    """Build a ``state_sampler(rng, B) -> (lines (B,F) in [0,1], label (B,))`` for
    ``train_deep``.

    With real readouts (``ro = (X, y, A)``): sample rows of the cached ``(X, y)`` -- the
    device stack must learn the policy behind the frozen front-end's actual embedding.
    With the proxy (``ro is None``): class-conditional noisy embeddings (overlapping
    clusters) whose label is XOR of two latent bits, so depth is genuinely required. Either
    way only the STATE source changes; the learning rule, eligibility, DFA, homeostasis and
    fault prior are identical to exp14.
    """
    if ro is not None:
        X, y, _A = ro
        N = len(y)

        def sampler(rng, B):
            idx = rng.integers(0, N, size=B)
            return X[idx], y[idx]
        return sampler

    # fixed class-mean templates (deterministic; the noise gives the spread)
    base = np.array([[1, 0, 1, 0, 0, 1, 0, 1],
                     [0, 1, 0, 1, 1, 0, 1, 0]], dtype=float)
    mu = np.resize(base, (2, F))

    def sampler(rng, B):
        b0 = rng.integers(2, size=B); b1 = rng.integers(2, size=B)
        y = (b0 ^ b1).astype(int)
        x = mu[y] + PROXY_NOISE * rng.standard_normal((B, F))
        return np.clip(x, 0.0, 1.0), y
    return sampler


def _hybrid_scale_run(job):
    """One (H, p, cond, seed) run of the exp16 sweep -> ``(H, p, cond, seed, final_rate)``.

    Module-level (not a closure) so it is picklable by ``multiprocessing.Pool`` in
    ``main()``. Reads the sweep config from the ``job`` tuple's trailing ``cfg`` dict
    ``(F, A, ro, trials)`` so no module-global mutable state has to cross the fork barrier.
    """
    H, p, cond, seed, cfg = job
    F, A, ro, trials = cfg["F"], cfg["A"], cfg["ro"], cfg["trials"]
    # Programmed-conductance fault prior (stuck-at + realistic asymmetric D2D). pf_on=False:
    # the Poole-Frenkel I-V term is a read/inference nonlinearity, out of scope for a rule
    # that credits the programmed weight (see entry_array_scale for the full rationale).
    fault = siox_fault_stack(p_stuck=p, sigma_g=SIGMA_G, sigma_g_on=SIGMA_G_ON, pf_on=False,
                             stuck_kind="off", seed=1000 * seed + int(100 * p))
    sampler = _make_sampler(ro, F)
    kw = CONDS[cond]
    rewards = train_deep(
        H=H, tau_leak=10.0, D=5.0, t_distract=3.0, distract_dur=0.3,
        trials=trials, seed0=seed, weight_fault=fault,
        early_stop=_early_stop_for(A),
        state_sampler=sampler, n_features=F, n_actions=A,
        **kw,
    )
    return (H, p, cond, seed, float(rewards[:, -200:].mean()))


def run_hybrid_scale(*, H_grid=(32, 128, 512), p_grid=(0.0, 0.2, 0.5), seeds=12,
                     trials=2000, readouts=None, pool=None):
    """exp16 Stage 2 (Tier 3): the width-scaling + fault-tolerance sweep behind the frozen
    front-end. Pure NumPy -- NO torch. Serial by default (a notebook can call it in-kernel);
    ``main()`` passes a ``multiprocessing.Pool`` in ``pool`` to parallelise the (H, p, cond,
    seed) grid.

    ``readouts`` is a path to the cached ``readouts.npz`` (Stage 1 output). If it is None or
    the file is absent, the sweep falls back to the HONEST proxy front-end (overlapping class
    clusters, F=8, XOR label, A=2) -- clearly recorded in the returned ``front_end`` field.
    The headline result should pass the real ``readouts.npz``.

    Conditions per (H, p): *hybrid_full* (the device stack: trace + DFA + homeostasis) and
    *no_trace* (eligibility-necessity control). Metric: final reward rate (mean over the last
    200 trials), summarised as mean + bootstrap 95% CI over seeds.

    Returns the result grid dict with the exact keys the driver wrote (H, p, conds, summary,
    readouts, front_end, F, A, seeds, trials, claim); no file I/O. ``main()`` writes it to
    ``exp16_hybrid_scale.npy``.
    """
    import os

    H_grid = [int(x) for x in H_grid]
    p_grid = [float(x) for x in p_grid]
    have_ro = bool(readouts) and os.path.exists(readouts)
    if have_ro:
        X, y, A = _load_readouts(readouts)
        ro, F = (X, y, A), X.shape[1]
    else:
        ro, F, A = None, PROXY_F, 2

    claim = (f"width-scaling + fault-tolerance robustness behind a "
             f"{'real frozen' if ro is not None else 'proxy'} front-end "
             f"(A={A}); homeostasis contribution is exp13's claim, NOT tested here")

    cfg = {"F": F, "A": A, "ro": ro, "trials": trials}
    jobs = [(H, p, cond, s, cfg) for H in H_grid for p in p_grid
            for cond in CONDS for s in range(seeds)]
    if pool is None:
        results = [_hybrid_scale_run(j) for j in jobs]
    else:
        results = pool.map(_hybrid_scale_run, jobs, chunksize=1)

    acc = {}
    for H, p, cond, s, val in results:
        acc.setdefault((H, p, cond), []).append(val)

    summary = {}
    for H in H_grid:
        for p in p_grid:
            for cond in CONDS:
                v = np.array(acc[(H, p, cond)])
                lo, hi = _boot_ci(v, seed=H + int(100 * p))
                summary[(H, p, cond)] = (float(v.mean()), lo, hi)

    return {"H": H_grid, "p": p_grid, "conds": list(CONDS), "summary": summary,
            "readouts": bool(have_ro), "front_end": "real" if have_ro else "proxy",
            "F": F, "A": A, "seeds": seeds, "trials": trials, "claim": claim}


def _boot_ci(vals, seed=0):
    """Percentile bootstrap 95% CI for the mean of ``vals`` (matches the original driver's
    inline bootstrap, N_BOOT resamples, keyed seed for reproducibility)."""
    rng = np.random.default_rng(seed)
    b = [vals[rng.integers(0, len(vals), len(vals))].mean() for _ in range(N_BOOT)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


# =============================================================================
# Full-scale reproduction CLI
# =============================================================================
def main(argv=None):
    r"""Full-scale reproduction CLI for the hybrid grids (writes ``data/results``).

    ``python -m mrl_trace.hybrid [--exp5] [--exp16] [--prep-readouts]
    [--readouts PATH] [--full|--quick]``

    With no experiment flag, runs exp5 + exp16. ``--full`` = published scale (exp5 20 seeds,
    exp16 H=32,128,512 x p=0,0.2,0.5 x 12 seeds x 2000 trials); ``--quick`` = a fast
    few-seed smoke run. ``--prep-readouts`` (Stage 1) trains the front-end and caches
    ``readouts.npz`` -- needs the ``hybrid`` extra. ``--readouts PATH`` points exp16 Stage 2
    at a cached readouts file; with no path it defaults to ``data/results/readouts.npz`` if
    present, else the honest proxy front-end.
    """
    import argparse
    import multiprocessing as mp
    import os
    from . import paths

    ap = argparse.ArgumentParser(description="Hybrid perception-front-end reproductions")
    ap.add_argument("--exp5", action="store_true",
                    help="hybrid decision (Fig 9) -> tier6_results.npy (needs torch)")
    ap.add_argument("--exp16", action="store_true",
                    help="width x fault sweep behind the front-end -> exp16_hybrid_scale.npy")
    ap.add_argument("--prep-readouts", action="store_true",
                    help="Stage 1: train front-end, cache readouts.npz (needs torch)")
    ap.add_argument("--readouts", default=None,
                    help="path to cached readouts.npz for exp16 Stage 2 "
                         "(default: data/results/readouts.npz if present, else proxy)")
    ap.add_argument("--workers", type=int, default=int(os.cpu_count() or 4),
                    help="worker processes for the exp16 sweep")
    ap.add_argument("--quick", action="store_true", help="fast few-seed smoke run")
    ap.add_argument("--full", action="store_true", help="published-scale run (default)")
    a = ap.parse_args(argv)
    run_all = not (a.exp5 or a.exp16 or a.prep_readouts)

    if a.prep_readouts:
        print("=== exp16 Stage 1: caching frozen front-end readouts ===")
        per_class = 200 if a.quick else 800
        steps = 120 if a.quick else 600
        out = prep_readouts(per_class=per_class, steps=steps)
        print(f"  wrote {out}")

    if a.exp5 or run_all:
        seeds = 6 if a.quick else _N_SEEDS
        trials = 400 if a.quick else 1500
        front_steps = 120 if a.quick else 600
        per_class = 200 if a.quick else 500
        print(f"=== exp5 hybrid decision (Fig 9): {seeds} seeds, {trials} trials ===")
        grid = run_hybrid_decision(seeds=seeds, front_steps=front_steps,
                                   per_class=per_class, trials=trials)
        paths.save_result("tier6_results.npy", grid)
        print(f"  wrote tier6_results.npy  acc={grid['front_acc']:.3f} "
              f"C1={grid['C1']} C2={grid['C2']} C3={grid['C3']}")

    if a.exp16 or run_all:
        # Default the readouts path to the bundled Stage-1 cache if the caller did not
        # supply one; falls back to the honest proxy if that file is absent.
        ro_path = a.readouts
        if ro_path is None:
            cand = paths.results_dir() / "readouts.npz"
            ro_path = str(cand) if cand.exists() else None
        if a.quick:
            H_grid, p_grid, seeds, trials = (32,), (0.0, 0.5), 4, 400
        else:
            H_grid, p_grid, seeds, trials = (32, 128, 512), (0.0, 0.2, 0.5), 12, 2000
        try:
            ctx = mp.get_context("fork")
        except ValueError:                                    # pragma: no cover
            ctx = mp.get_context()
        front = "real" if (ro_path and os.path.exists(ro_path)) else "proxy"
        print(f"=== exp16 hybrid-scale sweep: H={list(H_grid)} p={list(p_grid)} "
              f"seeds={seeds} front_end={front} ===")
        with ctx.Pool(a.workers) as pool:
            grid = run_hybrid_scale(H_grid=H_grid, p_grid=p_grid, seeds=seeds,
                                    trials=trials, readouts=ro_path, pool=pool)
        for H in grid["H"]:
            for p in grid["p"]:
                line = f"  H={H:4d} p={p:.2f}: "
                for cond in grid["conds"]:
                    m, lo, hi = grid["summary"][(H, p, cond)]
                    line += f"{cond}={m:.2f}[{lo:.2f},{hi:.2f}] "
                print(line)
        paths.save_result("exp16_hybrid_scale.npy", grid)
        print(f"  wrote exp16_hybrid_scale.npy  front_end={grid['front_end']}")


if __name__ == "__main__":
    main()
