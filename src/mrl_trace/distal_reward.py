"""Distal-reward credit-assignment tasks (manuscript Figs 4 and 5).

These are the open-loop credit-assignment demonstrations that precede the
closed-loop bandit of :mod:`mrl_trace.bandit`: a cue synapse undergoes a
coincidence, a reward arrives after an action--reward delay ``D``, and
reward-uncorrelated distractor synapses are active throughout. A correct learner
raises the cue weight and leaves the distractors near baseline despite the delay.

Two settings, matching the paper:

- ``trace_level`` (Fig 4): the device eligibility trace is precomputed as a kernel
  (from :meth:`TransientGate.trace`) and applied by its lag at reward time; success
  is the cue-to-distractor weight ratio.
- ``spiking`` (Fig 5): a full LIF output neuron with the device gate integrated
  online per synapse from real Poisson spike coincidences; success is the
  saturation of the cue synapse toward its conductance bound.

Both expose a per-seed ``train_*`` and a vectorised ``sweep_*`` that returns
per-seed arrays (so bootstrapped CIs can be computed). The retention time
``tau_leak`` sets the maximum learnable delay in both.
"""
from __future__ import annotations

import numpy as np

from .device import TransientGate, tau_r, tau_d, K_STAGES
from .neurons import lif_step, TAU_M, V_TH
from .learning import LTD_BIAS

__all__ = [
    "trace_kernel", "abstract_kernel", "train_trace_level",
    "SpikingGateBank", "train_spiking", "cue_saturation", "W_INIT", "W_MAX",
    "trace_ratio", "run_trace_window",
]

W_INIT, W_MAX = 0.5, 1.0


# ----------------------------------------------------------------------------
# Trace-level task (Fig 4): precomputed eligibility kernel + lag-based credit
# ----------------------------------------------------------------------------
def trace_kernel(t_grid, tau_leak, V=0.9):
    """Device eligibility kernel e(t) on ``t_grid`` for a coincidence at t=0."""
    g = TransientGate(V=V, tau_leak=tau_leak, dt=float(t_grid[1] - t_grid[0]))
    return g.trace(t_grid, coincidence_at=0.0, coincidence_dur=0.3)


def abstract_kernel(t_grid, tau):
    """Control: an abstract exponential eligibility kernel."""
    return np.exp(-np.asarray(t_grid, float) / tau)


def train_trace_level(tau_leak, D, *, N=8, trials=1500, T=300.0, dt=0.1,
                      eta=0.05, seed=0, abstract=False, no_trace=False):
    """One seed of the trace-level distal-reward task; returns final weights ``w``
    (``w[0]`` cue, ``w[1:]`` distractors).

    ``abstract=True`` uses an exponential kernel instead of the device trace;
    ``no_trace=True`` is the control with a delta kernel (eligibility only at the
    exact reward instant, so no credit survives any delay ``D>0``)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, dt)
    nt = len(t)
    if no_trace:
        kern = np.zeros(nt)
        kern[0] = 1.0
    elif abstract:
        kern = abstract_kernel(t, tau_leak)
    else:
        kern = trace_kernel(t, tau_leak)
    w = np.full(N, W_INIT)
    baseline = 0.5
    for _ in range(trials):
        rewarded = rng.random() < 0.5
        t_reward = 0.1 * T + D
        if t_reward >= T:
            continue
        ri = int(t_reward / dt)
        e = np.zeros(N)
        if rewarded:
            lag = ri - int(0.1 * T / dt)
            e[0] = kern[lag] if 0 <= lag < nt else 0.0
        for i in range(1, N):
            t_d = rng.uniform(0.02 * T, 0.95 * T)
            lag = ri - int(t_d / dt)
            e[i] = kern[lag] if 0 <= lag < nt else 0.0
        R = 1.0 if rewarded else 0.0
        baseline += 0.02 * (R - baseline)
        w = np.clip(w + eta * (R - baseline) * e, 0.0, 2.0)
    return w


# ----------------------------------------------------------------------------
# Spiking task (Fig 5): online per-synapse device gate + LIF output neuron
# ----------------------------------------------------------------------------
class SpikingGateBank:
    """``N`` independent device gates (the Fig-5 per-synapse eligibility bank).

    Equivalent to a vectorised :class:`TransientGate` over an ``(N,)`` grid, with
    the signed/leak-dominant drive and ``[-Vnmax, Vnmax]`` bound."""

    def __init__(self, N, V=0.9, tau_leak=5.0, k=K_STAGES, dt=1e-3, Vnmax=1.0):
        self.N, self.k, self.dt, self.Vnmax = N, k, dt, Vnmax
        self.alpha = k / tau_r(V)
        self.tau_d = tau_d(V)
        self.tau_leak = tau_leak
        self.vn = np.zeros((N, k))
        self.vsc = np.zeros(N)

    def reset(self):
        self.vn[:] = 0.0
        self.vsc[:] = 0.0

    def step(self, drive):
        dt, a, Vm = self.dt, self.alpha, self.Vnmax
        new = self.vn.copy()
        prev = drive
        for j in range(self.k):
            new[:, j] = self.vn[:, j] + dt * (
                a * prev * (Vm - np.abs(self.vn[:, j])) - self.vn[:, j] / self.tau_leak)
            prev = self.vn[:, j]
        self.vsc += dt * (a * drive * (Vm - np.abs(self.vsc)) - self.vsc / self.tau_d)
        self.vn = np.clip(new, -Vm, Vm)
        return self.vn[:, -1] / Vm


def _spiking_trial(bank, w, rewarded, D, *, dt=1e-3, cue_rate=200.0, dist_rate=40.0,
                   N=8, rng=None, eta=0.05, baseline=0.5, ltd_bias=LTD_BIAS):
    cue_window = (0.5, 0.5 + 2.0)          # seconds-long correlated cue epoch
    bank.reset()
    v_out = 0.0
    ri = int((cue_window[1] + D) / dt)
    nsteps = ri + 2
    e_at_reward = np.zeros(N)
    for n in range(nsteps):
        t = n * dt
        rates = np.full(N, dist_rate)
        if rewarded and cue_window[0] <= t < cue_window[1]:
            rates[0] = cue_rate
        pre = rng.random(N) < rates * dt
        charge = np.dot(w, pre.astype(float))
        v_out, post = lif_step(v_out, charge, dt)
        drive = (pre.astype(float) * (1.0 if post else -ltd_bias)) / dt * 1e-3
        e = bank.step(drive)
        if n == ri:
            e_at_reward = e.copy()
    R = 1.0 if rewarded else 0.0
    w = np.clip(w + eta * (R - baseline) * e_at_reward, 0.0, W_MAX)
    return w, R


def train_spiking(tau_leak, D, *, N=8, trials=400, dt=1e-3, seed=0, eta=0.2):
    """One seed of the full-spiking distal-reward task; returns final weights ``w``
    (``w[0]`` cue saturation toward the bound is the success measure)."""
    rng = np.random.default_rng(seed)
    bank = SpikingGateBank(N, tau_leak=tau_leak, dt=dt)
    w = np.full(N, W_INIT)
    baseline = 0.5
    for _ in range(trials):
        rewarded = rng.random() < 0.5
        w, R = _spiking_trial(bank, w, rewarded, D, dt=dt, N=N, rng=rng,
                              baseline=baseline, eta=eta)
        baseline += 0.02 * (R - baseline)
    return w


def cue_saturation(w):
    """Saturation of the cue synapse toward its bound: (w0 - W_INIT)/(W_MAX - W_INIT)."""
    return (w[0] - W_INIT) / (W_MAX - W_INIT)


# =============================================================================
# Experiment cores (the distal-reward studies that compose ``train_trace_level``)
#
# Each ``run_*`` returns the result grid as a plain dict -- no file I/O, no
# plotting, no stdout.  Notebooks call these at a small (quick) seed count and
# render the figures inline; ``main()`` (below) calls them at the published
# 20-seed scale and writes the grid under ``data/results/`` for the notebooks to
# render.  This mirrors the pattern in :mod:`mrl_trace.bandit`.
# =============================================================================

# Fig 4 (tier1) sweep axes: action->reward delay grid, the three device
# retention times, and the no-trace (delta-kernel) control. "learned" if the
# cue-to-distractor weight ratio reaches PASS. These constants match the
# 20-seed driver exactly so the published grid is bit-reproducible.
TIER1_DELAYS = [1, 2, 5, 10, 20, 40, 70, 100]
# (label, tau_leak, no_trace?) ; "none" is the no-trace control (delta kernel)
TIER1_VARIANTS = [("gate_tl20", 20.0, False), ("gate_tl5", 5.0, False),
                  ("gate_tl1", 1.0, False), ("none", None, True)]
TIER1_PASS = 2.0


def trace_ratio(w):
    """Cue-to-distractor weight ratio -- the trace-level success measure."""
    return w[0] / max(np.mean(w[1:]), 1e-6)


def _trace_cell(tau_leak, D, no_trace, seeds):
    """Per-seed cue/distractor ratios for one (variant, delay) cell (serial)."""
    if no_trace:
        # the no-trace control uses a delta kernel (no eligibility persistence)
        return np.array([trace_ratio(train_trace_level(0.0, D, seed=s, no_trace=True))
                         for s in range(seeds)])
    return np.array([trace_ratio(train_trace_level(tau_leak, D, seed=s))
                     for s in range(seeds)])


def run_trace_window(*, seeds=20, delays=TIER1_DELAYS, variants=TIER1_VARIANTS,
                     PASS=TIER1_PASS):
    """Fig 4 grid (tier1): trace-level credit-assignment window.

    Learned cue-to-distractor weight ratio versus action--reward delay ``D``, for
    three device retention times ``tau_leak in {20, 5, 1}`` s and a no-trace control
    (delta kernel), at ``seeds`` seeds with bootstrapped 95% CIs. The device trace is
    cheap here (precomputed kernel), so the seeds run sequentially.

    A cell is "learned" (contributes to ``max_learn``) if its mean ratio reaches
    ``PASS`` (=2.0): the cue synapse is at least twice the distractor weight, so
    credit is correctly assigned to the cue despite the delay. The maximum learnable
    delay per variant is the retention-window signature the manuscript reports.

    Returns the result dict (per-variant mean ratios + bootstrap CIs, max learnable
    delay per variant); no file I/O. Dict keys (``delays``, ``ratios``,
    ``ratios_ci``, ``max_learn``, ``n_seeds``) match the published grid exactly.
    """
    from .stats import bootstrap_ci
    ratios, ratios_ci = {}, {}
    for name, tl, no_trace in variants:
        row, row_ci = [], []
        for D in delays:
            r = _trace_cell(tl, D, no_trace, seeds)
            row.append(float(r.mean()))
            row_ci.append(bootstrap_ci(r))
        ratios[name] = row
        ratios_ci[name] = row_ci

    def _maxd(name):
        ds = [d for d, v in zip(delays, ratios[name]) if v >= PASS]
        return max(ds) if ds else 0

    return {"delays": list(delays), "ratios": ratios, "ratios_ci": ratios_ci,
            "max_learn": {name: _maxd(name) for name, _, _ in variants},
            "n_seeds": seeds}


def _trace_cell_star(args):
    """Pool worker: unpack a (variant, delay) cell and return its per-seed ratios."""
    name, tl, no_trace, D, seeds = args
    return (name, D, _trace_cell(tl, D, no_trace, seeds))


def run_trace_window_parallel(*, seeds=20, delays=TIER1_DELAYS,
                              variants=TIER1_VARIANTS, PASS=TIER1_PASS,
                              processes=None):
    """``run_trace_window`` computed with a ``multiprocessing.Pool`` over the
    coarse (variant x delay) cell axis (for the full-scale ``main()`` driver).

    Identical grid dict and science to :func:`run_trace_window`; only the cell loop
    is parallelised. NOT for in-notebook use -- notebooks call the serial core.
    """
    import multiprocessing as mp
    from .stats import bootstrap_ci
    jobs = [(name, tl, no_trace, D, seeds)
            for name, tl, no_trace in variants for D in delays]
    with mp.Pool(processes=processes) as pool:
        cells = pool.map(_trace_cell_star, jobs)
    per_cell = {(name, D): r for name, D, r in cells}

    ratios, ratios_ci = {}, {}
    for name, _tl, _nt in variants:
        row, row_ci = [], []
        for D in delays:
            r = per_cell[(name, D)]
            row.append(float(r.mean()))
            row_ci.append(bootstrap_ci(r))
        ratios[name] = row
        ratios_ci[name] = row_ci

    def _maxd(name):
        ds = [d for d, v in zip(delays, ratios[name]) if v >= PASS]
        return max(ds) if ds else 0

    return {"delays": list(delays), "ratios": ratios, "ratios_ci": ratios_ci,
            "max_learn": {name: _maxd(name) for name, _, _ in variants},
            "n_seeds": seeds}


def main(argv=None):
    """Full-scale reproduction CLI for the distal-reward grids (writes ``data/results``).

    ``python -m mrl_trace.distal_reward [--tier1] [--full|--quick]``
    With no experiment flag, runs all. ``--full`` = 20 seeds (published); ``--quick``
    = a fast few-seed smoke run. The trace-level cells are independent, so the
    full-scale run parallelises over the (variant x delay) grid with a Pool.
    """
    import argparse
    from . import paths
    ap = argparse.ArgumentParser(description="Distal-reward credit-assignment reproductions")
    ap.add_argument("--tier1", action="store_true",
                    help="Fig 4 trace-level window -> tier1_gate_results.npy")
    ap.add_argument("--quick", action="store_true", help="fast few-seed smoke run")
    ap.add_argument("--full", action="store_true", help="published 20-seed run (default)")
    a = ap.parse_args(argv)
    run_all = not a.tier1
    seeds = 4 if a.quick else 20

    if a.tier1 or run_all:
        print(f"=== Fig 4 trace-level, N={seeds} seeds, bootstrap 95% CIs ===")
        grid = run_trace_window_parallel(seeds=seeds)
        for name, _tl, _nt in TIER1_VARIANTS:
            print(f"{name:11s} " + " ".join(f"{v:5.2f}" for v in grid["ratios"][name]))
        print("max learnable delay:", grid["max_learn"])
        paths.save_result("tier1_gate_results.npy", grid)
        print("  wrote tier1_gate_results.npy")


if __name__ == "__main__":
    main()
