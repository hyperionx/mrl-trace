"""Distal-reward credit-assignment tasks (manuscript Figs 4 and 5).

These are the open-loop credit-assignment demonstrations that precede the
closed-loop bandit of :mod:`mrl_trace.bandit`: a cue synapse undergoes a
coincidence, a reward arrives after an action--reward delay ``D``, and
reward-uncorrelated distractor synapses are active throughout. A correct learner
raises the cue weight and leaves the distractors near baseline despite the delay.

Two settings, matching the paper:

- ``trace_level`` (Fig 4): the device eligibility trace is precomputed as a kernel
  (from :meth:`CascadeEligibilityGate.trace`) and applied by its lag at reward time; success
  is the cue-to-distractor weight ratio.
- ``spiking`` (Fig 5): a full LIF output neuron with the device gate integrated
  online per synapse from real Poisson spike coincidences; success is the
  saturation of the cue synapse toward its conductance bound.

Both expose a per-seed ``train_*`` and a vectorised ``sweep_*`` that returns
per-seed arrays (so bootstrapped CIs can be computed). Sweeping ``tau_leak`` changes
the simulated delay sensitivity in both; it is not an independently tested physical law.
"""
from __future__ import annotations

import numpy as np

from .device import (CascadeEligibilityGate, LinearErlangEligibilityGate,
                     decay_matched_exponential_tau, tau_r, K_STAGES)
from .model_specs import LINEAR_MODEL_ID, PRIMARY_MODEL_ID
from .neurons import lif_step, TAU_M, V_TH
from .learning import LTD_BIAS

DISTAL_METHOD_PROVENANCE = {
    "status": "proposed",
    "established_basis": ["three-factor delayed-reward learning"],
    "repository_adaptation": (
        "The cascade eligibility surrogate is driven by the repository's signed "
        "coincidence rule in trace-level and LIF credit tasks."
    ),
    "claim_limit": (
        "Simulation evidence with deliberately swept retention; the cascade inherits "
        "the approximation limits recorded by CASCADE_METHOD_PROVENANCE."
    ),
}

__all__ = [
    "trace_kernel", "abstract_kernel", "train_trace_level",
    "SpikingGateBank", "LinearErlangSpikingGateBank", "train_spiking",
    "cue_saturation", "W_INIT", "W_MAX",
    "trace_ratio", "run_trace_window", "run_spiking_saturation",
    "DISTAL_METHOD_PROVENANCE",
]

W_INIT, W_MAX = 0.5, 1.0


# ----------------------------------------------------------------------------
# Trace-level task (Fig 4): precomputed eligibility kernel + lag-based credit
# ----------------------------------------------------------------------------
def trace_kernel(t_grid, tau_leak, V=0.9, k=K_STAGES, tau_r_override=None,
                 beta_leak=1.0, gate_model=PRIMARY_MODEL_ID):
    """Device eligibility kernel e(t) on ``t_grid`` for a coincidence at t=0."""
    gate_class = (
        CascadeEligibilityGate if gate_model == PRIMARY_MODEL_ID
        else LinearErlangEligibilityGate if gate_model == LINEAR_MODEL_ID
        else None
    )
    if gate_class is None:
        raise ValueError(f"unknown gate model {gate_model!r}")
    g = gate_class(V=V, tau_leak=tau_leak, k=k,
                   tau_r_override=tau_r_override, beta_leak=beta_leak,
                   dt=float(t_grid[1] - t_grid[0]))
    return g.trace(t_grid, coincidence_at=0.0, coincidence_dur=0.3)


def abstract_kernel(t_grid, tau):
    """Control: an abstract exponential eligibility kernel."""
    return np.exp(-np.asarray(t_grid, float) / tau)


def train_trace_level(tau_leak, D, *, N=8, trials=1500, T=300.0, dt=0.1,
                      eta=0.05, seed=0, abstract=False, no_trace=False, V=0.9,
                      k=K_STAGES, tau_r_override=None, beta_leak=1.0,
                      gate_model=PRIMARY_MODEL_ID):
    """One seed of the trace-level distal-reward task; returns final weights ``w``
    (``w[0]`` cue, ``w[1:]`` distractors).

    ``abstract=True`` uses a single exponential fitted to the device surrogate's
    post-peak 80--10% decay band instead of merely reusing its nominal retention;
    ``no_trace=True`` is the control with a delta kernel (eligibility only at the
    exact reward instant, so no credit survives any delay ``D>0``)."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, dt)
    nt = len(t)
    if no_trace:
        kern = np.zeros(nt)
        kern[0] = 1.0
    elif abstract:
        matched_tau = decay_matched_exponential_tau(
            tau_leak, V=V, k=k, tau_r_override=tau_r_override,
            beta_leak=beta_leak, gate_model=gate_model,
        )
        kern = abstract_kernel(t, matched_tau)
    else:
        kern = trace_kernel(t, tau_leak, V=V, k=k,
                            tau_r_override=tau_r_override,
                            beta_leak=beta_leak, gate_model=gate_model)
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
    """Primary nonlinear headroom gate over ``N`` synapses."""

    model_id = PRIMARY_MODEL_ID

    def __init__(self, N, V=0.9, tau_leak=5.0, k=K_STAGES, dt=1e-3,
                 Vnmax=1.0, tau_r_override=None, beta_leak=1.0):
        self.N, self.k, self.dt, self.Vnmax = int(N), int(k), float(dt), float(Vnmax)
        fitted_tau_r = tau_r(V) if tau_r_override is None else float(tau_r_override)
        self.alpha = self.k / fitted_tau_r
        self.beta_leak = float(beta_leak)
        tl = np.asarray(tau_leak, dtype=float)
        if (self.beta_leak <= 0 or not np.isfinite(self.beta_leak)
                or not np.all(np.isfinite(tl)) or np.any(tl <= 0)
                or self.Vnmax <= 0):
            raise ValueError("gate scales and leakage parameters must be positive")
        if tl.ndim == 0:
            self.tau_leak = tl
        elif tl.shape == (N,):
            self.tau_leak = tl[:, None]
        else:
            raise ValueError("tau_leak must be scalar or have shape (N,)")
        self._t_since = np.full((N, 1), dt) if beta_leak != 1.0 else None
        self.vn = np.zeros((N, self.k))

    def reset(self):
        self.vn[:] = 0.0
        if self._t_since is not None:
            self._t_since[:] = self.dt

    def step(self, drive):
        drive = np.asarray(drive, dtype=float)
        if drive.shape != (self.N,):
            raise ValueError(f"drive must have shape ({self.N},)")
        previous_fraction = np.empty_like(self.vn)
        previous_fraction[:, 0] = drive
        previous_fraction[:, 1:] = self.vn[:, :-1] / self.Vnmax
        if self.beta_leak == 1.0:
            leak_rate = 1.0 / self.tau_leak
        else:
            self._t_since = np.where(
                np.abs(drive)[:, None] > 1e-9, self.dt, self._t_since + self.dt
            )
            tau = self.tau_leak
            leak_rate = (self.beta_leak / tau) * np.power(
                np.clip(self._t_since / tau, 1e-6, None), self.beta_leak - 1.0
            )
        new = self.vn + self.dt * (
            self.alpha * previous_fraction * (self.Vnmax - np.abs(self.vn))
            - self.vn * leak_rate
        )
        self.vn = np.clip(new, -self.Vnmax, self.Vnmax)
        return self.vn[:, -1] / self.Vnmax


class LinearErlangSpikingGateBank:
    """Linear Erlang-exact sensitivity over ``N`` synapses.

    Equivalent to a vectorised :class:`LinearErlangEligibilityGate` over an ``(N,)``
    grid: the same linear Erlang cascade, signed/leak-dominant drive,
    age-dependent leakage option and ``[-Vnmax, Vnmax]`` bound."""

    model_id = LINEAR_MODEL_ID

    def __init__(self, N, V=0.9, tau_leak=5.0, k=K_STAGES, dt=1e-3, Vnmax=1.0,
                 tau_r_override=None, beta_leak=1.0):
        self.N, self.k, self.dt, self.Vnmax = N, k, dt, Vnmax
        fitted_tau_r = tau_r(V) if tau_r_override is None else float(tau_r_override)
        self.alpha = k / fitted_tau_r
        self.beta_leak = float(beta_leak)
        tl = np.asarray(tau_leak, dtype=float)
        if (self.beta_leak <= 0 or not np.isfinite(self.beta_leak)
                or not np.all(np.isfinite(tl)) or np.any(tl <= 0)):
            raise ValueError("beta_leak and all tau_leak values must be finite and positive")
        if tl.ndim == 0:
            self.tau_leak = tl
        elif tl.shape == (N,):
            self.tau_leak = tl[:, None]
        else:
            raise ValueError("tau_leak must be scalar or have shape (N,)")
        self._t_since = np.full((N, 1), dt) if beta_leak != 1.0 else None
        self.vn = np.zeros((N, k))

    def reset(self):
        self.vn[:] = 0.0
        if self._t_since is not None:
            self._t_since[:] = self.dt

    def step(self, drive):
        dt, a, Vm = self.dt, self.alpha, self.Vnmax
        drive = np.asarray(drive, dtype=float)
        if drive.shape != (self.N,):
            raise ValueError(f"drive must have shape ({self.N},)")
        vn = self.vn
        prev = np.empty_like(vn)
        prev[:, 0] = Vm * drive
        prev[:, 1:] = vn[:, :-1]
        if self.beta_leak == 1.0:
            leak_rate = 1.0 / self.tau_leak
        else:
            self._t_since = np.where(
                np.abs(drive)[:, None] > 1e-9, dt, self._t_since + dt
            )
            tau = self.tau_leak
            leak_rate = (self.beta_leak / tau) * np.power(
                np.clip(self._t_since / tau, 1e-6, None), self.beta_leak - 1.0
            )
        new = vn + dt * (a * (prev - vn) - vn * leak_rate)
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


def train_spiking(tau_leak, D, *, N=8, trials=400, dt=1e-3, seed=0, eta=0.2,
                  V=0.9, k=K_STAGES, tau_r_override=None, beta_leak=1.0,
                  gate_model=PRIMARY_MODEL_ID):
    """One seed of the full-spiking distal-reward task; returns final weights ``w``
    (``w[0]`` cue saturation toward the bound is the success measure)."""
    rng = np.random.default_rng(seed)
    bank_cls = {
        PRIMARY_MODEL_ID: SpikingGateBank,
        LINEAR_MODEL_ID: LinearErlangSpikingGateBank,
    }.get(gate_model)
    if bank_cls is None:
        raise ValueError(f"unknown gate_model: {gate_model!r}")
    bank = bank_cls(N, tau_leak=tau_leak, dt=dt, V=V, k=k,
                    tau_r_override=tau_r_override, beta_leak=beta_leak)
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


def _spiking_saturation_cell(args):
    """Spawn-safe worker for one retention/delay cell of the Fig. 5 sweep."""
    (name, tau_leak, delay, seeds, trials, dt, eta, V, k, tau_r_override,
     beta_leak) = args
    values = np.asarray([
        cue_saturation(train_spiking(
            tau_leak, delay, trials=trials, dt=dt, seed=seed, eta=eta,
            V=V, k=k, tau_r_override=tau_r_override, beta_leak=beta_leak
        ))
        for seed in range(seeds)
    ], dtype=float)
    return name, delay, values


def run_spiking_saturation(*, seeds=20, delays=(1, 2, 5, 10, 20, 40),
                           variants=(("gate_tl10", 10.0),
                                     ("gate_tl2", 2.0),
                                     ("gate_tl0.5", 0.5)),
                           trials=400, dt=1e-3, eta=0.2, workers=1, V=0.9,
                           k=K_STAGES, tau_r_override=None, beta_leak=1.0):
    """Fig. 5 grid: live full-spiking credit saturation.

    This is the missing thin sweep driver around the preserved per-seed
    :func:`train_spiking` implementation.  It changes no model equations or
    random-number behaviour: independent ``(retention, delay)`` cells may merely
    be dispatched to spawn-safe CPU processes.  Per-seed values are retained so
    the appendix figure can calculate honest uncertainty bands.
    """
    jobs = [
        (name, tau_leak, delay, int(seeds), int(trials), float(dt), float(eta),
         float(V), int(k), tau_r_override, float(beta_leak))
        for name, tau_leak in variants for delay in delays
    ]
    if workers in (None, "auto"):
        import os
        workers = min(6, max(1, (os.cpu_count() or 2) // 2))
    workers = int(workers)
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn")
        ) as pool:
            cells = list(pool.map(_spiking_saturation_cell, jobs))
    else:
        cells = [_spiking_saturation_cell(job) for job in jobs]

    by_cell = {(name, delay): values for name, delay, values in cells}
    sat_seeds = {
        name: {delay: by_cell[(name, delay)] for delay in delays}
        for name, _tau_leak in variants
    }
    saturation = {
        name: [float(np.mean(sat_seeds[name][delay])) for delay in delays]
        for name, _tau_leak in variants
    }
    return {
        "delays": list(delays),
        "saturation": saturation,
        "sat_seeds": sat_seeds,
        "n_seeds": int(seeds),
        "trials": int(trials),
        "dt": float(dt),
        "beta_leak": float(beta_leak),
        "retention_definition": "deliberately_swept",
        "method_provenance": DISTAL_METHOD_PROVENANCE,
    }


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


def _trace_cell(tau_leak, D, no_trace, seeds, V=0.9, k=K_STAGES,
                tau_r_override=None):
    """Per-seed cue/distractor ratios for one (variant, delay) cell (serial)."""
    if no_trace:
        # the no-trace control uses a delta kernel (no eligibility persistence)
        return np.array([trace_ratio(train_trace_level(
            0.0, D, seed=s, no_trace=True, V=V, k=k,
            tau_r_override=tau_r_override))
                         for s in range(seeds)])
    return np.array([trace_ratio(train_trace_level(
        tau_leak, D, seed=s, V=V, k=k, tau_r_override=tau_r_override))
                     for s in range(seeds)])


def run_trace_window(*, seeds=20, delays=TIER1_DELAYS, variants=TIER1_VARIANTS,
                     PASS=TIER1_PASS, V=0.9, k=K_STAGES, tau_r_override=None):
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
    ratios, ratios_ci, seed_ratios = {}, {}, {}
    for name, tl, no_trace in variants:
        row, row_ci, row_seeds = [], [], {}
        for D in delays:
            r = _trace_cell(tl, D, no_trace, seeds, V=V, k=k,
                            tau_r_override=tau_r_override)
            row.append(float(r.mean()))
            row_ci.append(bootstrap_ci(r))
            row_seeds[D] = np.asarray(r, float)
        ratios[name] = row
        ratios_ci[name] = row_ci
        seed_ratios[name] = row_seeds

    def _maxd(name):
        ds = [d for d, v in zip(delays, ratios[name]) if v >= PASS]
        return max(ds) if ds else 0

    return {"delays": list(delays), "ratios": ratios, "ratios_ci": ratios_ci,
            "seed_ratios": seed_ratios,
            "max_learn": {name: _maxd(name) for name, _, _ in variants},
            "n_seeds": seeds, "retention_definition": "deliberately_swept",
            "method_provenance": DISTAL_METHOD_PROVENANCE}


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
            "n_seeds": seeds, "retention_definition": "deliberately_swept",
            "method_provenance": DISTAL_METHOD_PROVENANCE}


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
