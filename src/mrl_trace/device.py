r"""Empirical SiO\ :sub:`x` current model and an Erlang-cascade eligibility surrogate.

The measured current is fitted empirically by a KWW rise multiplied by a screening
decay.  The learning simulations use a separate linear cascade-plus-leak state.  With
no discharge and a unit step, its continuous-time final-stage response is exactly the
Erlang CDF used by the model-identification workflow.  That Erlang response is only a
compact approximation to the KWW rise: the cascade is not a state-space realisation of
the complete product-current fit, and its integer depth is not an identified microscopic
trap count.

Fitted field-acceleration laws (manuscript Eqs. (5.8)/(5.9), ``V`` = ``|bias|`` in volts)::

    tau_r(V) = 1.45e2 * exp(-2.9 V)   s     (cascade rise)
    tau_d(V) = 1.17e4 * exp(-3.9 V)   s     (space-charge decay)
    beta = 2 (fixed shape exponent),  k = 3 representative cascade stages

``tau_leak`` is either a directly fitted discharge time or a deliberately swept model
parameter, as recorded by each experiment.  The repository does not demonstrate
electrode or fabrication control of this quantity.

The gate integrates by forward Euler and is forward-only (the learning rule is a
local three-factor rule, not BPTT), so there is no autograd-stiffness concern.
"""
from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np

__all__ = ["tau_r", "tau_d", "BETA", "K_STAGES", "CascadeEligibilityGate",
           "TransientGate", "CASCADE_METHOD_PROVENANCE", "KWW_METHOD_PROVENANCE",
           "fit_kww_laws", "simulate_habituation", "KWW_VOLTAGES",
           "decay_matched_exponential_tau"]

#: Historical compact-model compression exponent of the Au rise.
BETA = 2.0
#: Historical compact representative (beta ~ 2 shape-matches k ~ 3); publication
#: analysis freezes its pooled-CV representative and reports depth uncertainty.
K_STAGES = 3


def tau_r(V: float) -> float:
    """Field-accelerated cascade *rise* time constant (s); ``V`` = ``|bias|`` (V)."""
    return 1.45e2 * np.exp(-2.9 * V)


def tau_d(V: float) -> float:
    """Field-accelerated space-charge *decay* time constant (s); ``V`` = ``|bias|`` (V)."""
    return 1.17e4 * np.exp(-3.9 * V)


CASCADE_METHOD_PROVENANCE = {
    "status": "adapted",
    "established_basis": ["sequential-state waiting times", "leaky eligibility traces"],
    "repository_adaptation": (
        "A linear Erlang cascade is shape-calibrated to the empirical KWW rise and "
        "combined with a specified leakage time and, when beta_leak differs from one, "
        "an age-dependent stretched-discharge hazard."
    ),
    "claim_limit": (
        "The no-leak step response matches the fitted Erlang candidate, but that "
        "candidate only approximates the KWW rise; it neither exactly realises the "
        "product-current model nor identifies a microscopic stage count. The "
        "stretched-discharge clock resets on drive and is a non-Markov surrogate."
    ),
}

KWW_METHOD_PROVENANCE = {
    "status": "empirical_fit",
    "established_basis": ["Kohlrausch-Williams-Watts empirical relaxation fitting"],
    "repository_adaptation": "Shared field laws are fitted across measured Au traces.",
    "claim_limit": "The empirical fit does not uniquely identify a microscopic mechanism.",
}


class CascadeEligibilityGate:
    """Stateful linear Erlang-cascade-plus-leak eligibility surrogate.

    State is ``k`` identical first-order low-pass nodes in series.  With a unit-step
    drive and no leakage, the final-stage continuous-time response is
    ``gammainc(k, k*t/tau_r)``: the same Erlang CDF evaluated during physical-model
    selection.  The implementation uses forward Euler, so agreement converges with
    ``dt``.  The output is the normalised final stage with ``tau_leak`` relaxation.
    For ``beta_leak != 1`` the leakage rate is the instantaneous
    stretched-exponential hazard evaluated at time since the most recent drive; this
    reset-clock construction is an explicit non-Markov approximation. Space charge is
    not part of this output; it remains in the separate current/habituation model.

    Parameters
    ----------
    V : float
        Bias magnitude (V); sets the field-accelerated rise ``tau_r``.
    tau_leak : float
        Retention/relaxation time constant (s); measured or deliberately swept as
        stated by the calling experiment.
    beta_leak : float
        Stretched-exponential discharge exponent. ``1`` is the historical single-rate
        approximation; empirical ITO-linked runs must pass and record their fitted value.
    k : int
        Number of sequential cascade stages.
    dt : float
        Integration step (s).
    vnmax : float
        State scale and symmetric numerical bound. A unit drive has magnitude
        ``vnmax`` inside the cascade and the returned eligibility is divided by it.
    shape : tuple[int, ...]
        Optional grid shape for a vectorised bank of independent gates (e.g.
        ``(n_state, n_action)`` for a crossbar).  Default ``()`` is a single gate.
    """

    def __init__(self, V: float = 0.9, tau_leak: float = 2.0, k: int = K_STAGES,
                 dt: float = 0.05, vnmax: float = 1.0, shape: tuple = (),
                 tau_r_override: float | None = None, beta_leak: float = 1.0):
        if (not np.isfinite(tau_leak) or not np.isfinite(beta_leak)
                or tau_leak <= 0 or beta_leak <= 0):
            raise ValueError("tau_leak and beta_leak must be finite and positive")
        self.V = V
        self.tau_leak = tau_leak
        self.k = k
        self.dt = dt
        self.vnmax = vnmax
        self.shape = tuple(shape)
        self.beta_leak = float(beta_leak)
        # per-stage rate: total rise matches the fitted tau_r(V) spread over k
        # sequential stages (Erlang-k).
        self.tau_r = float(tau_r(V) if tau_r_override is None else tau_r_override)
        self.alpha = k / self.tau_r
        self.reset()

    def reset(self) -> None:
        """Zero all internal states."""
        self.vn = np.zeros(self.shape + (self.k,))
        self._t_since = np.full(self.shape, self.dt, dtype=float)

    def step(self, drive):
        """Advance one ``dt`` under coincidence ``drive`` and return ``e(t)``.

        ``drive`` is the instantaneous (optionally signed) coincidence input,
        broadcast over ``shape``.  Returns the normalised final-stage occupancy.
        """
        dt, a, vm = self.dt, self.alpha, self.vnmax
        drive = np.asarray(drive, dtype=float)
        self._t_since = np.where(
            np.abs(drive) > 1e-9, dt, self._t_since + dt
        )
        if self.beta_leak == 1.0:
            leak_rate = 1.0 / self.tau_leak
        else:
            leak_rate = (self.beta_leak / self.tau_leak) * np.power(
                np.clip(self._t_since / self.tau_leak, 1e-6, None),
                self.beta_leak - 1.0,
            )
        new = self.vn.copy()
        # A unit drive corresponds to the state scale ``vm``.  Every stage uses the
        # upstream node's OLD value, which is the forward-Euler discretisation of
        #
        #   dv_1/dt = alpha * (vm * drive - v_1) - h(a) * v_1
        #   dv_m/dt = alpha * (v_{m-1} - v_m) - h(a) * v_m.
        #
        # Without leakage, the continuous-time unit-step response of v_k / vm is
        # gammainc(k, alpha*t), exactly the Erlang candidate used during fitting.
        prev = vm * drive
        for j in range(self.k):
            vj = self.vn[..., j]
            new[..., j] = vj + dt * (
                a * (prev - vj) - vj * leak_rate
            )
            prev = vj
        self.vn = np.clip(new, -vm, vm)
        return self.vn[..., -1] / vm

    def trace(self, t_grid, coincidence_at: float, coincidence_dur: float = 0.2,
              normalise: bool = True) -> np.ndarray:
        """Full ``e(t)`` over ``t_grid`` for one coincidence (single-gate only).

        A coincidence of duration ``coincidence_dur`` starts at ``coincidence_at``.
        With ``normalise`` the trace is divided by its peak.
        """
        if self.shape:
            raise ValueError("trace() is for a single gate (shape=()).")
        self.reset()
        t_grid = np.asarray(t_grid, float)
        out = np.zeros_like(t_grid)
        for i, tt in enumerate(t_grid):
            d = 1.0 if coincidence_at <= tt < coincidence_at + coincidence_dur else 0.0
            out[i] = self.step(d)
        if normalise:
            pk = out.max()
            return out / pk if pk > 0 else out
        return out


@lru_cache(maxsize=256)
def decay_matched_exponential_tau(
    tau_leak: float,
    V: float = 0.9,
    k: int = K_STAGES,
    tau_r_override: float | None = None,
    beta_leak: float = 1.0,
    coincidence_dur: float = 0.3,
) -> float:
    """Fit the exponential control to the surrogate's post-peak decay.

    Matching nominal ``tau_leak`` values is not decay matching because every cascade
    stage also contains the intrinsic ``-alpha*v`` term.  This deterministic protocol
    simulates one standard coincidence pulse, normalises the last-stage response, and
    fits ``log(e/e_peak)`` against time over the predeclared 80--10% post-peak band.
    The returned time constant is used only by the single-exponential control; it does
    not alter or reinterpret the fitted ITO retention.
    """
    tau_leak = float(tau_leak)
    V = float(V)
    k = int(k)
    beta_leak = float(beta_leak)
    coincidence_dur = float(coincidence_dur)
    fitted_tau_r = tau_r(V) if tau_r_override is None else float(tau_r_override)
    if (not np.isfinite(tau_leak) or tau_leak <= 0 or k < 1
            or not np.isfinite(fitted_tau_r) or fitted_tau_r <= 0
            or not np.isfinite(beta_leak) or beta_leak <= 0
            or not np.isfinite(coincidence_dur) or coincidence_dur <= 0):
        raise ValueError("matching parameters must be finite and positive")

    # alpha=k/tau_r bounds the asymptotic decay even for a stretched-leak hazard.
    # A fixed dense grid makes the matching rule independent of each task's solver dt.
    horizon = max(5.0 * coincidence_dur, 16.0 * fitted_tau_r)
    t_grid = np.linspace(0.0, horizon, 12001)
    gate = CascadeEligibilityGate(
        V=V, tau_leak=tau_leak, k=k, dt=float(t_grid[1]),
        tau_r_override=fitted_tau_r, beta_leak=beta_leak,
    )
    response = gate.trace(
        t_grid, coincidence_at=0.0, coincidence_dur=coincidence_dur,
        normalise=True,
    )
    peak_index = int(np.argmax(response))
    tail = response[peak_index:]
    elapsed = t_grid[peak_index:] - t_grid[peak_index]
    mask = (tail <= 0.80) & (tail >= 0.10)
    if np.count_nonzero(mask) < 3:
        mask = (tail < 0.95) & (tail > 1e-6)
    if np.count_nonzero(mask) < 3:
        raise RuntimeError("device kernel did not expose an identifiable decay band")
    slope, _intercept = np.polyfit(elapsed[mask], np.log(tail[mask]), 1)
    matched_tau = -1.0 / float(slope)
    if not np.isfinite(matched_tau) or matched_tau <= 0:
        raise RuntimeError("post-peak exponential decay fit was not identifiable")
    return matched_tau


class TransientGate(CascadeEligibilityGate):
    """Deprecated compatibility alias for :class:`CascadeEligibilityGate`."""

    def __init__(self, *args, tau_d_override=None, **kwargs):
        warnings.warn(
            "TransientGate is deprecated; use CascadeEligibilityGate. "
            "tau_d_override never affected the eligibility readout and is ignored.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# =============================================================================
# Device-model FIT + SIM (the Chapter 5 provenance that produces the fitted laws
# this module's ``tau_r``/``tau_d``/``BETA``/``K_STAGES`` primitives report, and
# the habituation regime demonstration).  Each ``run_*``/``fit_*``/``simulate_*``
# core is SERIAL, takes its inputs as kwargs, and RETURNS a plain dict with NO
# file I/O, NO plotting, NO stdout -- exactly what a notebook calls in-kernel.
# ``main()`` (below) is the full-scale driver: it runs the cores and writes the
# fixtures under ``data/device_model/`` via ``paths``.
#
# Absorbed drivers (relocated verbatim in science; see per-function docstrings):
#   experiments/device_model/py_model/kww_final.py       -> fit_kww_laws() + main()
#   experiments/device_model/py_model/sim_habituation.py -> simulate_habituation() + main()
# =============================================================================

#: Bias magnitudes (V) of the measured gold-device traces the KWW law is fit to.
KWW_VOLTAGES = [0.8, 0.9, 1.1, 1.2, 1.4, 1.5]


def _kww_bias_from_name(basename: str) -> float:
    """Signed bias (V) parsed from a ``trace_V{m|p}<mag>_tr<n>.csv`` filename."""
    import re
    m = re.search(r"V([mp])(\d+(?:\.\d+)?)", basename)
    return (-1 if m.group(1) == "m" else 1) * float(m.group(2))


def _kww_r2(y, f):
    """Coefficient of determination of a fit ``f`` to data ``y``."""
    return 1 - np.sum((y - f) ** 2) / np.sum((y - np.mean(y)) ** 2)


def _kww_load_traces(export_dir, voltages=KWW_VOLTAGES, n_grid=300):
    """Load + per-bias-average the measured gold traces from ``export_dir``.

    Reads every ``trace_*.csv`` (columns ``time,current``), takes ``|current|``,
    keeps finite positive samples, groups by rounded ``|bias|``, and for each bias
    in ``voltages`` returns ``(t_grid, mean_I)`` interpolated onto a common grid
    truncated to the shortest trace at that bias.  Traces at biases outside
    ``voltages`` (e.g. the 1.7/1.8 V set) are loaded but not used, exactly as the
    original fitter did.
    """
    import glob
    import os
    byV = {}
    for c in sorted(glob.glob(os.path.join(str(export_dir), "trace_*.csv"))):
        arr = np.genfromtxt(c, delimiter=",", names=True)
        t = np.asarray(arr["time"], float)
        I = np.abs(np.asarray(arr["current"], float))
        m = np.isfinite(t) & np.isfinite(I) & (I > 0)
        key = round(abs(_kww_bias_from_name(os.path.basename(c))), 2)
        byV.setdefault(key, []).append((t[m], I[m]))
    data = {}
    for V in voltages:
        trs = byV[V]
        tmax = min(tr[0][-1] for tr in trs)
        tg = np.linspace(0, tmax, n_grid)
        data[V] = (tg, np.mean([np.interp(tg, t, I) for t, I in trs], axis=0))
    return data


def _kww_beta_to_k():
    """Approximate ``beta <-> k`` shape-matching bridge.

    Fits the KWW compressed-exponential rise ``1 - exp(-(t/tau_r)^beta)`` to the
    normalised Erlang-``k`` CDF for ``k = 1..5``; the returned ``{k: beta}`` map is
    obtained by fitting one family to the other. Thus ``beta ~ 2`` is represented
    compactly by ``k ~ 3``; it does not uniquely observe three microscopic stages.
    """
    from scipy.optimize import curve_fit
    from scipy.special import gammainc
    t = np.linspace(0, 10, 500)
    out = {}
    for k in range(1, 6):
        rise = gammainc(k, t)
        rise = rise / rise[-1]
        p, _ = curve_fit(lambda t, tr, b: 1 - np.exp(-(t / abs(tr)) ** abs(b)),
                         t, rise, p0=[3, 1.5], maxfev=20000)
        out[k] = round(abs(p[1]), 2)
    return out


def _kww_fit_global(data, beta, voltages=KWW_VOLTAGES):
    """Global shape fit: field-accelerated ``tau_r``/``tau_d`` exp laws, ``beta``
    fixed (dispersion bias-independent), per-voltage amplitude ``A`` + offset ``C``.

    Model (compact closed form):
        I(t) = A(V) [1 - exp(-(t/tau_r(V))^beta)] exp(-t/tau_d(V)) + C(V)
    with ``tau_r(V) = tr0 exp(-cr V)`` and ``tau_d(V) = td0 exp(-cd V)``.

    Returns ``(laws, rows)`` where ``laws`` holds ``beta, tr0, cr, td0, cd`` and
    ``rows[V]`` holds the per-voltage ``R2, A, C, tau_r, tau_d``.
    """
    from scipy.optimize import least_squares

    def shape(V, th, t):
        tr0, cr, td0, cd = th
        tr = abs(tr0) * np.exp(-cr * V)
        td = abs(td0) * np.exp(-cd * V)
        return (1 - np.exp(-(t / tr) ** beta)) * np.exp(-t / td)

    def resid(th):
        ch = []
        for V in voltages:
            tg, Ia = data[V]
            s = shape(V, th, tg)
            M = np.vstack([s, np.ones_like(s)]).T
            coef, *_ = np.linalg.lstsq(M, Ia, rcond=None)
            ch.append((M @ coef - Ia) / Ia.max())
        return np.concatenate(ch)

    r = least_squares(resid, [150, 2.9, 11000, 3.8], method="trf", max_nfev=8000)
    tr0, cr, td0, cd = r.x
    rows = {}
    for V in voltages:
        tg, Ia = data[V]
        s = shape(V, r.x, tg)
        M = np.vstack([s, np.ones_like(s)]).T
        coef, *_ = np.linalg.lstsq(M, Ia, rcond=None)
        rows[V] = dict(R2=_kww_r2(Ia, M @ coef), A=float(coef[0]), C=float(coef[1]),
                       tau_r=abs(tr0) * np.exp(-cr * V), tau_d=abs(td0) * np.exp(-cd * V))
    return dict(beta=beta, tr0=abs(tr0), cr=cr, td0=abs(td0), cd=cd), rows


def fit_kww_laws(export_dir=None, *, beta=BETA, voltages=KWW_VOLTAGES):
    r"""Lock the final KWW dispersive-kernel coefficients for the Chapter 5 Sec.5.3.2
    rewrite, and the ``beta <-> sequential-stage-count (k)`` bridge (absorbed from
    ``experiments/device_model/py_model/kww_final.py``).

    Model (compact closed form)::

        I(t) = A(V) [1 - exp(-(t/tau_r(V))^beta)] exp(-t/tau_d(V)) + C(V)

    ``beta``   compression exponent, HELD CONSTANT across bias (dispersion is
               bias-independent);
    ``tau_r``  rise time constant, field-accelerated ``tr0 exp(-cr V)``;
    ``tau_d``  decay time constant, field-accelerated ``td0 exp(-cd V)``;
    ``A, C``   per-voltage amplitude/offset (each trace sets its own scale, as the
               published model does).

    Mechanistic representation: a cascade of ``k`` sequential first-order
    trap-filling steps (Erlang-``k`` rise) can approximate the KWW rise. The
    ``beta<->k`` map is a fitted shape match, so ``beta ~ 2`` motivates ``k ~ 3``
    without uniquely identifying a microscopic stage count.
    These are the very laws this module's :func:`tau_r`/:func:`tau_d` report.

    Reads the measured gold traces from ``export_dir`` (defaults to
    ``paths.gold_export_dir()``); performs NO file I/O of its own.  Returns the plain
    dict later serialised to ``kww_final.json`` -- ``{"laws", "beta_to_k", "rows"}``.
    """
    if export_dir is None:
        from . import paths
        export_dir = paths.gold_export_dir()
    data = _kww_load_traces(export_dir, voltages=voltages)
    laws, rows = _kww_fit_global(data, beta, voltages=voltages)
    bk = _kww_beta_to_k()
    return {"laws": laws, "beta_to_k": bk,
            "rows": {f"{V}": rows[V] for V in voltages},
            "method_provenance": KWW_METHOD_PROVENANCE,
            "cascade_interpretation": CASCADE_METHOD_PROVENANCE}


# --- habituation regime simulation (Fig 5.15) hyperparameters ---------------
# See simulate_habituation() docstring for the full mean-field rationale.  These
# place the model in the depression-dominated regime; the qualitative result
# (settle-low at 10 Hz, recover at 1 Hz) is robust across the ranges given, not
# tuned to a single point.
HABIT_K = 3            #: trap-cascade stages (from the beta~2 Erlang-k fit, = K_STAGES)
HABIT_A = 3.0          #: cascade charging gain (LARGE -> fast-saturating, rate-insensitive baseline)
HABIT_TL_TRAP = 8.0    #: trap leak time constant (s) ~ fitted tau_r at mid-bias (~8 s at 1.0 V)
HABIT_SCR = 0.05       #: space-charge charging rate per unit drive (SLOW -> integrates the rate)
HABIT_SCRELAX = 0.06   #: space-charge relaxation rate (~tens of s, the tau_d regime)
HABIT_DEPTH = 0.85     #: space-charge suppression depth (<1 -> partial, as in Fig 4l)
HABIT_VNMAX = 2.0      #: saturated trap/space-charge occupancy bound (arbitrary units)


def _habit_rate_protocol(t):
    """Ch4 Fig 4l spike-rate protocol (Hz): 1 Hz, then 10 Hz on [108,168) s, then 1 Hz."""
    return 10.0 if (108 <= t < 168) else 1.0


def simulate_habituation(*, k=HABIT_K, a=HABIT_A, tl_trap=HABIT_TL_TRAP,
                         scr=HABIT_SCR, screlax=HABIT_SCRELAX, depth=HABIT_DEPTH,
                         vnmax=HABIT_VNMAX, t_end=276.0, n_grid=4000):
    """Reproduce the Fig 5.15 habituation demonstration (absorbed from
    ``experiments/device_model/py_model/sim_habituation.py``).

    The full two-branch dispersive model -- a fast-saturating trap cascade plus a
    slow space-charge depression branch -- is driven by the Chapter 4 Fig 4l spike-
    rate protocol (1 Hz -> 10 Hz -> 1 Hz) in the depression-dominated regime, to show
    the model reproduces the measured rate habituation.

    IMPORTANT: this is a DEMONSTRATION of the model's regime behaviour, NOT a fit to
    the Fig 4l data (that measurement is on a different device family and the raw
    envelope was not re-digitised).  The default hyperparameters place the model in
    the depression-dominated regime and are reported in full so the figure is
    reproducible; the qualitative result is robust across the ranges in the module
    constants, not tuned to a single point.

    Mean-field (rate-driven) rationale: the device timescales (seconds to minutes)
    are far slower than the inter-spike interval (0.1-1 s), so the device integrates
    the RATE -- exactly the quantity that drives habituation -- and resolving
    individual ms spikes would change nothing while costing ~1000x more compute.  The
    fast-saturating trap branch (large ``a``) sets a rate-insensitive baseline; the
    slow space-charge branch (small ``scr``) integrates the rate: it cannot relax
    between closely-spaced 10 Hz spikes (accumulates -> suppresses) but does relax
    between sparse 1 Hz spikes (-> recovers).  This rate separation is the entire
    physical mechanism.  Net current ``I = trap_occupancy (1 - depth * space_charge)``.

    SERIAL, no file I/O: returns the plain dict later saved to ``habit_data.npz``
    (``tg, I, f, xk, s``) plus diagnostic ``I/Imax`` checkpoints and a pass flag.
    """
    from scipy.integrate import solve_ivp

    def rhs(t, y):
        x = y[:k]
        s = y[k]
        f = _habit_rate_protocol(t)
        dx = np.zeros(k)
        dx[0] = a * f * (vnmax - x[0]) - x[0] / tl_trap
        for i in range(1, k):
            dx[i] = a * f * (x[i - 1] / vnmax) * (vnmax - x[i]) - x[i] / tl_trap
        ds = scr * f * (1 - s) - screlax * s     # space charge integrates the rate
        return list(dx) + [ds]

    tg = np.linspace(0, t_end, n_grid)
    sol = solve_ivp(rhs, (0, t_end), [0] * k + [0], t_eval=tg,
                    method="LSODA", rtol=1e-7, atol=1e-10)
    xk = sol.y[k - 1]
    s = sol.y[k]
    I = xk * (1 - depth * s)                      # net current: trap minus depression
    f = np.array([_habit_rate_protocol(t) for t in tg])

    def at(tt):
        return I[np.argmin(np.abs(tg - tt))] / I.max()

    checks = {"settle_1Hz_t100": float(at(100)), "end_10Hz_t165": float(at(165)),
              "recover_t270": float(at(270))}
    reproduced = bool(at(165) < at(100) * 0.9 and at(270) > at(165) * 1.1)
    return {"tg": tg, "I": I, "f": f, "xk": xk, "s": s,
            "checks": checks, "reproduced": reproduced,
            "method_provenance": {
                "status": "adapted",
                "established_basis": ["coupled trap and space-charge rate equations"],
                "repository_adaptation": (
                    "A mean-field pulse-rate protocol is simulated using separately "
                    "active trap and screening states."
                ),
                "claim_limit": (
                    "This qualitative simulation is not an identified microscopic "
                    "mechanism or a learning-result payload."
                ),
            }}


def main(argv=None):
    r"""Full-scale device-model driver: refit the KWW laws and rerun the habituation
    demonstration, writing the Chapter 5 fixtures under ``data/device_model/``.

    ``python -m mrl_trace.device [--kww] [--habituation] [--quick|--full]``
    With no experiment flag, runs both.  ``--kww`` writes ``kww_final.json`` (the
    fitted field-acceleration coefficients + ``beta<->k`` bridge); ``--habituation``
    writes ``habit_data.npz`` (the Fig 5.15 traces).  ``--quick`` uses a coarser ODE
    grid for the habituation sim; the KWW fit is already fast and is unchanged by the
    flag (it reads the fixed measured-trace set either way).
    """
    import argparse
    import json
    from . import paths
    ap = argparse.ArgumentParser(description="SiOx device-model fit + habituation sim")
    ap.add_argument("--kww", action="store_true",
                    help="refit KWW laws -> data/device_model/kww_final.json")
    ap.add_argument("--habituation", action="store_true",
                    help="rerun habituation sim -> data/device_model/habit_data.npz")
    ap.add_argument("--quick", action="store_true", help="coarser ODE grid (habituation)")
    ap.add_argument("--full", action="store_true", help="published grid (default)")
    a = ap.parse_args(argv)
    run_all = not (a.kww or a.habituation)

    if a.kww or run_all:
        print("=== KWW dispersive-kernel fit (Chapter 5 Table 5.2) ===")
        res = fit_kww_laws()
        laws, rows = res["laws"], res["rows"]
        print(f"FINAL KWW global law (beta fixed = {laws['beta']}):")
        print(f"  tau_r(V) = {laws['tr0']:.1f} * exp(-{laws['cr']:.2f} V)  [s]")
        print(f"  tau_d(V) = {laws['td0']:.0f} * exp(-{laws['cd']:.2f} V)  [s]")
        print(f"  beta<->stages shape match: {res['beta_to_k']}  "
              "(beta~2 is represented by k~3; exact depth is not identified)")
        print(f"{'V':>5s} {'R2':>7s} {'tau_r':>8s} {'tau_d':>9s} {'A(nA)':>8s}")
        r2s = []
        for V in KWW_VOLTAGES:
            d = rows[f"{V}"]
            r2s.append(d["R2"])
            print(f"{V:5.2f} {d['R2']:7.3f} {d['tau_r']:8.2f} {d['tau_d']:9.1f} {d['A']*1e9:8.1f}")
        print(f"median R2 = {np.median(r2s):.3f}  min = {min(r2s):.3f}")
        out = paths.device_model_dir() / "kww_final.json"
        json.dump(res, open(out, "w"), indent=2)
        print(f"  saved {out}")

    if a.habituation or run_all:
        n_grid = 800 if a.quick else 4000
        print(f"=== habituation regime demonstration (Fig 5.15, n_grid={n_grid}) ===")
        h = simulate_habituation(n_grid=n_grid)
        out = paths.device_model_dir() / "habit_data.npz"
        np.savez(out, tg=h["tg"], I=h["I"], f=h["f"], xk=h["xk"], s=h["s"])
        c = h["checks"]
        print(f"  I/Imax: 1Hz-settle(t=100)={c['settle_1Hz_t100']:.2f}  "
              f"10Hz-end(t=165)={c['end_10Hz_t165']:.2f}  "
              f"recover(t=270)={c['recover_t270']:.2f}")
        print("  habituation reproduced" if h["reproduced"] else "  CHECK params")
        print(f"  saved {out}")


if __name__ == "__main__":
    main()
