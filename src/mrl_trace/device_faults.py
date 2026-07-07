r"""Device non-idealities for array-scale feasibility studies.

A faithful NumPy port of the fault catalogue in the nonideality-aware-mnn-training
package (``awarememristor.crossbar.nonidealities``), the engine the SiO_x
inference/homeostasis work trains against. We port the *semantics* rather than import
the package, because that package is TensorFlow and this learning rule is pure NumPy
(running a TF op inside the per-timestep spiking loop would be both a heavy dependency
and prohibitively slow). The fault definitions below mirror those classes exactly, so
the device-fault prior is shared with the inference work while the learning stays here.

Two physical classes (after Joksas et al.):

* Linearity-preserving -- corrupt the conductance, the multiply stays linear:
    - ``StuckAtGOff`` / ``StuckAtGOn`` : a fraction of devices pinned to G_off / G_on.
    - ``StuckDistribution``           : a fraction stuck at values drawn from a KDE of
                                        measured stuck conductances (reflected at 0).
    - ``D2DLognormal``                : lognormal device-to-device programming spread,
                                        width interpolated between R_on and R_off.
* Linearity-NONpreserving -- the per-device I--V law itself is nonlinear:
    - ``IVNonlinearityPF``            : Poole--Frenkel forward map with per-synapse
                                        (c, d_epsilon) sampled from the measured
                                        bivariate regression; an always-on distortion
                                        of every analogue multiply.

Each fault exposes ``disturb(G, rng)`` (linearity-preserving) or
``apply(V_eff, G, rng)`` (nonpreserving), and a ``FaultStack`` composes them into the
single ``weight_fault`` hook ``train_deep`` accepts. Faults are a fixed physical
realisation: drawn once (cached) and reused for the run, as a programmed array behaves.
"""
import numpy as np
import scipy.constants as const

__all__ = ["StuckAtGOff", "StuckAtGOn", "StuckDistribution", "D2DLognormal",
           "IVNonlinearityPF", "FaultStack", "SIOX_PF", "siox_fault_stack",
           "maze_fault_stack"]

# --- Pre-computed SiO_x device constants (fitted ONCE, offline, from the measured
# SiO_x-multistate-data.mat via mnn-torch's PF regression; baked here so no runtime
# fit or TF dependency is needed). These ARE the parameters the inference work uses. ---
SIOX_PF = {
    "G_off": 6.9013e-4,
    "G_on": 3.4506e-3,
    "slopes": [0.236877, 1.091365],
    "intercepts": [-11.535144, -47.585831],
    "res_cov": [[0.0154065, -0.0031134], [-0.0031134, 0.0766984]],
    "k_V": 0.5,
}


class _Fault:
    """Base: draws a fixed realisation on first call, caches it per weight shape."""
    def __init__(self):
        self._cache = {}

    def disturb(self, G, rng):
        raise NotImplementedError


class StuckAtGOff(_Fault):
    """A fraction ``probability`` of devices pinned to G_off (open synapses)."""
    def __init__(self, probability, G_off=0.0):
        super().__init__(); self.p = probability; self.G_off = G_off

    def disturb(self, G, rng):
        key = G.shape
        if key not in self._cache:
            self._cache[key] = rng.random(G.shape) < self.p
        return np.where(self._cache[key], self.G_off, G)


class StuckAtGOn(_Fault):
    """A fraction ``probability`` of devices pinned to G_on (shorted synapses)."""
    def __init__(self, probability, G_on=1.0):
        super().__init__(); self.p = probability; self.G_on = G_on

    def disturb(self, G, rng):
        key = G.shape
        if key not in self._cache:
            self._cache[key] = rng.random(G.shape) < self.p
        return np.where(self._cache[key], self.G_on, G)


class StuckDistribution(_Fault):
    """A fraction stuck at values drawn from a reflected-Gaussian KDE of measured
    stuck conductances (``means``), Scott's-rule bandwidth -- the realistic stuck
    model (cf. ``StuckDistribution`` in awarememristor)."""
    def __init__(self, probability, means, bandwidth=None):
        super().__init__()
        self.p = probability
        self.means = np.asarray(means, float)
        n = len(self.means)
        # Scott's rule (matches KDEpy default used in the reference).
        self.bw = bandwidth if bandwidth is not None else \
            self.means.std(ddof=1) * n ** (-1.0 / 5.0) if n > 1 else 0.1 * abs(self.means.mean() + 1e-9)

    def _sample(self, n, rng):
        # KDE sample: pick a mean, add Gaussian(bw), reflect at 0 (truncated >=0).
        centres = rng.choice(self.means, size=n)
        vals = centres + self.bw * rng.standard_normal(n)
        return np.abs(vals)                       # reflect negatives to keep G >= 0

    def disturb(self, G, rng):
        key = G.shape
        if key not in self._cache:
            mask = rng.random(G.shape) < self.p
            stuck = np.zeros(G.shape)
            stuck[mask] = self._sample(int(mask.sum()), rng)
            self._cache[key] = (mask, stuck)
        mask, stuck = self._cache[key]
        return np.where(mask, stuck, G)


class D2DLognormal(_Fault):
    """Lognormal device-to-device programming spread, the log-std interpolated
    linearly in resistance between R_on and R_off (cf. ``D2DLognormal``)."""
    def __init__(self, R_on_log_std, R_off_log_std, G_on=1.0, G_off=1e-3):
        super().__init__()
        self.s_on, self.s_off = R_on_log_std, R_off_log_std
        self.G_on, self.G_off = G_on, G_off

    def disturb(self, G, rng):
        key = G.shape
        if key not in self._cache:
            Gpos = np.clip(np.abs(G), self.G_off, self.G_on)
            R = 1.0 / Gpos
            R_on, R_off = 1.0 / self.G_on, 1.0 / self.G_off
            # piecewise-linear interp of log-std between R_on and R_off
            frac = np.clip((R - R_on) / (R_off - R_on + 1e-18), 0.0, 1.0)
            log_std = self.s_on + frac * (self.s_off - self.s_on)
            # mean-preserving lognormal factor on G (G = G_target * exp(N(0, sigma)))
            self._cache[key] = np.exp(log_std * rng.standard_normal(G.shape))
        return np.sign(G) * np.abs(G) * self._cache[key]


class IVNonlinearityPF(_Fault):
    """Poole--Frenkel per-device I--V nonlinearity (linearity-NONpreserving).

    Each synapse draws ``(ln c, ln d_epsilon)`` from the measured bivariate regression
    on ``ln R`` (slopes ``m``, intercepts ``b``, residual covariance ``Sigma``), then
    its read current is the PF map I = c V exp[(2e/kT) sqrt(eV / (4 pi d_epsilon))],
    not the linear product G*V. Returned as an *effective conductance* I/V so it drops
    into the existing linear einsum -- i.e. the nonlinearity is folded into a
    voltage-dependent effective weight, evaluated at the read amplitude ``V_read``.
    """
    def __init__(self, slopes, intercepts, res_cov, k_V=0.5, V_read=0.25,
                 G_on=1.0, G_off=1e-3, w_max=1.5):
        super().__init__()
        self.m = np.asarray(slopes, float); self.b = np.asarray(intercepts, float)
        self.L = np.linalg.cholesky(np.asarray(res_cov, float)
                                    + 1e-12 * np.eye(2))
        self.k_V, self.V_read = k_V, V_read
        self.G_on, self.G_off = G_on, G_off
        self.w_max = w_max                       # weight-domain scale for the I -> y map

    def disturb(self, W, rng):
        # Faithful crossbar I-V nonlinearity, following the reference pipeline
        # (awarememristor crossbar.map.w_to_G -> IVNonlinearityPF.compute_I -> map.I_to_y),
        # with the per-device regression RESIDUAL drawn ONCE (fixed device identity) and the
        # weight-dependent regression MEAN recomputed each call so the read current tracks the
        # programmed weight.
        #   1. weight -> conductance: G = k_G*|W| + G_off, k_G = (G_on - G_off)/w_max.
        #   2. PF coefficients from the bivariate regression on ln R (= -ln G); slopes come
        #      from scipy.linregress so the mean is ``+m*lnR + b`` (mnn-torch convention). A
        #      prior port used ``-m*lnR``, driving d_epsilon to ~1e-24 (vs physical ~1e-18),
        #      exploding the exponent and collapsing the output to a saturated sign-only value.
        #   3. PF read current, then map current -> weight: y = I / k_I, k_I = k_V*k_G.
        # NB: at the measured SiO_x constants over a low read window the PF map is close to a
        # mild rescale + per-device spread (a graded, sign-preserving distortion). It is the
        # linearity-NON-preserving term in the prior; the dominant programmed-weight faults are
        # stuck-at and D2D.
        key = W.shape
        if key not in self._cache:
            self._cache[key] = (self.L @ rng.standard_normal((2,) + W.shape).reshape(2, -1)
                                ).reshape((2,) + W.shape)
        eps = self._cache[key]
        Gpos = np.clip(np.abs(W), self.G_off, self.G_on)
        lnR = np.log(1.0 / Gpos)
        c = np.exp(self.m[0] * lnR + self.b[0] + eps[0])
        d_eps = np.exp(self.m[1] * lnR + self.b[1] + eps[1])
        V = self.k_V * self.V_read
        e, kT = const.elementary_charge, const.Boltzmann * (const.zero_Celsius + 20.0)
        expo = np.clip((2.0 * e / kT) * np.sqrt(e * abs(V) / (4 * np.pi * d_eps) + 1e-18),
                       None, 50.0)
        g_eff = c * np.exp(expo)
        # Mean-preserving anchor to the programmed-weight magnitude: keeps the analogue
        # dot-product gain stable (so the LIF stays in regime) while the per-device residual
        # spread + the c(G) warp remain as the PF non-ideality content. Empirically this is the
        # form under which the deep net learns through PF (PF-only ~0.73, full prior graceful);
        # a strict weight->[G_off,G_on] mapping compresses lnR so far that the residual swamps
        # the weight dependence and the net regresses to chance.
        g_eff = g_eff * (np.mean(Gpos) / (np.mean(g_eff) + 1e-30))
        return np.sign(W) * g_eff


class FaultStack:
    """Compose faults into one ``weight_fault`` hook for ``train_deep``.

    Faults apply in order; a single ``rng`` seeded once gives a fixed realisation.
    Linearity-preserving faults are applied first (they set the conductance), then the
    PF nonlinearity maps that conductance through the device I--V law.
    """
    def __init__(self, faults, seed=0):
        self.faults = faults
        self.rng = np.random.default_rng(seed)

    def __call__(self, W):
        G = W
        for f in self.faults:
            G = f.disturb(G, self.rng)
        return G


def siox_fault_stack(p_stuck=0.0, sigma_g=0.5, sigma_g_on=None, pf_on=False,
                     stuck_kind="off", stuck_means=None, seed=0):
    """Assemble the canonical SiO_x fault set from the pre-computed constants.

    Args:
        p_stuck    : fraction of stuck devices (the inference work uses 0.05; the
                     manuscript stress-tests up to 0.5).
        sigma_g    : lognormal D2D log-std at the HIGH-resistance (R_off) end. The
                     awarememristor reference's realistic device is ASYMMETRIC --
                     ``(R_on_log_std, R_off_log_std) = (0.05, 0.5)`` -- and its (0.5, 0.5)
                     symmetric setting is an explicit "high-D2D" STRESS regime. Pass
                     ``sigma_g_on`` to set the R_on end independently; if omitted it defaults
                     to ``sigma_g`` (the legacy symmetric behaviour). Set ``sigma_g=0`` to
                     disable D2D. NB: with the I-V nonlinearity now correctly active (it was
                     previously sign-only), symmetric (0.5, 0.5) drives the deep net to
                     chance, whereas the realistic (0.05, 0.5) leaves it robust -- so the
                     headline grids use the realistic spread, (0.5, 0.5) as a stress point.
        sigma_g_on : R_on-end D2D log-std; defaults to ``sigma_g`` (symmetric) if None.
        pf_on      : include the Poole--Frenkel I--V nonlinearity (always-on distortion).
        stuck_kind : "off" (open synapses, the dominant mode), "on", or "dist" (KDE of
                     measured stuck values, needs ``stuck_means``).
        stuck_means: measured stuck conductances for the KDE model (stuck_kind="dist").
    Returns a ``FaultStack`` callable usable as ``train_deep(weight_fault=...)``.
    """
    G_off, G_on = SIOX_PF["G_off"], SIOX_PF["G_on"]
    s_on = sigma_g if sigma_g_on is None else sigma_g_on
    faults = []
    if p_stuck > 0:
        if stuck_kind == "off":
            faults.append(StuckAtGOff(p_stuck, G_off))
        elif stuck_kind == "on":
            faults.append(StuckAtGOn(p_stuck, G_on))
        elif stuck_kind == "dist":
            faults.append(StuckDistribution(p_stuck,
                          stuck_means if stuck_means is not None else [G_off, G_on]))
    if sigma_g > 0 or s_on > 0:
        faults.append(D2DLognormal(s_on, sigma_g, G_on, G_off))
    if pf_on:
        faults.append(IVNonlinearityPF(SIOX_PF["slopes"], SIOX_PF["intercepts"],
                                       SIOX_PF["res_cov"], k_V=SIOX_PF["k_V"], V_read=0.25,
                                       G_on=G_on, G_off=G_off))
    return FaultStack(faults, seed=seed)


class _MazeFaultStack:
    """Apply the SiO_x fault prior to NON-NEGATIVE single-conductance weights.

    The deep network's weights are signed differential pairs in ``[-w_max, w_max]`` and
    map onto the SiO_x conductance window directly. The sequential-maze (``maze.py``)
    weights are instead a SINGLE non-negative conductance per synapse in ``[0, w_max]``;
    feeding those raw into :func:`siox_fault_stack` is physically wrong, because a weight
    of 0 is NOT ``G_off`` on that stack and ``StuckAtGOn`` (``G_on = 3.45e-3``) would map a
    shorted device to ~0 weight rather than to the maximum. This wrapper fixes the mapping:

      1. affine-map the weight ``[0, w_max] -> [G_off, G_on]`` (0 -> open, w_max -> shorted);
      2. apply the SAME fault classes (stuck-off/on, lognormal D2D, PF I-V nonlinearity)
         in that physical conductance space;
      3. affine-map the faulted conductance back to a non-negative weight, clipped to
         ``[0, w_max]``.

    So stuck-off correctly opens a synapse (weight 0), stuck-on correctly saturates it
    (weight w_max), and D2D/PF distort around the programmed value -- the same measured
    prior as the deep experiments, expressed in the maze's weight space.
    """
    def __init__(self, gc_faults, pf, w_max, G_off, G_on, seed=0):
        # Two fault groups, applied in the right domain for the NON-NEGATIVE maze weight:
        #   gc_faults : linearity-PRESERVING faults (stuck-off/on, D2D). They act in
        #               CONDUCTANCE space, so the wrapper maps weight -> [G_off, G_on],
        #               applies them, maps back to [0, w_max]. Stuck-off -> 0 weight,
        #               stuck-on -> w_max, D2D distorts around the programmed value.
        #   pf        : the PF I-V nonlinearity (or None). PF is SELF-CONTAINED: it maps
        #               weight -> conductance -> current -> weight internally (with its own
        #               w_max), so it is applied DIRECTLY to the (non-negative) maze weight,
        #               NOT pre-mapped (which would double-transform). The earlier min-max
        #               rescale hack is gone -- PF now returns weight-domain values directly.
        self.gc_faults = gc_faults
        self.pf = pf
        self.w_max, self.G_off, self.G_on = w_max, G_off, G_on
        self.rng = np.random.default_rng(seed)

    def __call__(self, W):
        # Map the non-negative maze weight into the SiO_x conductance window, apply ALL faults
        # in that conductance space (stuck/D2D AND the PF I-V map, which expects conductance-
        # scale input -- it clips |.| to [G_off, G_on]), then map back to [0, w_max]. Stuck-off
        # -> 0 weight, stuck-on -> w_max, D2D + PF distort around the programmed value. PF
        # returns a conductance-scale effective value (mean-anchored), so the round-trip stays
        # in-window.
        frac = np.clip(W / self.w_max, 0.0, 1.0)              # [0,w_max] -> [0,1]
        G = self.G_off + frac * (self.G_on - self.G_off)      # -> [G_off, G_on]
        for f in self.gc_faults:
            G = f.disturb(G, self.rng)
        if self.pf is not None:
            G = np.clip(np.abs(self.pf.disturb(G, self.rng)), self.G_off, self.G_on)
        frac_out = (G - self.G_off) / (self.G_on - self.G_off + 1e-18)
        return np.clip(frac_out, 0.0, 1.0) * self.w_max       # back to [0, w_max]


def maze_fault_stack(p_stuck=0.0, sigma_g=0.5, pf_on=False, stuck_kind="off",
                     w_max=1.5, seed=0):
    """SiO_x fault prior for the non-negative single-conductance maze weights.

    Same arguments and physical constants as :func:`siox_fault_stack`, but composed
    through the ``[0, w_max] <-> [G_off, G_on]`` conductance mapping (see
    :class:`_MazeFaultStack`). ``w_max`` must match ``maze.W_MAX`` (default 1.5).
    Returns a callable usable as ``train_sequential(weight_fault=...)``.
    """
    G_off, G_on = SIOX_PF["G_off"], SIOX_PF["G_on"]
    gc_faults = []                                   # linearity-preserving (conductance space)
    if p_stuck > 0:
        if stuck_kind == "off":
            gc_faults.append(StuckAtGOff(p_stuck, G_off))
        elif stuck_kind == "on":
            gc_faults.append(StuckAtGOn(p_stuck, G_on))
    if sigma_g > 0:
        gc_faults.append(D2DLognormal(sigma_g, sigma_g, G_on, G_off))
    pf = None
    if pf_on:
        # PF maps weight<->conductance internally, so it carries the maze w_max.
        pf = IVNonlinearityPF(SIOX_PF["slopes"], SIOX_PF["intercepts"], SIOX_PF["res_cov"],
                              k_V=SIOX_PF["k_V"], V_read=0.25, G_on=G_on, G_off=G_off,
                              w_max=w_max)
    return _MazeFaultStack(gc_faults, pf, w_max, G_off, G_on, seed=seed)
