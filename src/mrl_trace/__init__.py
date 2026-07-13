r"""mrl-trace: a SiO\ :sub:`x` device transient as a hardware eligibility trace.

Reference implementation for the manuscript *"A Memristive Device Transient as a
Hardware Eligibility Trace for Three-Factor Reinforcement Learning"* (APL Machine
Learning).  The package exposes, as small composable pieces:

- :mod:`mrl_trace.device`   -- the fitted SiO\ :sub:`x` transient and the
  ``TransientGate`` eligibility-trace generator (fitted device physics);
- :mod:`mrl_trace.neurons`  -- LIF neuron updates (scalar and batched);
- :mod:`mrl_trace.learning` -- the signed coincidence kernel and the
  three-factor reward-modulated update ``dw = eta (R - b) e``;
- :mod:`mrl_trace.bandit`   -- the contextual spiking bandit (the closed-loop
  RL task) with the device gate, an abstract-trace control, and a no-trace control;
- :mod:`mrl_trace.selectivity` -- the interval-selectivity and vector-timer
  tasks that read out the transient's tuning to inter-cue delay;
- :mod:`mrl_trace.extensions`  -- multi-timescale, working-memory + short-term
  consolidation, device-TD and beta-sensitivity extension studies;
- :mod:`mrl_trace.hybrid`   -- the hybrid vision front-end feeding the spiking
  RL readout (raw-pixel orientation task);
- :mod:`mrl_trace.paths`    -- resolves the bundled ``data/`` result grids and
  device-model fixtures (one source of truth for notebooks and ``python -m`` runs).

The ``experiments/`` directory reproduces every quantitative figure in the paper as
notebooks that call this package's ``run_*`` cores; the result grids live in ``data/``.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mrl-trace")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .device import (
    TransientGate,
    tau_r,
    tau_d,
    BETA,
    K_STAGES,
    fit_kww_laws,
    simulate_habituation,
    KWW_VOLTAGES,
)
from .neurons import lif_step, lif_step_batched
from .learning import signed_coincidence, three_factor_update, RewardBaseline
from .bandit import (
    GateBankBatched,
    AbstractTrace,
    train,
    reward_rate,
    trials_to_criterion,
    W_INIT,
    W_MAX,
    run_reversal,
    run_learning_and_window,
    run_scaling,
    run_remedies,
)
from .distal_reward import (
    trace_kernel,
    abstract_kernel,
    train_trace_level,
    SpikingGateBank,
    train_spiking,
    cue_saturation,
    run_trace_window,
    run_spiking_saturation,
    trace_ratio,
)
from .maze import (
    run_sequential,
    run_dmax_law,
    run_long_horizon,
    run_long_horizon_faults,
    running_rate,
    interp_dmax,
)
from .deep import (
    run_deep_local,
    run_deep_dms,
    run_array_scale,
    run_dms,
    run_dms_all,
    final_rate,
    main,
)
from .distal_cue import run_distal_cue, main as distal_cue_main
from .probselect import (
    run_probselect,
    PST_HP,
    PST_HOMEO,
    PST_SEEDS,
    PST_TRIALS,
    PST_CHANCE,
    PST_CRIT,
)
from .dopamine import (
    run_dopamine_shallow,
    run_dopamine_deep,
    make_shuffled_pools,
    HP,
    HOMEO,
    REW,
)
from .biosignal import (
    build_reward_pools,
    run_biosignal_reward,
    run_eeg_capstone,
    EEG_DATA_DEFAULT,
    EEG_POOLS_CACHE,
)
from .selectivity import (
    run_interval_selectivity,
    run_vector_timer,
    peak_lag,
    selectivity_ratio,
    aliasing_pair,
    run_interval,
)
from .extensions import (
    run_multitimescale,
    run_wm_stc,
    wm_isolated,
    run_device_td,
    run_beta_sensitivity,
    dmax_law,
    load_measured_tau,
    CHANCE,
    CRIT,
    TAU_BAND,
    BETAS,
    V_INIT,
    V_MAX,
    ETA_V,
)
from .hybrid import (
    IMG,
    C,
    NOISE,
    make_grating,
    make_batch,
    run_frontend,
    run_readout_pool,
    run_hybrid_decision,
    prep_readouts,
    run_hybrid_scale,
    final_rate as hybrid_final_rate,
)
from .stats import bootstrap_ci, summarise
from . import paths
from .paths import data_dir, results_dir, device_model_dir, save_result, load_result

__all__ = [
    "__version__",
    # device
    "TransientGate",
    "tau_r",
    "tau_d",
    "BETA",
    "K_STAGES",
    "fit_kww_laws",
    "simulate_habituation",
    "KWW_VOLTAGES",
    # neurons
    "lif_step",
    "lif_step_batched",
    # learning
    "signed_coincidence",
    "three_factor_update",
    "RewardBaseline",
    # bandit / RL task
    "GateBankBatched",
    "AbstractTrace",
    "train",
    "reward_rate",
    "trials_to_criterion",
    "W_INIT",
    "W_MAX",
    "run_reversal",
    "run_learning_and_window",
    "run_scaling",
    "run_remedies",
    # distal-reward credit-assignment tasks (Figs 4, 5)
    "trace_kernel",
    "abstract_kernel",
    "train_trace_level",
    "SpikingGateBank",
    "train_spiking",
    "cue_saturation",
    "run_trace_window",
    "run_spiking_saturation",
    "trace_ratio",
    # sequential-maze / distal-credit tasks (D_max retention law, long horizon)
    "run_sequential",
    "run_dmax_law",
    "run_long_horizon",
    "run_long_horizon_faults",
    "running_rate",
    "interp_dmax",
    # deep / multi-layer crossbar tasks (deep-local, DMS, array scaling)
    "run_deep_local",
    "run_deep_dms",
    "run_array_scale",
    "run_dms",
    "run_dms_all",
    "final_rate",
    "main",
    # distal-cue working-memory task
    "run_distal_cue",
    "distal_cue_main",
    # probabilistic selection (choose-A / avoid-B)
    "run_probselect",
    "PST_HP",
    "PST_HOMEO",
    "PST_SEEDS",
    "PST_TRIALS",
    "PST_CHANCE",
    "PST_CRIT",
    # measured-dopamine reward (Jeong 2022, DANDI 000351)
    "run_dopamine_shallow",
    "run_dopamine_deep",
    "make_shuffled_pools",
    "HP",
    "HOMEO",
    "REW",
    # biosignal reward (EEG capstone)
    "build_reward_pools",
    "run_biosignal_reward",
    "run_eeg_capstone",
    "EEG_DATA_DEFAULT",
    "EEG_POOLS_CACHE",
    # interval selectivity / vector timer
    "run_interval_selectivity",
    "run_vector_timer",
    "peak_lag",
    "selectivity_ratio",
    "aliasing_pair",
    "run_interval",
    # extensions (multi-timescale, WM+STC, device-TD, beta sensitivity)
    "run_multitimescale",
    "run_wm_stc",
    "wm_isolated",
    "run_device_td",
    "run_beta_sensitivity",
    "dmax_law",
    "load_measured_tau",
    "CHANCE",
    "CRIT",
    "TAU_BAND",
    "BETAS",
    "V_INIT",
    "V_MAX",
    "ETA_V",
    # hybrid vision front-end + RL readout
    "IMG",
    "C",
    "NOISE",
    "make_grating",
    "make_batch",
    "run_frontend",
    "run_readout_pool",
    "run_hybrid_decision",
    "prep_readouts",
    "run_hybrid_scale",
    "hybrid_final_rate",
    # statistics
    "bootstrap_ci",
    "summarise",
    # paths / data-grid resolution (single source of truth for data/)
    "paths",
    "data_dir",
    "results_dir",
    "device_model_dir",
    "save_result",
    "load_result",
]
