r"""mrl-trace: SiO\ :sub:`x` transients as physical eligibility surrogates.

Reference implementation for the manuscript *"Memristive transients as physical
eligibility traces for all-local reinforcement learning"*.  The package
exposes, as small composable pieces:

- :mod:`mrl_trace.device`   -- the empirical current fit and separate
  ``CascadeEligibilityGate`` computational surrogate;
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
- :mod:`mrl_trace.paths`    -- resolves compatible local fixtures and opt-in result
  archives while rejecting scientifically withdrawn archive names.

The ``experiments/`` directory contains notebook-owned live analyses. Corrected
publication-scale statistics require the declared raw fixtures and external seed
archive; they are not bundled or silently reconstructed from historical grids.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("mrl-trace")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from .device import (
    CascadeEligibilityGate,
    TransientGate,
    tau_r,
    tau_d,
    BETA,
    K_STAGES,
    fit_kww_laws,
    simulate_habituation,
    KWW_VOLTAGES,
    CASCADE_METHOD_PROVENANCE,
    KWW_METHOD_PROVENANCE,
)
from .neurons import lif_step, lif_step_batched
from .learning import (signed_coincidence, unsigned_coincidence, coincidence_drive,
                       three_factor_update, RewardBaseline, SIGNED_RULE_PROVENANCE,
                       THREE_FACTOR_PROVENANCE)
from .bandit import (
    GateBankBatched,
    AbstractTrace,
    train,
    reward_rate,
    trials_to_criterion,
    W_INIT,
    W_MAX,
    run_reversal,
    run_reversal_grid,
    calibrate_reversal_scales,
    run_learning_and_window,
    run_scaling,
    run_remedies,
    run_signed_rule_ablations,
    BANDIT_METHOD_PROVENANCE,
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
    ActionSequenceTrack,
    DelayedCuedChoice,
    TMaze,
    ConventionalRstdpTrace,
    ShallowEpropPolicyTrace,
    EpropTrace,
    calibrate_comparator_scales,
    tune_comparator_learning_rates,
    run_action_sequence,
    run_delayed_cued_choice,
    run_sequential,
    run_retention_delay_curve,
    run_dmax_law,
    run_retention_delay_curve_adaptive,
    run_dmax_adaptive,
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
    DEEP_METHOD_PROVENANCE,
)
from .distal_cue import (run_distal_cue, main as distal_cue_main,
                         DISTAL_CUE_METHOD_PROVENANCE)
from .probselect import (
    run_probselect,
    PST_HP,
    PST_HOMEO,
    PST_SEEDS,
    PST_TRIALS,
    PST_CHANCE,
    PST_CRIT,
    PST_METHOD_PROVENANCE,
)
from .dopamine import (load_session as load_dopamine_session, pair_cue_events,
                       reward_aligned_epochs, reject_legacy_capstone)
from .biosignal import (
    build_reward_pools,
    run_biosignal_reward,
    run_eeg_capstone,
    EEG_DATA_DEFAULT,
    EEG_POOLS_CACHE,
    EEG_METHOD_PROVENANCE,
)
from .selectivity import (
    run_interval_selectivity,
    run_vector_timer,
    peak_lag,
    selectivity_ratio,
    aliasing_pair,
    run_interval,
    SELECTIVITY_METHOD_PROVENANCE,
    PREDICTIVE_DELAY_GRID,
)
from .extensions import (
    run_multitimescale,
    run_wm_stc,
    wm_isolated,
    run_device_td,
    run_beta_sensitivity,
    load_measured_tau,
    CHANCE,
    CRIT,
    TAU_BAND,
    BETAS,
    V_INIT,
    V_MAX,
    ETA_V,
    EXTENSIONS_METHOD_PROVENANCE,
    RETENTION_DELAY_METHOD_PROVENANCE,
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
    HYBRID_METHOD_PROVENANCE,
)
from .stats import bootstrap_ci, summarise
from . import paths
from .paths import data_dir, results_dir, device_model_dir, save_result, load_result

__all__ = [
    "__version__",
    # device
    "CascadeEligibilityGate",
    "TransientGate",
    "tau_r",
    "tau_d",
    "BETA",
    "K_STAGES",
    "fit_kww_laws",
    "simulate_habituation",
    "KWW_VOLTAGES",
    "CASCADE_METHOD_PROVENANCE",
    "KWW_METHOD_PROVENANCE",
    # neurons
    "lif_step",
    "lif_step_batched",
    # learning
    "signed_coincidence",
    "unsigned_coincidence",
    "coincidence_drive",
    "three_factor_update",
    "RewardBaseline",
    "SIGNED_RULE_PROVENANCE",
    "THREE_FACTOR_PROVENANCE",
    # bandit / RL task
    "GateBankBatched",
    "AbstractTrace",
    "train",
    "reward_rate",
    "trials_to_criterion",
    "W_INIT",
    "W_MAX",
    "run_reversal",
    "run_reversal_grid",
    "calibrate_reversal_scales",
    "run_learning_and_window",
    "run_scaling",
    "run_remedies",
    "run_signed_rule_ablations",
    "BANDIT_METHOD_PROVENANCE",
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
    # multi-decision and historical delayed-choice tasks
    "ActionSequenceTrack",
    "DelayedCuedChoice",
    "ConventionalRstdpTrace",
    "ShallowEpropPolicyTrace",
    "calibrate_comparator_scales",
    "tune_comparator_learning_rates",
    "run_action_sequence",
    "run_delayed_cued_choice",
    "run_retention_delay_curve",
    "run_retention_delay_curve_adaptive",
    # deprecated compatibility names
    "TMaze",
    "EpropTrace",
    "run_sequential",
    "run_dmax_law",
    "run_dmax_adaptive",
    # additional distal-credit tasks
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
    "DEEP_METHOD_PROVENANCE",
    # distal-cue working-memory task
    "run_distal_cue",
    "distal_cue_main",
    "DISTAL_CUE_METHOD_PROVENANCE",
    # probabilistic selection (choose-A / avoid-B)
    "run_probselect",
    "PST_HP",
    "PST_HOMEO",
    "PST_SEEDS",
    "PST_TRIALS",
    "PST_CHANCE",
    "PST_CRIT",
    "PST_METHOD_PROVENANCE",
    # validated descriptive dopamine analysis (no omission/learning inference)
    "load_dopamine_session",
    "pair_cue_events",
    "reward_aligned_epochs",
    "reject_legacy_capstone",
    # biosignal reward (EEG capstone)
    "build_reward_pools",
    "run_biosignal_reward",
    "run_eeg_capstone",
    "EEG_DATA_DEFAULT",
    "EEG_POOLS_CACHE",
    "EEG_METHOD_PROVENANCE",
    # interval selectivity / vector timer
    "run_interval_selectivity",
    "run_vector_timer",
    "peak_lag",
    "selectivity_ratio",
    "aliasing_pair",
    "run_interval",
    "SELECTIVITY_METHOD_PROVENANCE",
    "PREDICTIVE_DELAY_GRID",
    # extensions (multi-timescale, WM+STC, device-TD, beta sensitivity)
    "run_multitimescale",
    "run_wm_stc",
    "wm_isolated",
    "run_device_td",
    "run_beta_sensitivity",
    "load_measured_tau",
    "CHANCE",
    "CRIT",
    "TAU_BAND",
    "BETAS",
    "V_INIT",
    "V_MAX",
    "ETA_V",
    "EXTENSIONS_METHOD_PROVENANCE",
    "RETENTION_DELAY_METHOD_PROVENANCE",
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
    "HYBRID_METHOD_PROVENANCE",
    # statistics
    "bootstrap_ci",
    "summarise",
    # paths / optional fixture and archive resolution
    "paths",
    "data_dir",
    "results_dir",
    "device_model_dir",
    "save_result",
    "load_result",
]
