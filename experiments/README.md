# Publication reproduction notebooks

These notebooks form the lightweight, examiner-facing reproduction path for the
MRL manuscript. They display every figure inline and do not write image files by
default.

- `00_device_physics_and_trace.ipynb`: measured transients, device physics, trace
  shapes, ITO summaries, and timescales.
- `01_distal_credit_ladder.ipynb`: four authored schematic generators
  plus live trace-level, full-spiking, and closed-loop credit experiments.
- `02_sequential_and_scaling.ipynb`: simulated retention-delay sensitivity, the
  four-decision `ActionSequenceTrack`, AULC-tuned calibrated comparators and rule ablations,
  array-size/remedy sweeps, and hybrid orientation.
- `03_deep_local_and_faults.ipynb`: distal-cue, deep-local, delayed-match,
  homeostasis, crossbar, array scaling, and fault panels.
- `04_biological_grounding.ipynb`: synthetic Frank learning, DANDI 001340
  photometry triage, and leakage-safe action-contingent logged dopamine replay.
- `05_extensions.ipynb`: interval, beta-sensitivity, long-horizon, reversal,
  multi-timescale, vector-timer, consolidation, and device-TD results.
- `06_nmi_predictive_linkage.ipynb`: authoritative replicate-level empirical-model comparison,
  direct-held-bias timing, cascade-representation sensitivity, simulated
  retention-delay and reversal-quantile analyses. Its complete orchestration and
  measured-data analysis are embedded in the notebook.
- `REPRODUCE.ipynb`: fans lightweight notebooks into isolated children and gives
  grid-heavy notebooks a spawn-safe worker pool. Figure ownership is validated
  inside each notebook rather than through an external manifest.

Analysis notebooks 00--06 expose the same controls: `RUN_PROFILE`, `DEVICE`, `WORKERS`,
`SAVE_FIGURES`, `OUTPUT_DIR`, `OVERWRITE`, `RUN_EXTERNAL_DATA`, and
`ALLOW_DATA_DOWNLOADS`. Environment variables use the same names with an `MRL_`
prefix. The default reduced profile makes no network request, but data-dependent
notebooks still require their declared restored fixtures and fail closed if those
inputs are absent. Core reference simulations remain NumPy/CPU workloads; the Torch
hybrid front end selects CUDA when available.

Historical numerical archives are not fabricated into seed arrays and are not
substitutes for corrected live runs. In particular, the DANDI 000351 dopamine-capstone,
transformed-retention and one-choice-task archives are evidentially incompatible.
External recordings are never
downloaded unless both external execution and downloads are enabled. For notebook 04,
install `.[repro,external]`, then set `MRL_RUN_EXTERNAL_DATA=1` and
`MRL_ALLOW_DATA_DOWNLOADS=1` only when downloading is intended. The workflow pins all
69 NWB assets from DANDI 001340 version `0.250221.0527`, verifies size and SHA-256,
and creates non-pickle state-free caches. Set `MRL_DANDI001340_RAW_DIR`,
`MRL_DANDI001340_CACHE_DIR`, and `MRL_DANDI001340_OUTPUT_DIR` to external locations.
Raw NWBs and generated caches remain untracked.

The 60 s detrending result is primary; 30/120 s are sensitivity reconstructions.
All 46 substantive sessions enter primary LOSO likelihood, while the predeclared
37-session separation subset is sensitivity-only. The verdict is **Conditional Go**:
replay feasibility, not biological learning or device-kernel superiority. Plain
dLight currently beats device weighting, the device result is inconclusive, and
outcome Q-learning is only a positive control. Recordings contain no counterfactual
dLight for actions the animal did not choose, so Frank/deep-XOR learning remains
synthetic. EEG helpers are descriptive and no decoded EEG value is resampled into
learning.

All notebooks that use a gate record a shared specification identity. The package
default is nonlinear `physical_headroom_v1`; HEAD's linear Erlang cascade is the
explicit `linear_erlang_v1` sensitivity. Notebook 06 evaluates KWW and both state-
space families at k=2--5 and writes a deterministic non-pickle predictive-linkage
manifest. Other notebooks call the specifications in `src/` directly and remain
runnable without notebook 06 or ITO data.

On the verified inputs, linear-Erlang k=3 is the best held-out Au representation
(NRMSE 0.0548), while the physical-headroom family is outside the 5% support set.
This keeps physical headroom as an equation-matching learning primary without an
empirically-preferred claim.
At the direct ITO retention quantiles, timing peaks are boundary-censored and the
retention--horizon relationship is not estimable; notebook 06 reports this as an
unresolved empirical linkage rather than promoting the internal kernel direction.

Sequential and shallow-DMS method rates are selected by mean unsmoothed pilot-seed
AULC only after fixed balanced RMS normalization, then evaluated on a disjoint seed
block. No-trace has eta zero. Reported device advantages require a wholly positive
paired device-minus-comparator AULC interval; otherwise the comparison is unresolved.
The primary sequential operating point is the synthetic 10 s retention, 12 s delay,
1.5 V, k=3, beta=1 configuration. Interval selectivity is the primary amplitude-
robust kernel-shape result and is checked across eta={0.01,0.1,0.5} for both models.
Tracked publication-scale JSON/CSV outputs and artifact-derived manuscript macros
live in `data/results/reference/`.

Numerical notebooks support `MRL_USE_ARCHIVED_RESULTS=1` where a matching archive
exists. The default remains live, and `publication` restores the larger authored
settings.

Notebook 06 runs automatically from `REPRODUCE.ipynb` in publication mode and can
be included in smoke/reduced reproduction with `MRL_RUN_PREDICTIVE_LINKAGE=1`.
It requires all 24 Au traces, the frozen 91-workbook January 2025 ITO cohort, and
the two declared June 2026 1 mV direct-read supplements, and fails closed when they
are missing. Other acquisitions may remain beside that cohort and receive an explicit
out-of-cohort disposition. Raw inputs are ignored by Git and excluded from package
artifacts; provide them through `MRL_TRACE_GOLD_DIR` and `MRL_TRACE_ITO_DIR`.
Publication mode requires an explicit `MRL_OUTPUT_DIR` and defaults both
`MRL_SAVE_RESULTS=1` and `MRL_SAVE_FIGURES=1`; disabling either fails closed. It may run locally or on any
CPU compute service, and its bulky per-seed archive belongs with the archival DOI.
The notebook's opening cells contain the exact laptop/cloud commands, workload table,
archive policy and compatibility record; no separate execution or audit document is
required.

Established components and repository adaptations are distinguished explicitly.
The signed coincidence rule, cascade eligibility surrogate, shallow e-prop-style
policy trace and combined DFA/device/homeostasis update are bespoke constructions;
their component citations do not establish the exact implemented update. Result
payloads carry a `method_provenance` record and the simulated retention-delay curve
is never presented as an independently validated physical law.

To save claimable figures, set `MRL_SAVE_FIGURES=1` and choose
`MRL_OUTPUT_DIR`. Existing files are protected unless `MRL_OVERWRITE=1`.
Read-only QA references, placeholders, and gated external panels are never written
as regenerated publication figures.
