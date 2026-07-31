# Publication reproduction notebooks

These notebooks form the lightweight, examiner-facing reproduction path for the
MRL manuscript. They display every figure inline and do not write image files by
default.

- `00_device_physics_and_trace.ipynb`: measured transients, device physics, trace
  shapes, ITO summaries, and timescales.
- `01_distal_credit_ladder.ipynb`: four authored schematic generators
  plus live trace-level, full-spiking, and closed-loop credit experiments.
- `02_sequential_and_scaling.ipynb`: simulated retention-delay sensitivity, the
  four-decision `ActionSequenceTrack`, calibrated comparators and rule ablations,
  array-size/remedy sweeps, and hybrid orientation.
- `03_deep_local_and_faults.ipynb`: distal-cue, deep-local, delayed-match,
  homeostasis, crossbar, array scaling, and fault panels.
- `04_biological_grounding.ipynb`: Frank-style task metrics, descriptive
  reward-aligned DANDI photometry, and OpenNeuro EEG workflows. The DANDI event
  stream contains cue-onset/cue-offset pairs and no verified omission class.
- `05_extensions.ipynb`: interval, beta-sensitivity, long-horizon, reversal,
  multi-timescale, vector-timer, consolidation, and device-TD results.
- `06_nmi_predictive_linkage.ipynb`: replicate-level empirical-model comparison,
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
substitutes for corrected live runs. In particular, dopamine-capstone,
transformed-retention and one-choice-task archives are evidentially incompatible.
External recordings are never
downloaded unless both external execution and downloads are enabled. For notebook 04,
install `.[repro,external]`, then set `MRL_RUN_EXTERNAL_DATA=1` and
`MRL_ALLOW_DATA_DOWNLOADS=1`. Its preparation cell range-streams the ten exact DANDI
000351 photometry/event sessions used by the recorded panel (skipping their large videos)
and downloads the ~96 MB OpenNeuro ds003474 participant `sub-001`. It then creates small
analysis caches and runs the reduced figures inline. Dopamine processing validates
two sound markers per recorded reward and produces descriptive reward-aligned epochs
only---never inferred omissions, a reward decoder or a learning pool. Later executions
reuse and verify
the local files; the full 98.5 GB/17.9 GB source datasets are not needed for this reduced
validation snapshot.

Numerical notebooks support `MRL_USE_ARCHIVED_RESULTS=1` where a matching archive
exists. The default remains live, and `publication` restores the larger authored
settings.

Notebook 06 runs automatically from `REPRODUCE.ipynb` in publication mode and can
be included in smoke/reduced reproduction with `MRL_RUN_PREDICTIVE_LINKAGE=1`.
It requires all 24 Au traces and 91 ITO workbooks and fails closed when they are
missing. Publication mode requires an explicit `MRL_OUTPUT_DIR` and defaults both
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
