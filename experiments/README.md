# Publication reproduction notebooks

These notebooks form the lightweight, examiner-facing reproduction path for the
MRL manuscript. They display every figure inline and do not write image files by
default.

- `00_device_physics_and_trace.ipynb`: measured transients, device physics, trace
  shapes, ITO summaries, and timescales.
- `01_distal_credit_ladder.ipynb`: four authored schematic generators
  plus live trace-level, full-spiking, and closed-loop credit experiments.
- `02_sequential_and_scaling.ipynb`: retention-delay scaling, sequential credit,
  array-size/remedy sweeps, and hybrid orientation.
- `03_deep_local_and_faults.ipynb`: distal-cue, deep-local, delayed-match,
  homeostasis, crossbar, array scaling, and fault panels.
- `04_biological_grounding.ipynb`: a live Frank-task reproduction plus genuine reduced
  DANDI dopamine and OpenNeuro EEG workflows.
- `05_extensions.ipynb`: interval, beta-sensitivity, long-horizon, reversal,
  multi-timescale, vector-timer, consolidation, and device-TD results.
- `REPRODUCE.ipynb`: validates the 41-figure manifest, fans lightweight notebooks
  into isolated children, and gives grid-heavy notebooks a spawn-safe worker pool.

All notebooks expose the same controls: `RUN_PROFILE`, `DEVICE`, `WORKERS`,
`SAVE_FIGURES`, `OUTPUT_DIR`, `OVERWRITE`, `RUN_EXTERNAL_DATA`, and
`ALLOW_DATA_DOWNLOADS`. Environment variables use the same names with an `MRL_`
prefix. The default reduced profile is offline. Core reference simulations remain
NumPy/CPU workloads; the Torch hybrid front end selects CUDA when available.

Published numerical summaries in `data/publication/aggregates.json` are audit
records, not fabricated seed arrays and not substitutes for the default live runs.
Exact visual targets are read-only QA references. External recordings are never
downloaded unless both external execution and downloads are enabled. For notebook 04,
install `.[repro,external]`, then set `MRL_RUN_EXTERNAL_DATA=1` and
`MRL_ALLOW_DATA_DOWNLOADS=1`. Its preparation cell range-streams the ten exact DANDI
000351 photometry/event sessions used by the recorded panel (skipping their large videos)
and downloads the ~96 MB OpenNeuro ds003474 participant `sub-001`. It then creates small
analysis caches and runs the reduced figures inline. Later executions reuse and verify
the local files; the full 98.5 GB/17.9 GB source datasets are not needed for this reduced
validation snapshot.

Numerical notebooks support `MRL_USE_ARCHIVED_RESULTS=1` where a matching archive
exists. The default remains live, and `publication` restores the larger authored
settings.

To save claimable figures, set `MRL_SAVE_FIGURES=1` and choose
`MRL_OUTPUT_DIR`. Existing files are protected unless `MRL_OVERWRITE=1`.
Read-only QA references, placeholders, and gated external panels are never written
as regenerated publication figures.
