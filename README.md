# mrl-trace

`mrl-trace` is the reference implementation for *Memristive transients as
eligibility traces for all-local reinforcement learning*. It models a measured
SiO<sub>x</sub> device transient as a physical eligibility trace and connects it
to spiking, sequential, deep-local, and biologically grounded learning studies.

## Installation

Python 3.10 or newer is required. This repository installs independently and
does not require a sibling checkout or a pre-existing named environment.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[repro]"
```

The base install contains the NumPy/SciPy device and learning models. The
`repro` extra adds the notebook, plotting, analysis, and optional Torch hybrid
stack; `external` adds adapters used by explicitly enabled biological-data
workflows; `test` and `release` provide test and package-build tools.

```bash
python -m pip install -e ".[repro,external,test]"
```

The scientific reference simulations remain CPU-vectorised. Independent
conditions can use spawn-safe process workers, while Torch perception workloads
select CUDA when an appropriate PyTorch build is installed.

## Reproducing the figures

Start Jupyter in the repository root and run `experiments/REPRODUCE.ipynb`:

```bash
python -m jupyter lab experiments/REPRODUCE.ipynb
```

The examiner-readable topic notebooks are:

- `00_device_physics_and_trace.ipynb`: measured transient physics, trace shapes,
  electrode studies, and timescales.
- `01_distal_credit_ladder.ipynb`: three-factor architecture and the trace,
  spiking, and closed-loop distal-credit ladder.
- `02_sequential_and_scaling.ipynb`: retention-delay scaling, T-maze,
  array-size/remedy sweeps, and the CUDA-capable hybrid study.
- `03_deep_local_and_faults.ipynb`: shallow/deep local learning, homeostasis,
  crossbar schematics, array scaling, and fault studies.
- `04_biological_grounding.ipynb`: Frank-task and gated dopamine/EEG studies.
- `05_extensions.ipynb`: beta-sensitivity validation, long-horizon, reversal,
  multi-timescale, capture, working-memory, and device-TD studies.
- `06_nmi_predictive_linkage.ipynb`: self-contained replicate-level Au/ITO model
  identification and the new timing, empirical-retention, and reversal analyses.

Together they catalogue 41 unique first-party, generator-backed figures: ten
used by the accompanying manuscript, 17 supplementary or contextual figures,
and 14 device or outlook figures. Third-party artwork and unexplained legacy
rasters are outside the reproduction manifest.

Every notebook exposes the same controls near the top:

```python
RUN_PROFILE = "reduced"       # reduced | publication | smoke
DEVICE = "auto"               # CUDA, then MPS, then CPU
WORKERS = "auto"
SAVE_FIGURES = False
OUTPUT_DIR = ...
OVERWRITE = False
RUN_EXTERNAL_DATA = False
ALLOW_DATA_DOWNLOADS = False
```

- `reduced` computes every offline numerical panel live from the measured fixtures
  and scientific models with smaller seed/trial budgets. It preserves the
  published condition ordering and panel structure without claiming identical
  confidence intervals.
- `publication` restores the larger authored budgets and is intentionally
  computationally expensive.
- `smoke` runs a minimal offline validation workload.

The reproduction driver fans lightweight notebooks into isolated child
interpreters. Grid-heavy notebooks then run in canonical order with spawn-safe
cell-level pools. This avoids nested pools and gives the full worker budget to the
longest simulations. Torch hybrid work selects CUDA when the installed Torch build
provides it; NumPy reference algorithms correctly report CPU.

Figures always display inline. Saving is opt-in, existing outputs are protected,
and writing to a manuscript directory requires explicit overwrite permission.
Read-only reference rasters cannot be exported as regenerated evidence. Published
means and confidence intervals remain audit records; they are not expanded into
synthetic seed runs or used to replace the default live computations.

The numerical notebooks accept `MRL_USE_ARCHIVED_RESULTS=1` where a matching full
sweep archive exists. This is always explicit: merely finding an archive never
changes a live default, and archived sample counts are reported as stored.

## Predictive-linkage extension

The NMI-facing additions live in `experiments/06_nmi_predictive_linkage.ipynb`,
not in a second source-level publication framework. It calls the package's scientific
models directly, embeds the physical fitting workflow needed to audit the raw Au and
ITO files, and writes no results by default. Profiles are `smoke`, `reduced`, and
`publication`; only the last is intended for final inference.

Set `MRL_SAVE_RESULTS=1` and `MRL_OUTPUT_DIR` to save per-trace, per-seed, and summary
artifacts to an external archival directory. Full result arrays and generated figures
are intentionally ignored by Git. SageMaker is optional: the same notebook runs on a
laptop, workstation, cluster, or ordinary cloud VM. Exact commands, workload sizes,
archive policy and historical-run provenance are contained in the notebook itself.

## External biological data

Dopamine and EEG downloads are disabled by default. Install the `external`
extra, set `RUN_EXTERNAL_DATA=True`, and set `ALLOW_DATA_DOWNLOADS=True` only
after reviewing the dataset instructions in the owning notebook. Absent or
invalid caches leave an explicit preparation gate and do not draw a proxy figure.
A cache is accepted only when its recorded sessions, preprocessing, and
provenance match the requested workflow.

## Repository layout

```text
src/mrl_trace/   device dynamics and reference learning algorithms
experiments/     publication-reproduction notebooks and manifests
data/            curated device fixtures, aggregates, and ignored local caches
tests/           package and reproduction-contract checks
```

## Citation and licence

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The software is
released under the MIT licence; see [`LICENSE`](LICENSE).
