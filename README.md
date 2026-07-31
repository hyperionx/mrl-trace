# mrl-trace

`mrl-trace` is the reference implementation for *Memristive transients as
eligibility traces for all-local reinforcement learning*. It fits measured
SiO<sub>x</sub> transients empirically and evaluates an approximate cascade-shaped
eligibility surrogate in spiking, sequential and deep-local learning studies. The
surrogate is not an exact state-space realization or an identified microscopic
trap cascade.

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
- `02_sequential_and_scaling.ipynb`: a simulated retention-delay design curve,
  controlled multi-decision action sequence, array-size/remedy sweeps, and the
  CUDA-capable hybrid study.
- `03_deep_local_and_faults.ipynb`: shallow/deep local learning, pilot-tuned then
  frozen homeostasis, crossbar schematics, array scaling, and explicitly scoped
  D2D/stuck-off fault studies.
- `04_biological_grounding.ipynb`: Frank-style task metrics, descriptive
  reward-aligned dopamine photometry, and gated EEG studies.
- `05_extensions.ipynb`: beta-sensitivity validation, long-horizon, reversal,
  multi-timescale, capture, working-memory, and device-TD studies.
- `06_nmi_predictive_linkage.ipynb`: self-contained replicate-level Au/ITO model
  comparison, direct-held-bias timing sweeps, simulated design-curve and reversal
  analyses.

Each analysis notebook (00--06) declares and validates its own compact figure registry. There is no
external publication-manifest dependency. Third-party artwork and unexplained
legacy rasters are outside the reproduction workflow.

Every analysis notebook exposes the same controls near the top:

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

- `reduced` computes offline numerical panels live with smaller seed/trial budgets
  once their declared compact measured fixtures have been restored. It preserves
  condition ordering and panel structure without claiming publication-scale
  confidence intervals.
- `publication` restores the larger authored budgets, requires all 24 Au traces
  and 91 ITO workbooks for predictive linkage, and is intentionally expensive.
- `smoke` runs a minimal offline validation workload.

This checkout does not silently substitute for missing measured inputs. Restore the
declared compact fixture bundle (and, for publication mode, the complete raw-data
archive) before executing data-dependent notebooks; those notebooks fail closed when
their inputs are absent. Large per-seed arrays belong in the archival DOI, not Git.

The reproduction driver fans lightweight notebooks into isolated child
interpreters. Grid-heavy notebooks then run in canonical order with spawn-safe
cell-level pools. This avoids nested pools and gives the full worker budget to the
longest simulations. Torch hybrid work selects CUDA when the installed Torch build
provides it; NumPy reference algorithms correctly report CPU.

Figures always display inline. Saving is opt-in, existing outputs are protected,
and writing to a manuscript directory requires explicit overwrite permission.
Read-only reference rasters cannot be exported as regenerated evidence. Historical
means and confidence intervals remain provenance records; they are not expanded into
synthetic seed runs or used to replace the default live computations.

Some numerical notebooks accept `MRL_USE_ARCHIVED_RESULTS=1` only for compatible
analyses. Historical dopamine-capstone, transformed-retention and one-choice
"T-maze" archives are provenance records, not valid replacements for corrected
results. Merely finding an archive never changes a live default.

## Method provenance

Three-factor reward modulation, conventional exponential traces, LIF neurons,
feedback alignment, firing-rate homeostasis, KWW fitting and the Frank task are
established components. The signed coincidence rule, cascade eligibility surrogate,
shallow e-prop-style policy trace, and combined DFA/device/homeostasis update are
repository-specific proposals or adaptations. Result dictionaries record
`method_provenance` with status, established basis, adaptation and claim limit.

## Predictive-linkage extension

The NMI-facing additions live in `experiments/06_nmi_predictive_linkage.ipynb`,
not in a second source-level publication framework. It calls the package's scientific
models directly, embeds the empirical fitting workflow needed to audit the raw Au and
ITO files, uses direct fitted held-bias retention without floor rescaling, and writes
no results by default in smoke/reduced mode. Profiles are `smoke`, `reduced`, and
`publication`; only the last is intended for final inference and it requires an
explicit external output directory while saving both result tables and figures.

For smoke/reduced runs, set `MRL_SAVE_RESULTS=1` and `MRL_OUTPUT_DIR` to save per-trace, per-seed, and summary
artifacts to an external archival directory. Full result arrays and generated figures
are intentionally ignored by Git. SageMaker is optional: the same notebook runs on a
laptop, workstation, cluster, or ordinary cloud VM. Exact commands, workload sizes,
archive policy and historical-run provenance are contained in the notebook itself.

## External biological data

Dopamine and EEG downloads are disabled by default. The DANDI event stream is
validated as two cue markers per rewarded trial; it supplies no verified omission
class, decoder target or learning-reward pool. The only dopamine output is
descriptive reward-aligned photometry. Install the `external`
extra, set `RUN_EXTERNAL_DATA=True`, and set `ALLOW_DATA_DOWNLOADS=True` only
after reviewing the dataset instructions in the owning notebook. Absent or
invalid caches leave an explicit preparation gate and do not draw a proxy figure.
A cache is accepted only when its recorded sessions, preprocessing, and
provenance match the requested workflow.

## Repository layout

```text
src/mrl_trace/   device dynamics and reference learning algorithms
experiments/     publication-reproduction notebooks and embedded figure registries
data/            optional restored fixtures, generated archives, and ignored caches
tests/           package and reproduction-contract checks
```

## Citation and licence

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The software is
released under the MIT licence; see [`LICENSE`](LICENSE).
