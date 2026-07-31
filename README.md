# mrl-trace

`mrl-trace` is the reference implementation for *Memristive transients as
eligibility traces for all-local reinforcement learning*. It fits measured
SiO<sub>x</sub> transients empirically and evaluates versioned cascade-shaped
eligibility models in spiking, sequential and deep-local learning studies. The
default `physical_headroom_v1` model uses normalized upstream occupancy, nonlinear
headroom and explicit leakage. The previous linear Erlang implementation remains
available as `linear_erlang_v1`; neither identifies a microscopic trap cascade.

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
  learning-rate-fair controlled multi-decision action sequence, array-size/remedy
  sweeps, and the CUDA-capable hybrid study.
- `03_deep_local_and_faults.ipynb`: shallow/deep local learning, pilot-tuned then
  frozen homeostasis, crossbar schematics, array scaling, and explicitly scoped
  D2D/stuck-off fault studies.
- `04_biological_grounding.ipynb`: synthetic Frank learning, pinned DANDI
  001340 photometry triage, and action-contingent logged dopamine replay.
- `05_extensions.ipynb`: beta-sensitivity validation, long-horizon, reversal,
  multi-timescale, capture, working-memory, and device-TD studies.
- `06_nmi_predictive_linkage.ipynb`: replicate-level Au/ITO model comparison,
  direct-held-bias timing sweeps, shared-model provenance, and simulated
  design-curve and reversal analyses.

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
  and the frozen 91-workbook January 2025 ITO cohort plus the two declared June
  2026 1 mV direct-read supplements for predictive linkage, and is intentionally
  expensive. Any other workbooks are hashed and disclosed as out-of-cohort additions.
- `smoke` runs a minimal offline validation workload.

Raw Au, ITO, NWB, EEG and other measured recordings are intentionally ignored and
are not packaged. Set `MRL_TRACE_GOLD_DIR` to the directory containing the 24 Au
trace CSVs plus `dataset.csv` and `manifest.csv`, and set `MRL_TRACE_ITO_DIR` to the
ITO workbook directory. This checkout never substitutes for missing inputs:
publication preflight fails without those Au files and either the verified raw ITO
cohort or a source-verified schema-2 direct-retention archive. Derived reference
JSON/CSV artifacts remain tracked; large per-seed arrays belong in the archival DOI.

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
analyses. Historical DANDI 000351 dopamine-capstone, transformed-retention and one-choice
"T-maze" archives are provenance records, not valid replacements for corrected
results. Merely finding an archive never changes a live default.

## Method provenance

Three-factor reward modulation, conventional exponential traces, LIF neurons,
feedback alignment, firing-rate homeostasis, KWW fitting and the Frank task are
established components. The signed coincidence rule, cascade eligibility surrogate,
shallow e-prop-style policy trace, and combined DFA/device/homeostasis update are
repository-specific proposals or adaptations. Result dictionaries record
`method_provenance` with status, established basis, adaptation and claim limit.

The sequential and shallow-DMS comparisons first calibrate reward-time effective-
update RMS on fixed balanced trials, then tune each learned method by unsmoothed
per-seed AULC on seeds 1000--1019 and evaluate once on seeds 2000--2019. Pilot grids
expand only when a pilot optimum is on a boundary; evaluation seeds never influence
normalization, expansion, or rate choice. The primary sequential point is explicitly
synthetic (`tau_leak=10 s`, `D=12 s`, `V=1.5`, `k=3`, `beta_leak=1`). A positive
device-minus-comparator AULC is called an advantage only when its paired bootstrap
interval is wholly positive. Interval selectivity, evaluated across eta 0.01, 0.1,
and 0.5 for both gate representations, is the amplitude-robust kernel-shape test.
The publication-scale outputs, pilot surfaces, per-seed metrics and generated TeX
macros are tracked under `data/results/reference/`; they are the numerical source
for the repaired manuscript table rather than manually copied values.
Run `python -m mrl_trace.publication_artifacts --figures-dir <directory>` to
render the fair sequential, shallow-DMS, and interval summaries from those tracked
artifacts without rerunning or retuning a benchmark.

## Predictive-linkage extension

Notebook 06 is the empirical-validation driver. Reusable candidate responses,
grouped held-out fitting, model selection and provenance construction live in
`src/mrl_trace/`; no experiment imports notebook code or execution state. It compares
KWW, physical-headroom k=2--5, and linear-Erlang k=2--5 on the same held-out Au
traces, reports every candidate within 5% of the best, keeps k=3 primary with k=4
sensitivity, and removes any empirically-preferred claim if the physical family is
outside that band. It uses direct fitted held-bias retention without floor rescaling
and writes no results by default in smoke/reduced mode. Profiles are `smoke`, `reduced`, and
`publication`; only the last is intended for final inference and it requires an
explicit external output directory while saving both result tables and figures.

The verified measured-data smoke run selects linear-Erlang k=3 (held-out NRMSE
0.0548); KWW and linear k=2--5 are within 5% of the best. Physical-headroom k=2--5
scores 0.1319--0.1320 and is not empirically preferred, so it remains primary only
as the equation-matching learning model. The 91-workbook primary ITO cohort yields
50 held-bias fits at the primary QC threshold; both separately acquired 1 mV
supplements pass QC and remain outside held-bias field-law estimation.
The publication-profile measured-retention timing sweep is unresolved: all four
device peaks are right-censored, all exponential peaks are left-censored, and only
one of six retention--horizon grid points resolves. Its direction is an internal
frozen-kernel consistency check, not independent timing validation.

For smoke/reduced runs, set `MRL_SAVE_RESULTS=1` and `MRL_OUTPUT_DIR` to save per-trace, per-seed, and summary
artifacts to an external archival directory. Full result arrays and generated figures
are intentionally ignored by Git. SageMaker is optional: the same notebook runs on a
laptop, workstation, cluster, or ordinary cloud VM. Exact commands, workload sizes,
archive policy and historical-run provenance are contained in the notebook itself.

## External biological data

Dopamine and EEG downloads are disabled by default. The replacement biological
validation pins [DANDI 001340](https://dandiarchive.org/dandiset/001340/0.250221.0527)
release `0.250221.0527` with a tracked 69-asset
path/ID/size/SHA-256 manifest. Preparation is an independent reconstruction of
the [published method](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1013226):
seconds-parameterized second-difference airPLS,
robust 415-to-470 nm Huber regression, and session z-scoring. It is not the source
authors' exact pipeline.

The primary analysis uses all 46 substantive photometry sessions and 39,020 aligned
trials; the predeclared 37-session quality subset is sensitivity-only. Replay
predicts each next choice before reading that trial's outcome or waveform, then
teacher-forces the animal's action and applies its intact continuous dLight segment.
The hidden rewarded port is transient QC information and is absent from caches and
learner interfaces.

The verdict is **Conditional Go**: action-contingent replay is feasible, but plain
continuous dLight currently outperforms device weighting, the device-versus-shuffled
result is inconclusive, and this dataset is weak evidence for device-kernel
superiority. Binary-outcome Q-learning is only a positive control. Counterfactual
dopamine is unavailable when a simulated agent chooses differently, so deep-XOR and
other closed-loop learning claims remain synthetic.

The replay's primary device condition is `physical_headroom_v1` at `V=0.9`,
`tau_leak=10 s`, `k=3`, `beta_leak=1`; `linear_erlang_v1` is a separately labelled
sensitivity and receives its own independently decay-matched exponential. Replay
manifests record both model-specification digests and, when present, notebook 06's
predictive-linkage manifest digest. The empirical manifest is never used to tune
replay parameters or inspect held-out mice.

Install `.[external]` and run explicitly:

```bash
python -m mrl_trace.dopamine prepare \
  --raw-dir /path/to/dandi001340 \
  --cache-dir /path/to/cache \
  --output-dir /path/to/output
python -m mrl_trace.dopamine evaluate \
  --cache-dir /path/to/cache \
  --output-dir /path/to/output
```

Add `--download` only after deliberately opting into network downloads. Preparation
writes deterministic QC CSV/PNG outputs and non-pickle state-free session caches.
Evaluation writes LOSO predictions, per-session scores, a JSON run manifest, and
`fig_dopamine_replay.png`. Descriptive EEG loading, epoching, and decoding helpers
remain, but decoded EEG values never enter a learning update.

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
