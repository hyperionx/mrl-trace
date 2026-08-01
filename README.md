# mrl-trace

Reference implementation for *Measured memristive transients provide an
interval-selective prior for local reinforcement learning*.

The repository fits measured SiO<sub>x</sub> current transients and evaluates their
use as local eligibility-state surrogates. The nonlinear physical-headroom cascade
is the equation-matching primary model; a linear Erlang cascade is retained as an
explicit sensitivity. Neither representation identifies a literal microscopic
stage count.

## Install

Python 3.10--3.12 is supported.

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[repro,test]"
```

Install `.[external]` only for explicitly enabled DANDI/NWB workflows. The base
package contains the NumPy/SciPy models; the `repro` extra adds notebooks,
plotting, Torch, and empirical-analysis dependencies.

## Reproduce

The source CLI has three fail-closed profiles:

```bash
python -m mrl_trace.reproduce --profile smoke
python -m mrl_trace.reproduce --profile reduced
python -m mrl_trace.reproduce --profile publication \
  --gold-dir /path/to/gold_export --ito-dir /path/to/ITO_data
```

- `smoke` checks analytic identities and tracked artifact contracts without raw
  data.
- `reduced` is a deterministic workstation regression using smaller live
  simulation budgets. The committed notebooks contain outputs from this profile.
- `publication` runs the full reference jobs and fails unless the declared Au and
  ITO inputs pass preflight.

The complete reduced notebook workflow is:

```bash
python -m jupyter lab experiments/REPRODUCE.ipynb
```

Notebook 06 runs automatically in reduced and publication profiles. The driver
uses isolated child interpreters and gives serial heavy notebooks the full worker
budget. On the reference Windows workstation, the complete live reduced run took
approximately three hours; use `smoke` for quick checks.

## Notebooks

| Notebook | Responsibility |
|---|---|
| `00_device_physics_and_trace.ipynb` | Measured transients, trace shapes, and timescales |
| `01_distal_credit_ladder.ipynb` | Trace-level, spiking, and closed-loop distal credit |
| `02_sequential_and_scaling.ipynb` | Fair sequential comparison, scaling, and remedies |
| `03_deep_local_and_faults.ipynb` | Synthetic deep-local, homeostasis, array, and fault sensitivities |
| `04_biological_grounding.ipynb` | Synthetic Frank task and opt-in DANDI logged replay |
| `05_extensions.ipynb` | Interval, reversal, multi-timescale, vector, and TD extensions |
| `06_nmi_predictive_linkage.ipynb` | Au/ITO model comparison, identifiability, theory, timing, and falsification predictions |

Every notebook records the reduced execution profile, embeds its outputs, declares
its owned figures, and exposes consistent environment controls. Figures display
inline; saving and overwriting remain explicit opt-ins.

## Data contract

Raw Au, ITO, NWB, and EEG recordings are ignored by Git and excluded from package
artifacts. Configure local measured inputs with:

```text
MRL_TRACE_GOLD_DIR   directory containing 24 Au trace CSVs, dataset.csv, manifest.csv
MRL_TRACE_ITO_DIR    directory containing the frozen 91-workbook cohort and two 1 mV supplements
```

Publication preflight refuses incomplete or unverified inputs. Tracked
`data/results/reference/` JSON/CSV/TeX files are compact derived evidence; large
per-seed arrays, raw recordings, and caches remain external.

Notebook 06 compares KWW and physical/linear cascade candidates by grouped
leave-one-bias-out prediction. The verified analysis resolves a compressed rise but
not a unique cascade depth; the nonlinear physical equation is not the best
descriptive fit to the Au traces. AICc/BIC are secondary working-likelihood checks
and do not override held-out prediction.

## Scientific scope

The main positive result is specialised interval selectivity: a rise followed by
discharge supplies a non-zero-lag temporal prior that is absent from an
instantaneous-rise exponential. Fair sequential and shallow-DMS comparisons do not
establish general superiority over tuned controls. Scalar readout aliases times on
opposite sides of the peak and therefore cannot encode temporal order.

Structural accounting concerns externally stored eligibility state under the stated
circuit abstraction. It is not a measured energy, area, throughput, or closed-loop
hardware advantage. Bias-conditioned preferred-lag curves and pulse-off bands are
untested model predictions.

The biological evidence is deliberately complementary. Reanalysis of the
Coddington closed-loop intervention gives a positive action-contingent dopamine
policy-learning result; a pinned 428-asset Jeong/DANDI 000351 analysis recovers the
reported inhibition direction across 13 mice and 82 conditioning sessions. DANDI
001340 then supplies a leakage-safe, action-contingent waveform bridge: the intact
physical dLight replay improves held-out likelihood over shuffled waveforms. Its
action-contingent replay result leaves discrimination among the closely matched
physical, linear, and exponential kernels unresolved. The evidence hierarchy and public-data
contracts are encoded in `claims.yaml`, the tracked reference artifacts, and the
publication-contract tests.

## External DANDI workflow

DANDI 001340 is pinned to release `0.250221.0527` with a tracked 69-asset digest
manifest. Preparation independently reconstructs airPLS detrending and robust
415-to-470 nm regression because the study's custom preprocessing and QC code were
not released.

```bash
python -m pip install -e ".[external]"
python -m mrl_trace.dopamine prepare \
  --raw-dir /path/to/dandi001340 \
  --cache-dir /path/to/cache \
  --output-dir /path/to/output
python -m mrl_trace.dopamine evaluate \
  --cache-dir /path/to/cache \
  --output-dir /path/to/output
```

Downloads occur only with `prepare --download`. Session caches are non-pickle and
exclude the hidden rewarded port. Evaluation predicts each choice before observing
the current outcome or waveform, teacher-forces the recorded action, and evaluates
held-out mice.

## Verification

```bash
python -m pytest -q
python -m build
```

`claims.yaml` maps stable claim IDs to artifact JSON pointers, producer commands,
seed protocols, model digests, and tolerances. Scientific boundaries are stated
where the corresponding results are described and enforced by the manuscript
contract tests.

## Layout

```text
src/mrl_trace/          device models, learning rules, analyses, reproduction CLI
experiments/            executed reduced notebooks and orchestration driver
data/results/reference/ tracked compact publication artifacts
tests/                  numerical and publication-contract tests
```

Citation metadata is in [`CITATION.cff`](CITATION.cff). The software is released
under the [`MIT licence`](LICENSE).
