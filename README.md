# mrl-trace

A memristive device transient as a hardware eligibility trace for three-factor reinforcement learning.

This repository contains the reference implementation for the manuscript's device-trace reinforcement-learning studies. The package models the SiOx transient as a hardware eligibility trace for spiking and local-learning tasks.

## Project Structure

- `src/mrl_trace/`: core package for device dynamics, learning rules, simulated tasks, biological reward interfaces, and extension studies.
- `experiments/`: live-first notebooks that recompute reduced-budget evidence for the publication-facing claims.
- `data/device_model/`: bundled measured-device fixtures used as live inputs.
- `data/results/`: optional publication-scale result grids used only when explicitly requested.

## Installation

Use the same environment as the companion `mnn-torch` checks:

```bash
conda run -n mnn_torch python -c "import mrl_trace; print(mrl_trace.__file__)"
```

Editable install is still supported:

```bash
pip install -e .
```

## Reproducing Experiments

The notebooks default to:

```python
RESULT_MODE = "live"
```

Live mode recomputes reduced-budget result panels directly from the package code. It is intended to support the mechanism, direction, and qualitative structure of the claims without requiring the full publication sweep.

Set:

```python
RESULT_MODE = "full_sweep_cache"
```

only to render committed publication-scale grids from `data/results/`. Cache-mode cells print provenance before loading.

The local dopamine live path uses extracted DANDI-derived session caches when present. On this machine the expected cache is `C:\tmp\da_cache` (also visible to the code as `/tmp/da_cache`). EEG/OpenNeuro data are external and are labelled as such in the biological notebook when absent.

## Notebooks

- `00_device_model.ipynb`: measured device transient, KWW fit, retention distribution, and habituation.
- `01_trace_and_credit_window.ipynb`: eligibility trace and delayed-credit window.
- `02_closed_loop_rl.ipynb`: closed-loop bandit, scaling, remedies, and sequential maze.
- `03_deep_local_learning.ipynb`: deep local learning, distractor task, fault diagnostic, and hybrid orientation.
- `04_temporal_selectivity.ipynb`: DMS, interval selectivity, and vector-timer orientation.
- `05_biological_grounding.ipynb`: Frank PST live-oriented run, real dopamine-cache path, and EEG external-data gate.
- `06_extensions.ipynb`: reversal, multi-timescale, WM/STC, device-TD, and beta sensitivity.
- `schematics.ipynb`: live-generated explanatory diagrams.
- `REPRODUCE_ALL.ipynb`: executes all notebooks in live mode.

The temporary workspace has been consumed. Useful dopamine extraction logic now lives as a guarded, disabled-by-default cell in `05_biological_grounding.ipynb`; one-off audit, patch, download, and exploration files are not part of the reproducibility surface.
