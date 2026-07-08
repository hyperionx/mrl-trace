# Experiments

These notebooks are live-first reproducibility artifacts for `mrl-trace`.

By default each notebook sets `RESULT_MODE = "live"` and recomputes reduced-budget evidence directly from the package. Set `RESULT_MODE = "full_sweep_cache"` only to render committed publication-scale grids under `data/results/`; cache-mode cells print their source before loading.

Run with the shared environment:

```bash
conda run -n mnn_torch python -c "import mrl_trace"
```

## Claim Alignment

- `00_device_model.ipynb`: live-backed measured transient, KWW fit, retention distribution, and habituation.
- `01_trace_and_credit_window.ipynb`: live-backed eligibility trace and delayed-credit window; live-oriented coarse `D_max` scaling.
- `02_closed_loop_rl.ipynb`: live-backed closed-loop learning and sequential maze; live-oriented scaling and remedy diagnostics.
- `03_deep_local_learning.ipynb`: live-backed deep local-learning and distractor panels; live-oriented array-scale and hybrid diagnostics.
- `04_temporal_selectivity.ipynb`: live-backed DMS and interval selectivity; live-oriented vector-timer trace response.
- `05_biological_grounding.ipynb`: live-oriented Frank PST and live-backed real local dopamine-cache panels; EEG claims are external-data gated when OpenNeuro data are absent.
- `06_extensions.ipynb`: live-backed reversal and WM/STC diagnostics; live-oriented multi-timescale, device-TD, and beta-sensitivity panels.

## External Data

The dopamine panel uses extracted DANDI-derived caches if available at `C:\tmp\da_cache` or `/tmp/da_cache`. The optional DANDI extraction helper is included in `05_biological_grounding.ipynb` behind `RUN_DANDI_EXTRACT = False`.

OpenNeuro EEG data are not bundled. Without an EEG reward-pool cache, the notebook renders an explicitly labelled external-data orientation panel rather than silently omitting the claim.

## Full-Sweep Regeneration

The notebooks include commented command patterns such as:

```bash
python -m mrl_trace.bandit --bandit --full
python -m mrl_trace.maze --exp6 --full
python -m mrl_trace.deep --exp7 --exp13 --exp14 --full
python -m mrl_trace.selectivity --exp10 --exp20 --full
python -m mrl_trace.dopamine --exp11 --full
python -m mrl_trace.extensions --exp19 --exp21 --exp22 --exp23 --full
```

Use these only when publication-scale seed counts and confidence intervals are required.
