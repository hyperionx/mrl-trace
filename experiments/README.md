# Reproduction notebooks

Notebooks 00--06 are the examiner-facing analysis record. They contain deterministic
outputs from the `reduced` profile and do not depend on execution state from another
notebook. Shared scientific functions and model specifications live in `src/`.

| Notebook | Scope |
|---|---|
| `00_device_physics_and_trace.ipynb` | Device physics, measured Au traces, and trace timescales |
| `01_distal_credit_ladder.ipynb` | Trace-level, spiking, and closed-loop credit assignment |
| `02_sequential_and_scaling.ipynb` | Multi-decision benchmark, fair controls, and scaling |
| `03_deep_local_and_faults.ipynb` | Synthetic deep-local and implementation sensitivities |
| `04_biological_grounding.ipynb` | Synthetic Frank task and opt-in DANDI 001340 replay |
| `05_extensions.ipynb` | Timing, reversal, multi-timescale, vector, and TD analyses |
| `06_nmi_predictive_linkage.ipynb` | Empirical model comparison, identifiability, theory, and frozen predictions |
| `REPRODUCE.ipynb` | Resource-aware execution of all topic notebooks |

Run the complete reduced workflow from the repository root:

```bash
python -m jupyter lab experiments/REPRODUCE.ipynb
```

The shared controls are `MRL_RUN_PROFILE`, `MRL_DEVICE`, `MRL_WORKERS`,
`MRL_SAVE_FIGURES`, `MRL_SAVE_RESULTS`, `MRL_OUTPUT_DIR`, `MRL_OVERWRITE`,
`MRL_RUN_EXTERNAL_DATA`, and `MRL_ALLOW_DATA_DOWNLOADS`. Reduced runs do not save
files or access external recordings unless explicitly requested.

Notebook 06 requires the 24 Au CSV traces and the declared ITO workbook cohort.
Point `MRL_TRACE_GOLD_DIR` and `MRL_TRACE_ITO_DIR` to local raw inputs; these files
remain ignored and are never packaged. Publication mode additionally requires an
external output directory and fails closed if result or figure saving is disabled.

Notebook 04 never downloads DANDI data by default. Enabling its external stage also
requires explicit raw/cache/output directories. Its DANDI 001340 result is a
**Conditional Go**: intact continuous dLight improves on shuffled waveforms in a
leakage-safe replay, while closely matched kernel shapes remain unresolved. The
separate Coddington and Jeong analyses ground causal dopamine learning and its
inhibitory direction; they do not select a device kernel.

Every notebook owns a unique figure registry and records method provenance. The
committed outputs must have non-null execution counts, no error outputs, and the
notebook-level `mrl_trace_execution.profile` value `reduced`; tests enforce this
contract.
