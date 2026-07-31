from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "experiments" / "06_nmi_predictive_linkage.ipynb"


class InlinePool:
    """Exercise the pool code path without platform-specific child startup."""

    def map(self, function, jobs, chunksize=1):
        return list(map(function, jobs))


def test_predictive_linkage_notebook_contract() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for control in (
        "MRL_RUN_PROFILE", "MRL_WORKERS", "MRL_SAVE_RESULTS", "MRL_OUTPUT_DIR",
        "smoke", "reduced", "publication",
    ):
        assert control in source
    assert "device_analysis" not in source
    assert "publication.py" not in source
    assert "mrl_trace.evidence" not in source


def test_parallel_selectivity_path_preserves_seed_order() -> None:
    from mrl_trace.selectivity import run_predictive_interval_sweep

    kwargs = dict(
        taus=(1.0,), seeds=2, trials=8, delay_grid=(0.5, 1.0, 2.0),
        k=2, V=0.9, tau_r_override=0.8, beta_leak=0.7,
        retention_definition="measured_held_bias_quantiles",
    )
    serial = run_predictive_interval_sweep(**kwargs)
    pooled = run_predictive_interval_sweep(**kwargs, pool=InlinePool())

    assert serial["learned_peak"] == pooled["learned_peak"]
    assert serial["diagnostics"] == pooled["diagnostics"]
    assert serial["grid_source"] == "predefined_absolute_common_grid"
    assert serial["retention_definition"] == "measured_held_bias_quantiles"
    assert serial["beta_leak"] == 0.7
    assert len(serial["rows"]) == len(pooled["rows"])
    for expected, observed in zip(serial["rows"], pooled["rows"]):
        assert expected.keys() == observed.keys()
        assert np.array_equal(expected["seed_values"], observed["seed_values"])


def test_predictive_sweep_uses_one_absolute_grid_and_reports_censoring() -> None:
    from mrl_trace.selectivity import run_predictive_interval_sweep

    result = run_predictive_interval_sweep(
        taus=(0.8, 2.0), seeds=1, trials=5,
        delay_grid=(0.5, 1.0, 2.0), k=2, V=0.9,
        tau_r_override=0.5,
    )
    for tau in result["taus"]:
        observed = sorted({
            row["delay_s"] for row in result["rows"] if row["tau_s"] == tau
        })
        assert observed == result["delay_grid"]
        for condition in ("device", "exp", "no_trace"):
            assert result["peak_censoring"][condition][tau] in {
                "none", "left", "right", "both",
            }
            low, high = result["preferred_band_by_condition"][condition][tau]
            assert 0.5 <= low <= high <= 2.0


def test_predictive_difference_bootstraps_independent_seed_clusters() -> None:
    from mrl_trace.selectivity import run_predictive_interval_sweep
    from mrl_trace.stats import bootstrap_ci

    result = run_predictive_interval_sweep(
        taus=(0.8, 1.5, 2.5), seeds=3, trials=8,
        delay_grid=(0.5, 1.0, 2.0, 4.0), k=2, V=0.9,
        tau_r_override=0.5,
    )
    paired_rows = []
    for tau in result["taus"]:
        predicted = result["predicted_tstar"][tau]
        design_delay = min(result["delay_grid"], key=lambda d: abs(d - predicted))
        cells = {
            row["condition"]: np.asarray(row["seed_values"], float)
            for row in result["rows"]
            if row["tau_s"] == tau and row["delay_s"] == design_delay
        }
        paired_rows.append(cells["device"] - cells["exp"])

    per_seed = np.stack(paired_rows).mean(axis=0)
    diagnostics = result["diagnostics"]
    assert np.allclose(
        diagnostics["device_minus_exponential_per_seed_cluster"], per_seed
    )
    assert diagnostics["device_minus_exponential_resampling_unit"] == (
        "seed_cluster_across_retention"
    )
    assert diagnostics["device_minus_exponential_independent_seed_count"] == 3
    assert np.allclose(
        diagnostics["device_minus_exponential_95ci"], bootstrap_ci(per_seed)
    )
