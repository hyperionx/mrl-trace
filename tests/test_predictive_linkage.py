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
        taus=(1.0,), seeds=2, trials=8, lag_factors=(0.7, 1.0, 1.4),
        k=2, V=0.9, tau_r_override=0.8,
    )
    serial = run_predictive_interval_sweep(**kwargs)
    pooled = run_predictive_interval_sweep(**kwargs, pool=InlinePool())

    assert serial["learned_peak"] == pooled["learned_peak"]
    assert serial["diagnostics"] == pooled["diagnostics"]
    assert len(serial["rows"]) == len(pooled["rows"])
    for expected, observed in zip(serial["rows"], pooled["rows"]):
        assert expected.keys() == observed.keys()
        assert np.array_equal(expected["seed_values"], observed["seed_values"])
