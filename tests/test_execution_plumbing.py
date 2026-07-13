"""Regression tests for release-safe path and package execution plumbing."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from mrl_trace import paths
from mrl_trace.distal_reward import run_spiking_saturation


def test_read_path_lookup_has_no_creation_side_effect(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "external-data"
    monkeypatch.setenv("MRL_TRACE_DATA_DIR", str(target))
    assert paths.data_dir() == target.resolve()
    assert not target.exists()


def test_legacy_data_environment_remains_compatible(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "legacy-data"
    monkeypatch.delenv("MRL_TRACE_DATA_DIR", raising=False)
    monkeypatch.setenv("SIOX_DATA_DIR", str(target))
    assert paths.data_dir() == target.resolve()


def test_save_result_creates_only_the_write_target(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "data"
    monkeypatch.setenv("MRL_TRACE_DATA_DIR", str(target))
    saved = paths.save_result("nested/example.npy", {"value": np.array([1.0])})
    assert saved.exists()
    loaded = np.load(saved, allow_pickle=True).item()
    assert loaded["value"].tolist() == [1.0]


def test_spiking_saturation_sweep_preserves_per_seed_evidence() -> None:
    result = run_spiking_saturation(
        seeds=2, delays=(1,), variants=(("gate_tl10", 10.0),),
        trials=2, dt=1e-2, workers=1,
    )
    values = result["sat_seeds"]["gate_tl10"][1]
    assert values.shape == (2,)
    assert np.isfinite(values).all()
    assert result["saturation"]["gate_tl10"] == [float(values.mean())]
    assert result["n_seeds"] == 2
