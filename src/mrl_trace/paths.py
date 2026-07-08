"""Filesystem paths for optional result grids and device-model fixtures.

The experiments in this package are reproduced by notebooks under ``experiments/``
that hold no code themselves: they call the ``run_*`` cores in the library and read
or write result grids under a single top-level ``data/`` directory. This module is
the one place that resolves ``data/`` so that both the notebooks and the module
``main()`` entry points (``python -m mrl_trace.<module> --full``) agree on
where results live, regardless of the current working directory.

Because the package is installed editable, ``__file__`` resolves inside the real
repository ``src/`` tree, so the repository root -- and its sibling ``data/`` -- is
discoverable from the module location. Set ``SIOX_DATA_DIR`` to override (e.g. for a
wheel install or CI where ``data/`` lives elsewhere).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

__all__ = [
    "data_dir",
    "results_dir",
    "device_model_dir",
    "gold_export_dir",
    "preregistration_dir",
    "save_result",
    "load_result",
]


def data_dir() -> Path:
    """Absolute path to the package ``data/`` directory (created if absent).

    Honours the ``SIOX_DATA_DIR`` environment variable; otherwise resolves to
    ``<repo-root>/data`` from this module's location (editable install).
    """
    env = os.environ.get("SIOX_DATA_DIR")
    if env:
        d = Path(env).expanduser().resolve()
    else:
        # paths.py -> mrl_trace -> src -> <repo-root>
        d = Path(__file__).resolve().parents[2] / "data"
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_dir() -> Path:
    """``data/results`` -- optional full-sweep caches used only when requested."""
    p = data_dir() / "results"
    p.mkdir(parents=True, exist_ok=True)
    return p


def device_model_dir() -> Path:
    """``data/device_model`` -- measured device-model fixtures.

    Holds ``kww_final.json``, ``habit_data.npz``, ``ito_decay_data.npz`` and the
    ``gold_export/`` measured device traces.
    """
    p = data_dir() / "device_model"
    p.mkdir(parents=True, exist_ok=True)
    return p


def gold_export_dir() -> Path:
    """``data/device_model/gold_export`` -- the measured gold-device CSV traces."""
    return device_model_dir() / "gold_export"


def preregistration_dir() -> Path:
    """``data/preregistration`` -- the pre-registration protocol documents."""
    return data_dir() / "preregistration"


def save_result(name: str, obj) -> Path:
    """Pickle-save ``obj`` to ``data/results/<name>`` (``.npy``), returning the path."""
    path = results_dir() / name
    np.save(path, obj, allow_pickle=True)
    return path


def load_result(name: str):
    """Load a result grid written by :func:`save_result` (``allow_pickle`` dict)."""
    return np.load(results_dir() / name, allow_pickle=True).item()
