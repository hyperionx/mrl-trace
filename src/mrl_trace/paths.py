"""Filesystem paths for optional local result archives and measured fixtures.

The experiments in this package are reproduced by self-contained notebooks under
``experiments/``. They keep figure construction beside the narrative while calling
shared ``run_*`` scientific cores and optionally reading or writing compatible
archives under a single top-level ``data/`` directory. Generated seed grids are not
bundled publication evidence. This module is the one place that resolves ``data/``
so that both the notebooks and the module
``main()`` entry points (``python -m mrl_trace.<module> --full``) agree on
where results live, regardless of the current working directory.

Because the package is installed editable, ``__file__`` resolves inside the real
repository ``src/`` tree, so the repository root -- and its sibling ``data/`` -- is
discoverable from the module location. Set ``MRL_TRACE_DATA_DIR`` to override (the
legacy ``SIOX_DATA_DIR`` spelling remains accepted for compatibility).
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
    "WITHDRAWN_RESULT_NAMES",
]

WITHDRAWN_RESULT_NAMES = frozenset({"exp11_dopamine_capstone.npy"})


def _reject_withdrawn_result(name: str) -> None:
    basename = Path(name).name
    if basename in WITHDRAWN_RESULT_NAMES:
        raise ValueError(
            f"{basename} is scientifically invalid and has been withdrawn: cue "
            "onset/offset markers were misclassified as separate reward/omission trials"
        )


def data_dir() -> Path:
    """Absolute path to the package ``data/`` directory.

    Honours ``MRL_TRACE_DATA_DIR`` (or legacy ``SIOX_DATA_DIR``); otherwise resolves to
    ``<repo-root>/data`` from this module's location (editable install).
    """
    env = os.environ.get("MRL_TRACE_DATA_DIR") or os.environ.get("SIOX_DATA_DIR")
    if env:
        d = Path(env).expanduser().resolve()
    else:
        # paths.py -> mrl_trace -> src -> <repo-root>
        d = Path(__file__).resolve().parents[2] / "data"
    return d


def results_dir() -> Path:
    """``data/results`` -- optional, locally generated compatible archives."""
    return data_dir() / "results"


def device_model_dir() -> Path:
    """``data/device_model`` -- measured device-model fixtures.

    A complete publication run requires the declared Au and ITO raw fixtures. Their
    absence is a preparation error, not permission to synthesize replacement data.
    """
    return data_dir() / "device_model"


def gold_export_dir() -> Path:
    """``data/device_model/gold_export`` -- the measured gold-device CSV traces."""
    return device_model_dir() / "gold_export"


def preregistration_dir() -> Path:
    """Legacy ``data/preregistration`` directory of retrospective protocol documents."""
    return data_dir() / "preregistration"


def save_result(name: str, obj) -> Path:
    """Pickle-save ``obj`` to ``data/results/<name>`` (``.npy``), returning the path."""
    _reject_withdrawn_result(name)
    path = results_dir() / name
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, obj, allow_pickle=True)
    return path


def load_result(name: str):
    """Load a result grid written by :func:`save_result` (``allow_pickle`` dict)."""
    _reject_withdrawn_result(name)
    return np.load(results_dir() / name, allow_pickle=True).item()
