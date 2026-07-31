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
import re
from pathlib import Path

import numpy as np

__all__ = [
    "data_dir",
    "results_dir",
    "device_model_dir",
    "gold_export_dir",
    "preregistration_dir",
    "ITO_PRIMARY_ACQUISITION_DATES",
    "ITO_SUPPLEMENTAL_NEAR_ZERO_FILENAMES",
    "select_ito_source_cohort",
    "publication_device_preflight",
    "save_result",
    "load_result",
    "WITHDRAWN_RESULT_NAMES",
]

WITHDRAWN_RESULT_NAMES = frozenset({"exp11_dopamine_capstone.npy"})
ITO_PRIMARY_ACQUISITION_DATES = frozenset({"01-15-2025", "01-16-2025"})
ITO_SUPPLEMENTAL_NEAR_ZERO_FILENAMES = frozenset({
    "vcv Site@1 Subsite 2-wire-resistor voltage bias#1 Run471 06-26-2026.xls",
    "vcv Site@1 Subsite 2-wire-resistor voltage bias#1 Run498 06-26-2026.xls",
})
_ITO_DATE_RE = re.compile(r" (?P<date>\d{2}-\d{2}-\d{4})\.xls$", re.IGNORECASE)


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


def select_ito_source_cohort(ito_raw_dir) -> dict:
    """Select the frozen ITO cohort plus the two declared 1 mV supplements.

    The 91 January 2025 workbooks define the held-bias/conditioning cohort.  Runs
    471 and 498 are the separately acquired 1 mV direct-read records required by the
    predeclared near-zero sensitivity.  Every other workbook is returned as an
    explicit out-of-cohort addition; no file is selected merely by directory order.
    """
    directory = Path(ito_raw_dir).expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"ITO raw directory not found: {directory}")
    all_workbooks = tuple(sorted(directory.glob("*.xls")))
    primary = []
    supplemental = []
    additional = []
    for path in all_workbooks:
        match = _ITO_DATE_RE.search(path.name)
        if match and match.group("date") in ITO_PRIMARY_ACQUISITION_DATES:
            primary.append(path)
        elif path.name in ITO_SUPPLEMENTAL_NEAR_ZERO_FILENAMES:
            supplemental.append(path)
        else:
            additional.append(path)
    if len(primary) != 91:
        raise FileNotFoundError(
            "ITO raw directory does not contain the frozen 91-workbook "
            f"January 2025 cohort (found {len(primary)} in {directory})"
        )
    if {path.name for path in supplemental} != ITO_SUPPLEMENTAL_NEAR_ZERO_FILENAMES:
        raise FileNotFoundError(
            "ITO raw directory does not contain both declared June 2026 1 mV "
            f"supplemental workbooks (found {len(supplemental)} in {directory})"
        )
    return {
        "directory": directory,
        "workbooks": tuple(sorted((*primary, *supplemental))),
        "primary_workbooks": tuple(primary),
        "supplemental_workbooks": tuple(supplemental),
        "additional_workbooks": tuple(additional),
        "primary_acquisition_dates": tuple(sorted(ITO_PRIMARY_ACQUISITION_DATES)),
        "supplemental_acquisition_date": "06-26-2026",
    }


def publication_device_preflight(*, gold_raw_dir=None, ito_raw_dir=None,
                                 ito_archive=None) -> dict:
    """Fail closed unless local Au data and verified direct ITO evidence exist."""
    configured_gold = gold_raw_dir or os.environ.get("MRL_TRACE_GOLD_DIR")
    gold = (Path(configured_gold).expanduser().resolve()
            if configured_gold else gold_export_dir())
    traces = sorted(gold.glob("trace_*.csv"))
    if len(traces) != 24 or not (gold / "dataset.csv").is_file() or not (
            gold / "manifest.csv").is_file():
        raise FileNotFoundError(
            "publication preflight requires 24 Au CSV traces plus dataset.csv and manifest.csv"
        )
    raw = Path(ito_raw_dir) if ito_raw_dir is not None else None
    if raw is not None:
        try:
            cohort = select_ito_source_cohort(raw)
        except FileNotFoundError:
            cohort = None
        if cohort is not None:
            return {
                "gold_traces": 24,
                "ito_evidence": "verified_raw_91_plus_2_near_zero_workbooks",
                "ito_primary_workbooks": len(cohort["primary_workbooks"]),
                "ito_supplemental_near_zero_workbooks": len(
                    cohort["supplemental_workbooks"]
                ),
                "ito_primary_acquisition_dates": list(
                    cohort["primary_acquisition_dates"]
                ),
                "ito_additional_workbooks_excluded": len(
                    cohort["additional_workbooks"]
                ),
            }
    archive = (device_model_dir() / "ito_decay_data.npz"
               if ito_archive is None else Path(ito_archive))
    if archive.is_file():
        with np.load(archive, allow_pickle=False) as payload:
            definition = str(np.asarray(
                payload.get("retention_definition", "")
            ).item())
            schema = int(np.asarray(
                payload.get("analysis_schema_version", 0)
            ).item())
            verified = bool(np.asarray(payload.get("source_verified", False)).item())
            source_digest = str(np.asarray(
                payload.get("source_manifest_sha256", "")
            ).item())
        if (definition == "direct_held_bias_tau" and schema >= 2 and verified
                and len(source_digest) == 64
                and all(character in "0123456789abcdefABCDEF"
                        for character in source_digest)):
            return {
                "gold_traces": 24,
                "ito_evidence": "source_verified_schema_2_archive",
                "source_manifest_sha256": source_digest.lower(),
            }
    raise FileNotFoundError(
        "publication preflight requires the 91-workbook January 2025 ITO cohort "
        "plus both declared June 2026 1 mV supplements, or a "
        "source-verified schema-2 direct-retention archive"
    )


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
