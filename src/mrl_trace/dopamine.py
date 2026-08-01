r"""DANDI 001340 preparation for action-contingent logged dopamine replay.

The source study describes airPLS detrending, robust 415-nm regression, and
session z-scoring, but does not release the custom preprocessing implementation
or its session-QC list.  This module therefore implements an explicit,
repository-owned approximation.  It is not represented as the authors' exact
pipeline.

Only recorded action, outcome, timing, and continuous dLight are written to a
``LoggedSession`` cache.  The hidden rewarded port (``state`` in the NWB trial
table) is used transiently to identify stochastic omissions for diagnostic QC
and is deliberately absent from the cache and replay API.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import rankdata

__all__ = [
    "DANDISET_ID",
    "DANDISET_VERSION",
    "DANDI001340_VERDICT",
    "DANDI001340_METHOD_PROVENANCE",
    "AssetRecord",
    "LoggedSession",
    "asset_manifest",
    "airpls",
    "robust_reference_regression",
    "preprocess_photometry",
    "download_dandi001340",
    "prepare_dandi001340",
    "load_logged_session",
]

DANDISET_ID = "001340"
DANDISET_VERSION = "0.250221.0527"
DANDI001340_VERDICT = (
    "Action-contingent replay supported; kernel discrimination unresolved"
)
CACHE_SCHEMA_VERSION = 1
PRIMARY_DETREND_WINDOW_S = 60.0
SENSITIVITY_WINDOWS_S = (30.0, 120.0)
OUTCOME_WINDOW_S = 1.0

EXPECTED_ASSETS = 69
EXPECTED_VALID_BEHAVIOR_TRIALS = 58_590
EXPECTED_BEHAVIOR_ONLY = 21
EXPECTED_TRUNCATED = frozenset({("BSD016", "p118"), ("BSD016", "p119")})
EXPECTED_SUBSTANTIVE = 46
EXPECTED_ALIGNED_TRIALS = 39_020
EXPECTED_QUALITY_SESSIONS = 38
EXPECTED_QUALITY_TRIALS = 33_330
HISTORICAL_UNVERIFIED_QUALITY_TARGET = {
    "n_sessions": 37,
    "n_trials": 32_373,
    "status": "not_reproduced_by_pinned_external_rerun_2026-08-01",
}

DANDI001340_METHOD_PROVENANCE = {
    "status": "adapted",
    "dataset": f"DANDI:{DANDISET_ID}@{DANDISET_VERSION}",
    "source_method_doi": "10.1371/journal.pcbi.1013226",
    "established_basis": [
        "fiber-photometry artifact regression",
        "airPLS baseline estimation",
        "logged behavioral replay",
    ],
    "repository_adaptation": (
        "A seconds-parameterized airPLS approximation and Huber IRLS reference "
        "regression reconstruct a continuous dLight signal from the released "
        "470/415-nm channels. Recorded choices are replayed sequentially."
    ),
    "claim_limit": (
        "The authors' custom preprocessing and QC list were not released. This "
        "pipeline establishes feasibility only; it is not evidence of biological "
        "learning, a closed-loop agent, or device-kernel superiority."
    ),
    "verdict": DANDI001340_VERDICT,
}


def _trapezoid(values, coordinates, *, axis=-1):
    implementation = getattr(np, "trapezoid", None)
    if implementation is not None:
        return implementation(values, coordinates, axis=axis)
    return np.trapz(values, coordinates, axis=axis)  # pragma: no cover - NumPy < 2


@dataclass(frozen=True)
class AssetRecord:
    """One immutable row of the pinned DANDI asset manifest."""

    path: str
    asset_id: str
    size: int
    sha256: str


@dataclass(frozen=True)
class LoggedSession:
    """State-free, non-pickle representation of one recorded replay session.

    ``waveform_offsets`` indexes ragged per-trial segments stored in the flat
    ``waveform_time_s`` and ``waveform_dlight_z`` arrays. Absolute timestamps are
    retained. ``trace_*`` stores the complete processed session so the continuous
    circular-shift control can be constructed without trial pooling.
    """

    mouse_id: str
    session_id: str
    trial_id: np.ndarray
    action: np.ndarray
    rewarded: np.ndarray
    center_in_s: np.ndarray
    center_out_s: np.ndarray
    side_in_s: np.ndarray
    outcome_s: np.ndarray
    waveform_offsets: np.ndarray
    waveform_time_s: np.ndarray
    waveform_dlight_z: np.ndarray
    trace_time_s: np.ndarray
    trace_dlight_z: np.ndarray
    source_sha256: str
    preprocessing: dict
    quality_pass: bool = False

    def __post_init__(self) -> None:
        n = len(self.trial_id)
        one_dimensional = (
            "trial_id", "action", "rewarded", "center_in_s", "center_out_s",
            "side_in_s", "outcome_s",
        )
        for name in one_dimensional:
            value = np.asarray(getattr(self, name))
            if value.ndim != 1 or len(value) != n:
                raise ValueError(f"{name} must be one-dimensional with {n} rows")
        if set(np.unique(self.action)).difference({-1, 1}):
            raise ValueError("action must encode left=-1 and right=1")
        if set(np.unique(self.rewarded)).difference({0, 1, False, True}):
            raise ValueError("rewarded must be binary")
        offsets = np.asarray(self.waveform_offsets)
        if (offsets.ndim != 1 or len(offsets) != n + 1 or offsets[0] != 0
                or np.any(np.diff(offsets) < 2)
                or offsets[-1] != len(self.waveform_time_s)
                or len(self.waveform_time_s) != len(self.waveform_dlight_z)):
            raise ValueError("invalid ragged waveform offsets")
        if (len(self.trace_time_s) != len(self.trace_dlight_z)
                or len(self.trace_time_s) < 2
                or np.any(np.diff(self.trace_time_s) <= 0)):
            raise ValueError("trace arrays must be matching and strictly increasing")
        if not all(np.isfinite(np.asarray(getattr(self, name), float)).all()
                   for name in one_dimensional[3:]):
            raise ValueError("trial timestamps must be finite")
        forbidden = {"state", "rewarded_port", "hidden_state"}
        metadata_keys: set[str] = set()

        def collect_keys(value) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    metadata_keys.add(str(key).lower())
                    collect_keys(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    collect_keys(child)

        collect_keys(self.preprocessing)
        if forbidden.intersection(metadata_keys):
            raise ValueError("hidden state must not be present in replay metadata")

    @property
    def n_trials(self) -> int:
        return int(len(self.trial_id))

    def waveform(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return the absolute timestamps and dLight samples for trial ``index``."""
        start, stop = self.waveform_offsets[index:index + 2]
        return (
            self.waveform_time_s[start:stop],
            self.waveform_dlight_z[start:stop],
        )

    def save(self, path: str | Path) -> Path:
        """Save a compressed cache without object arrays or pickle."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target,
            cache_schema_version=np.asarray(CACHE_SCHEMA_VERSION, dtype=np.int16),
            mouse_id=np.asarray(self.mouse_id),
            session_id=np.asarray(self.session_id),
            trial_id=np.asarray(self.trial_id, dtype=np.int32),
            action=np.asarray(self.action, dtype=np.int8),
            rewarded=np.asarray(self.rewarded, dtype=np.int8),
            center_in_s=np.asarray(self.center_in_s, dtype=np.float64),
            center_out_s=np.asarray(self.center_out_s, dtype=np.float64),
            side_in_s=np.asarray(self.side_in_s, dtype=np.float64),
            outcome_s=np.asarray(self.outcome_s, dtype=np.float64),
            waveform_offsets=np.asarray(self.waveform_offsets, dtype=np.int64),
            waveform_time_s=np.asarray(self.waveform_time_s, dtype=np.float64),
            waveform_dlight_z=np.asarray(self.waveform_dlight_z, dtype=np.float64),
            trace_time_s=np.asarray(self.trace_time_s, dtype=np.float64),
            trace_dlight_z=np.asarray(self.trace_dlight_z, dtype=np.float64),
            source_sha256=np.asarray(self.source_sha256),
            preprocessing_json=np.asarray(json.dumps(
                self.preprocessing, sort_keys=True, separators=(",", ":")
            )),
            quality_pass=np.asarray(bool(self.quality_pass)),
        )
        return target


def load_logged_session(path: str | Path) -> LoggedSession:
    """Load and schema-validate a :class:`LoggedSession` without pickle."""
    with np.load(Path(path), allow_pickle=False) as value:
        version = int(value["cache_schema_version"])
        if version != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported LoggedSession schema {version}; "
                f"expected {CACHE_SCHEMA_VERSION}"
            )
        names = (
            "trial_id", "action", "rewarded", "center_in_s", "center_out_s",
            "side_in_s", "outcome_s", "waveform_offsets", "waveform_time_s",
            "waveform_dlight_z", "trace_time_s", "trace_dlight_z",
        )
        arrays = {name: np.asarray(value[name]) for name in names}
        return LoggedSession(
            mouse_id=str(value["mouse_id"]),
            session_id=str(value["session_id"]),
            source_sha256=str(value["source_sha256"]),
            preprocessing=json.loads(str(value["preprocessing_json"])),
            quality_pass=bool(value["quality_pass"]),
            **arrays,
        )


def asset_manifest() -> tuple[AssetRecord, ...]:
    """Return the tracked manifest for the published 69-asset release."""
    manifest = resources.files("mrl_trace").joinpath("dandi001340_assets.csv")
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = tuple(
            AssetRecord(
                path=row["path"],
                asset_id=row["asset_id"],
                size=int(row["size"]),
                sha256=row["sha256"],
            )
            for row in csv.DictReader(stream)
        )
    if len(rows) != EXPECTED_ASSETS or len({row.path for row in rows}) != len(rows):
        raise RuntimeError("DANDI 001340 asset manifest is incomplete or duplicated")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_asset(raw_dir: Path, record: AssetRecord) -> Path:
    nested = raw_dir / Path(record.path)
    flat = raw_dir / Path(record.path).name
    return nested if nested.exists() or not flat.exists() else flat


def download_dandi001340(
    raw_dir: str | Path,
    *,
    allow_download: bool = False,
) -> tuple[Path, ...]:
    """Verify or explicitly download every pinned DANDI asset.

    Network access is refused unless ``allow_download=True``. Existing files are
    always checked against the tracked size and SHA-256.
    """
    root = Path(raw_dir)
    verified: list[Path] = []
    for record in asset_manifest():
        target = root / Path(record.path)
        if target.is_file():
            if target.stat().st_size != record.size or _sha256(target) != record.sha256:
                raise ValueError(f"source-data integrity check failed: {target}")
            verified.append(target)
            continue
        if not allow_download:
            raise FileNotFoundError(
                f"missing pinned DANDI asset {record.path}; pass "
                "allow_download=True only after explicitly enabling external downloads"
            )
        import requests

        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".part")
        url = (
            "https://api.dandiarchive.org/api/assets/"
            f"{record.asset_id}/download/"
        )
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        stream.write(block)
        if temporary.stat().st_size != record.size or _sha256(temporary) != record.sha256:
            temporary.unlink(missing_ok=True)
            raise ValueError(f"downloaded source failed integrity check: {record.path}")
        temporary.replace(target)
        verified.append(target)
    return tuple(verified)


def _airpls_lambda(dt: float, window_s: float) -> float:
    if not np.isfinite(dt) or dt <= 0 or not np.isfinite(window_s) or window_s <= 2 * dt:
        raise ValueError("dt and detrending window must be finite and positive")
    return float(np.power(2.0 * np.sin(np.pi * dt / window_s), -4))


def airpls(
    signal,
    timestamps,
    *,
    window_s: float = PRIMARY_DETREND_WINDOW_S,
    max_iter: int = 50,
    tol: float = 1e-3,
) -> tuple[np.ndarray, dict]:
    """Estimate a smooth baseline with an explicit airPLS approximation.

    The Whittaker second-difference penalty is parameterized in seconds through
    ``lambda = [2 sin(pi*dt/window)]^-4``. This is a repository convention,
    required because the source study did not publish its airPLS parameters.
    """
    y = np.asarray(signal, dtype=float)
    t = np.asarray(timestamps, dtype=float)
    if (y.ndim != 1 or t.ndim != 1 or y.shape != t.shape or len(y) < 5
            or not np.isfinite(y).all() or not np.isfinite(t).all()
            or np.any(np.diff(t) <= 0)):
        raise ValueError("airPLS inputs must be finite matching monotonic vectors")
    dt = float(np.median(np.diff(t)))
    lam = _airpls_lambda(dt, float(window_s))
    difference = sparse.diags(
        (np.ones(len(y) - 2), -2 * np.ones(len(y) - 2), np.ones(len(y) - 2)),
        (0, 1, 2),
        shape=(len(y) - 2, len(y)),
        format="csc",
    )
    penalty = lam * (difference.T @ difference)
    weights = np.ones(len(y), dtype=float)
    scale = max(float(np.sum(np.abs(y))), np.finfo(float).eps)
    converged = False
    baseline = np.zeros_like(y)
    iteration = 0
    for iteration in range(1, int(max_iter) + 1):
        system = sparse.diags(weights, format="csc") + penalty
        baseline = np.asarray(spsolve(system, weights * y), dtype=float)
        residual = y - baseline
        negative = residual < 0
        negative_mass = float(np.sum(np.abs(residual[negative])))
        if not negative.any() or negative_mass / scale < tol:
            converged = True
            break
        weights.fill(0.0)
        exponent = np.minimum(
            iteration * np.abs(residual[negative]) / max(negative_mass, 1e-12),
            50.0,
        )
        weights[negative] = np.exp(exponent)
        endpoint = float(np.max(weights[negative]))
        weights[0] = endpoint
        weights[-1] = endpoint
    if not np.isfinite(baseline).all():
        raise RuntimeError("airPLS baseline was not finite")
    return baseline, {
        "algorithm": "repository_airpls_second_difference",
        "window_s": float(window_s),
        "dt_s": dt,
        "lambda": lam,
        "max_iter": int(max_iter),
        "iterations": int(iteration),
        "tolerance": float(tol),
        "converged": bool(converged),
    }


def robust_reference_regression(
    reference,
    signal,
    *,
    c: float = 1.345,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Fit ``signal = intercept + slope*reference`` with Huber IRLS."""
    x = np.asarray(reference, dtype=float)
    y = np.asarray(signal, dtype=float)
    if (x.ndim != 1 or y.shape != x.shape or len(x) < 3
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError("robust regression inputs must be finite matching vectors")
    design = np.column_stack((np.ones(len(x)), x))
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    converged = False
    iteration = 0
    for iteration in range(1, int(max_iter) + 1):
        residual = y - design @ coef
        center = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - center)))
        scale = max(scale, np.finfo(float).eps)
        standardized = np.abs(residual) / scale
        weights = np.ones_like(standardized)
        outside = standardized > c
        weights[outside] = c / standardized[outside]
        root_weight = np.sqrt(weights)
        new_coef = np.linalg.lstsq(
            design * root_weight[:, None], y * root_weight, rcond=None
        )[0]
        if np.linalg.norm(new_coef - coef) <= tol * (1 + np.linalg.norm(coef)):
            coef = new_coef
            converged = True
            break
        coef = new_coef
    fitted = design @ coef
    return fitted, {
        "algorithm": "huber_irls",
        "huber_c": float(c),
        "max_iter": int(max_iter),
        "iterations": int(iteration),
        "tolerance": float(tol),
        "converged": bool(converged),
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
    }


def preprocess_photometry(
    signal_470,
    timestamps_470,
    signal_415,
    timestamps_415,
    *,
    detrend_window_s: float = PRIMARY_DETREND_WINDOW_S,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return common timestamps and session-z-scored, artifact-corrected dLight."""
    y470 = np.asarray(signal_470, dtype=float)
    t470 = np.asarray(timestamps_470, dtype=float)
    y415 = np.asarray(signal_415, dtype=float)
    t415 = np.asarray(timestamps_415, dtype=float)
    baseline470, air470 = airpls(
        y470, t470, window_s=detrend_window_s
    )
    baseline415, air415 = airpls(
        y415, t415, window_s=detrend_window_s
    )
    residual470 = y470 - baseline470
    residual415 = y415 - baseline415
    if min(t470[-1], t415[-1]) <= max(t470[0], t415[0]):
        raise ValueError("470/415-nm recordings have no substantive timestamp overlap")
    # The released channel clocks differ by fractions of a sample at their
    # boundaries. np.interp provides deterministic endpoint continuation there
    # while preserving every actual 470-nm timestamp.
    time = t470
    reference = np.interp(time, t415, residual415)
    signal = residual470
    fitted, regression = robust_reference_regression(reference, signal)
    corrected = signal - fitted
    sd = float(np.std(corrected))
    if not np.isfinite(sd) or sd <= np.finfo(float).eps:
        raise ValueError("artifact-corrected dLight has zero or invalid variance")
    zscore = (corrected - float(np.mean(corrected))) / sd
    metadata = {
        "implementation": "mrl_trace_independent_approximation",
        "detrend_window_s": float(detrend_window_s),
        "airpls_470": air470,
        "airpls_415": air415,
        "reference_regression": regression,
        "session_zscore_mean": float(np.mean(corrected)),
        "session_zscore_sd": sd,
        "outcome_debase": "performed during trial replay",
    }
    return time, zscore, metadata


def _decode_strings(values) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def _behavior_arrays(nwb) -> dict:
    if "intervals/trials" not in nwb:
        raise ValueError("NWB file is missing intervals/trials")
    trials = nwb["intervals/trials"]
    required = (
        "trial", "action", "rewarded", "state", "center_in", "center_out",
        "side_in", "outcome", "last_side_out",
    )
    missing = [name for name in required if name not in trials]
    if missing:
        raise ValueError(f"NWB trial table is missing columns: {missing}")
    return {
        "trial_id": np.asarray(trials["trial"], dtype=np.int32),
        "action_text": _decode_strings(trials["action"][()]),
        "state_text": _decode_strings(trials["state"][()]),
        "rewarded": np.asarray(trials["rewarded"], dtype=float),
        "center_in_s": np.asarray(trials["center_in"], dtype=float),
        "center_out_s": np.asarray(trials["center_out"], dtype=float),
        "side_in_s": np.asarray(trials["side_in"], dtype=float),
        "outcome_s": np.asarray(trials["outcome"], dtype=float),
        "last_side_out_s": np.asarray(trials["last_side_out"], dtype=float),
    }


def _valid_behavior_mask(values: dict) -> np.ndarray:
    action = values["action_text"]
    state = values["state_text"]
    rewarded = values["rewarded"]
    center_in = values["center_in_s"]
    center_out = values["center_out_s"]
    side_in = values["side_in_s"]
    outcome = values["outcome_s"]
    return (
        np.isin(action, ("left", "right"))
        & np.isin(state, ("left", "right"))
        & np.isin(rewarded, (0.0, 1.0))
        & np.isfinite(center_in)
        & np.isfinite(center_out)
        & np.isfinite(side_in)
        & np.isfinite(outcome)
        & (center_in <= center_out)
        & (center_out <= side_in)
        & (side_in <= outcome)
    )


def _session_identity(path: Path, values: dict) -> tuple[str, str]:
    stem = path.stem
    if "_ses-" in stem and stem.startswith("sub-"):
        left, session = stem.split("_ses-", 1)
        return left.removeprefix("sub-"), session
    mouse = str(values["state_text"].shape[0])
    return f"unknown-{mouse}", stem


def _photometry_status(nwb, values: dict) -> tuple[str, str]:
    if "acquisition" not in nwb:
        return "behavior_only", "415/470-nm acquisition groups are absent"
    acquisition = nwb["acquisition"]
    required = ("fp_series_415nm", "fp_series_470nm")
    if any(name not in acquisition for name in required):
        return "behavior_only", "one or both photometry channels are absent"
    lengths = [len(acquisition[name]["data"]) for name in required]
    if min(lengths) < 1_000:
        return "truncated", f"photometry contains only {min(lengths)} samples"
    t415 = np.asarray(acquisition["fp_series_415nm"]["timestamps"], dtype=float)
    t470 = np.asarray(acquisition["fp_series_470nm"]["timestamps"], dtype=float)
    behavior = values["outcome_s"][np.isfinite(values["outcome_s"])]
    overlap_start, overlap_stop = max(t415[0], t470[0]), min(t415[-1], t470[-1])
    if (overlap_start >= overlap_stop or not len(behavior)
            or behavior.max() < overlap_start or behavior.min() > overlap_stop):
        return "truncated", "photometry timestamps do not overlap behavior"
    return "substantive", ""


def _bootstrap_difference(
    rewarded_values: np.ndarray,
    omitted_values: np.ndarray,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    # Chunking bounds memory for long sessions while retaining exact deterministic
    # percentile-bootstrap semantics.
    differences = np.empty(n_boot, dtype=float)
    chunk = 500
    for start in range(0, n_boot, chunk):
        stop = min(start + chunk, n_boot)
        nr = stop - start
        reward_mean = rewarded_values[
            rng.integers(0, len(rewarded_values), size=(nr, len(rewarded_values)))
        ].mean(axis=1)
        omission_mean = omitted_values[
            rng.integers(0, len(omitted_values), size=(nr, len(omitted_values)))
        ].mean(axis=1)
        differences[start:stop] = reward_mean - omission_mean
    lo, hi = np.percentile(differences, (2.5, 97.5))
    return float(lo), float(hi)


def _auc(labels, scores) -> float:
    y = np.asarray(labels, dtype=bool)
    score = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = rankdata(score, method="average")
    return float(
        (ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    )


def _prepare_one(
    path: Path,
    record: AssetRecord,
    *,
    detrend_window_s: float,
) -> tuple[LoggedSession | None, dict, dict | None]:
    import h5py

    with h5py.File(path, "r") as nwb:
        values = _behavior_arrays(nwb)
        valid = _valid_behavior_mask(values)
        mouse_id, session_id = _session_identity(path, values)
        status, reason = _photometry_status(nwb, values)
        base_qc = {
            "mouse_id": mouse_id,
            "session_id": session_id,
            "asset_path": record.path,
            "source_sha256": record.sha256,
            "status": status,
            "rejection_reason": reason,
            "n_behavior_trials": int(len(valid)),
            "n_valid_behavior_trials": int(valid.sum()),
            "n_aligned_trials": 0,
            "n_rewarded": 0,
            "n_stochastic_omission": 0,
            "separation_mean": float("nan"),
            "separation_ci_low": float("nan"),
            "separation_ci_high": float("nan"),
            "separation_auc": float("nan"),
            "quality_pass": False,
        }
        if status != "substantive":
            return None, base_qc, None

        acquisition = nwb["acquisition"]
        signal415 = np.asarray(acquisition["fp_series_415nm"]["data"], dtype=float)
        time415 = np.asarray(acquisition["fp_series_415nm"]["timestamps"], dtype=float)
        signal470 = np.asarray(acquisition["fp_series_470nm"]["data"], dtype=float)
        time470 = np.asarray(acquisition["fp_series_470nm"]["timestamps"], dtype=float)
        trace_time, trace_dlight, preprocessing = preprocess_photometry(
            signal470, time470, signal415, time415,
            detrend_window_s=detrend_window_s,
        )
        preprocessing.update({
            "dataset": f"DANDI:{DANDISET_ID}",
            "dataset_version": DANDISET_VERSION,
            "asset_id": record.asset_id,
            "source_sha256": record.sha256,
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "eligibility_event": "center_out",
            "outcome_window_s": OUTCOME_WINDOW_S,
        })
        complete = (
            valid
            & (values["center_out_s"] >= trace_time[0])
            & (values["outcome_s"] + OUTCOME_WINDOW_S <= trace_time[-1])
            # The last trial commonly lacks a side-out marker even though its
            # required outcome waveform is complete. A finite marker beyond the
            # recording, as in BSD019/p148 trial 1072, proves truncation.
            & (
                ~np.isfinite(values["last_side_out_s"])
                | (values["last_side_out_s"] <= trace_time[-1])
            )
        )
        indices = np.flatnonzero(complete)
        if not len(indices):
            base_qc.update(
                status="truncated",
                rejection_reason="no complete behavior/photometry trial overlap",
            )
            return None, base_qc, None

        offsets = [0]
        segment_times: list[np.ndarray] = []
        segment_values: list[np.ndarray] = []
        diagnostic_grid = np.linspace(0.0, OUTCOME_WINDOW_S, 101)
        diagnostic_traces = np.empty((len(indices), len(diagnostic_grid)))
        for row, source_index in enumerate(indices):
            start = values["center_out_s"][source_index]
            stop = values["outcome_s"][source_index] + OUTCOME_WINDOW_S
            select = (trace_time > start) & (trace_time < stop)
            time = np.r_[start, trace_time[select], stop]
            dlight = np.r_[
                np.interp(start, trace_time, trace_dlight),
                trace_dlight[select],
                np.interp(stop, trace_time, trace_dlight),
            ]
            if len(time) < 2:
                raise RuntimeError("aligned trial unexpectedly has fewer than two samples")
            segment_times.append(time)
            segment_values.append(dlight)
            offsets.append(offsets[-1] + len(time))
            outcome = values["outcome_s"][source_index]
            baseline = float(np.interp(outcome, trace_time, trace_dlight))
            diagnostic_traces[row] = np.interp(
                outcome + diagnostic_grid, trace_time, trace_dlight
            ) - baseline

        action_text = values["action_text"][indices]
        state_text = values["state_text"][indices]
        rewarded = values["rewarded"][indices].astype(np.int8)
        action = np.where(action_text == "right", 1, -1).astype(np.int8)
        correct_action = action_text == state_text
        stochastic_omission = correct_action & (rewarded == 0)
        reward_label = rewarded == 1
        plain_response = _trapezoid(
            diagnostic_traces, diagnostic_grid, axis=1
        )
        rewarded_values = plain_response[reward_label]
        omitted_values = plain_response[stochastic_omission]
        session_seed = int.from_bytes(
            hashlib.sha256(f"{mouse_id}/{session_id}".encode()).digest()[:4],
            "little",
        )
        if len(rewarded_values) >= 20 and len(omitted_values) >= 20:
            ci_low, ci_high = _bootstrap_difference(
                rewarded_values, omitted_values, seed=session_seed
            )
        else:
            ci_low, ci_high = float("nan"), float("nan")
        separation = (
            float(rewarded_values.mean() - omitted_values.mean())
            if len(rewarded_values) and len(omitted_values) else float("nan")
        )
        labels = np.r_[np.ones(len(rewarded_values), dtype=bool),
                       np.zeros(len(omitted_values), dtype=bool)]
        scores = np.r_[rewarded_values, omitted_values]
        quality_pass = bool(
            len(rewarded_values) >= 20
            and len(omitted_values) >= 20
            and np.isfinite(ci_low)
            and ci_low > 0
        )
        session = LoggedSession(
            mouse_id=mouse_id,
            session_id=session_id,
            trial_id=values["trial_id"][indices],
            action=action,
            rewarded=rewarded,
            center_in_s=values["center_in_s"][indices],
            center_out_s=values["center_out_s"][indices],
            side_in_s=values["side_in_s"][indices],
            outcome_s=values["outcome_s"][indices],
            waveform_offsets=np.asarray(offsets, dtype=np.int64),
            waveform_time_s=np.concatenate(segment_times),
            waveform_dlight_z=np.concatenate(segment_values),
            trace_time_s=trace_time,
            trace_dlight_z=trace_dlight,
            source_sha256=record.sha256,
            preprocessing=preprocessing,
            quality_pass=quality_pass,
        )
        base_qc.update({
            "n_aligned_trials": session.n_trials,
            "n_rewarded": int(reward_label.sum()),
            "n_stochastic_omission": int(stochastic_omission.sum()),
            "separation_mean": separation,
            "separation_ci_low": ci_low,
            "separation_ci_high": ci_high,
            "separation_auc": _auc(labels, scores),
            "quality_pass": quality_pass,
        })
        diagnostics = {
            "rewarded_traces": diagnostic_traces[reward_label],
            "omitted_traces": diagnostic_traces[stochastic_omission],
            "grid_s": diagnostic_grid,
            "diagnostic_labels": np.where(
                reward_label | stochastic_omission,
                reward_label.astype(np.int8),
                -1,
            ),
        }
        return session, base_qc, diagnostics


def _write_qc(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_triage(
    path: Path,
    rows: list[dict],
    reward_sum: np.ndarray,
    reward_n: int,
    omission_sum: np.ndarray,
    omission_n: int,
    grid_s: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    status_order = ("substantive", "behavior_only", "truncated")
    counts = [sum(row["status"] == name for row in rows) for name in status_order]
    fig, (ax_trace, ax_qc) = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ax_trace.plot(grid_s, reward_sum / reward_n, color="#2f7d5c",
                  label=f"reward (n={reward_n:,})")
    ax_trace.plot(grid_s, omission_sum / omission_n, color="#b44b48",
                  label=f"stochastic omission (n={omission_n:,})")
    ax_trace.axhline(0, color="0.7", lw=0.8)
    ax_trace.set(xlabel="time from outcome (s)", ylabel="outcome-debased dLight (z)",
                 title="continuous outcome response")
    ax_trace.legend(frameon=False, fontsize=8)
    ax_trace.spines[["top", "right"]].set_visible(False)
    ax_qc.bar(range(3), counts, color=("#3aa07a", "#9aa6b2", "#c0392b"))
    ax_qc.set_xticks(range(3), ("substantive", "behavior-only", "truncated"))
    ax_qc.tick_params(axis="x", rotation=25)
    ax_qc.set(ylabel="sessions", title="explicit session disposition")
    ax_qc.spines[["top", "right"]].set_visible(False)
    fig.suptitle("DANDI 001340 triage - independent preprocessing")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _validate_expected(report: dict, rows: list[dict], *, primary: bool) -> None:
    if report["n_assets"] != EXPECTED_ASSETS:
        raise RuntimeError(f"expected {EXPECTED_ASSETS} assets; found {report['n_assets']}")
    if report["n_valid_behavior_trials"] != EXPECTED_VALID_BEHAVIOR_TRIALS:
        raise RuntimeError(
            "valid behavioral trial count drifted: "
            f"{report['n_valid_behavior_trials']} != {EXPECTED_VALID_BEHAVIOR_TRIALS}"
        )
    if report["n_behavior_only_sessions"] != EXPECTED_BEHAVIOR_ONLY:
        raise RuntimeError("behavior-only session disposition drifted")
    observed_truncated = {
        (row["mouse_id"], row["session_id"])
        for row in rows if row["status"] == "truncated"
    }
    if observed_truncated != EXPECTED_TRUNCATED:
        raise RuntimeError(f"truncated sessions drifted: {observed_truncated}")
    if report["n_substantive_sessions"] != EXPECTED_SUBSTANTIVE:
        raise RuntimeError("substantive photometry session count drifted")
    if report["n_aligned_trials"] != EXPECTED_ALIGNED_TRIALS:
        raise RuntimeError(
            f"aligned trial count drifted: {report['n_aligned_trials']} "
            f"!= {EXPECTED_ALIGNED_TRIALS}"
        )
    if primary and (
        report["n_quality_sessions"] != EXPECTED_QUALITY_SESSIONS
        or report["n_quality_trials"] != EXPECTED_QUALITY_TRIALS
        or len(report["quality_mice"]) != 5
    ):
        raise RuntimeError(
            "quality sensitivity set drifted: "
            f"{report['n_quality_sessions']} sessions/"
            f"{report['n_quality_trials']} trials/"
            f"{len(report['quality_mice'])} mice"
        )


def prepare_dandi001340(
    raw_dir: str | Path,
    cache_dir: str | Path,
    output_dir: str | Path,
    *,
    detrend_windows_s: Iterable[float] = (
        PRIMARY_DETREND_WINDOW_S, *SENSITIVITY_WINDOWS_S,
    ),
    allow_download: bool = False,
    verify_hashes: bool = True,
    validate_expected: bool = True,
) -> dict:
    """Prepare pinned NWBs, state-free caches, QC tables, and triage figures.

    The 60-s window is primary. Additional windows are complete sensitivity
    reruns with independent cache subdirectories. Raw recordings are never
    downloaded unless ``allow_download=True``.
    """
    raw_root = Path(raw_dir)
    cache_root = Path(cache_dir)
    output_root = Path(output_dir)
    records = asset_manifest()
    if allow_download:
        download_dandi001340(raw_root, allow_download=True)

    sources: list[tuple[AssetRecord, Path]] = []
    for record in records:
        path = _resolve_asset(raw_root, record)
        if not path.is_file():
            raise FileNotFoundError(f"missing DANDI source asset: {record.path}")
        if path.stat().st_size != record.size:
            raise ValueError(f"source size mismatch: {path}")
        if verify_hashes and _sha256(path) != record.sha256:
            raise ValueError(f"source SHA-256 mismatch: {path}")
        sources.append((record, path))

    windows = tuple(float(value) for value in detrend_windows_s)
    if not windows or PRIMARY_DETREND_WINDOW_S not in windows:
        raise ValueError("detrend_windows_s must include the primary 60-s window")
    reports: dict[str, dict] = {}
    primary_rows: list[dict] | None = None
    primary_plot = None
    for window in windows:
        rows: list[dict] = []
        sessions: list[LoggedSession] = []
        overlap_labels: list[np.ndarray] = []
        labeled_device_overlaps: list[np.ndarray] = []
        labeled_linear_overlaps: list[np.ndarray] = []
        all_device_overlaps: list[np.ndarray] = []
        all_exponential_overlaps: list[np.ndarray] = []
        all_linear_overlaps: list[np.ndarray] = []
        all_linear_exponential_overlaps: list[np.ndarray] = []
        grid = np.linspace(0.0, OUTCOME_WINDOW_S, 101)
        reward_sum = np.zeros_like(grid)
        omission_sum = np.zeros_like(grid)
        reward_n = omission_n = 0
        window_cache = cache_root / f"window_{int(window)}s"
        for record, path in sources:
            session, qc, diagnostics = _prepare_one(
                path, record, detrend_window_s=window
            )
            rows.append(qc)
            if session is not None:
                session.save(
                    window_cache
                    / f"sub-{session.mouse_id}_ses-{session.session_id}.npz"
                )
                sessions.append(session)
                reward = diagnostics["rewarded_traces"]
                omission = diagnostics["omitted_traces"]
                reward_sum += reward.sum(axis=0)
                omission_sum += omission.sum(axis=0)
                reward_n += len(reward)
                omission_n += len(omission)
                # Reward/omission labels exist only in this transient QC scope.
                # They are never serialized or exposed to the replay learner.
                diagnostic_labels = diagnostics["diagnostic_labels"]
                diagnostic = diagnostic_labels >= 0
                from .dopamine_replay import trial_modulators
                device_overlap = trial_modulators(session, "device")
                exponential_overlap = trial_modulators(
                    session, "matched_exponential"
                )
                linear_overlap = trial_modulators(session, "linear_device")
                linear_exponential_overlap = trial_modulators(
                    session, "linear_matched_exponential"
                )
                overlap_labels.append(diagnostic_labels[diagnostic])
                labeled_device_overlaps.append(device_overlap[diagnostic])
                labeled_linear_overlaps.append(linear_overlap[diagnostic])
                all_device_overlaps.append(device_overlap)
                all_exponential_overlaps.append(exponential_overlap)
                all_linear_overlaps.append(linear_overlap)
                all_linear_exponential_overlaps.append(
                    linear_exponential_overlap
                )

        labels_all = np.concatenate(overlap_labels)
        device_labeled = np.concatenate(labeled_device_overlaps)
        linear_labeled = np.concatenate(labeled_linear_overlaps)
        device_all = np.concatenate(all_device_overlaps)
        exponential_all = np.concatenate(all_exponential_overlaps)
        linear_all = np.concatenate(all_linear_overlaps)
        linear_exponential_all = np.concatenate(
            all_linear_exponential_overlaps
        )
        quality_sessions = [
            session for session in sessions if session.quality_pass
        ]
        report = {
            "verdict": DANDI001340_VERDICT,
            "dataset": f"DANDI:{DANDISET_ID}@{DANDISET_VERSION}",
            "detrend_window_s": window,
            "n_assets": len(rows),
            "n_valid_behavior_trials": int(sum(
                row["n_valid_behavior_trials"] for row in rows
            )),
            "n_behavior_only_sessions": int(sum(
                row["status"] == "behavior_only" for row in rows
            )),
            "n_truncated_sessions": int(sum(
                row["status"] == "truncated" for row in rows
            )),
            "n_substantive_sessions": len(sessions),
            "n_aligned_trials": int(sum(session.n_trials for session in sessions)),
            "n_quality_sessions": len(quality_sessions),
            "n_quality_trials": int(sum(
                session.n_trials for session in quality_sessions
            )),
            "mice": sorted({session.mouse_id for session in sessions}),
            "quality_mice": sorted({
                session.mouse_id for session in quality_sessions
            }),
            "device_overlap_reward_omission_auc": _auc(
                labels_all == 1, device_labeled
            ),
            "device_exponential_overlap_correlation": float(np.corrcoef(
                device_all, exponential_all
            )[0, 1]),
            "linear_device_overlap_reward_omission_auc": _auc(
                labels_all == 1, linear_labeled
            ),
            "linear_device_exponential_overlap_correlation": float(np.corrcoef(
                linear_all, linear_exponential_all
            )[0, 1]),
            "physical_linear_overlap_correlation": float(np.corrcoef(
                device_all, linear_all
            )[0, 1]),
            "method_provenance": DANDI001340_METHOD_PROVENANCE,
        }
        if validate_expected:
            _validate_expected(
                report, rows, primary=math.isclose(window, PRIMARY_DETREND_WINDOW_S)
            )
        qc_name = (
            "dandi001340_qc.csv"
            if math.isclose(window, PRIMARY_DETREND_WINDOW_S)
            else f"dandi001340_qc_{int(window)}s.csv"
        )
        _write_qc(output_root / qc_name, rows)
        reports[str(int(window))] = report
        if math.isclose(window, PRIMARY_DETREND_WINDOW_S):
            primary_rows = rows
            primary_plot = (
                reward_sum, reward_n, omission_sum, omission_n, grid
            )

    assert primary_rows is not None and primary_plot is not None
    _plot_triage(
        output_root / "dandi001340_triage_traces.png",
        primary_rows,
        *primary_plot,
    )
    _plot_triage(
        output_root / "fig_dopamine_triage.png",
        primary_rows,
        *primary_plot,
    )
    run_manifest = {
        "schema_version": 1,
        "reports": reports,
        "primary_window_s": PRIMARY_DETREND_WINDOW_S,
        "sensitivity_windows_s": list(SENSITIVITY_WINDOWS_S),
        "source_manifest": "mrl_trace/dandi001340_assets.csv",
        "source_assets": [
            {
                "path": record.path,
                "asset_id": record.asset_id,
                "size": record.size,
                "sha256": record.sha256,
            }
            for record in records
        ],
        "preprocessing_parameters": {
            "timestamp_basis": "actual_NWB_timestamps",
            "reference_alignment": "linear_415_onto_470",
            "airpls_penalty_order": 2,
            "airpls_lambda": "[2*sin(pi*dt/window_s)]^-4",
            "airpls_max_iterations": 50,
            "airpls_convergence": 1e-3,
            "huber_irls_c": 1.345,
            "huber_irls_max_iterations": 50,
            "huber_irls_convergence": 1e-8,
            "session_standardization": "mean_zero_population_sd_one",
            "quality_min_rewarded_trials": 20,
            "quality_min_stochastic_omission_trials": 20,
            "quality_bootstrap_resamples": 10_000,
            "quality_bootstrap_ci": 0.95,
            "quality_rule": "reward_minus_stochastic_omission_ci_wholly_positive",
        },
        "historical_unverified_quality_target":
            HISTORICAL_UNVERIFIED_QUALITY_TARGET,
        "quality_target_resolution": (
            "The pinned external rerun selected 38 sessions and 33,330 aligned "
            "trials at every declared detrending window. The earlier 37-session/"
            "32,373-trial target was not reproduced and is not forced by a "
            "post-hoc session exclusion."
        ),
        "claim": (
            "Feasibility of logged action-contingent replay only; no positive "
            "biological-learning or device-superiority result."
        ),
        "triage_trace": {
            "time_from_outcome_s": primary_plot[4].tolist(),
            "reward_mean_dlight_z": (
                primary_plot[0] / primary_plot[1]
            ).tolist(),
            "stochastic_omission_mean_dlight_z": (
                primary_plot[2] / primary_plot[3]
            ).tolist(),
            "n_rewarded": int(primary_plot[1]),
            "n_stochastic_omission": int(primary_plot[3]),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "dandi001340_preparation_manifest.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(run_manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return run_manifest


def main(argv=None) -> None:
    """Prepare or evaluate the DANDI 001340 logged-replay workflow."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--raw-dir", required=True)
    prepare.add_argument("--cache-dir", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--download", action="store_true")
    prepare.add_argument("--no-verify-hashes", action="store_true")
    prepare.add_argument("--no-validate-expected", action="store_true")
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--cache-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--quality-only", action="store_true")
    arguments = parser.parse_args(argv)

    if arguments.command == "prepare":
        report = prepare_dandi001340(
            arguments.raw_dir,
            arguments.cache_dir,
            arguments.output_dir,
            allow_download=arguments.download,
            verify_hashes=not arguments.no_verify_hashes,
            validate_expected=not arguments.no_validate_expected,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    from .dopamine_replay import evaluate_logged_replay_loso, write_replay_artifacts

    sessions = [
        load_logged_session(path)
        for path in sorted(
            (Path(arguments.cache_dir) / "window_60s").glob("*.npz")
        )
    ]
    if arguments.quality_only:
        sessions = [session for session in sessions if session.quality_pass]
    evaluation = evaluate_logged_replay_loso(sessions)
    write_replay_artifacts(evaluation, arguments.output_dir)
    print(json.dumps(evaluation["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
