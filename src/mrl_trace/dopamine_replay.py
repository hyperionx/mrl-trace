"""Leakage-safe logged replay of DANDI 001340 actions and continuous dLight."""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import expit

from .device import (
    CascadeEligibilityGate, LinearErlangEligibilityGate,
    decay_matched_exponential_tau,
)
from .dopamine import (
    DANDI001340_METHOD_PROVENANCE,
    DANDI001340_VERDICT,
    LoggedSession,
)
from .model_specs import (
    PRIMARY_MODEL_ID, LINEAR_MODEL_ID, device_model_spec, file_sha256,
)
from .stats import bootstrap_ci

__all__ = [
    "REPLAY_CONDITIONS",
    "ReplayParameters",
    "trial_modulators",
    "run_logged_replay",
    "evaluate_logged_replay_loso",
    "write_replay_artifacts",
]

REPLAY_CONDITIONS = (
    "previous_choice",
    "plain_dlight",
    "device",
    "matched_exponential",
    "linear_device",
    "linear_matched_exponential",
    "no_trace",
    "shuffled_device",
    "shifted_device",
    "outcome_rl",
)

DEVICE_V = 0.9
DEVICE_TAU_LEAK_S = 10.0
DEVICE_K = 3
DEVICE_BETA_LEAK = 1.0
ELIGIBILITY_PULSE_S = 0.3
KERNEL_DT_S = 0.005
KERNEL_MATCH_HORIZON_S = 120.0


def _predictive_linkage_metadata(reference=None) -> dict:
    """Describe an optional notebook-06 manifest without consuming its estimates."""
    if reference is None:
        candidate = (Path(__file__).resolve().parents[2] / "data" / "results" /
                     "reference" / "predictive_linkage_manifest.json")
        if not candidate.exists():
            return {"available": False, "used_for_computation": False}
        reference = candidate
    if isinstance(reference, dict):
        encoded = json.dumps(
            reference, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return {
            "available": True,
            "sha256": hashlib.sha256(encoded).hexdigest(),
            "used_for_computation": False,
        }
    path = Path(reference)
    if not path.exists():
        return {"available": False, "used_for_computation": False}
    repository_root = Path(__file__).resolve().parents[2]
    try:
        display_path = path.resolve().relative_to(repository_root).as_posix()
    except ValueError:
        display_path = str(path)
    return {
        "available": True,
        "sha256": file_sha256(path),
        "path": display_path,
        "used_for_computation": False,
    }


def _trapezoid(values, coordinates):
    implementation = getattr(np, "trapezoid", None)
    if implementation is not None:
        return implementation(values, coordinates)
    return np.trapz(values, coordinates)  # pragma: no cover - NumPy < 2


@dataclass(frozen=True)
class ReplayParameters:
    """Choice intercept, perseveration, and positive update rate."""

    bias: float = 0.0
    kappa: float = 0.0
    eta: float = 0.0

    def as_dict(self) -> dict:
        return {
            "bias": float(self.bias),
            "kappa": float(self.kappa),
            "eta": float(self.eta),
        }


@lru_cache(maxsize=256)
def _kernel_tables(horizon_s: float, gate_model: str = PRIMARY_MODEL_ID
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError("eligibility horizon must be finite and positive")
    if horizon_s > KERNEL_MATCH_HORIZON_S:
        raise ValueError(
            f"trial horizon exceeds frozen {KERNEL_MATCH_HORIZON_S:g}-s kernel"
        )
    full_grid, full_device, full_exponential = _frozen_kernel_tables(gate_model)
    stop = int(np.ceil(horizon_s / KERNEL_DT_S)) + 1
    return (
        full_grid[:stop],
        full_device[:stop],
        full_exponential[:stop],
    )


@lru_cache(maxsize=2)
def _frozen_kernel_tables(gate_model=PRIMARY_MODEL_ID
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = np.arange(
        0.0,
        KERNEL_MATCH_HORIZON_S + 0.5 * KERNEL_DT_S,
        KERNEL_DT_S,
    )
    gate_cls = {
        PRIMARY_MODEL_ID: CascadeEligibilityGate,
        LINEAR_MODEL_ID: LinearErlangEligibilityGate,
    }.get(gate_model)
    if gate_cls is None:
        raise ValueError(f"unknown gate_model: {gate_model!r}")
    gate = gate_cls(
        V=DEVICE_V,
        tau_leak=DEVICE_TAU_LEAK_S,
        k=DEVICE_K,
        beta_leak=DEVICE_BETA_LEAK,
        dt=KERNEL_DT_S,
    )
    device = gate.trace(
        grid,
        coincidence_at=0.0,
        coincidence_dur=ELIGIBILITY_PULSE_S,
        normalise=False,
    )
    matched_tau = decay_matched_exponential_tau(
        DEVICE_TAU_LEAK_S,
        V=DEVICE_V,
        k=DEVICE_K,
        beta_leak=DEVICE_BETA_LEAK,
        coincidence_dur=ELIGIBILITY_PULSE_S,
        gate_model=gate_model,
    )
    exponential = np.exp(-grid / matched_tau)
    device_area = float(_trapezoid(device, grid))
    exponential_area = float(_trapezoid(exponential, grid))
    if device_area > 0 and exponential_area > 0:
        exponential *= device_area / exponential_area
    return grid, device, exponential


def _shifted_values(
    session: LoggedSession,
    sample_time: np.ndarray,
) -> np.ndarray:
    start, stop = session.trace_time_s[0], session.trace_time_s[-1]
    duration = stop - start
    shifted_time = start + np.mod(
        sample_time - start + duration / 2.0, duration
    )
    return np.interp(
        shifted_time, session.trace_time_s, session.trace_dlight_z
    )


def _base_modulators(
    session: LoggedSession,
    *,
    shifted: bool = False,
    gate_model: str = PRIMARY_MODEL_ID,
) -> dict[str, np.ndarray]:
    """Compute plain/device/exponential overlaps for each intact trial segment."""
    max_elapsed = max(
        float(session.waveform(i)[0][-1] - session.center_out_s[i])
        for i in range(session.n_trials)
    )
    kernel_time, device_kernel, exponential_kernel = _kernel_tables(
        max_elapsed, gate_model
    )
    plain = np.empty(session.n_trials, dtype=float)
    device = np.empty(session.n_trials, dtype=float)
    exponential = np.empty(session.n_trials, dtype=float)
    for index in range(session.n_trials):
        time, recorded = session.waveform(index)
        if shifted:
            modulator = _shifted_values(session, time)
            at_outcome = _shifted_values(
                session, np.asarray([session.outcome_s[index]])
            )[0]
        else:
            modulator = recorded
            at_outcome = float(np.interp(
                session.outcome_s[index], time, recorded
            ))
        modulator = modulator - at_outcome
        elapsed = time - session.center_out_s[index]
        e_device = np.interp(elapsed, kernel_time, device_kernel)
        e_exponential = np.interp(elapsed, kernel_time, exponential_kernel)
        plain[index] = float(_trapezoid(modulator, time))
        device[index] = float(_trapezoid(modulator * e_device, time))
        exponential[index] = float(_trapezoid(
            modulator * e_exponential, time
        ))
    return {
        "plain_dlight": plain,
        "device": device,
        "matched_exponential": exponential,
    }


def trial_modulators(
    session: LoggedSession,
    condition: str,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Return one post-choice update scalar per trial for ``condition``.

    Shuffling permutes complete within-session device-overlap segments. Shifting
    circularly offsets the complete continuous session trace by half its duration
    before reconstructing trial segments.
    """
    if condition not in REPLAY_CONDITIONS:
        raise ValueError(f"unknown replay condition {condition!r}")
    if condition in {"previous_choice", "no_trace"}:
        return np.zeros(session.n_trials, dtype=float)
    if condition == "outcome_rl":
        return np.asarray(session.rewarded, dtype=float)
    if condition == "shifted_device":
        return _base_modulators(
            session, shifted=True, gate_model=PRIMARY_MODEL_ID
        )["device"]
    if condition in {"linear_device", "linear_matched_exponential"}:
        linear = _base_modulators(session, gate_model=LINEAR_MODEL_ID)
        return linear[
            "device" if condition == "linear_device" else "matched_exponential"
        ]
    base = _base_modulators(session, gate_model=PRIMARY_MODEL_ID)
    if condition == "shuffled_device":
        stable = int.from_bytes(
            hashlib.sha256(
                f"{session.mouse_id}/{session.session_id}/{seed}".encode()
            ).digest()[:8],
            "little",
        )
        return base["device"][np.random.default_rng(stable).permutation(
            session.n_trials
        )]
    return base[condition]


def _coerce_parameters(value) -> ReplayParameters:
    if isinstance(value, ReplayParameters):
        return value
    if isinstance(value, dict):
        return ReplayParameters(**value)
    values = np.asarray(value, dtype=float).ravel()
    if len(values) != 3:
        raise ValueError("parameters must provide bias, kappa, and eta")
    return ReplayParameters(*map(float, values))


def run_logged_replay(
    session: LoggedSession,
    condition: str,
    parameters: ReplayParameters | dict | tuple,
    *,
    modulators: np.ndarray | None = None,
    seed: int = 0,
) -> dict:
    """Predict before outcome, then teacher-force and update from recorded data.

    Predictions at trial ``t`` depend only on earlier recorded trials. The current
    recorded action is used only after its prediction to select the eligible weight.
    """
    if condition not in REPLAY_CONDITIONS:
        raise ValueError(f"unknown replay condition {condition!r}")
    params = _coerce_parameters(parameters)
    if not np.isfinite((params.bias, params.kappa, params.eta)).all():
        raise ValueError("replay parameters must be finite")
    if params.eta < 0:
        raise ValueError("eta must be non-negative")
    if modulators is None:
        modulators = trial_modulators(session, condition, seed=seed)
    modulators = np.asarray(modulators, dtype=float)
    if modulators.shape != (session.n_trials,) or not np.isfinite(modulators).all():
        raise ValueError("modulators must be a finite value per trial")

    weights = np.zeros(2, dtype=float)  # left, right
    previous_action = 0.0
    probability_right = np.empty(session.n_trials, dtype=float)
    weights_before = np.empty((session.n_trials, 2), dtype=float)
    for index in range(session.n_trials):
        weights_before[index] = weights
        logit = (
            params.bias
            + params.kappa * previous_action
            + weights[1] - weights[0]
        )
        probability_right[index] = float(expit(np.clip(logit, -30.0, 30.0)))

        # Teacher forcing happens after the probability is recorded. It follows the
        # animal's action even when the model assigned the other action higher probability.
        recorded_action = session.action[index]
        chosen = 1 if recorded_action == 1 else 0
        if condition == "outcome_rl":
            weights[chosen] += params.eta * (
                float(session.rewarded[index]) - weights[chosen]
            )
        elif condition not in {"previous_choice", "no_trace"}:
            weights[chosen] += params.eta * modulators[index]
        previous_action = float(recorded_action)

    observed_right = session.action == 1
    probability_observed = np.where(
        observed_right, probability_right, 1.0 - probability_right
    )
    trial_log_loss = -np.log(np.clip(probability_observed, 1e-12, 1.0))
    score_mask = np.arange(session.n_trials) > 0
    return {
        "mouse_id": session.mouse_id,
        "session_id": session.session_id,
        "condition": condition,
        "trial_id": session.trial_id.copy(),
        "recorded_action": session.action.copy(),
        "recorded_outcome": session.rewarded.copy(),
        "probability_right": probability_right,
        "trial_log_loss": trial_log_loss,
        "score_mask": score_mask,
        "weights_before": weights_before,
        "parameters": params.as_dict(),
    }


def _objective_parameters(raw: np.ndarray, condition: str) -> ReplayParameters:
    if condition in {"previous_choice", "no_trace"}:
        return ReplayParameters(float(raw[0]), float(raw[1]), 0.0)
    if condition == "outcome_rl":
        return ReplayParameters(
            float(raw[0]), float(raw[1]), float(expit(raw[2]))
        )
    return ReplayParameters(
        float(raw[0]), float(raw[1]), float(np.exp(raw[2]))
    )


def _fit_parameters(
    sessions: list[LoggedSession],
    condition: str,
    *,
    seed: int,
    n_starts: int = 8,
    modulators_by_session: dict[tuple[str, str], np.ndarray] | None = None,
) -> ReplayParameters:
    if not sessions:
        raise ValueError("at least one training session is required")
    modulators = (
        {
            (session.mouse_id, session.session_id):
            trial_modulators(session, condition, seed=seed)
            for session in sessions
        }
        if modulators_by_session is None
        else modulators_by_session
    )
    dimension = 2 if condition in {"previous_choice", "no_trace"} else 3
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]
    if dimension == 3:
        bounds.append((-8.0, 3.0) if condition != "outcome_rl" else (-8.0, 8.0))

    if condition != "outcome_rl":
        observed, previous, accumulated = [], [], []
        for session in sessions:
            modulator = modulators[(session.mouse_id, session.session_id)]
            signed_update = np.asarray(session.action, float) * modulator
            effect_before = np.r_[0.0, np.cumsum(signed_update)[:-1]]
            observed.append((session.action[1:] == 1).astype(float))
            previous.append(session.action[:-1].astype(float))
            accumulated.append(effect_before[1:])
        observed = np.concatenate(observed)
        previous = np.concatenate(previous)
        accumulated = np.concatenate(accumulated)

        def objective(raw):
            params = _objective_parameters(raw, condition)
            logit = (
                params.bias
                + params.kappa * previous
                + params.eta * accumulated
            )
            # Stable Bernoulli negative log likelihood.
            loss = np.logaddexp(0.0, logit) - observed * logit
            return float(loss.mean())
    else:
        outcome_design = []
        for session in sessions:
            trial_index = np.arange(session.n_trials)
            action_design = []
            for action in (-1, 1):
                positions = np.flatnonzero(session.action == action)
                action_design.append((
                    positions,
                    np.asarray(session.rewarded[positions], dtype=float),
                    np.searchsorted(positions, trial_index, side="left") - 1,
                ))
            outcome_design.append((
                action_design,
                (session.action[1:] == 1).astype(float),
                session.action[:-1].astype(float),
            ))

        def objective(raw):
            params = _objective_parameters(raw, condition)
            losses = []
            for action_design, observed, previous in outcome_design:
                values_before = []
                for positions, outcomes, previous_occurrence in action_design:
                    after_update = lfilter(
                        [params.eta],
                        [1.0, -(1.0 - params.eta)],
                        outcomes,
                    )
                    value = np.zeros(len(previous_occurrence), dtype=float)
                    available = previous_occurrence >= 0
                    value[available] = after_update[
                        previous_occurrence[available]
                    ]
                    values_before.append(value)
                logit = (
                    params.bias
                    + params.kappa * previous
                    + (values_before[1] - values_before[0])[1:]
                )
                losses.append(
                    np.logaddexp(0.0, logit) - observed * logit
                )
            return float(np.concatenate(losses).mean())

    rng = np.random.default_rng(seed)
    starts = [np.zeros(dimension)]
    starts.extend(
        rng.uniform(
            [bound[0] for bound in bounds],
            [bound[1] for bound in bounds],
            size=(max(0, n_starts - 1), dimension),
        )
    )
    best = None
    for start in starts:
        result = minimize(objective, start, method="L-BFGS-B", bounds=bounds)
        if best is None or result.fun < best.fun:
            best = result
    if best is None or not np.isfinite(best.fun):
        raise RuntimeError(f"could not fit replay condition {condition}")
    return _objective_parameters(best.x, condition)


def _session_key(session: LoggedSession) -> tuple[str, str]:
    return session.mouse_id, session.session_id


def evaluate_logged_replay_loso(
    sessions: list[LoggedSession] | tuple[LoggedSession, ...],
    *,
    conditions: tuple[str, ...] = REPLAY_CONDITIONS,
    seed: int = 0,
    n_starts: int = 8,
    n_boot: int = 10_000,
    include_quality_sensitivity: bool = True,
    predictive_linkage_manifest=None,
) -> dict:
    """Fit on four mice and score pre-outcome choices from the held-out mouse."""
    sessions = list(sessions)
    if not sessions:
        raise ValueError("no logged sessions were supplied")
    if len({_session_key(session) for session in sessions}) != len(sessions):
        raise ValueError("mouse/session identifiers must be unique")
    unknown = set(conditions).difference(REPLAY_CONDITIONS)
    if unknown:
        raise ValueError(f"unknown replay conditions: {sorted(unknown)}")
    mice = sorted({session.mouse_id for session in sessions})
    if len(mice) < 2:
        raise ValueError("leave-one-mouse-out evaluation needs at least two mice")

    prediction_rows: list[dict] = []
    session_rows: list[dict] = []
    parameters: dict[str, dict[str, dict]] = {}
    for condition_index, condition in enumerate(conditions):
        condition_seed = seed + 1009 * condition_index
        all_modulators = {
            _session_key(session):
            (
                np.zeros(session.n_trials, dtype=float)
                if condition == "outcome_rl"
                else trial_modulators(session, condition, seed=condition_seed)
            )
            for session in sessions
        }
        parameters[condition] = {}
        for mouse_index, held_out in enumerate(mice):
            training = [
                session for session in sessions if session.mouse_id != held_out
            ]
            testing = [
                session for session in sessions if session.mouse_id == held_out
            ]
            fold_seed = condition_seed + 37 * mouse_index
            fitted = _fit_parameters(
                training,
                condition,
                seed=fold_seed,
                n_starts=n_starts,
                modulators_by_session=all_modulators,
            )
            parameters[condition][held_out] = fitted.as_dict()
            for session in testing:
                result = run_logged_replay(
                    session,
                    condition,
                    fitted,
                    modulators=all_modulators[_session_key(session)],
                    seed=fold_seed,
                )
                mask = result["score_mask"]
                scored_loss = result["trial_log_loss"][mask]
                session_rows.append({
                    "condition": condition,
                    "held_out_mouse": held_out,
                    "session_id": session.session_id,
                    "n_scored_trials": int(mask.sum()),
                    "log_loss": float(scored_loss.mean()),
                })
                for index in np.flatnonzero(mask):
                    prediction_rows.append({
                        "condition": condition,
                        "held_out_mouse": held_out,
                        "session_id": session.session_id,
                        "trial_id": int(result["trial_id"][index]),
                        "recorded_action": int(result["recorded_action"][index]),
                        "recorded_outcome": int(result["recorded_outcome"][index]),
                        "probability_right": float(
                            result["probability_right"][index]
                        ),
                        "log_loss": float(result["trial_log_loss"][index]),
                    })

    pooled: dict[str, float] = {}
    per_mouse: dict[str, dict[str, float]] = {}
    for condition in conditions:
        rows = [row for row in prediction_rows if row["condition"] == condition]
        pooled[condition] = float(np.mean([row["log_loss"] for row in rows]))
        per_mouse[condition] = {
            mouse: float(np.mean([
                row["log_loss"] for row in rows
                if row["held_out_mouse"] == mouse
            ]))
            for mouse in mice
        }

    bootstrap = None
    paired_comparisons = {}
    if "device" in conditions:
        by_key = {
            (row["condition"], row["held_out_mouse"], row["session_id"]):
            row["log_loss"]
            for row in session_rows
        }
        keys = sorted({
            (row["held_out_mouse"], row["session_id"])
            for row in session_rows if row["condition"] == "device"
        })
        comparators = [condition for condition in conditions if condition != "device"]
        for comparator_index, comparator in enumerate(comparators):
            differences = np.asarray([
                by_key[("device", *key)] - by_key[(comparator, *key)]
                for key in keys
            ])
            lo, hi = bootstrap_ci(
                differences,
                n_boot=n_boot,
                seed=seed + comparator_index,
            )
            paired_comparisons[comparator] = {
                "comparison": f"device_minus_{comparator}",
                "mean": float(differences.mean()),
                "ci95": [lo, hi],
                "negative_is_better": True,
                "n_sessions": len(differences),
            }
        bootstrap = paired_comparisons.get("shuffled_device")

    summary = {
        "verdict": DANDI001340_VERDICT,
        "n_mice": len(mice),
        "n_sessions": len(sessions),
        "n_trials": int(sum(session.n_trials for session in sessions)),
        "n_scored_trials": int(sum(session.n_trials - 1 for session in sessions)),
        "pooled_log_loss": pooled,
        "per_mouse_log_loss": per_mouse,
        "device_minus_shuffled_bootstrap": bootstrap,
        "physical_device_paired_comparisons": paired_comparisons,
        "interpretation": (
            "Action-contingent logged replay is feasible. The physical device "
            "improves slightly over shuffled and shifted signals, but its paired "
            "intervals versus plain dLight, its matched exponential, and the "
            "linear device include zero. Biological learning and device-kernel "
            "superiority are therefore not established."
        ),
        "configuration": {
            "choice_model": (
                "P(right)=sigmoid(bias+kappa*previous_action+w_right-w_left)"
            ),
            "prediction_timing": "before current action teacher forcing and outcome",
            "first_trial_scored": False,
            "weights_reset": "per_session",
            "eligibility_event": "center_out",
            "eligibility_pulse_s": ELIGIBILITY_PULSE_S,
            "outcome_window_s": 1.0,
            "outcome_debase": (
                "subtract interpolated dLight at outcome from the complete "
                "center-out through outcome+1s segment"
            ),
            "integration": "trapezoidal",
            "device": {
                "gate_model": PRIMARY_MODEL_ID,
                "V": DEVICE_V,
                "tau_leak_s": DEVICE_TAU_LEAK_S,
                "k": DEVICE_K,
                "beta": DEVICE_BETA_LEAK,
                "solver_dt_s": KERNEL_DT_S,
                "kernel_normalization": "none",
                "area_matching_horizon_s": KERNEL_MATCH_HORIZON_S,
            },
            "matched_exponential": (
                "post-peak decay matched to the frozen device kernel and "
                "rescaled to equal finite-horizon area"
            ),
            "linear_sensitivity": {
                "gate_model": LINEAR_MODEL_ID,
                "matched_exponential_recomputed_independently": True,
            },
            "shuffle_control": (
                "deterministic within-session permutation of intact complete "
                "waveform-segment overlaps"
            ),
            "shift_control": (
                "half-session circular shift of the complete continuous dLight trace"
            ),
            "optimizer": {
                "method": "bounded_deterministic_multistart_L-BFGS-B",
                "n_starts": int(n_starts),
                "seed": int(seed),
                "bias_bounds": [-5.0, 5.0],
                "kappa_bounds": [-5.0, 5.0],
                "continuous_log_eta_bounds": [-8.0, 3.0],
                "outcome_logit_eta_bounds": [-8.0, 8.0],
            },
            "session_bootstrap_resamples": int(n_boot),
            "model_specifications": {
                model_id: device_model_spec(model_id)
                for model_id in (PRIMARY_MODEL_ID, LINEAR_MODEL_ID)
            },
            "predictive_linkage_manifest": _predictive_linkage_metadata(
                predictive_linkage_manifest
            ),
        },
        "method_provenance": DANDI001340_METHOD_PROVENANCE,
    }
    evaluation = {
        "summary": summary,
        "parameters_by_training_fold": parameters,
        "session_scores": session_rows,
        "predictions": prediction_rows,
    }
    quality_sessions = [session for session in sessions if session.quality_pass]
    if (
        include_quality_sensitivity
        and len(quality_sessions) >= 2
        and len(quality_sessions) < len(sessions)
        and len({session.mouse_id for session in quality_sessions}) >= 2
    ):
        sensitivity = evaluate_logged_replay_loso(
            quality_sessions,
            conditions=conditions,
            seed=seed,
            n_starts=n_starts,
            n_boot=n_boot,
            include_quality_sensitivity=False,
            predictive_linkage_manifest=predictive_linkage_manifest,
        )
        evaluation["quality_filtered_sensitivity"] = sensitivity
        summary["quality_filtered_sensitivity"] = sensitivity["summary"]
    return evaluation


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_replay_artifacts(evaluation: dict, output_dir: str | Path) -> dict:
    """Write predictions, session scores, summary JSON, and comparison figure."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "dandi001340_loso_predictions.csv"
    session_path = output / "dandi001340_replay_sessions.csv"
    summary_path = output / "dandi001340_replay_manifest.json"
    figure_path = output / "fig_dopamine_replay.png"
    _write_csv(prediction_path, evaluation["predictions"])
    _write_csv(session_path, evaluation["session_scores"])
    manifest_payload = {
        "source_revision": None,
        "summary": evaluation["summary"],
        "parameters_by_training_fold":
            evaluation["parameters_by_training_fold"],
        "artifacts": {
            "predictions": prediction_path.name,
            "session_scores": session_path.name,
            "figure": figure_path.name,
        },
    }
    from .publication_artifacts import _source_revision
    manifest_payload["source_revision"] = _source_revision(
        Path(__file__).resolve().parents[2]
    )
    preparation_manifest = output / "dandi001340_preparation_manifest.json"
    if preparation_manifest.is_file():
        manifest_payload["preparation_manifest"] = {
            "path": preparation_manifest.name,
            "sha256": file_sha256(preparation_manifest),
        }
    if "quality_filtered_sensitivity" in evaluation:
        sensitivity = evaluation["quality_filtered_sensitivity"]
        manifest_payload["quality_filtered_sensitivity"] = {
            "summary": sensitivity["summary"],
            "parameters_by_training_fold":
                sensitivity["parameters_by_training_fold"],
        }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest_payload, stream, indent=2, sort_keys=True)
        stream.write("\n")

    import matplotlib.pyplot as plt

    condition_order = [
        condition for condition in (
            "plain_dlight", "device", "matched_exponential",
            "linear_device", "linear_matched_exponential", "no_trace",
            "shuffled_device", "shifted_device",
        )
        if condition in evaluation["summary"]["pooled_log_loss"]
    ]
    session_rows = evaluation["session_scores"]
    by_session = {
        (row["condition"], row["held_out_mouse"], row["session_id"]):
        row["log_loss"]
        for row in session_rows
    }
    baseline_keys = sorted({
        (row["held_out_mouse"], row["session_id"])
        for row in session_rows if row["condition"] == "previous_choice"
    })
    means, errors = [], [[], []]
    for index, condition in enumerate(condition_order):
        values = np.asarray([
            by_session[(condition, *key)]
            - by_session[("previous_choice", *key)]
            for key in baseline_keys
        ])
        mean = 1e4 * float(values.mean())
        lo, hi = bootstrap_ci(values, n_boot=10_000, seed=index)
        means.append(mean)
        errors[0].append(mean - 1e4 * lo)
        errors[1].append(1e4 * hi - mean)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    colors = [
        "#3aa07a", "#2f4b8f", "#d49b37", "#6f7fae",
        "#c7a95a", "#aaaaaa", "#b44b48", "#8b6bb1",
    ][:len(condition_order)]
    y = np.arange(len(condition_order))
    ax.errorbar(
        means, y, xerr=np.asarray(errors), fmt="none",
        ecolor="0.45", elinewidth=2.0, capsize=3,
    )
    ax.scatter(means, y, c=colors, s=42, zorder=3)
    labels = {
        "plain_dlight": "plain dLight",
        "device": "physical device",
        "matched_exponential": "physical matched exp.",
        "linear_device": "linear device",
        "linear_matched_exponential": "linear matched exp.",
        "no_trace": "no trace",
        "shuffled_device": "shuffled physical",
        "shifted_device": "shifted physical",
    }
    ax.set_yticks(y, [labels[name] for name in condition_order])
    ax.invert_yaxis()
    ax.axvline(0.0, color="0.35", lw=0.8)
    ax.set_xlabel(r"session-paired $\Delta$ log loss vs previous choice ($\times 10^4$)")
    outcome_loss = evaluation["summary"]["pooled_log_loss"].get("outcome_rl")
    suffix = (
        f"; outcome-RL control = {outcome_loss:.3f}"
        if outcome_loss is not None else ""
    )
    ax.set_title(
        f"DANDI 001340 logged replay - intact dLight versus controls{suffix}",
        loc="left",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", color="0.88", lw=0.5)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return {
        "predictions": str(prediction_path),
        "session_scores": str(session_path),
        "manifest": str(summary_path),
        "figure": str(figure_path),
    }
