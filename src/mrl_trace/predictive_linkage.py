"""Reusable empirical model-comparison primitives driven by notebook 06.

These functions operate on measured response traces and never configure a learning
experiment.  The separation is intentional: empirical representation scores may be
recorded beside a shared model-specification digest, but cannot mutate that spec or
provide fitted parameters to synthetic or logged-replay workflows.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import gammainc

from .model_specs import PRIMARY_MODEL_ID, LINEAR_MODEL_ID


_BOOTSTRAP_TRACES = None
_BOOTSTRAP_GROUP_INDICES = None
_BOOTSTRAP_MAX_NFEV = 1200


def _initialise_kww_bootstrap_worker(traces, group_indices, max_nfev):
    global _BOOTSTRAP_TRACES, _BOOTSTRAP_GROUP_INDICES, _BOOTSTRAP_MAX_NFEV
    _BOOTSTRAP_TRACES = traces
    _BOOTSTRAP_GROUP_INDICES = group_indices
    _BOOTSTRAP_MAX_NFEV = int(max_nfev)


def _fit_kww_bootstrap_indices(draw):
    sample = []
    for group_draw, source_indices in zip(draw, _BOOTSTRAP_GROUP_INDICES):
        sample.extend(_BOOTSTRAP_TRACES[source_indices[int(index)]]
                      for index in group_draw)
    return fit_candidate(sample, "kww", max_nfev=_BOOTSTRAP_MAX_NFEV)


GOLD_TRACE_RE = re.compile(r"trace_V([mp])(\d+\.\d+)_tr(\d+)\.csv")


def load_gold_traces(directory, *, n_grid: int = 240) -> list[dict]:
    """Load the 24 Au CSV records as replicate-level trace dictionaries.

    This source-level loader is shared by notebook 06, the reproduction CLI and
    tests.  Biases 1.7/1.8 V are loaded but remain outside model selection.
    """
    directory = Path(directory)
    traces: list[dict] = []
    for path in sorted(directory.glob("trace_*.csv")):
        match = GOLD_TRACE_RE.fullmatch(path.name)
        if match is None:
            continue
        sign, magnitude, trial = match.groups()
        array = np.genfromtxt(path, delimiter=",", names=True)
        time = np.asarray(array["time"], dtype=float)
        current = np.abs(np.asarray(array["current"], dtype=float))
        valid = np.isfinite(time) & np.isfinite(current) & (time >= 0) & (current > 0)
        time, current = time[valid], current[valid]
        if time.size < 20:
            raise ValueError(f"too few valid Au samples in {path.name}")
        indices = np.unique(np.linspace(
            0, time.size - 1, min(int(n_grid), time.size)
        ).astype(int))
        traces.append({
            "filename": path.name,
            "bias": float(magnitude),
            "signed_bias": (-1 if sign == "m" else 1) * float(magnitude),
            "trial": int(trial),
            "time": time[indices] - time[indices][0],
            "current": current[indices],
        })
    if len(traces) != 24:
        raise ValueError(f"expected 24 Au traces, found {len(traces)}")
    return traces


def candidate_response(time, bias_v, theta, representation, *, k=None):
    """Return a unit-scale candidate transient on the supplied timestamps.

    ``theta`` stores ``log(tau_r0), c_r, log(tau_d0), c_d`` and, for KWW,
    ``log(beta)``.  Physical-headroom candidates integrate the same normalized
    upstream/headroom/leakage equation used by the learning gate.  Linear candidates
    retain the prior Erlang-CDF rise multiplied by the explicit discharge envelope.
    """
    time = np.asarray(time, dtype=float)
    theta = np.asarray(theta, dtype=float)
    if time.ndim != 1 or np.any(~np.isfinite(time)) or np.any(time < 0):
        raise ValueError("time must be a finite non-negative one-dimensional array")
    tau_r = math.exp(theta[0]) * math.exp(-float(theta[1]) * abs(float(bias_v)))
    tau_d = math.exp(theta[2]) * math.exp(-float(theta[3]) * abs(float(bias_v)))
    if representation == "kww":
        beta = math.exp(theta[4])
        rise = 1.0 - np.exp(-np.power(time / tau_r, beta))
        return rise * np.exp(-time / tau_d)
    if k is None or int(k) < 1:
        raise ValueError("state-space candidates require a positive k")
    k = int(k)
    if representation == LINEAR_MODEL_ID:
        return gammainc(k, k * time / tau_r) * np.exp(-time / tau_d)
    if representation != PRIMARY_MODEL_ID:
        raise ValueError(f"unknown representation: {representation!r}")
    if time.size == 0:
        return time.copy()
    order = np.argsort(time)
    sorted_time = time[order]
    unique_time, inverse = np.unique(sorted_time, return_inverse=True)
    alpha = k / tau_r

    def rhs(_, state):
        upstream = np.empty(k, dtype=float)
        upstream[0] = 1.0
        upstream[1:] = state[:-1]
        return alpha * upstream * (1.0 - np.abs(state)) - state / tau_d

    if unique_time[-1] == 0:
        sorted_values = np.zeros_like(sorted_time)
    else:
        solution = solve_ivp(
            rhs, (0.0, float(unique_time[-1])), np.zeros(k),
            t_eval=unique_time, method="LSODA", rtol=1e-6, atol=1e-9,
        )
        if not solution.success:
            raise RuntimeError(f"physical candidate integration failed: {solution.message}")
        sorted_values = solution.y[-1, inverse]
    values = np.empty_like(sorted_values)
    values[order] = sorted_values
    return values


def affine_trace_score(shape, observed):
    """Fit amplitude/offset and return prediction plus range-normalised RMSE."""
    shape = np.asarray(shape, dtype=float)
    observed = np.asarray(observed, dtype=float)
    if shape.shape != observed.shape or shape.ndim != 1:
        raise ValueError("shape and observed must be same-length vectors")
    design = np.column_stack((shape, np.ones_like(shape)))
    coefficient, *_ = np.linalg.lstsq(design, observed, rcond=None)
    prediction = design @ coefficient
    scale = max(float(np.ptp(observed)), float(np.max(np.abs(observed))), 1e-15)
    score = float(np.sqrt(np.mean(np.square(prediction - observed))) / scale)
    return prediction, float(coefficient[0]), float(coefficient[1]), score


def _candidate_shapes(traces, theta, representation, *, k=None):
    """Evaluate candidate shapes, batching physical ODEs across bias groups."""
    traces = list(traces)
    if representation != PRIMARY_MODEL_ID:
        return [candidate_response(trace["time"], trace["bias"], theta,
                                   representation, k=k) for trace in traces]
    if k is None or int(k) < 1:
        raise ValueError("physical candidates require a positive k")
    k = int(k)
    biases = sorted({float(trace["bias"]) for trace in traces})
    all_time = np.unique(np.concatenate([
        np.asarray(trace["time"], dtype=float) for trace in traces
    ]))
    if all_time.size == 0 or all_time[-1] == 0:
        return [np.zeros_like(np.asarray(trace["time"], dtype=float))
                for trace in traces]
    theta = np.asarray(theta, dtype=float)
    rise = np.asarray([
        math.exp(theta[0]) * math.exp(-float(theta[1]) * abs(bias))
        for bias in biases
    ])
    decay = np.asarray([
        math.exp(theta[2]) * math.exp(-float(theta[3]) * abs(bias))
        for bias in biases
    ])
    alpha = k / rise

    def rhs(_, flat_state):
        state = flat_state.reshape(len(biases), k)
        upstream = np.empty_like(state)
        upstream[:, 0] = 1.0
        upstream[:, 1:] = state[:, :-1]
        derivative = (
            alpha[:, None] * upstream * (1.0 - np.abs(state))
            - state / decay[:, None]
        )
        return derivative.ravel()

    solution = solve_ivp(
        rhs, (0.0, float(all_time[-1])), np.zeros(len(biases) * k),
        t_eval=all_time, method="LSODA", rtol=1e-6, atol=1e-9,
    )
    if not solution.success:
        raise RuntimeError(f"physical candidate integration failed: {solution.message}")
    state = solution.y.reshape(len(biases), k, all_time.size)
    bias_index = {bias: index for index, bias in enumerate(biases)}
    return [
        np.interp(np.asarray(trace["time"], dtype=float), all_time,
                  state[bias_index[float(trace["bias"])], -1])
        for trace in traces
    ]


def fit_candidate(traces: Iterable[dict], representation, *, k=None,
                  max_nfev=2500):
    """Robustly fit one shared field law with per-trace amplitude and offset."""
    traces = list(traces)
    if not traces:
        raise ValueError("traces must not be empty")
    initial = np.asarray([math.log(145.0), 2.9, math.log(11700.0), 3.9])
    lower = np.asarray([math.log(0.1), 0.0, math.log(1.0), 0.0])
    upper = np.asarray([math.log(1e5), 10.0, math.log(1e7), 10.0])
    if representation == "kww":
        initial = np.r_[initial, math.log(2.0)]
        lower = np.r_[lower, math.log(0.4)]
        upper = np.r_[upper, math.log(4.0)]

    def residual(theta):
        chunks = []
        shapes = _candidate_shapes(traces, theta, representation, k=k)
        for trace, shape in zip(traces, shapes):
            observed = np.asarray(trace["current"], dtype=float)
            prediction, amplitude, _, _ = affine_trace_score(shape, observed)
            scale = max(float(np.ptp(observed)),
                        float(np.max(np.abs(observed))), 1e-15)
            chunk = (prediction - observed) / scale
            # Optimizers require a fixed residual dimension.  Keep one amplitude
            # penalty slot per trace even when the fitted amplitude is positive.
            amplitude_penalty = max(
                0.0, min(10.0, 1.0 - amplitude / scale)
            ) if amplitude <= 0 else 0.0
            chunk = np.r_[chunk, amplitude_penalty]
            chunks.append(chunk)
        return np.concatenate(chunks)

    result = least_squares(
        residual, initial, bounds=(lower, upper), method="trf",
        loss="soft_l1", f_scale=0.02, max_nfev=int(max_nfev),
    )
    if not result.success:
        raise RuntimeError(
            f"fit failed for {representation} k={k}: {result.message}"
        )
    return result.x


def score_candidate(traces: Iterable[dict], theta, representation, *, k=None):
    traces = list(traces)
    shapes = _candidate_shapes(traces, theta, representation, k=k)
    rows = []
    for trace, shape in zip(traces, shapes):
        _, amplitude, offset, nrmse = affine_trace_score(
            shape,
            trace["current"],
        )
        rows.append({
            "bias": float(trace["bias"]),
            "trial": int(trace.get("trial", 0)),
            "nrmse": nrmse,
            "amplitude": amplitude,
            "offset": offset,
        })
    return rows


def grouped_held_out_scores(traces: Iterable[dict], candidates, *, group_key="bias"):
    """Fit on all-but-one group and score every frozen candidate on that group."""
    traces = list(traces)
    groups = sorted({float(trace[group_key]) for trace in traces})
    rows = []
    for held_out in groups:
        training = [trace for trace in traces if float(trace[group_key]) != held_out]
        testing = [trace for trace in traces if float(trace[group_key]) == held_out]
        for candidate in candidates:
            name = str(candidate["name"])
            representation = str(candidate["representation"])
            k = candidate.get("k")
            theta = fit_candidate(training, representation, k=k)
            for row in score_candidate(testing, theta, representation, k=k):
                rows.append({"held_out_group": held_out, "candidate": name, **row})
    means = {
        str(candidate["name"]): float(np.mean([
            row["nrmse"] for row in rows
            if row["candidate"] == str(candidate["name"])
        ]))
        for candidate in candidates
    }
    return {"rows": rows, "mean_nrmse": means, "groups": groups}


def bootstrap_held_out_support(comparison: dict, *, samples: int = 2_000,
                               seed: int = 20260801,
                               tolerance: float = 0.05) -> dict:
    """Repeat-stratified bootstrap of already held-out predictions.

    Each all-but-one-bias fit remains frozen.  Within every held-out bias, recorded
    repeats are resampled with replacement and all candidate scores are recomputed
    from the paired prediction rows.  This is deliberately labelled a *prediction
    rescore* bootstrap, not a computationally prohibitive full model-refit bootstrap.
    """
    if int(samples) < 1 or not 0 <= float(tolerance) < 1:
        raise ValueError("samples must be positive and tolerance non-negative")
    rows = list(comparison["rows"])
    candidates = sorted({str(row["candidate"]) for row in rows})
    groups = sorted({float(row["held_out_group"]) for row in rows})
    trials_by_group = {
        group: sorted({int(row["trial"]) for row in rows
                       if float(row["held_out_group"]) == group})
        for group in groups
    }
    repeats = {len(value) for value in trials_by_group.values()}
    if len(repeats) != 1:
        raise ValueError("every held-out bias must contain the same repeat count")
    n_repeat = repeats.pop()
    lookup = {
        (float(row["held_out_group"]), int(row["trial"]), str(row["candidate"])):
        float(row["nrmse"])
        for row in rows
    }
    score = np.empty((len(groups), n_repeat, len(candidates)), dtype=float)
    for group_index, group in enumerate(groups):
        for repeat_index, trial in enumerate(trials_by_group[group]):
            for candidate_index, candidate in enumerate(candidates):
                score[group_index, repeat_index, candidate_index] = lookup[
                    (group, trial, candidate)
                ]
    rng = np.random.default_rng(int(seed))
    boot = np.empty((int(samples), len(candidates)), dtype=float)
    for start in range(0, int(samples), 1000):
        stop = min(int(samples), start + 1000)
        draw = rng.integers(
            0, n_repeat, size=(stop - start, len(groups), n_repeat)
        )
        gathered = np.take_along_axis(
            np.broadcast_to(score[None, ...],
                            (stop - start, *score.shape)),
            draw[..., None], axis=2,
        )
        boot[start:stop] = gathered.mean(axis=(1, 2))
    best = np.min(boot, axis=1)
    within = boot <= (1.0 + float(tolerance)) * best[:, None]
    winners = np.argmin(boot, axis=1)
    summaries = {}
    for index, candidate in enumerate(candidates):
        low, high = np.percentile(boot[:, index], (2.5, 97.5))
        summaries[candidate] = {
            "point_mean_nrmse": float(comparison["mean_nrmse"][candidate]),
            "bootstrap_mean_nrmse": float(np.mean(boot[:, index])),
            "bootstrap_95ci": [float(low), float(high)],
            "best_frequency": float(np.mean(winners == index)),
            "within_5pct_frequency": float(np.mean(within[:, index])),
        }
    linear_depths = [name for name in candidates if name.startswith("linear_k")]
    strongly_supported = [
        name for name in linear_depths
        if summaries[name]["within_5pct_frequency"] >= 0.95
    ]
    excluded = [
        name for name in linear_depths
        if summaries[name]["within_5pct_frequency"] <= 0.05
    ]
    depth_identified = bool(
        len(strongly_supported) == 1
        and len(excluded) == len(linear_depths) - 1
    )
    return {
        "method": "repeat_stratified_bootstrap_of_frozen_lobo_predictions",
        "full_refit": False,
        "samples": int(samples), "seed": int(seed),
        "relative_support_tolerance": float(tolerance),
        "groups": groups, "repeats_per_group": n_repeat,
        "candidate_summary": summaries,
        "cascade_depth_identified": depth_identified,
        "depth_identification_rule": (
            "exactly one linear k has within-tolerance frequency >=0.95 and all "
            "other linear k candidates have frequency <=0.05"
        ),
        "interpretation": (
            "Uncertainty in held-out prediction scores is resampled at the recorded "
            "repeat level; training-fold kinetics remain frozen."
        ),
    }


def working_information_criteria(traces: Iterable[dict], candidates) -> dict:
    """Secondary AICc/BIC check under an explicitly Gaussian working likelihood."""
    traces = list(traces)
    rows = []
    for candidate in candidates:
        representation = str(candidate["representation"])
        k = candidate.get("k")
        theta = fit_candidate(traces, representation, k=k)
        residual_chunks = []
        for trace in traces:
            prediction, _, _, _ = affine_trace_score(
                candidate_response(trace["time"], trace["bias"], theta,
                                   representation, k=k),
                trace["current"],
            )
            residual_chunks.append(
                np.asarray(prediction) - np.asarray(trace["current"], dtype=float)
            )
        residual = np.concatenate(residual_chunks)
        n = int(residual.size)
        rss = max(float(np.sum(np.square(residual))), np.finfo(float).tiny)
        # Shared kinetics + two affine nuisance terms per trace + one noise scale.
        parameter_count = int(theta.size + 2 * len(traces) + 1)
        log_likelihood = -0.5 * n * (
            math.log(2.0 * math.pi) + 1.0 + math.log(rss / n)
        )
        aic = 2.0 * parameter_count - 2.0 * log_likelihood
        correction = (
            2.0 * parameter_count * (parameter_count + 1)
            / (n - parameter_count - 1)
            if n > parameter_count + 1 else math.inf
        )
        rows.append({
            "candidate": str(candidate["name"]), "observations": n,
            "parameter_count": parameter_count,
            "working_log_likelihood": log_likelihood,
            "aicc": aic + correction,
            "bic": parameter_count * math.log(n) - 2.0 * log_likelihood,
            "rss": rss,
        })
    min_aicc = min(row["aicc"] for row in rows)
    min_bic = min(row["bic"] for row in rows)
    for row in rows:
        row["delta_aicc"] = float(row["aicc"] - min_aicc)
        row["delta_bic"] = float(row["bic"] - min_bic)
    return {
        "method": "secondary_gaussian_working_likelihood",
        "primary_for_model_selection": False,
        "assumptions": (
            "Common sampled time grid; independent Gaussian residual working model; "
            "global kinetics plus per-trace amplitude/offset and one noise scale."
        ),
        "claim_limit": (
            "Trace residuals are temporally correlated and the kinetics were fitted "
            "with robust loss; held-out prediction takes precedence."
        ),
        "rows": rows,
    }


def bootstrap_kww_parameters(traces: Iterable[dict], *, samples: int = 2_000,
                             seed: int = 20260802, workers: int = 1,
                             max_nfev: int = 1200) -> dict:
    """Refit KWW field-law parameters after repeat-stratified resampling."""
    traces = list(traces)
    groups = sorted({float(trace["bias"]) for trace in traces})
    by_group = {
        group: [trace for trace in traces if float(trace["bias"]) == group]
        for group in groups
    }
    if any(len(group) < 2 for group in by_group.values()):
        raise ValueError("each bias needs at least two recorded repeats")
    rng = np.random.default_rng(int(seed))
    group_indices = []
    trace_index = {id(trace): index for index, trace in enumerate(traces)}
    for group in groups:
        group_indices.append([trace_index[id(trace)] for trace in by_group[group]])
    draws = []
    for _ in range(int(samples)):
        draw = []
        for group in groups:
            records = by_group[group]
            draw.append(rng.integers(0, len(records), size=len(records)).tolist())
        draws.append(draw)

    if int(workers) > 1:
        with ProcessPoolExecutor(
            max_workers=int(workers), initializer=_initialise_kww_bootstrap_worker,
            initargs=(traces, group_indices, max_nfev),
        ) as executor:
            theta = np.asarray(list(executor.map(
                _fit_kww_bootstrap_indices, draws, chunksize=20
            )))
    else:
        _initialise_kww_bootstrap_worker(traces, group_indices, max_nfev)
        theta = np.asarray([_fit_kww_bootstrap_indices(draw) for draw in draws])
    transformed = {
        "tau_r0_s": np.exp(theta[:, 0]),
        "rise_field_coefficient_per_v": theta[:, 1],
        "tau_d0_s": np.exp(theta[:, 2]),
        "decay_field_coefficient_per_v": theta[:, 3],
        "beta_fill": np.exp(theta[:, 4]),
    }
    summary = {}
    for name, values in transformed.items():
        low, high = np.percentile(values, (2.5, 97.5))
        summary[name] = {
            "mean": float(np.mean(values)), "median": float(np.median(values)),
            "bootstrap_95ci": [float(low), float(high)],
        }
    return {
        "method": "repeat_stratified_full_refit_kww_parameter_bootstrap",
        "samples": int(samples), "seed": int(seed), "groups": groups,
        "workers": int(workers), "parameters": summary,
        "parameter_draws": {
            name: values.tolist() for name, values in transformed.items()
        },
        "compressed_rise_resolved": bool(
            summary["beta_fill"]["bootstrap_95ci"][0] > 1.0
        ),
    }


def score_out_of_domain_candidates(training: Iterable[dict], testing: Iterable[dict],
                                   candidates) -> dict:
    """Fit on the in-domain bias set and score untouched high-bias diagnostics."""
    training, testing = list(training), list(testing)
    rows = []
    for candidate in candidates:
        representation = str(candidate["representation"])
        k = candidate.get("k")
        theta = fit_candidate(training, representation, k=k)
        for row in score_candidate(testing, theta, representation, k=k):
            rows.append({"candidate": str(candidate["name"]), **row})
    return {
        "rows": rows,
        "mean_nrmse": {
            str(candidate["name"]): float(np.mean([
                row["nrmse"] for row in rows
                if row["candidate"] == str(candidate["name"])
            ]))
            for candidate in candidates
        },
        "interpretation": "untouched 1.7/1.8-V extrapolation diagnostic; no refitting",
    }


def default_candidates():
    """KWW plus physical and linear state-space representations at k=2--5."""
    return (
        {"name": "kww", "representation": "kww"},
        *(
            {"name": f"physical_k{k}", "representation": PRIMARY_MODEL_ID, "k": k}
            for k in range(2, 6)
        ),
        *(
            {"name": f"linear_k{k}", "representation": LINEAR_MODEL_ID, "k": k}
            for k in range(2, 6)
        ),
    )


__all__ = [
    "load_gold_traces", "candidate_response", "affine_trace_score",
    "fit_candidate", "score_candidate", "grouped_held_out_scores",
    "bootstrap_held_out_support", "working_information_criteria",
    "bootstrap_kww_parameters", "score_out_of_domain_candidates",
    "default_candidates",
]
