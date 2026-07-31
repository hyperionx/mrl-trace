"""Reusable empirical model-comparison primitives driven by notebook 06.

These functions operate on measured response traces and never configure a learning
experiment.  The separation is intentional: empirical representation scores may be
recorded beside a shared model-specification digest, but cannot mutate that spec or
provide fitted parameters to synthetic or logged-replay workflows.
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
from scipy.special import gammainc

from .model_specs import PRIMARY_MODEL_ID, LINEAR_MODEL_ID


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
        for trace in traces:
            observed = np.asarray(trace["current"], dtype=float)
            shape = candidate_response(
                trace["time"], trace["bias"], theta, representation, k=k
            )
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
    rows = []
    for trace in traces:
        _, amplitude, offset, nrmse = affine_trace_score(
            candidate_response(
                trace["time"], trace["bias"], theta, representation, k=k
            ),
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
    "candidate_response", "affine_trace_score", "fit_candidate",
    "score_candidate", "grouped_held_out_scores", "default_candidates",
]
