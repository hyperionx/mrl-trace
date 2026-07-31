"""Versioned device-model identities shared by every scientific workflow.

The specifications in this module are configuration and provenance records, not
results selected by a notebook.  Notebook 06 evaluates these frozen candidates;
the remaining experiments consume the same identifiers directly and therefore do
not depend on notebook execution order or on an empirical result manifest.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

MODEL_SPEC_SCHEMA_VERSION = 1
PRIMARY_MODEL_ID = "physical_headroom_v1"
LINEAR_MODEL_ID = "linear_erlang_v1"

_TAU_R_LAW = {
    "expression": "145*exp(-2.9*abs(V))",
    "prefactor_s": 145.0,
    "field_coefficient_per_v": 2.9,
    "source": "global Au transient fit; kww_final.json",
    "source_sha256": "c261d052b796a31c7278010315c71e8572268755e8bcd8a527018d00bc552724",
}

_BASE_SPECS = {
    PRIMARY_MODEL_ID: {
        "model_id": PRIMARY_MODEL_ID,
        "equation_family": "nonlinear_normalized_headroom_cascade",
        "primary": True,
        "default_voltage_v": 0.9,
        "default_cascade_depth": 3,
        "default_beta_leak": 1.0,
        "upstream_transfer": "drive for stage 1; v_previous/V_max thereafter",
        "claim_limit": (
            "Equation-matching computational device model. Its effective cascade "
            "depth is a representation parameter, not an identified microscopic "
            "trap count."
        ),
    },
    LINEAR_MODEL_ID: {
        "model_id": LINEAR_MODEL_ID,
        "equation_family": "linear_erlang_cascade",
        "primary": False,
        "default_voltage_v": 0.9,
        "default_cascade_depth": 3,
        "default_beta_leak": 1.0,
        "upstream_transfer": "V_max*drive for stage 1; raw v_previous thereafter",
        "claim_limit": (
            "Erlang-exact computational sensitivity. It is not the nonlinear "
            "headroom equation and does not identify microscopic stages."
        ),
    },
}


def _canonical_digest(payload: dict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def device_model_spec(model_id: str = PRIMARY_MODEL_ID) -> dict:
    """Return an immutable-by-convention copy of a frozen model specification."""
    model_id = str(model_id)
    if model_id not in _BASE_SPECS:
        raise ValueError(
            f"unknown gate model {model_id!r}; expected one of {tuple(_BASE_SPECS)}"
        )
    payload = {
        "schema_version": MODEL_SPEC_SCHEMA_VERSION,
        **deepcopy(_BASE_SPECS[model_id]),
        "tau_r_law": deepcopy(_TAU_R_LAW),
    }
    payload["tau_r_law"]["digest_sha256"] = _canonical_digest(_TAU_R_LAW)
    payload["spec_digest_sha256"] = _canonical_digest(payload)
    return payload


def model_spec_digest(model_id: str = PRIMARY_MODEL_ID) -> str:
    return str(device_model_spec(model_id)["spec_digest_sha256"])


def known_model_ids() -> tuple[str, ...]:
    return tuple(_BASE_SPECS)


def select_supported_state_space(scores: dict[str, float], tolerance: float = 0.05) -> dict:
    """Select candidates within ``tolerance`` of the lowest held-out error."""
    if not scores:
        raise ValueError("scores must not be empty")
    values = {str(key): float(value) for key, value in scores.items()}
    if any(not (value >= 0.0 and value < float("inf")) for value in values.values()):
        raise ValueError("candidate scores must be finite and non-negative")
    tolerance = float(tolerance)
    if not 0.0 <= tolerance < 1.0:
        raise ValueError("tolerance must be in [0, 1)")
    best_name = min(values, key=lambda name: (values[name], name))
    best_score = values[best_name]
    threshold = best_score * (1.0 + tolerance)
    supported = sorted(name for name, value in values.items() if value <= threshold)
    physical_values = [
        value for name, value in values.items()
        if name == PRIMARY_MODEL_ID or name.startswith("physical_k")
    ]
    physical_score = min(physical_values) if physical_values else None
    return {
        "best_candidate": best_name,
        "best_score": best_score,
        "relative_tolerance": tolerance,
        "threshold": threshold,
        "supported_candidates": supported,
        "physical_within_tolerance": (
            None if physical_score is None else bool(physical_score <= threshold)
        ),
    }


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_predictive_linkage_manifest(
    *,
    candidate_scores: dict[str, float],
    source_digests: dict[str, str],
    parameter_estimates: dict,
    retention_definition: str,
    software_versions: dict[str, str],
    tolerance: float = 0.05,
) -> dict:
    """Build a deterministic, JSON-safe notebook-06 provenance manifest."""
    selection = select_supported_state_space(candidate_scores, tolerance=tolerance)
    payload = {
        "schema_version": 1,
        "analysis": "predictive_device_model_linkage",
        "candidate_scores": {
            key: float(candidate_scores[key]) for key in sorted(candidate_scores)
        },
        "selection": selection,
        "model_specifications": {
            model_id: device_model_spec(model_id) for model_id in known_model_ids()
        },
        "source_digests": dict(sorted(source_digests.items())),
        "parameter_estimates": parameter_estimates,
        "retention_definition": str(retention_definition),
        "software_versions": dict(sorted(software_versions.items())),
        "claim_limit": (
            "Empirical representation scoring is independent of downstream learning "
            "performance and does not identify microscopic cascade depth."
        ),
    }
    payload["manifest_digest_sha256"] = _canonical_digest(payload)
    return payload


__all__ = [
    "MODEL_SPEC_SCHEMA_VERSION",
    "PRIMARY_MODEL_ID",
    "LINEAR_MODEL_ID",
    "device_model_spec",
    "model_spec_digest",
    "known_model_ids",
    "select_supported_state_space",
    "file_sha256",
    "build_predictive_linkage_manifest",
]
