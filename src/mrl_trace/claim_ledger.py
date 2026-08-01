"""Validation of manuscript claims against tracked machine-readable artifacts."""
from __future__ import annotations

import json
import math
from pathlib import Path

from .model_specs import known_model_ids, model_spec_digest


def _pointer(payload, pointer: str):
    if not pointer.startswith("/"):
        raise ValueError(f"JSON pointer must start with '/': {pointer}")
    value = payload
    for raw in pointer.lstrip("/").split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


def load_claim_ledger(path) -> dict:
    """Load ``claims.yaml``; JSON is used as a dependency-free YAML subset."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("claims"), list):
        raise ValueError("claim ledger must use schema version 1 and contain claims")
    return payload


def validate_claim_ledger(path, *, repository_root=None,
                          manuscript_path=None) -> dict:
    ledger_path = Path(path).resolve()
    root = Path(repository_root or ledger_path.parent).resolve()
    manuscript = (Path(manuscript_path).read_text(encoding="utf-8")
                  if manuscript_path is not None else None)
    payload = load_claim_ledger(ledger_path)
    seen, rows = set(), []
    for claim in payload["claims"]:
        claim_id = str(claim["claim_id"])
        if claim_id in seen:
            raise ValueError(f"duplicate claim id: {claim_id}")
        seen.add(claim_id)
        artifact = (root / claim["artifact"]).resolve()
        if not artifact.is_file():
            raise FileNotFoundError(f"claim {claim_id} artifact missing: {artifact}")
        document = json.loads(artifact.read_text(encoding="utf-8"))
        value = _pointer(document, claim["json_pointer"])
        if "expected" in claim:
            expected = claim["expected"]
            tolerance = float(claim.get("tolerance", 0.0))
            if isinstance(expected, (int, float)) and isinstance(value, (int, float)):
                if not math.isclose(float(value), float(expected), abs_tol=tolerance,
                                    rel_tol=tolerance):
                    raise AssertionError(
                        f"claim {claim_id} drifted: {value!r} != {expected!r}"
                    )
            elif value != expected:
                raise AssertionError(
                    f"claim {claim_id} drifted: {value!r} != {expected!r}"
                )
        model_id = claim.get("model_id")
        if model_id is not None:
            if model_id not in known_model_ids():
                raise ValueError(f"claim {claim_id} uses unknown model {model_id}")
            declared = claim.get("model_spec_digest")
            if declared != model_spec_digest(model_id):
                raise AssertionError(f"claim {claim_id} model digest drifted")
        anchor = claim.get("manuscript_anchor")
        if manuscript is not None and anchor and anchor not in manuscript:
            raise AssertionError(f"claim {claim_id} manuscript anchor not found")
        for required in ("producer", "seed_protocol"):
            if not str(claim.get(required, "")).strip():
                raise ValueError(f"claim {claim_id} lacks {required}")
        rows.append({"claim_id": claim_id, "artifact": str(artifact), "value": value})
    return {"schema_version": 1, "claims_validated": len(rows), "rows": rows}


__all__ = ["load_claim_ledger", "validate_claim_ledger"]
