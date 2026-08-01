"""Clean-clone reproduction command for smoke, reduced and publication profiles."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from .dopamine import DANDI001340_VERDICT
from .kernel_theory import (
    linear_impulse_peak,
    linear_rectangular_peak,
    linear_rectangular_response,
)
from .paths import gold_export_dir, publication_device_preflight, results_dir
from .primitive_evidence import (
    build_primitive_evidence,
    render_primitive_figures,
    write_primitive_artifacts,
    write_primitive_tex_macros,
)


def _smoke_identities() -> dict:
    tau_r_s, tau_leak_s, k, duration = 4.0, 10.0, 3, 0.01
    impulse_peak = linear_impulse_peak(tau_r_s, tau_leak_s, k)
    finite_peak = linear_rectangular_peak(tau_r_s, tau_leak_s, k, duration)
    time = np.linspace(0.0, 20.0, 4001)
    response = linear_rectangular_response(
        time, tau_r_s, tau_leak_s, k, duration
    )
    numerical_peak = float(time[int(np.argmax(response))])
    if abs(numerical_peak - finite_peak) > 0.01:
        raise RuntimeError("finite-pulse analytic/numerical smoke identity failed")
    return {
        "linear_impulse_peak_s": impulse_peak,
        "linear_finite_pulse_peak_s": finite_peak,
        "sampled_numerical_peak_s": numerical_peak,
    }


def _validate_dandi_reference() -> dict:
    reference = results_dir() / "reference" / "dandi001340_replay_manifest.json"
    if not reference.is_file():
        raise FileNotFoundError(
            "--with-dandi requires the tracked DANDI replay manifest; raw rerun "
            "remains an independent opt-in workflow"
        )
    payload = json.loads(reference.read_text(encoding="utf-8"))
    summary = payload.get("summary", payload)
    text = json.dumps(summary)
    if DANDI001340_VERDICT not in text:
        raise RuntimeError(
            "DANDI reference does not retain the Conditional Go verdict"
        )
    return {"path": str(reference), "validated": True}


def reproduce(*, profile: str, output_dir=None, gold_dir=None, ito_dir=None,
              ito_archive=None, with_dandi: bool = False) -> dict:
    if profile not in {"smoke", "reduced", "publication"}:
        raise ValueError("profile must be smoke, reduced or publication")
    output = Path(output_dir or (
        Path.cwd() / "reproduction" / profile
    )).resolve()
    preflight = None
    selected_gold = Path(gold_dir).resolve() if gold_dir else gold_export_dir()
    if profile == "publication":
        preflight = publication_device_preflight(
            gold_raw_dir=selected_gold, ito_raw_dir=ito_dir,
            ito_archive=ito_archive,
        )
    tracked_identifiability = (
        results_dir() / "reference" / "identifiability_reference.json"
    )
    result = build_primitive_evidence(
        profile=profile,
        gold_dir=(selected_gold if profile != "smoke" and selected_gold.is_dir()
                  else None),
        identifiability_reference=(tracked_identifiability
                                   if profile == "reduced" else None),
    )
    artifacts = write_primitive_artifacts(result, output)
    figures = render_primitive_figures(result, output)
    macros = write_primitive_tex_macros(result, output / "primitive_macros.tex")
    report = {
        "profile": profile, "output_dir": str(output),
        "smoke_identities": _smoke_identities(),
        "publication_preflight": preflight,
        "primitive_manifest_digest_sha256": result["manifest_digest_sha256"],
        "artifacts": artifacts, "figures": figures, "tex_macros": str(macros),
        "dandi": _validate_dandi_reference() if with_dandi else {
            "requested": False,
            "note": "DANDI raw preparation/replay remains explicit and opt-in",
        },
    }
    (output / "reproduction_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "reduced", "publication"),
                        default="smoke")
    parser.add_argument("--output-dir")
    parser.add_argument("--gold-dir", default=os.environ.get("MRL_TRACE_GOLD_DIR"))
    parser.add_argument("--ito-dir", default=os.environ.get("MRL_TRACE_ITO_DIR"))
    parser.add_argument("--ito-archive")
    parser.add_argument("--with-dandi", action="store_true")
    arguments = parser.parse_args(argv)
    report = reproduce(
        profile=arguments.profile, output_dir=arguments.output_dir,
        gold_dir=arguments.gold_dir, ito_dir=arguments.ito_dir,
        ito_archive=arguments.ito_archive, with_dandi=arguments.with_dandi,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
