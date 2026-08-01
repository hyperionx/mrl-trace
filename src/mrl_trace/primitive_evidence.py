"""Publication artifact orchestration for the focused primitive-evidence analysis."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .falsification import build_falsification_predictions
from .kernel_theory import kernel_phase_map, kernel_theory_manifest
from .predictive_linkage import (
    bootstrap_held_out_support,
    bootstrap_kww_parameters,
    default_candidates,
    grouped_held_out_scores,
    load_gold_traces,
    score_out_of_domain_candidates,
    working_information_criteria,
)
from .timing_benchmarks import (
    ABSTRACT_GAMMA_ID,
    MATCHED_EXPONENTIAL_ID,
    MATCHING_REGIMES,
    frozen_kernel_bank,
    run_matched_timing_benchmark,
)


PRIMITIVE_EVIDENCE_SCHEMA_VERSION = 1
TRAIN_BIASES = frozenset((0.8, 0.9, 1.1, 1.2, 1.4, 1.5))
OOD_BIASES = frozenset((1.7, 1.8))


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _digest(payload) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path, payload) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return path


def write_csv(path, rows) -> Path:
    path = Path(path)
    rows = [_jsonable(row) for row in rows]
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_identifiability_reference(gold_dir, *, profile: str) -> dict:
    """Run grouped prediction, support uncertainty and secondary diagnostics."""
    if profile not in {"reduced", "publication"}:
        raise ValueError("identifiability requires reduced or publication profile")
    n_grid = 120 if profile == "reduced" else 300
    support_samples = 200 if profile == "reduced" else 2_000
    parameter_samples = 40 if profile == "reduced" else 2_000
    traces = load_gold_traces(gold_dir, n_grid=n_grid)
    training = [trace for trace in traces if float(trace["bias"]) in TRAIN_BIASES]
    testing = [trace for trace in traces if float(trace["bias"]) in OOD_BIASES]
    candidates = default_candidates()
    comparison = grouped_held_out_scores(training, candidates)
    support = bootstrap_held_out_support(
        comparison, samples=support_samples, seed=20260801, tolerance=0.05
    )
    parameter_bootstrap = bootstrap_kww_parameters(
        training, samples=parameter_samples, seed=20260802,
        workers=4 if profile == "publication" else 1,
    )
    ood = score_out_of_domain_candidates(training, testing, candidates)
    information = working_information_criteria(training, candidates)
    best = min(comparison["mean_nrmse"], key=comparison["mean_nrmse"].get)
    threshold = 1.05 * comparison["mean_nrmse"][best]
    supported = sorted(
        name for name, value in comparison["mean_nrmse"].items()
        if value <= threshold
    )
    result = {
        "schema_version": PRIMITIVE_EVIDENCE_SCHEMA_VERSION,
        "analysis": "au_representation_identifiability",
        "profile": profile,
        "protocol": {
            "training_biases_v": sorted(TRAIN_BIASES),
            "untouched_diagnostic_biases_v": sorted(OOD_BIASES),
            "sampled_points_per_trace": n_grid,
            "support_bootstrap_samples": support_samples,
            "parameter_refit_bootstrap_samples": parameter_samples,
            "primary_statistic": "grouped_leave_one_bias_out_nrmse",
            "relative_support_tolerance": 0.05,
        },
        "grouped_lobo": comparison,
        "support_bootstrap": support,
        "kww_parameter_bootstrap": parameter_bootstrap,
        "out_of_domain_diagnostic": ood,
        "working_information_criteria": information,
        "selection": {
            "best_candidate": best,
            "threshold": threshold,
            "supported_candidates": supported,
            "cascade_depth_identified": support["cascade_depth_identified"],
        },
        "interpretation": (
            "The data resolve a compressed rise when the beta interval excludes one, "
            "but cascade depth is not identified unless the predeclared support rule passes."
        ),
        "claim_limit": (
            "All 18 in-domain records are repeated traces from one Au device; held-out "
            "bias prediction and repeat uncertainty do not establish device-population "
            "generality or microscopic stage count."
        ),
    }
    result["manifest_digest_sha256"] = _digest(result)
    return result


def build_primitive_evidence(*, profile: str, gold_dir=None,
                             identifiability_reference=None) -> dict:
    """Build smoke, reduced or publication primitive-evidence payloads."""
    if profile not in {"smoke", "reduced", "publication"}:
        raise ValueError("profile must be smoke, reduced or publication")
    theory_kwargs = {
        "voltages": (0.5, 0.9, 1.5) if profile == "smoke" else
                    (0.5, 0.7, 0.9, 1.2, 1.5),
        "tau_leaks_s": (2.0, 10.0) if profile == "smoke" else
                       (1.0, 2.0, 5.0, 10.0, 20.0),
        "depths": (3,) if profile == "smoke" else (2, 3, 4),
        "durations_s": (0.3,) if profile == "smoke" else (0.05, 0.3),
        "v_max_values": (1.0,) if profile == "smoke" else (0.5, 1.0, 2.0),
        "dt_s": 0.02 if profile == "smoke" else 0.01,
    }
    theory_rows = kernel_phase_map(**theory_kwargs)
    theory = kernel_theory_manifest(theory_rows, protocol={
        **{key: list(value) if isinstance(value, tuple) else value
           for key, value in theory_kwargs.items()},
        "profile": profile,
    })
    timing = run_matched_timing_benchmark(
        trials_per_block={"smoke": 4, "reduced": 16, "publication": 64}[profile],
        bootstrap_samples={"smoke": 100, "reduced": 1_000,
                           "publication": 10_000}[profile],
    )
    identifiability = None
    parameter_bootstrap = None
    if profile != "smoke":
        if gold_dir is not None and Path(gold_dir).is_dir():
            identifiability = build_identifiability_reference(gold_dir, profile=profile)
        elif profile == "reduced" and identifiability_reference is not None:
            reference_path = Path(identifiability_reference)
            if not reference_path.is_file():
                raise FileNotFoundError(
                    "reduced profile needs local Au traces or the tracked "
                    "identifiability reference"
                )
            identifiability = json.loads(reference_path.read_text(encoding="utf-8"))
            identifiability = {**identifiability,
                "reduced_profile_reuse": {
                    "source": str(reference_path),
                    "interpretation": (
                        "Tracked publication inference reused without refitting raw Au; "
                        "theory and timing tasks are rerun at reduced scale."
                    ),
                }}
        else:
            raise FileNotFoundError(f"{profile} profile requires --gold-dir")
        parameter_bootstrap = identifiability["kww_parameter_bootstrap"]
    falsification = build_falsification_predictions(
        parameter_bootstrap=parameter_bootstrap,
        samples={"smoke": 10, "reduced": 40, "publication": 400}[profile],
        dt_s=0.05 if profile == "smoke" else 0.02,
    )
    result = {
        "schema_version": PRIMITIVE_EVIDENCE_SCHEMA_VERSION,
        "analysis": "focused_primitive_evidence_strengthening",
        "profile": profile,
        "kernel_theory": theory,
        "identifiability": identifiability,
        "timing_benchmark": timing,
        "falsification_predictions": falsification,
        "verdicts": {
            "physical_is_globally_gamma": False,
            "bias_is_demonstrated_programmability": False,
            "timing_is_measured_hardware_learning": False,
            "falsification_predictions_are_tested": False,
            "energy_advantage_claimed": False,
        },
    }
    result["manifest_digest_sha256"] = _digest(result)
    return result


def write_primitive_artifacts(result: dict, output_dir) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "reference": write_json(output / "primitive_evidence_reference.json", result),
        "kernel_theory": write_json(output / "kernel_theory_reference.json",
                                    result["kernel_theory"]),
        "timing": write_json(output / "timing_benchmark_reference.json",
                             result["timing_benchmark"]),
        "falsification": write_json(output / "falsification_predictions.json",
                                    result["falsification_predictions"]),
    }
    write_csv(output / "kernel_phase_map.csv", result["kernel_theory"]["rows"])
    write_csv(output / "timing_evaluation.csv",
              result["timing_benchmark"]["per_block_metrics"])
    if result["identifiability"] is not None:
        paths["identifiability"] = write_json(
            output / "identifiability_reference.json", result["identifiability"]
        )
        write_csv(output / "identifiability_lobo.csv",
                  result["identifiability"]["grouped_lobo"]["rows"])
        write_csv(output / "identifiability_ood.csv",
                  result["identifiability"]["out_of_domain_diagnostic"]["rows"])
    return {key: str(value) for key, value in paths.items()}


def write_primitive_tex_macros(result: dict, path) -> Path:
    """Generate manuscript values directly from the primitive reference payload."""
    timing = result["timing_benchmark"]
    rms = timing["summary"]["pilot_rms"]
    names = {
        "physical_headroom_v1": "TimingPhysical",
        "linear_erlang_v1": "TimingLinear",
        ABSTRACT_GAMMA_ID: "TimingGamma",
        MATCHED_EXPONENTIAL_ID: "TimingExponential",
    }
    lines = ["% Generated from primitive_evidence_reference.json; do not edit."]
    for kernel, token in names.items():
        lines.append(
            f"\\newcommand{{\\{token}Loss}}{{{rms[kernel]['log_loss']:.3f}}}"
        )
        lines.append(
            f"\\newcommand{{\\{token}Accuracy}}{{{rms[kernel]['top1_accuracy']:.3f}}}"
        )
    identifiability = result.get("identifiability")
    if identifiability is not None:
        selection = identifiability["selection"]
        lines.append(
            f"\\newcommand{{\\IdentifiabilityBestCandidate}}"
            f"{{{selection['best_candidate'].replace('_', r'\_')}}}"
        )
        best_score = identifiability["grouped_lobo"]["mean_nrmse"][
            selection["best_candidate"]
        ]
        lines.append(
            f"\\newcommand{{\\IdentifiabilityBestNRMSE}}{{{best_score:.4f}}}"
        )
        beta = identifiability["kww_parameter_bootstrap"]["parameters"]["beta_fill"]
        lines.extend((
            f"\\newcommand{{\\BetaFillBootstrapMedian}}{{{beta['median']:.2f}}}",
            f"\\newcommand{{\\BetaFillBootstrapLo}}{{{beta['bootstrap_95ci'][0]:.2f}}}",
            f"\\newcommand{{\\BetaFillBootstrapHi}}{{{beta['bootstrap_95ci'][1]:.2f}}}",
        ))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def render_primitive_figures(result: dict, output_dir) -> dict[str, str]:
    """Render the two replacement main-text figures from tracked payloads."""
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    colors = {
        "physical_headroom_v1": "#2a9d6f", "linear_erlang_v1": "#3c78b5",
        ABSTRACT_GAMMA_ID: "#8757a5", MATCHED_EXPONENTIAL_ID: "#7f7f7f",
    }
    labels = {
        "physical_headroom_v1": "physical headroom",
        "linear_erlang_v1": "linear Erlang",
        ABSTRACT_GAMMA_ID: "abstract Gamma",
        MATCHED_EXPONENTIAL_ID: "matched exponential",
    }

    theory_rows = result["kernel_theory"]["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    selected = [row for row in theory_rows
                if row["cascade_depth"] == 3 and row["duration_s"] == 0.3
                and row["v_max"] == 1.0]
    for voltage in sorted({row["voltage_v"] for row in selected}):
        rows = sorted((row for row in selected if row["voltage_v"] == voltage),
                      key=lambda row: row["tau_leak_s"])
        axes[0, 0].plot([row["tau_leak_s"] for row in rows],
                        [row["linear_finite_pulse_peak_s"] for row in rows],
                        marker="o", label=f"{voltage:g} V")
    axes[0, 0].set(xscale="log", yscale="log", xlabel=r"$\tau_{leak}$ (s)",
                   ylabel=r"linear $t^*$ (s)", title="(a) analytic crossover")
    axes[0, 0].legend(frameon=False, fontsize=7)
    scatter = axes[0, 1].scatter(
        [row["physical_peak_occupancy_fraction"] for row in selected],
        [row["physical_peak_relative_departure"] for row in selected],
        c=[math.log10(row["rho_alpha_tau_leak"]) for row in selected],
        cmap="viridis", s=28,
    )
    axes[0, 1].set(xlabel="peak occupancy / $V_{max}$",
                   ylabel="departure from low-signal peak",
                   title="(b) nonlinear headroom departure")
    colorbar = fig.colorbar(scatter, ax=axes[0, 1], fraction=0.046, pad=0.04)
    colorbar.set_label(r"$\log_{10}(\alpha\tau_{leak})$", fontsize=8)

    ident = result.get("identifiability")
    if ident is None:
        axes[1, 0].text(0.5, 0.5, "Au identifiability requires\nreduced/publication profile",
                        ha="center", va="center")
        axes[1, 1].axis("off")
    else:
        names = list(ident["grouped_lobo"]["mean_nrmse"])
        short = {
            "kww": "KWW", **{f"linear_k{k}": f"L{k}" for k in range(2, 6)},
            **{f"physical_k{k}": f"P{k}" for k in range(2, 6)},
        }
        support = ident["support_bootstrap"]["candidate_summary"]
        means = [ident["grouped_lobo"]["mean_nrmse"][name] for name in names]
        low = [support[name]["bootstrap_95ci"][0] for name in names]
        high = [support[name]["bootstrap_95ci"][1] for name in names]
        x = np.arange(len(names))
        axes[1, 0].errorbar(x, means,
                            yerr=[np.asarray(means)-low, np.asarray(high)-means],
                            fmt="o", color="#3c78b5", capsize=2)
        axes[1, 0].set_xticks(x, [short[name] for name in names], fontsize=7)
        axes[1, 0].set(ylabel="held-out NRMSE",
                       title="(c) grouped leave-one-bias-out")
        axes[1, 0].text(0.02, 0.96, "L: linear Erlang; P: physical headroom",
                        transform=axes[1, 0].transAxes, va="top", fontsize=6.5)
        frequencies = [support[name]["within_5pct_frequency"] for name in names]
        axes[1, 1].bar(x, frequencies, color="#8757a5")
        axes[1, 1].axhline(0.95, color="0.3", ls="--", lw=0.8)
        axes[1, 1].set_xticks(x, [short[name] for name in names], fontsize=7)
        axes[1, 1].set(ylabel="within-5% bootstrap frequency", ylim=(0, 1.05),
                       title="(d) depth is not identified")
    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="0.9", lw=0.5)
    fig.tight_layout()
    theory_path = output / "fig_kernel_identifiability.png"
    fig.savefig(theory_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    timing = result["timing_benchmark"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    bank = frozen_kernel_bank(np.linspace(0, 60, 3001))
    for name, values in bank["values"].items():
        values = np.asarray(values)
        axes[0, 0].plot(bank["time_s"], values / max(float(values.max()), 1e-30),
                        color=colors[name], label=labels[name])
    axes[0, 0].set(xlabel="event-to-reward lag (s)", ylabel="unit-peak response",
                   title="(a) frozen kernel shapes")
    axes[0, 0].legend(frameon=False, fontsize=6.8)
    rows = [row for row in timing["per_block_metrics"]
            if row["matching"] == "pilot_rms"
            and row["task"] == "four_event_interval_attribution"]
    for name in bank["values"]:
        selected_rows = sorted((row for row in rows if row["kernel"] == name),
                               key=lambda row: row["target_delay_s"])
        axes[0, 1].plot([row["target_delay_s"] for row in selected_rows],
                        [row["log_loss"] for row in selected_rows], marker="o",
                        ms=3, color=colors[name], label=labels[name])
    axes[0, 1].set(xlabel="predeclared target delay (s)", ylabel="held-out log loss",
                   title="(b) interval attribution")
    order_rows = [row for row in timing["per_block_metrics"]
                  if row["matching"] == "pilot_rms"
                  and row["task"] == "two_cue_temporal_order"]
    names = list(bank["values"])
    means = [np.mean([row["top1_accuracy"] for row in order_rows
                     if row["kernel"] == name]) for name in names]
    axes[1, 0].bar(np.arange(len(names)), means,
                   color=[colors[name] for name in names])
    axes[1, 0].set_xticks(np.arange(len(names)),
                          [labels[name] for name in names], rotation=25,
                          ha="right", fontsize=7)
    axes[1, 0].set(ylabel="held-out accuracy", ylim=(0, 1.02),
                   title="(c) temporal order and scalar limit")
    prediction = result["falsification_predictions"]
    peak_rows = prediction["bias_conditioned_preferred_lag"]
    x = np.asarray([row["voltage_v"] for row in peak_rows])
    y = np.asarray([row["median_preferred_lag_s"] for row in peak_rows])
    lo = np.asarray([row["simultaneous_95band_s"][0] for row in peak_rows])
    hi = np.asarray([row["simultaneous_95band_s"][1] for row in peak_rows])
    axes[1, 1].fill_between(x, lo, hi, color="#2a9d6f", alpha=0.25)
    axes[1, 1].plot(x, y, color="#2a9d6f", marker="o")
    axes[1, 1].set(xlabel="model bias (V)", ylabel="predicted preferred lag (s)",
                   title="(d) untested frozen prediction")
    for axis in axes.ravel():
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="0.9", lw=0.5)
    fig.tight_layout()
    timing_path = output / "fig_timing_falsification.png"
    fig.savefig(timing_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"kernel_identifiability": str(theory_path),
            "timing_falsification": str(timing_path)}


__all__ = [
    "PRIMITIVE_EVIDENCE_SCHEMA_VERSION", "TRAIN_BIASES", "OOD_BIASES",
    "build_identifiability_reference", "build_primitive_evidence",
    "write_primitive_artifacts", "write_primitive_tex_macros",
    "render_primitive_figures",
]
