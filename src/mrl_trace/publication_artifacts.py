"""Writers and CLI for tracked, artifact-derived benchmark references."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np


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


def _source_revision(root: Path) -> str:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        return revision + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(payload), stream, indent=2, sort_keys=True,
                  allow_nan=False)
        stream.write("\n")


def sequential_reference(result: dict, *, source_revision: str) -> tuple[dict, list, list]:
    evaluation_rows = []
    pilot_rows = []
    seeds = result["evaluation_seeds"]
    for condition, metrics in result["per_seed_metrics"].items():
        eta = result["tuning"]["conditions"][condition]["selected_eta"]
        for index, seed in enumerate(seeds):
            evaluation_rows.append({
                "condition": condition, "seed": int(seed),
                "selected_eta": float(eta), "aulc": float(metrics["aulc"][index]),
                "final_reward": float(metrics["final_reward"][index]),
                "criterion_time": int(metrics["criterion_time"][index]),
                "right_censored": bool(metrics["right_censored"][index]),
            })
        for rate, score in result["tuning"]["conditions"][condition]["scores"].items():
            if score is None:
                continue
            for index, seed in enumerate(result["tuning"]["tuning_seeds"]):
                pilot_rows.append({
                    "condition": condition, "seed": int(seed), "eta": float(rate),
                    "aulc": float(score["per_seed_aulc"][index]),
                    "final_reward": float(score["per_seed_final_reward"][index]),
                })
    payload = {
        "schema_version": 1, "analysis": "sequential_learning_rate_fair",
        "source_revision": source_revision,
        "protocol": {
            "task": result["task"], "episodes": result["episodes"],
            "tau_leak_s": result["tau_leak"], "delay_s": result["delay"],
            "voltage_v": 1.5, "cascade_depth": 3, "beta_leak": result["beta_leak"],
            "calibration": result["calibration"]["protocol"],
            "selection_rule": result["tuning"]["selection_rule"],
            "seed_partition": result["seed_partition"],
            "positive_difference_is_better": True,
        },
        "model_specifications": result["model_specifications"],
        "calibration": result["calibration"], "tuning": result["tuning"],
        "per_seed_metrics": evaluation_rows,
        "paired_bootstrap": result["paired_bootstrap"],
        "verdicts": {
            name: row["verdict"] for name, row in result["paired_bootstrap"].items()
        },
    }
    return payload, evaluation_rows, pilot_rows


def dms_reference(result: dict, *, source_revision: str) -> tuple[dict, list, list]:
    seeds = result["seed_partition"]["evaluation"]
    evaluation_rows = []
    pilot_rows = []
    for condition in result["raw"]:
        eta = result["tuning"]["conditions"][condition]["selected_eta"]
        for index, seed in enumerate(seeds):
            evaluation_rows.append({
                "condition": condition, "seed": int(seed),
                "selected_eta": float(eta),
                "aulc": float(result["aulc"][condition][index]),
                "final_300_reward": float(result["finals"][condition][index]),
            })
        for rate, score in result["tuning"]["conditions"][condition]["scores"].items():
            if score is None:
                continue
            for index, seed in enumerate(result["tuning"]["tuning_seeds"]):
                pilot_rows.append({
                    "condition": condition, "seed": int(seed), "eta": float(rate),
                    "aulc": float(score["per_seed_aulc"][index]),
                })
    payload = {
        "schema_version": 1, "analysis": "shallow_dms_learning_rate_fair",
        "source_revision": source_revision,
        "protocol": {
            "trials": result["trials"], "final_window": result["final_window"],
            "seed_partition": result["seed_partition"],
            "selection_rule": result["tuning"]["selection_rule"],
            "positive_difference_is_better": True,
        },
        "model_specifications": result["model_specifications"],
        "calibration": result["calibration"], "tuning": result["tuning"],
        "per_seed_metrics": evaluation_rows,
        "paired_bootstrap": result["paired_bootstrap"],
        "verdicts": {
            name: row["verdict"] for name, row in result["paired_bootstrap"].items()
        },
    }
    return payload, evaluation_rows, pilot_rows


def interval_reference(result: dict, *, source_revision: str) -> tuple[dict, list, list]:
    rows = []
    for model_id, model in result["eta_sweep"].items():
        for eta, record in model["by_eta"].items():
            for condition, values in (
                ("device", record["device"]),
                ("matched_exponential", record["matched_exponential"]),
            ):
                for seed, value in enumerate(values):
                    rows.append({
                        "model_id": model_id, "eta": float(eta),
                        "condition": condition, "seed": int(seed),
                        "selectivity": float(value),
                    })
    payload = {
        "schema_version": 1, "analysis": "interval_selectivity_eta_sensitivity",
        "source_revision": source_revision,
        "protocol": {
            "eta_values": sorted({row["eta"] for row in rows}),
            "seeds": result["seeds"], "trials": result["trials"],
            "require_device_above_one": True,
            "require_matched_exponential_below_one": True,
        },
        "model_specifications": result["model_specifications"],
        "per_seed_metrics": rows,
        "summary": {
            model_id: {
                "preferred_delay": model["preferred_delay"],
                "direction_holds_all_eta": model["direction_holds_all_eta"],
                "means": {
                    str(eta): {
                        "device": record["device_mean"],
                        "matched_exponential": record["matched_exponential_mean"],
                    }
                    for eta, record in model["by_eta"].items()
                },
            }
            for model_id, model in result["eta_sweep"].items()
        },
        "verdict": (
            "kernel_shape_direction_supported"
            if result["eta_direction_required"] else "kernel_shape_direction_failed"
        ),
    }
    return payload, rows, []


def write_reference_artifacts(result: dict, name: str, output_dir,
                              *, source_revision: str | None = None) -> dict:
    output = Path(output_dir)
    root = Path(__file__).resolve().parents[2]
    revision = source_revision or _source_revision(root)
    builders = {
        "sequential": sequential_reference,
        "dms": dms_reference,
        "interval": interval_reference,
    }
    if name not in builders:
        raise ValueError(f"unknown reference artifact {name!r}")
    builder = builders[name]
    payload, evaluation, pilot = builder(result, source_revision=revision)
    json_path = output / f"{name}_reference.json"
    evaluation_path = output / f"{name}_evaluation.csv"
    pilot_path = output / f"{name}_pilot.csv"
    _write_json(json_path, payload)
    _write_csv(evaluation_path, evaluation)
    if pilot:
        _write_csv(pilot_path, pilot)
    return {"json": str(json_path), "evaluation": str(evaluation_path),
            "pilot": str(pilot_path)}


def write_tex_macros(references: dict[str, dict], path) -> None:
    """Generate compact manuscript macros directly from reference payloads."""
    lines = ["% Generated from tracked benchmark reference JSON; do not edit values."]
    for analysis, payload in sorted(references.items()):
        if analysis not in {"sequential", "dms"}:
            continue
        prefix = "Sequential" if analysis == "sequential" else "DMS"
        rows = payload["per_seed_metrics"]
        conditions = sorted({row["condition"] for row in rows})
        for condition in conditions:
            selected = [row for row in rows if row["condition"] == condition]
            field = "aulc"
            mean = float(np.mean([row[field] for row in selected]))
            token = "".join(part.title() for part in condition.split("_"))
            lines.append(f"\\newcommand{{\\{prefix}{token}AULC}}{{{mean:.3f}}}")
            selected_eta = payload["tuning"]["conditions"][condition]["selected_eta"]
            lines.append(
                f"\\newcommand{{\\{prefix}{token}Eta}}{{{float(selected_eta):g}}}"
            )
        for comparator, record in sorted(payload["paired_bootstrap"].items()):
            token = "".join(part.title() for part in comparator.split("_"))
            lines.extend((
                f"\\newcommand{{\\{prefix}Minus{token}AULC}}{{{float(record['mean']):.3f}}}",
                f"\\newcommand{{\\{prefix}Minus{token}Lo}}{{{float(record['ci95'][0]):.3f}}}",
                f"\\newcommand{{\\{prefix}Minus{token}Hi}}{{{float(record['ci95'][1]):.3f}}}",
            ))
    interval = references.get("interval")
    if interval is not None:
        for model_id, record in sorted(interval["summary"].items()):
            prefix = "IntervalPhysical" if model_id.startswith("physical") else "IntervalLinear"
            ordered = sorted(record["means"].items(), key=lambda item: float(item[0]))
            if len(ordered) != 3:
                raise ValueError("interval TeX macros require the declared three-rate sweep")
            for token, (eta, values) in zip(("Low", "Mid", "High"), ordered):
                lines.append(
                    f"\\newcommand{{\\{prefix}DeviceEta{token}}}"
                    f"{{{float(values['device']):.2f}}}"
                )
                lines.append(
                    f"\\newcommand{{\\{prefix}ExpEta{token}}}"
                    f"{{{float(values['matched_exponential']):.2f}}}"
                )
    codesign = references.get("codesign")
    if codesign is not None:
        benchmark = codesign["benchmark"]
        lines.append(
            f"\\newcommand{{\\CodesignPhysicalPreferredLag}}"
            f"{{{float(benchmark['protocol']['physical_preferred_lag_s']):.2f}}}"
        )
        method_prefixes = {
            "physical_headroom_v1": "CodesignPhysical",
            "linear_erlang_v1": "CodesignLinear",
            "physical_decay_matched_exponential": "CodesignExponential",
            "learned_signed_exponential_k3": "CodesignLearned",
        }
        for method, prefix in method_prefixes.items():
            record = benchmark["summary"][method]
            lines.append(
                f"\\newcommand{{\\{prefix}OverallLoss}}{{{float(record['log_loss']):.3f}}}"
            )
            for regime, values in sorted(record["by_regime"].items()):
                token = "".join(part.title() for part in regime.split("_"))
                lines.append(
                    f"\\newcommand{{\\{prefix}{token}Loss}}"
                    f"{{{float(values['log_loss']):.3f}}}"
                )
                lines.append(
                    f"\\newcommand{{\\{prefix}{token}Accuracy}}"
                    f"{{{float(values['top1_accuracy']):.3f}}}"
                )
        for comparator, record in sorted(benchmark["paired_bootstrap"].items()):
            token = method_prefixes[comparator].removeprefix("Codesign")
            lines.extend((
                f"\\newcommand{{\\CodesignPhysicalMinus{token}Loss}}"
                f"{{{float(record['mean_physical_minus_comparator']):.3f}}}",
                f"\\newcommand{{\\CodesignPhysicalMinus{token}Lo}}"
                f"{{{float(record['ci95'][0]):.3f}}}",
                f"\\newcommand{{\\CodesignPhysicalMinus{token}Hi}}"
                f"{{{float(record['ci95'][1]):.3f}}}",
            ))
        resources = {
            row["implementation"]: row
            for row in codesign["resource_accounting"]["rows"]
        }
        resource_names = {
            "physical_headroom_in_material": "Physical",
            "physical_decay_matched_exponential": "Exponential",
            "digital_linear_erlang_k3": "Linear",
            "learned_signed_exponential_k3": "Learned",
        }
        for implementation, token in resource_names.items():
            row = resources[implementation]
            lines.append(
                f"\\newcommand{{\\Codesign{token}ExternalBits}}"
                f"{{{int(row['external_state_bits_per_synapse'])}}}"
            )
            lines.append(
                f"\\newcommand{{\\Codesign{token}SharedCoefficients}}"
                f"{{{int(row['shared_coefficient_words'])}}}"
            )
        preferred = [
            float(row["preferred_lag_s"])
            for row in codesign["timing_phase_envelope"]
        ]
        lines.append(
            f"\\newcommand{{\\CodesignPhaseMinimum}}{{{min(preferred):.2f}}}"
        )
        lines.append(
            f"\\newcommand{{\\CodesignPhaseMaximum}}{{{max(preferred):.2f}}}"
        )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mean_interval(values, *, seed: int, n_boot: int = 10_000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, values.size, size=(int(n_boot), values.size))
    samples = values[indices].mean(axis=1)
    low, high = np.percentile(samples, (2.5, 97.5))
    return float(values.mean()), float(low), float(high)


def write_reference_figures(reference_dir, output_dir) -> dict[str, str]:
    """Render deterministic headline figures from tracked reference JSON only."""
    import matplotlib.pyplot as plt

    reference_dir = Path(reference_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sequential = json.loads(
        (reference_dir / "sequential_reference.json").read_text(encoding="utf-8")
    )
    dms = json.loads(
        (reference_dir / "dms_reference.json").read_text(encoding="utf-8")
    )
    interval = json.loads(
        (reference_dir / "interval_reference.json").read_text(encoding="utf-8")
    )
    colors = {
        "device": "#3aa07a", "linear_device": "#4c78a8",
        "exponential": "#2f4b8f", "abstract": "#2f4b8f",
        "conventional_rstdp": "#7f7f7f", "shallow_eprop": "#e0a93b",
        "no_trace": "#b8c0c8",
    }
    labels = {
        "device": "physical", "linear_device": "linear",
        "exponential": "matched exp.", "abstract": "matched exp.",
        "conventional_rstdp": "R-STDP", "shallow_eprop": "e-prop-style",
        "no_trace": "no trace",
    }

    def grouped(payload, field):
        result = {}
        for row in payload["per_seed_metrics"]:
            result.setdefault(row["condition"], []).append(float(row[field]))
        return result

    seq_order = (
        "device", "linear_device", "exponential", "shallow_eprop",
        "conventional_rstdp", "no_trace",
    )
    seq_values = grouped(sequential, "aulc")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    x = np.arange(len(seq_order))
    summaries = [_mean_interval(seq_values[name], seed=7000 + index)
                 for index, name in enumerate(seq_order)]
    means = np.asarray([row[0] for row in summaries])
    errors = np.vstack((
        means - np.asarray([row[1] for row in summaries]),
        np.asarray([row[2] for row in summaries]) - means,
    ))
    axes[0].bar(x, means, color=[colors[name] for name in seq_order],
                yerr=errors, capsize=2.5)
    axes[0].axhline(0.5 ** 4, color="0.45", ls="--", lw=0.9)
    axes[0].set(ylabel="held-out AULC", title="(a) 20 evaluation seeds", ylim=(0, 1.04))
    etas = [sequential["tuning"]["conditions"][name]["selected_eta"]
            for name in seq_order]
    axes[1].bar(x, etas, color=[colors[name] for name in seq_order])
    axes[1].set_yscale("symlog", linthresh=1e-3)
    axes[1].set(ylabel="pilot-selected $\\eta$", title="(b) frozen before evaluation")
    for axis in axes:
        axis.set_xticks(x, [labels[name] for name in seq_order], rotation=35,
                        ha="right", fontsize=7.5)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", color="0.88", lw=0.5)
    fig.tight_layout()
    sequential_path = output_dir / "fig_sequential_fair.png"
    fig.savefig(sequential_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    dms_order = ("device", "linear_device", "abstract", "no_trace")
    dms_values = grouped(dms, "aulc")
    fig, axis = plt.subplots(figsize=(5.4, 3.1))
    x = np.arange(len(dms_order))
    summaries = [_mean_interval(dms_values[name], seed=8000 + index)
                 for index, name in enumerate(dms_order)]
    means = np.asarray([row[0] for row in summaries])
    errors = np.vstack((means - np.asarray([row[1] for row in summaries]),
                        np.asarray([row[2] for row in summaries]) - means))
    axis.bar(x, means, color=[colors[name] for name in dms_order],
             yerr=errors, capsize=2.5)
    axis.axhline(0.5, color="0.45", ls="--", lw=0.9)
    axis.set_xticks(x, [labels[name] for name in dms_order], rotation=25,
                    ha="right", fontsize=8)
    axis.set(ylabel="held-out AULC", title="Shallow DMS: 20 evaluation seeds",
             ylim=(0.45, 1.02))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="0.88", lw=0.5)
    fig.tight_layout()
    dms_path = output_dir / "fig_shallow_dms_fair.png"
    fig.savefig(dms_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for model_index, (model_id, title) in enumerate((
        ("physical_headroom_v1", "physical headroom"),
        ("linear_erlang_v1", "linear Erlang sensitivity"),
    )):
        axis = axes[model_index]
        rows = [row for row in interval["per_seed_metrics"]
                if row["model_id"] == model_id]
        for condition, color, marker in (
            ("device", "#3aa07a", "o"),
            ("matched_exponential", "#2f4b8f", "s"),
        ):
            eta_values = sorted({float(row["eta"]) for row in rows})
            summaries = [
                _mean_interval(
                    [row["selectivity"] for row in rows
                     if row["condition"] == condition and float(row["eta"]) == eta],
                    seed=9000 + model_index * 10 + eta_index,
                )
                for eta_index, eta in enumerate(eta_values)
            ]
            means = np.asarray([summary[0] for summary in summaries])
            axis.errorbar(
                eta_values, means,
                yerr=np.vstack((means - np.asarray([s[1] for s in summaries]),
                                np.asarray([s[2] for s in summaries]) - means)),
                color=color, marker=marker, lw=1.5, capsize=2.5,
                label="device" if condition == "device" else "matched exponential",
            )
        axis.axhline(1.0, color="0.5", ls="--", lw=0.9)
        axis.set_xscale("log")
        axis.set(xlabel="learning rate $\\eta$", title=f"({chr(97 + model_index)}) {title}")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color="0.88", lw=0.5)
    axes[0].set_ylabel("selectivity $S=w_{pref}/w_{late}$")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    interval_path = output_dir / "fig_interval_fair.png"
    fig.savefig(interval_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    result = {
        "sequential": str(sequential_path),
        "dms": str(dms_path),
        "interval": str(interval_path),
    }
    codesign_path = reference_dir / "codesign_reference.json"
    if codesign_path.exists():
        from .codesign import render_codesign_figure
        codesign = json.loads(codesign_path.read_text(encoding="utf-8"))
        result["codesign"] = render_codesign_figure(
            codesign, output_dir / "fig_codesign.png"
        )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/results/reference")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--dms", action="store_true")
    parser.add_argument("--interval", action="store_true")
    parser.add_argument("--codesign", action="store_true")
    parser.add_argument(
        "--figures-dir",
        help="render tracked reference summaries into this directory",
    )
    args = parser.parse_args(argv)
    if not (args.sequential or args.dms or args.interval or args.codesign
            or args.figures_dir):
        parser.error("select a benchmark and/or --figures-dir")
    output = Path(args.output_dir)
    payloads = {}
    if args.sequential:
        from .maze import run_action_sequence
        kwargs = ({
            "episodes": 20, "tuning_episodes": 10,
            "calibration_trajectories": 4, "learning_rates": (0.1, 1.0),
            "tuning_seeds": (1000, 1001), "evaluation_seeds": (2000, 2001),
            "max_boundary_expansions": 0, "bootstrap_resamples": 100,
            "D": 0.02, "dt": 0.02, "step_dur": 0.02,
        } if args.smoke else {})
        result = run_action_sequence(workers=args.workers, **kwargs)
        paths = write_reference_artifacts(result, "sequential", output)
        payloads["sequential"] = json.loads(Path(paths["json"]).read_text())
    if args.dms:
        from .deep import run_dms_all
        kwargs = ({
            "trials": 20, "tuning_trials": 10, "calibration_trials": 4,
            "learning_rates": (0.1, 1.0), "tuning_seeds": (1000, 1001),
            "evaluation_seeds": (2000, 2001), "max_boundary_expansions": 0,
            "bootstrap_resamples": 100, "G": 0.16, "t_distract": 0.08,
            "dt": 0.02, "cue_dur": 0.04, "distract_dur": 0.04,
        } if args.smoke else {})
        result = run_dms_all(workers=args.workers, **kwargs)
        paths = write_reference_artifacts(result, "dms", output)
        payloads["dms"] = json.loads(Path(paths["json"]).read_text())
    if args.interval:
        from .selectivity import run_interval_selectivity
        kwargs = {"seeds": 2, "trials": 20} if args.smoke else {}
        result = run_interval_selectivity(**kwargs)
        paths = write_reference_artifacts(result, "interval", output)
        payloads["interval"] = json.loads(Path(paths["json"]).read_text())
    if args.codesign:
        from .codesign import build_codesign_reference, write_codesign_artifacts
        kwargs = ({
            "trials_per_block": 4, "learned_multistarts": 1,
            "bootstrap_resamples": 100,
            "phase_kwargs": {
                "voltages": (0.9,), "depths": (3,),
                "retention_definitions": (("smoke", 1.5, "smoke"),),
            },
        } if args.smoke else {})
        codesign = build_codesign_reference(**kwargs)
        write_codesign_artifacts(
            codesign,
            json_path=output / "codesign_reference.json",
            block_csv_path=output / "codesign_evaluation.csv",
            phase_csv_path=output / "codesign_phase.csv",
        )
        payloads["codesign"] = codesign
    for existing in ("sequential", "dms", "interval", "codesign"):
        path = output / f"{existing}_reference.json"
        if existing not in payloads and path.exists():
            payloads[existing] = json.loads(path.read_text(encoding="utf-8"))
    write_tex_macros(payloads, output / "benchmark_macros.tex")
    if args.figures_dir:
        write_reference_figures(output, args.figures_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "sequential_reference", "dms_reference", "interval_reference",
    "write_reference_artifacts",
    "write_tex_macros", "write_reference_figures", "main",
]
