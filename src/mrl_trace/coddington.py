"""Causal dopamine-learning reanalysis of Coddington, Lindo & Dudman (2023).

The public Figshare release contains analysed, trial-aligned behaviour and
photometry for the closed-loop VTA--DA intervention.  This module reproduces a
source-style mouse-by-training-block analysis and keeps a mouse-resampling
sensitivity alongside it.  It tests action-contingent dopamine modulation of
learning; it does not compare device-kernel shapes.
"""
from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.stats import f as f_distribution

__all__ = [
    "CODDINGTON_ARTICLE_DOI",
    "CODDINGTON_DATA_DOI",
    "CODDINGTON_EXPECTED_MD5",
    "CODDINGTON_GROUPS",
    "CoddingtonSubject",
    "load_coddington_dataset",
    "evaluate_coddington_causal_learning",
    "write_coddington_tex_macros",
    "write_coddington_artifacts",
]

CODDINGTON_ARTICLE_DOI = "10.1038/s41586-022-05614-z"
CODDINGTON_DATA_DOI = "10.25378/janelia.21816054.v1"
CODDINGTON_EXPECTED_MD5 = "23b0b229d92ab9bf26ab9989946eafb4"
CODDINGTON_EXPECTED_SIZE = 271_014_539
CODDINGTON_PRIMARY_TRIALS = 800
CODDINGTON_BLOCK_SIZE = 100
CODDINGTON_GROUPS = {
    "control": (1, 2, 4, 6, 9, 11, 15, 16, 19),
    "stimLick-": (3, 5, 10, 14, 18, 20),
    "stimLick+": (7, 8, 12, 13, 17),
    "stim+Lick+": (21, 22, 23, 24),
}


def _digest(path: Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_digest(payload: dict) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _group_for_mouse(mouse_id: int) -> str:
    for group, ids in CODDINGTON_GROUPS.items():
        if mouse_id in ids:
            return group
    raise ValueError(f"mouse {mouse_id} is absent from the frozen group map")


@dataclass(frozen=True)
class CoddingtonSubject:
    """Analysis fields for one mouse; raw video and waveforms remain external."""

    mouse_id: int
    group: str
    session_id: np.ndarray
    lick_state: np.ndarray
    stimulation: np.ndarray
    reward_collection_latency_ms: np.ndarray
    reward_dopamine_z: np.ndarray
    preparatory_lick_hz: np.ndarray
    baseline_lick_hz: np.ndarray

    @property
    def n_trials(self) -> int:
        return int(self.preparatory_lick_hz.size)

    def baseline_corrected_preparatory_lick(self) -> np.ndarray:
        return self.preparatory_lick_hz - self.baseline_lick_hz


def _vector(record, name: str, *, limit: int | None = None) -> np.ndarray:
    values = np.asarray(record[name]).reshape(-1).astype(float, copy=False)
    return values if limit is None else values[:limit]


def load_coddington_dataset(
    path,
    *,
    verify_source: bool = True,
) -> tuple[CoddingtonSubject, ...]:
    """Load the public ``seshMerge.mat`` without copying raw data into the repo."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if verify_source:
        if source.stat().st_size != CODDINGTON_EXPECTED_SIZE:
            raise ValueError("Coddington source size does not match Figshare v1")
        if _digest(source, "md5") != CODDINGTON_EXPECTED_MD5:
            raise ValueError("Coddington source MD5 does not match Figshare v1")

    archive = loadmat(
        source,
        variable_names=["seshMerge"],
        struct_as_record=True,
        squeeze_me=False,
    )
    if "seshMerge" not in archive:
        raise ValueError("Coddington archive lacks seshMerge")
    records = np.asarray(archive["seshMerge"]).reshape(-1)
    if records.size != 24:
        raise ValueError(f"expected 24 Coddington mice, found {records.size}")
    required = {
        "seshID", "lickState", "stimState", "latency", "rewDA",
        "prepLick", "baseLick",
    }
    if records.dtype.names is None or not required.issubset(records.dtype.names):
        missing = required.difference(records.dtype.names or ())
        raise ValueError(f"Coddington archive lacks fields: {sorted(missing)}")

    subjects = []
    for mouse_id, record in enumerate(records, start=1):
        analysed_trials = _vector(record, "prepLick").size
        if mouse_id <= 20 and analysed_trials != CODDINGTON_PRIMARY_TRIALS:
            raise ValueError(
                f"mouse {mouse_id} has {analysed_trials} analysed trials, expected 800"
            )
        arrays = {
            "session_id": _vector(record, "seshID", limit=analysed_trials),
            "lick_state": _vector(record, "lickState", limit=analysed_trials),
            "stimulation": _vector(record, "stimState", limit=analysed_trials),
            "reward_collection_latency_ms": _vector(
                record, "latency", limit=analysed_trials
            ),
            "reward_dopamine_z": _vector(record, "rewDA", limit=analysed_trials),
            "preparatory_lick_hz": _vector(record, "prepLick"),
            "baseline_lick_hz": _vector(record, "baseLick"),
        }
        if any(values.size != analysed_trials for values in arrays.values()):
            raise ValueError(f"mouse {mouse_id} contains misaligned analysed fields")
        subjects.append(CoddingtonSubject(
            mouse_id=mouse_id,
            group=_group_for_mouse(mouse_id),
            **arrays,
        ))
    return tuple(subjects)


def _block_means(subject: CoddingtonSubject) -> np.ndarray:
    values = subject.baseline_corrected_preparatory_lick()
    if values.size < CODDINGTON_PRIMARY_TRIALS:
        raise ValueError("primary Coddington analysis requires 800 trials per mouse")
    return np.asarray([
        np.nanmean(values[start:start + CODDINGTON_BLOCK_SIZE])
        for start in range(0, CODDINGTON_PRIMARY_TRIALS, CODDINGTON_BLOCK_SIZE)
    ])


def _source_style_two_way(
    stim_lick_minus: np.ndarray,
    stim_lick_plus: np.ndarray,
) -> dict:
    """Reproduce the source-style contingency x 100-trial-block statistic."""
    if stim_lick_minus.shape[1:] != (8,) or stim_lick_plus.shape[1:] != (8,):
        raise ValueError("Coddington primary comparison requires eight blocks")
    rows = []
    for group_value, matrix in ((1, stim_lick_minus), (0, stim_lick_plus)):
        for subject_values in matrix:
            rows.extend(
                (group_value, block, value)
                for block, value in enumerate(subject_values)
            )
    group = np.asarray([row[0] for row in rows], dtype=float)
    block = np.asarray([row[1] for row in rows], dtype=int)
    outcome = np.asarray([row[2] for row in rows], dtype=float)
    if not np.all(np.isfinite(outcome)):
        raise ValueError("block means must be finite")

    cell_design = np.column_stack([
        ((group == group_value) & (block == block_id)).astype(float)
        for group_value in (0, 1)
        for block_id in range(8)
    ])
    cell_fit = cell_design @ np.linalg.lstsq(
        cell_design, outcome, rcond=None
    )[0]
    residual = outcome - cell_fit
    residual_df = int(outcome.size - np.linalg.matrix_rank(cell_design))
    residual_mse = float(np.dot(residual, residual) / residual_df)

    additive_design = np.column_stack([
        np.ones(outcome.size),
        group,
        *[(block == block_id).astype(float) for block_id in range(1, 8)],
    ])
    coefficients = np.linalg.lstsq(additive_design, outcome, rcond=None)[0]
    covariance = residual_mse * np.linalg.inv(
        additive_design.T @ additive_design
    )
    contrast = float(coefficients[1])
    contrast_se = float(np.sqrt(covariance[1, 1]))
    f_value = float(np.square(contrast / contrast_se))
    p_value = float(f_distribution.sf(f_value, 1, residual_df))
    return {
        "contrast_stimLick_minus_minus_plus_hz": contrast,
        "standard_error_hz": contrast_se,
        "f_value": f_value,
        "numerator_df": 1,
        "denominator_df": residual_df,
        "p_value": p_value,
        "analysis_unit": "mouse_by_100_trial_block",
        "claim_limit": (
            "Source-style two-way analysis; the separate mouse-resampling "
            "sensitivity treats the randomized animal as the sole resampling unit."
        ),
    }


def _mouse_bootstrap(
    stim_lick_minus: np.ndarray,
    stim_lick_plus: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    minus_mouse = np.mean(stim_lick_minus, axis=1)
    plus_mouse = np.mean(stim_lick_plus, axis=1)
    estimates = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        minus = minus_mouse[
            rng.integers(0, minus_mouse.size, size=minus_mouse.size)
        ]
        plus = plus_mouse[
            rng.integers(0, plus_mouse.size, size=plus_mouse.size)
        ]
        estimates[index] = np.mean(minus) - np.mean(plus)
    return {
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "estimate_hz": float(np.mean(minus_mouse) - np.mean(plus_mouse)),
        "ci95_hz": np.quantile(estimates, (0.025, 0.975)).tolist(),
        "bootstrap_probability_positive": float(np.mean(estimates > 0)),
    }


def _dopamine_boost(subjects: list[CoddingtonSubject], *, seed: int) -> dict:
    values = []
    for subject in subjects:
        valid = (
            np.isfinite(subject.reward_dopamine_z)
            & np.isfinite(subject.stimulation)
        )
        stimulated = valid & (subject.stimulation == 1)
        unstimulated = valid & (subject.stimulation == 0)
        if not np.any(stimulated) or not np.any(unstimulated):
            raise ValueError(f"mouse {subject.mouse_id} lacks a stimulation contrast")
        values.append(float(
            np.mean(subject.reward_dopamine_z[stimulated])
            - np.mean(subject.reward_dopamine_z[unstimulated])
        ))
    values_array = np.asarray(values)
    rng = np.random.default_rng(seed)
    bootstrap = np.asarray([
        np.mean(values_array[
            rng.integers(0, values_array.size, size=values_array.size)
        ])
        for _ in range(10_000)
    ])
    return {
        "per_mouse_z": values_array.tolist(),
        "mean_z": float(np.mean(values_array)),
        "ci95_z": np.quantile(bootstrap, (0.025, 0.975)).tolist(),
    }


def evaluate_coddington_causal_learning(
    source,
    *,
    n_resamples: int = 10_000,
    seed: int = 20_260_801,
    verify_source: bool = True,
) -> dict:
    """Evaluate the frozen calibrated-stimulation contingency contrast."""
    path = Path(source)
    subjects = load_coddington_dataset(path, verify_source=verify_source)
    primary = [subject for subject in subjects if subject.mouse_id <= 20]
    by_group = {
        group: [subject for subject in primary if subject.group == group]
        for group in ("control", "stimLick-", "stimLick+")
    }
    block_values = {
        group: np.stack([_block_means(subject) for subject in group_subjects])
        for group, group_subjects in by_group.items()
    }
    minus, plus = block_values["stimLick-"], block_values["stimLick+"]
    block_contrast = np.mean(minus, axis=0) - np.mean(plus, axis=0)
    source_style = _source_style_two_way(minus, plus)
    mouse_sensitivity = _mouse_bootstrap(
        minus, plus, n_resamples=n_resamples, seed=seed
    )

    block_rows = []
    for group, group_subjects in by_group.items():
        for subject, values in zip(group_subjects, block_values[group]):
            for block, value in enumerate(values, start=1):
                block_rows.append({
                    "mouse_id": subject.mouse_id,
                    "group": group,
                    "block": block,
                    "trial_start": (block - 1) * CODDINGTON_BLOCK_SIZE + 1,
                    "trial_stop": block * CODDINGTON_BLOCK_SIZE,
                    "baseline_corrected_preparatory_lick_hz": float(value),
                })

    result = {
        "schema_version": 1,
        "analysis": "coddington_action_contingent_dopamine_learning",
        "source": {
            "article_doi": CODDINGTON_ARTICLE_DOI,
            "dataset_doi": CODDINGTON_DATA_DOI,
            "figshare_file_id": 38_710_665,
            "size_bytes": int(path.stat().st_size),
            "md5": _digest(path, "md5"),
            "sha256": _digest(path, "sha256"),
        },
        "protocol": {
            "primary_mice": 20,
            "primary_trials_per_mouse": CODDINGTON_PRIMARY_TRIALS,
            "block_size_trials": CODDINGTON_BLOCK_SIZE,
            "groups": {key: list(value) for key, value in CODDINGTON_GROUPS.items()},
            "outcome": "preparatory_lick_hz_minus_baseline_lick_hz",
            "primary_test": "source_style_two_way_contingency_by_block",
            "sensitivity": "paired_group_mouse_bootstrap",
            "interpretation": (
                "Causal grounding of action-contingent dopaminergic learning; "
                "not evidence for a particular eligibility-kernel shape."
            ),
        },
        "group_block_mean_hz": {
            group: np.mean(values, axis=0).tolist()
            for group, values in block_values.items()
        },
        "stimLick_minus_minus_plus_by_block_hz": block_contrast.tolist(),
        "all_block_contrasts_positive": bool(np.all(block_contrast > 0)),
        "source_style_inference": source_style,
        "mouse_resampling_sensitivity": mouse_sensitivity,
        "dopamine_manipulation_check": {
            "stimLick-": _dopamine_boost(by_group["stimLick-"], seed=seed + 1),
            "stimLick+": _dopamine_boost(by_group["stimLick+"], seed=seed + 2),
        },
        "verdict": "positive_action_contingent_causal_learning",
        "claim_limits": [
            "The intervention randomizes mice to action-contingent dopamine protocols; it does not randomize device kernels.",
            "The source-style block analysis is positive; the conservative animal-only bootstrap interval is reported as sensitivity.",
            "The result grounds the third-factor learning architecture, not memristor hardware efficacy or device-kernel superiority.",
        ],
        "block_rows": block_rows,
    }
    digest_payload = dict(result)
    digest_payload.pop("block_rows")
    result["manifest_digest_sha256"] = _canonical_digest(digest_payload)
    return result


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    colours = {
        "control": "#202020",
        "stimLick-": "#18864b",
        "stimLick+": "#7b3294",
    }
    rows = result["block_rows"]
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 3.5), constrained_layout=True)

    for group in ("control", "stimLick-", "stimLick+"):
        mouse_ids = sorted({row["mouse_id"] for row in rows if row["group"] == group})
        matrix = np.asarray([
            [
                row["baseline_corrected_preparatory_lick_hz"]
                for row in rows
                if row["group"] == group and row["mouse_id"] == mouse_id
            ]
            for mouse_id in mouse_ids
        ])
        mean = np.mean(matrix, axis=0)
        sem = np.std(matrix, axis=0, ddof=1) / np.sqrt(matrix.shape[0])
        block = np.arange(1, 9)
        axes[0].plot(block, mean, marker="o", lw=2, color=colours[group], label=group)
        axes[0].fill_between(block, mean - sem, mean + sem, color=colours[group], alpha=0.18)
    axes[0].set(xlabel="100-trial block", ylabel="Cued minus baseline licking (Hz)")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("Action-contingent learning")

    contrast = np.asarray(result["stimLick_minus_minus_plus_by_block_hz"])
    axes[1].axhline(0, color="0.65", lw=1)
    axes[1].plot(np.arange(1, 9), contrast, marker="o", color="#006d8f", lw=2)
    inference = result["source_style_inference"]
    axes[1].set(xlabel="100-trial block", ylabel="stimLick− minus stimLick+ (Hz)")
    axes[1].set_title(
        rf"Contingency contrast  $F_{{1,{inference['denominator_df']}}}={inference['f_value']:.2f}$"
        + "\n" + rf"$p={inference['p_value']:.2g}$"
    )

    checks = result["dopamine_manipulation_check"]
    names = ["stimLick−", "stimLick+"]
    keys = ["stimLick-", "stimLick+"]
    means = [checks[key]["mean_z"] for key in keys]
    intervals = np.asarray([checks[key]["ci95_z"] for key in keys])
    error = np.vstack([np.asarray(means) - intervals[:, 0], intervals[:, 1] - np.asarray(means)])
    axes[2].bar(names, means, color=[colours[key] for key in keys], alpha=0.85)
    axes[2].errorbar(names, means, yerr=error, fmt="none", color="black", capsize=3)
    axes[2].axhline(0, color="0.65", lw=1)
    axes[2].set(ylabel="Stimulated − unstimulated reward DA (z)")
    axes[2].set_title("Manipulation check")

    for label, axis in zip("abc", axes):
        axis.text(-0.16, 1.06, label, transform=axis.transAxes, fontweight="bold", fontsize=12)
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=300, bbox_inches="tight", metadata={"Software": "mrl_trace.coddington"})
    plt.close(figure)


def write_coddington_tex_macros(result: dict, path) -> Path:
    """Generate stable TeX values from the causal-learning reference payload."""
    inference = result["source_style_inference"]
    sensitivity = result["mouse_resampling_sensitivity"]
    minus = result["dopamine_manipulation_check"]["stimLick-"]
    plus = result["dopamine_manipulation_check"]["stimLick+"]
    lines = [
        "% Generated from coddington_causal_reference.json; do not edit.",
        f"\\newcommand{{\\CoddingtonContrastHz}}{{{inference['contrast_stimLick_minus_minus_plus_hz']:.3f}}}",
        f"\\newcommand{{\\CoddingtonFValue}}{{{inference['f_value']:.2f}}}",
        f"\\newcommand{{\\CoddingtonDenominatorDf}}{{{inference['denominator_df']}}}",
        f"\\newcommand{{\\CoddingtonPValue}}{{{inference['p_value']:.2g}}}",
        f"\\newcommand{{\\CoddingtonBootstrapProbability}}{{{100.0 * sensitivity['bootstrap_probability_positive']:.1f}\\%}}",
        f"\\newcommand{{\\CoddingtonBootstrapLo}}{{{sensitivity['ci95_hz'][0]:.3f}}}",
        f"\\newcommand{{\\CoddingtonBootstrapHi}}{{{sensitivity['ci95_hz'][1]:.3f}}}",
        f"\\newcommand{{\\CoddingtonMinusDABoost}}{{{minus['mean_z']:.2f}}}",
        f"\\newcommand{{\\CoddingtonPlusDABoost}}{{{plus['mean_z']:.2f}}}",
        f"\\newcommand{{\\CoddingtonManifestDigest}}{{\\texttt{{{result['manifest_digest_sha256'][:12]}}}}}",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_coddington_artifacts(result: dict, output_dir) -> dict:
    """Write tracked derived artifacts; never writes the external MATLAB source."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "coddington_causal_reference.json"
    csv_path = output / "coddington_causal_blocks.csv"
    figure_path = output / "fig_coddington_causal.png"
    macros_path = output / "coddington_macros.tex"
    payload = dict(result)
    rows = payload.pop("block_rows")
    _write_json(json_path, payload)
    _write_csv(csv_path, rows)
    _write_figure(result, figure_path)
    write_coddington_tex_macros(result, macros_path)
    return {
        "reference": str(json_path),
        "blocks": str(csv_path),
        "figure": str(figure_path),
        "macros": str(macros_path),
    }
