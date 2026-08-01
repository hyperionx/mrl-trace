"""Independent directional replication of Jeong et al. (2022), DANDI 000351.

The analysis follows the public Figure 6 source code for its cohort and
anticipatory-lick definition.  It range-reads only the event log from each NWB;
raw photometry and video remain external.  The result is a secondary causal
directionality check, not an eligibility-kernel comparison.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

__all__ = [
    "JEONG_DANDISET_ID",
    "JEONG_ARTICLE_DOI",
    "JEONG_SOURCE_CODE_URL",
    "JEONG_CONTROL_SUBJECTS",
    "JEONG_INHIBITED_SUBJECTS",
    "fetch_jeong000351_manifest",
    "load_jeong000351_manifest",
    "evaluate_jeong_directional_learning",
    "write_jeong_artifacts",
    "write_jeong_tex_macros",
]

JEONG_DANDISET_ID = "000351"
JEONG_ARTICLE_DOI = "10.1126/science.abq6740"
JEONG_SOURCE_CODE_URL = "https://github.com/namboodirilab/ANCCR"
JEONG_API_ROOT = "https://api.dandiarchive.org/api"
JEONG_EXPECTED_ASSET_COUNT = 428

# Frozen from analysis/fig6/learnign_curve_sequential.py in the source repository.
JEONG_CONTROL_SUBJECTS = (
    "HJ-FP-datWT-stGtACR-F1",
    "HJ-FP-WT-stGtACR-F1",
    "HJ-FP-WT-stGtACR-F2",
    "HJ-FP-WT-stGtACR-F3",
    "HJ-FP-WT-stGtACR-M1",
    "HJ-FP-WT-stGtACR-M2",
    "HJ-FP-WT-stGtACR-M3",
)
JEONG_INHIBITED_SUBJECTS = tuple(
    f"HJ-FP-datHT-stGtACR-M{mouse}" for mouse in range(2, 8)
)
_SELECTED_SUBJECTS = JEONG_CONTROL_SUBJECTS + JEONG_INHIBITED_SUBJECTS
_DAY_PATTERN = re.compile(r"_ses-Day(\d+)")


def _canonical_digest(payload) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _asset_detail(asset: dict) -> dict:
    import requests

    response = requests.get(
        f"{JEONG_API_ROOT}/assets/{asset['asset_id']}/", timeout=60
    )
    response.raise_for_status()
    detail = response.json()
    digests = detail.get("digest", {})
    content_urls = detail.get("contentUrl", [])
    s3_url = next(
        (url for url in content_urls if "dandiarchive.s3.amazonaws.com" in url),
        content_urls[-1] if content_urls else "",
    )
    return {
        "path": asset["path"],
        "asset_id": asset["asset_id"],
        "blob_id": asset.get("blob") or "",
        "size_bytes": int(asset["size"]),
        "sha256": digests.get("dandi:sha2-256", ""),
        "dandi_etag": digests.get("dandi:dandi-etag", ""),
        "content_url": s3_url,
        "created": asset.get("created", ""),
        "modified": asset.get("modified", ""),
    }


def fetch_jeong000351_manifest(output_path, *, workers: int = 12) -> dict:
    """Pin every asset in the mutable DANDI draft, including SHA-256 digests."""
    import requests

    listing_url = (
        f"{JEONG_API_ROOT}/dandisets/{JEONG_DANDISET_ID}/versions/draft/"
        "assets/?page_size=500"
    )
    response = requests.get(listing_url, timeout=60)
    response.raise_for_status()
    listing = response.json()
    if int(listing["count"]) != JEONG_EXPECTED_ASSET_COUNT:
        raise ValueError(
            f"DANDI {JEONG_DANDISET_ID} draft has {listing['count']} assets; "
            f"the frozen analysis expects {JEONG_EXPECTED_ASSET_COUNT}"
        )
    rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [pool.submit(_asset_detail, asset) for asset in listing["results"]]
        for future in as_completed(futures):
            rows.append(future.result())
    rows.sort(key=lambda row: row["path"])
    if any(not row["sha256"] or not row["content_url"] for row in rows):
        raise ValueError("DANDI manifest contains an asset without digest or content URL")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "dandiset_id": JEONG_DANDISET_ID,
        "version": "draft",
        "asset_count": len(rows),
        "total_size_bytes": int(sum(row["size_bytes"] for row in rows)),
        "manifest_sha256": _canonical_digest(rows),
        "path": str(target),
    }


def load_jeong000351_manifest(path) -> tuple[dict, ...]:
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as stream:
        rows = tuple(csv.DictReader(stream))
    if len(rows) != JEONG_EXPECTED_ASSET_COUNT:
        raise ValueError(
            f"Jeong manifest has {len(rows)} assets, expected {JEONG_EXPECTED_ASSET_COUNT}"
        )
    required = {
        "path", "asset_id", "blob_id", "size_bytes", "sha256", "content_url"
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError("Jeong manifest schema is incomplete")
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)) or paths != sorted(paths):
        raise ValueError("Jeong manifest paths must be unique and sorted")
    if any(len(row["sha256"]) != 64 for row in rows):
        raise ValueError("Jeong manifest contains an invalid SHA-256 digest")
    return rows


def _subject_and_day(path: str) -> tuple[str, int] | None:
    subject = path.split("/", 1)[0]
    if subject.startswith("sub-"):
        subject = subject[4:]
    match = _DAY_PATTERN.search(path)
    if subject not in _SELECTED_SUBJECTS or match is None:
        return None
    day = int(match.group(1))
    # The public Figure 6 code treats the first chronological asset as the
    # random-reward session.  Most paths say RandomRewards; M6/M7 call it only
    # Day1.nwb.  Day 1 is therefore excluded independent of the path suffix.
    if day == 1 or "RandomRewards" in path:
        return None
    return subject, day


def _read_eventlog(url: str, *, retries: int = 3) -> tuple[np.ndarray, np.ndarray]:
    import fsspec
    import h5py

    error = None
    for attempt in range(retries):
        try:
            with fsspec.open(
                url, "rb", block_size=2 * 1024 * 1024, cache_type="blockcache"
            ) as remote:
                with h5py.File(remote, "r") as nwb:
                    event_index = np.asarray(
                        nwb["acquisition/eventlog/eventindex"][:], dtype=int
                    )
                    event_time = np.asarray(
                        nwb["acquisition/eventlog/eventtime"][:], dtype=float
                    )
            if event_index.shape != event_time.shape:
                raise ValueError("misaligned Jeong event log")
            # Some source NWBs encode the final session-end marker (event 0)
            # with timestamp 0 after otherwise monotone events.  The authors'
            # Figure 6 loader retains the record, while the analysis uses only
            # cue and lick events.  Validate the relevant stream accordingly.
            analysis_time = event_time[event_index != 0]
            if not np.all(np.isfinite(analysis_time)) or not np.all(
                np.diff(analysis_time) >= 0
            ):
                raise ValueError("non-monotone Jeong cue/lick event stream")
            return event_index, event_time
        except Exception as exc:  # pragma: no cover - network retry path
            error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"failed to range-read {url}") from error


def _session_anticipatory_licks(row: dict) -> dict:
    parsed = _subject_and_day(row["path"])
    if parsed is None:
        raise ValueError("asset is outside the frozen Figure 6 cohort")
    subject, day = parsed
    event_index, event_time = _read_eventlog(row["content_url"])
    licks = event_time[event_index == 5]
    paired_cues = event_time[event_index == 15]
    if paired_cues.size < 2 or paired_cues.size % 2:
        raise ValueError(f"{row['path']} does not contain paired CS1/CS2 events")
    cs1 = paired_cues[::2]
    trial_values = np.asarray([
        np.count_nonzero((licks >= cue) & (licks < cue + 3.0))
        - np.count_nonzero((licks >= cue - 3.0) & (licks < cue))
        for cue in cs1
    ], dtype=float)
    return {
        "subject": subject,
        "group": "inhibited" if subject in JEONG_INHIBITED_SUBJECTS else "control",
        "day": day,
        "path": row["path"],
        "asset_id": row["asset_id"],
        "n_trials": int(trial_values.size),
        "mean_anticipatory_licks": float(np.mean(trial_values)),
    }


def _independent_bootstrap(
    control: np.ndarray,
    inhibited: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_resamples, dtype=float)
    for index in range(n_resamples):
        control_draw = control[rng.integers(0, control.size, control.size)]
        inhibited_draw = inhibited[
            rng.integers(0, inhibited.size, inhibited.size)
        ]
        estimates[index] = np.mean(control_draw) - np.mean(inhibited_draw)
    return {
        "estimate_control_minus_inhibited": float(np.mean(control) - np.mean(inhibited)),
        "ci95": np.quantile(estimates, (0.025, 0.975)).tolist(),
        "probability_positive": float(np.mean(estimates > 0)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
    }


def evaluate_jeong_directional_learning(
    manifest_path,
    *,
    workers: int = 6,
    n_resamples: int = 10_000,
    seed: int = 20_260_805,
) -> dict:
    """Range-read the frozen Figure 6 cohort and test the reported direction."""
    manifest = load_jeong000351_manifest(manifest_path)
    selected = [row for row in manifest if _subject_and_day(row["path"]) is not None]
    session_rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(_session_anticipatory_licks, row): row for row in selected}
        for future in as_completed(futures):
            session_rows.append(future.result())
    session_rows.sort(key=lambda row: (row["subject"], row["day"]))

    by_subject = {}
    for row in session_rows:
        by_subject.setdefault(row["subject"], []).append(row)
    if set(by_subject) != set(_SELECTED_SUBJECTS):
        missing = sorted(set(_SELECTED_SUBJECTS).difference(by_subject))
        raise ValueError(f"Jeong Figure 6 cohort is incomplete: {missing}")
    if any(len(rows) < 5 for rows in by_subject.values()):
        raise ValueError("each Jeong Figure 6 mouse requires at least five sessions")

    subject_rows = []
    for subject in _SELECTED_SUBJECTS:
        rows = by_subject[subject]
        values = np.asarray([row["mean_anticipatory_licks"] for row in rows])
        source_display = np.concatenate((values[:3], values[-2:]))
        subject_rows.append({
            "subject": subject,
            "group": rows[0]["group"],
            "n_sessions": len(rows),
            "first_session": float(values[0]),
            "final_session": float(values[-1]),
            "acquisition_change": float(values[-1] - values[0]),
            "source_display_sessions": source_display.tolist(),
        })

    control = [row for row in subject_rows if row["group"] == "control"]
    inhibited = [row for row in subject_rows if row["group"] == "inhibited"]
    control_change = np.asarray([row["acquisition_change"] for row in control])
    inhibited_change = np.asarray([row["acquisition_change"] for row in inhibited])
    control_final = np.asarray([row["final_session"] for row in control])
    inhibited_final = np.asarray([row["final_session"] for row in inhibited])
    change = _independent_bootstrap(
        control_change, inhibited_change, n_resamples=n_resamples, seed=seed
    )
    final = _independent_bootstrap(
        control_final, inhibited_final, n_resamples=n_resamples, seed=seed + 1
    )
    t_change = ttest_ind(control_change, inhibited_change, equal_var=False)
    t_final = ttest_ind(control_final, inhibited_final, equal_var=False)
    directional = bool(
        change["estimate_control_minus_inhibited"] > 0
        and final["estimate_control_minus_inhibited"] > 0
    )
    result = {
        "schema_version": 1,
        "analysis": "jeong000351_figure6_directional_replication",
        "source": {
            "article_doi": JEONG_ARTICLE_DOI,
            "dandiset_id": JEONG_DANDISET_ID,
            "dandiset_version": "draft",
            "source_code": JEONG_SOURCE_CODE_URL,
            "manifest_asset_count": len(manifest),
            "manifest_sha256": _canonical_digest(list(manifest)),
        },
        "protocol": {
            "cohort_source": "public analysis/fig6/learnign_curve_sequential.py",
            "control_subjects": list(JEONG_CONTROL_SUBJECTS),
            "inhibited_subjects": list(JEONG_INHIBITED_SUBJECTS),
            "excluded_source_subject": "HJ-FP-datHT-stGtACR-M8",
            "event_codes": {"lick": 5, "paired_cues": 15},
            "outcome": "licks_CS1_to_reward_minus_pre_CS1_3s",
            "display_sessions": "first three and last two sessions per source code",
            "analysis_unit": "mouse",
        },
        "session_count": len(session_rows),
        "subject_count": len(subject_rows),
        "subject_rows": subject_rows,
        "session_rows": session_rows,
        "inference": {
            "acquisition_change": change,
            "final_session": final,
            "welch_change_t": float(t_change.statistic),
            "welch_change_p": float(t_change.pvalue),
            "welch_final_t": float(t_final.statistic),
            "welch_final_p": float(t_final.pvalue),
        },
        "directionally_consistent": directional,
        "verdict": (
            "directionally_consistent_secondary_causal_replication"
            if directional else "direction_not_replicated"
        ),
        "claim_limits": [
            "The Dandiset is a pinned snapshot of a mutable draft, not a published DANDI version.",
            "Genotype and opsin status define groups; this is not a randomized kernel comparison.",
            "The fixed CS2-to-reward inhibition window tests dopamine-dependent learning, not eligibility-kernel shape.",
            "The independent reanalysis is described as directional unless the mouse-bootstrap interval is wholly positive.",
        ],
    }
    digest_payload = dict(result)
    digest_payload.pop("session_rows")
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


def write_jeong_tex_macros(result: dict, path) -> Path:
    change = result["inference"]["acquisition_change"]
    final = result["inference"]["final_session"]
    lines = [
        "% Generated from jeong000351_directional_reference.json; do not edit.",
        f"\\newcommand{{\\JeongSubjectCount}}{{{result['subject_count']}}}",
        f"\\newcommand{{\\JeongSessionCount}}{{{result['session_count']}}}",
        f"\\newcommand{{\\JeongAcquisitionDifference}}{{{change['estimate_control_minus_inhibited']:.2f}}}",
        f"\\newcommand{{\\JeongAcquisitionLo}}{{{change['ci95'][0]:.2f}}}",
        f"\\newcommand{{\\JeongAcquisitionHi}}{{{change['ci95'][1]:.2f}}}",
        f"\\newcommand{{\\JeongFinalDifference}}{{{final['estimate_control_minus_inhibited']:.2f}}}",
        f"\\newcommand{{\\JeongFinalLo}}{{{final['ci95'][0]:.2f}}}",
        f"\\newcommand{{\\JeongFinalHi}}{{{final['ci95'][1]:.2f}}}",
    ]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def _write_figure(result: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    subjects = result["subject_rows"]
    colors = {"control": "#202020", "inhibited": "#c43c39"}
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), constrained_layout=True)
    for group in ("control", "inhibited"):
        rows = [row for row in subjects if row["group"] == group]
        matrix = np.asarray([row["source_display_sessions"] for row in rows])
        for values in matrix:
            axes[0].plot(np.arange(5), values, color=colors[group], alpha=0.22, lw=0.8)
        mean = np.mean(matrix, axis=0)
        sem = np.std(matrix, axis=0, ddof=1) / np.sqrt(matrix.shape[0])
        axes[0].errorbar(
            np.arange(5), mean, yerr=sem, marker="o", color=colors[group],
            capsize=2, lw=2, label=f"{group} (n={len(rows)})",
        )
    axes[0].axhline(0, color="0.7", lw=0.8)
    axes[0].set_xticks(np.arange(5), ["1", "2", "3", "n−1", "n"])
    axes[0].set(
        xlabel="conditioning session",
        ylabel="anticipatory licks above baseline",
        title="Source-defined learning trajectory",
    )
    axes[0].legend(frameon=False, fontsize=8)

    positions = {"control": 0, "inhibited": 1}
    rng = np.random.default_rng(19)
    for group in ("control", "inhibited"):
        rows = [row for row in subjects if row["group"] == group]
        values = np.asarray([row["acquisition_change"] for row in rows])
        x = positions[group] + rng.uniform(-0.09, 0.09, size=values.size)
        axes[1].scatter(x, values, color=colors[group], s=28, alpha=0.8)
        axes[1].errorbar(
            positions[group], np.mean(values),
            yerr=np.std(values, ddof=1) / np.sqrt(values.size),
            marker="_", markersize=20, color="black", capsize=3, lw=1.2,
        )
    axes[1].axhline(0, color="0.7", lw=0.8)
    axes[1].set_xticks([0, 1], ["control", "inhibited"])
    axes[1].set(
        ylabel="final minus first session (licks)",
        title="Mouse-level acquisition change",
    )
    change = result["inference"]["acquisition_change"]
    axes[1].text(
        0.04, 0.96,
        f"control − inhibited = {change['estimate_control_minus_inhibited']:.2f}\n"
        f"95% bootstrap [{change['ci95'][0]:.2f}, {change['ci95'][1]:.2f}]",
        transform=axes[1].transAxes, va="top", fontsize=8,
    )
    for label, axis in zip("ab", axes):
        axis.text(-0.15, 1.06, label, transform=axis.transAxes, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    figure.savefig(path, dpi=300, bbox_inches="tight", metadata={"Software": "mrl_trace.jeong"})
    plt.close(figure)


def write_jeong_artifacts(result: dict, output_dir) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference = output / "jeong000351_directional_reference.json"
    sessions = output / "jeong000351_directional_sessions.csv"
    figure = output / "fig_jeong_directional.png"
    macros = output / "jeong_macros.tex"
    payload = dict(result)
    rows = payload.pop("session_rows")
    _write_json(reference, payload)
    _write_csv(sessions, rows)
    _write_figure(result, figure)
    write_jeong_tex_macros(result, macros)
    return {
        "reference": str(reference),
        "sessions": str(sessions),
        "figure": str(figure),
        "macros": str(macros),
    }
