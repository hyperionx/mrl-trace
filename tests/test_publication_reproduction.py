from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT / "data" / "publication"
EXPERIMENTS = ROOT / "experiments"


def test_manifest_contract() -> None:
    manifest = json.loads((PUB / "figure_manifest.json").read_text(encoding="utf-8"))
    figures = manifest["figures"]
    assert manifest["figure_count"] == len(figures) == 41
    assert len({row["filename"] for row in figures}) == 41
    assert sum(row["tier"] == "main" for row in figures) == 10
    required = {"filename", "tier", "profile", "device", "seeds", "data_hash",
                "reference_hash", "provenance_class", "claim_status", "notebook"}
    assert all(required <= set(row) for row in figures)
    assert {row["provenance_class"] for row in figures} <= set(manifest["provenance_classes"])


def test_active_manuscript_figure_names() -> None:
    expected = {
        "fig_measured_transient.png", "fig_architecture.png", "fig_tier1_window.png",
        "fig_tier2_saturation.png", "fig_rl_curve.png", "fig_dmax_dense.png",
        "fig_distal_cue_deep.png", "fig_deep_local.png", "fig_deep_dms.png",
        "fig_dopamine_capstone.png",
    }
    manifest = json.loads((PUB / "figure_manifest.json").read_text(encoding="utf-8"))
    assert {row["filename"] for row in manifest["figures"] if row["tier"] == "main"} == expected
    manuscript = ROOT.parent / "manuscript-MRL" / "main.tex"
    if manuscript.exists():
        active = "\n".join(line for line in manuscript.read_text(encoding="utf-8").splitlines()
                           if not line.lstrip().startswith("%"))
        included = set(re.findall(r"includegraphics[^{}]*\{(fig_[^{}]+\.png)\}", active))
        assert included == expected


def test_curated_aggregates_are_summary_records() -> None:
    aggregates = json.loads((PUB / "aggregates.json").read_text(encoding="utf-8"))
    assert "fig_tier1_window.png" in aggregates
    assert "fig_rl_curve.png" not in aggregates  # a delay table is not a learning curve
    assert "tier3_delay_retention_summary" in aggregates
    assert aggregates["fig_dmax_dense.png"]["series"]["Dmax"][-1] == 637.9
    # Archive contains explicit published statistics, never arrays labelled as raw seeds.
    assert all("raw" not in payload and "seed_values" not in payload for payload in aggregates.values())


def test_curated_fixture_and_reference_hashes() -> None:
    fixtures = json.loads((PUB / "fixture_hashes.json").read_text(encoding="utf-8"))
    for relative, expected in fixtures.items():
        target = ROOT / relative
        assert target.is_file()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == expected

    manifest = json.loads((PUB / "figure_manifest.json").read_text(encoding="utf-8"))
    for row in manifest["figures"]:
        if row["reference_hash"] is None:
            continue
        target = PUB / "references" / row["filename"]
        assert target.is_file()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == row["reference_hash"]
        raw = target.read_bytes()
        assert raw[:8] == b"\x89PNG\r\n\x1a\n"
        width, height = int.from_bytes(raw[16:20], "big"), int.from_bytes(raw[20:24], "big")
        assert width >= 400 and height >= 250


def test_notebook_controls_and_ownership() -> None:
    manifest = json.loads((PUB / "figure_manifest.json").read_text(encoding="utf-8"))
    controls = ["RUN_PROFILE", "DEVICE", "WORKERS", "SAVE_FIGURES", "OUTPUT_DIR",
                "OVERWRITE", "RUN_EXTERNAL_DATA", "ALLOW_DATA_DOWNLOADS"]
    for notebook in sorted({row["notebook"] for row in manifest["figures"]}):
        text = (EXPERIMENTS / notebook).read_text(encoding="utf-8")
        assert all(control in text for control in controls)
        assert "SAVE_FIGURES = os.getenv" in text
        assert "render_entry" in text or "def finish(" in text


def test_device_notebook_is_fully_live_and_inline() -> None:
    text = (EXPERIMENTS / "00_device_physics_and_trace.ipynb").read_text(encoding="utf-8")
    assert "render_entry" not in text
    assert "reference-only" not in text
    assert "not regenerated" not in text.lower()
    assert "def finish(" in text
    assert "display(fig)" in text
    assert "plt.close(fig)" in text
    assert text.count('image/png') == 7


def test_distal_credit_notebook_is_fully_live_unique_and_appendix_faithful() -> None:
    text = (EXPERIMENTS / "01_distal_credit_ladder.ipynb").read_text(encoding="utf-8")
    assert "render_entry" not in text
    assert "reference only - not regenerated" not in text.lower()
    assert "reference only — not regenerated" not in text.lower()
    assert "AGGREGATES" not in text
    assert "run_trace_window" in text
    assert "run_spiking_saturation" in text
    assert "from mrl_trace.bandit import train" in text
    assert 'MRL_CHILD_PROCESS' in text
    assert 'RESOLVED_WORKERS = 1' in text
    assert "display(fig)" in text
    assert "plt.close(fig)" in text
    assert text.count('image/png') == 7
    for filename in {
        "fig_architecture.png", "fig_eligibility_electron.png",
        "fig_bio_silicon_map.png", "fig_tier1_window.png",
        "fig_tier2_saturation.png", "fig_rl_curve.png", "fig_crossbar_rl.png",
    }:
        assert text.count(filename) >= 2  # owning cell plus executed provenance record


def test_distal_credit_manifest_uses_live_or_authored_provenance() -> None:
    manifest = json.loads((PUB / "figure_manifest.json").read_text(encoding="utf-8"))
    owned = {
        row["filename"]: row for row in manifest["figures"]
        if row["notebook"] == "01_distal_credit_ladder.ipynb"
    }
    assert len(owned) == 7
    assert {row["provenance_class"] for row in owned.values()} == {
        "immutable-authored", "live-reduced"
    }
    assert owned["fig_tier1_window.png"]["seeds"] == 6
    assert owned["fig_tier2_saturation.png"]["seeds"] == 4
    assert owned["fig_rl_curve.png"]["seeds"] == 6


def test_external_data_is_gated() -> None:
    manifest = json.loads((PUB / "figure_manifest.json").read_text(encoding="utf-8"))
    external = [row for row in manifest["figures"] if row["provenance_class"] == "external-gated"]
    assert {row["filename"] for row in external} == {
        "fig_dopamine_signal.png", "fig_dopamine_capstone.png",
        "fig_probselect_eeg.png", "fig_eeg_combined.png",
    }
    assert all(row["claim_status"] == "external-data-required" for row in external)


def test_topic_notebook_ownership_is_unique_and_current() -> None:
    manifest = json.loads((PUB / "figure_manifest.json").read_text(encoding="utf-8"))
    expected_counts = {
        "00_device_physics_and_trace.ipynb": 7,
        "01_distal_credit_ladder.ipynb": 7,
        "02_sequential_and_scaling.ipynb": 4,
        "03_deep_local_and_faults.ipynb": 8,
        "04_biological_grounding.ipynb": 5,
        "05_extensions.ipynb": 10,
    }
    actual = {name: 0 for name in expected_counts}
    for row in manifest["figures"]:
        actual[row["notebook"]] += 1
        source = (EXPERIMENTS / row["notebook"]).read_text(encoding="utf-8")
        assert row["filename"] in source
    assert actual == expected_counts
    assert sum(actual.values()) == 41


def test_remaining_offline_notebooks_compute_live_without_raster_replay() -> None:
    for name in (
        "02_sequential_and_scaling.ipynb",
        "03_deep_local_and_faults.ipynb",
        "05_extensions.ipynb",
    ):
        text = (EXPERIMENTS / name).read_text(encoding="utf-8")
        assert "render_entry" not in text
        assert "AGGREGATES" not in text
        assert "plt.imread" not in text
        assert "REFERENCE ONLY - NOT REGENERATED" not in text.upper()
        assert "display(fig)" in text
        assert "plt.close(fig)" in text
    extensions = (EXPERIMENTS / "05_extensions.ipynb").read_text(encoding="utf-8")
    assert "MRL_CHILD_PROCESS" in extensions
    assert "RESOLVED_WORKERS = 1" in extensions


def test_live_manifest_provenance_for_notebooks_02_03_and_05() -> None:
    manifest = json.loads((PUB / "figure_manifest.json").read_text(encoding="utf-8"))
    rows = {row["filename"]: row for row in manifest["figures"]}
    for filename in {
        "fig_dmax_dense.png", "fig_sequential.png", "fig_scaling.png", "fig_hybrid.png",
        "fig_distal_cue_deep.png", "fig_deep_local.png", "fig_deep_dms.png",
        "fig_shallow_dms.png", "fig_array_faults.png", "fig_array_scale.png",
        "fig_betaval.png", "fig_interval.png", "fig_reversal.png",
        "fig_multitimescale.png", "fig_vector_timer.png", "fig_wm_stc.png",
        "fig_device_td.png", "fig_beta_sensitivity.png", "fig_long_horizon.png",
    }:
        assert rows[filename]["provenance_class"] == "live-reduced"
        assert rows[filename]["claim_status"] == "reduced-validation"
        assert rows[filename]["data_hash"] is None
    assert rows["fig_capture.png"]["provenance_class"] == "live-exact"
    assert rows["fig_homeostasis_stab.png"]["provenance_class"] == "immutable-authored"
    assert rows["fig_deep_crossbar.png"]["provenance_class"] == "immutable-authored"


def test_extensions_use_valid_measured_tau_population() -> None:
    from mrl_trace.extensions import load_measured_tau

    tau, source = load_measured_tau()
    assert len(tau) >= 10
    assert tau.min() >= 0.56
    assert tau.max() <= 7.72
    assert "rejected fits=" in source


def test_resource_aware_reproduce_driver() -> None:
    text = (EXPERIMENTS / "REPRODUCE.ipynb").read_text(encoding="utf-8")
    assert "OUTER_FAN" in text
    assert "INNER_PARALLEL" in text
    assert "MRL_CHILD_PROCESS" in text
    assert "mrl-trace-venv" in text
    assert "--ExecutePreprocessor.timeout=7200" in text


def test_public_material_does_not_expose_dissertation_routing() -> None:
    public = [ROOT / "README.md", EXPERIMENTS / "README.md"]
    public.extend(EXPERIMENTS.glob("[0-9][0-9]_*.ipynb"))
    for path in public:
        text = path.read_text(encoding="utf-8").lower()
        assert "appendix" not in text
        assert not re.search(r"chapter\s+[1-9]", text)
