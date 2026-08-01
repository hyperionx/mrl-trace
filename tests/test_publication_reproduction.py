from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
TOPIC_NOTEBOOKS = tuple(
    EXPERIMENTS / name
    for name in (
        "00_device_physics_and_trace.ipynb",
        "01_distal_credit_ladder.ipynb",
        "02_sequential_and_scaling.ipynb",
        "03_deep_local_and_faults.ipynb",
        "04_biological_grounding.ipynb",
        "05_extensions.ipynb",
    )
)
PREDICTIVE_NOTEBOOK = EXPERIMENTS / "06_nmi_predictive_linkage.ipynb"
REPRODUCE_NOTEBOOK = EXPERIMENTS / "REPRODUCE.ipynb"
REPRODUCTION_NOTEBOOKS = (*TOPIC_NOTEBOOKS, PREDICTIVE_NOTEBOOK, REPRODUCE_NOTEBOOK)


def _notebook(path: Path) -> dict:
    """Read a notebook without requiring nbformat as a test dependency."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert isinstance(payload["cells"], list)
    return payload


def _source(path: Path, *, code_only: bool = False) -> str:
    cells = _notebook(path)["cells"]
    if code_only:
        cells = [cell for cell in cells if cell["cell_type"] == "code"]
    return "\n".join("".join(cell.get("source", ())) for cell in cells)


def _tex_macros(path: Path) -> dict[str, str]:
    """Read scalar generated macros used for cross-repository number checks."""
    return dict(
        re.findall(
            r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$",
            path.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )


def _literal_assignment(path: Path, name: str):
    """Return a literal notebook-level assignment such as the OWNED registry."""
    found = []
    for index, cell in enumerate(_notebook(path)["cells"]):
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell.get("source", ())), filename=f"{path.name}:cell-{index}")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                found.append(ast.literal_eval(node.value))
    assert len(found) == 1, f"expected one literal {name} registry in {path.name}"
    return found[0]


def _public_materials() -> tuple[Path, ...]:
    return (ROOT / "README.md", EXPERIMENTS / "README.md", *REPRODUCTION_NOTEBOOKS)


def test_all_reproduction_notebooks_parse_and_every_code_cell_compiles() -> None:
    assert all(path.is_file() for path in REPRODUCTION_NOTEBOOKS)
    for path in REPRODUCTION_NOTEBOOKS:
        notebook = _notebook(path)
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                compile(
                    "".join(cell.get("source", ())),
                    f"{path.name}:cell-{index}",
                    "exec",
                )


def test_committed_notebooks_embed_successful_reduced_outputs() -> None:
    for path in REPRODUCTION_NOTEBOOKS:
        notebook = _notebook(path)
        assert notebook["metadata"]["mrl_trace_execution"] == {
            "embedded_outputs": True,
            "external_data": False,
            "profile": "reduced",
        }, path.name
        code_cells = [
            cell for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        assert code_cells, path.name
        assert all(
            cell.get("execution_count") is not None for cell in code_cells
        ), path.name
        assert not [
            output
            for cell in code_cells
            for output in cell.get("outputs", ())
            if output.get("output_type") == "error"
        ], path.name


def test_topic_notebooks_embed_common_controls_and_compact_unique_registries() -> None:
    controls = {
        "RUN_PROFILE",
        "DEVICE",
        "WORKERS",
        "SAVE_FIGURES",
        "OUTPUT_DIR",
        "OVERWRITE",
        "RUN_EXTERNAL_DATA",
        "ALLOW_DATA_DOWNLOADS",
    }
    all_owned: list[str] = []

    for path in (*TOPIC_NOTEBOOKS, PREDICTIVE_NOTEBOOK):
        code = _source(path, code_only=True)
        missing = {control for control in controls if control not in code}
        assert not missing, f"{path.name} lacks common controls: {sorted(missing)}"
        assert "FIGURE_REPORT = []" in code
        assert "method_provenance" in _source(path)

        owned = tuple(_literal_assignment(path, "OWNED"))
        assert owned
        assert len(owned) == len(set(owned))
        assert all(re.fullmatch(r"fig_[a-z0-9_]+\.png", item) for item in owned)
        assert all(item in code for item in owned)
        all_owned.extend(owned)

    # Ownership is notebook-local, but no figure can silently have two owners.
    assert len(all_owned) == len(set(all_owned))


def test_reproduce_explicitly_orchestrates_notebooks_without_external_manifest() -> None:
    source = _source(REPRODUCE_NOTEBOOK)
    for path in TOPIC_NOTEBOOKS:
        assert f'"{path.name}"' in source
    assert '"06_nmi_predictive_linkage.ipynb"' in source
    assert "TOPIC_NOTEBOOKS" in source
    assert "RUN_PREDICTIVE_LINKAGE" in source
    assert 'RUN_PROFILE == "publication"' in source
    assert "subprocess.run" in source
    assert '"-m", "nbconvert"' in source
    assert "OUTER_FAN" in source and "INNER_PARALLEL" in source
    assert "MRL_SAVE_RESULTS" in source
    assert "publication reproduction requires saved result tables and figures" in source
    assert 'KERNEL_NAME = os.getenv("MRL_KERNEL_NAME", "python3")' in source
    assert '"-1" if RUN_PROFILE == "publication"' in source
    assert "mrl-trace-venv" not in source

    combined = "\n".join(_source(path) for path in REPRODUCTION_NOTEBOOKS).lower()
    assert "figure_manifest.json" not in combined
    assert "data/publication" not in combined
    assert "data\\publication" not in combined
    package_manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8").lower()
    assert "data/publication" not in package_manifest
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8").lower()
    assert "data/publication" not in gitignore


def test_notebooks_compute_live_and_do_not_replay_raster_results() -> None:
    forbidden = (
        "render_entry",
        "plt.imread",
        "mpimg.imread",
        "matplotlib.image.imread",
        "pil.image.open",
        "image.open(",
        "aggregates.json",
    )
    for path in (*TOPIC_NOTEBOOKS, PREDICTIVE_NOTEBOOK):
        code = _source(path, code_only=True).lower()
        assert not [marker for marker in forbidden if marker in code], path.name
        assert "display(fig)" in code or path == PREDICTIVE_NOTEBOOK


def test_withdrawn_claims_and_archives_are_absent_from_public_materials() -> None:
    # Explanations may say that a historical capstone or transformed retention is
    # invalid. These exact identifiers and affirmative claims must not reappear.
    forbidden = (
        "exp11_dopamine_capstone.npy",
        "fig_dopamine_capstone.png",
        "tau_fill_corrected_s",
        "tau/(1-rho)",
        "tau / (1 - rho)",
        "run_dmax_law",
        "faithful e-prop",
        "offline replay",
        "per-trial verdict",
        "balanced accuracy 0.89",
        "auc 0.92",
        "1.586 s",
        "measured at ~25 aj",
    )
    for path in _public_materials():
        text = path.read_text(encoding="utf-8").lower()
        matches = [phrase for phrase in forbidden if phrase in text]
        assert not matches, f"withdrawn public material in {path.name}: {matches}"


def test_direct_retention_loader_fails_closed_and_rejects_legacy_archives(tmp_path: Path) -> None:
    from mrl_trace.extensions import load_measured_tau

    missing = tmp_path / "missing-ito-fits.npz"
    with pytest.raises(FileNotFoundError, match="measured ITO retention archive unavailable"):
        load_measured_tau(archive=missing)

    legacy = tmp_path / "legacy-transformed-retention.npz"
    np.savez(
        legacy,
        tau=np.linspace(0.6, 2.0, 12),
        retention_definition="floor_corrected_tau",
        analysis_schema_version=1,
    )
    with pytest.raises(ValueError, match="legacy or transformed"):
        load_measured_tau(archive=legacy)

    current = tmp_path / "direct-held-bias-retention.npz"
    expected = np.linspace(0.6, 2.0, 12)
    np.savez(
        current,
        tau=expected,
        retention_definition="direct_held_bias_tau",
        analysis_schema_version=2,
    )
    observed, description = load_measured_tau(archive=current)
    np.testing.assert_allclose(observed, expected)
    assert "direct held-bias fits" in description


def test_optional_measured_retention_fixture_uses_current_schema() -> None:
    from mrl_trace.extensions import load_measured_tau

    fixture = ROOT / "data" / "device_model" / "ito_decay_data.npz"
    if not fixture.is_file():
        pytest.skip("optional direct-held-bias ITO fit fixture is not present in this checkout")
    tau, source = load_measured_tau(archive=fixture)
    assert tau.size >= 10
    assert np.isfinite(tau).all()
    assert (tau > 0).all()
    assert "direct held-bias" in source


def test_predictive_notebook_records_direct_retention_and_fails_closed_on_raw_counts() -> None:
    source = _source(PREDICTIVE_NOTEBOOK)
    assert 'retention_definition": "direct_held_bias_tau"' in source
    assert 'analysis_schema_version": 2' in source
    assert 'len(list(candidate.glob("trace_*.csv"))) == 24' in source
    assert "select_ito_source_cohort(candidate)" in source
    assert '"primary_january_2025"' in source
    assert '"supplemental_near_zero_june_2026"' in source
    assert '"later_addition"' in source
    assert "later acquisition outside frozen January 2025 cohort" in source
    assert "expected 24 Au traces" in source
    assert "91-workbook primary cohort and two 1 mV supplements not found" in source
    assert 'assert not any(row["qc_status"] == "pending_fit"' in source
    assert '"device_identity": "unrecoverable"' in source
    assert 'NEAR_ZERO_MAX_V = 0.002' in source
    assert '"measurement_regime": _measurement_regime(bias_v)' in source
    assert 'row["measurement_regime"] == "held_bias"' in source
    assert '"near_zero_read"' in source
    assert "publication mode requires result tables and figures to be saved" in source
    assert "kaplan_meier_recovery_summary" in source
    for artifact in (
        "raw_manifest.csv", "raw_manifest.json", "gold_cv_rows.csv",
        "gold_fit_rows.csv", "ito_fit_rows.csv",
        "predictive_interval_seeds.csv", "retention_delay_seeds.csv",
        "reversal_seeds.csv", "figure_sources.json", "run_report.json",
    ):
        assert artifact in source
    assert '"evidential_use": "publication-scale" if RUN_PROFILE == "publication" else "feasibility-only"' in source


def test_public_material_does_not_expose_dissertation_routing() -> None:
    public = (ROOT / "README.md", EXPERIMENTS / "README.md", *TOPIC_NOTEBOOKS)
    for path in public:
        text = path.read_text(encoding="utf-8").lower()
        assert "appendix" not in text
        assert not re.search(r"chapter\s+[1-9]", text)


def test_companion_manuscript_uses_artifact_macros_and_shared_model_ids() -> None:
    candidates = (
        (ROOT.parent / "manuscript-MRL" / "main.tex",
         ROOT.parent / "manuscript-MRL" / "benchmark_macros.tex"),
        (ROOT.parent / "Manuscript-MRL" / "manuscript.tex",
         ROOT.parent / "Manuscript-MRL" / "benchmark_macros.tex"),
    )
    pair = next(((tex, macros) for tex, macros in candidates if tex.is_file()), None)
    if pair is None:
        pytest.skip("companion manuscript checkout is not present")
    manuscript, manuscript_macros = pair
    reference_macros = ROOT / "data/results/reference/benchmark_macros.tex"
    assert manuscript_macros.read_text(encoding="utf-8") == (
        reference_macros.read_text(encoding="utf-8")
    )
    source = manuscript.read_text(encoding="utf-8")
    assert "physical\\_headroom\\_v1" in source or "physical headroom" in source
    assert "linear\\_erlang\\_v1" in source or "linear Erlang" in source
    assert "fig_sequential_fair.png" in source
    assert "fig_interval_fair.png" in source
    assert "Conditional" + " Go" in source
    assert "coddington2023mesolimbic" in source
    assert "jeong2022mesolimbic" in source
    assert "fig_coddington_causal.png" in source
    assert "fig_jeong_directional.png" in source


def test_companion_dissertation_uses_current_artifacts_and_withdraws_stale_claims() -> None:
    dissertation = ROOT.parent / "phd-dissertation"
    chapter = dissertation / "Chapter7" / "chapter7.tex"
    if not chapter.is_file():
        pytest.skip("companion dissertation checkout is not present")

    source_files = tuple(
        dissertation / name
        for name in (
            "Abstract/abstract.tex", "Impact/impact.tex", "Chapter1/chapter1.tex",
            "Chapter5/chapter5.tex", "Chapter6/chapter6.tex",
            "Chapter7/chapter7.tex", "Chapter8/chapter8.tex",
        )
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in source_files)
    lowered = combined.lower()
    for stale in (
        "eeg_reward_capstone", "reward_pools",
        "over-credits the near-reward distractor",
        "three microscopic stages",
    ):
        assert stale not in lowered

    assert "physical\\_headroom\\_v1" not in combined
    assert "linear\\_erlang\\_v1" not in combined
    assert "Conditional" + " Go" in combined
    assert "positive causal" in lowered
    assert "directionally consistent" in lowered
    assert "fig_codesign.png" in combined
    assert "fig_sequential_fair.png" in combined
    assert "fig_shallow_dms_fair.png" in combined
    assert "fig_dopamine_replay.png" in combined
    assert "fig_coddington_causal.png" in combined
    assert "fig_jeong_directional.png" in combined

    chapter7 = (dissertation / "Chapter7" / "chapter7.tex").read_text(
        encoding="utf-8"
    )
    macro_names = (
        "benchmark_macros.tex",
        "primitive_macros.tex",
        "coddington_macros.tex",
        "jeong_macros.tex",
    )
    assert all(not (dissertation / "Chapter7" / name).exists() for name in macro_names)
    assert all(name not in chapter7 for name in macro_names)

    benchmark = _tex_macros(ROOT / "data/results/reference/benchmark_macros.tex")
    primitive = _tex_macros(ROOT / "data/results/reference/primitive_macros.tex")
    coddington = _tex_macros(ROOT / "data/results/reference/coddington_macros.tex")
    jeong = _tex_macros(ROOT / "data/results/reference/jeong_macros.tex")
    assert (
        f"The nonlinear physical primary reaches AULC "
        f"{benchmark['SequentialDeviceAULC']}."
    ) in chapter7
    assert (
        f"full KWW refits give $\\beta_{{fill}}={primitive['BetaFillBootstrapMedian']}$ "
        f"[{primitive['BetaFillBootstrapLo']}, {primitive['BetaFillBootstrapHi']}]"
    ) in chapter7
    assert (
        f"groups separate by {coddington['CoddingtonContrastHz']}\\,Hz"
    ) in chapter7
    assert (
        f"reproduced across {jeong['JeongSubjectCount']} mice and "
        f"{jeong['JeongSessionCount']} conditioning sessions"
    ) in chapter7
