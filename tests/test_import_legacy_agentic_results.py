"""SPEC-003-014 R6/AC7 legacy agentic result import."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import agentic_suite  # noqa: E402
import import_legacy_agentic_results as sut  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _legacy_source(tmp_path: Path, cases: tuple[str, ...] = ("EVAL-001", "EVAL-002")) -> Path:
    source = tmp_path / "nightshift-results" / "model-a-hermes"
    runs = []
    for index, case_id in enumerate(cases, start=1):
        row = {
            "eval_id": case_id,
            "model": "model-a",
            "harness": "hermes",
            "scoring_version": 2,
            "status": "pass" if index == 1 else "timeout",
            "timestamp": f"2026-08-23T0{index}:00:00Z",
            "timeout_limit_s": 1800,
            "max_turns": 30,
            "toolsets": "file,terminal,code_execution",
        }
        runs.append(row)
        _write_json(source / case_id / "result.json", row)
        (source / case_id / "hermes.log").write_bytes(f"log-{case_id}\n".encode())
    _write_json(
        source / "summary.json",
        {
            "harness": "hermes",
            "runs": runs,
            "generated": "2026-08-23T03:00:00+00:00",
            "total_runs": len(runs),
            "total_pass": sum(row["status"] == "pass" for row in runs),
            "total_failed": sum(row["status"] == "failed" for row in runs),
            "total_timeout": sum(row["status"] == "timeout" for row in runs),
        },
    )
    return source


def _import(source: Path, results_root: Path, **kwargs):
    return sut.import_legacy_results(
        source,
        results_root=results_root,
        run_id="legacy-test-run",
        selected_cases=["EVAL-001", "EVAL-002"],
        context=131072,
        **kwargs,
    )


def test_import_copies_byte_identical_files_and_writes_schema_valid_manifest(tmp_path):
    source = _legacy_source(tmp_path)
    before = sut.tree_snapshot(source)

    result = _import(source, tmp_path / "results")
    manifest = json.loads(result.manifest_path.read_text())

    assert sut.tree_snapshot(source) == before
    assert manifest["provenance"] == "imported-legacy-layout"
    assert manifest["models"] == ["model-a"]
    assert manifest["cases"] == ["EVAL-001", "EVAL-002"]
    assert manifest["case_statuses"] == {"EVAL-001": "pass", "EVAL-002": "timeout"}
    assert manifest["legacy_import"]["source_absolute_path"] == str(source.resolve())
    assert manifest["legacy_import"]["source_summary_sha256"] == hashlib.sha256(
        (source / "summary.json").read_bytes()
    ).hexdigest()
    assert manifest["legacy_import"]["source_aggregate_sha256"] == before["sha256"]
    assert manifest["legacy_import"]["destination_aggregate_sha256"] == before["sha256"]
    assert manifest["legacy_import"]["files"] == before["files"]
    assert (result.output_path / "legacy-artifacts" / "EVAL-001" / "hermes.log").read_bytes() == (
        source / "EVAL-001" / "hermes.log"
    ).read_bytes()
    agentic_suite.validate_manifest(manifest)


def test_import_rejects_incomplete_or_unselected_summary(tmp_path):
    source = _legacy_source(tmp_path, cases=("EVAL-001",))
    with pytest.raises(sut.ImportError, match="missing selected cases.*EVAL-002"):
        _import(source, tmp_path / "results")
    assert not (tmp_path / "results").exists()

    source = _legacy_source(tmp_path / "other")
    (source / "EVAL-002" / "result.json").unlink()
    with pytest.raises(sut.ImportError, match="case result is absent"):
        _import(source, tmp_path / "other-results")


def test_import_rejects_source_mutation_and_removes_staging(tmp_path):
    source = _legacy_source(tmp_path)
    mutated = False

    def mutating_copy(src: Path, dst: Path) -> None:
        nonlocal mutated
        shutil.copy2(src, dst)
        if not mutated:
            mutated = True
            (source / "EVAL-001" / "hermes.log").write_bytes(b"changed-during-import\n")

    results_root = tmp_path / "results"
    with pytest.raises(sut.ImportError, match="source changed during import"):
        _import(source, results_root, copy_file=mutating_copy)
    assert not (results_root / "legacy-test-run").exists()
    assert not list(results_root.glob(".legacy-test-run.import-*"))


def test_import_refuses_existing_target_without_changing_it(tmp_path):
    source = _legacy_source(tmp_path)
    target = tmp_path / "results" / "legacy-test-run"
    target.mkdir(parents=True)
    marker = target / "owned.txt"
    marker.write_bytes(b"keep")
    with pytest.raises(sut.ImportError, match="target already exists"):
        _import(source, tmp_path / "results")
    assert marker.read_bytes() == b"keep"


def test_dry_run_validates_and_reports_without_writes(tmp_path):
    source = _legacy_source(tmp_path)
    results_root = tmp_path / "results"
    result = _import(source, results_root, dry_run=True)
    assert result.dry_run is True
    assert result.manifest["provenance"] == "imported-legacy-layout"
    assert result.manifest["canonical_output_path"].endswith("legacy-test-run")
    assert not results_root.exists()


def test_imported_provenance_is_distinct_from_native_manifest(tmp_path):
    source = _legacy_source(tmp_path)
    imported = _import(source, tmp_path / "results", dry_run=True).manifest
    native = agentic_suite.build_manifest(
        ["model-a"], agentic_suite.load_json(agentic_suite.AGENTIC_SUITE_PATH),
        {"hermes": "1"}, 131072, ["EVAL-001"], tmp_path / "native",
        {"model-a": None}, dry_run=True,
    )
    assert imported["provenance"] == "imported-legacy-layout"
    assert native["provenance"] == "native-unified"
    assert "legacy_import" in imported
    assert "legacy_import" not in native


def test_import_records_git_commit_and_portable_source_path_when_available(tmp_path):
    repo = tmp_path / "Nightshift"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    source = _legacy_source(repo)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()

    manifest = _import(source, tmp_path / "results", dry_run=True).manifest
    legacy = manifest["legacy_import"]
    assert legacy["source_git_commit"] == commit
    assert legacy["source_portable_path"] == "Nightshift/nightshift-results/model-a-hermes"


def test_missing_completed_summary_is_rejected_before_write(tmp_path):
    source = _legacy_source(tmp_path)
    (source / "summary.json").unlink()
    results_root = tmp_path / "results"
    with pytest.raises(sut.ImportError, match="completed summary is unreadable"):
        _import(source, results_root)
    assert not results_root.exists()


def test_cli_dry_run_prints_clear_diagnostics_and_writes_nothing(tmp_path, capsys):
    source = _legacy_source(tmp_path)
    rc = sut.main([
        str(source), "--results-root", str(tmp_path / "results"),
        "--run-id", "cli-dry", "--evals", "EVAL-001", "EVAL-002", "--dry-run",
    ])
    output = capsys.readouterr().out
    assert rc == 0
    assert "DRY RUN" in output
    assert "source aggregate SHA-256" in output
    assert "imported-legacy-layout" in output
    assert not (tmp_path / "results").exists()


def test_selected_import_excludes_partial_unselected_case_and_hashes_exact_bytes(tmp_path):
    source = _legacy_source(tmp_path, cases=("EVAL-001",))
    partial = source / "EVAL-002"
    partial.mkdir()
    (partial / "hermes.log").write_bytes(b"partial-invalid-evidence\n")

    result = sut.import_legacy_results(
        source,
        results_root=tmp_path / "results",
        run_id="selected-only",
        selected_cases=["EVAL-001"],
        context=131072,
    )
    artifacts = result.output_path / "legacy-artifacts"
    manifest = result.manifest

    assert (artifacts / "summary.json").is_file()
    assert (artifacts / "EVAL-001" / "result.json").is_file()
    assert not (artifacts / "EVAL-002").exists()
    assert {row["path"].split("/", 1)[0] for row in manifest["legacy_import"]["files"]} == {
        "summary.json", "EVAL-001",
    }
    copied = sut.tree_snapshot(artifacts)
    assert manifest["legacy_import"]["source_aggregate_sha256"] == copied["sha256"]
    assert manifest["legacy_import"]["destination_aggregate_sha256"] == copied["sha256"]
    assert manifest["legacy_import"]["source_summary_sha256"] == hashlib.sha256(
        (source / "summary.json").read_bytes()
    ).hexdigest()
    assert manifest["legacy_import"]["source_absolute_path"] == str(source.resolve())
