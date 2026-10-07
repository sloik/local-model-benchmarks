#!/usr/bin/env python3
"""Import a completed legacy Hermes result directory as immutable benchmark evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from agentic_suite import (
    AGENTIC_SUITE_PATH,
    DEFAULT_CONTEXT,
    ROOT,
    RESULTS_ROOT,
    load_json,
    resolve_context,
    validate_agentic_suite,
    validate_manifest,
)


class ImportError(RuntimeError):
    """Legacy evidence is incomplete, mutable, unsafe, or conflicts with a target."""


@dataclass(frozen=True)
class ImportResult:
    output_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    dry_run: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_snapshot(root: Path) -> dict[str, Any]:
    """Hash every regular source byte using stable portable relative paths."""
    if not root.is_dir():
        raise ImportError(f"legacy source is not a directory: {root}")
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ImportError(f"legacy source contains unsupported symlink: {path}")
        if path.is_file():
            rows.append({
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            })
    if not rows:
        raise ImportError(f"legacy source contains no files: {root}")
    aggregate = hashlib.sha256(
        "".join(
            f"{row['path']}\0{row['size_bytes']}\0{row['sha256']}\n" for row in rows
        ).encode("utf-8")
    ).hexdigest()
    return {"sha256": aggregate, "files": rows}


def selected_evidence_snapshot(
    source: Path, summary_path: Path, selected_cases: Sequence[str]
) -> dict[str, Any]:
    """Hash exactly the bytes eligible for a selected legacy import.

    The canonical destination layout is always ``summary.json`` plus one
    directory per selected case.  Unselected directories are deliberately not
    observed or copied: they may contain an interrupted next case and are not
    evidence for the selected run.
    """
    if summary_path.is_symlink() or not summary_path.is_file():
        raise ImportError(f"legacy summary must be one regular file: {summary_path}")
    entries: list[tuple[Path, str]] = [(summary_path, "summary.json")]
    for case_id in selected_cases:
        case_root = source / case_id
        if case_root.is_symlink() or not case_root.is_dir():
            raise ImportError(f"selected case directory is absent: {case_root}")
        for path in sorted(case_root.rglob("*")):
            if path.is_symlink():
                raise ImportError(f"legacy source contains unsupported symlink: {path}")
            if path.is_file():
                relative = path.relative_to(case_root).as_posix()
                entries.append((path, f"{case_id}/{relative}"))
    rows = [
        {"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path, relative in sorted(entries, key=lambda item: item[1])
    ]
    aggregate = hashlib.sha256(
        "".join(
            f"{row['path']}\0{row['size_bytes']}\0{row['sha256']}\n" for row in rows
        ).encode("utf-8")
    ).hexdigest()
    return {"sha256": aggregate, "files": rows}


def _load_summary(path: Path) -> dict[str, Any]:
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ImportError(f"completed summary is unreadable: {path}: {exc}") from exc
    if not isinstance(summary, dict):
        raise ImportError("completed summary must be a JSON object")
    return summary


def _validate_summary(
    source: Path, summary_path: Path, selected_cases: Sequence[str]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = _load_summary(summary_path)
    runs = summary.get("runs")
    if not isinstance(runs, list) or not runs or not all(isinstance(row, dict) for row in runs):
        raise ImportError("completed summary must contain a non-empty runs list")
    by_case: dict[str, dict[str, Any]] = {}
    allowed_statuses = {"pass", "failed", "timeout"}
    for row in runs:
        case_id = row.get("eval_id")
        if not isinstance(case_id, str) or not case_id:
            raise ImportError("completed summary run is missing eval_id")
        if case_id in by_case:
            raise ImportError(f"completed summary contains duplicate case {case_id}")
        if row.get("status") not in allowed_statuses:
            raise ImportError(f"completed summary has invalid status for {case_id}: {row.get('status')}")
        if not isinstance(row.get("model"), str) or not row["model"]:
            raise ImportError(f"completed summary run {case_id} is missing model")
        by_case[case_id] = row

    missing = sorted(set(selected_cases) - by_case.keys())
    if missing:
        raise ImportError(f"completed summary is missing selected cases: {', '.join(missing)}")
    if summary.get("total_runs") != len(runs):
        raise ImportError("completed summary total_runs does not match runs")
    expected_counts = {
        "total_pass": sum(row["status"] == "pass" for row in runs),
        "total_failed": sum(row["status"] == "failed" for row in runs),
        "total_timeout": sum(row["status"] == "timeout" for row in runs),
    }
    for key, expected in expected_counts.items():
        if summary.get(key) != expected:
            raise ImportError(f"completed summary {key} does not match runs")

    selected_rows = [by_case[case_id] for case_id in selected_cases]
    for row in selected_rows:
        result_path = source / row["eval_id"] / "result.json"
        if not result_path.is_file():
            raise ImportError(f"selected case result is absent: {result_path}")
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ImportError(f"selected case result is unreadable: {result_path}: {exc}") from exc
        for key in ("eval_id", "model", "status"):
            if result.get(key) != row.get(key):
                raise ImportError(
                    f"selected case result disagrees with completed summary for {row['eval_id']} {key}"
                )
    return summary, selected_rows


def _source_git_identity(source: Path) -> tuple[str | None, str]:
    try:
        root_proc = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "--show-toplevel"],
            capture_output=True, text=True, timeout=10,
        )
        commit_proc = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, source.name
    if root_proc.returncode != 0 or commit_proc.returncode != 0:
        return None, source.name
    git_root = Path(root_proc.stdout.strip()).resolve()
    try:
        relative = source.resolve().relative_to(git_root).as_posix()
    except ValueError:
        return commit_proc.stdout.strip(), source.name
    portable = f"{git_root.name}/{relative}" if relative != "." else git_root.name
    return commit_proc.stdout.strip(), portable


def _common_int(rows: Sequence[dict[str, Any]], key: str, fallback: int) -> int:
    values = {row.get(key) for row in rows if isinstance(row.get(key), int)}
    if len(values) > 1:
        raise ImportError(f"completed summary has inconsistent {key} values: {sorted(values)}")
    return next(iter(values), fallback)


def _tools(rows: Sequence[dict[str, Any]], fallback: list[str]) -> list[str]:
    values = [row.get("toolsets") for row in rows if row.get("toolsets") is not None]
    normalized = [
        tuple(item for item in value.split(",") if item) if isinstance(value, str)
        else tuple(value) if isinstance(value, list) and all(isinstance(item, str) for item in value)
        else None
        for value in values
    ]
    if any(value is None for value in normalized):
        raise ImportError("completed summary toolsets must be a comma string or string list")
    distinct = set(normalized)
    if len(distinct) > 1:
        raise ImportError("completed summary has inconsistent toolsets values")
    if not distinct:
        return fallback
    return list(next(iter(distinct)))


def _default_run_id(source: Path, now: datetime) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", source.name).strip("-") or "legacy"
    return f"{now.strftime('%Y-%m-%dT%H%M%SZ')}_agentic-import-{slug}"


def _validate_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ImportError(f"run_id must be one portable directory name: {run_id!r}")


def _build_import_manifest(
    *, source: Path, summary_path: Path, summary_hash: str,
    source_snapshot: dict[str, Any], destination_snapshot: dict[str, Any],
    selected_rows: Sequence[dict[str, Any]], suite: dict[str, Any], context: int,
    output_path: Path, run_id: str, imported_at: datetime,
) -> dict[str, Any]:
    source_commit, source_portable = _source_git_identity(source)
    models = sorted({row["model"] for row in selected_rows})
    cases = [row["eval_id"] for row in selected_rows]
    statuses = {row["eval_id"]: row["status"] for row in selected_rows}
    timestamps = sorted(
        row["timestamp"] for row in selected_rows if isinstance(row.get("timestamp"), str)
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "suite_id": suite["suite_id"],
        "models": models,
        "harness": {"name": "hermes", "version": "legacy-unreported"},
        "requested_context": resolve_context(context),
        "observed_contexts": {model: None for model in models},
        "observation_state": "legacy-import-context-not-observed",
        "timeout_seconds": _common_int(
            selected_rows, "timeout_limit_s", suite["policy"]["timeout_seconds"]
        ),
        "minimum_free_memory_mb": _common_int(
            selected_rows, "minimum_free_memory_mb", suite["policy"]["minimum_free_memory_mb"]
        ),
        "max_turns": _common_int(selected_rows, "max_turns", suite["policy"]["max_turns"]),
        "tools": _tools(selected_rows, list(suite["policy"]["toolsets"])),
        "dependencies": {
            "layout": "Nightshift run_hermes_evals.sh legacy output",
            "source_git_commit": source_commit,
        },
        "suite_identity": {
            "path": str(AGENTIC_SUITE_PATH.relative_to(ROOT)),
            "sha256": _sha256(AGENTIC_SUITE_PATH),
        },
        "model_artifacts": {model: {"state": "legacy-summary-only"} for model in models},
        "fixture_identity": {
            "state": "legacy-layout-artifact-tree",
            "sha256": source_snapshot["sha256"],
        },
        "scorer_identity": {
            "state": "legacy-layout-summary",
            "summary_sha256": summary_hash,
        },
        "cases": cases,
        "case_statuses": statuses,
        "started_at": timestamps[0] if timestamps else imported_at.isoformat(),
        "canonical_output_path": str(output_path.resolve()),
        "provenance": "imported-legacy-layout",
        "legacy_import": {
            "source_absolute_path": str(source.resolve()),
            "source_portable_path": source_portable,
            "source_git_commit": source_commit,
            "source_summary_path": str(summary_path.resolve()),
            "source_summary_sha256": summary_hash,
            "source_aggregate_sha256": source_snapshot["sha256"],
            "destination_aggregate_sha256": destination_snapshot["sha256"],
            "imported_at": imported_at.isoformat(),
            "files": source_snapshot["files"],
        },
    }
    validate_manifest(manifest)
    return manifest


def import_legacy_results(
    source: Path | str,
    *,
    results_root: Path = RESULTS_ROOT,
    run_id: str | None = None,
    summary_path: Path | None = None,
    selected_cases: Sequence[str] | None = None,
    context: int = DEFAULT_CONTEXT,
    dry_run: bool = False,
    copy_file: Callable[[Path, Path], Any] = shutil.copy2,
    now: datetime | None = None,
) -> ImportResult:
    source = Path(source).expanduser().resolve()
    suite = load_json(AGENTIC_SUITE_PATH)
    validate_agentic_suite(suite)
    known_cases = [case["id"] for case in suite["cases"]]
    selected = list(selected_cases or known_cases)
    unknown = sorted(set(selected) - set(known_cases))
    if unknown:
        raise ImportError(f"unknown selected cases: {', '.join(unknown)}")
    if len(selected) != len(set(selected)):
        raise ImportError("selected cases contain duplicates")
    resolve_context(context)

    summary = (summary_path or source / "summary.json").expanduser().resolve()
    _summary_value, selected_rows = _validate_summary(source, summary, selected)
    summary_hash = _sha256(summary)
    source_before = selected_evidence_snapshot(source, summary, selected)
    imported_at = now or datetime.now(timezone.utc)
    run_id = run_id or _default_run_id(source, imported_at)
    _validate_run_id(run_id)
    results_root = Path(results_root).expanduser().resolve()
    output_path = results_root / run_id
    if output_path.exists():
        raise ImportError(f"target already exists; refusing overwrite: {output_path}")

    if dry_run:
        manifest = _build_import_manifest(
            source=source, summary_path=summary, summary_hash=summary_hash,
            source_snapshot=source_before, destination_snapshot=source_before,
            selected_rows=selected_rows, suite=suite, context=context,
            output_path=output_path, run_id=run_id, imported_at=imported_at,
        )
        return ImportResult(output_path, output_path / "run-manifest.json", manifest, True)

    results_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.import-", dir=results_root))
    try:
        artifacts = staging / "legacy-artifacts"
        artifacts.mkdir()
        for row in source_before["files"]:
            src = summary if row["path"] == "summary.json" else source / row["path"]
            dst = artifacts / row["path"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            copy_file(src, dst)
        source_after = selected_evidence_snapshot(source, summary, selected)
        if source_after != source_before or _sha256(summary) != summary_hash:
            raise ImportError("source changed during import; no canonical run was published")
        destination = tree_snapshot(artifacts)
        if destination != source_before:
            raise ImportError("destination byte hashes do not match source after copy")
        manifest = _build_import_manifest(
            source=source, summary_path=summary, summary_hash=summary_hash,
            source_snapshot=source_before, destination_snapshot=destination,
            selected_rows=selected_rows, suite=suite, context=context,
            output_path=output_path, run_id=run_id, imported_at=imported_at,
        )
        (staging / "run-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if output_path.exists():
            raise ImportError(f"target appeared during import; refusing overwrite: {output_path}")
        os.rename(staging, output_path)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return ImportResult(output_path, output_path / "run-manifest.json", manifest, False)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Completed legacy result directory")
    parser.add_argument("--summary", type=Path, help="Explicit completed summary (default: SOURCE/summary.json)")
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--run-id", help="Canonical target directory name; default is timestamped")
    parser.add_argument("--evals", nargs="+", help="Expected selected EVAL IDs; default is all five")
    parser.add_argument("--ctx", type=int, default=DEFAULT_CONTEXT, help="Requested legacy run context")
    parser.add_argument("--dry-run", action="store_true", help="Validate and hash; perform no writes")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = import_legacy_results(
            args.source, results_root=args.results_root, run_id=args.run_id,
            summary_path=args.summary, selected_cases=args.evals, context=args.ctx,
            dry_run=args.dry_run,
        )
    except (ImportError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    legacy = result.manifest["legacy_import"]
    print("DRY RUN — no files written" if result.dry_run else "IMPORT COMPLETE")
    print(f"provenance: {result.manifest['provenance']}")
    print(f"source: {legacy['source_absolute_path']}")
    print(f"source aggregate SHA-256: {legacy['source_aggregate_sha256']}")
    print(f"files verified: {len(legacy['files'])}")
    print(f"target: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
