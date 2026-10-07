#!/usr/bin/env python3
"""
run_provenance.py — Shared runtime/context provenance block (SPEC-003-020).

A tok/s figure, a fleet score, or a judge verdict is runtime-relative (recorded benchmark reliability finding), and the leaderboard carried no runtime label anywhere: 0/38 runs with
a parsable run_info.json. This module is the ONE shared builder both
scripts/run_benchmark.py (the runner) and scripts/evaluate.py (the scorer)
call, so their provenance blocks are structurally identical by construction
rather than by two independently-maintained implementations drifting apart.

Requirements: no external packages (stdlib only) — the schema validator below
is a minimal hand-rolled check against suites/schemas/run-info.schema.json's
specific shape, not a general JSON Schema implementation.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BENCH_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCH_DIR / "results"
SCHEMA_PATH = BENCH_DIR / "suites" / "schemas" / "run-info.schema.json"
RUN_INDEX_PATH = BENCH_DIR / "reports" / "run-index.json"

REQUIRED_FIELDS = (
    "runtime_name", "runtime_version", "model_artifact",
    "quantization", "speculative_decoding", "actual_context_length",
)


class ProvenanceSchemaError(Exception):
    """A provenance block failed schema validation."""


def fetch_runtime_version() -> str:
    """Query the live serving stack's own version string. Never hardcoded.

    Returns 'unknown' (not an exception) if the stack exposes no queryable
    version at the moment of the call — a live run must still complete and
    record an explicit 'unknown' per R2, not abort mid-run over a version
    string. The Gap Protocol's "stop immediately" applies to the historical
    run-index builder inventing a value, not to this graceful degradation.
    """
    try:
        proc = subprocess.run(["lms", "--version"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if proc.returncode != 0:
        return "unknown"
    text = proc.stdout.strip()
    return text or "unknown"


def build_provenance_block(model_info: dict | None, settings_gate: dict | None = None) -> dict:
    """Build the provenance block for a NEW run from live LM Studio metadata.

    `model_info` is one entry from LM Studio's native `/api/v1/models`
    response (or `lms ls --json`-shaped) — publisher/key/quantization/
    loaded_instances. `settings_gate` is run_benchmark.py's own
    validate_model_settings() output, preferred for actual_context_length
    since it reflects the context the run actually validated against.
    """
    model_info = model_info or {}

    publisher = model_info.get("publisher")
    key = model_info.get("key") or model_info.get("modelKey")
    if publisher and key:
        model_artifact = f"{publisher}/{key}"
    elif key:
        model_artifact = key
    else:
        model_artifact = "unknown"

    quantization = (model_info.get("quantization") or {}).get("name") or "unknown"

    actual_context_length = None
    if settings_gate and settings_gate.get("actual_context_length") is not None:
        actual_context_length = settings_gate["actual_context_length"]
    else:
        instances = model_info.get("loaded_instances") or []
        if instances:
            actual_context_length = (instances[0].get("config") or {}).get("context_length")
    if actual_context_length is None:
        actual_context_length = "unknown"

    return {
        "runtime_name": "lmstudio",
        "runtime_version": fetch_runtime_version(),
        "model_artifact": model_artifact,
        "quantization": quantization,
        # LM Studio's API does not expose whether speculative decoding was
        # active for a given load — record the gap explicitly rather than
        # guess "false" (R2: an undeterminable field is 'unknown', not a
        # plausible-looking guess).
        "speculative_decoding": "unknown",
        "actual_context_length": actual_context_length,
    }


def historical_provenance_from_run_info(run_info: dict) -> dict:
    """Build the run-index entry for an EXISTING run — never inferred (R4).

    Only `settings_gate.actual_context_length`, when the historical
    run_info.json itself recorded it (the 5/38 "gated" runs), is used —
    everything else is 'unavailable', including runtime_name, which is never
    guessed as "lmstudio" from base_url even though every historical run in
    fact used it (the spec's own Problem section: base_url is an implicit
    signal, not a label).
    """
    context = "unavailable"
    settings_gate = run_info.get("settings_gate")
    if isinstance(settings_gate, dict) and isinstance(settings_gate.get("actual_context_length"), int):
        context = settings_gate["actual_context_length"]

    return {
        "runtime_name": "unavailable",
        "runtime_version": "unavailable",
        "model_artifact": "unavailable",
        "quantization": "unavailable",
        "speculative_decoding": "unavailable",
        "actual_context_length": context,
    }


def validate_provenance(block: dict) -> None:
    """Minimal validator against suites/schemas/run-info.schema.json's shape.

    Raises ProvenanceSchemaError naming every violation found (not just the
    first), so a caller can report a complete diagnosis.
    """
    errors = []
    if not isinstance(block, dict):
        raise ProvenanceSchemaError(f"provenance block must be an object, got {type(block).__name__}")

    for field in REQUIRED_FIELDS:
        if field not in block:
            errors.append(f"missing required field {field!r}")

    for field in ("runtime_name", "runtime_version", "model_artifact", "quantization", "speculative_decoding"):
        if field in block and not (isinstance(block[field], str) and block[field]):
            errors.append(f"{field!r} must be a non-empty string, got {block.get(field)!r}")

    if "actual_context_length" in block:
        value = block["actual_context_length"]
        valid = (isinstance(value, int) and not isinstance(value, bool) and value >= 1) or (
            isinstance(value, str) and value in ("unknown", "unavailable")
        )
        if not valid:
            errors.append(
                f"'actual_context_length' must be a positive integer or 'unknown'/'unavailable', got {value!r}"
            )

    extra = set(block.keys()) - set(REQUIRED_FIELDS)
    if extra:
        errors.append(f"unexpected field(s): {sorted(extra)}")

    if errors:
        raise ProvenanceSchemaError("; ".join(errors))


def build_run_index(results_dir: Path = RESULTS_DIR, registry: dict | None = None) -> dict:
    """R4/R5: classify every results/ directory as attributed/unattributable
    (matching SPEC-003-018's resolver) and record its recoverable provenance.
    Read-only against results/.
    """
    import model_identity  # local import — keeps this module runnable standalone

    if registry is None:
        registry = model_identity.load_registry()
    attribution = model_identity.attribute_all(registry, results_dir=results_dir)
    attributed_dirs = {name for names in attribution["attributed"].values() for name in names}

    index: dict[str, dict] = {}
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        classification = "attributed" if entry.name in attributed_dirs else "unattributable"
        run_info_path = entry / "run_info.json"
        run_info: dict = {}
        if run_info_path.is_file():
            try:
                run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                run_info = {}

        existing = run_info.get("provenance")
        provenance = existing if isinstance(existing, dict) else historical_provenance_from_run_info(run_info)

        index[entry.name] = {"classification": classification, "provenance": provenance}

    return index


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run-provenance tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build-index", help="Build reports/run-index.json from results/")
    p_validate = sub.add_parser("validate", help="Validate a run_info.json's provenance block")
    p_validate.add_argument("path", type=Path)

    args = parser.parse_args()

    if args.command == "build-index":
        index = build_run_index()
        RUN_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        RUN_INDEX_PATH.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {RUN_INDEX_PATH} ({len(index)} directories)")
    elif args.command == "validate":
        run_info = json.loads(args.path.read_text(encoding="utf-8"))
        try:
            validate_provenance(run_info.get("provenance"))
        except ProvenanceSchemaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print("ok")

    return 0


if __name__ == "__main__":
    sys.exit(main())
