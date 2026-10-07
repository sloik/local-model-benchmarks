#!/usr/bin/env python3
"""
model_identity.py — Canonical model-identity resolver (SPEC-003-018).

Resolving a model to its benchmark runs by directory-name substring silently
returns the wrong model's score (e.g. installed `qwen3.8-27b` is a substring
of the deleted, rejected `qwen3.8-27b-mlx` run directory). This module fixes
identity by an explicit alias table, never by substring/prefix/fuzzy match,
and attributes result directories only from their own `run_info.json`
`model_id` field — never from the directory name.

Usage (CLI):
    python3 scripts/model_identity.py resolve qwen3.8-27b
    python3 scripts/model_identity.py runs-for "qwen/qwen3.6-27b"
    python3 scripts/model_identity.py attribute-all
    python3 scripts/model_identity.py build-registry [--write]

Requirements: no external packages (stdlib only). `build-registry`'s live
refresh of installed models needs the `lms` CLI on PATH; every other command
works from the committed models/registry.json alone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
BENCH_DIR = SCRIPT_DIR.parent
REGISTRY_PATH = BENCH_DIR / "models" / "registry.json"
RESULTS_DIR = BENCH_DIR / "results"

LIFECYCLE_STATES = frozenset({
    "installed",
    "deleted-retained-evidence",
    "deleted-rejected",
})


class ModelIdentityError(Exception):
    """Base class for model-identity resolution failures."""


class UnknownModelKeyError(ModelIdentityError):
    def __init__(self, key: str):
        super().__init__(f"unknown model key {key!r} — not in registry as a canonical key or alias")
        self.key = key


class AmbiguousModelKeyError(ModelIdentityError):
    def __init__(self, key: str, candidates: list[str]):
        super().__init__(
            f"ambiguous model key {key!r} matches multiple canonical entries: "
            + ", ".join(sorted(candidates))
        )
        self.key = key
        self.candidates = sorted(candidates)


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    """Load the registry, dropping the non-model `_meta` entry."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if k != "_meta"}


def resolve(key: str, registry: dict | None = None) -> str:
    """Return the canonical key for `key` (a canonical key or a historical alias).

    Never matches by substring, prefix, or fuzzy comparison — only an exact
    match against a canonical key or a declared alias. Raises
    UnknownModelKeyError if no entry matches, or AmbiguousModelKeyError if
    more than one canonical entry claims the same key (a registry-authoring
    defect: no two entries should ever share an alias).
    """
    if registry is None:
        registry = load_registry()

    matches = [
        canonical
        for canonical, entry in registry.items()
        if key == canonical or key in entry.get("aliases", [])
    ]
    if not matches:
        raise UnknownModelKeyError(key)
    if len(matches) > 1:
        raise AmbiguousModelKeyError(key, matches)
    return matches[0]


def runs_for(key: str, registry: dict | None = None, results_dir: Path = RESULTS_DIR) -> list[str]:
    """Return every result-directory name whose run_info.json model_id resolves to `key`'s canonical entry."""
    if registry is None:
        registry = load_registry()
    canonical = resolve(key, registry)

    matches = []
    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        run_info = entry / "run_info.json"
        if not run_info.is_file():
            continue
        try:
            data = json.loads(run_info.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        model_id = data.get("model_id")
        if not model_id:
            continue
        try:
            if resolve(model_id, registry) == canonical:
                matches.append(entry.name)
        except ModelIdentityError:
            continue
    return matches


def attribute_all(registry: dict | None = None, results_dir: Path = RESULTS_DIR) -> dict:
    """Attribute every results/ directory to its canonical model, or the unattributable bucket.

    A directory is `unattributable` only when it has no parsable run_info.json
    or no model_id field — never guessed from its name (R3). Per the Gap
    Protocol, a model_id that IS present but maps to no known registry entry
    raises UnknownModelKeyError rather than being silently bucketed — that is
    a registry gap to fix, not a directory to discard.
    """
    if registry is None:
        registry = load_registry()

    attributed: dict[str, list[str]] = {}
    unattributable: list[str] = []

    for entry in sorted(results_dir.iterdir()):
        if not entry.is_dir():
            continue
        run_info = entry / "run_info.json"
        if not run_info.is_file():
            unattributable.append(entry.name)
            continue
        try:
            data = json.loads(run_info.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            unattributable.append(entry.name)
            continue
        model_id = data.get("model_id")
        if not model_id:
            unattributable.append(entry.name)
            continue
        canonical = resolve(model_id, registry)  # raises on an unmapped model_id — Gap Protocol
        attributed.setdefault(canonical, []).append(entry.name)

    return {"attributed": attributed, "unattributable": sorted(unattributable)}


def fetch_installed_models() -> list[dict]:
    """Return the live `lms ls --json` inventory. Never a hardcoded list (recorded benchmark reliability finding)."""
    proc = subprocess.run(["lms", "ls", "--json"], capture_output=True, text=True, timeout=30)
    proc.check_returncode()
    return json.loads(proc.stdout)


def build_registry(existing: dict | None = None) -> dict:
    """Refresh installed entries from live `lms ls --json`; preserve deleted/curated entries untouched.

    Never renames or removes a deleted entry, and never merges an installed
    model's aliases into a different canonical key — this only updates the
    `modelKey`/`artifact`/`quantization`/`lifecycle_state` fields of entries
    that are currently installed, and adds newly-installed models with a
    single self-alias.
    """
    registry = dict(existing) if existing else {}
    installed = fetch_installed_models()
    installed_keys = set()

    for model in installed:
        if model.get("type") != "llm":
            continue
        model_key = model.get("modelKey")
        if not model_key:
            continue
        installed_keys.add(model_key)
        entry = registry.get(model_key, {})
        entry["modelKey"] = model_key
        entry["artifact"] = model.get("path", model.get("indexedModelIdentifier", ""))
        entry["quantization"] = (model.get("quantization") or {}).get("name")
        entry["lifecycle_state"] = "installed"
        entry.setdefault("aliases", [])
        if model_key not in entry["aliases"]:
            entry["aliases"].append(model_key)
        registry[model_key] = entry

    # A previously-installed entry that's no longer in `lms ls` output is left
    # exactly as-is (its lifecycle_state is a human/curated call, R6-adjacent —
    # this resolver never rewrites deleted-model history on its own).
    return registry


def main() -> int:
    parser = argparse.ArgumentParser(description="Canonical model-identity resolver")
    sub = parser.add_subparsers(dest="command", required=True)

    p_resolve = sub.add_parser("resolve", help="Resolve a key/alias to its canonical model key")
    p_resolve.add_argument("key")

    p_runs = sub.add_parser("runs-for", help="List result directories for a model key/alias")
    p_runs.add_argument("key")

    sub.add_parser("attribute-all", help="Attribute every results/ directory")

    p_build = sub.add_parser("build-registry", help="Refresh installed entries from `lms ls --json`")
    p_build.add_argument("--write", action="store_true", help="Write the refreshed registry back to disk")

    args = parser.parse_args()

    try:
        if args.command == "resolve":
            print(resolve(args.key))
        elif args.command == "runs-for":
            for name in runs_for(args.key):
                print(name)
        elif args.command == "attribute-all":
            result = attribute_all()
            print(json.dumps(
                {
                    "attributed_count": sum(len(v) for v in result["attributed"].values()),
                    "unattributable_count": len(result["unattributable"]),
                    "attributed": result["attributed"],
                    "unattributable": result["unattributable"],
                },
                indent=2,
            ))
        elif args.command == "build-registry":
            existing = load_registry() if REGISTRY_PATH.is_file() else {}
            refreshed = build_registry(existing)
            if args.write:
                out = {"_meta": json.loads(REGISTRY_PATH.read_text())["_meta"]} if REGISTRY_PATH.is_file() else {}
                out.update(refreshed)
                REGISTRY_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
                print(f"wrote {REGISTRY_PATH}")
            else:
                print(json.dumps(refreshed, indent=2))
    except ModelIdentityError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
