#!/usr/bin/env python3
"""CLI for the benchmark-owned agentic coding suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agentic_suite import (
    AGENTIC_SUITE_PATH,
    DEFAULT_CONTEXT,
    MAX_FREE_MEMORY_MB,
    MAX_TIMEOUT_SECONDS,
    REGISTRY_PATH,
    ROOT,
    CommandExecutor,
    build_manifest,
    load_json,
    preflight,
    resolve_credential,
    run_models,
    select_cases,
    validate_agentic_suite,
    validate_registry,
)


def bounded_positive(name: str, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"{name} must be an integer") from exc
        if not 1 <= parsed <= maximum:
            raise argparse.ArgumentTypeError(f"{name} must be in range 1..{maximum}")
        return parsed
    return parse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", action="append", nargs="+", required=True,
        help="LM Studio model key(s); group values or repeat for comparison arms",
    )
    parser.add_argument("--evals", nargs="*", help="Optional EVAL IDs; default is all five")
    parser.add_argument("--ctx", type=int, default=DEFAULT_CONTEXT, help="Exact context for every arm")
    parser.add_argument(
        "--timeout", type=bounded_positive("timeout", MAX_TIMEOUT_SECONDS),
        help=f"Per-case Hermes timeout in seconds (1..{MAX_TIMEOUT_SECONDS})",
    )
    parser.add_argument(
        "--min-free-mem", type=bounded_positive("min-free-mem", MAX_FREE_MEMORY_MB),
        help=f"Minimum free/reclaimable memory in MB (1..{MAX_FREE_MEMORY_MB})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Deprecated compatibility flag; native runs are always new and never overwrite evidence",
    )
    parser.add_argument("--api-root", default=os.getenv("LM_STUDIO_API_ROOT", "http://127.0.0.1:1234"))
    parser.add_argument("--dry-run", action="store_true", help="Print offline policy/manifest; no runtime probe, model invocation, or output write")
    args = parser.parse_args()

    registry = load_json(REGISTRY_PATH)
    suite = load_json(AGENTIC_SUITE_PATH)
    models = [model for group in args.model for model in group]
    if args.timeout is not None:
        suite["policy"]["timeout_seconds"] = args.timeout
    if args.min_free_mem is not None:
        suite["policy"]["minimum_free_memory_mb"] = args.min_free_mem
    suite["dependencies"]["lm_studio_api"] = args.api_root
    validate_registry(registry)
    validate_agentic_suite(suite)
    dependencies = {"preflight": "not-run (offline dry-run)", "hermes": "unknown (not probed)"} if args.dry_run else preflight(suite)
    cases = select_cases(suite, args.evals)
    if args.force:
        print(
            "INFO: --force is compatibility-only and deprecated; native timestamped runs "
            "always create new directories and never overwrite or skip existing evidence."
        )
    print(json.dumps({"dependencies": dependencies, "policy": suite["policy"]}, indent=2))

    if args.dry_run:
        output = ROOT / "results" / "DRY-RUN-NOT-WRITTEN"
        manifest = build_manifest(
            models, suite, dependencies, args.ctx, [case["id"] for case in cases],
            output, {model: None for model in models}, dry_run=True,
            force_requested=args.force,
        )
        print(json.dumps(manifest, indent=2))
        return

    credential = resolve_credential(dict(os.environ))
    if credential is None:
        parser.error("LM Studio credential disappeared after preflight")
    token, _credential_source = credential
    output = run_models(
        models, suite, dependencies, args.ctx, args.evals, args.api_root, token,
        CommandExecutor(), force_requested=args.force,
    )
    print(f"Agentic benchmark complete: {output}")


if __name__ == "__main__":
    main()
