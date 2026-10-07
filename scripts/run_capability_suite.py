#!/usr/bin/env python3
"""
run_capability_suite.py -- deterministic, provider-neutral runner for the
general capability suite (SPEC-003-016).

Three execution profiles (R1): `structured_snapshot` (one bounded request
over supplied JSON), `incremental_event` (a multi-checkpoint trace delivered
without restating accumulated state), `tool_loop` (allowlisted read-only
tools with frozen results, mirroring scripts/cat07_harness.py's pattern).

Every function here takes a `subject_fn(messages, tools) -> response_dict`
callable rather than calling LM Studio directly -- unit tests inject a fake
subject_fn with controlled responses (per this spec's own Notes: "Unit tests
must use fake responses and frozen transcripts. No live inference is
required to finish the implementation contract"). A real LM Studio-backed
subject_fn is a thin adapter over cat07_harness._call_model's pattern, wired
in only by the Live Execution Checklist items, never by the test suite.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
BENCH_DIR = HERE.parent
CAPABILITY_DIR = BENCH_DIR / "suites" / "capability"

sys.path.insert(0, str(CAPABILITY_DIR))
import contract as capability_contract  # noqa: E402
import taxonomy  # noqa: E402


class RunnerError(Exception):
    pass


# ── R2: fail-closed runtime identity ──────────────────────────────────────────

def assert_identity(requested: dict, observed: dict) -> None:
    """R2/AC2: fail-closed identity check. `requested` is the catalog key /
    expected profile the run was configured for; `observed` is what the
    response/runtime actually reported. Any mismatch raises -- never a
    warning, never a silent substitution."""
    for field in ("model_key", "runtime"):
        if requested.get(field) != observed.get(field):
            raise RunnerError(
                f"identity mismatch on {field}: requested {requested.get(field)!r}, "
                f"observed {observed.get(field)!r}"
            )
    if observed.get("context_length") is None:
        raise RunnerError("observed identity missing context_length -- cannot assert fail-closed")
    if requested.get("min_context") and observed["context_length"] < requested["min_context"]:
        raise RunnerError(
            f"observed context {observed['context_length']} below requested minimum "
            f"{requested['min_context']}"
        )


# ── R3: bounded state/tool loop, read-only allowlist ──────────────────────────

def _allowed_tool_names(case: dict) -> set[str]:
    return {t["name"] for t in case.get("tools", [])}


def execute_tool_call(case: dict, tool_name: str, args: dict) -> dict:
    """Only an exact allowlisted, schema-declared tool name may be called; its
    result is the case's own frozen result, never executed live. Anything else
    (mutation-sounding name, undeclared tool) is refused, not silently ignored."""
    allowed = _allowed_tool_names(case)
    if tool_name not in allowed:
        return {"error": "tool_not_allowlisted", "tool_name": tool_name, "contained": True}
    entry = next(t for t in case["tools"] if t["name"] == tool_name)
    return {"result": entry["frozen_result"], "contained": True}


def run_case_structured_snapshot(case: dict, subject_fn, identity: dict) -> dict:
    packet = capability_contract.serialize_for_subject(case)
    response = subject_fn([{"role": "user", "content": json.dumps(packet)}], tools=[])
    assert_identity({"model_key": identity["model_key"], "runtime": identity["runtime"]},
                     response.get("identity", {}))
    return {"profile": "structured_snapshot", "final_response": response.get("structured_output"),
            "tool_calls_made": [], "error": response.get("error")}


def run_case_incremental_event(case: dict, subject_fn, identity: dict) -> dict:
    checkpoints = case.get("checkpoints", [])
    if not checkpoints:
        return {"profile": "incremental_event", "final_response": None, "tool_calls_made": [],
                "error": "no_checkpoints_declared"}
    last_response = None
    for cp in checkpoints:
        # R3: preserve incremental checkpoint order; never restate accumulated state.
        response = subject_fn([{"role": "user", "content": cp["text"]}], tools=[])
        assert_identity({"model_key": identity["model_key"], "runtime": identity["runtime"]},
                         response.get("identity", {}))
        last_response = response
    return {"profile": "incremental_event", "final_response": last_response.get("structured_output"),
            "tool_calls_made": [], "error": last_response.get("error")}


def run_case_tool_loop(case: dict, subject_fn, identity: dict, max_turns: int = 5) -> dict:
    packet = capability_contract.serialize_for_subject(case)
    messages = [{"role": "user", "content": json.dumps(packet)}]
    tool_calls_made = []
    final_response = None
    error = None

    for _turn in range(max_turns):
        response = subject_fn(messages, tools=case.get("tools", []))
        assert_identity({"model_key": identity["model_key"], "runtime": identity["runtime"]},
                         response.get("identity", {}))
        calls = response.get("tool_calls") or []
        if not calls:
            final_response = response.get("structured_output")
            error = response.get("error")
            break
        for call in calls:
            result = execute_tool_call(case, call["name"], call.get("args", {}))
            tool_calls_made.append({"name": call["name"], "args": call.get("args", {}), "result": result})
        messages.append({"role": "assistant", "tool_calls": calls})
        messages.append({"role": "tool", "content": json.dumps([tc["result"] for tc in tool_calls_made[-len(calls):]])})
    else:
        error = "turn_limit_exhausted"

    return {"profile": "tool_loop", "final_response": final_response,
            "tool_calls_made": tool_calls_made, "error": error}


_PROFILE_RUNNERS = {
    "structured_snapshot": run_case_structured_snapshot,
    "incremental_event": run_case_incremental_event,
    "tool_loop": run_case_tool_loop,
}


def run_case(case: dict, subject_fn, identity: dict, supported_profiles: set[str]) -> dict:
    """R1: dispatch to the case's required profile. An unsupported profile is
    `runtime_unsupported` -- never a silent fallback to a different profile."""
    required_profile = case.get("execution_profile", "structured_snapshot")
    if required_profile not in taxonomy.EXECUTION_PROFILES:
        raise RunnerError(f"case {case['case_id']} declares unknown profile {required_profile!r}")
    if required_profile not in supported_profiles:
        return {"profile": required_profile, "final_response": None, "tool_calls_made": [],
                "error": "runtime_unsupported", "coverage": False}
    result = _PROFILE_RUNNERS[required_profile](case, subject_fn, identity)
    result["coverage"] = True
    return result


# ── R9: safe dry-run preflight ────────────────────────────────────────────────

def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dry_run_preflight(corpus_path: Path = CAPABILITY_DIR / "corpus.json",
                       scorer_path: Path = CAPABILITY_DIR / "scorer.py") -> dict:
    """R9/AC9: validates hashes and structure without loading a model or
    writing a result. Returns a plan; never mutates anything."""
    corpus = capability_contract.load_corpus(corpus_path)
    capability_contract.assert_corpus_meets_minimums(corpus)
    return {
        "corpus_sha256": _sha256_file(corpus_path),
        "scorer_sha256": _sha256_file(scorer_path),
        "case_count": len(corpus["cases"]),
        "model_loaded": False,
        "result_written": False,
        "ownership_plan": "no runtime inventory queried in dry-run-only mode",
    }


# ── R6: pin and hash the full run tuple ───────────────────────────────────────

def compute_run_digest(model_key: str, adapter: str | None, prompt_version: str,
                        sampling: dict, parser_profile: str, corpus_sha256: str,
                        scorer_sha256: str) -> str:
    """R6: one digest over the full declared tuple. Changing ANY component
    changes the digest, which is the mechanism that 'prevents comparison'
    (AC6) between two runs that aren't actually the same configuration."""
    tuple_repr = json.dumps({
        "model_key": model_key, "adapter": adapter, "prompt_version": prompt_version,
        "sampling": sampling, "parser_profile": parser_profile,
        "corpus_sha256": corpus_sha256, "scorer_sha256": scorer_sha256,
    }, sort_keys=True)
    return hashlib.sha256(tuple_repr.encode("utf-8")).hexdigest()


# ── R9: ownership/cleanup preflight decision (pure function over a supplied
#        runtime inventory -- fixture-testable without a live LM Studio call) ──

def preflight_ownership_check(inventory: dict, requested_model_key: str,
                               required_parser_profile: str | None = None) -> dict:
    """`inventory` is the caller-supplied runtime snapshot (from a real `lms ps`
    call, or a test fixture) -- this function makes no network/subprocess call
    itself, so it is fully offline-testable. Returns
    {"eligible": bool, "reason": str, "cleanup_targets": [...]}."""
    if inventory.get("busy"):
        return {"eligible": False, "reason": "busy_runtime", "cleanup_targets": []}
    if inventory.get("active_eval_owner") not in (None, "self"):
        return {"eligible": False, "reason": "active_eval_owned_by_other", "cleanup_targets": []}

    loaded = inventory.get("loaded_instances", [])
    owned_by_this_run = [i for i in loaded if i.get("run_owner") == "self"]
    unowned = [i for i in loaded if i.get("run_owner") not in ("self", None)]
    if unowned:
        return {"eligible": False, "reason": "unowned_instance_present", "cleanup_targets": []}

    target = next((i for i in loaded if i.get("model_key") == requested_model_key), None)
    if target is None:
        return {"eligible": False, "reason": "requested_model_not_loaded", "cleanup_targets": []}

    if required_parser_profile and target.get("parser_profile") != required_parser_profile:
        return {"eligible": False, "reason": "incompatible_parser", "cleanup_targets": []}

    if not inventory.get("corpus_scorer_hash_confirmed", False):
        return {"eligible": False, "reason": "missing_hash_confirmation", "cleanup_targets": []}

    # Only exact run-owned instances are ever cleanup targets -- never a
    # pre-existing instance this run did not itself load.
    cleanup_targets = [i["model_key"] for i in owned_by_this_run]
    return {"eligible": True, "reason": "ok", "cleanup_targets": cleanup_targets}


def cleanup_after_run(inventory: dict, cleanup_targets: list, cleanup_succeeded: bool) -> dict:
    """R9: 'cleanup uncertainty stops subsequent arms.' Never silently
    proceeds to a next arm when cleanup outcome is unknown or failed."""
    if not cleanup_succeeded:
        return {"safe_to_continue": False, "reason": "cleanup_failure_or_uncertain"}
    remaining_owned = [i for i in inventory.get("loaded_instances", [])
                        if i.get("model_key") in cleanup_targets]
    if remaining_owned:
        return {"safe_to_continue": False, "reason": "ownership_loss_detected"}
    return {"safe_to_continue": True, "reason": "ok"}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        print(json.dumps(dry_run_preflight(), indent=2))
    else:
        print("only --dry-run is implemented without explicit live-run authorization (SPEC-003-016 R9)")
        sys.exit(1)
