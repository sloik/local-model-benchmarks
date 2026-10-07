"""Tests for scripts/run_capability_suite.py (SPEC-003-016 R1-R3, R9).

Everything here uses a fake subject_fn -- no live model call, no LM Studio
connection (matches this spec's own Notes: "Unit tests must use fake
responses and frozen transcripts").
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_BENCH_DIR = _HERE.parent
_SCRIPTS_DIR = _BENCH_DIR / "scripts"
_CAPABILITY_DIR = _BENCH_DIR / "suites" / "capability"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


contract = _load("capability_contract", _CAPABILITY_DIR / "contract.py")
runner = _load("run_capability_suite", _SCRIPTS_DIR / "run_capability_suite.py")

CORPUS = contract.load_corpus()
IDENTITY = {"model_key": "fake-model", "runtime": "fake-runtime", "context_length": 131072}


def _snapshot_case():
    case = dict(CORPUS["cases"][0])
    case["execution_profile"] = "structured_snapshot"
    return case


def _incremental_case():
    case = next(c for c in CORPUS["cases"] if "checkpoints" in c)
    case = dict(case)
    case["execution_profile"] = "incremental_event"
    return case


def _tool_loop_case():
    case = dict(CORPUS["cases"][0])
    case["execution_profile"] = "tool_loop"
    case["tools"] = [{"name": "get_status", "frozen_result": {"state": "nominal"}}]
    return case


def _fake_subject(structured_output=None, tool_calls=None, error=None, identity=None):
    """Returns a subject_fn closure that ignores its args and always returns
    the same fixed fake response -- a frozen transcript, not a live call."""
    def subject_fn(messages, tools):
        return {
            "structured_output": structured_output,
            "tool_calls": tool_calls or [],
            "error": error,
            "identity": identity or IDENTITY,
        }
    return subject_fn


def _gold_final(case):
    return contract.build_gold_response(case)


# ── R1/AC1: three execution profiles ─────────────────────────────────────────

def test_structured_snapshot_profile_runs_one_bounded_request():
    case = _snapshot_case()
    subject_fn = _fake_subject(structured_output=_gold_final(case))
    result = runner.run_case(case, subject_fn, IDENTITY, supported_profiles={"structured_snapshot"})
    assert result["profile"] == "structured_snapshot"
    assert result["coverage"] is True
    assert result["final_response"] is not None
    assert result["tool_calls_made"] == []


def test_incremental_event_profile_preserves_checkpoint_order():
    case = _incremental_case()
    call_log = []

    def subject_fn(messages, tools):
        call_log.append(messages[0]["content"])
        return {"structured_output": _gold_final(case), "tool_calls": [], "error": None, "identity": IDENTITY}

    result = runner.run_case(case, subject_fn, IDENTITY, supported_profiles={"incremental_event"})
    assert result["profile"] == "incremental_event"
    assert result["coverage"] is True
    # one subject_fn call per checkpoint, in declared order
    expected_texts = [cp["text"] for cp in case["checkpoints"]]
    assert call_log == expected_texts


def test_incremental_event_with_no_checkpoints_declared_errors_cleanly():
    case = _snapshot_case()  # has no "checkpoints" field
    case["execution_profile"] = "incremental_event"
    subject_fn = _fake_subject()
    result = runner.run_case(case, subject_fn, IDENTITY, supported_profiles={"incremental_event"})
    assert result["error"] == "no_checkpoints_declared"


def test_tool_loop_profile_executes_an_eligible_allowlisted_tool():
    case = _tool_loop_case()
    call_count = {"n": 0}

    def subject_fn(messages, tools):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return {"structured_output": None,
                    "tool_calls": [{"name": "get_status", "args": {}}],
                    "error": None, "identity": IDENTITY}
        return {"structured_output": _gold_final(case), "tool_calls": [], "error": None, "identity": IDENTITY}

    result = runner.run_case(case, subject_fn, IDENTITY, supported_profiles={"tool_loop"})
    assert result["profile"] == "tool_loop"
    assert result["coverage"] is True
    assert len(result["tool_calls_made"]) == 1
    assert result["tool_calls_made"][0]["name"] == "get_status"
    assert result["tool_calls_made"][0]["result"]["result"] == {"state": "nominal"}


def test_unsupported_profile_returns_runtime_unsupported_with_no_verdict_path():
    case = _tool_loop_case()
    subject_fn = _fake_subject(structured_output=_gold_final(case))
    result = runner.run_case(case, subject_fn, IDENTITY, supported_profiles={"structured_snapshot"})
    assert result["error"] == "runtime_unsupported"
    assert result["coverage"] is False
    assert result["final_response"] is None


# ── R2/AC2: fail-closed runtime identity ─────────────────────────────────────

def test_correct_identity_passes():
    runner.assert_identity(
        {"model_key": "m1", "runtime": "lmstudio"},
        {"model_key": "m1", "runtime": "lmstudio", "context_length": 131072},
    )


def test_identity_alias_collision_fails_closed():
    with pytest.raises(runner.RunnerError):
        runner.assert_identity(
            {"model_key": "requested-model", "runtime": "lmstudio"},
            {"model_key": "different-alias-loaded-instead", "runtime": "lmstudio", "context_length": 131072},
        )


def test_missing_response_identity_fails_closed():
    with pytest.raises(runner.RunnerError):
        runner.assert_identity(
            {"model_key": "m1", "runtime": "lmstudio"},
            {"model_key": "m1", "runtime": "lmstudio"},  # no context_length
        )


def test_context_drift_below_minimum_fails_closed():
    with pytest.raises(runner.RunnerError):
        runner.assert_identity(
            {"model_key": "m1", "runtime": "lmstudio", "min_context": 131072},
            {"model_key": "m1", "runtime": "lmstudio", "context_length": 32768},
        )


def test_mid_run_instance_replacement_invalidates_the_case():
    case = _snapshot_case()

    def subject_fn(messages, tools):
        return {"structured_output": _gold_final(case), "tool_calls": [], "error": None,
                "identity": {"model_key": "a-completely-different-model", "runtime": "fake-runtime",
                              "context_length": 131072}}

    with pytest.raises(runner.RunnerError):
        runner.run_case(case, subject_fn, IDENTITY, supported_profiles={"structured_snapshot"})


# ── R3: bounded state/tool loop, read-only allowlist ─────────────────────────

def test_undeclared_tool_is_refused_not_executed():
    case = _tool_loop_case()
    result = runner.execute_tool_call(case, "delete_all_data", {"target": "everything"})
    assert result["error"] == "tool_not_allowlisted"
    assert result["contained"] is True


def test_allowlisted_tool_returns_only_its_frozen_result():
    case = _tool_loop_case()
    result = runner.execute_tool_call(case, "get_status", {})
    assert result["result"] == {"state": "nominal"}


def test_tool_loop_turn_limit_exhaustion_is_a_controlled_error():
    case = _tool_loop_case()

    def subject_fn(messages, tools):
        # always asks for another tool call -- never emits a final answer
        return {"structured_output": None, "tool_calls": [{"name": "get_status", "args": {}}],
                "error": None, "identity": IDENTITY}

    result = runner.run_case_tool_loop(case, subject_fn, IDENTITY, max_turns=3)
    assert result["error"] == "turn_limit_exhausted"
    assert len(result["tool_calls_made"]) == 3


def test_frozen_case_bytes_are_never_mutated_by_a_tool_attempt():
    import json
    case = _tool_loop_case()
    before = json.dumps(case, sort_keys=True)
    runner.execute_tool_call(case, "path/traversal/../../etc/passwd", {})
    runner.execute_tool_call(case, "get_status", {})
    after = json.dumps(case, sort_keys=True)
    assert before == after


# ── R9/AC9: safe dry-run preflight ────────────────────────────────────────────

def test_dry_run_never_loads_a_model_or_writes_a_result():
    plan = runner.dry_run_preflight()
    assert plan["model_loaded"] is False
    assert plan["result_written"] is False
    assert plan["case_count"] >= 120


def test_dry_run_records_corpus_and_scorer_hashes():
    plan = runner.dry_run_preflight()
    assert len(plan["corpus_sha256"]) == 64
    assert len(plan["scorer_sha256"]) == 64


def test_dry_run_is_read_only_and_leaves_results_directory_untouched():
    import subprocess
    before = subprocess.run(["git", "status", "--porcelain", "results/"],
                             cwd=_BENCH_DIR, capture_output=True, text=True).stdout
    runner.dry_run_preflight()
    after = subprocess.run(["git", "status", "--porcelain", "results/"],
                            cwd=_BENCH_DIR, capture_output=True, text=True).stdout
    assert before == after


# ── R6: run-digest hashing ────────────────────────────────────────────────────

def _digest_kwargs(**overrides):
    base = dict(model_key="m1", adapter=None, prompt_version="v1",
                sampling={"temperature": 0.1}, parser_profile="native",
                corpus_sha256="a" * 64, scorer_sha256="b" * 64)
    base.update(overrides)
    return base


def test_run_digest_is_deterministic():
    d1 = runner.compute_run_digest(**_digest_kwargs())
    d2 = runner.compute_run_digest(**_digest_kwargs())
    assert d1 == d2


@pytest.mark.parametrize("field,value", [
    ("model_key", "m2"), ("adapter", "some-adapter"), ("prompt_version", "v2"),
    ("sampling", {"temperature": 0.5}), ("parser_profile", "other"),
    ("corpus_sha256", "c" * 64), ("scorer_sha256", "d" * 64),
])
def test_run_digest_changes_when_any_tuple_component_changes(field, value):
    baseline = runner.compute_run_digest(**_digest_kwargs())
    changed = runner.compute_run_digest(**_digest_kwargs(**{field: value}))
    assert baseline != changed, f"changing {field} did not change the digest"


# ── R9: ownership/cleanup preflight fixtures ──────────────────────────────────

def _inventory(**overrides):
    base = {
        "busy": False, "active_eval_owner": None,
        "loaded_instances": [{"model_key": "target-model", "run_owner": "self", "parser_profile": "native"}],
        "corpus_scorer_hash_confirmed": True,
    }
    base.update(overrides)
    return base


def test_ownership_check_eligible_on_clean_owned_inventory():
    result = runner.preflight_ownership_check(_inventory(), "target-model")
    assert result["eligible"] is True
    assert result["cleanup_targets"] == ["target-model"]


def test_ownership_check_rejects_busy_runtime():
    result = runner.preflight_ownership_check(_inventory(busy=True), "target-model")
    assert result["eligible"] is False
    assert result["reason"] == "busy_runtime"


def test_ownership_check_rejects_active_eval_owned_by_other():
    result = runner.preflight_ownership_check(_inventory(active_eval_owner="other-session"), "target-model")
    assert result["eligible"] is False
    assert result["reason"] == "active_eval_owned_by_other"


def test_ownership_check_rejects_unowned_instance_present():
    # "unowned" here means owned by a DIFFERENT run/session -- distinct from a
    # pre-existing instance (run_owner=None), which AC9 explicitly allows to
    # coexist untouched. A foreign run_owner is the genuinely risky case.
    inv = _inventory(loaded_instances=[
        {"model_key": "target-model", "run_owner": "self", "parser_profile": "native"},
        {"model_key": "someone-elses-model", "run_owner": "a-different-run-id", "parser_profile": "native"},
    ])
    result = runner.preflight_ownership_check(inv, "target-model")
    assert result["eligible"] is False
    assert result["reason"] == "unowned_instance_present"


def test_ownership_check_allows_a_pre_existing_untouched_instance():
    # AC9: "a pre-existing model" is its own named, ALLOWED fixture --
    # run_owner=None (never loaded by any eval run) must not block eligibility
    # and must never appear in cleanup_targets.
    inv = _inventory(loaded_instances=[
        {"model_key": "target-model", "run_owner": "self", "parser_profile": "native"},
        {"model_key": "pre-existing-untouched-model", "run_owner": None, "parser_profile": "native"},
    ])
    result = runner.preflight_ownership_check(inv, "target-model")
    assert result["eligible"] is True
    assert "pre-existing-untouched-model" not in result["cleanup_targets"]


def test_ownership_check_rejects_incompatible_parser():
    result = runner.preflight_ownership_check(_inventory(), "target-model", required_parser_profile="special-parser")
    assert result["eligible"] is False
    assert result["reason"] == "incompatible_parser"


def test_ownership_check_rejects_missing_hash_confirmation():
    result = runner.preflight_ownership_check(_inventory(corpus_scorer_hash_confirmed=False), "target-model")
    assert result["eligible"] is False
    assert result["reason"] == "missing_hash_confirmation"


def test_ownership_check_rejects_requested_model_not_loaded():
    result = runner.preflight_ownership_check(_inventory(), "a-different-model")
    assert result["eligible"] is False
    assert result["reason"] == "requested_model_not_loaded"


def test_ownership_check_never_targets_a_pre_existing_instance_for_cleanup():
    inv = _inventory(loaded_instances=[
        {"model_key": "target-model", "run_owner": "self", "parser_profile": "native"},
        {"model_key": "pre-existing-model", "run_owner": "pre-existing", "parser_profile": "native"},
    ])
    # pre-existing (non-None, non-self owner) is treated as "unowned by this run"
    # and must block eligibility rather than being silently swept into cleanup.
    result = runner.preflight_ownership_check(inv, "target-model")
    assert result["eligible"] is False
    assert "pre-existing-model" not in result["cleanup_targets"]


def test_cleanup_after_run_blocks_on_uncertain_cleanup():
    result = runner.cleanup_after_run(_inventory(), ["target-model"], cleanup_succeeded=False)
    assert result["safe_to_continue"] is False
    assert result["reason"] == "cleanup_failure_or_uncertain"


def test_cleanup_after_run_detects_ownership_loss():
    inv = _inventory()  # target-model still shows up as loaded after "cleanup"
    result = runner.cleanup_after_run(inv, ["target-model"], cleanup_succeeded=True)
    assert result["safe_to_continue"] is False
    assert result["reason"] == "ownership_loss_detected"


def test_cleanup_after_run_succeeds_when_target_actually_gone():
    inv = _inventory(loaded_instances=[])  # cleanup actually removed it
    result = runner.cleanup_after_run(inv, ["target-model"], cleanup_succeeded=True)
    assert result["safe_to_continue"] is True
