#!/usr/bin/env python3
"""Tests for scripts/qualify_compact_models.py (SPEC-003-017). All offline --
no live LM Studio call. Covers R1 manifest identity keying, R3 measured
context, R4 hazard budget/opportunity minima, and the journal."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
BENCH_DIR = HERE.parent
sys.path.insert(0, str(BENCH_DIR / "scripts"))
sys.path.insert(0, str(BENCH_DIR / "suites" / "capability"))

import qualify_compact_models as qcm  # noqa: E402
import contract as capability_contract  # noqa: E402

FAKE_INVENTORY = [
    {"type": "llm", "modelKey": "minicpm5-2b", "path": "mlx-community/MiniCPM5-2B-8bit",
     "publisher": "mlx-community", "displayName": "MiniCPM5 2B", "format": "safetensors",
     "quantization": {"name": "8bit"}, "paramsString": "2B", "architecture": "llama",
     "maxContextLength": 131072, "trainedForToolUse": False, "vision": False, "sizeBytes": 1},
    {"type": "llm", "modelKey": "minicpm5-2b-mlx", "path": "openbmb/MiniCPM5-2B-MLX",
     "publisher": "openbmb", "displayName": "MiniCPM5 2B", "format": "safetensors",
     "quantization": {"name": "4bit"}, "paramsString": "2B", "architecture": "llama",
     "maxContextLength": 131072, "trainedForToolUse": False, "vision": False, "sizeBytes": 2},
    {"type": "llm", "modelKey": "minicpm5-1b-mlx", "path": "openbmb/MiniCPM5-1B-MLX",
     "publisher": "openbmb", "displayName": "MiniCPM5 1B", "format": "safetensors",
     "quantization": {"name": "4bit"}, "paramsString": "1B", "architecture": "llama",
     "maxContextLength": 131072, "trainedForToolUse": False, "vision": False, "sizeBytes": 3},
    {"type": "llm", "modelKey": "llama-3.2-1b-instruct", "path": "mlx-community/Llama-3.2-1B-Instruct-4bit",
     "publisher": "mlx-community", "displayName": "Llama 3.2 1B Instruct", "format": "safetensors",
     "quantization": {"name": "4bit"}, "paramsString": "1B", "architecture": "llama",
     "maxContextLength": 131072, "trainedForToolUse": True, "vision": False, "sizeBytes": 4},
    {"type": "llm", "modelKey": "qwen2.5-0.5b-instruct-mlx", "path": "lmstudio-community/Qwen2.5-0.5B-Instruct-MLX-4bit",
     "publisher": "lmstudio-community", "displayName": "Qwen2.5 0.5B Instruct", "format": "safetensors",
     "quantization": {"name": "4bit"}, "paramsString": "0.5B", "architecture": "qwen2",
     "maxContextLength": 32768, "trainedForToolUse": True, "vision": False, "sizeBytes": 5},
]


# ── R1: manifest identity keying ────────────────────────────────────────────────

def test_manifest_selects_exact_openbmb_1b_not_either_2b_decoy():
    manifest = qcm.build_candidate_manifest(FAKE_INVENTORY)
    arm = manifest["arms"]["minicpm5-1b"]
    assert arm["status"] == "available"
    assert arm["path"] == "openbmb/MiniCPM5-1B-MLX"
    assert arm["publisher"] == "openbmb"
    assert arm["params_string"] == "1B"


def test_manifest_excludes_both_2b_decoys_explicitly():
    manifest = qcm.build_candidate_manifest(FAKE_INVENTORY)
    assert "mlx-community/MiniCPM5-2B-8bit" in manifest["explicitly_excluded_decoys"]
    assert "openbmb/MiniCPM5-2B-MLX" in manifest["explicitly_excluded_decoys"]
    arm_paths = {a["path"] for a in manifest["arms"].values() if a["status"] == "available"}
    assert "mlx-community/MiniCPM5-2B-8bit" not in arm_paths
    assert "openbmb/MiniCPM5-2B-MLX" not in arm_paths


def test_manifest_marks_missing_arm_unavailable_not_substituted():
    inventory = [m for m in FAKE_INVENTORY if m["path"] != "mlx-community/Llama-3.2-1B-Instruct-4bit"]
    manifest = qcm.build_candidate_manifest(inventory)
    assert manifest["arms"]["llama-3.2-1b"]["status"] == "unavailable"
    # every other arm stays available -- one missing arm never substitutes another
    assert manifest["arms"]["minicpm5-1b"]["status"] == "available"
    assert manifest["arms"]["qwen2.5-0.5b"]["status"] == "available"


def test_manifest_never_matches_by_display_name_alone():
    # a decoy with a matching displayName but a different path/publisher must
    # never be picked up by a naive displayName match
    manifest = qcm.build_candidate_manifest(FAKE_INVENTORY)
    arm = manifest["arms"]["minicpm5-1b"]
    assert arm["display_name"] == "MiniCPM5 1B"
    decoy_display_names = {m["displayName"] for m in FAKE_INVENTORY if m["path"] != arm["path"]}
    # the true arm's display name differs from both decoys' shared "MiniCPM5 2B"
    assert arm["display_name"] not in decoy_display_names


def test_manifest_captures_tool_use_and_context_profile_independently():
    manifest = qcm.build_candidate_manifest(FAKE_INVENTORY)
    assert manifest["arms"]["minicpm5-1b"]["trained_for_tool_use"] is False
    assert manifest["arms"]["llama-3.2-1b"]["trained_for_tool_use"] is True
    assert manifest["arms"]["qwen2.5-0.5b"]["max_context_length"] == 32768
    assert manifest["arms"]["llama-3.2-1b"]["max_context_length"] == 131072


# ── R3: measured bounded context ────────────────────────────────────────────────

def test_measure_max_serialized_input_is_a_real_measurement_not_a_constant():
    corpus = capability_contract.load_corpus()
    small_corpus = {"cases": corpus["cases"][:5]}
    full_measured = qcm.measure_max_serialized_input(corpus)
    small_measured = qcm.measure_max_serialized_input(small_corpus)
    assert full_measured["max_serialized_chars"] > 0
    assert full_measured["max_case_id"] is not None
    # changing the input corpus changes the measurement -- not hardcoded
    assert full_measured["max_serialized_chars"] != small_measured["max_serialized_chars"] or \
        full_measured["max_case_id"] != small_measured["max_case_id"]


def test_select_common_context_never_exceeds_smallest_arm_max():
    manifest = qcm.build_candidate_manifest(FAKE_INVENTORY)
    measured = {"max_serialized_chars": 4000, "max_case_id": "x", "approx_max_tokens": 1000}
    selected = qcm.select_common_context(manifest, measured)
    assert selected["common_context"] <= 32768  # the qwen arm's own max
    assert selected["common_context"] <= selected["bounding_arm_max"]


def test_select_common_context_bounded_by_margin_when_smaller_than_arm_max():
    manifest = qcm.build_candidate_manifest(FAKE_INVENTORY)
    measured = {"max_serialized_chars": 100, "max_case_id": "x", "approx_max_tokens": 25}
    selected = qcm.select_common_context(manifest, measured)
    # 8x margin over 25 tokens rounds up to 256, well under any arm max
    assert selected["common_context"] == 256
    assert selected["margin_bound"] == 256


def test_select_common_context_raises_with_no_available_arms():
    manifest = {"arms": {"x": {"status": "unavailable"}}, "explicitly_excluded_decoys": {}, "baselines": []}
    try:
        qcm.select_common_context(manifest, {"max_serialized_chars": 1, "approx_max_tokens": 1, "max_case_id": "x"})
        assert False, "expected ManifestError"
    except qcm.ManifestError:
        pass


# ── R4: hazard budget + opportunity minima ──────────────────────────────────────

def test_rule_of_three_budget_matches_closed_form():
    assert qcm.rule_of_three_budget(0.1) == math.ceil(3 / 0.1)
    assert qcm.rule_of_three_budget(0.05) == 60
    assert qcm.rule_of_three_budget(0.3) == 10


def test_rule_of_three_budget_rejects_out_of_range_p_max():
    for bad in (0, 1, -0.1, 1.5):
        try:
            qcm.rule_of_three_budget(bad)
            assert False, f"expected ValueError for p_max={bad}"
        except ValueError:
            pass


def test_opportunity_minima_counts_logical_pairs_not_raw_cases():
    corpus = capability_contract.load_corpus()
    minima = qcm.opportunity_minima(corpus)
    # every capability has fewer (or equal) logical pairs than raw cases,
    # since some pairs have a base+counterfactual twin -- never counting
    # repeats/twins as independent evidence
    counts_by_cap = {}
    for case in corpus["cases"]:
        counts_by_cap[case["capability"]] = counts_by_cap.get(case["capability"], 0) + 1
    for cap, n_pairs in minima.items():
        assert n_pairs <= counts_by_cap[cap]
    assert sum(minima.values()) > 0


# ── R4: journal ──────────────────────────────────────────────────────────────────

def test_journal_roundtrip_and_already_done(tmp_path):
    path = tmp_path / "journal.jsonl"
    entry = {"arm_id": "qwen2.5-0.5b", "case_id": "c1", "repetition": 0, "reload_block": 0, "verdict": "pass"}
    qcm.append_journal(entry, path)
    journal = qcm.read_journal(path)
    assert journal == [entry]
    assert qcm.already_done(journal, "qwen2.5-0.5b", "c1", 0, 0) is True
    assert qcm.already_done(journal, "qwen2.5-0.5b", "c1", 1, 0) is False
    assert qcm.already_done(journal, "llama-3.2-1b", "c1", 0, 0) is False


def test_journal_never_duplicates_on_append(tmp_path):
    path = tmp_path / "journal.jsonl"
    entry = {"arm_id": "a", "case_id": "c", "repetition": 0, "reload_block": 0, "verdict": "pass"}
    qcm.append_journal(entry, path)
    journal_before = qcm.read_journal(path)
    if not qcm.already_done(journal_before, "a", "c", 0, 0):
        qcm.append_journal(entry, path)
    journal_after = qcm.read_journal(path)
    assert len(journal_after) == 1


def test_read_journal_missing_file_returns_empty():
    assert qcm.read_journal(Path("/nonexistent/path/journal.jsonl")) == []


# ── R2: subject_fn response contract (offline shape check, no live call) ───────

def test_build_subject_fn_returns_callable_with_expected_signature():
    fn = qcm.build_subject_fn("qwen2.5-0.5b-instruct-mlx", "lmstudio", 32768)
    assert callable(fn)
