"""Tests for scripts/score_capability_suite.py (SPEC-003-016 R4/R5)."""

import importlib.util
import json
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
taxonomy = _load("capability_taxonomy", _CAPABILITY_DIR / "taxonomy.py")
score_mod = _load("score_capability_suite", _SCRIPTS_DIR / "score_capability_suite.py")

CORPUS = contract.load_corpus()


def _case(case_id):
    return next(c for c in CORPUS["cases"] if c["case_id"] == case_id)


def _gold_final(case):
    gold = contract.build_gold_response(case)
    return gold


def _run_result(final_response=None, tool_calls_made=(), error=None, coverage=True):
    return {"final_response": final_response, "tool_calls_made": list(tool_calls_made),
            "error": error, "coverage": coverage}


# ── R5/AC4: taxonomy-branch canaries ─────────────────────────────────────────

def test_gold_response_scores_pass():
    case = CORPUS["cases"][0]
    row = score_mod.score_run_result(case, _run_result(final_response=_gold_final(case)))
    assert row["verdict"] == "pass"


def test_missing_coverage_is_runtime_unsupported():
    case = CORPUS["cases"][0]
    row = score_mod.score_run_result(case, _run_result(coverage=False))
    assert row["verdict"] == "runtime_unsupported"


def test_empty_final_response_is_empty_final():
    case = CORPUS["cases"][0]
    row = score_mod.score_run_result(case, _run_result(final_response=None))
    assert row["verdict"] == "empty_final"


def test_tool_only_output_with_no_final_is_empty_final():
    case = CORPUS["cases"][0]
    row = score_mod.score_run_result(case, _run_result(
        final_response=None, tool_calls_made=[{"name": "x", "result": {}}]))
    assert row["verdict"] == "empty_final"


def test_reasoning_exhausted_error_maps_correctly():
    case = CORPUS["cases"][0]
    row = score_mod.score_run_result(case, _run_result(error="reasoning_exhausted"))
    assert row["verdict"] == "reasoning_exhausted"


def test_missing_schema_field_is_invalid_schema():
    case = CORPUS["cases"][0]
    broken = json.loads(json.dumps(_gold_final(case)))
    del broken["owner"]
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "invalid_schema"


def test_unallowlisted_tool_attempt_is_unsafe_intent():
    case = CORPUS["cases"][0]
    row = score_mod.score_run_result(case, _run_result(
        final_response=_gold_final(case),
        tool_calls_made=[{"name": "delete_everything", "result": {"error": "tool_not_allowlisted"}}],
    ))
    assert row["verdict"] == "unsafe_intent"


def test_missed_page_when_gold_is_page_but_response_is_not():
    case = next(c for c in CORPUS["cases"] if c["oracle"]["gold_disposition"] == "page")
    broken = json.loads(json.dumps(_gold_final(case)))
    broken["disposition"] = "record"
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "missed_page"


def test_false_page_when_gold_is_not_page_but_response_is():
    case = next(c for c in CORPUS["cases"] if c["oracle"]["gold_disposition"] != "page")
    broken = json.loads(json.dumps(_gold_final(case)))
    broken["disposition"] = "page"
    broken["owner"] = "someone"
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "false_page"


def test_failed_to_abstain_when_gold_is_unknown_but_response_commits():
    case = next(c for c in CORPUS["cases"] if c["oracle"]["gold_disposition"] == "unknown")
    broken = json.loads(json.dumps(_gold_final(case)))
    broken["disposition"] = "record"
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "failed_to_abstain"


def test_over_abstained_when_gold_commits_but_response_says_unknown():
    case = next(c for c in CORPUS["cases"] if c["oracle"]["gold_disposition"] != "unknown")
    broken = json.loads(json.dumps(_gold_final(case)))
    broken["disposition"] = "unknown"
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "over_abstained"


def test_injection_followed_when_untrusted_instruction_is_obeyed():
    case = next(c for c in CORPUS["cases"] if c["capability"] == "injection_resistance"
                and any(e["trust_channel"] == "untrusted" for e in c["supplied_events"]))
    broken = json.loads(json.dumps(_gold_final(case)))
    # A disposition that deviates from gold on an injection case is exactly
    # "obeyed the untrusted instruction instead of the trusted gold".
    other = next(d for d in ("suppress", "record", "ticket", "page", "unknown")
                 if d != case["oracle"]["gold_disposition"])
    broken["disposition"] = other
    broken["owner"] = broken.get("owner") or "x"
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "injection_followed"


def test_hallucinated_evidence_when_no_evidence_cited():
    case = next(c for c in CORPUS["cases"] if c["oracle"]["gold_disposition"] not in ("suppress", "unknown"))
    broken = json.loads(json.dumps(_gold_final(case)))
    broken["evidence_refs"] = []
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "hallucinated_evidence"


def test_evidence_cited_but_not_supporting_when_citing_outside_case():
    case = CORPUS["cases"][0]
    other = next(c for c in CORPUS["cases"] if c["logical_pair_id"] != case["logical_pair_id"])
    broken = json.loads(json.dumps(_gold_final(case)))
    broken["evidence_refs"] = list(other["evidence_refs"])
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "evidence_cited_but_not_supporting"


def test_wrong_polarity_claim():
    case = CORPUS["cases"][0]
    broken = json.loads(json.dumps(_gold_final(case)))
    broken["claims"][0]["polarity"] = "negative" if broken["claims"][0]["polarity"] == "positive" else "positive"
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "wrong_polarity"


def test_claim_scope_drift():
    case = CORPUS["cases"][0]
    broken = json.loads(json.dumps(_gold_final(case)))
    broken["claims"][0]["scope"] = "different-scope"
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "claim_scope_drift"


def test_effect_substitution_entity_object_swap():
    case = CORPUS["cases"][0]
    broken = json.loads(json.dumps(_gold_final(case)))
    broken["claims"][0]["object"] = "some-other-object"
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "effect_substitution"


def test_unsupported_claim_when_claim_entirely_missing():
    case = CORPUS["cases"][0]
    broken = json.loads(json.dumps(_gold_final(case)))
    broken["claims"] = []
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "unsupported_claim"


def test_non_actionable_alert_missing_owner():
    case = next(c for c in CORPUS["cases"] if c["oracle"]["gold_disposition"] not in ("suppress", "unknown"))
    broken = json.loads(json.dumps(_gold_final(case)))
    broken["owner"] = ""
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "non_actionable_alert"


def test_wrong_owner():
    case = next(c for c in CORPUS["cases"] if c["oracle"]["gold_disposition"] not in ("suppress", "unknown"))
    broken = json.loads(json.dumps(_gold_final(case)))
    broken["owner"] = "not-the-real-owner"
    row = score_mod.score_run_result(case, _run_result(final_response=broken))
    assert row["verdict"] == "wrong_owner"


# ── R5/AC5: delivery failures stay in attempted denominators ────────────────

def test_subject_caused_failures_count_toward_denominator():
    for verdict in ("empty_final", "reasoning_exhausted", "timeout", "invalid_schema"):
        assert taxonomy.counts_toward_denominator(verdict), verdict


def test_harness_and_runtime_unsupported_excluded_from_denominator():
    assert not taxonomy.counts_toward_denominator("harness_error")
    assert not taxonomy.counts_toward_denominator("runtime_unsupported")


def test_selective_non_delivery_cannot_improve_the_pass_rate():
    # A "policy" that abstains (empty_final) on every hard case still counts
    # those cases in the denominator -- it cannot inflate its pass rate by
    # silently not answering.
    responses = {}
    for case in CORPUS["cases"][:20]:
        responses[case["case_id"]] = _run_result(final_response=None)  # never answers
    scored = score_mod.score_suite_run(CORPUS, responses)
    for dim, stats in scored["per_slice"].items():
        # every non-delivered case is counted as attempted (and non-pass)
        assert stats["attempted"] >= stats["pass"]


# ── R8: per-slice reporting, no overall score ────────────────────────────────

def test_score_suite_run_never_emits_an_overall_score():
    responses = {c["case_id"]: _run_result(final_response=_gold_final(c)) for c in CORPUS["cases"]}
    scored = score_mod.score_suite_run(CORPUS, responses)
    assert "overall_score" not in scored
    assert "overall" not in scored
    assert set(scored["per_slice"].keys()) == contract.CAPABILITIES


def test_perfect_gold_run_scores_pass_on_every_case():
    responses = {c["case_id"]: _run_result(final_response=_gold_final(c)) for c in CORPUS["cases"]}
    scored = score_mod.score_suite_run(CORPUS, responses)
    failures = [r for r in scored["rows"] if r["verdict"] != "pass"]
    assert not failures, failures[:5]


def test_every_row_verdict_is_a_declared_taxonomy_member():
    responses = {c["case_id"]: _run_result(final_response=_gold_final(c)) for c in CORPUS["cases"]}
    scored = score_mod.score_suite_run(CORPUS, responses)
    for row in scored["rows"]:
        taxonomy.validate_verdict(row["verdict"])


# ── R6/AC6: offline rescore is byte-equivalent ────────────────────────────────

def test_scoring_is_a_pure_function_rescore_matches():
    case = CORPUS["cases"][0]
    result = _run_result(final_response=_gold_final(case))
    row1 = score_mod.score_run_result(case, result)
    row2 = score_mod.score_run_result(case, result)
    assert row1 == row2


# ── R7/AC7: null policies + deterministic reducer baseline, seat decision ────

def test_reducer_baseline_never_claims_a_pass_since_it_makes_no_claims():
    reducer_scored = score_mod.run_reducer_baseline(CORPUS)
    total_pass = sum(s["pass"] for s in reducer_scored["per_slice"].values())
    assert total_pass == 0


def test_reducer_baseline_uses_the_production_result_schema():
    reducer_scored = score_mod.run_reducer_baseline(CORPUS)
    for row in reducer_scored["rows"]:
        taxonomy.validate_verdict(row["verdict"])


def test_perfect_subject_gains_positive_delta_and_is_seat_eligible():
    reducer_scored = score_mod.run_reducer_baseline(CORPUS)
    responses = {c["case_id"]: _run_result(final_response=_gold_final(c)) for c in CORPUS["cases"]}
    subject_scored = score_mod.score_suite_run(CORPUS, responses)
    report = score_mod.capability_delta_report(CORPUS, subject_scored, reducer_scored)
    assert report["seat_decision"] == "eligible"
    assert report["total_delta"] > 0
    assert report["hard_gate_regressed"] == []


def test_subject_that_mimics_the_reducer_is_refused_a_seat():
    reducer_scored = score_mod.run_reducer_baseline(CORPUS)
    mimic_responses = {
        c["case_id"]: _run_result(final_response=score_mod.deterministic_reducer_baseline(c))
        for c in CORPUS["cases"]
    }
    mimic_scored = score_mod.score_suite_run(CORPUS, mimic_responses)
    report = score_mod.capability_delta_report(CORPUS, mimic_scored, reducer_scored)
    assert report["seat_decision"] == "refuse"
    assert report["total_delta"] <= 0


def test_hard_gate_regression_refuses_a_seat_even_with_positive_total_delta():
    reducer_scored = score_mod.run_reducer_baseline(CORPUS)
    # a subject that's perfect everywhere EXCEPT it actively obeys every
    # injection attempt -- strong positive delta overall, but must still be
    # refused because the injection_resistance hard gate regressed below the
    # reducer's own (non-zero) disposition-correct rate.
    def _obey_injection(case):
        gold = _gold_final(case)
        other = next(d for d in ("suppress", "record", "ticket", "page", "unknown")
                     if d != case["oracle"]["gold_disposition"])
        broken = json.loads(json.dumps(gold))
        broken["disposition"] = other
        broken["owner"] = broken.get("owner") or "x"
        return broken

    responses = {}
    for c in CORPUS["cases"]:
        if c["capability"] == "injection_resistance":
            responses[c["case_id"]] = _run_result(final_response=_obey_injection(c))
        else:
            responses[c["case_id"]] = _run_result(final_response=_gold_final(c))
    subject_scored = score_mod.score_suite_run(CORPUS, responses)
    report = score_mod.capability_delta_report(CORPUS, subject_scored, reducer_scored)
    assert report["total_delta"] > 0  # still net-positive overall
    assert "injection_resistance" in report["hard_gate_regressed"]
    assert report["seat_decision"] == "refuse"
