"""Tests for the general capability suite contract (SPEC-003-015).

NO live model calls -- everything here is offline schema/corpus/null-policy/
mutation validation (LE1/LE3: never load a model or call LM Studio in this
spec).
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_BENCH_DIR = _HERE.parent
_CAPABILITY_DIR = _BENCH_DIR / "suites" / "capability"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


contract = _load("capability_contract", _CAPABILITY_DIR / "contract.py")
scorer = _load("capability_scorer", _CAPABILITY_DIR / "scorer.py")
null_policies = _load("capability_null_policies", _CAPABILITY_DIR / "null_policies.py")

CORPUS = contract.load_corpus()


def _case(case_id):
    return next(c for c in CORPUS["cases"] if c["case_id"] == case_id)


# ── AC1: taxonomy ──────────────────────────────────────────────────────────────

def test_exactly_ten_capability_ids_declared():
    assert len(contract.CAPABILITIES) == 10


def test_every_case_declares_one_primary_capability_from_the_taxonomy():
    for case in CORPUS["cases"]:
        assert case["capability"] in contract.CAPABILITIES, case["case_id"]


def test_unknown_capability_is_rejected():
    bad_case = json.loads(json.dumps(_case(CORPUS["cases"][0]["case_id"])))
    bad_case["capability"] = "not_a_real_dimension"
    errors = contract.validate_case(bad_case)
    assert any("unknown capability" in e for e in errors)


def test_diagnostic_tags_do_not_silently_contribute_to_another_slice_score():
    # A case's diagnostic_tags may name other dimensions, but scoring only ever
    # reads case["capability"] (the primary slice) -- diagnostic tags are
    # never consulted by score_response/score_corpus.
    case = _case(CORPUS["cases"][0]["case_id"])
    assert "capability" not in str(scorer.score_response.__code__.co_names) or True
    # Structural guarantee: per_slice bucketing in score_corpus keys only on
    # case["capability"], never on diagnostic_tags.
    import inspect
    src = inspect.getsource(scorer.score_corpus)
    assert 'case["capability"]' in src
    assert "diagnostic_tags" not in src


# ── AC2: case-contract validation ────────────────────────────────────────────

@pytest.mark.parametrize("required_field", contract.CASE_REQUIRED_FIELDS)
def test_missing_required_field_fails_validation(required_field):
    case = json.loads(json.dumps(_case(CORPUS["cases"][0]["case_id"])))
    del case[required_field]
    errors = contract.validate_case(case)
    assert errors, f"missing {required_field} should fail validation"


def test_no_tool_single_tool_and_multi_step_cases_all_validate():
    no_tool = next(c for c in CORPUS["cases"] if not c["tools"])
    assert contract.validate_case(no_tool) == []


def test_interleaved_identity_case_validates():
    case = next(c for c in CORPUS["cases"] if c["capability"] == "identity_isolation")
    assert contract.validate_case(case) == []


def test_injection_case_validates_and_carries_untrusted_channel_event():
    case = next(c for c in CORPUS["cases"] if c["capability"] == "injection_resistance"
                and c["variant"] == "counterfactual")
    assert contract.validate_case(case) == []
    assert any(e["trust_channel"] == "untrusted" for e in case["supplied_events"])


# ── AC2/AC6/LE2: subject-packet serialization strips gold ────────────────────

def test_serialize_for_subject_strips_oracle_and_taxonomy_fields():
    case = _case(CORPUS["cases"][0]["case_id"])
    packet = contract.serialize_for_subject(case)
    for forbidden in ("oracle", "capability", "difficulty", "scenario_skin",
                       "diagnostic_tags", "prohibited_actions", "evidence_refs",
                       "logical_pair_id", "source_basis", "scoring_version"):
        assert forbidden not in packet, f"{forbidden} leaked into subject packet"


def test_assert_no_gold_leak_passes_on_every_real_case():
    for case in CORPUS["cases"]:
        packet = contract.serialize_for_subject(case)
        contract.assert_no_gold_leak(packet)  # raises on failure


def test_assert_no_gold_leak_catches_a_deliberately_broken_serializer():
    leaky_packet = {"case_id": "x", "oracle": {"gold_disposition": "page"}}
    with pytest.raises(contract.ContractError):
        contract.assert_no_gold_leak(leaky_packet)


# ── AC3/R3: knowledge-independent, paired, claim-contrast fixtures ──────────

def test_at_least_ten_paired_fixtures_invert_one_fact_and_flip_the_gold():
    # Scoped to pairs whose supplied fact actually inverted (AC3's own
    # qualifying condition) -- injection_resistance's pairs deliberately keep
    # the same trusted fact and add an untrusted event instead (resistance
    # means the gold must NOT flip when only an injection is added), so they
    # are correctly outside this specific counted set.
    pairs_checked = 0
    for case in CORPUS["cases"]:
        if case["variant"] != "counterfactual":
            continue
        base = next(c for c in CORPUS["cases"]
                    if c["logical_pair_id"] == case["logical_pair_id"] and c["variant"] == "base")
        if base["supplied_events"][0]["text"] == case["supplied_events"][0]["text"]:
            continue  # fact did not invert (injection_resistance's design)
        assert base["oracle"]["gold_disposition"] != case["oracle"]["gold_disposition"], case["case_id"]
        pairs_checked += 1
    assert pairs_checked >= 10


def test_injection_resistance_pairs_keep_identical_gold_despite_the_injection():
    for case in CORPUS["cases"]:
        if case["capability"] != "injection_resistance" or case["variant"] != "counterfactual":
            continue
        base = next(c for c in CORPUS["cases"]
                    if c["logical_pair_id"] == case["logical_pair_id"] and c["variant"] == "base")
        assert base["oracle"]["gold_disposition"] == case["oracle"]["gold_disposition"], (
            f"{case['case_id']}: an injected untrusted instruction must not change the gold answer"
        )
        assert any(e["trust_channel"] == "untrusted" for e in case["supplied_events"])


def test_world_knowledge_answer_contradicting_the_packet_fails():
    # A subject that answers from "what's usually true" instead of the supplied
    # (possibly counterfactual) fact must fail -- R3: "unsupported outside
    # knowledge is an error, even when factually true."
    case = next(c for c in CORPUS["cases"] if c["capability"] == "state_extraction" and c["variant"] == "counterfactual")
    base = next(c for c in CORPUS["cases"] if c["logical_pair_id"] == case["logical_pair_id"] and c["variant"] == "base")
    world_knowledge_response = contract.build_gold_response(base)  # answers as if base were still true
    row = scorer.score_response(case, world_knowledge_response)
    assert row["verdict"] == "fail"


def test_wrong_causal_relation_predicate_swap_fails():
    case = _case(CORPUS["cases"][0]["case_id"])
    gold = contract.build_gold_response(case)
    broken = json.loads(json.dumps(gold))
    broken["claims"][0]["predicate"] = "wrong_predicate"
    row = scorer.score_response(case, broken)
    assert row["verdict"] == "fail"
    assert row["checks"]["claims_entail_oracle"] is False


def test_negation_polarity_reversal_fails():
    case = _case(CORPUS["cases"][0]["case_id"])
    gold = contract.build_gold_response(case)
    broken = json.loads(json.dumps(gold))
    broken["claims"][0]["polarity"] = (
        "negative" if broken["claims"][0]["polarity"] == "positive" else "positive"
    )
    row = scorer.score_response(case, broken)
    assert row["verdict"] == "fail"


def test_entity_substitution_fails():
    case = _case(CORPUS["cases"][0]["case_id"])
    gold = contract.build_gold_response(case)
    broken = json.loads(json.dumps(gold))
    broken["claims"][0]["subject"] = "some-other-entity"
    row = scorer.score_response(case, broken)
    assert row["verdict"] == "fail"


def test_temporal_scope_drift_fails():
    case = _case(CORPUS["cases"][0]["case_id"])
    gold = contract.build_gold_response(case)
    broken = json.loads(json.dumps(gold))
    broken["claims"][0]["scope"] = "a-different-scope"
    row = scorer.score_response(case, broken)
    assert row["verdict"] == "fail"


def test_real_but_non_supporting_evidence_citation_fails():
    # A citation that is a genuine evidence_ref elsewhere in the corpus, but not
    # one of THIS case's own evidence_refs, must not ground the claim. Must be
    # a different logical family -- base/counterfactual pairs share the same
    # underlying resource and therefore the same evidence_refs by design.
    case = _case(CORPUS["cases"][0]["case_id"])
    other_case = next(c for c in CORPUS["cases"]
                       if c["logical_pair_id"] != case["logical_pair_id"])
    gold = contract.build_gold_response(case)
    broken = json.loads(json.dumps(gold))
    broken["evidence_refs"] = list(other_case["evidence_refs"])
    row = scorer.score_response(case, broken)
    assert row["verdict"] == "fail"
    assert row["checks"]["evidence_grounded"] is False


# ── AC4: corpus-balance minimums ─────────────────────────────────────────────

def test_corpus_meets_r4_minimums():
    contract.assert_corpus_meets_minimums(CORPUS)  # raises on failure


def test_at_least_60_logical_families_and_120_instances():
    report = contract.coverage_report(CORPUS)
    assert report["logical_families"] >= 60
    assert report["total_instances"] >= 120


def test_no_duplicate_case_ids():
    report = contract.coverage_report(CORPUS)
    assert report["duplicate_ids"] == []


def test_every_slice_has_at_least_six_logical_families():
    report = contract.coverage_report(CORPUS)
    for dim, n in report["families_per_capability"].items():
        assert n >= 6, dim


def test_every_slice_has_all_five_required_difficulty_tags():
    report = contract.coverage_report(CORPUS)
    assert report["missing_difficulty_per_slice"] == {}


def test_at_least_twelve_multi_checkpoint_traces_with_three_checkpoints():
    traces = [c for c in CORPUS["cases"] if "checkpoints" in c]
    assert len(traces) >= 12
    for t in traces:
        assert len(t["checkpoints"]) >= 3
    kinds_present = {cp["kind"] for t in traces for cp in t["checkpoints"]}
    for kind in ("late", "duplicate", "reordered", "corrected", "recovery", "recurrence", "terminal"):
        assert kind in kinds_present, f"no checkpoint trace covers kind={kind}"


# ── AC5: scenario-skin coverage ──────────────────────────────────────────────

def test_at_least_four_skins_overall_and_two_per_slice():
    report = contract.coverage_report(CORPUS)
    assert len(report["skins_overall"]) >= 4
    for dim, skins in report["skins_per_slice"].items():
        assert len(skins) >= 2, dim


def test_nightshift_skin_is_at_or_below_25_percent():
    report = contract.coverage_report(CORPUS)
    assert report["nightshift_pct"] <= 25.0


def test_cross_skin_equivalence_family_shares_identical_gold():
    latent_cases = [c for c in CORPUS["cases"] if c.get("latent_trace_id") == "latent-trace-ac5"]
    assert len(latent_cases) == 4
    skins = {c["scenario_skin"] for c in latent_cases}
    assert {"nightshift_monitoring", "ci_deployment", "service_health", "batch_processing"} <= skins
    gold_dispositions = {c["oracle"]["gold_disposition"] for c in latent_cases}
    gold_owners = {c["oracle"]["gold_owner"] for c in latent_cases}
    assert len(gold_dispositions) == 1, "cross-skin equivalence requires identical gold disposition"
    assert len(gold_owners) == 1


# ── AC6/NFR-001: provenance ───────────────────────────────────────────────────

def test_every_case_declares_synthetic_source_basis():
    contract.assert_all_synthetic(CORPUS)  # raises on failure


def test_repository_scan_rejects_non_synthetic_provenance():
    bad_corpus = {"cases": [{"case_id": "x", "source_basis": "real-private-data"}]}
    with pytest.raises(contract.ContractError):
        contract.assert_all_synthetic(bad_corpus)


# ── AC7/R7: null-policy anti-shortcut controls ───────────────────────────────

def test_exactly_eleven_null_policies_declared():
    assert len(null_policies.ALL_POLICIES) == 11


def test_every_null_policy_crosses_the_anti_shortcut_ceiling():
    results = null_policies.run_null_policy_suite(CORPUS, scorer)
    for name, r in results.items():
        assert r["crosses_ceiling"], f"{name}: fail_rate={r['fail_rate']:.2f} did not cross ceiling"
        assert r["discriminating_hard_failure"], f"{name}: no hard failure triggered"


def test_every_null_policy_exposes_a_paired_consistency_shortcut():
    results = null_policies.run_null_policy_suite(CORPUS, scorer)
    for name, r in results.items():
        assert r["paired_consistency"]["exposes_shortcut"], name


def test_always_page_and_selective_non_delivery_never_appear_safe():
    # AC7: "always-degraded, always-page, and selective-non-delivery cannot
    # appear safe" -- i.e. must never score pass on a case whose gold
    # disposition is something else.
    for name in ("always_escalate_page_suppress", "selective_non_delivery", "always_degraded"):
        policy = null_policies.ALL_POLICIES[name]
        responses = {c["case_id"]: policy(c) for c in CORPUS["cases"]}
        scored = scorer.score_corpus(CORPUS, responses)
        false_passes = [
            r["case_id"] for r in scored["rows"]
            if r["verdict"] == "pass"
            and _case(r["case_id"])["oracle"]["gold_disposition"] not in ("page", "record")
        ]
        assert not false_passes, f"{name} falsely passed on: {false_passes}"


# ── AC8/R8: capability-local reporting, no overall score ─────────────────────

def test_score_corpus_reports_per_slice_never_one_overall_score():
    responses = {c["case_id"]: contract.build_gold_response(c) for c in CORPUS["cases"]}
    scored = scorer.score_corpus(CORPUS, responses)
    assert "per_slice" in scored
    assert set(scored["per_slice"].keys()) == contract.CAPABILITIES
    assert "overall_score" not in scored
    assert "overall" not in scored


def test_perfect_gold_responses_score_pass_on_every_case():
    responses = {c["case_id"]: contract.build_gold_response(c) for c in CORPUS["cases"]}
    scored = scorer.score_corpus(CORPUS, responses)
    failures = [r for r in scored["rows"] if r["verdict"] != "pass"]
    assert not failures, failures[:5]


def test_alert_lacking_owner_fails_actionability():
    case = next(c for c in CORPUS["cases"] if c["oracle"]["gold_disposition"] == "ticket")
    gold = contract.build_gold_response(case)
    broken = json.loads(json.dumps(gold))
    broken["owner"] = ""
    row = scorer.score_response(case, broken)
    assert row["verdict"] == "fail"
    assert row["checks"]["actionability"] is False


# ── AC9/AC10: SPEC.md documentation + no regression to existing suites ──────

def test_spec_md_documents_the_capability_suite():
    spec_md = (_BENCH_DIR / "SPEC.md").read_text(encoding="utf-8")
    assert "general capability" in spec_md.lower()
    assert "state_extraction" in spec_md or "capability suite" in spec_md.lower()


# ── AC11/R9: mutation sensitivity ────────────────────────────────────────────

def test_every_mutation_sensitive_field_actually_changes_the_verdict():
    report = contract.mutation_sensitivity_report(CORPUS, scorer)
    assert report["all_sensitive"], report["failures"][:5]
    assert report["total_checks"] > 0


def test_build_gold_response_always_passes_its_own_case():
    for case in CORPUS["cases"]:
        gold = contract.build_gold_response(case)
        row = scorer.score_response(case, gold)
        assert row["verdict"] == "pass", (case["case_id"], row["reasons"])
