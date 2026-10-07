#!/usr/bin/env python3
"""
contract.py -- schema validation, corpus-coverage checks, and subject-packet
serialization for the general capability suite (SPEC-003-015).

No external packages (stdlib only) -- a hand-rolled validator against
suites/schemas/capability-case.schema.json's specific shape, matching this
project's established convention (scripts/run_provenance.py's
validate_provenance(), scripts/model_identity.py) rather than depending on
the `jsonschema` package, which is not a declared project dependency.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).parent
BENCH_DIR = HERE.parent.parent
CORPUS_PATH = HERE / "corpus.json"

CAPABILITIES = frozenset({
    "state_extraction", "transition_reasoning", "grounding_abstention",
    "tool_routing", "bounded_planning", "correction_recovery",
    "identity_isolation", "escalation_safety", "injection_resistance",
    "operational_reporting",
})

SKINS = frozenset({
    "abstract_workflow", "ci_deployment", "service_health", "batch_processing",
    "document_workflow", "portfolio_research", "nightshift_monitoring",
})

DIFFICULTIES = frozenset({"basic", "healthy", "adversarial", "contradictory", "insufficient_evidence"})
DISPOSITIONS = frozenset({"suppress", "record", "ticket", "page", "unknown"})
REQUIRED_DIFFICULTY_SET = frozenset({"basic", "healthy", "adversarial", "contradictory", "insufficient_evidence"})

CASE_REQUIRED_FIELDS = (
    "case_id", "logical_pair_id", "variant", "capability", "diagnostic_tags",
    "difficulty", "scenario_skin", "source_basis", "scoring_version",
    "trusted_instructions", "supplied_events", "tools", "expected_output_schema",
    "prohibited_actions", "evidence_refs", "oracle",
)
ORACLE_REQUIRED_FIELDS = (
    "observation_time", "trust_channel_policy", "risk_tier", "error_costs",
    "authoritative_signal_ids", "state_liveness_policy", "grace_phase",
    "required_escalation", "permitted_dispositions", "ownership_slo_policy",
    "claim_atoms", "gold_disposition", "gold_severity", "gold_owner",
    "gold_next_probe", "mutation_sensitive_fields",
)


class ContractError(Exception):
    """A case or corpus fails the capability contract."""


def load_corpus(path: Path = CORPUS_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ── AC1/AC2: per-case schema validation ───────────────────────────────────────

def validate_case(case: dict) -> list[str]:
    """Return a list of violations (empty = valid). AC2: missing logical-pair
    ID, evidence oracle, prohibited-action list, source basis, or scoring
    version fails before inference."""
    errors = []

    for field in CASE_REQUIRED_FIELDS:
        if field not in case:
            errors.append(f"missing required field {field!r}")
    if errors:
        return errors  # can't check deeper without the required shape

    if case["capability"] not in CAPABILITIES:
        errors.append(f"unknown capability {case['capability']!r} -- AC1 rejects unscored capabilities")
    if case["scenario_skin"] not in SKINS:
        errors.append(f"unknown scenario_skin {case['scenario_skin']!r}")
    if case["difficulty"] not in DIFFICULTIES:
        errors.append(f"unknown difficulty {case['difficulty']!r}")
    if case["source_basis"] != "synthetic":
        errors.append(f"source_basis must be 'synthetic', got {case['source_basis']!r} (NFR-001/AC6)")
    if not case["logical_pair_id"]:
        errors.append("empty logical_pair_id")
    if not case["prohibited_actions"]:
        errors.append("empty prohibited_actions list")
    if not case["scoring_version"]:
        errors.append("empty scoring_version")
    if not case["evidence_refs"]:
        errors.append("empty evidence_refs -- AC2 requires an evidence oracle")

    oracle = case["oracle"]
    for field in ORACLE_REQUIRED_FIELDS:
        if field not in oracle:
            errors.append(f"oracle missing required field {field!r}")
    if "gold_disposition" in oracle and oracle["gold_disposition"] not in DISPOSITIONS:
        errors.append(f"oracle.gold_disposition {oracle['gold_disposition']!r} not in {sorted(DISPOSITIONS)}")
    if "mutation_sensitive_fields" in oracle and not oracle["mutation_sensitive_fields"]:
        errors.append("oracle.mutation_sensitive_fields must be non-empty (R9)")

    for event in case.get("supplied_events", []):
        if event.get("trust_channel") not in ("trusted", "untrusted"):
            errors.append(f"supplied_event {event.get('event_id')} has invalid trust_channel")

    return errors


# ── AC2/AC6/LE2: subject-packet serialization strips the oracle ──────────────

_SUBJECT_VISIBLE_FIELDS = (
    "case_id", "trusted_instructions", "supplied_events", "tools",
    "expected_output_schema", "checkpoints",
)


def serialize_for_subject(case: dict) -> dict:
    """The only fields a subject model may see. Strips oracle, logical_pair_id,
    capability, difficulty, scenario_skin, diagnostic_tags, prohibited_actions,
    evidence_refs, scoring_version, source_basis, latent_trace_id -- anything
    that names the gold, the taxonomy slice, or scorer internals (R2, LE2)."""
    return {k: case[k] for k in _SUBJECT_VISIBLE_FIELDS if k in case}


def assert_no_gold_leak(packet: dict) -> None:
    """AC6: the subject request serializer demonstrably omits qualification gold."""
    forbidden = ("oracle", "gold_disposition", "gold_severity", "gold_owner",
                 "mutation_sensitive_fields", "claim_atoms", "prohibited_actions")
    blob = json.dumps(packet)
    for term in forbidden:
        if term in blob:
            raise ContractError(f"subject packet leaks gold-adjacent field: {term!r}")


# ── AC4/AC5: corpus-level coverage report ─────────────────────────────────────

def coverage_report(corpus: dict) -> dict:
    cases = corpus["cases"]
    ids = [c["case_id"] for c in cases]
    duplicate_ids = [i for i in set(ids) if ids.count(i) > 1]

    by_capability: dict[str, list[dict]] = {}
    for c in cases:
        by_capability.setdefault(c["capability"], []).append(c)

    logical_families = {c["logical_pair_id"] for c in cases}
    checkpoint_traces = [c for c in cases if "checkpoints" in c and len(c["checkpoints"]) >= 3]

    skins_overall = {c["scenario_skin"] for c in cases}
    skins_per_slice = {dim: {c["scenario_skin"] for c in group} for dim, group in by_capability.items()}
    nightshift_count = sum(1 for c in cases if c["scenario_skin"] == "nightshift_monitoring")

    difficulty_per_slice = {dim: {c["difficulty"] for c in group} for dim, group in by_capability.items()}
    missing_difficulty = {
        dim: sorted(REQUIRED_DIFFICULTY_SET - diffs)
        for dim, diffs in difficulty_per_slice.items() if REQUIRED_DIFFICULTY_SET - diffs
    }

    return {
        "total_instances": len(cases),
        "logical_families": len(logical_families),
        "duplicate_ids": duplicate_ids,
        "instances_per_capability": {dim: len(group) for dim, group in by_capability.items()},
        "families_per_capability": {
            dim: len({c["logical_pair_id"] for c in group}) for dim, group in by_capability.items()
        },
        "checkpoint_traces": len(checkpoint_traces),
        "skins_overall": sorted(skins_overall),
        "skins_per_slice": {dim: sorted(s) for dim, s in skins_per_slice.items()},
        "nightshift_pct": (nightshift_count / len(cases) * 100) if cases else 0.0,
        "missing_difficulty_per_slice": missing_difficulty,
    }


def assert_corpus_meets_minimums(corpus: dict) -> None:
    """AC4/AC5/AC6: hard-fail if the generated corpus doesn't meet R4-R6's numeric floors."""
    report = coverage_report(corpus)
    if report["duplicate_ids"]:
        raise ContractError(f"duplicate case_id(s): {report['duplicate_ids']}")
    if report["logical_families"] < 60:
        raise ContractError(f"only {report['logical_families']} logical families, need >= 60")
    if report["total_instances"] < 120:
        raise ContractError(f"only {report['total_instances']} instances, need >= 120")
    for dim, n in report["families_per_capability"].items():
        if n < 6:
            raise ContractError(f"{dim}: only {n} logical families, need >= 6")
    if report["checkpoint_traces"] < 12:
        raise ContractError(f"only {report['checkpoint_traces']} multi-checkpoint traces, need >= 12")
    if len(report["skins_overall"]) < 4:
        raise ContractError(f"only {len(report['skins_overall'])} skins overall, need >= 4")
    for dim, skins in report["skins_per_slice"].items():
        if len(skins) < 2:
            raise ContractError(f"{dim}: only {len(skins)} skins, need >= 2")
    if report["nightshift_pct"] > 25.0:
        raise ContractError(f"nightshift_monitoring is {report['nightshift_pct']:.1f}% of corpus, must be <= 25%")
    if report["missing_difficulty_per_slice"]:
        raise ContractError(f"slices missing required difficulty tags: {report['missing_difficulty_per_slice']}")


# ── AC6/NFR-001: provenance scan ──────────────────────────────────────────────

def assert_all_synthetic(corpus: dict) -> None:
    non_synthetic = [c["case_id"] for c in corpus["cases"] if c.get("source_basis") != "synthetic"]
    if non_synthetic:
        raise ContractError(f"non-synthetic source_basis on: {non_synthetic}")


# ── R9/AC11: mutation-sensitivity tests ───────────────────────────────────────

def build_gold_response(case: dict) -> dict:
    """Construct the response a perfectly-correct subject would give -- derived
    straight from the oracle, so scoring it against the case's own (unmutated)
    oracle must always be 'pass'. Used as the mutation-test baseline."""
    oracle = case["oracle"]
    return {
        "claims": [dict(a) for a in oracle["claim_atoms"]],
        "disposition": oracle["gold_disposition"],
        "severity": oracle["gold_severity"],
        "owner": oracle["gold_owner"],
        "next_probe": oracle["gold_next_probe"],
        "evidence_refs": list(case["evidence_refs"]),
    }


def _set_path(d: dict, path: str, value):
    """Set a dotted/indexed path like 'oracle.claim_atoms.0.polarity'."""
    parts = path.split(".")
    node = d
    for part in parts[:-1]:
        node = node[int(part)] if part.isdigit() else node[part]
    last = parts[-1]
    if last.isdigit():
        node[int(last)] = value
    else:
        node[last] = value


def _get_path(d: dict, path: str):
    parts = path.split(".")
    node = d
    for part in parts:
        node = node[int(part)] if part.isdigit() else node[part]
    return node


def mutation_sensitivity_report(corpus: dict, scorer_module) -> dict:
    """R9/AC11: for every declared oracle.mutation_sensitive_fields entry on
    every case, flip that field and prove the scorer verdict changes relative
    to the gold response scored against the unmutated oracle."""
    results = []
    for case in corpus["cases"]:
        gold = build_gold_response(case)
        baseline = scorer_module.score_response(case, gold)
        if baseline["verdict"] != "pass":
            results.append({
                "case_id": case["case_id"], "field": None, "ok": False,
                "reason": f"gold response itself did not pass: {baseline['reasons']}",
            })
            continue

        for field in case["oracle"].get("mutation_sensitive_fields", []):
            mutated_case = json.loads(json.dumps(case))
            try:
                current = _get_path(mutated_case, field)
            except (KeyError, IndexError, TypeError) as exc:
                results.append({"case_id": case["case_id"], "field": field, "ok": False,
                                 "reason": f"cannot resolve path: {exc}"})
                continue

            if isinstance(current, str) and current in DISPOSITIONS:
                other = next(d for d in sorted(DISPOSITIONS) if d != current)
                _set_path(mutated_case, field, other)
            elif isinstance(current, str) and current in ("positive", "negative"):
                _set_path(mutated_case, field, "negative" if current == "positive" else "positive")
            elif isinstance(current, bool):
                _set_path(mutated_case, field, not current)
            elif isinstance(current, str):
                _set_path(mutated_case, field, current + "-MUTATED")
            else:
                _set_path(mutated_case, field, None)

            mutated_verdict = scorer_module.score_response(mutated_case, gold)
            changed = mutated_verdict["verdict"] != baseline["verdict"]
            results.append({
                "case_id": case["case_id"], "field": field, "ok": changed,
                "reason": None if changed else "mutating this field did not change the verdict",
            })

    failures = [r for r in results if not r["ok"]]
    return {"total_checks": len(results), "failures": failures, "all_sensitive": not failures}


if __name__ == "__main__":
    import sys
    corpus = load_corpus()
    errors = []
    for case in corpus["cases"]:
        case_errors = validate_case(case)
        if case_errors:
            errors.append((case["case_id"], case_errors))
    if errors:
        for case_id, errs in errors:
            print(f"{case_id}: {errs}")
        sys.exit(1)
    try:
        assert_corpus_meets_minimums(corpus)
        assert_all_synthetic(corpus)
    except ContractError as exc:
        print(f"error: {exc}")
        sys.exit(1)
    print("OK:", json.dumps(coverage_report(corpus), indent=2))
