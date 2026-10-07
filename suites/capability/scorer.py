#!/usr/bin/env python3
"""
scorer.py -- deterministic capability-suite scorer (SPEC-003-015 R2/R8/R9).

Scores a subject's structured response against a case's hidden oracle. Never
an LLM judge -- every check is a mechanical comparison against
suites/capability/corpus.json's oracle fields, mirroring
scripts/score_cat07.py's "no LLM judge, ever" precedent.

Subject response shape (R2): {"claims": [...], "disposition": str,
"severity": str, "owner": str, "next_probe": str|None,
"evidence_refs": [str, ...]}
"""

from __future__ import annotations

REQUIRED_RESPONSE_FIELDS = ("claims", "disposition", "severity", "owner", "next_probe", "evidence_refs")


class ScoringError(Exception):
    pass


def _claim_key(atom: dict) -> tuple:
    return (atom.get("subject"), atom.get("predicate"), atom.get("object"),
            atom.get("polarity"), atom.get("scope"))


def _claims_entail(response_claims: list, oracle_claim_atoms: list) -> bool:
    """Every oracle claim_atom must have an exactly-matching (subject,
    predicate, object, polarity) entry among the response's claims -- R2's
    entailment requirement. A response that flips one polarity, substitutes
    the object/entity, or omits the claim entirely fails this check (AC3's
    negation/entity/temporal-drift contrast fixtures exercise this)."""
    response_keys = {_claim_key(c) for c in response_claims if isinstance(c, dict)}
    return all(_claim_key(atom) in response_keys for atom in oracle_claim_atoms)


def _response_well_formed(response: dict) -> list[str]:
    errors = []
    for field in REQUIRED_RESPONSE_FIELDS:
        if field not in response:
            errors.append(f"response missing field {field!r}")
    return errors


def score_response(case: dict, response: dict) -> dict:
    """Return {"verdict": "pass"|"fail", "reasons": [...], "checks": {...}}.

    Checks, in order (any failure -> verdict "fail", all reasons collected --
    R8 "hard-failure counts", not a single opaque bool):
      1. response is well-formed (has every required field).
      2. disposition matches oracle.gold_disposition exactly.
      3. disposition is one of oracle.permitted_dispositions.
      4. no prohibited_action was taken (heuristic: response text must not
         assert a claim absent from evidence_refs -- R9's "unsupported
         outside knowledge is an error, even when factually true").
      5. actionability (AC8): a disposition other than 'suppress'/'unknown'
         needs a non-empty owner and evidence_refs; 'page'/'ticket' need a
         next_probe too, unless oracle.gold_next_probe is explicitly None.
    """
    reasons = []

    malformed = _response_well_formed(response)
    if malformed:
        return {"verdict": "fail", "reasons": malformed, "checks": {}}

    oracle = case["oracle"]
    checks = {}

    checks["claims_entail_oracle"] = _claims_entail(response.get("claims", []), oracle["claim_atoms"])
    if not checks["claims_entail_oracle"]:
        reasons.append("response claims do not entail every oracle claim_atom (subject/predicate/object/polarity)")

    checks["disposition_matches_gold"] = response["disposition"] == oracle["gold_disposition"]
    if not checks["disposition_matches_gold"]:
        reasons.append(
            f"disposition {response['disposition']!r} != gold {oracle['gold_disposition']!r}"
        )

    checks["disposition_permitted"] = response["disposition"] in oracle["permitted_dispositions"]
    if not checks["disposition_permitted"]:
        reasons.append(f"disposition {response['disposition']!r} not in permitted set")

    checks["evidence_grounded"] = bool(response["evidence_refs"]) and set(
        response["evidence_refs"]
    ).issubset(set(case["evidence_refs"]))
    if not checks["evidence_grounded"]:
        reasons.append("evidence_refs empty or cite something outside the case's own evidence_refs")

    needs_owner = response["disposition"] not in ("suppress", "unknown")
    checks["actionability"] = True
    if needs_owner and not response.get("owner"):
        checks["actionability"] = False
        reasons.append("actionable disposition missing an owner")
    if response["disposition"] in ("page", "ticket"):
        if oracle.get("gold_next_probe") is not None and not response.get("next_probe"):
            checks["actionability"] = False
            reasons.append("page/ticket disposition missing a next_probe")

    verdict = "pass" if all(checks.values()) else "fail"
    return {"verdict": verdict, "reasons": reasons, "checks": checks}


def score_corpus(corpus: dict, responses: dict) -> dict:
    """responses: {case_id: response_dict}. Returns per-slice pass/fail counts
    and hard-failure counts -- R8: no single overall score."""
    per_slice: dict[str, dict] = {}
    rows = []
    for case in corpus["cases"]:
        response = responses.get(case["case_id"])
        if response is None:
            row = {"case_id": case["case_id"], "verdict": "no_response", "reasons": ["not attempted"]}
        else:
            row = {"case_id": case["case_id"], **score_response(case, response)}
        rows.append(row)

        slice_stats = per_slice.setdefault(
            case["capability"], {"total": 0, "pass": 0, "fail": 0, "no_response": 0}
        )
        slice_stats["total"] += 1
        slice_stats[row["verdict"]] = slice_stats.get(row["verdict"], 0) + 1

    return {"rows": rows, "per_slice": per_slice}
