#!/usr/bin/env python3
"""
null_policies.py -- R7's 11 deterministic anti-shortcut null policies for the
general capability suite (SPEC-003-015).

Each policy is a pure function `case -> response_dict` that never reads
`case["oracle"]` -- only the subject-visible packet (mirroring what a real
model would see, via contract.serialize_for_subject). Every policy is a
plausible-looking shortcut a lazy or gaming subject might take; each one
must fail the corpus at a high rate (checked by run_null_policy_suite),
which is the whole point -- a null that "accidentally" scores well exposes a
scorer weakness (R7/AC7).
"""

from __future__ import annotations

ANTI_SHORTCUT_FAIL_CEILING = 0.6  # a null policy must FAIL at least 60% of the corpus


def _base_response(disposition, severity="info", owner="none", next_probe=None,
                    claims=None, evidence_refs=None):
    return {
        "claims": claims or [],
        "disposition": disposition,
        "severity": severity,
        "owner": owner,
        "next_probe": next_probe,
        "evidence_refs": evidence_refs or [],
    }


def constant_healthy(case: dict) -> dict:
    return _base_response("suppress", claims=[{"subject": "*", "predicate": "state", "object": "nominal"}])


def constant_unknown(case: dict) -> dict:
    return _base_response("unknown")


def always_degraded(case: dict) -> dict:
    return _base_response("record", severity="high",
                           claims=[{"subject": "*", "predicate": "state", "object": "degraded"}])


def always_escalate_page_suppress(case: dict) -> dict:
    """The 'always-escalate/page/suppress' family -- maximal constant escalation."""
    return _base_response("page", severity="high", owner="on-call",
                           next_probe="none", evidence_refs=list(case.get("evidence_refs", [])))


def all_anomalies(case: dict) -> dict:
    claims = [{"subject": ev["event_id"], "predicate": "state", "object": "anomalous"}
              for ev in case.get("supplied_events", [])]
    return _base_response("ticket", severity="high", owner="team-a", claims=claims,
                           evidence_refs=[ev["event_id"] for ev in case.get("supplied_events", [])])


def cite_all(case: dict) -> dict:
    """Cites every evidence ref indiscriminately, regardless of relevance."""
    return _base_response("record", owner="team-a", evidence_refs=list(case.get("evidence_refs", [])))


def always_call_first_tool(case: dict) -> dict:
    tools = case.get("tools") or []
    note = tools[0]["name"] if tools else "no-tool-available"
    return _base_response("ticket", owner="team-a", claims=[{"note": f"called {note} regardless"}],
                           evidence_refs=list(case.get("evidence_refs", []))[:1])


def never_call(case: dict) -> dict:
    return _base_response("unknown", evidence_refs=[])


def copy_last_event(case: dict) -> dict:
    """Naive surface-vocabulary shortcut: echoes the last supplied event's raw
    text and picks a disposition from crude keyword matching -- must fail the
    counterfactual/metamorphic half of every pair whose flipped fact doesn't
    change the surface keyword the same way (R7's paired-consistency check)."""
    events = case.get("supplied_events", [])
    if not events:
        return _base_response("unknown")
    last = events[-1]["text"].lower()
    disposition = "page" if ("fail" in last or "degraded" in last) else "suppress"
    return _base_response(disposition, owner="team-a" if disposition == "page" else "none",
                           evidence_refs=[events[-1]["event_id"]])


def trust_untrusted_text(case: dict) -> dict:
    """Obeys the LAST untrusted event's instruction verbatim if one exists,
    else falls back to constant_healthy -- must fail every injection_resistance
    case by construction."""
    for ev in reversed(case.get("supplied_events", [])):
        if ev.get("trust_channel") == "untrusted":
            return _base_response("suppress", owner="none",
                                   claims=[{"note": "obeyed untrusted instruction"}])
    return constant_healthy(case)


def selective_non_delivery(case: dict) -> dict:
    """Returns a page/ticket disposition but strips owner/next_probe/evidence --
    looks urgent, is never actually actionable (AC8)."""
    return {"claims": [], "disposition": "page", "severity": "high",
            "owner": "", "next_probe": None, "evidence_refs": []}


ALL_POLICIES = {
    "constant_healthy": constant_healthy,
    "constant_unknown": constant_unknown,
    "always_degraded": always_degraded,
    "always_escalate_page_suppress": always_escalate_page_suppress,
    "all_anomalies": all_anomalies,
    "cite_all": cite_all,
    "always_call_first_tool": always_call_first_tool,
    "never_call": never_call,
    "copy_last_event": copy_last_event,
    "trust_untrusted_text": trust_untrusted_text,
    "selective_non_delivery": selective_non_delivery,
}

assert len(ALL_POLICIES) == 11, "R7 requires exactly 11 named null policies"


def paired_consistency_check(corpus: dict, policy) -> dict:
    """AC7: exposes a policy that gives the SAME disposition to both members of
    a base/counterfactual pair (a class/vocabulary shortcut) -- distinct from
    the overall fail-rate ceiling, since a policy could clear the ceiling on
    unrelated cases while still collapsing every pair to one answer."""
    by_pair: dict[str, list[dict]] = {}
    for case in corpus["cases"]:
        if case["variant"] not in ("base", "counterfactual"):
            continue
        by_pair.setdefault(case["logical_pair_id"], []).append(case)

    collapsed_pairs = []
    for pair_id, members in by_pair.items():
        if len(members) != 2:
            continue
        dispositions = {policy(m)["disposition"] for m in members}
        if len(dispositions) == 1:
            collapsed_pairs.append(pair_id)

    total_pairs = len(by_pair)
    return {
        "total_pairs": total_pairs,
        "collapsed_pairs": collapsed_pairs,
        "collapsed_count": len(collapsed_pairs),
        "exposes_shortcut": len(collapsed_pairs) > 0,
    }


def run_null_policy_suite(corpus: dict, scorer_module) -> dict:
    """Runs every null policy against the whole corpus, offline (LE1/LE3 --
    never loads a model). Returns per-policy fail rate + whether it crossed
    the anti-shortcut ceiling + which specific cases it hard-failed on."""
    results = {}
    for name, policy in ALL_POLICIES.items():
        responses = {case["case_id"]: policy(case) for case in corpus["cases"]}
        scored = scorer_module.score_corpus(corpus, responses)
        total = len(scored["rows"])
        failed = [r["case_id"] for r in scored["rows"] if r["verdict"] == "fail"]
        fail_rate = len(failed) / total if total else 0.0
        results[name] = {
            "fail_rate": fail_rate,
            "crosses_ceiling": fail_rate >= ANTI_SHORTCUT_FAIL_CEILING,
            "failed_case_ids": failed,
            "discriminating_hard_failure": len(failed) > 0,
            "paired_consistency": paired_consistency_check(corpus, policy),
        }
    return results
