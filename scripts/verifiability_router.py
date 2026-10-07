#!/usr/bin/env python3
"""Verifiability Router (A1) — tag an atomic sub-question V0 / V1 / V2 and route it.

Spec: specs/SPEC-B02-router-detector-rate.md  (A1 section, AC-A1-1..5)
Plan: Improvement-Plan-Agent-Panel-2026-07-17.md  (item A1; folds in rejected C1's cost readout)
Findings: Local-Model-Reliability-Findings.md §6 (route by task, not by leaderboard rank)

WHAT THIS ENCODES (and what it deliberately does NOT do)
    This is PLUMBING that encodes the CORE routing rule. It does not *detect* the invisible
    unwarranted-attribution failure (~9%, correct numbers, invented causation) — a figure check
    is structurally blind to it. It *refuses to expose* a local model to that class by routing
    every judgement-shaped question to the frontier.

    - V0 -- answerable by grep/Python/schema against a clean labelled table. Field must be on a
            closed whitelist of pre-declared atomic fields. Local admissible, verified free.
    - V1 -- one claim vs one clean-table cell (Apple-FM-checkable presence/lookup). The plan's
            unresolved middle (adjustment-aware extraction). NO proven V1 verifier exists, so
            per plan §"The disagreement" and findings §6 the router never emits V1 on its own:
            the V1 middle DEFAULTS TO V2. We do not pretend a mechanism exists there.
    - V2 -- any "does the source support / explain / cause" judgement, OR anything touching a
            filing's numbers where a one-off / adjustment / causal term could appear. Frontier.

SAFETY PROPERTY (load-bearing — do not weaken)
    Anything not PROVABLY V0 routes UP to V2. Ambiguity routes up, never down. V0 is the narrow
    provable case, never the default. This is what keeps the relocation of invisibility (to
    authoring-time tagging) from being a silent hole.

    `cost` is a STATIC READOUT folded in from the rejected C1 proposal. It is emitted for audit
    only and is NEVER an input to the routing decision.

Usage:
    verifiability_router.py --question "Ile wyniosly przychody w Q1'26?" --field przychody
    verifiability_router.py --question "Czy wyniki Q1'26 tlumacza spadek prognozy EPS?"
"""
import argparse
import json
import sys

# Closed whitelist of pre-declared atomic fields. A field NOT on this set can never be V0
# (AC-A1-2). Extend it only with fields that have a mechanical check against a clean labelled
# table -- "looks atomic" is not enough.
V0_WHITELIST = {
    # DIA-origin (presentation-chart line items)
    "przychody", "wolumen_badan", "zysk_netto_akcjonariuszom", "recurring_ebitda",
    "dlug_netto", "dywidenda_na_akcje", "cfo", "fcf", "capex", "ebitda",
    # universal financial-statement line items (added 2026-07-17 from the CRI dogfood — the DIA-only
    # set routed clean extractions like cash/equity/total-assets to frontier, defeating the cost
    # saving; these are single-number statement lines verifiable against a clean table). The causal
    # lexicon still forces any "why/how/explain" question to V2 regardless of field, so widening the
    # whitelist cannot mis-route an attribution question.
    "zysk_netto", "strata_netto", "zysk_operacyjny", "strata_operacyjna",
    "srodki_pieniezne", "kapital_wlasny", "kapital_podstawowy", "aktywa_razem",
    "suma_bilansowa", "zobowiazania", "zobowiazania_dlugoterminowe", "zobowiazania_krotkoterminowe",
    "naleznosci", "zapasy", "przeplywy_operacyjne", "przeplywy_inwestycyjne", "przeplywy_finansowe",
    "dotacje", "eps", "marza_ebitda", "marza_netto",
}

# Causal / adjustment lexicon. Any match -> V2 regardless of field (AC-A1-3): these words mark a
# "does the source explain / cause" judgement or an adjustment basis (one-off) where the invisible
# fabrication lives. Stored as diacritic-FOLDED stems so a single entry catches (a) the inflected
# Polish form ("tlumaczy" / "tlumacza" / "tlumaczyc"), (b) the accented source ("tlumaczy") and
# (c) its ASCII-transliterated copy -- transliteration is why we match on folded stems, not on the
# literal accented word. The spec lexicon each stem derives from is in the trailing comment.
_CAUSAL_STEMS = (
    "tlumacz",     # tlumaczy / tlumacza  (explains)
    "wyjasni",     # wyjasnia / wyjasniaja (explains)
    "dlaczego",    # why
    "przyczyn",    # przyczyna / przyczyny (cause)
    "mechanizm",   # mechanism
    "wylaczen",    # po wylaczeniu (after excluding)
    "oczyszczon",  # oczyszczony (adjusted / cleaned)
    "jednorazow",  # jednorazowy (one-off)
    "one-off",
    "wplyw",       # wplyw / wplywa (impact)
)

# Fold Polish diacritics to their ASCII base so accented and transliterated text match identically.
_PL_FOLD = str.maketrans({
    "ą": "a", "ć": "c", "ę": "e", "ł": "l", "ń": "n",
    "ó": "o", "ś": "s", "ż": "z", "ź": "z",
    "Ą": "a", "Ć": "c", "Ę": "e", "Ł": "l", "Ń": "n",
    "Ó": "o", "Ś": "s", "Ż": "z", "Ź": "z",
})

# Static cost readout (rejected C1, folded in). Placeholder rates; a readout, never a routing input.
_LOCAL_USD = 0.0
_FRONTIER_USD = 1.40

# Tier risk order: V0 (local, riskiest if mis-tagged) < V1 < V2 (frontier, safest).
_TIER_RANK = {"V0": 0, "V1": 1, "V2": 2}


def _fold(text):
    return text.translate(_PL_FOLD).lower()


def _causal_hit(question):
    """Return the first causal/adjustment stem found in `question`, or None."""
    folded = _fold(question)
    for stem in _CAUSAL_STEMS:
        if stem in folded:
            return stem
    return None


def _cost():
    return {"local_usd": _LOCAL_USD, "frontier_usd": _FRONTIER_USD}


def route(question, field=None):
    """Tag an atomic sub-question V0/V1/V2 and route it local or frontier.

    Returns {"tier", "route", "reason", "cost"}. `cost` is an audit readout only.
    """
    if not isinstance(question, str):
        raise ValueError("question must be a string, got %r" % type(question).__name__)
    if field is not None and not isinstance(field, str):
        raise ValueError("field must be a string or None, got %r" % type(field).__name__)

    # 1. Causal/adjustment lexicon overrides everything, including a whitelisted field (AC-A1-3).
    hit = _causal_hit(question)
    if hit:
        return {
            "tier": "V2",
            "route": "frontier",
            "reason": "causal/adjustment term '%s' -> V2; source-support judgement never routes local" % hit,
            "cost": _cost(),
        }

    # 2. V0 iff the field is on the closed whitelist (AC-A1-2). Normalise case/diacritics/whitespace.
    if field is not None and _fold(field.strip()) in V0_WHITELIST:
        return {
            "tier": "V0",
            "route": "local",
            "reason": "field '%s' on V0_WHITELIST and no causal term -> V0; verifiable free against clean table" % field,
            "cost": _cost(),
        }

    # 3. SAFETY DEFAULT: not provably V0 -> V2. Ambiguity routes UP, never down. The V1 middle has
    #    no proven verifier (plan §disagreement, findings §6), so it collapses here too.
    if field is None:
        why = "no field supplied"
    else:
        why = "field '%s' not on V0_WHITELIST" % field
    return {
        "tier": "V2",
        "route": "frontier",
        "reason": "not provably V0 (%s) -> V2; ambiguity routes up" % why,
        "cost": _cost(),
    }


def mis_tag_audit(labelled):
    """Audit router tags against a human/frontier re-tag (AC-A1-4).

    Each item needs a `gold_tier` (the trusted re-tag) and either a `tier` (the router's tag) or a
    `question` (+ optional `field`) from which the router tag is computed. Reports the disagreement
    rate SPLIT BY DIRECTION, and counts the one catastrophic mode -- a hidden-V2 item tagged V0 --
    SEPARATELY. That single subset is the silent hole the whitelist+default-V2 exists to bound;
    a bare disagreement rate would let it hide inside the total.
    """
    n = len(labelled)
    disagreements = 0
    routed_down = 0        # tagged a LOWER tier than gold -> under-escalated -> risky
    routed_up = 0          # tagged a HIGHER tier than gold -> over-escalated -> wasteful
    hidden_v2_as_v0 = 0    # gold V2 tagged V0 -> the one catastrophic mode

    for item in labelled:
        if "gold_tier" not in item:
            raise ValueError("each item needs a 'gold_tier': %r" % item)
        gold = item["gold_tier"]
        if "tier" in item:
            pred = item["tier"]
        elif "question" in item:
            pred = route(item["question"], item.get("field"))["tier"]
        else:
            raise ValueError("each item needs a 'tier' or a 'question': %r" % item)
        if pred not in _TIER_RANK or gold not in _TIER_RANK:
            raise ValueError("tiers must be V0/V1/V2: pred=%r gold=%r" % (pred, gold))

        if pred == gold:
            continue
        disagreements += 1
        if _TIER_RANK[pred] < _TIER_RANK[gold]:
            routed_down += 1
            if gold == "V2" and pred == "V0":
                hidden_v2_as_v0 += 1
        else:
            routed_up += 1

    return {
        "n": n,
        "disagreements": disagreements,
        "disagreement_rate": round(disagreements / n, 4) if n else 0.0,
        "routed_down": routed_down,   # risky: local when frontier was warranted
        "routed_up": routed_up,       # wasteful: frontier when local would have sufficed
        "hidden_v2_as_v0": hidden_v2_as_v0,  # CATASTROPHIC: counted on its own
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verifiability Router (A1)")
    ap.add_argument("--question", required=True, help="the atomic sub-question text")
    ap.add_argument("--field", default=None, help="atomic field name (for the V0 whitelist check)")
    a = ap.parse_args(argv)
    print(json.dumps(route(a.question, a.field), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
