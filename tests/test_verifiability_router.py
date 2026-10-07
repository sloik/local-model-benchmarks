"""Tests for the Verifiability Router (A1). Pure logic — NO model calls anywhere.

Spec ACs: AC-A1-1..5 in specs/SPEC-B02-router-detector-rate.md.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from verifiability_router import route, mis_tag_audit, V0_WHITELIST  # noqa: E402


# --- AC-A1-1: shape -----------------------------------------------------------------------------

def test_route_returns_full_shape():
    r = route("Ile wyniosly przychody w Q1'26?", "przychody")
    assert set(r) == {"tier", "route", "reason", "cost"}
    assert r["tier"] in {"V0", "V1", "V2"}
    assert r["route"] in {"local", "frontier"}
    assert isinstance(r["reason"], str) and r["reason"]
    assert set(r["cost"]) == {"local_usd", "frontier_usd"}


# --- AC-A1-5: the three canonical DIA cases -----------------------------------------------------

def test_dia_attribution_question_routes_v2():
    # DIA §5 attribution question (inflected "tlumacza") must route V2/frontier.
    r = route("Czy wyniki Q1'26 tlumacza spadek prognozy EPS?")
    assert r["tier"] == "V2"
    assert r["route"] == "frontier"


def test_dia_attribution_question_with_accents_routes_v2():
    # Same question with Polish diacritics — must fold and still route V2.
    r = route("Czy wyniki Q1'26 tłumaczą spadek prognozy EPS?")
    assert r["tier"] == "V2"


def test_ex_item_recurring_ebitda_routes_v2():
    # The +4,3 mln ex-item question ("po wyłączeniu") is an adjustment basis -> V2.
    r = route("Recurring EBITDA po wyłączeniu przeszacowania?")
    assert r["tier"] == "V2"
    assert r["route"] == "frontier"


def test_clean_extraction_routes_v0():
    r = route("Ile wyniosły przychody w Q1'26?", field="przychody")
    assert r["tier"] == "V0"
    assert r["route"] == "local"


# --- AC-A1-2: whitelist gate --------------------------------------------------------------------

def test_field_not_on_whitelist_routes_v2():
    r = route("Ile wyniosla marza brutto w Q1'26?", field="marza_brutto")
    assert r["tier"] == "V2"
    assert r["route"] == "frontier"


def test_no_field_never_v0():
    r = route("Ile wyniosly przychody w Q1'26?")
    assert r["tier"] == "V2"


def test_field_case_and_diacritics_normalised_to_v0():
    r = route("Ile wyniosl dlug netto?", field="  DŁUG_NETTO ")
    assert r["tier"] == "V0"


# --- AC-A1-3: causal lexicon overrides a whitelisted field --------------------------------------

def test_causal_lexicon_overrides_whitelisted_field():
    # Whitelisted field, but a causal term in the question -> V2, never V0.
    r = route("Dlaczego wzrosly przychody w Q1'26?", field="przychody")
    assert r["tier"] == "V2"
    assert r["route"] == "frontier"


def test_ex_item_term_overrides_whitelisted_recurring_ebitda():
    r = route("Recurring EBITDA po wyłączeniu przeszacowania?", field="recurring_ebitda")
    assert r["tier"] == "V2"


@pytest.mark.parametrize("q", [
    "Co tłumaczy spadek?",
    "Co wyjaśnia ten wynik?",
    "Jaka jest przyczyna spadku?",
    "Jaki jest mechanizm?",
    "Wynik oczyszczony o zdarzenia?",
    "Zysk jednorazowy w kwartale?",
    "This is a one-off item?",
    "Jaki wpływ na wynik?",
])
def test_each_causal_stem_routes_v2_even_with_whitelisted_field(q):
    assert route(q, field="przychody")["tier"] == "V2"


def test_ascii_folded_causal_term_still_matches():
    # Transliterated source (no diacritics) must still hit the lexicon.
    assert route("Czy raport tlumaczy spadek prognozy?", field="przychody")["tier"] == "V2"


# --- AC-A1-4: mis-tag audit, catastrophic mode counted separately -------------------------------

def test_mis_tag_audit_counts_hidden_v2_as_v0_separately():
    labelled = [
        {"tier": "V0", "gold_tier": "V2"},  # catastrophic: hidden V2 tagged V0
        {"tier": "V2", "gold_tier": "V0"},  # wasteful over-escalation
        {"tier": "V0", "gold_tier": "V0"},  # agree
        {"tier": "V2", "gold_tier": "V2"},  # agree
    ]
    audit = mis_tag_audit(labelled)
    assert audit["n"] == 4
    assert audit["disagreements"] == 2
    assert audit["disagreement_rate"] == 0.5
    assert audit["hidden_v2_as_v0"] == 1        # counted in its own bucket
    assert audit["routed_down"] == 1
    assert audit["routed_up"] == 1


def test_mis_tag_audit_derives_tier_from_question():
    labelled = [
        # router tags this V0; gold says V2 -> catastrophic, computed via route()
        {"question": "Ile wyniosly przychody?", "field": "przychody", "gold_tier": "V2"},
        {"question": "Dlaczego spadek?", "gold_tier": "V2"},  # router V2, agrees
    ]
    audit = mis_tag_audit(labelled)
    assert audit["hidden_v2_as_v0"] == 1
    assert audit["disagreements"] == 1


def test_mis_tag_audit_empty_is_safe():
    audit = mis_tag_audit([])
    assert audit["n"] == 0
    assert audit["disagreement_rate"] == 0.0
    assert audit["hidden_v2_as_v0"] == 0


# --- cost is a readout, never a routing input ---------------------------------------------------

def test_cost_is_static_readout():
    local = route("Ile wyniosly przychody?", "przychody")
    frontier = route("Dlaczego spadek?", "przychody")
    # identical static cost regardless of which way the decision went
    assert local["cost"] == frontier["cost"] == {"local_usd": 0.0, "frontier_usd": 1.40}


# --- invalid-input tests (DevKB python.md: small CLIs need invalid-input tests) ------------------

def test_empty_question_and_none_field_routes_up_safely():
    # Empty question, None field: no crash, and the safety default routes UP to V2.
    r = route("", None)
    assert r["tier"] == "V2"
    assert r["route"] == "frontier"


def test_non_string_question_raises():
    with pytest.raises(ValueError):
        route(None)


def test_non_string_field_raises():
    with pytest.raises(ValueError):
        route("Ile przychody?", field=123)


def test_mis_tag_audit_missing_gold_tier_raises():
    with pytest.raises(ValueError):
        mis_tag_audit([{"tier": "V0"}])


# --- CLI ----------------------------------------------------------------------------------------

def test_cli_prints_routing_json(capsys):
    import json
    from verifiability_router import main
    rc = main(["--question", "Czy wyniki Q1'26 tłumaczą spadek prognozy EPS?"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["tier"] == "V2"
    assert out["route"] == "frontier"


def test_whitelist_is_nonempty_closed_set():
    assert isinstance(V0_WHITELIST, set)
    assert len(V0_WHITELIST) >= 9
