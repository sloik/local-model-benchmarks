#!/usr/bin/env python3
"""Unit tests for A2 — Atomic Causal Re-ask Detector.

Spec: specs/SPEC-B02-router-detector-rate.md §A2 (AC-A2-1..5)

ALL model calls are MOCKED. A fake `ask_fn` returns canned (content, err) tuples, so no
network / model is touched. The tests exercise the STRUCTURAL verdict directly: verbatim
grounding, negation guard, both-terms, effect-substitution defence, unscorable handling, and
the AC5-style null-detector separation.
"""
import pathlib
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

from causal_reask_detector import detect  # noqa: E402
# Imported ONLY to prove the verdict diverges from the lexical path — NOT used by detect.
from score_cat06 import is_refusal  # noqa: E402

MODEL = "test-model"


def stub(content, err=None):
    """A fake ask_fn returning a fixed (content, err). Matches ask(model, messages, api_key,
    max_tokens=..., timeout=...)."""
    def _fn(model, messages, api_key, max_tokens=8192, timeout=900):
        return content, err
    return _fn


# --- AC-A2-2: negated sentence containing BOTH terms -> asserts_causation=False -------------
def test_negated_span_with_both_terms_is_false():
    source = "Wzrost kosztow nie spowodowal spadku prognozy EPS w tym kwartale."
    reply = "SPAN: Wzrost kosztow nie spowodowal spadku prognozy EPS w tym kwartale."
    res = detect(source, effect="spadek prognozy EPS", cause="wzrost kosztow",
                 model=MODEL, ask_fn=stub(reply))
    assert res["asserts_causation"] is False
    assert "negation" in res["reason"]


# --- AC-A2-3: span not verbatim in source -> rejected --------------------------------------
def test_non_verbatim_span_is_rejected():
    source = "Marza spadla wskutek wyzszych kosztow energii."
    # A fluent, real-looking, but INVENTED sentence — not present in the source.
    reply = "SPAN: Wzrost kosztow spowodowal spadek prognozy EPS."
    res = detect(source, effect="spadek prognozy EPS", cause="wzrost kosztow",
                 model=MODEL, ask_fn=stub(reply))
    assert res["asserts_causation"] is False
    assert "verbatim" in res["reason"]


# --- A true assertion: verbatim + both terms + no negation -> True --------------------------
def test_true_assertion():
    source = "Zdaniem zarzadu, wzrost kosztow operacyjnych spowodowal spadek prognozy EPS."
    reply = ("SPAN: Zdaniem zarzadu, wzrost kosztow operacyjnych spowodowal "
             "spadek prognozy EPS.")
    res = detect(source, effect="spadek prognozy EPS", cause="wzrost kosztow",
                 model=MODEL, ask_fn=stub(reply))
    assert res["asserts_causation"] is True
    assert res["span"] and "wzrost kosztow" in res["span"]


# --- The DIA case: cite the negated "nie opublikowal prognoz" sentence -> False -------------
def test_dia_fixture_would_have_caught_dia():
    # The retained DIA source sentence (LEADERBOARD.md:151-shaped): contains the key noun
    # 'prognoz' but inside a negation. A lexical co-occurrence gate greenlights the fabrication;
    # the detector must return False.
    source = "Zarzad Spolki nie opublikowal prognoz finansowych na rok 2026."
    reply = "SPAN: Zarzad Spolki nie opublikowal prognoz finansowych na rok 2026."
    res = detect(source, effect="spadek prognozy EPS", cause="wzrost kosztow",
                 model=MODEL, ask_fn=stub(reply))
    assert res["asserts_causation"] is False


# --- Effect-substitution guard: model swaps the absent effect for a source-present one -------
def test_effect_substitution_does_not_pass():
    # Source asserts a DIFFERENT causal pair (spadek marzy <- wzrost kosztow). The model cites
    # that real sentence, but the CALLER asked about "spadek prognozy EPS". Because effect/cause
    # are the caller's terms (not read from the model), the both-terms gate fails: the span
    # carries no "prognoz*"/EPS effect. So the substitution cannot pass.
    source = "Spadek marzy wynika z wzrostu kosztow operacyjnych."
    reply = "SPAN: Spadek marzy wynika z wzrostu kosztow operacyjnych."
    res = detect(source, effect="spadek prognozy EPS", cause="wzrost kosztow",
                 model=MODEL, ask_fn=stub(reply))
    assert res["asserts_causation"] is False
    assert "both" in res["reason"]


# --- BRAK / declined -> False (no span) -----------------------------------------------------
def test_brak_is_false():
    source = "Zarzad nie opublikowal prognoz finansowych."
    res = detect(source, effect="spadek prognozy EPS", cause="wzrost kosztow",
                 model=MODEL, ask_fn=stub("BRAK"))
    assert res["asserts_causation"] is False
    assert res["span"] is None


# --- EMPTY_ALL_REASONING -> UNSCORABLE (None), never a refusal/False -------------------------
def test_empty_all_reasoning_is_unscorable():
    res = detect("dowolne zrodlo", effect="X", cause="Y", model=MODEL,
                 ask_fn=stub(None, "EMPTY_ALL_REASONING(reasoning_tokens=8192)"))
    assert res["asserts_causation"] is None
    assert "EMPTY_ALL_REASONING" in res["reason"]


# --- Verdict is STRUCTURAL, not lexical -----------------------------------------------------
def test_verdict_is_structural_not_lexical():
    # The model's PROSE contains a refusal keyword ("brak danych") that is_refusal() fires on,
    # yet it also cites a valid, verbatim, asserting span. A lexical verdict would call this a
    # refusal (False); the structural detector correctly returns True.
    source = "Wzrost kosztow operacyjnych spowodowal spadek prognozy EPS."
    reply = ("Na pierwszy rzut oka wyglada to jak brak danych, ale jednak zrodlo to twierdzi.\n"
             "SPAN: Wzrost kosztow operacyjnych spowodowal spadek prognozy EPS.")
    assert is_refusal(reply) is True  # lexical path would mis-score this as a refusal
    res = detect(source, effect="spadek prognozy EPS", cause="wzrost kosztow",
                 model=MODEL, ask_fn=stub(reply))
    assert res["asserts_causation"] is True


# --- AC-A2-5: null detectors must NOT match the real detector's accuracy --------------------
# A labelled, MIXED fixture. Each item carries a ground-truth label and the reply a realistic
# model would give. The real detector reads the reply structurally; the two null "detectors"
# ignore it and emit a constant. The test fails if `detect` ever collapses to a constant.
_FIXTURE = [
    # (source, effect, cause, model_reply, label)
    ("Wzrost kosztow operacyjnych spowodowal spadek prognozy EPS w tym kwartale.",
     "spadek prognozy EPS", "wzrost kosztow",
     "SPAN: Wzrost kosztow operacyjnych spowodowal spadek prognozy EPS w tym kwartale.",
     True),
    ("Wzrost kosztow nie spowodowal spadku prognozy EPS.",
     "spadek prognozy EPS", "wzrost kosztow",
     "SPAN: Wzrost kosztow nie spowodowal spadku prognozy EPS.",
     False),
    ("Marza spadla wskutek wyzszych kosztow energii.",
     "spadek prognozy EPS", "wzrost kosztow",
     "SPAN: Wzrost kosztow spowodowal spadek prognozy EPS.",  # not verbatim
     False),
    ("Zarzad nie opublikowal prognoz finansowych.",
     "spadek prognozy EPS", "wzrost kosztow",
     "BRAK",
     False),
    ("Zdaniem spolki, wzrost kosztow energii wywolal spadek prognozy EPS.",
     "spadek prognozy EPS", "wzrost kosztow",
     "SPAN: Zdaniem spolki, wzrost kosztow energii wywolal spadek prognozy EPS.",
     True),
]


def _accuracy(preds, labels):
    return sum(1 for p, l in zip(preds, labels) if p == l) / len(labels)


def test_null_detectors_do_not_match_real_detector():
    labels = [item[4] for item in _FIXTURE]

    real = []
    for source, effect, cause, reply, _ in _FIXTURE:
        res = detect(source, effect, cause, MODEL, ask_fn=stub(reply))
        real.append(res["asserts_causation"])

    real_acc = _accuracy(real, labels)
    always_asserts_acc = _accuracy([True] * len(labels), labels)   # constant "yes"
    always_refuses_acc = _accuracy([False] * len(labels), labels)  # constant BRAK

    # The real detector must SEPARATE the nulls: strictly beat both constants. This assertion
    # would FAIL if `detect` collapsed to a constant output (it would then equal one null and
    # be beaten-or-tied by the other).
    assert real_acc > always_asserts_acc
    assert real_acc > always_refuses_acc
    assert real_acc == 1.0  # gets every mixed-label item right on this fixture
