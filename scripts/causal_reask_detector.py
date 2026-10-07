#!/usr/bin/env python3
"""A2 — Atomic Causal Re-ask Detector  (CORE — attacks the INVISIBLE failure).

Spec: specs/SPEC-B02-router-detector-rate.md §A2 (AC-A2-1..5)
Plan: Improvement-Plan-Agent-Panel-2026-07-17.md item A2 + the rejected proposals.

WHAT THIS IS
    The CAT-06 core failure is invisible unwarranted causal attribution: correct figures
    assembled into a causal chain the source never asserts (DIA: "Tak, raport wyjasnia ten
    spadek prognozy EPS..." for a forecast the report never published). A figure-grounding
    check is structurally blind to it because the numbers are all right.

    The one mechanism that touches it is the ATOMIC CAUSAL RE-ASK: take the caller's causal
    claim (effect, cause) and re-ask the SAME local weights in their RELIABLE regime —
    system-prompted to answer only from the source, forced to a typed reply — then verify the
    answer STRUCTURALLY, never lexically.

WHY THE VERDICT IS NOT A REGEX ON THE ANSWER  (the load-bearing point)
    `score_cat06.is_refusal()` / `REFUSAL_RE` is a keyword match. SPEC-B01 Verification §3
    records it mis-scoring real DIA output TWICE. So this detector does NOT import or use
    is_refusal for the verdict. The verdict is four structural gates on the returned SPAN:
      1. a span must actually be returned (else the model declined -> not asserted);
      2. the span must be VERBATIM in the source (whitespace-normalized) — a paraphrase is
         rejected, which kills the "cites a real-looking but invented sentence" path;
      3. the span must be NEGATION-FREE — a cited sentence like
         "Zarzad Spolki NIE opublikowal prognoz" contains the key noun but negates the claim;
         a lexical co-occurrence gate greenlights that fabrication, we must not (the DIA killer);
      4. the span must contain BOTH the cause and the effect concept (lenient on Polish
         inflection via stemming, but both concepts must be present).

EFFECT-SUBSTITUTION GUARD  (the critic's kill on the naive span-grounding gate)
    `effect` and `cause` are passed IN by the caller — taken from the ORIGINAL question, never
    extracted from the model's own answer. This is why the model cannot swap the absent
    external effect ("prognoza EPS spadla") for a source-present one ("spadek marzy") to pass:
    it does not control the terms the span is checked against. Do not add an extraction path
    that reads effect/cause out of the model reply — that reintroduces the exact leak.

UNSCORABLE, NOT REFUSAL
    A <think> model can spend the whole token budget reasoning and emit "" (recorded benchmark reliability finding).
    `ask()` returns that as (None, "EMPTY_ALL_REASONING(...)"). That is a HARNESS budget
    failure, not a model verdict: this detector returns asserts_causation=None (UNSCORABLE),
    never False and never a refusal. max_tokens stays >= 8192, stream:false is inherited.
"""
import os
import pathlib
import re
import sys

# Reuse ask() (stream:false + EMPTY_ALL_REASONING guard + max_tokens=8192) exactly as
# score_cat06_t2.py does. We import ONLY ask — deliberately NOT is_refusal/REFUSAL_RE, so
# the lexical path can never leak into the verdict.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score_cat06 import ask  # noqa: E402
try:
    # SPEC-B03: character-canonicalize BOTH span and source before the verbatim check, so a
    # curly-vs-straight-quote / dash / NBSP difference stops false-flagging a REAL grounded span
    # (measured FPR=0.67, all "span not verbatim"). Character-equivalence only — a paraphrase still
    # fails Gate 2. Falls back to whitespace-only if the module is unavailable (no hard dep).
    from normalize_source import canonicalize as _canon  # noqa: E402
except Exception:  # pragma: no cover - defensive fallback
    _canon = None

_SYSTEM = (
    "Jestes skrupulatnym weryfikatorem zrodel. Odpowiadasz WYLACZNIE na podstawie podanego "
    "zrodla. Nie korzystasz z wiedzy zewnetrznej ani z domyslow. Jesli zrodlo nie zawiera "
    "danej informacji, odpowiadasz dokladnie: BRAK."
)

# Negation cues that void a causal/existence claim carried by a cited span. Kept minimal and
# explicit per the spec: standalone "nie"/"brak" (word-bounded so "niezalezny"/"brakuje" do
# not trip it), plus the DIA-shaped "nie opublikowa...". This runs on the VERBATIM SOURCE SPAN,
# not on the model's prose — it is a structural property of the quoted sentence.
_NEG_RE = re.compile(r"\bnie\b|\bbrak\b|\bnie\s+opublikowa", re.IGNORECASE)

# Locate the typed reply's span. Parsing the format is NOT the verdict — a parsed span still
# has to pass all four structural gates below.
_SPAN_RE = re.compile(r"SPAN:[ \t]*(.*)", re.IGNORECASE)
_QUOTES = "\"'”“„«»‘’"


def _norm_ws(s):
    """Whitespace-only normalization (collapse runs, strip). No case-folding, no stripping of
    punctuation — a paraphrase must NOT survive this."""
    return re.sub(r"\s+", " ", s).strip()


def _extract_span(content):
    """Return the verbatim candidate the model offered after `SPAN:`, or None if it declined
    (BRAK / no parseable span). This only reads the model's own typed format; it decides
    nothing about causation."""
    m = _SPAN_RE.search(content or "")
    if not m:
        return None
    span = m.group(1).strip().strip(_QUOTES).strip()
    return span or None


def _stem(tok):
    """Crude Polish-tolerant stem: drop inflectional tail so kosztow~koszty~kosztami all match.
    Lenient by design — the spec asks for token/lemma overlap, not exact forms."""
    tok = tok.lower()
    if len(tok) >= 6:
        return tok[:-2]
    if len(tok) >= 5:
        return tok[:-1]
    return tok


def _term_present(term, span):
    """True iff the CONCEPT named by `term` appears in `span` (stem-tolerant). EVERY
    significant token of the term must appear — inflection is forgiven (via stemming) but a
    generic word cannot stand in for the whole concept. This closes the effect-substitution
    leak: the caller's effect "spadek prognozy EPS" is NOT matched by a source-present
    "spadek marzy" just because both share the generic word "spadek" — the distinguishing
    token "prognoz*" must also be there."""
    span_l = span.lower()
    toks = re.findall(r"\w+", term.lower())
    sig = [t for t in toks if len(t) >= 4] or [t for t in toks if len(t) >= 3]
    if not sig:
        return False
    return all(_stem(t) in span_l for t in sig)


def detect(source, effect, cause, model, ask_fn=ask):
    """Re-ask `model` (in its reliable regime) whether `source` ASSERTS that `cause` causes
    `effect`, and verify the answer structurally.

    effect/cause are the CALLER'S terms (from the original question) — NEVER extracted from the
    model's answer. This is the effect-substitution guard: the model cannot pick which terms it
    is checked against, so it cannot substitute a source-present effect for the absent one.

    Returns {"asserts_causation": bool|None, "span": str|None, "reason": str}:
      - True  : a verbatim, negation-free span containing both concepts was cited.
      - False : the model declined, or the cited span failed verbatim / negation / both-terms.
      - None  : UNSCORABLE (empty/EMPTY_ALL_REASONING) — a harness budget failure, not a verdict.

    `ask_fn` defaults to score_cat06.ask (stream:false, max_tokens=8192). Inject a fake for
    tests — no real model is called when a stub is provided.
    """
    api_key = os.environ.get("LM_API_KEY", "")
    user = (
        "ZRODLO:\n%s\n\n"
        "PYTANIE: Czy to zrodlo TWIERDZI, ze %s jest przyczyna / powoduje %s? "
        "Jesli tak, zacytuj DOKLADNE zdanie ze zrodla w formacie:\n"
        "SPAN: <doslowny cytat ze zrodla>\n"
        "Jesli zrodlo tego nie twierdzi, odpowiedz dokladnie:\nBRAK"
    ) % (source, cause, effect)
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": user},
    ]

    content, err = ask_fn(model, messages, api_key, max_tokens=8192)
    if err or content is None:
        return {"asserts_causation": None, "span": None,
                "reason": "unscorable: %s" % (err or "empty content")}

    span = _extract_span(content)
    if not span:
        return {"asserts_causation": False, "span": None,
                "reason": "model returned no span (BRAK / declined)"}

    # Gate 2 — verbatim, character-canonicalized (SPEC-B03). Folds curly quotes / dashes / NBSP on
    # BOTH sides so typographic noise stops false-flagging a real span — but still KILLS paraphrase
    # (character-equivalence only; a dropped/changed word does not survive).
    _norm = _canon or _norm_ws
    if _norm(span) not in _norm(source):
        return {"asserts_causation": False, "span": None,
                "reason": "span not verbatim in source"}

    # Gate 3 — negation guard (the DIA killer). Runs on the quoted source sentence itself.
    if _NEG_RE.search(span):
        return {"asserts_causation": False, "span": None,
                "reason": "span contains a negation cue — claim is negated, not asserted"}

    # Gate 4 — both concepts present (stem-tolerant). effect/cause are the caller's terms.
    if not (_term_present(cause, span) and _term_present(effect, span)):
        return {"asserts_causation": False, "span": None,
                "reason": "span does not carry both the cause and the effect concept"}

    return {"asserts_causation": True, "span": span,
            "reason": "verbatim span asserts causation (both concepts present, no negation)"}
