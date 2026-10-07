"""Tests for SPEC-B03 evidence normalization. No model calls."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))
from normalize_source import canonicalize, normalize_report, normalize_structure  # noqa: E402
from causal_reask_detector import detect  # noqa: E402


# --- AC1: character folding + whitespace collapse, idempotent ---
def test_curly_quotes_folded():
    assert canonicalize("Q1’26 i „cytat”") == "Q1'26 i \"cytat\""


def test_dashes_folded():
    assert canonicalize("a–b—c−d") == "a-b-c-d"


def test_nbsp_and_whitespace_collapsed():
    assert canonicalize("1 070,3 mln   zł\n\n") == "1 070,3 mln zł"


def test_idempotent():
    s = "Q1’26 — 1 070,3 mln"
    assert canonicalize(canonicalize(s)) == canonicalize(s)


# --- AC2 (MANDATORY): a paraphrase must STILL fail the verbatim gate after canonicalization ---
def test_paraphrase_still_fails():
    # source and span differ by ONE dropped word — canonicalization must NOT rescue it.
    source = "Spadek marży Recurring EBITDA wynika głównie z większej dynamiki kosztów."
    paraphrase = "Spadek marży wynika głównie z większej dynamiki kosztów."  # 'Recurring EBITDA' dropped
    assert canonicalize(paraphrase) not in canonicalize(source)


def test_paraphrase_changed_word_still_fails():
    source = "Spadek marży wynika z wzrostu kosztów operacyjnych."
    changed = "Spadek marży wynika z spadku kosztów operacyjnych."  # wzrostu -> spadku
    assert canonicalize(changed) not in canonicalize(source)


# --- AC3: a curly-quote-only difference now MATCHES (the FPR fix) ---
def test_curly_quote_diff_now_matches():
    source_mangled = "Dynamika przychodów zgodna z oczekiwaniami w Q1’26."   # curly apostrophe
    span_straight = "Dynamika przychodów zgodna z oczekiwaniami w Q1'26."          # straight
    # before: whitespace-only norm would NOT match (different apostrophe byte)
    assert span_straight not in source_mangled
    # after canonicalize: matches
    assert canonicalize(span_straight) in canonicalize(source_mangled)


def test_nbsp_number_diff_now_matches():
    source = "Dług netto wzrósł do 1 070,3 mln zł."   # NBSP in the number
    span = "Dług netto wzrósł do 1 070,3 mln zł."          # normal space
    assert canonicalize(span) in canonicalize(source)


# --- AC4: canonicalization does not break the detector's gates (uses injected mock ask) ---
def _mk_ask(reply):
    def fake(model, messages, api_key=None, max_tokens=8192, timeout=900):
        return reply, None
    return fake


def test_detector_negation_guard_survives_canonicalization():
    src = normalize_report("Zarząd Spółki nie opublikował prognoz finansowych.")
    ask_fn = _mk_ask("SPAN: Zarząd Spółki nie opublikował prognoz finansowych.")
    r = detect(src, "spadek prognozy EPS", "brak prognoz", "m", ask_fn=ask_fn)
    assert r["asserts_causation"] is False  # negation still caught


def test_detector_true_assertion_with_curly_source():
    # source has curly apostrophe; model quotes with straight — after B03 this now asserts True
    src = normalize_report("Spadek marży w Q1’26 wynika głównie z większej dynamiki kosztów operacyjnych.")
    ask_fn = _mk_ask("SPAN: Spadek marży w Q1'26 wynika głównie z większej dynamiki kosztów operacyjnych.")
    r = detect(src, "spadek marży", "dynamika kosztów operacyjnych", "m", ask_fn=ask_fn)
    assert r["asserts_causation"] is True


# --- B03 residual: colon-bullet causation joining (the working-capital / cost-of-finance shape) ---
def test_colon_bullet_joined_onto_one_line():
    # effect clause ends in a colon; the cause is on the following bullet line (DIA lines ~280-281).
    src = (
        "       + Kapitał obrotowy netto wzrósł w Q1'26 o 45,1 mln rdr, co wynika głównie z:\n"
        "                   • wzrostu poziomu należności o 59,0 mln zł rdr spowodowany wzrostem sprzedaży.\n"
        "                   • wzrostu poziomu zapasów o 17,2 mln zł rdr związanego z wzrostem sprzedaży.\n"
    )
    out = normalize_structure(src)
    # effect and cause now share ONE physical line: no newline between "z:" and the cause.
    joined = [ln for ln in out.split("\n") if "wynika głównie z:" in ln][0]
    assert "wynika głównie z: wzrostu poziomu należności" in joined
    # the bullet glyph is gone from the causal span (a model's clean quote will now match verbatim)
    assert "•" not in joined
    # both bullet causes folded into the same line
    assert "wzrostu poziomu zapasów" in joined


def test_colon_bullet_join_survives_normalize_report():
    # end-to-end: after structure-then-fold, a clean (bullet-free) causal quote is a substring of source.
    src = (
        "Wzrost kosztów finansowych o 1,0 mln rdr wynika głównie z:\n"
        "  • wzrostu odsetek od leasingu o 2,4 mln zł.\n"
    )
    quote = "wynika głównie z: wzrostu odsetek od leasingu"
    # before B03 the bullet glyph "• " sat between "z:" and the cause, so the clean quote missed
    assert canonicalize(quote) not in canonicalize(src)
    assert canonicalize(quote) in normalize_report(src)


# --- B03 residual: chart-number interleaving (the T1 chart trap) ---
def test_chart_interleave_prose_isolated():
    # a bar-chart number run sharing a line with a trailing prose clause (DIA line ~87).
    line = "      224,6          221,2           243,1        6,9        6,6         6,9    jest efektem zmiany struktury sprzedaży poprzez"
    out = normalize_structure(line)
    lines = out.split("\n")
    # the prose clause now stands on its own line, findable WITHOUT the number soup
    prose_lines = [ln for ln in lines if "jest efektem zmiany struktury" in ln]
    assert len(prose_lines) == 1
    assert prose_lines[0].strip() == "jest efektem zmiany struktury sprzedaży poprzez"
    assert not any(ch.isdigit() for ch in prose_lines[0])
    # the numbers are on a different line, not fused with the prose
    assert any("224,6" in ln and "jest efektem" not in ln for ln in lines)


def test_pure_number_row_not_split():
    # a label/number row with no >=3-word prose clause must be left alone (conservative).
    line = "      4,9               B2B                     6,9        6,6          6,9     B2B"
    assert normalize_structure(line) == line


def test_two_column_table_prose_line_not_shredded():
    # DIA line ~218: a table row (left column) sharing the physical line with a right-column causal
    # clause. Words are scattered across the line, so chart-separation must NOT split it (that would
    # shred "...wynika głównie z:"). This two-column interleave is a documented residual, not mangled.
    line = "Koszty finansowe   -16,2   -17,2   -1,0   6,2%   + Wzrost kosztów finansowych o 1,0 mln rdr wynika głównie z:"
    assert normalize_structure(line) == line


def test_ordinary_prose_with_scattered_numbers_untouched():
    # the net-debt sentence: numbers are scattered (never >=3 in a row) — must NOT be split/mangled.
    line = "+ Dług netto wzrósł do poziomu 1070,3 mln zł w wyniku wzrostu zobowiązań z tytułu leasingu (+80,6 mln zł)."
    assert normalize_structure(line) == line


# --- Anti-regression (CRITICAL): two SEPARATE unrelated sentences must NOT be joined ---
def test_two_separate_sentences_not_joined():
    # neither line is a colon+bullet continuation — a false causal span must not be fabricated.
    src = (
        "Spadek marży wynika z wzrostu kosztów operacyjnych.\n"
        "Zysk netto wzrósł o 10% rdr w Q1'26.\n"
    )
    out = normalize_structure(src)
    assert out == src  # unchanged: no colon-bullet, no chart line
    # the two sentences remain on separate lines (not merged into one span)
    assert "kosztów operacyjnych. Zysk netto" not in out


def test_colon_line_without_bullet_not_joined():
    # a colon-terminated line followed by an ORDINARY sentence (not a bullet) must not join.
    src = "Uwaga do wyniku:\nZysk netto wzrósł o 10% rdr.\n"
    out = normalize_structure(src)
    assert out == src


# --- Idempotence of the full pipeline ---
def test_normalize_report_idempotent():
    src = (
        "Kapitał obrotowy netto wzrósł, co wynika głównie z:\n"
        "  • wzrostu należności o 59,0 mln zł.\n"
        "224,6 221,2 243,1 6,9 6,6 6,9 jest efektem zmiany struktury sprzedaży poprzez\n"
        "Spadek marży wynika z wzrostu kosztów.\n"
    )
    once = normalize_report(src)
    assert normalize_report(once) == once


def test_normalize_structure_idempotent():
    src = (
        "Efekt wynika głównie z:\n"
        "  • wzrostu należności o 59,0 mln zł.\n"
        "224,6 221,2 243,1 6,9 6,6 6,9 jest efektem zmiany struktury sprzedaży poprzez\n"
    )
    once = normalize_structure(src)
    assert normalize_structure(once) == once
