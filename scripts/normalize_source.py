#!/usr/bin/env python3
"""Evidence normalization — character-canonicalization for robust verbatim matching.

Spec: specs/SPEC-B03-evidence-normalization.md

WHY: the A2 detector's Gate 2 ("span verbatim in source") measured FPR=0.67 — false-flags where the
model quoted a REAL grounded sentence whose text differed from the pdftotext-mangled source only by
curly-vs-straight quotes, dash variants, or NBSP. That is the evidence-normalization bottleneck the
whole DIA session converged on (it also causes Apple FM's 0/3->5/6 swing and the T1 chart trap).

THE LINE THIS MUST NOT CROSS: Gate 2 kills paraphrase/invented citations — the core anti-fabrication
property. Canonicalization folds ONLY visually-equivalent characters. It does NOT strip words,
semantic-case-fold, or fuzzy-match. A genuine paraphrase (a dropped or changed word) MUST still fail
the gate. `tests/test_normalize_source.py::test_paraphrase_still_fails` is the guard.
"""
import argparse
import pathlib
import re
import sys

# curly/typographic quotes -> ASCII
_QUOTE_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",  # single
    "“": '"', "”": '"', "„": '"', "‟": '"',  # double
    "′": "'", "″": '"',                                  # primes
    "«": '"', "»": '"',                                  # guillemets
}
# dash / minus variants -> hyphen-minus
_DASH_MAP = {
    "‐": "-", "‑": "-", "‒": "-", "–": "-",
    "—": "-", "―": "-", "−": "-",
}
# spaces (NBSP, narrow NBSP, thin, hair, figure, zero-width) -> normal space / removed
_SPACE_MAP = {
    " ": " ", " ": " ", " ": " ", " ": " ",
    " ": " ", " ": " ", "﻿": "", "​": "",
}
_TRANS = {ord(k): v for m in (_QUOTE_MAP, _DASH_MAP, _SPACE_MAP) for k, v in m.items()}

# --- structure normalization (SPEC-B03 AC6 residual: fragmented_multi_span) ---
# A bullet line: leading whitespace, a bullet glyph, whitespace, then a LETTER (not a digit — that
# keeps chart-number rows like "-16,2 -17,2" from ever reading as bullets). Bullets in the DIA source
# begin with lowercase Polish words ("wzrostu", "spadku") or "+ Wzrost".
_BULLET = r"[•+\-–—*·‣▪]"
_BULLET_LINE_RE = re.compile(r"^\s*" + _BULLET + r"\s+[^\W\d_]", re.UNICODE)
_BULLET_STRIP_RE = re.compile(r"^\s*" + _BULLET + r"\s+", re.UNICODE)
# A numeric token: 224,6  6,9  84,9%  -16,2  1070,3  1,8x  +9,2%  (Polish comma-decimal, optional %/x).
_NUM_TOKEN_RE = re.compile(r"^[+\-]?\d[\d.,]*[%xX]?$")


def _is_num(tok):
    return bool(_NUM_TOKEN_RE.match(tok))


def _is_word(tok):
    return bool(tok) and tok[0].isalpha()


def _first_run(kinds, want, minlen):
    """First [start, end) maximal run of `want` in `kinds` with length >= minlen, else None."""
    i, n = 0, len(kinds)
    while i < n:
        if kinds[i] == want:
            j = i
            while j < n and kinds[j] == want:
                j += 1
            if j - i >= minlen:
                return (i, j)
            i = j
        else:
            i += 1
    return None


def _join_colon_bullets(lines):
    """Join a colon-terminated clause with its immediately-following bullet lines into one line.

    "X ... co wynika głównie z:" + "• wzrostu ..." -> "X ... co wynika głównie z: wzrostu ...".
    CONSERVATIVE: fires ONLY when the current stripped line ends in ':' AND the very next line starts
    with a bullet glyph followed by a word. Two ordinary sentences (no colon, or next line not a
    bullet) are never merged — that is what protects the verbatim/causation guarantee.
    """
    out = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if line.rstrip().endswith(":") and i + 1 < n and _BULLET_LINE_RE.match(lines[i + 1]):
            merged = line.rstrip()
            j = i + 1
            while j < n and _BULLET_LINE_RE.match(lines[j]):
                merged = merged + " " + _BULLET_STRIP_RE.sub("", lines[j].strip())
                j += 1
            out.append(merged)
            i = j
        else:
            out.append(line)
            i += 1
    return out


def _separate_chart_line(line):
    """If a line carries BOTH a run of >=3 numeric tokens AND a prose clause of >=3 words, split the
    prose onto its own line. Returns a list of lines, or None to leave the line unchanged.

    This isolates a bar-chart number run ("224,6 221,2 243,1 6,9 6,6 6,9") from a trailing/leading
    prose clause ("jest efektem zmiany struktury sprzedaży poprzez") that pdftotext -layout dropped
    onto the same physical line. CONSERVATIVE: needs a real >=3-word clause AND a >=3-number run, so
    ordinary prose (no long number run) and pure number/label rows (no long word run) are untouched.
    """
    tokens = line.split()
    if len(tokens) < 6:
        return None
    kinds = ["n" if _is_num(t) else ("w" if _is_word(t) else "o") for t in tokens]
    if _first_run(kinds, "n", 3) is None:
        return None
    span = _first_run(kinds, "w", 3)
    if span is None:
        return None
    i, j = span
    # Only fire on the clean "number-block + single prose-block" shape: EVERY word token must belong
    # to this one clause. If words are scattered (a two-column table row like "Koszty finansowe -16,2
    # ... wynika głównie z:", where prose and table share a physical line) we leave the line untouched
    # rather than shred the clause. That messy two-column interleave stays a documented residual.
    if kinds.count("w") != (j - i):
        return None
    segs = [seg for seg in (tokens[:i], tokens[i:j], tokens[j:]) if seg]
    if len(segs) < 2:
        return None
    return [" ".join(seg) for seg in segs]


def normalize_structure(text):
    """Deterministic, conservative line-structure repair BEFORE character folding.

    Two heuristics, both pattern-only (no model calls, stdlib only):
      1. colon-bullet joining  — rejoin a "...wynika z:" effect clause with its following bullet cause.
      2. chart-noise separation — split a same-line number-run + prose clause so prose stands alone.
    When a line matches neither pattern it is returned byte-for-byte unchanged (never mangled).
    Idempotent: re-running finds no colon+bullet adjacency and no mixed number/prose line to split.
    """
    if not text:
        return ""
    lines = _join_colon_bullets(text.split("\n"))
    out = []
    for line in lines:
        seg = _separate_chart_line(line)
        out.extend(seg if seg is not None else [line])
    return "\n".join(out)


def canonicalize(text):
    """Fold visually-equivalent characters and collapse whitespace. Idempotent.

    Character-equivalence ONLY — no word stripping, no case-folding, no fuzzy match. This is what
    makes it safe to run on BOTH the source and a quoted span before comparison: it removes the
    typographic noise that broke verbatim matching, without letting a paraphrase through.
    """
    if not text:
        return ""
    t = text.translate(_TRANS)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def normalize_report(text):
    """Source-prep pass over a full report: structure repair first (colon-bullet join + chart-noise
    separation), then character folding. Structure-before-fold matters — joining strips the bullet
    glyph that would otherwise sit between an effect clause and its cause in the folded source."""
    return canonicalize(normalize_structure(text))


def main():
    ap = argparse.ArgumentParser(description="Canonicalize source text for robust verbatim matching.")
    ap.add_argument("--in", dest="inp", required=True)
    a = ap.parse_args()
    p = pathlib.Path(a.inp)
    if not p.exists():
        ap.error("no such file: %s" % a.inp)
    sys.stdout.write(normalize_report(p.read_text(encoding="utf-8")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
