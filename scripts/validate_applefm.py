#!/usr/bin/env python3
"""Claim-level validator using Apple Foundation Model (~4K ctx) as a cheap second pass.

Two stages, because Apple FM cannot hold the 53K-token source:
  stage 1 (grep, deterministic) — is the claimed number present in the source AT ALL?
                                  absent => hallucination. No model call needed.
  stage 2 (Apple FM, semantic)  — the number IS in the source; is it attached to the
                                  label/period the analysis claims? grep cannot do this,
                                  and it is what catches the Q4'25-as-Q1'25 trap.

Usage: validate_applefm.py <analysis.md> <source.txt> <out.json>
"""
import json
import pathlib
import re
import sys
import time
import urllib.request

FM_URL = "http://127.0.0.1:1976/v1/chat/completions"
NBSP = " "
NUM_RE = re.compile(r"(?<![\w,.])(\d{1,4}(?:[\s ]\d{3})*(?:,\d+)?)(?![\w])")


def normalize(n):
    return n.replace(NBSP, " ").strip()


def fm_ask(prompt, retries=2):
    body = json.dumps({
        "model": "system",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 60,
        "temperature": 0,
        # fm serve streams SSE by default; without this the body is not JSON
        "stream": False,
    }).encode("utf-8")
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(
                FM_URL, body, {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                payload = json.loads(r.read())
            return payload["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # validator must survive a single bad claim
            if attempt == retries:
                return "ERROR: %s" % exc
            time.sleep(2)
    return "ERROR"


def find_evidence(num, src_lines, width=2):
    """Locate num in the source; return a small window around the first hit."""
    variants = {num, num.replace(" ", ""), num.replace(" ", NBSP)}
    for i, line in enumerate(src_lines):
        squashed = re.sub(r"[\s ]+", " ", line)
        if any(v in squashed or v in squashed.replace(" ", "") for v in variants):
            lo, hi = max(0, i - width), min(len(src_lines), i + width + 1)
            window = "\n".join(
                re.sub(r"[ \t]{2,}", " ", l).strip() for l in src_lines[lo:hi])
            return window[:1200]  # stay well inside Apple FM's ~4K window
    return None


def extract_claims(analysis_text):
    """A claim = one analysis line asserting at least one non-trivial number."""
    claims = []
    for line in analysis_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or set(s) <= set("|-: "):
            continue
        nums = [normalize(m.group(1)) for m in NUM_RE.finditer(s)]
        nums = [n for n in nums if "," in n or len(n.replace(" ", "")) >= 3]
        if nums:
            claims.append({"line": s[:300], "numbers": nums})
    return claims


def build_prompt(claim_line, number, evidence):
    return (
        "Jestes weryfikatorem danych finansowych. Sprawdzasz JEDNA liczbe.\n\n"
        'TWIERDZENIE Z ANALIZY:\n"%s"\n\n'
        'FRAGMENT ZRODLA (jedyne dopuszczalne zrodlo prawdy):\n"""\n%s\n"""\n\n'
        "PYTANIE: czy liczba %s jest w tym fragmencie przypisana do TEGO SAMEGO "
        "wskaznika i TEGO SAMEGO okresu co w twierdzeniu?\n"
        "Uwaga: raport zestawia Q1'25, Q4'25 i Q1'26 obok siebie - liczba moze "
        "istniec, ale dotyczyc innego okresu.\n"
        "Odpowiedz JEDNYM slowem: PASS (zgadza sie), FAIL (zla etykieta lub okres), "
        "UNSURE (fragment nie rozstrzyga)."
        % (claim_line, evidence, number)
    )


def classify(raw):
    up = raw.upper()
    if up.startswith("ERROR"):
        return "ERROR"
    if "FAIL" in up:
        return "FAIL_WRONG_LABEL"
    if "PASS" in up:
        return "PASS"
    if "UNSURE" in up:
        return "UNSURE"
    return "UNPARSED"


def main():
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: validate_applefm.py <analysis.md> <source.txt> <out.json>")
    analysis_p, source_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
    analysis = pathlib.Path(analysis_p).read_text(encoding="utf-8")
    src_lines = pathlib.Path(source_p).read_text(encoding="utf-8").splitlines()

    claims = extract_claims(analysis)
    results = []
    t0 = time.time()
    for claim in claims:
        for num in claim["numbers"]:
            evidence = find_evidence(num, src_lines)
            if evidence is None:
                results.append({
                    "number": num, "claim": claim["line"], "stage": "grep",
                    "verdict": "FAIL_NOT_IN_SOURCE", "evidence": None,
                    "fm_raw": None,
                })
                continue
            raw = fm_ask(build_prompt(claim["line"], num, evidence))
            results.append({
                "number": num, "claim": claim["line"], "stage": "fm",
                "verdict": classify(raw), "evidence": evidence[:400],
                "fm_raw": raw[:200],
            })

    summary = {}
    for r in results:
        summary[r["verdict"]] = summary.get(r["verdict"], 0) + 1
    out = {
        "analysis": analysis_p,
        "claims_with_numbers": len(claims),
        "numbers_checked": len(results),
        "fm_calls": sum(1 for r in results if r["stage"] == "fm"),
        "elapsed_s": round(time.time() - t0, 1),
        "summary": summary,
        "results": results,
    }
    pathlib.Path(out_p).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    head = {k: out[k] for k in
            ("analysis", "claims_with_numbers", "numbers_checked", "fm_calls",
             "elapsed_s", "summary")}
    print(json.dumps(head, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
