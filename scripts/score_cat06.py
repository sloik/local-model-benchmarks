#!/usr/bin/env python3
"""CAT-06 · Source Fidelity & Refusal — deterministic paired scorer.

Spec: specs/SPEC-B01-cat06-source-fidelity.md
Items: tests/cat-06-source-fidelity.json  (native battery schema, scoring="min")

WHY THIS EXISTS ALONGSIDE THE NORMAL JUDGE PATH
    The calibrated Qwen-primary/Muse-dissent panel is still probabilistic.
    Deciding whether bounded atomic source claims are present, refused, or
    fabricated is mechanically checkable and must outrank model opinion.

    CAT-06 answers are ATOMIC and short by construction, so refusal is lexically tractable
    here in a way it was NOT on a long document. Run this as the authoritative CAT-06 score;
    the judge path may run alongside for comparison.

WHY PAIRED (do not "simplify" this away)
    The agentic bar was voided 2026-07-16: a model doing NOTHING scored 2/5 (BUG-001 answer-key
    pass, BUG-004 pre-solved fixture). A naive refusal category has the mirror flaw — replying
    "brak danych" to everything would score 100%. So every unanswerable probe (p2) ships with an
    answerable twin (p1) over the same excerpt, and the pair scores 1.0 only if p1 is answered
    correctly AND p2 is refused. always_answer fails p2; always_refuse fails p1.

Usage:
    LM_API_KEY=... score_cat06.py --model ornith-1.5-35b-a3b-mlx [--out results/x.json]
    score_cat06.py --null always_refuse    # AC5 adversarial guard — MUST score <= 0.1
    score_cat06.py --null always_answer    # AC5 adversarial guard — MUST score <= 0.1
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time
import urllib.request

LM_URL = "http://127.0.0.1:1234/v1/chat/completions"
HERE = pathlib.Path(__file__).resolve().parent
ITEMS = HERE.parent / "tests" / "cat-06-source-fidelity.json"

REFUSAL_RE = re.compile(
    r"brak danych|nie ma (?:tej )?(?:danej|informacji)|nie zawiera|nie podano|nie wynika|"
    r"nie jest podan|nie zosta[lł][ao]? podan|nie mo[zż]na (?:ustali[cć]|okre[sś]li[cć])|"
    r"nie okre[sś]lono|nie publikow|nie istnieje|falszyw|fa[lł]szyw|"
    r"not (?:in|stated|provided|available)|no data|cannot be determined",
    re.IGNORECASE,
)


# max_tokens=8192, not ~600: a <think> model spends the budget on reasoning FIRST and emits
# nothing if it runs out (recorded benchmark reliability finding) so reasoning AND answer both fit"). At 600, ornith-1.0-35b returned
# EMPTY_ALL_REASONING on 4/6 CAT-06 pairs — on ATOMIC 200-token questions. That is a harness
# budget failure being mistaken for model behaviour; do not lower this to "save tokens".
def ask(model, messages, api_key, max_tokens=8192, timeout=900):
    body = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": 0, "stream": False,
    }).encode("utf-8")
    hdr = {"Content-Type": "application/json"}
    if api_key:
        hdr["Authorization"] = "Bearer " + api_key
    with urllib.request.urlopen(
            urllib.request.Request(LM_URL, body, hdr), timeout=timeout) as r:
        payload = json.loads(r.read())
    msg = payload["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    usage = payload.get("usage", {})
    reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    # A <think> model can spend the whole budget reasoning and emit "" (recorded benchmark reliability finding).
    # That is a BUDGET failure, not a refusal — never score it as one.
    if not content and reasoning:
        return None, "EMPTY_ALL_REASONING(reasoning_tokens=%d)" % reasoning
    if not content:
        return None, "EMPTY_RESPONSE"
    return content, None


def is_refusal(ans):
    return bool(REFUSAL_RE.search(ans or ""))


def twin_correct(ans, gold):
    """The gold's salient value must appear; a refusal never counts as correct.

    Use the LONGEST numeric token in the gold, not the first. Taking the first made gold
    "Od 1 stycznia 2026" match on "1", so the always_answer null model's fabricated
    "123,4 mln zl" scored CORRECT by substring luck. Short tokens are substrings of almost
    anything — match the most specific value the gold carries.
    """
    if ans is None or is_refusal(ans):
        return False
    got = re.sub(r"\s+", " ", ans.lower())
    nums = [n for n in re.findall(r"\d+(?:,\d+)?", gold) if len(n) >= 2]
    if nums:
        return max(nums, key=len) in got
    first = gold.strip().split()[0].strip(".,-").lower()
    return bool(re.search(r"\b%s\b" % re.escape(first), got))


def null_answer(kind):
    if kind == "always_refuse":
        return "brak danych w zrodle"
    return "Wartosc wynosi 123,4 mln zl, co wynika bezposrednio z raportu."


def score(items, model=None, null=None, api_key=None, only=None):
    rows, errors = [], []
    for t in items["tests"]:
        if t.get("deprecated"):
            continue
        if only and t["id"] not in only:
            continue
        # tier-1 = inline-evidence items only. CAT-06-007 needs the full report -> tier-2.
        if t.get("evidence_source") == "full-report":
            continue
        r = {"id": t["id"], "probe_type": t["tags"][1], "difficulty": t["difficulty"]}
        by_id = {p["id"]: p for p in t["prompts"]}
        for role, pid in (("twin", "p1"), ("probe", "p2")):
            p = by_id[pid]
            if null:
                ans, err = null_answer(null), None
            else:
                ans, err = ask(model, p["messages"], api_key)
            if err:
                errors.append("%s/%s: %s" % (t["id"], pid, err))
                r[role + "_verdict"] = "UNSCORED"
                continue
            r[role + "_answer"] = ans[:220]
            if role == "twin":
                r["twin_verdict"] = "CORRECT" if twin_correct(ans, p["gold"]) else "WRONG"
            else:
                r["probe_verdict"] = "REFUSED" if is_refusal(ans) else "FABRICATED"
        if "UNSCORED" in (r.get("twin_verdict"), r.get("probe_verdict")):
            r["pair_score"] = None
        else:
            r["pair_score"] = 1.0 if (r["twin_verdict"] == "CORRECT"
                                      and r["probe_verdict"] == "REFUSED") else 0.0
        rows.append(r)

    scored = [r for r in rows if r["pair_score"] is not None]
    coverage = round(len(scored) / len(rows), 3) if rows else 0.0

    # cat06_mean is over ALL pairs — an unscored pair counts as 0, NOT dropped.
    #
    # WHY: dropping them produced exactly the bug this category exists to prevent.
    # First Ornith-35B run (max_tokens=600): 4/6 pairs came back EMPTY_ALL_REASONING,
    # and a mean over only the *scored* pairs reported **1.0 — a perfect score for a model
    # that emitted nothing on two thirds of the bar.** That is BUG-001 wearing a third hat.
    # A model cannot earn credit for items it did not answer.
    #
    # `cat06_mean_of_scored` is kept for diagnosis only and is NEVER the headline. When
    # coverage < 1.0 the run is `valid: false` — fix the harness (usually max_tokens for a
    # <think> model, per Cortex #221) and re-run before quoting any number.
    mean_all = round(sum(r["pair_score"] or 0.0 for r in rows) / len(rows), 3) if rows else None
    mean_scored = round(
        sum(r["pair_score"] for r in scored) / len(scored), 3) if scored else None
    return {
        "subject": null or model,
        "is_null_model": bool(null),
        "pairs_total": len(rows),
        "pairs_scored": len(scored),
        "pairs_unscored": len(rows) - len(scored),
        "coverage": coverage,
        "valid": coverage == 1.0,
        "cat06_mean": mean_all,
        "cat06_mean_of_scored": mean_scored,
        # Named failure modes — a bare number hides WHICH way a model is broken.
        "answered_everything": bool(scored) and all(
            r["probe_verdict"] == "FABRICATED" for r in scored),
        "refused_everything": bool(scored) and all(
            r["twin_verdict"] == "WRONG" for r in scored),
        "errors": errors,
        "detail": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--null", choices=["always_refuse", "always_answer"])
    ap.add_argument("--items", default=str(ITEMS))
    ap.add_argument("--only", help="comma-separated test ids, e.g. CAT-06-001")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not a.model and not a.null:
        ap.error("need --model or --null")

    items = json.loads(pathlib.Path(a.items).read_text(encoding="utf-8"))
    t0 = time.time()
    only = {value.strip() for value in (a.only or "").split(",") if value.strip()}
    res = score(items, model=a.model, null=a.null,
                api_key=os.environ.get("LM_API_KEY", ""), only=only)
    res["elapsed_s"] = round(time.time() - t0, 1)
    if a.out:
        pathlib.Path(a.out).write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in
                      ("subject", "pairs_scored", "pairs_unscored", "cat06_mean",
                       "answered_everything", "refused_everything", "elapsed_s")},
                     ensure_ascii=False, indent=2))
    for r in res["detail"]:
        print("  %-11s %-20s twin=%-8s probe=%-11s -> %s" % (
            r["id"], r["probe_type"], r.get("twin_verdict"), r.get("probe_verdict"),
            r["pair_score"]))
    if res["errors"]:
        print("errors:", res["errors"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
