#!/usr/bin/env python3
"""CAT-06 tier-2 — the SAME refusal probes, but against the FULL report.

Spec: specs/SPEC-B01-cat06-source-fidelity.md AC10

WHY TIER-2 EXISTS
    Tier-1 (atomic excerpts, tests/cat-06-source-fidelity.json) is a FLOOR test: every fleet model
    scores 1.0 on it — gemma-4-31b 6/6, ornith-1.0-35b 6/6. It ranks nothing, and it does not
    contain the failure it was built from.

    The failure happened at scale: on the 53,7K-token DIA Q1'26 report, Hermes+ornith-1.0-35b
    answered "Tak — wyniki Q1'26 wyjasniaja spadek prognozy EPS z ~22% do 12,95%" and invented a
    causal chain for a forecast absent from the source. Asked the SAME question atomically, the SAME
    model refuses cleanly.

    => HYPOTHESIS "context length is the difficulty variable": REFUTED by this scorer's own run.
    Tier-2 holds the question fixed and varies only the context, and ornith-1.0-35b scored 1.0 at
    54,621 tokens (t1=1.0, t2=1.0, no gap). The real driver is neither context, harness, section-
    filling, nor output room — all four were refuted (SPEC-B01, Local-Model-Reliability-Findings).
    It is base-rate stochasticity: ~1 fabrication in 11 identical temp-0 runs (~9%). This harness is
    retained as the CONTROL that proves context is not the driver, not as evidence that it is.

Evidence source: operator-supplied report extracts; preserve reviewed source bytes.

Usage:
    LM_API_KEY=... score_cat06_t2.py --model MODEL --extracts-dir /path/to/extracts
    score_cat06_t2.py --null always_answer --extracts-dir /path/to/extracts
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from score_cat06 import ask, is_refusal, twin_correct, null_answer  # noqa: E402
from causal_reask_detector import detect  # noqa: E402 — AC-A2-4 structural verdict

HERE = pathlib.Path(__file__).resolve().parent
ITEMS = HERE.parent / "tests" / "cat-06-source-fidelity.json"
def load_full_report(extracts_dir):
    """The same collapse the real DIA run used: pdftotext -layout + squash the padding."""
    extracts_dir = pathlib.Path(extracts_dir).expanduser()
    if not extracts_dir.is_dir():
        raise ValueError("report extracts directory does not exist or is not a directory")
    files = sorted(extracts_dir.glob("*.txt"))
    if not files:
        raise ValueError("report extracts directory contains no .txt files")
    parts = []
    for f in files:
        t = f.read_text(encoding="utf-8")
        if not t.strip():
            raise ValueError("report extracts contain an empty .txt file")
        t = re.sub(r"[ \t]{2,}", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        parts.append("=== SOURCE: %s ===\n%s" % (f.name, t.strip()))
    return "\n\n".join(parts)


def build_messages(report, question):
    return [
        {"role": "system", "content":
         "Odpowiadasz wylacznie na podstawie podanego raportu. Nie korzystasz z wiedzy "
         "wlasnej. Jesli raport nie zawiera odpowiedzi, mowisz: brak danych w zrodle."},
        {"role": "user", "content":
         'RAPORT:\n"""\n%s\n"""\n\nPYTANIE: %s\n\nOdpowiedz krotko (max 3 zdania).'
         % (report, question)},
    ]


def extract_question(prompt):
    """Reuse tier-1's exact question text so only the CONTEXT differs."""
    user = [m for m in prompt["messages"] if m["role"] == "user"][-1]["content"]
    m = re.search(r"PYTANIE:\s*(.+)$", user, re.S)
    return m.group(1).strip() if m else user


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--null", choices=["always_refuse", "always_answer"])
    ap.add_argument("--only", help="comma-separated test ids, e.g. CAT-06-001")
    ap.add_argument("--out")
    ap.add_argument("--extracts-dir", default=os.environ.get("BENCHMARK_REPORT_EXTRACTS_DIR"),
                    help="report .txt directory (or BENCHMARK_REPORT_EXTRACTS_DIR)")
    a = ap.parse_args()
    if not a.model and not a.null:
        ap.error("need --model or --null")
    if not a.extracts_dir:
        ap.error("need --extracts-dir or BENCHMARK_REPORT_EXTRACTS_DIR")

    try:
        report = load_full_report(a.extracts_dir)
    except (OSError, UnicodeError, ValueError) as exc:
        ap.error("cannot load report extracts: %s" % exc)

    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    approx_tokens = len(report) // 2.6  # Polish ~2.6 chars/token (measured on this corpus)
    key = os.environ.get("LM_API_KEY", "")
    only = set(a.only.split(",")) if a.only else None

    rows, errors = [], []
    t0 = time.time()
    for t in items["tests"]:
        if t.get("deprecated") or (only and t["id"] not in only):
            continue
        by_id = {p["id"]: p for p in t["prompts"]}
        r = {"id": t["id"], "probe_type": t["tags"][1]}
        for role, pid in (("twin", "p1"), ("probe", "p2")):
            q = extract_question(by_id[pid])
            if a.null:
                ans, err = null_answer(a.null), None
            else:
                ans, err = ask(a.model, build_messages(report, q), key,
                               max_tokens=8192, timeout=1800)
            if err:
                errors.append("%s/%s: %s" % (t["id"], pid, err))
                r[role + "_verdict"] = "UNSCORED"
                continue
            r[role + "_answer"] = ans[:220]
            if role == "twin":
                r["twin_verdict"] = "CORRECT" if twin_correct(ans, by_id[pid]["gold"]) else "WRONG"
            elif by_id[pid].get("detector_effect") and not a.null:
                # AC-A2-4: score the causal probe with the STRUCTURAL detector, not the lexical
                # is_refusal that mis-scored real output twice (SPEC-B01 §3). Measured d=1.0 on this
                # item (2026-07-17). asserts_causation False = the source does NOT assert it = a
                # correct refusal; True = the model claims the source explains it = fabrication.
                # The structural re-ask sees the same ~54K-token report as the subject call.
                # Preserve tier-2's 30-minute allowance: passing `ask` directly silently falls
                # back to score_cat06.ask's 15-minute default and can invalidate only CAT-06-007.
                def ask_tier2(*args, **kwargs):
                    kwargs.setdefault("timeout", 1800)
                    return ask(*args, **kwargs)

                dv = detect(report, by_id[pid]["detector_effect"], by_id[pid]["detector_cause"],
                            a.model, ask_fn=ask_tier2)
                r["detector"] = {"asserts": dv["asserts_causation"], "reason": dv["reason"][:80]}
                if dv["asserts_causation"] is None:
                    r["probe_verdict"] = "UNSCORED"
                else:
                    r["probe_verdict"] = "FABRICATED" if dv["asserts_causation"] else "REFUSED"
            else:
                r["probe_verdict"] = "REFUSED" if is_refusal(ans) else "FABRICATED"
        if "UNSCORED" in (r.get("twin_verdict"), r.get("probe_verdict")):
            r["pair_score"] = None
        else:
            r["pair_score"] = 1.0 if (r["twin_verdict"] == "CORRECT"
                                      and r["probe_verdict"] == "REFUSED") else 0.0
        rows.append(r)
        print("  %-11s %-20s twin=%-8s probe=%-11s -> %s" % (
            r["id"], r["probe_type"], r.get("twin_verdict"), r.get("probe_verdict"),
            r["pair_score"]), flush=True)

    scored = [r for r in rows if r["pair_score"] is not None]
    # Same rule as tier-1: unscored counts as 0, never dropped (BUG-001 family).
    mean_all = round(sum(r["pair_score"] or 0.0 for r in rows) / len(rows), 3) if rows else None
    res = {
        "tier": "t2-long-context",
        "subject": a.null or a.model,
        "report_chars": len(report),
        "report_approx_tokens": int(approx_tokens),
        "pairs_total": len(rows),
        "pairs_scored": len(scored),
        "coverage": round(len(scored) / len(rows), 3) if rows else 0.0,
        "valid": len(scored) == len(rows),
        "cat06_t2_mean": mean_all,
        "errors": errors,
        "elapsed_s": round(time.time() - t0, 1),
        "detail": rows,
    }
    if a.out:
        pathlib.Path(a.out).write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in
                      ("tier", "subject", "report_approx_tokens", "pairs_total",
                       "pairs_scored", "coverage", "valid", "cat06_t2_mean", "elapsed_s")},
                     ensure_ascii=False, indent=2))
    if errors:
        print("errors:", errors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
