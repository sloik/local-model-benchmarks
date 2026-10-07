#!/usr/bin/env python3
"""CAT-05 · Classification & Triage — deterministic label-match scorer.

Spec: specs/SPEC-B04-cat05-judge.md
Items: tests/cat-05-klasyfikacja.json  (native battery schema, scoring="min")

WHY THIS EXISTS ALONGSIDE THE NORMAL JUDGE PATH
    The calibrated Qwen-primary/Muse-dissent panel remains semantic and can still
    disagree or leak multi-prompt criteria across answers. Exact classification
    labels are mechanically decidable, so model opinion must not override them.

    CAT-05 answers are ENUMERABLE by construction: every prompt's system message lists the
    label set ("Classify ... as one of: raport_kwartalny / ... / inne") and asks the model to
    "Respond ONLY with the label". Deciding "is the chosen label the gold label?" is then a
    normalized string match — no judge, no rubric, no weak-classifier-grading-classification.
    Run this as the authoritative LABEL score; the judge path may run alongside for the
    justification-quality nuance it (and only it) can grade.

WHAT THIS DOES *NOT* GRADE (honest scope — the residual is the judge's / a panel's job)
    The item rubrics award 5 for "correct label + concise justification referencing specific
    features" and 4 for "correct label, justification missing/generic". This scorer decides
    label-correctness ONLY (the core classification decision). The 4-vs-5 justification nuance
    is the residual SPEC-B04 recommends a stronger judge / judge-panel for.

ANTI-GAMING (inherited from CAT-06's paired discipline — non-negotiable)
    A constant-output model ("always raport_kwartalny") must not score well. Two mechanisms:
      1. scoring="min" per item — a multi-prompt item is won only if EVERY prompt is correct,
         so any constant label fails the moment two prompts disagree.
      2. CONTRASTIVE subset — the gaming-resistant headline (`cat05_label_mean_contrastive`)
         is taken only over items whose prompts carry >=2 DISTINCT gold labels. Over that
         subset every constant-output null scores 0.0, exactly as CAT-06 pairs guarantee.
    Single-prompt items are still deterministically label-parseable and are scored, but a
    single (document -> label) example CANNOT tell a real classifier from a model hardwired to
    that label — so they are reported separately (`not_null_resistant`) and excluded from the
    contrastive headline. SPEC-B04's top recommendation is to pair them, CAT-06-style.

Usage:
    LM_API_KEY=... score_cat05_deterministic.py --model ornith-1.5-35b-a3b-mlx [--out results/x.json]
    score_cat05_deterministic.py --run-dir results/2026-... [--out results/...json]
    score_cat05_deterministic.py --null raport_kwartalny   # anti-gaming guard — see report
    score_cat05_deterministic.py --null inne                # worst-case constant (single-prompt inne)
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
ITEMS = HERE.parent / "tests" / "cat-05-klasyfikacja.json"


# ── label parsing ──────────────────────────────────────────────────────────────
def normalize_label(s):
    """Lowercase, collapse whitespace/hyphens to a single underscore.

    'Raport Kwartalny' / 'raport-kwartalny' / 'raport_kwartalny' all -> 'raport_kwartalny'.
    The item rubrics explicitly tolerate this ("'raport kwartalny' = 'raport_kwartalny'").
    """
    s = s.strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def parse_label_set(system_content):
    """Extract the enumerable label set from a system prompt's 'one of: A / B / ... ' clause.

    Taxonomy varies per prompt (4-class, 5-class, 6-class), so it is read from the prompt,
    not hardcoded. Returns normalized labels.
    """
    m = re.search(r"one of:\s*([a-z0-9_/ \-]+)", system_content, re.IGNORECASE)
    if not m:
        return []
    chunk = m.group(1)
    return [normalize_label(p) for p in chunk.split("/") if normalize_label(p)]


def extract_label(response, label_set):
    """Return the label from `label_set` that appears EARLIEST in the response, or None.

    Earliest-position wins because the system prompt instructs "Respond ONLY with the label"
    first — so the answer label leads, and a forbidden label merely *mentioned* in the
    justification ("...not analiza_analityczna...") cannot outrank it. Matching is word-bounded
    (protects the common Polish word 'inne' inside 'innych'/'innego') and accepts the
    underscore and space forms of each label.
    """
    text = response.lower()
    best_pos, best_label = None, None
    for lab in label_set:
        norm = normalize_label(lab)
        variants = {norm, norm.replace("_", " ")}
        for v in variants:
            m = re.search(r"\b" + re.escape(v) + r"\b", text)
            if m and (best_pos is None or m.start() < best_pos):
                best_pos, best_label = m.start(), norm
    return best_label


# ── gold interpretation ────────────────────────────────────────────────────────
def gold_targets(gold):
    """(correct_labels, acceptable_labels) for a flat enumerable gold, or (None, None).

    Flat/enumerable = has `required` or `required_one_of` (a fixed label set the answer must
    hit). `required_per_document` (the CAT-05-010 batch item) is NOT flat -> (None, None) ->
    routed to needs-judge, never force-scored.
    """
    if "required_per_document" in gold:
        return None, None
    correct = gold.get("required") or gold.get("required_one_of")
    if not correct:
        return None, None
    correct = {normalize_label(x) for x in correct}
    acceptable = {normalize_label(x) for x in gold.get("acceptable", [])}
    return correct, acceptable


def is_deterministic(test):
    """True iff every prompt in the item has a flat enumerable single-target gold."""
    for p in test["prompts"]:
        correct, _ = gold_targets(p.get("gold", {}))
        if correct is None:
            return False
    return True


def is_contrastive(test):
    """True iff NO single label is a pass (correct OR acceptable) for every prompt.

    This is the CAT-06-pairing property generalized: an item defeats constant-output gaming
    under min-scoring only if the intersection of the per-prompt PASS sets is empty — then no
    constant label can win all prompts. Using correct-∪-acceptable (not just correct) is load-
    bearing: CAT-05-004 has 2 distinct required labels yet BOTH prompts accept 'raport_kwartalny'
    (p1 as secondary, p2 as required_one_of), so a constant 'raport_kwartalny' would win it — it
    is therefore NOT null-resistant despite the label variety.
    """
    pass_sets = []
    for p in test["prompts"]:
        correct, acceptable = gold_targets(p.get("gold", {}))
        if correct is None:
            return False
        pass_sets.append(correct | acceptable)
    common = set.intersection(*pass_sets) if pass_sets else set()
    return len(pass_sets) >= 2 and not common


def verdict(answer, prompt):
    """CORRECT / ACCEPTABLE / WRONG / UNPARSEABLE for one prompt's answer.

    ACCEPTABLE = the label is in the gold's secondary-accept set; counts as a label pass but is
    reported so a run can see how much rode on secondary accepts.
    """
    correct, acceptable = gold_targets(prompt["gold"])
    if correct is None:
        return "NEEDS_JUDGE"
    label_set = parse_label_set(prompt["messages"][0]["content"])
    # Union with gold labels as a safety net if the system prompt phrasing changes.
    label_set = list({*label_set, *correct, *acceptable})
    chosen = extract_label(answer or "", label_set)
    if chosen is None:
        return "UNPARSEABLE"
    if chosen in correct:
        return "CORRECT"
    if chosen in acceptable:
        return "ACCEPTABLE"
    return "WRONG"


# ── model / null / injected answer sourcing ──────────────────────────────────────
# Classification answers are short, but a <think> model still spends its budget on reasoning
# FIRST and emits nothing if it runs out (recorded benchmark reliability finding). Keep max_tokens large.
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
    if not content and reasoning:
        return None, "EMPTY_ALL_REASONING(reasoning_tokens=%d)" % reasoning
    if not content:
        return None, "EMPTY_RESPONSE"
    return content, None


def get_answer(test, prompt, model, null, injected, api_key):
    """Source one prompt's answer: injected fixture > constant null > live model."""
    if injected is not None:
        return injected.get((test["id"], prompt["id"]), ""), None
    if null is not None:
        return null, None
    return ask(model, prompt["messages"], api_key)


def load_run_answers(run_dir):
    """Load exact saved canonical responses keyed by (test_id, prompt_id)."""
    raw_dir = pathlib.Path(run_dir) / "raw"
    if not raw_dir.is_dir():
        raise ValueError("missing raw response directory: %s" % raw_dir)
    answers = {}
    for raw_file in sorted(raw_dir.glob("CAT-05-*_*.txt")):
        test_id, prompt_id = raw_file.stem.rsplit("_", 1)
        answers[(test_id, prompt_id)] = raw_file.read_text(encoding="utf-8")
    return answers


# ── scoring ──────────────────────────────────────────────────────────────────────
def score(items, model=None, null=None, injected=None, api_key=None):
    rows, errors = [], []
    for t in items["tests"]:
        if t.get("deprecated"):
            continue
        det = is_deterministic(t)
        contrastive = det and is_contrastive(t)
        r = {
            "id": t["id"],
            "difficulty": t.get("difficulty"),
            "gradeable": "deterministic" if det else "needs-judge",
            "contrastive": contrastive,
            "not_null_resistant": det and not contrastive,
        }
        if not det:
            # Genuinely open-ended for a label-match scorer (batch/structured gold).
            # Route to the judge — do NOT invent a score (that would be a false pass).
            r["item_score"] = None
            r["prompt_verdicts"] = None
            rows.append(r)
            continue

        verdicts = []
        for p in t["prompts"]:
            ans, err = get_answer(t, p, model, null, injected, api_key)
            if err:
                errors.append("%s/%s: %s" % (t["id"], p["id"], err))
                verdicts.append("UNSCORED")
                continue
            verdicts.append(verdict(ans, p))
        r["prompt_verdicts"] = verdicts
        if "UNSCORED" in verdicts:
            r["item_score"] = None  # harness failure, not a model result — never 0-credit it
        else:
            passed = all(v in ("CORRECT", "ACCEPTABLE") for v in verdicts)
            r["item_score"] = 1.0 if passed else 0.0
        rows.append(r)

    det_rows = [r for r in rows if r["gradeable"] == "deterministic"]
    judge_rows = [r for r in rows if r["gradeable"] == "needs-judge"]
    contr_rows = [r for r in det_rows if r["contrastive"]]
    weak_rows = [r for r in det_rows if r["not_null_resistant"]]

    def _mean(rs):
        scored = [r for r in rs if r["item_score"] is not None]
        return round(sum(r["item_score"] for r in scored) / len(scored), 3) if scored else None

    return {
        "subject": null and ("null:" + null) or model,
        "is_null_model": null is not None,
        "items_active": len(rows),
        "items_deterministic": len(det_rows),
        "items_needs_judge": len(judge_rows),
        "items_contrastive": len(contr_rows),
        "items_not_null_resistant": len(weak_rows),
        "coverage": round(len(det_rows) / len(rows), 3) if rows else 0.0,
        # Authoritative gaming-resistant score — over contrastive items only.
        "cat05_label_mean_contrastive": _mean(contr_rows),
        # Full deterministic mean (includes single-prompt items — read WITH the null ceiling).
        "cat05_label_mean_all": _mean(det_rows),
        "needs_judge_ids": [r["id"] for r in judge_rows],
        "errors": errors,
        "detail": rows,
    }


def constant_null_score(items, label, subset="contrastive"):
    """Score of a model that answers `label` to everything. subset: contrastive|all."""
    res = score(items, null=normalize_label(label))
    key = ("cat05_label_mean_contrastive" if subset == "contrastive"
           else "cat05_label_mean_all")
    return res[key]


def null_ceiling(items, subset="all"):
    """(best_label, best_score) — the highest score any constant-output null achieves.

    The anti-gaming headline: a real model must beat this to demonstrate it is classifying and
    not exploiting the label base rate.
    """
    labels = set()
    for t in items["tests"]:
        if t.get("deprecated"):
            continue
        for p in t["prompts"]:
            labels |= set(parse_label_set(p["messages"][0]["content"]))
            correct, acc = gold_targets(p.get("gold", {}))
            if correct:
                labels |= correct
    best_label, best = None, -1.0
    for lab in sorted(labels):
        s = constant_null_score(items, lab, subset=subset) or 0.0
        if s > best:
            best_label, best = lab, s
    return best_label, round(best, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--run-dir", help="score saved CAT-05 responses from a canonical run")
    ap.add_argument("--null", help="constant label emitted for every prompt (anti-gaming guard)")
    ap.add_argument("--items", default=str(ITEMS))
    ap.add_argument("--out")
    ap.add_argument("--ceiling", action="store_true",
                    help="report the constant-output null ceiling and exit")
    a = ap.parse_args()

    items = json.loads(pathlib.Path(a.items).read_text(encoding="utf-8"))

    if a.ceiling:
        lab_all, sc_all = null_ceiling(items, subset="all")
        lab_c, sc_c = null_ceiling(items, subset="contrastive")
        print(json.dumps({
            "null_ceiling_all_deterministic": {"label": lab_all, "score": sc_all},
            "null_ceiling_contrastive": {"label": lab_c, "score": sc_c},
        }, ensure_ascii=False, indent=2))
        return 0

    selected = sum(value is not None for value in (a.model, a.run_dir, a.null))
    if selected != 1:
        ap.error("choose exactly one of --model, --run-dir, or --null")

    t0 = time.time()
    injected = load_run_answers(a.run_dir) if a.run_dir else None
    res = score(items, model=a.model, null=a.null and normalize_label(a.null),
                injected=injected, api_key=os.environ.get("LM_API_KEY", ""))
    if a.run_dir:
        res["subject"] = pathlib.Path(a.run_dir).name
        res["answer_source"] = "saved-run"
    res["elapsed_s"] = round(time.time() - t0, 1)
    if a.out:
        pathlib.Path(a.out).write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: res[k] for k in (
        "subject", "items_active", "items_deterministic", "items_needs_judge",
        "items_contrastive", "coverage", "cat05_label_mean_contrastive",
        "cat05_label_mean_all", "elapsed_s")}, ensure_ascii=False, indent=2))
    for r in res["detail"]:
        print("  %-11s %-13s contrastive=%-5s %-30s -> %s" % (
            r["id"], r["gradeable"], r["contrastive"],
            str(r["prompt_verdicts"]), r["item_score"]))
    if res["errors"]:
        print("errors:", res["errors"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
