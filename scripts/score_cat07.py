#!/usr/bin/env python3
"""CAT-07 · Tool Use / Agentic Retrieval -- deterministic scorer.

Spec: SPEC.md (R3, AC3)
Items: tests/cat-07-tool-use.json
Harness: scripts/cat07_harness.py (R2 -- records every tool call the model makes)

WHY THIS EXISTS -- NO LLM JUDGE, EVER, IN THIS SCORING PATH (AC3, non-negotiable)
    The whole point of CAT-07 is verifying a model did not fabricate what a tool "returned". An
    LLM judge has the exact same blind spot the subject model does -- it cannot tell a plausible
    fabrication from a real tool result any more reliably than the model that produced it. Every
    verdict below is a mechanical check against `scripts/cat07_harness.py`'s recorded transcript
    (tool name called, args, the harness's own frozen-corpus lookup result) -- never a model
    opinion. Mirrors `score_cat05_deterministic.py` / `score_cat06.py`'s precedent exactly.

VERDICT TAXONOMY (AC3's original five, plus one addition -- SPEC-003-012/SPEC-003-011 Decision,
2026-09-16 -- the only sanctioned extension to date, made via the same dedicated-review process
SPEC-003-006 established; not a quiet patch)
    wrong_tool                -- the expected tool was never called (includes answering without
                                  calling any tool at all).
    bad_args                  -- the expected tool WAS called, but with an argument that fails the
                                  case's schema-level check (wrong enum value, url missing the
                                  target video id, company/date the corpus has never heard of).
    stale_snapshot_mismatch   -- args passed schema validation and named a REAL resource, but the
                                  harness's hermetic lookup resolved to a DIFFERENT real snapshot
                                  than the one the case needs (same company, wrong date is the
                                  canonical case -- R3d). The model then answers with data that is
                                  real, just for the wrong date.
    hallucinated_content      -- the correct resource was fetched, but the final answer does not
                                  reflect what the tool actually returned (missing the case's
                                  required fact, or asserting a disallowed one).
    pass                      -- correct tool, valid args, correct resource, faithful answer.
    resolved_but_uncommitted  -- (SPEC-003-012) a harness error occurred (typically
                                  max_turns_exhausted) but the decisive call to expected_tool was
                                  already schema-valid AND resolved to the correct real resource --
                                  the model reached the right answer and simply never committed to
                                  a final answer, a distinct and more specific finding than generic
                                  UNSCORED. Never 0-credited, same as UNSCORED.
    (UNSCORED is the remaining harness-failure escape hatch -- see score_case -- for every other
    harness-error case (the model never reached the resource at all, or the transcript is
    otherwise unusable), and is never counted as a model failure, following CAT-05/06's "never
    0-credit a harness failure" rule.)

Usage:
    LM_API_KEY=... score_cat07.py --model qwen/qwen3.6-27b [--out results/x/cat07-scores.json]
    score_cat07.py --model qwen/qwen3.6-27b --run-tag full   # writes results/<run_id>/ like run_benchmark.py
    score_cat07.py --null never_call          # AC-adjacent adversarial guard -- MUST score <= 0.1
    score_cat07.py --null always_call_fixed   # AC-adjacent adversarial guard -- MUST score <= 0.2
    score_cat07.py --run-dir results/<run_id> # rescore previously-saved raw07/ transcripts (AC4)
"""
import argparse
import json
import os
import pathlib
import re
import sys
import time
from datetime import datetime

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from cat07_harness import load_corpus, run_case  # noqa: E402

ITEMS = HERE.parent / "tests" / "cat-07-tool-use.json"
RESULTS_DIR = HERE.parent / "results"


def _norm(s):
    """Lowercase, collapse whitespace, and strip markdown emphasis markers (*_`).

    A model that faithfully reports "build a **graph** based on memory" must not be scored as
    hallucinated_content just because it bolded the word -- markdown formatting is not content.
    Discovered on the first live fleet run (qwen/qwen3.6-27b, CAT-07-T-002): the un-stripped
    literal "**graph**" defeated a substring check for "a graph", producing a false
    hallucinated_content verdict on an answer that was actually correct.
    """
    s = re.sub(r"[*_`]", "", s or "")
    return " ".join(s.lower().split())


# SPEC-003-010 Decision (2026-09-16): a forbidden string is only a genuine hallucination when it
# is ASSERTED, not when it is mentioned only to rule it out. A fixed, narrow set of contrastive/
# negation markers, checked in the same sentence as the forbidden term -- not a general sentiment
# classifier, which this scorer deliberately avoids (AC3's "no LLM judge, ever" applies equally to
# any fuzzy NLP heuristic standing in for one). This can only ever turn a hallucinated_content
# verdict into a pass (has_required is checked independently and unaffected), never the reverse --
# a genuinely wrong answer that never states the required correct content still fails on
# has_required regardless of this check.
_CONTRAST_MARKERS = (
    "not ", "n't ", "unlike ", "instead of ", "rather than ", "as opposed to ",
    "is not ", "was not ", "ruled out", "rejecting", "excluded",
)


def _split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def _forbidden_asserted(final_answer_norm, forbidden_norm):
    """True if `forbidden_norm` appears in some sentence with no contrastive/negation marker."""
    for sentence in _split_sentences(final_answer_norm):
        if forbidden_norm in sentence and not any(m in sentence for m in _CONTRAST_MARKERS):
            return True
    return False


def _find_call(tool_calls_made, expected_tool):
    """First recorded call matching expected_tool, or None. AC3a checks presence ANYWHERE in the
    transcript, not just the first call the model made -- a model that calls the wrong tool first
    and then self-corrects still gets credit for eventually calling the right one; a model that
    never calls it does not.
    """
    for call in tool_calls_made:
        if call["name"] == expected_tool:
            return call
    return None


def _check_arg(value, spec):
    """One argument's schema-level check (R1: schema-level, not exact-string equality)."""
    if spec.get("type") == "string" and not isinstance(value, str):
        return False
    if "enum" in spec and value not in spec["enum"]:
        return False
    if "must_contain" in spec and spec["must_contain"] not in (value or ""):
        return False
    if "must_contain_any" in spec:
        if not any(needle.lower() in (value or "").lower() for needle in spec["must_contain_any"]):
            return False
    return True


def _args_valid(args, expected_args_schema):
    for key, spec in expected_args_schema.items():
        if key not in args:
            return False
        if not _check_arg(args[key], spec):
            return False
    return True


def _call_resolved_correctly(call, case):
    """True if `call` (the first decisive call to expected_tool) is schema-valid AND resolved to
    a real, correct corpus resource -- i.e. everything except the final-answer text is already
    right. Mirrors the checks score_case performs after the error branch, without the final-answer
    step (SPEC-003-012 R3 / resolved_but_uncommitted)."""
    if call is None or not _args_valid(call["args"], case["expected_args_schema"]):
        return False
    raw_result = call["raw_result"]
    if isinstance(raw_result, dict) and raw_result.get("error"):
        return False
    if call.get("exact_match") is False:
        return False
    return True


def score_case(case, transcript):
    """Apply the deterministic verdict logic to one case's recorded transcript. Returns a row."""
    row = {"id": case["id"], "task_type": case["task_type"], "difficulty": case.get("difficulty")}

    if transcript.get("error"):
        # SPEC-003-012/SPEC-003-011 Decision: a harness error (typically max_turns_exhausted) that
        # nonetheless shows a valid, correctly-resolved decisive call is a DIFFERENT, more specific
        # finding than a generic harness failure -- the model reached the right resource and simply
        # never committed to a final answer. Distinguished as its own verdict, not zero-credited
        # (same "never 0-credit a harness failure" convention as UNSCORED) and not conflated with a
        # harness error where the model never reached the resource at all.
        call = _find_call(transcript["tool_calls_made"], case["expected_tool"])
        if _call_resolved_correctly(call, case):
            row["verdict"] = "resolved_but_uncommitted"
            row["reason"] = (
                "decisive call to '%s' resolved correctly, but no final answer was produced: %s"
                % (case["expected_tool"], transcript["error"])
            )
            row["case_score"] = None
            return row
        row["verdict"] = "UNSCORED"
        row["reason"] = transcript["error"]
        row["case_score"] = None
        return row

    tool_calls_made = transcript["tool_calls_made"]
    expected_tool = case["expected_tool"]
    call = _find_call(tool_calls_made, expected_tool)

    if call is None:
        row["verdict"] = "wrong_tool"
        row["reason"] = (
            "no call to '%s' found (no_call=%s, tools called: %s)"
            % (expected_tool, transcript["no_call"],
               [c["name"] for c in tool_calls_made] or "none")
        )
        row["case_score"] = 0.0
        return row

    if not _args_valid(call["args"], case["expected_args_schema"]):
        row["verdict"] = "bad_args"
        row["reason"] = "args %r fail schema %r" % (call["args"], case["expected_args_schema"])
        row["case_score"] = 0.0
        return row

    raw_result = call["raw_result"]
    if isinstance(raw_result, dict) and raw_result.get("error"):
        # Schema-shaped args that still resolved to nothing in the frozen corpus (e.g. a company
        # or video id this source has never heard of, even though the string "looks" plausible).
        row["verdict"] = "bad_args"
        row["reason"] = "tool call resolved to no corpus entry: %r" % raw_result
        row["case_score"] = 0.0
        return row

    if call.get("exact_match") is False:
        row["verdict"] = "stale_snapshot_mismatch"
        row["reason"] = (
            "args named a real resource but resolved to %r, not the case's frozen_source_key %r"
            % (call.get("resolved_key"), case.get("frozen_source_key"))
        )
        row["case_score"] = 0.0
        return row

    final_answer = _norm(transcript.get("final_answer"))
    checks = case.get("checks", {})
    required_any = checks.get("final_answer_must_contain_any", [])
    forbidden = checks.get("final_answer_must_not_contain", [])

    has_required = (not required_any) or any(_norm(s) in final_answer for s in required_any)
    has_forbidden = any(_forbidden_asserted(final_answer, _norm(s)) for s in forbidden)

    if not has_required or has_forbidden:
        row["verdict"] = "hallucinated_content"
        row["reason"] = (
            "final answer missing required fact(s) %r and/or contains forbidden %r -- answer=%r"
            % (required_any, forbidden, (transcript.get("final_answer") or "")[:200])
        )
        row["case_score"] = 0.0
        return row

    row["verdict"] = "pass"
    row["reason"] = "correct tool, valid args, correct resource, faithful answer"
    row["case_score"] = 1.0
    return row


def score(items, corpus, model=None, null=None, injected=None, api_key=None):
    """Run (or replay) every active case and score it. `injected` is {case_id: transcript} for
    --run-dir replay -- no live calls, no re-execution against the corpus (AC4 reproducibility).
    """
    rows = []
    for case in items["tests"]:
        if case.get("deprecated"):
            continue
        if injected is not None:
            transcript = injected.get(case["id"])
            if transcript is None:
                rows.append({"id": case["id"], "verdict": "UNSCORED",
                             "reason": "no saved transcript for this case", "case_score": None})
                continue
        else:
            # SPEC-003 R4: multi-hop cases declare max_turns in the case JSON (a resolve call +
            # the decisive fetch call + a final answer turn needs more than the 3-turn default);
            # additive only -- cases without the field are unaffected (default stays 3).
            transcript = run_case(case, corpus, model=model, null=null, api_key=api_key,
                                   max_turns=case.get("max_turns", 3))
        rows.append({**score_case(case, transcript), "transcript": transcript})

    scored = [r for r in rows if r["case_score"] is not None]
    coverage = round(len(scored) / len(rows), 3) if rows else 0.0
    mean_all = round(sum(r["case_score"] or 0.0 for r in rows) / len(rows), 3) if rows else None
    mean_scored = round(sum(r["case_score"] for r in scored) / len(scored), 3) if scored else None

    by_verdict = {}
    for r in rows:
        by_verdict[r["verdict"]] = by_verdict.get(r["verdict"], 0) + 1

    return {
        "subject": null and ("null:" + null) or model,
        "is_null_model": bool(null),
        "cases_total": len(rows),
        "cases_scored": len(scored),
        "cases_unscored": len(rows) - len(scored),
        "coverage": coverage,
        "valid": coverage == 1.0,
        # cat07_mean is over ALL cases -- an UNSCORED case is EXCLUDED (harness failure, not a
        # model result), but a scored failure (wrong_tool/bad_args/...) counts as 0, matching
        # CAT-05/06's "never let a dropped item look like a perfect score" rule.
        "cat07_mean": mean_all,
        "cat07_mean_of_scored": mean_scored,
        "verdict_counts": by_verdict,
        "detail": rows,
    }


def _save_raw07(run_dir, rows):
    raw_dir = pathlib.Path(run_dir) / "raw07"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        (raw_dir / (r["id"] + ".json")).write_text(
            json.dumps(r["transcript"], ensure_ascii=False, indent=2), encoding="utf-8"
        )


def _load_run_dir(run_dir):
    raw_dir = pathlib.Path(run_dir) / "raw07"
    if not raw_dir.is_dir():
        raise ValueError("missing raw07 directory: %s" % raw_dir)
    injected = {}
    for f in sorted(raw_dir.glob("*.json")):
        injected[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return injected


def _sanitize_model_id(model_id):
    return model_id.replace("/", "--").replace(" ", "_").replace(":", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--null", choices=["never_call", "always_call_fixed"])
    ap.add_argument("--run-dir", help="rescore previously saved raw07/ transcripts (no live calls)")
    ap.add_argument("--items", default=str(ITEMS))
    ap.add_argument("--corpus-dir")
    ap.add_argument("--out")
    ap.add_argument("--run-tag", help="write results/<date>_<model>_<tag>/ like run_benchmark.py")
    a = ap.parse_args()

    selected = sum(value is not None for value in (a.model, a.null, a.run_dir))
    if selected != 1:
        ap.error("choose exactly one of --model, --null, or --run-dir")

    items = json.loads(pathlib.Path(a.items).read_text(encoding="utf-8"))
    corpus = load_corpus(pathlib.Path(a.corpus_dir)) if a.corpus_dir else load_corpus()

    injected = _load_run_dir(a.run_dir) if a.run_dir else None
    t0 = time.time()
    res = score(items, corpus, model=a.model, null=a.null,
                injected=injected, api_key=os.environ.get("LM_API_KEY", ""))
    res["elapsed_s"] = round(time.time() - t0, 1)
    if a.run_dir:
        res["subject"] = pathlib.Path(a.run_dir).name
        res["answer_source"] = "saved-run"

    run_dir = None
    if a.run_tag and a.model:
        date_str = datetime.now().strftime("%Y-%m-%d")
        run_id = "%s_%s_%s" % (date_str, _sanitize_model_id(a.model), a.run_tag)
        run_dir = RESULTS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _save_raw07(run_dir, res["detail"])
        out_path = a.out or str(run_dir / "cat07-scores.json")
    else:
        out_path = a.out

    # transcripts are large (full tool-call payloads) -- keep them in raw07/ or --out, not stdout.
    printable = {k: v for k, v in res.items() if k != "detail"}
    printable["detail"] = [{k: v for k, v in r.items() if k != "transcript"} for r in res["detail"]]

    if out_path:
        pathlib.Path(out_path).write_text(
            json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
