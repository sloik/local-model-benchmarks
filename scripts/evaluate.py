#!/usr/bin/env python3
"""
evaluate.py — Evaluation of raw model responses by an evaluator model.

For each raw response in run_dir/raw/, retrieves criteria and gold from JSON,
sends to the evaluator model and parses the 1-5 score + justification.
Writes results to run_dir/scores.json.

Usage:
    python evaluate.py --run-dir results/2026-03-15_qwen2.5-72b--mlx-4bit/
    python evaluate.py --run-dir results/... --evaluator-type local --evaluator-model qwen/qwen3.6-27b
    python evaluate.py --run-dir results/... --evaluator-type claude

Default evaluator: Claude Sonnet via Anthropic API (ANTHROPIC_API_KEY environment variable).
Local evaluator: any model in LM Studio via --evaluator-type local.

Requirements:
    pip3 install openai anthropic --break-system-packages
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

from code_validator import validate_code, format_for_evaluator
import report as report_module
import run_provenance

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
BENCH_DIR   = SCRIPT_DIR.parent
TESTS_DIR   = BENCH_DIR / "tests"

LM_STUDIO_BASE  = os.getenv("LM_STUDIO_BASE",  "http://127.0.0.1:1234/v1")
LM_STUDIO_TOKEN = os.getenv("LM_STUDIO_TOKEN", "")

DEFAULT_LOCAL_EVALUATOR_MODEL = "qwen/qwen3.6-27b"
DEFAULT_LOCAL_EVALUATOR_CONTEXT_LENGTH = 131072
CANONICAL_LOCAL_EVALUATORS = frozenset({
    "qwen/qwen3.6-27b",
    "muse-glimmer-30b",
})
LOCAL_EVALUATOR_FALLBACK_ORDER = (
    "muse-glimmer-30b",
    "gemma-4-31b-it",
    "ornith-1.5-35b-a3b-mlx",
)

CAT_FILES = {
    "01": "cat-01-ekstrakcja.json",
    "02": "cat-02-destylacja.json",
    "03": "cat-03-reasoning.json",
    "04": "cat-04-kod.json",
    "05": "cat-05-klasyfikacja.json",
}
# CAT-06 and CAT-07 (SPEC-001) are intentionally absent from CAT_FILES / this judge dispatch path.
# CAT-07 in particular MUST NOT be scored by an LLM judge (AC3): the whole point of the category is
# verifying a model did not fabricate a tool result, and an LLM judge has the same blind spot the
# subject model does. `scripts/score_cat07.py` is CAT-07's sole, deterministic scorer.

# Cache tests to avoid loading JSON multiple times
_test_cache: dict[str, dict] = {}


def load_all_tests() -> dict[str, dict]:
    """Load all tests from all categories. Key: test_id."""
    global _test_cache
    if _test_cache:
        return _test_cache
    for cat_file in CAT_FILES.values():
        path = TESTS_DIR / cat_file
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for test in data["tests"]:
            _test_cache[test["id"]] = test
    return _test_cache


def get_test_and_prompt(test_id: str, prompt_id: str) -> tuple[dict, dict] | tuple[None, None]:
    """Return (test, prompt) for given test_id and prompt_id."""
    tests = load_all_tests()
    test = tests.get(test_id)
    if not test:
        return None, None
    for p in test["prompts"]:
        if p["id"] == prompt_id:
            return test, p
    return test, None


def build_eval_prompt(criteria: str, gold: dict, raw_response: str) -> list[dict]:
    """Build messages for the evaluator model."""
    gold_str = ""
    if gold.get("required"):
        gold_str += "Required elements in the response:\n"
        for item in gold["required"]:
            gold_str += f"  • {item}\n"
    if gold.get("forbidden"):
        gold_str += "\nForbidden elements (errors/hallucinations):\n"
        for item in gold["forbidden"]:
            gold_str += f"  • {item}\n"
    if gold.get("sample_answer"):
        gold_str += f"\nSample correct answer:\n{gold['sample_answer']}"

    user_content = f"""[SCORING CRITERIA]
{criteria}

[GOLD REFERENCE]
{gold_str}

[MODEL RESPONSE TO EVALUATE]
{raw_response}

---
Rate the response in this format (EXACTLY like this):
SCORE: [number 1-5]
JUSTIFICATION: [1-3 sentences explaining the score]"""

    return [
        {
            "role": "system",
            "content": (
                "You are a precise evaluator of language models. "
                "You evaluate the response based on the provided criteria and reference. "
                "Be rigorous — a score of 5 is excellence, not 'good enough'. "
                "Respond ONLY in the provided format: SCORE: [1-5] on the first line, "
                "JUSTIFICATION: on the second."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def parse_eval_response(text: str) -> tuple[int | None, str]:
    """Extract score (1-5) and justification from evaluator response."""
    score = None
    justification = ""

    # Look for "SCORE: N" or "Score: N" or "score: N"
    score_match = re.search(r"SCORE\s*:\s*([1-5])", text, re.IGNORECASE)
    if score_match:
        score = int(score_match.group(1))

    # Look for "JUSTIFICATION: ..."
    just_match = re.search(r"JUSTIFICATION\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if just_match:
        justification = just_match.group(1).strip()

    # Fallback: if format not found, check if there's just a digit
    if score is None:
        digit_match = re.search(r"\b([1-5])\b", text)
        if digit_match:
            score = int(digit_match.group(1))
            justification = f"[score extracted from: {text[:100]}]"

    return score, justification


VALID_PROMPT_SCORES = frozenset({1, 2, 3, 4, 5})


def is_unchecked(result: dict) -> bool:
    """True when a prompt result carries no usable verdict.

    The judge scale is 1-5, so a score outside it was never a verdict. Covers
    both artifact shapes: current runs mark an unparseable verdict with
    score=None and unchecked=True, while pre-SPEC-B06 artifacts recorded it as
    score 0 with a [PARSE ERROR] justification. UNCHECKED is its own category —
    never fold it into a pass or a fail.
    """
    if result.get("unchecked"):
        return True
    return result.get("score") not in VALID_PROMPT_SCORES


def aggregate_results(results: list[dict],
                      all_tests_data: dict) -> tuple[list[dict], dict]:
    """Build test_results and summary from prompt results.

    Unchecked prompts enter no mean, min, or aggregate. A test whose prompts are
    all unchecked gets test_score=None and leaves the category and overall means
    entirely, rather than contributing a floor value. Because that moves the
    denominator, summary["unchecked"] carries the counts that keep the mean
    honest — a reader must not need to open prompt_results to see them.
    """
    grouped: dict[str, list[dict]] = {}
    for r in results:
        grouped.setdefault(r["test_id"], []).append(r)

    tests_summary = []
    unchecked_prompts = 0
    tests_affected: list[str] = []
    tests_excluded: list[str] = []

    for tid, entries in grouped.items():
        valid     = [e for e in entries if not is_unchecked(e)]
        unchecked = [e for e in entries if is_unchecked(e)]
        unchecked_prompts += len(unchecked)
        if unchecked:
            tests_affected.append(tid)

        scoring = all_tests_data.get(tid, {}).get("scoring", "min")
        scores  = [e["score"] for e in valid]
        if scores:
            test_score = min(scores) if scoring == "min" else (sum(scores) / len(scores))
            test_score = round(test_score, 2)
        else:
            test_score = None
            tests_excluded.append(tid)

        tests_summary.append({
            "test_id":       tid,
            "category":      tid[:6],
            "scoring":       scoring,
            "prompt_scores": scores,
            "test_score":    test_score,
            "unchecked":     test_score is None,
            "unchecked_prompt_ids": [e["prompt_id"] for e in unchecked],
        })

    cat_scores: dict[str, list[float]] = {}
    for ts in tests_summary:
        if ts["test_score"] is None:
            continue
        cat_scores.setdefault(ts["category"], []).append(ts["test_score"])

    summary: dict = {}
    all_means: list[float] = []
    for cat, s_list in sorted(cat_scores.items()):
        mean = sum(s_list) / len(s_list)
        summary[cat] = {"mean": round(mean, 2), "min": min(s_list), "max": max(s_list), "n": len(s_list)}
        all_means.extend(s_list)
    if all_means:
        summary["overall"] = {"mean": round(sum(all_means) / len(all_means), 2), "n": len(all_means)}

    summary["unchecked"] = {
        "prompts":            unchecked_prompts,
        "tests_total":        len(tests_summary),
        "tests_scored":       len(all_means),
        "tests_affected":     len(tests_affected),
        "tests_excluded":     len(tests_excluded),
        "tests_affected_ids": sorted(tests_affected),
        "tests_excluded_ids": sorted(tests_excluded),
    }
    return tests_summary, summary


def select_repair_targets(artifact: dict,
                          available_prompts) -> set[tuple[str, str]]:
    """Prompts an existing artifact cannot account for.

    Two kinds qualify: a verdict that came back unparseable, and a raw response
    that was never judged at all. Both need a raw file on disk to re-judge, so
    anything absent from `available_prompts` is skipped. Every other prompt is
    left alone — repairing must not touch a verdict that was actually made.
    """
    available = set(available_prompts)
    judged = {(r["test_id"], r["prompt_id"]): r
              for r in artifact.get("prompt_results", [])}
    targets = {key for key, r in judged.items() if is_unchecked(r) and key in available}
    targets |= available - judged.keys()
    return targets


def merge_repaired_results(existing_results: list[dict], existing_errors: list[dict],
                           new_results: list[dict],
                           new_errors: list[dict]) -> tuple[list[dict], list[dict]]:
    """Fold freshly judged prompts into an existing artifact.

    Entries that were not re-judged are carried through as the same objects, in
    their original order, so every already-valid verdict stays byte-identical.
    Errors belonging to a repaired prompt are dropped — they described a missing
    verdict that now exists.
    """
    repaired = {(r["test_id"], r["prompt_id"]): r for r in new_results}

    merged = []
    seen = set()
    for r in existing_results:
        key = (r["test_id"], r["prompt_id"])
        merged.append(repaired.get(key, r))
        seen.add(key)
    merged.extend(r for r in new_results
                  if (r["test_id"], r["prompt_id"]) not in seen)

    repaired_files = {f"{tid}_{pid}.txt" for tid, pid in repaired}
    kept_errors = [
        e for e in existing_errors
        if (e.get("test_id"), e.get("prompt_id")) not in repaired
        and e.get("file") not in repaired_files
    ]
    return merged, kept_errors + new_errors


def print_summary(summary: dict) -> None:
    """Console rendering of the summary block. Only category/overall entries
    carry a mean; the unchecked block is reported separately."""
    for cat, stats in summary.items():
        if "mean" not in stats:
            continue
        bar = "█" * int(stats["mean"]) + "░" * (5 - int(stats["mean"]))
        print(f"   {cat}: {stats['mean']:.2f}/5  {bar}  (n={stats['n']})")
    unchecked = summary.get("unchecked", {})
    if unchecked.get("prompts"):
        print(f"   ⚠️  UNCHECKED: {unchecked['prompts']} prompt(s) had no parseable verdict "
              f"— {unchecked['tests_scored']}/{unchecked['tests_total']} tests scored, "
              f"{unchecked['tests_excluded']} excluded")
        if unchecked["tests_excluded_ids"]:
            print(f"      excluded: {', '.join(unchecked['tests_excluded_ids'])}")


def call_evaluator_claude(messages: list[dict], model: str = "claude-sonnet-4-6",
                          max_tokens: int = 512) -> str:
    """Call evaluator via Anthropic API."""
    try:
        import anthropic
    except ImportError:
        print("❌ Missing anthropic package. Run: pip3 install anthropic --break-system-packages")
        sys.exit(1)

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ Missing ANTHROPIC_API_KEY environment variable.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
    system_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
    user_msgs  = [m for m in messages if m["role"] != "system"]

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_msg,
        messages=user_msgs,
    )
    return response.content[0].text


def call_evaluator_local(messages: list[dict], base_url: str, token: str,
                          model: str, max_tokens: int = 512) -> str:
    """Call evaluator via LM Studio API."""
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ Missing openai package. Run: pip3 install openai --break-system-packages")
        sys.exit(1)

    client = OpenAI(api_key=token, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


# ── Argo Evaluator Mode (Claude in Cowork session) ─────────────────────────

def generate_argo_tasks(run_dir: Path, raw_dir: Path, run_info: dict, force: bool) -> Path:
    """
    Generate argo_eval_tasks.json file with tasks to be evaluated by Argo.

    Each task contains: criteria, gold, raw response, and empty fields
    score/justification to be filled in by Argo.
    """
    tasks_file = run_dir / "argo_eval_tasks.json"
    if tasks_file.exists() and not force:
        print(f"ℹ️  argo_eval_tasks.json already exists. Use --force to overwrite.")
        return tasks_file

    model_id = run_info.get("model_id", run_dir.name)
    raw_files = sorted(raw_dir.glob("*.txt"))

    if not raw_files:
        print(f"❌ No .txt files in {raw_dir}")
        sys.exit(1)

    all_tests = load_all_tests()
    tasks = []
    errors = []

    for raw_file in raw_files:
        stem  = raw_file.stem
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            print(f"  ⚠️  Unrecognized filename: {raw_file.name}")
            continue

        test_id, prompt_id = parts[0], parts[1]
        test, prompt = get_test_and_prompt(test_id, prompt_id)

        if test is None:
            errors.append({"file": raw_file.name, "error": "test not found"})
            continue
        if prompt is None:
            errors.append({"file": raw_file.name, "error": "prompt not found"})
            continue

        # Skip deprecated tests (marked with deprecated: true)
        if test.get("deprecated", False):
            print(f"  ⏭️  {test_id} — skipped (deprecated test)")
            continue

        raw_response = raw_file.read_text(encoding="utf-8").strip()

        # Inference errors immediately get score=1
        if raw_response.startswith("[ERROR:"):
            tasks.append({
                "task_id":            f"{test_id}_{prompt_id}",
                "test_id":            test_id,
                "prompt_id":          prompt_id,
                "category":           test_id[:6],
                "test_name":          test.get("name", ""),
                "prompt_description": prompt.get("description", ""),
                "criteria":           test.get("criteria", ""),
                "gold":               prompt.get("gold", {}),
                "raw_response":       raw_response,
                "score":              1,
                "justification":      f"Inference error: {raw_response[:200]}",
            })
        else:
            tasks.append({
                "task_id":            f"{test_id}_{prompt_id}",
                "test_id":            test_id,
                "prompt_id":          prompt_id,
                "category":           test_id[:6],
                "test_name":          test.get("name", ""),
                "prompt_description": prompt.get("description", ""),
                "criteria":           test.get("criteria", ""),
                "gold":               prompt.get("gold", {}),
                "raw_response":       raw_response,
                "score":              None,
                "justification":      None,
            })

    pending_count = sum(1 for t in tasks if t["score"] is None)

    output = {
        "run_id":       run_dir.name,
        "model_id":     model_id,
        "generated_at": datetime.now().isoformat(),
        "argo_status":  "pending",
        "tasks":        tasks,
        "errors":       errors,
    }

    tasks_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ Generated argo_eval_tasks.json")
    print(f"   Tasks to evaluate: {pending_count} / {len(tasks)}")
    print(f"   File: {tasks_file}")
    print(f"\n{'='*60}")
    print("🤖 ARGO — evaluation instructions:")
    print(f"{'='*60}")
    print(f"""
  Read the file and evaluate each task where score == null:

    File:  {tasks_file}

  For each task:
    1. Read the "criteria" field — this is the 1-5 scoring scale
    2. Check "gold" — required/forbidden elements
    3. Evaluate "raw_response" according to the criteria
    4. Set "score" (int 1-5) and "justification" (1-3 sentences)

  After evaluating all tasks, change "argo_status" to "complete"
  and run the finalization:

    python evaluate.py --run-dir {run_dir} \\
                       --evaluator-type argo --finalize
""")
    print("="*60)

    return tasks_file


def provenance_for_scoring(run_info: dict, model_id: str, base_url: str) -> dict:
    """R3: the same shared builder both scoring paths call — one writer, not two.

    Prefers the run's own recorded provenance (written by run_benchmark.py at
    inference time, R2's "label from the first request"); falls back to a
    fresh live lookup for a run scored against an older run_info.json that
    predates this field, and to an all-'unknown' block if even that fails —
    never a crash, since a provenance gap must not block scoring.
    """
    existing = run_info.get("provenance")
    if isinstance(existing, dict):
        try:
            run_provenance.validate_provenance(existing)
            return existing
        except run_provenance.ProvenanceSchemaError:
            pass  # fall through to a fresh live build

    try:
        model_info = fetch_native_local_model_metadata(base_url, LM_STUDIO_TOKEN).get(model_id)
    except (OSError, ValueError, json.JSONDecodeError):
        model_info = None
    return run_provenance.build_provenance_block(model_info, run_info.get("settings_gate"))


def update_leaderboard_from_score_output(output: dict) -> None:
    """Call the shared leaderboard writer directly (SPEC-003-019 R1) — no manual
    'Next step: python report.py' handoff. Never fails the scoring run: a
    leaderboard-write error is reported and swallowed, since the score
    artifact (already written to disk) is the run's real deliverable."""
    try:
        labels = report_module.determine_labels(output["summary"])
        report_module.update_leaderboard(output, labels)
    except Exception as exc:  # noqa: BLE001 — leaderboard write is best-effort
        print(f"  ⚠️  leaderboard update failed: {exc}", file=sys.stderr)


def finalize_argo_scores(run_dir: Path, run_info: dict, force: bool, no_leaderboard: bool = False,
                          base_url: str = LM_STUDIO_BASE) -> None:
    """
    Load the filled argo_eval_tasks.json and generate standard scores.json.
    """
    tasks_file = run_dir / "argo_eval_tasks.json"
    scores_file = run_dir / "scores.json"

    if not tasks_file.exists():
        print(f"❌ Missing argo_eval_tasks.json in {run_dir}")
        print(f"   First run: python evaluate.py --run-dir {run_dir} --evaluator-type argo")
        sys.exit(1)

    if scores_file.exists() and not force:
        print(f"ℹ️  scores.json already exists. Use --force to overwrite.")
        sys.exit(0)

    data = json.loads(tasks_file.read_text(encoding="utf-8"))

    # Check if all tasks are evaluated
    pending = [t for t in data["tasks"] if t["score"] is None]
    if pending:
        print(f"❌ Not all tasks evaluated — missing {len(pending)}:")
        for t in pending:
            print(f"   • {t['task_id']}")
        print(f"\n   Complete evaluations in: {tasks_file}")
        sys.exit(1)

    model_id   = data.get("model_id", run_dir.name)
    eval_model = "Argo"

    # Build prompt_results structure
    results = []
    errors  = data.get("errors", [])
    for t in data["tasks"]:
        results.append({
            "test_id":       t["test_id"],
            "prompt_id":     t["prompt_id"],
            "category":      t["category"],
            "score":         t["score"],
            "justification": t["justification"] or "",
            "evaluator":     eval_model,
        })

    # Calculate test_score (scoring="min") — unchecked prompts enter no aggregate
    tests_summary, summary = aggregate_results(results, load_all_tests())

    # Write scores.json
    model_id = run_info.get("model_id", run_dir.name)
    output = {
        "run_id":         run_dir.name,
        "model_id":       model_id,
        "eval_date":      datetime.now().strftime("%Y-%m-%d"),
        "evaluator":      eval_model,
        "evaluator_type": "argo",
        "provenance":     provenance_for_scoring(run_info, model_id, base_url),
        "prompt_results": results,
        "test_results":   tests_summary,
        "summary":        summary,
        "errors":         errors,
    }

    scores_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    n_unchecked = summary["unchecked"]["prompts"]
    print(f"\n✅ scores.json written: {scores_file}")
    print(f"   Scored: {len(results) - n_unchecked}/{len(results)} prompts | "
          f"Unchecked: {n_unchecked} | Errors: {len(errors)}")
    if summary:
        print("\n📊 Summary:")
        print_summary(summary)
    if not no_leaderboard:
        update_leaderboard_from_score_output(output)
    print(f"\n   Report: python report.py --run-dir {run_dir}")


def parse_evaluator_model_map(map_str: str) -> dict[str, str]:
    """
    Parse evaluator model map string like 'CAT-04:google/gemma-4-31b,CAT-03:other-model'.
    Returns dict: {"CAT-04": "google/gemma-4-31b", ...}
    """
    result = {}
    if not map_str:
        return result
    for pair in map_str.split(","):
        pair = pair.strip()
        if ":" not in pair:
            print(f"  ⚠️  Invalid evaluator-model-map entry (missing ':'): {pair}")
            continue
        cat, model = pair.split(":", 1)
        cat = cat.strip().upper()
        model = model.strip()
        if not cat or not model:
            print(f"  ⚠️  Invalid evaluator-model-map entry: {pair}")
            continue
        result[cat] = model
    return result


# Qwen3.6 is the calibrated primary judge across every category. Retain the map
# mechanism for explicit experiments, but do not silently route a category to a
# different judge: that would make the score set internally incomparable.
DEFAULT_LOCAL_EVALUATOR_MODEL_MAP: dict[str, str] = {}


def resolve_evaluator_model_map(map_str: str, evaluator_type: str) -> dict[str, str]:
    """CLI map merged over the local defaults; CLI entries always win.
    Defaults apply only to the local evaluator (their model ids are LM Studio models)."""
    result = parse_evaluator_model_map(map_str)
    if evaluator_type == "local":
        for cat, model in DEFAULT_LOCAL_EVALUATOR_MODEL_MAP.items():
            result.setdefault(cat, model)
    return result


def fetch_available_local_models(base_url: str, token: str) -> set[str]:
    """Return downloaded LM Studio LLM keys from the native model catalog."""
    api_root = base_url.rstrip("/")
    if api_root.endswith("/v1"):
        api_root = api_root[:-3]
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(f"{api_root}/api/v1/models", headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read())
    models = payload.get("models", payload.get("data", []))
    return {
        model.get("key", model.get("id", ""))
        for model in models
        if model.get("type", "llm") == "llm"
        and model.get("key", model.get("id", ""))
    }


def fetch_native_local_model_metadata(base_url: str, token: str) -> dict[str, dict]:
    """Return native LM Studio metadata keyed by model key."""
    api_root = base_url.rstrip("/")
    if api_root.endswith("/v1"):
        api_root = api_root[:-3]
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    request = urllib.request.Request(f"{api_root}/api/v1/models", headers=headers)
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read())
    models = payload.get("models", payload.get("data", []))
    return {
        model.get("key", model.get("id", "")): model
        for model in models
        if model.get("type", "llm") == "llm" and model.get("key", model.get("id", ""))
    }


def validate_local_evaluator_preflight(
    evaluator_models: set[str],
    native_models: dict[str, dict],
    minimum_context_length: int,
    *,
    allow_experimental_evaluator: bool,
) -> dict:
    """Fail closed unless every local judge is approved, loaded, and context-ready."""
    if minimum_context_length <= 0:
        raise ValueError("local evaluator context baseline must be positive")

    records = []
    for model_id in sorted(evaluator_models):
        if not allow_experimental_evaluator and model_id not in CANONICAL_LOCAL_EVALUATORS:
            raise ValueError(
                f"non-canonical local evaluator '{model_id}'; use "
                "--allow-experimental-evaluator only for an explicitly experimental score"
            )
        model = native_models.get(model_id)
        if model is None:
            raise ValueError(f"local evaluator '{model_id}' is absent from LM Studio native metadata")
        instances = model.get("loaded_instances") or []
        if not instances:
            raise ValueError(
                f"local evaluator '{model_id}' is not loaded; load it before scoring"
            )
        active_context = (instances[0].get("config") or {}).get("context_length")
        if not isinstance(active_context, int):
            raise ValueError(f"local evaluator '{model_id}' has no integer active context length")
        if active_context < minimum_context_length:
            raise ValueError(
                f"local evaluator '{model_id}' context {active_context} is below required "
                f"baseline {minimum_context_length}"
            )
        max_context = model.get("max_context_length")
        if isinstance(max_context, int) and max_context < minimum_context_length:
            raise ValueError(
                f"local evaluator '{model_id}' maximum context {max_context} is below "
                f"required baseline {minimum_context_length}"
            )
        records.append({
            "model_id": model_id,
            "active_context_length": active_context,
            "max_context_length": max_context,
        })

    return {
        "policy": "canonical_local_judge_with_minimum_context",
        "minimum_context_length": minimum_context_length,
        "allow_experimental_evaluator": allow_experimental_evaluator,
        "evaluators": records,
    }


def resolve_local_evaluator_selection(
    evaluator_model: str,
    evaluator_model_map: dict[str, str],
    subject_model: str,
    available_models: set[str],
    *,
    allow_self_evaluation: bool,
) -> tuple[str, dict[str, str]]:
    """Validate local judges and replace self-evaluation with an installed peer."""
    selected = {evaluator_model, *evaluator_model_map.values()}
    missing = sorted(selected - available_models)
    if missing:
        raise ValueError(
            "Local evaluator model(s) are not installed: "
            f"{', '.join(missing)}. Available LLMs: {', '.join(sorted(available_models))}"
        )

    if allow_self_evaluation:
        return evaluator_model, dict(evaluator_model_map)

    def alternative() -> str:
        for candidate in LOCAL_EVALUATOR_FALLBACK_ORDER:
            if candidate != subject_model and candidate in available_models:
                return candidate
        raise ValueError(
            f"No installed independent evaluator is available for subject {subject_model}."
        )

    resolved_model = alternative() if evaluator_model == subject_model else evaluator_model
    resolved_map = {
        category: alternative() if model == subject_model else model
        for category, model in evaluator_model_map.items()
    }
    return resolved_model, resolved_map


def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark — Evaluator")
    parser.add_argument("--run-dir",          required=True,
                        help="Folder with results from run_benchmark.py (e.g., results/2026-03-15_...)")
    parser.add_argument("--evaluator-type",   default="claude",
                        choices=["claude", "local", "argo"],
                        help="Evaluator type: claude (Anthropic API), local (LM Studio), argo (Claude in Cowork)")
    parser.add_argument("--evaluator-model",  default=None,
                        help="Evaluator model (default: claude-sonnet-4-6 or qwen/qwen3.6-27b)")
    parser.add_argument("--evaluator-model-map", default=None,
                        help="Per-category evaluator model overrides. Format: CAT-XX:model_id,CAT-YY:model_id. "
                             "Categories not in the map use --evaluator-model default. "
                             "Example: CAT-04:google/gemma-4-31b")
    parser.add_argument("--base-url",              default=LM_STUDIO_BASE,
                        help=f"LM Studio API base URL (for --evaluator-type local)")
    parser.add_argument("--evaluator-max-tokens",  type=int, default=None,
                        help="Max tokens per evaluator response "
                             "(default: 8192 for local, 512 for claude/argo)")
    parser.add_argument("--force",            action="store_true",
                        help="Overwrite existing scores.json / argo_eval_tasks.json")
    parser.add_argument("--repair-unchecked", action="store_true",
                        help="Re-judge only the unchecked or missing prompts in an existing "
                             "score artifact and merge them back in, leaving every verdict "
                             "that was actually made untouched")
    parser.add_argument("--output-file", default="scores.json",
                        help="Score filename within --run-dir (default: scores.json)")
    parser.add_argument("--allow-self-evaluation", action="store_true",
                        help="Research-only: permit a subject model to judge its own responses")
    parser.add_argument("--allow-experimental-evaluator", action="store_true",
                        help="Permit a non-canonical local judge; recorded in the score artifact")
    parser.add_argument("--evaluator-context-length", type=int,
                        default=DEFAULT_LOCAL_EVALUATOR_CONTEXT_LENGTH,
                        help="Minimum active context required for local judges (default: 131072)")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate local judge selection and context, then exit without scoring")
    parser.add_argument("--finalize",         action="store_true",
                        help="[only with --evaluator-type argo] Load filled argo_eval_tasks.json and write scores.json")
    parser.add_argument("--no-leaderboard", action="store_true",
                        help="Do not update leaderboard.csv after writing the score artifact")
    args = parser.parse_args()

    if args.repair_unchecked and args.evaluator_type == "argo":
        parser.error("--repair-unchecked is not supported for --evaluator-type argo")
    if args.preflight_only and args.evaluator_type != "local":
        parser.error("--preflight-only is available only with --evaluator-type local")

    # Default token limit depends on evaluator type:
    #   local (LM Studio) → 8192 (less = noticeable quality drop)
    #   claude / argo     → 512  (evaluator response is brief; API cost savings)
    if args.evaluator_max_tokens is None:
        args.evaluator_max_tokens = 8192 if args.evaluator_type == "local" else 512

    run_dir = Path(args.run_dir)
    raw_dir = run_dir / "raw"
    output_name = Path(args.output_file).name
    if output_name != args.output_file or not output_name.endswith(".json"):
        parser.error("--output-file must be a plain .json filename")
    scores_file = run_dir / output_name

    # ── Argo Mode ────────────────────────────────────────────────────────────
    if args.evaluator_type == "argo":
        # Load run metadata
        run_info_file = run_dir / "run_info.json"
        run_info = json.loads(run_info_file.read_text()) if run_info_file.exists() else {}

        if args.finalize:
            print(f"🤖 Argo — finalizing scores from argo_eval_tasks.json")
            finalize_argo_scores(run_dir, run_info, force=args.force, no_leaderboard=args.no_leaderboard,
                                  base_url=args.base_url)
        else:
            if not raw_dir.exists():
                print(f"❌ Folder raw/ does not exist: {raw_dir}")
                sys.exit(1)
            print(f"🤖 Argo — generating evaluation tasks")
            generate_argo_tasks(run_dir, raw_dir, run_info, force=args.force)
        return

    if not raw_dir.exists():
        print(f"❌ Folder raw/ does not exist: {raw_dir}")
        sys.exit(1)

    # Repair reads the existing artifact and writes back a merged one, so it is
    # the one mode where an existing file is the precondition rather than a block.
    existing_artifact: dict = {}
    if args.repair_unchecked:
        if not scores_file.exists():
            print(f"❌ --repair-unchecked needs an existing artifact: {scores_file}")
            sys.exit(1)
        existing_artifact = json.loads(scores_file.read_text(encoding="utf-8"))
    elif scores_file.exists() and not args.force:
        print(f"ℹ️  scores.json already exists. Use --force to overwrite, "
              f"or --repair-unchecked to re-judge only unchecked prompts.")
        sys.exit(0)

    # ── Parse evaluator model map (CLI over SPEC-B04 local defaults) ──────────
    evaluator_model_map = resolve_evaluator_model_map(
        args.evaluator_model_map or "", args.evaluator_type
    )

    # ── Set up default evaluator ──────────────────────────────────────────────
    if args.evaluator_type == "claude":
        eval_model = args.evaluator_model or "claude-sonnet-4-6"
        def make_call_eval(model_id):
            return lambda msgs: call_evaluator_claude(
                msgs, model=model_id, max_tokens=args.evaluator_max_tokens
            )
        print(f"🧑‍⚖️  Evaluator: Anthropic {eval_model} (max_tokens={args.evaluator_max_tokens})")
    else:
        eval_model = args.evaluator_model or DEFAULT_LOCAL_EVALUATOR_MODEL
        def make_call_eval(model_id):
            return lambda msgs: call_evaluator_local(
                msgs, base_url=args.base_url, token=LM_STUDIO_TOKEN, model=model_id,
                max_tokens=args.evaluator_max_tokens
            )
        print(f"🧑‍⚖️  Evaluator: local {eval_model} (max_tokens={args.evaluator_max_tokens})")

    if evaluator_model_map:
        print(f"📋 Per-category overrides: {evaluator_model_map}")

    # ── Load run metadata ──────────────────────────────────────────────────────
    run_info_file = run_dir / "run_info.json"
    run_info = json.loads(run_info_file.read_text()) if run_info_file.exists() else {}
    model_id = run_info.get("model_id", run_dir.name)

    # ── Validate installed judges + prevent accidental self-evaluation ────────
    local_evaluator_preflight = None
    if args.evaluator_type == "local":
        try:
            available_models = fetch_available_local_models(args.base_url, LM_STUDIO_TOKEN)
            old_eval_model = eval_model
            eval_model, evaluator_model_map = resolve_local_evaluator_selection(
                eval_model,
                evaluator_model_map,
                model_id,
                available_models,
                allow_self_evaluation=args.allow_self_evaluation,
            )
            local_evaluator_preflight = validate_local_evaluator_preflight(
                {eval_model, *evaluator_model_map.values()},
                fetch_native_local_model_metadata(args.base_url, LM_STUDIO_TOKEN),
                args.evaluator_context_length,
                allow_experimental_evaluator=args.allow_experimental_evaluator,
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            parser.error(str(exc))
        if old_eval_model != eval_model:
            print(f"  ⚠️  Self-evaluation detected: {old_eval_model} cannot evaluate itself")
            print(f"  ↳  Swapped default evaluator → {eval_model}")
        print(
            "✅ Local evaluator preflight passed: "
            f"context ≥ {local_evaluator_preflight['minimum_context_length']} | "
            + ", ".join(
                f"{record['model_id']}={record['active_context_length']}"
                for record in local_evaluator_preflight["evaluators"]
            )
        )
        if args.preflight_only:
            print(json.dumps(local_evaluator_preflight, indent=2))
            return

    # ── Set up code validator client (for CAT-04 LLM extraction) ──────────────
    code_validator_client = None
    code_validator_model = None
    if args.evaluator_type == "local":
        try:
            from openai import OpenAI
            code_validator_client = OpenAI(api_key=LM_STUDIO_TOKEN, base_url=args.base_url)
            code_validator_model = eval_model
            print(f"🔍 Code validator: LLM extraction via {code_validator_model}")
        except ImportError:
            print("  ⚠️  openai not installed — code extraction will use regex only")

    # ── Evaluate each raw file ────────────────────────────────────────────────
    raw_files = sorted(raw_dir.glob("*.txt"))
    if not raw_files:
        print(f"❌ No .txt files in {raw_dir}")
        sys.exit(1)

    print(f"📁 Run: {run_dir.name}")
    print(f"📄 Files to evaluate: {len(raw_files)}")

    repair_targets = None
    if args.repair_unchecked:
        # Only prompts the loop can actually judge. A deprecated or unknown test
        # never produces a prompt_result, so offering it as a repair target would
        # make it a permanent one — "nothing to repair" would never be reachable.
        available = []
        for f in raw_files:
            parts = f.stem.rsplit("_", 1)
            if len(parts) != 2:
                continue
            test, prompt = get_test_and_prompt(parts[0], parts[1])
            if test is None or prompt is None or test.get("deprecated", False):
                continue
            available.append((parts[0], parts[1]))

        repair_targets = select_repair_targets(existing_artifact, available)
        if not repair_targets:
            print("✅ Nothing to repair — every prompt already carries a verdict.")
            return
        print(f"🔧 Repair mode: re-judging {len(repair_targets)} of {len(available)} prompt(s); "
              f"all other verdicts are left untouched")

    results = []
    errors  = []

    for raw_file in raw_files:
        # Parse test_id and prompt_id from filename, e.g., "CAT-05-001_p1.txt"
        stem   = raw_file.stem  # "CAT-05-001_p1"
        parts  = stem.rsplit("_", 1)
        if len(parts) != 2:
            print(f"  ⚠️  Unrecognized filename: {raw_file.name}")
            continue

        test_id, prompt_id = parts[0], parts[1]

        if repair_targets is not None and (test_id, prompt_id) not in repair_targets:
            continue

        test, prompt = get_test_and_prompt(test_id, prompt_id)

        if test is None:
            print(f"  ⚠️  Test {test_id} not found in JSON")
            errors.append({"file": raw_file.name, "error": "test not found"})
            continue
        if prompt is None:
            print(f"  ⚠️  Prompt {prompt_id} not found in {test_id}")
            errors.append({"file": raw_file.name, "error": "prompt not found"})
            continue

        # Skip deprecated tests (marked with deprecated: true)
        if test.get("deprecated", False):
            print(f"  ⏭️  {test_id}_{prompt_id} — skipped (deprecated test)")
            continue

        raw_response = raw_file.read_text(encoding="utf-8").strip()
        category = test_id[:6]  # e.g., "CAT-04"

        # An empty final answer is a failed inference. Sending it to an LLM
        # judge lets the judge reconstruct an answer from the rubric/gold and
        # can award a false-positive score.
        if not raw_response:
            print(f"  ❌ {test_id}_{prompt_id} — empty response, score=1")
            results.append({
                "test_id": test_id,
                "prompt_id": prompt_id,
                "category": category,
                "score": 1,
                "justification": "Inference returned an empty final response.",
                "evaluator": "deterministic-empty-response-check",
            })
            continue

        # Determine which evaluator model to use for this category
        cat_eval_model = evaluator_model_map.get(category, eval_model)
        call_eval = make_call_eval(cat_eval_model)

        # Check if it's an inference error
        if raw_response.startswith("[ERROR:"):
            print(f"  ❌ {test_id}_{prompt_id} — inference error, score=1")
            results.append({
                "test_id":    test_id,
                "prompt_id":  prompt_id,
                "category":   category,
                "score":      1,
                "justification": f"Inference error: {raw_response[:200]}",
                "evaluator":  cat_eval_model,
            })
            continue

        eval_model_tag = f" [{cat_eval_model}]" if cat_eval_model != eval_model else ""
        print(f"  → {test_id}_{prompt_id}{eval_model_tag}", end="", flush=True)

        # ── CAT-04 code pre-filter ────────────────────────────────────────
        code_validation_note = ""
        if category == "CAT-04":
            validation = validate_code(
                raw_response,
                client=code_validator_client,
                model=code_validator_model,
            )
            code_validation_note = format_for_evaluator(validation)
            if code_validation_note:
                method = validation.get("extraction_method", "?")
                print(f" [code:{method}→{'FAIL' if validation.get('error') else 'OK'}]", end="")

        # Build prompt for evaluator
        eval_messages = build_eval_prompt(
            criteria=test["criteria"],
            gold=prompt["gold"],
            raw_response=raw_response + code_validation_note,
        )

        try:
            eval_response = call_eval(eval_messages)
            score, justification = parse_eval_response(eval_response)

            if score is None:
                # No verdict was made. Recording a 0 here would average a failed
                # measurement into the result — under scoring="min" it drives the
                # whole test to zero. Keep it UNCHECKED and preserve the raw text,
                # which is the only evidence of what the judge actually said.
                raw_head = eval_response[:300]
                print(f" ⚠️  UNCHECKED — unable to parse score from: {eval_response[:100]}")
                errors.append({
                    "file":              raw_file.name,
                    "test_id":           test_id,
                    "prompt_id":         prompt_id,
                    "error":             "unparseable judge verdict",
                    "raw_response_head": raw_head,
                })
                results.append({
                    "test_id":       test_id,
                    "prompt_id":     prompt_id,
                    "category":      category,
                    "score":         None,
                    "unchecked":     True,
                    "justification": f"[UNCHECKED] no parseable verdict; raw: {raw_head}",
                    "evaluator":     cat_eval_model,
                })
                continue

            print(f" → {score}/5")
            results.append({
                "test_id":       test_id,
                "prompt_id":     prompt_id,
                "category":      category,
                "score":         score,
                "justification": justification,
                "evaluator":     cat_eval_model,
            })
        except Exception as e:
            print(f" ❌ evaluation error: {e}")
            errors.append({"file": raw_file.name, "error": str(e)})

    # ── Merge into the existing artifact when repairing ───────────────────────
    if repair_targets is not None:
        results, errors = merge_repaired_results(
            existing_artifact.get("prompt_results", []),
            existing_artifact.get("errors", []),
            results,
            errors,
        )

    # ── Aggregate (scoring="min"); unchecked prompts enter no aggregate ───────
    tests_summary, summary = aggregate_results(results, load_all_tests())

    # ── Write score artifact ─────────────────────────────────────────────────
    # Collect unique evaluator models used
    evaluators_used = sorted(set(r["evaluator"] for r in results))
    self_evaluated = model_id in {eval_model, *evaluator_model_map.values()}

    output = {
        "run_id":        run_dir.name,
        "model_id":      model_id,
        "eval_date":     datetime.now().strftime("%Y-%m-%d"),
        "evaluator":     eval_model,
        "evaluator_type": args.evaluator_type,
        "self_evaluation_allowed": args.allow_self_evaluation,
        "evaluation_independence": (
            "non-independent-self-evaluation" if self_evaluated else "independent"
        ),
        "evaluator_model_map": evaluator_model_map if evaluator_model_map else None,
        "evaluators_used": evaluators_used,
        "local_evaluator_preflight": local_evaluator_preflight,
        "provenance": provenance_for_scoring(run_info, model_id, args.base_url),
        "prompt_results": results,
        "test_results":  tests_summary,
        "summary":       summary,
        "errors":        errors,
    }

    scores_file.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    n_unchecked = summary["unchecked"]["prompts"]
    print(f"\n✅ score artifact written: {scores_file}")
    print(f"   Scored: {len(results) - n_unchecked}/{len(results)} prompts | "
          f"Unchecked: {n_unchecked} | Errors: {len(errors)}")
    if n_unchecked:
        print(f"   Repair with: python evaluate.py --run-dir {run_dir} "
              f"--output-file {output_name} --repair-unchecked")
    if summary:
        print("\n📊 Summary:")
        print_summary(summary)
    if not args.no_leaderboard:
        update_leaderboard_from_score_output(output)
    print(f"\n   Report: python report.py --run-dir {run_dir}")


if __name__ == "__main__":
    main()
