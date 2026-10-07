#!/usr/bin/env python3
"""
run_benchmark.py — Inference runner for local LLM model benchmarking.

Loads tests from tests/cat-XX-*.json, sends each prompt to LM Studio API,
and saves raw responses to results/{run_id}/raw/{test_id}_{prompt_id}.txt

Usage:
    python run_benchmark.py --model ornith-1.5-35b-a3b-mlx
    python run_benchmark.py --model gemma-4-31b-it --categories 05 01
    python run_benchmark.py --model qwen/qwen3.6-27b --test-ids CAT-05-001

Requirements:
    pip3 install openai --break-system-packages

LM Studio Configuration:
    Endpoint: http://127.0.0.1:1234/v1 by default.
    Override LM_STUDIO_BASE when invoking from another host or VM.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
import run_provenance  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
BENCH_DIR    = SCRIPT_DIR.parent
TESTS_DIR    = BENCH_DIR / "tests"
RESULTS_DIR  = BENCH_DIR / "results"

# ── LM Studio Configuration ──────────────────────────────────────────────────
LM_STUDIO_BASE  = os.getenv("LM_STUDIO_BASE",  "http://127.0.0.1:1234/v1")
LM_STUDIO_TOKEN = os.getenv("LM_STUDIO_TOKEN", "")
DEFAULT_CONTEXT_LENGTH = 131072

CAT_FILES = {
    "01": "cat-01-ekstrakcja.json",
    "02": "cat-02-destylacja.json",
    "03": "cat-03-reasoning.json",
    "04": "cat-04-kod.json",
    "05": "cat-05-klasyfikacja.json",
}
# CAT-06 (source fidelity) and CAT-07 (tool use / agentic retrieval, SPEC-001) are intentionally
# absent from CAT_FILES. This dict feeds the single-turn, no-`tools=`-param inference path above,
# which both categories cannot use: CAT-06 needs deterministic refusal detection, and CAT-07 needs
# a multi-turn tool-calling loop with a hermetic tool executor (`scripts/cat07_harness.py`). Both
# are standalone deterministic-only scorers -- `scripts/score_cat06.py` and `scripts/score_cat07.py`
# -- that run their own inference and write into the same `results/<run_id>/` shape this script
# produces (see SPEC.md §2 CAT-07 scoring-caveat note). Do not add "06"/"07" here.


def load_tests(categories: list[str], test_ids: list[str] | None) -> list[dict]:
    """Load tests from JSON files. Filters by categories and/or test IDs.

    Skips tests marked with deprecated: true to prevent stale tests from being scored.
    """
    all_tests = []
    for cat in categories:
        cat_file = TESTS_DIR / CAT_FILES[cat]
        if not cat_file.exists():
            print(f"⚠️  Category file does not exist: {cat_file}")
            continue
        with open(cat_file) as f:
            data = json.load(f)
        for test in data["tests"]:
            # Skip deprecated tests (marked with deprecated: true)
            if test.get("deprecated", False):
                continue
            if test_ids is None or test["id"] in test_ids:
                test["_category_file"] = str(cat_file)
                all_tests.append(test)
    return all_tests


def sanitize_model_id(model_id: str) -> str:
    """Replace slashes and spaces with hyphens — safe folder name."""
    return model_id.replace("/", "--").replace(" ", "_").replace(":", "-")


def validate_model_settings(model_info: dict, expected_context_length: int,
                            max_tokens: int, retry_max_tokens: int) -> dict:
    """Validate the active LM Studio settings before a live benchmark starts."""
    if not model_info:
        raise ValueError("selected model is absent from LM Studio native model metadata")
    instances = model_info.get("loaded_instances") or []
    if not instances:
        raise ValueError(
            f"model {model_info.get('key', '<unknown>')} has no loaded instance; "
            "load it with the required context before running"
        )
    actual_context = (instances[0].get("config") or {}).get("context_length")
    if not isinstance(actual_context, int):
        raise ValueError("LM Studio did not report an integer active context length")
    if actual_context < expected_context_length:
        raise ValueError(
            f"context {actual_context} is below required minimum {expected_context_length}; "
            "reload the model with at least the required context before running"
        )
    model_max_context = model_info.get("max_context_length")
    if isinstance(model_max_context, int) and model_max_context < expected_context_length:
        raise ValueError(
            f"required context {expected_context_length} exceeds model maximum "
            f"{model_max_context}"
        )
    if max_tokens <= 0 or retry_max_tokens <= 0:
        raise ValueError("generation ceilings must be positive")
    if retry_max_tokens < max_tokens:
        raise ValueError(
            f"retry ceiling {retry_max_tokens} is smaller than standard ceiling {max_tokens}"
        )
    return {
        "policy": "minimum_context_and_equal_generation_ceilings",
        "minimum_context_length": expected_context_length,
        "actual_context_length": actual_context,
        "model_max_context_length": model_max_context,
        "max_tokens": max_tokens,
        "retry_max_tokens": retry_max_tokens,
    }


def native_models_url(base_url: str) -> str:
    """Derive LM Studio's native metadata endpoint from an OpenAI-compatible base URL."""
    marker = "/v1"
    root = base_url[:-len(marker)] if base_url.endswith(marker) else base_url.rstrip("/")
    return f"{root}/api/v1/models"


def fetch_native_model_info(base_url: str, model_id: str) -> dict:
    """Read one model's native LM Studio metadata, including its active load config."""
    try:
        with urllib.request.urlopen(native_models_url(base_url), timeout=10) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read LM Studio native model metadata: {exc}") from exc
    return next((model for model in payload.get("models", []) if model.get("key") == model_id), {})


def run_inference(client: Any, model_id: str, messages: list[dict],
                  max_tokens: int = 32768, temperature: float = 0.1,
                  reasoning_effort: str | None = None,
                  top_p: float | None = None,
                  top_k: int | None = None,
                  reasoning_strength: str | None = None) -> tuple[str, float, str]:
    """Send prompt to model. Returns (response, elapsed_seconds, finish_reason)."""
    t0 = time.time()
    try:
        request_messages = messages
        if reasoning_strength is not None:
            request_messages = [
                {"role": "system", "content": f"Reasoning strength: {reasoning_strength}."},
                *messages,
            ]
        request = {
            "model": model_id,
            "messages": request_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if reasoning_effort is not None:
            request["reasoning_effort"] = reasoning_effort
        if top_p is not None:
            request["top_p"] = top_p
        if top_k is not None:
            # The OpenAI-compatible wire format accepts top_k, but the OpenAI
            # Python client's typed Chat Completions signature does not.
            request["extra_body"] = {"top_k": top_k}
        response = client.chat.completions.create(
            **request,
        )
        elapsed = time.time() - t0
        choice = response.choices[0]
        return choice.message.content or "", elapsed, choice.finish_reason or "unknown"
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    ❌ API Error: {e}")
        return f"[ERROR: {e}]", elapsed, "error"


def acquire_keep_awake() -> subprocess.Popen | None:
    """Prevent macOS idle/system sleep until this process exits."""
    if sys.platform != "darwin":
        return None
    caffeinate = shutil.which("caffeinate")
    if caffeinate is None:
        raise RuntimeError("macOS caffeinate command is unavailable")
    # -w ties the assertion to this process; it is released automatically even
    # if the benchmark is interrupted or killed.
    process = subprocess.Popen(
        [caffeinate, "-ims", "-w", str(os.getpid())],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("☕ macOS sleep prevention active for this benchmark process")
    return process


def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark — Inference Runner")
    parser.add_argument("--model",       required=True,
                        help="Model ID in LM Studio (e.g. mlx-community/Qwen2.5-72B-Instruct-4bit)")
    parser.add_argument("--categories",  nargs="+", default=list(CAT_FILES.keys()),
                        choices=list(CAT_FILES.keys()),
                        help="Categories to run (default: all)")
    parser.add_argument("--test-ids",    nargs="+",
                        help="Run only selected tests (e.g. CAT-05-001 CAT-01-001)")
    parser.add_argument("--max-tokens",  type=int, default=32768,
                        help="Standard generation ceiling (default: 32768)")
    parser.add_argument("--retry-max-tokens", type=int, default=81920,
                        help="One retry ceiling for empty or length-truncated responses (default: 81920; 0 disables)")
    parser.add_argument("--context-length", type=int, default=DEFAULT_CONTEXT_LENGTH,
                        help="Minimum active LM Studio context required for the run (default: 131072)")
    parser.add_argument("--reasoning-effort", choices=["none", "minimal", "low", "medium", "high", "xhigh"],
                        help="Optional LM Studio/OpenAI reasoning mode recorded in run_info.json")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Sampling temperature (default: 0.1)")
    parser.add_argument("--top-p", type=float,
                        help="Optional nucleus-sampling probability")
    parser.add_argument("--top-k", type=int,
                        help="Optional top-k sampling limit")
    parser.add_argument("--reasoning-strength", choices=["low", "medium", "high", "xhigh"],
                        help="Optional model-card reasoning strength injected as a system instruction")
    parser.add_argument("--base-url",    default=LM_STUDIO_BASE,
                        help=f"LM Studio API base URL (default: {LM_STUDIO_BASE})")
    parser.add_argument("--run-tag",     default="",
                        help="Tag distinguishing multiple runs on the same day "
                             "(e.g. ctx4k, v2). Appended to run_id: {date}_{model}_{tag}")
    parser.add_argument("--force",       action="store_true",
                        help="Overwrite existing raw response files (re-run inference)")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Show what would be sent — without API calls")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Validate LM Studio model and equal-settings gate, then exit")
    parser.add_argument("--no-keep-awake", action="store_true",
                        help="Do not run macOS caffeinate during the benchmark")
    args = parser.parse_args()

    # ── Load tests ──────────────────────────────────────────────────────────────
    tests = load_tests(args.categories, args.test_ids)
    if not tests:
        print("❌ No tests to run. Check --categories and --test-ids.")
        sys.exit(1)

    n_prompts = sum(len(t["prompts"]) for t in tests)
    print(f"📋 Tests: {len(tests)} tests, {n_prompts} prompts")
    print(f"🤖 Model: {args.model}")

    # ── Create run_id and results folder ───────────────────────────────────────
    date_str  = datetime.now().strftime("%Y-%m-%d")
    model_str = sanitize_model_id(args.model)
    tag_str   = f"_{args.run_tag}" if args.run_tag else ""
    run_id    = f"{date_str}_{model_str}{tag_str}"
    run_dir   = RESULTS_DIR / run_id
    raw_dir   = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Results: {run_dir}")
    print(f"BENCHMARK_RUN_DIR={run_dir}", flush=True)  # machine-parseable for run_all.sh

    if args.dry_run:
        print("\n🔍 DRY RUN — prompts that will be sent:")
        for test in tests:
            for p in test["prompts"]:
                print(f"  {test['id']}_{p['id']}: {len(p['messages'])} messages")
        sys.exit(0)

    # ── Connect to LM Studio ────────────────────────────────────────────────────
    try:
        from openai import OpenAI
    except ImportError:
        print("❌ Missing openai package. Install project dependencies before a live run.")
        sys.exit(1)
    client = OpenAI(api_key=LM_STUDIO_TOKEN, base_url=args.base_url)

    # Check availability
    try:
        models = client.models.list()
        available = [m.id for m in models.data]
        if args.model not in available:
            print(f"⚠️  Model '{args.model}' is not loaded in LM Studio.")
            print(f"   Available models: {available}")
            sys.exit(1)
        print(f"✅ LM Studio available. Loaded model: {args.model}")
    except Exception as e:
        print(f"❌ Cannot connect to LM Studio ({args.base_url}): {e}")
        print("   Start LM Studio's local server or override LM_STUDIO_BASE.")
        sys.exit(1)

    # Native metadata is the only authoritative source for the active context.
    try:
        model_info = fetch_native_model_info(args.base_url, args.model)
        settings_gate = validate_model_settings(
            model_info, args.context_length, args.max_tokens, args.retry_max_tokens
        )
        print(
            "✅ Settings gate passed: "
            f"context={settings_gate['actual_context_length']} | "
            f"max_tokens={settings_gate['max_tokens']} | "
            f"retry={settings_gate['retry_max_tokens']}"
        )
    except ValueError as e:
        print(f"❌ Settings gate failed: {e}")
        sys.exit(1)
    if args.preflight_only:
        print(json.dumps(settings_gate, indent=2))
        sys.exit(0)

    try:
        keep_awake_process = None if args.no_keep_awake else acquire_keep_awake()
    except RuntimeError as e:
        print(f"❌ Cannot establish sleep prevention: {e}")
        sys.exit(1)

    # ── Run inference ───────────────────────────────────────────────────────────
    # SPEC-003-020: stamp runtime/context provenance from the first request —
    # not retroactively (recorded benchmark reliability finding). Validated against the
    # schema before the first write; a malformed block fails the run rather
    # than persisting an artifact that would silently fail later validation.
    provenance = run_provenance.build_provenance_block(model_info, settings_gate)
    run_provenance.validate_provenance(provenance)

    run_meta = {
        "run_id":     run_id,
        "model_id":   args.model,
        "start_time": datetime.now().isoformat(),
        "base_url":   args.base_url,
        "max_tokens": args.max_tokens,
        "retry_max_tokens": args.retry_max_tokens,
        "settings_gate": settings_gate,
        "provenance": provenance,
        "keep_awake": keep_awake_process is not None,
        "reasoning_effort": args.reasoning_effort,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "reasoning_strength": args.reasoning_strength,
        "tests_run":  [],
    }

    # Persist the validated settings before the first prompt.  A long run may be
    # interrupted, but its artifact must still prove which comparison settings
    # passed the gate.
    (run_dir / "run_info.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total_elapsed = 0.0
    for test in tests:
        print(f"\n📝 {test['id']}: {test['name']}")
        for p in test["prompts"]:
            out_file = raw_dir / f"{test['id']}_{p['id']}.txt"
            if out_file.exists() and not args.force:
                print(f"  ⏭️  {p['id']} — skipped (file exists, use --force to re-run)")
                continue

            # LM Studio can reconfigure or reload an instance while a long
            # benchmark is running. Re-check immediately before every prompt
            # so mid-run context drift cannot silently contaminate results.
            try:
                current_info = fetch_native_model_info(args.base_url, args.model)
                settings_gate = validate_model_settings(
                    current_info, args.context_length, args.max_tokens, args.retry_max_tokens
                )
            except ValueError as e:
                run_meta["settings_gate"]["runtime_failure"] = str(e)
                run_meta["settings_gate"]["runtime_failure_time"] = datetime.now().isoformat()
                (run_dir / "run_info.json").write_text(
                    json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(f"❌ Settings gate failed during run: {e}")
                sys.exit(1)

            print(f"  → {p['id']} ({p['description'][:60]}...)", end="", flush=True)
            response_text, elapsed, finish_reason = run_inference(
                client, args.model, p["messages"], max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                reasoning_strength=args.reasoning_strength,
            )
            total_elapsed += elapsed

            retry = None
            needs_retry = not response_text.strip() or finish_reason == "length"
            if needs_retry and args.retry_max_tokens > args.max_tokens:
                reason = "empty" if not response_text.strip() else "length"
                print(f" ↻ {reason}; retrying at {args.retry_max_tokens}", end="", flush=True)
                retry_text, retry_elapsed, retry_finish_reason = run_inference(
                    client, args.model, p["messages"], max_tokens=args.retry_max_tokens,
                    reasoning_effort=args.reasoning_effort,
                    temperature=args.temperature, top_p=args.top_p, top_k=args.top_k,
                    reasoning_strength=args.reasoning_strength,
                )
                total_elapsed += retry_elapsed
                retry = {
                    "reason": reason,
                    "max_tokens": args.retry_max_tokens,
                    "elapsed_s": round(retry_elapsed, 2),
                    "finish_reason": retry_finish_reason,
                    "output_len": len(retry_text),
                }
                response_text = retry_text
                elapsed += retry_elapsed
                finish_reason = retry_finish_reason

            out_file.write_text(response_text, encoding="utf-8")
            print(f" ✓ ({elapsed:.1f}s)")

            run_meta["tests_run"].append({
                "test_id":   test["id"],
                "prompt_id": p["id"],
                "elapsed_s": round(elapsed, 2),
                "output_len": len(response_text),
                "finish_reason": finish_reason,
                "retry": retry,
                "error": response_text.startswith("[ERROR:"),
            })

    # ── Save metadata ───────────────────────────────────────────────────────────
    run_meta["end_time"]      = datetime.now().isoformat()
    run_meta["total_elapsed"] = round(total_elapsed, 2)
    (run_dir / "run_info.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n✅ Completed. Time: {total_elapsed:.1f}s | Results: {run_dir}")
    print(f"   Next step: python evaluate.py --run-dir {run_dir}")


if __name__ == "__main__":
    main()
