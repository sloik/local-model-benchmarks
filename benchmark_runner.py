import os
#!/usr/bin/env python3
"""
Inwestomat Benchmark Runner
============================
Phase 0 — Learning exercise for the maintainer.

Complete all three TODO sections (EXERCISE 1, 2, 3).
Run the script after each one to test your progress.
If stuck: read the hints first, then ask Argo for a nudge.

Usage (run from vault root or benchmarks folder):
    python _Tools/benchmarks/benchmark_runner.py
    python _Tools/benchmarks/benchmark_runner.py --category cat-02-dystylacja --test 001
    python _Tools/benchmarks/benchmark_runner.py --model mistral-7b-instruct
"""

import json
import argparse
from pathlib import Path       # like URL in Swift, but for file paths
from datetime import datetime

import requests                # pip install requests  ← like URLSession, but 1 line instead of 20


# ── Configuration ──────────────────────────────────────────────────────────────
# Path(__file__) is this script's location. .parent goes up one directory.
# So TESTS_DIR = _Tools/benchmarks/tests/
TESTS_DIR   = Path(__file__).parent / "tests"
RESULTS_DIR = Path(__file__).parent / "results"

LM_STUDIO_HOST = "http://localhost:1234"
LM_STUDIO_URL = f"{LM_STUDIO_HOST}/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.6-27b"  # quality-first background default

LM_TOKEN = os.getenv("LM_STUDIO_TOKEN", "")

# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE 1 of 3 — File I/O + JSON
# ══════════════════════════════════════════════════════════════════════════════
def load_test(category: str, test_id: str) -> dict:
    """
    Load and return a single test from a category JSON file.

    File location : tests/cat-01-ekstrakcja.json
    JSON structure: { "tests": [ { "id": "001", "name": "...", "prompts": [...] }, ... ] }
    """
    path = TESTS_DIR / f"{category}.json"   # f-string: like "\(variable)" in Swift

    if not path.exists():
        raise FileNotFoundError(f"File not found {path}")

    # ── TODO 2: Open the file and parse its JSON ──────────────────────────────
    with open(path) as test_file:
        json_data = json.load(test_file)

    # ── TODO 3: Find the test matching test_id ────────────────────────────────
    tests = json_data["tests"]
    for test in tests:
        if test["id"] == test_id:
            return test

    # ── TODO 4: Raise a helpful error if nothing was found ────────────────────
    available_ids = [t["id"] for t in tests]
    raise ValueError(f"Test '{test_id}' not found. Available: {available_ids}")


# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE 2 of 3 — HTTP request to LM Studio
# ══════════════════════════════════════════════════════════════════════════════
def call_lm_studio(messages: list[dict], model: str) -> str:
    """
    Send a messages array to LM Studio and return the response text.

    LM Studio exposes an OpenAI-compatible API. The messages format is:
    [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
    Our test JSON already has messages in exactly this shape — no conversion needed.
    """

    # ── TODO 1: Build the request payload ────────────────────────────────────
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 2000
    }

    # ── TODO 2: Send the POST request ────────────────────────────────────────
    try:
        response = requests.post(
            LM_STUDIO_URL,
            json=payload,
            timeout=120,
            headers= {"Authorization" : f"Bearer {LM_TOKEN}" }
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to LM Studio. Is it running on {LM_STUDIO_HOST}?"
        )

    # ── TODO 3: Extract and return the response text ──────────────────────────
    return response.json()["choices"][0]["message"]["content"]


# ══════════════════════════════════════════════════════════════════════════════
# EXERCISE 3 of 3 — Putting it all together
# ══════════════════════════════════════════════════════════════════════════════
def run_test(category: str, test_id: str, model: str) -> Path:
    """
    Run all prompts for a single test and save results to disk.
    Returns the path to the saved result file.
    """
    print(f"\n{'='*60}")
    print(f"  Category : {category}")
    print(f"  Test     : {test_id}")
    print(f"  Model    : {model}")
    print(f"{'='*60}")

    test = load_test(category=category, test_id=test_id)

    prompts = test["prompts"]
    print(f"  Name    : {test['name']}")
    print(f"  Prompts : {len(prompts)}\n")

    results = []
    for prompt in prompts:
        print(f"── Prompt {prompt['id']}: {prompt['description'][:70]}")

        lm_response = call_lm_studio(messages=prompt["messages"], model=model)

        print(f"\n   Response:\n{lm_response[:400]}\n")
        results.append({
            "prompt_id": prompt["id"],
            "description": prompt["description"],
            "response": lm_response
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output = {
        "category": category,
        "test_id":  test_id,
        "test_name": test["name"],
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "criteria": test.get("criteria"),
        "results": results
    }

    safe_model = model.replace("/", "_").replace(":", "_")
    filename   = f"{category}_{test_id}_{safe_model}.json"

    out_path = RESULTS_DIR / filename

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved → {out_path}")
    return out_path


# ══════════════════════════════════════════════════════════════════════════════
# Entry point — nothing to change below this line, this part is done for you.
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """
    argparse parses command-line arguments.
    Gives us: python benchmark_runner.py --category CAT-01 --test CAT-01-001 --model llama
    """
    parser = argparse.ArgumentParser(description="Inwestomat Benchmark Runner")
    parser.add_argument("--category", "-c", default="cat-01-ekstrakcja",
                        help="Category filename without .json  (default: cat-01-ekstrakcja)")
    parser.add_argument("--test",     "-t", default="CAT-01-001",
                        help="Test ID to run  (default: CAT-01-001)")
    parser.add_argument("--model",    "-m", default=DEFAULT_MODEL,
                        help=f"Model name as shown in LM Studio  (default: {DEFAULT_MODEL})")
    return parser.parse_args()


# `if __name__ == "__main__":` runs only when executed directly, not when imported.
# In Swift this is like your @main entry point.
if __name__ == "__main__":
    args = parse_args()
    run_test(
        category=args.category,
        test_id=args.test,
        model=args.model,
    )
