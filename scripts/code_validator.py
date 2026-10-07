#!/usr/bin/env python3
"""
code_validator.py — Pre-filter for CAT-04 (code) evaluation.

Extracts Python code from raw model responses, checks syntax and runtime.
Results are appended to the evaluation prompt so the LLM evaluator
can see whether the code actually works.

Usage as module:
    from code_validator import validate_code
    result = validate_code(raw_response, client=openai_client, model="qwen/qwen3.6-27b")

Usage standalone (test):
    python code_validator.py "print('hello world')"
    python code_validator.py --file raw_response.txt
"""

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_code_regex(raw_response: str) -> str | None:
    """
    Extract Python code from markdown fences (```python ... ``` or ``` ... ```).
    Returns the longest code block found, or None if no fences detected.
    """
    # Try ```python first, then generic ```
    patterns = [
        r"```python\s*\n(.*?)```",
        r"```py\s*\n(.*?)```",
        r"```\s*\n(.*?)```",
    ]
    best = None
    for pat in patterns:
        matches = re.findall(pat, raw_response, re.DOTALL)
        for m in matches:
            code = m.strip()
            if code and (best is None or len(code) > len(best)):
                best = code
    return best


def extract_code_llm(raw_response: str, client, model: str) -> str | None:
    """
    Use a local LLM to extract Python code from a raw response.
    Fallback for when regex can't find markdown-fenced code.

    Args:
        raw_response: The full raw model response
        client: OpenAI-compatible client (e.g., from openai.OpenAI)
        model: Model ID to use for extraction (e.g., "qwen/qwen3.6-27b")

    Returns:
        Extracted Python code string, or None if extraction failed.
    """
    extraction_prompt = [
        {
            "role": "system",
            "content": (
                "You are a code extraction tool. "
                "Extract ONLY the Python code from the following text. "
                "Output the raw Python code only — no explanation, no markdown fences, "
                "no commentary. If there is no Python code, output exactly: NO_CODE"
            ),
        },
        {
            "role": "user",
            "content": raw_response[:8000],  # limit to avoid token overflow
        },
    ]

    try:
        response = client.chat.completions.create(
            model=model,
            messages=extraction_prompt,
            max_tokens=4096,
            temperature=0.0,
        )
        result = (response.choices[0].message.content or "").strip()

        if result == "NO_CODE" or not result:
            return None

        # Clean up: remove markdown fences if the LLM added them anyway
        if result.startswith("```"):
            inner = extract_code_regex(result)
            if inner:
                return inner
            # Strip fences manually
            lines = result.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            return "\n".join(lines).strip() or None

        return result
    except Exception as e:
        print(f"  ⚠️  LLM code extraction failed: {e}", file=sys.stderr)
        return None


def extract_code(raw_response: str, client=None, model: str | None = None) -> str | None:
    """
    Extract Python code from a raw model response.
    Tries regex first; if no code found and client provided, falls back to LLM.

    Args:
        raw_response: The full raw model response
        client: Optional OpenAI-compatible client for LLM fallback
        model: Optional model ID for LLM fallback

    Returns:
        Extracted Python code string, or None.
    """
    # Fast path: regex
    code = extract_code_regex(raw_response)
    if code:
        return code

    # Slow path: LLM extraction
    if client and model:
        return extract_code_llm(raw_response, client, model)

    return None


def check_syntax(code: str) -> tuple[bool, str | None]:
    """
    Check Python syntax using ast.parse().

    Returns:
        (True, None) if syntax is valid.
        (False, error_message) if syntax error found.
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def check_execution(code: str, timeout: int = 10) -> tuple[bool, int, str | None]:
    """
    Execute code in a subprocess with timeout.
    Uses a temporary file to avoid shell escaping issues.

    Returns:
        (runs_ok, exit_code, stderr_or_none)
    """
    # Safety: don't execute code that imports dangerous modules
    dangerous_patterns = [
        r"\bos\.system\b",
        r"\bsubprocess\b",
        r"\bshutil\.rmtree\b",
        r"\b__import__\b",
        r"\beval\s*\(",
        r"\bexec\s*\(",
    ]
    for pat in dangerous_patterns:
        if re.search(pat, code):
            return False, -1, f"Skipped execution: potentially dangerous pattern ({pat})"

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )

        os.unlink(tmp_path)

        if result.returncode == 0:
            return True, 0, None
        else:
            stderr = result.stderr.strip()
            # Truncate long errors
            if len(stderr) > 500:
                stderr = stderr[:500] + "..."
            return False, result.returncode, stderr

    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False, -1, f"Execution timed out after {timeout}s"
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return False, -1, f"Execution error: {e}"


def validate_code(raw_response: str, client=None, model: str | None = None,
                  timeout: int = 10) -> dict:
    """
    Full validation pipeline: extract → syntax check → runtime check.

    Args:
        raw_response: The full raw model response
        client: Optional OpenAI-compatible client for LLM code extraction
        model: Optional model ID for LLM code extraction
        timeout: Execution timeout in seconds

    Returns:
        {
            "has_code": bool,
            "code": str | None,        # extracted code (for debugging)
            "extraction_method": str,   # "regex", "llm", or "none"
            "syntax_ok": bool | None,   # None if no code found
            "runs_ok": bool | None,     # None if no code or syntax error
            "error": str | None,        # error message if any check failed
        }
    """
    # Step 1: Extract code
    code_regex = extract_code_regex(raw_response)
    if code_regex:
        code = code_regex
        method = "regex"
    elif client and model:
        code = extract_code_llm(raw_response, client, model)
        method = "llm" if code else "none"
    else:
        code = None
        method = "none"

    if not code:
        return {
            "has_code": False,
            "code": None,
            "extraction_method": method,
            "syntax_ok": None,
            "runs_ok": None,
            "error": "No Python code found in response",
        }

    # Step 2: Syntax check
    syntax_ok, syntax_error = check_syntax(code)
    if not syntax_ok:
        return {
            "has_code": True,
            "code": code,
            "extraction_method": method,
            "syntax_ok": False,
            "runs_ok": None,
            "error": syntax_error,
        }

    # Step 3: Runtime check
    runs_ok, exit_code, runtime_error = check_execution(code, timeout=timeout)
    return {
        "has_code": True,
        "code": code,
        "extraction_method": method,
        "syntax_ok": True,
        "runs_ok": runs_ok,
        "error": runtime_error,
    }


def format_for_evaluator(validation: dict) -> str:
    """
    Format validation results as a note to append to the evaluation prompt.
    Only called when there are issues worth noting.

    Returns:
        A string to append to the evaluator prompt, or empty string if all OK.
    """
    if not validation["has_code"]:
        return "\n\n[CODE VALIDATION NOTE: No Python code could be extracted from the response.]"

    parts = []
    if not validation["syntax_ok"]:
        parts.append(f"SYNTAX ERROR: {validation['error']}")
    elif not validation["runs_ok"]:
        parts.append(f"RUNTIME ERROR: {validation['error']}")

    if not parts:
        return ""  # All checks passed — no note needed

    return "\n\n[CODE VALIDATION NOTE: " + "; ".join(parts) + "]"


# ── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Code Validator — standalone test")
    parser.add_argument("code", nargs="?", help="Python code string to validate")
    parser.add_argument("--file", help="File containing raw model response")
    args = parser.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    elif args.code:
        text = f"```python\n{args.code}\n```"
    else:
        # Demo with a known-broken example
        text = """Here's a function:
```python
def parse_numbers(text):
    cleaned = text.replace(' ', '')
    parts = cleaned.split(' ')
    return [float(p) for p in parts]

print(parse_numbers("1 234 567"))
```
"""
        print("Running demo with known-broken code...\n")

    result = validate_code(text)
    import json
    # Don't print the full code in output
    display = {k: v for k, v in result.items() if k != "code"}
    display["code_length"] = len(result["code"]) if result["code"] else 0
    print(json.dumps(display, indent=2, ensure_ascii=False))

    note = format_for_evaluator(result)
    if note:
        print(f"\nEvaluator note:{note}")
