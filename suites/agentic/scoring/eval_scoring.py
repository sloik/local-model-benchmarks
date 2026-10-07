#!/usr/bin/env python3
"""Turn one eval run's artifacts into the numbers result.json records (BUG-001).

Both zsh runners shell out to this instead of parsing their own logs with sed.
Three defects it exists to prevent:

1. Test counts. `sed -n 's/.*\\([0-9][0-9]*\\) passed.*/\\1/p'` keeps only the LAST
   digit — the greedy `.*` eats the rest ("13 passed" -> 3, "10 passed" -> 0).
   Counts come from pytest's own --junitxml report, the same authority
   score_reference.py uses. The summary-line parse below is the fallback for runs
   that have no junit (timeout-skipped) and the only path open to the backfill of
   history written before --junitxml existed; its regex anchors the digits.

2. Model-call evidence. A harness that crashed before reaching the model produces
   the same untouched stub as a correct EVAL-004 refusal, and scored the same
   point. `model_invoked` separates them.

3. Token counts. The aider parse required a literal "k", so sub-1k lines
   ("Tokens: 462 sent") matched nothing and recorded 0 — indistinguishable from a
   dead harness. Both forms parse here.

CLI (the runners read these 7 stdout lines in order):
    passed, failed, errors, total, tokens_sent_k, tokens_received_k, model_invoked

tokens_* print as "null" when the harness does not report usage: 0 means the model
was never reached, and conflating the two is defect 2 all over again.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

# Anchored on the digits: \d+ is greedy from the START of the number, so "13
# passed" yields 13. The sed it replaces backtracked to the last digit alone.
_COUNTS = {
    "passed": re.compile(r"(\d+)\s+passed\b"),
    "failed": re.compile(r"(\d+)\s+failed\b"),
    "errors": re.compile(r"(\d+)\s+errors?\b"),
}
# pytest closes every run with a duration: "13 passed in 0.05s". Requiring it keeps
# a test *named* test_x_passed out of the summary-line search.
_DURATION = re.compile(r"\bin\s+\d+(?:\.\d+)?s\b")

# "Tokens: 3.8k sent, 6.0k received." and "Tokens: 462 sent, 89 received."
_AIDER_TOKENS = re.compile(
    r"Tokens:\s*([\d.]+)\s*([kKmM]?)\s+sent,\s*([\d.]+)\s*([kKmM]?)\s+received",
)
_HERMES_SESSION = re.compile(r"^session_id:\s*\S+", re.MULTILINE)

_UNIT = {"": 1.0, "k": 1e3, "m": 1e6}


def _zero_counts() -> dict:
    return {"passed": 0, "failed": 0, "errors": 0, "total": 0}


def counts_from_junit(junit_path: Path) -> dict:
    """Counts from pytest's own report. Never scrapes the human summary line."""
    root = ET.parse(junit_path).getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    if suite is None:
        return _zero_counts()

    out = _zero_counts()
    for case in suite.findall("testcase"):
        if case.find("failure") is not None:
            out["failed"] += 1
        elif case.find("error") is not None:
            out["errors"] += 1
        elif case.find("skipped") is not None:
            continue
        else:
            out["passed"] += 1
    out["total"] = out["passed"] + out["failed"] + out["errors"]
    return out


def counts_from_summary(text: str) -> dict:
    """Counts from a retained pytest.log's summary line.

    Recovery path only: history predates --junitxml, and a timeout-skipped run
    never produced one. Live scoring uses counts_from_junit.
    """
    line = ""
    for candidate in reversed(text.splitlines()):
        if _DURATION.search(candidate) and any(r.search(candidate) for r in _COUNTS.values()):
            line = candidate
            break

    out = _zero_counts()
    for name, rx in _COUNTS.items():
        m = rx.search(line)
        out[name] = int(m.group(1)) if m else 0
    out["total"] = out["passed"] + out["failed"] + out["errors"]
    return out


def counts_from_artifacts(junit_path: Path | None, pytest_log: Path | None) -> dict:
    """junitxml where it exists, retained log where it does not."""
    if junit_path is not None and junit_path.exists():
        try:
            return counts_from_junit(junit_path)
        except ET.ParseError:
            pass  # truncated report (killed mid-write): fall back to the log
    if pytest_log is not None and pytest_log.exists():
        return counts_from_summary(pytest_log.read_text(errors="replace"))
    return _zero_counts()


def tokens_from_aider_log(text: str) -> dict:
    """aider's last usage line, in thousands. Handles "3.8k sent" and "462 sent".

    Returns Nones when aider never printed usage — it crashed or was killed before
    a model round-trip completed. That is not zero usage; it is no measurement.
    """
    matches = _AIDER_TOKENS.findall(text)
    if not matches:
        return {"sent_k": None, "received_k": None}
    sent, sent_unit, received, received_unit = matches[-1]
    try:
        return {
            "sent_k": round(float(sent) * _UNIT[sent_unit.lower()] / 1000, 4),
            "received_k": round(float(received) * _UNIT[received_unit.lower()] / 1000, 4),
        }
    except (ValueError, KeyError):
        return {"sent_k": None, "received_k": None}


def model_invoked(harness: str, log_text: str) -> bool:
    """Did a real model call happen? The evidence EVAL-004 credit now requires.

    aider prints its usage line only after a completed round-trip; the 2026-07-14
    scipy-crash batch died in repo-map import and printed none (0/5 logs), while
    every genuine run has one (41/41 non-killed logs).

    hermes reports no token usage at all, so its evidence is the session line it
    emits once the agent loop has actually run (20/20 retained logs).
    """
    if harness == "aider":
        return (tokens_from_aider_log(log_text)["sent_k"] or 0) > 0
    if harness == "hermes":
        return bool(_HERMES_SESSION.search(log_text))
    raise ValueError(f"unknown harness: {harness}")


def _read(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return path.read_text(errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harness", required=True, choices=["aider", "hermes"])
    ap.add_argument("--junit", type=Path)
    ap.add_argument("--pytest-log", type=Path)
    ap.add_argument("--harness-log", type=Path, help="aider.log or hermes.log")
    ap.add_argument("--json", action="store_true", help="emit one JSON object instead")
    args = ap.parse_args()

    counts = counts_from_artifacts(args.junit, args.pytest_log)
    log_text = _read(args.harness_log)

    if args.harness == "aider":
        tokens = tokens_from_aider_log(log_text)
    else:
        # hermes emits no usage line. Recording 0 would read as "the model was
        # never called" — the exact confusion BUG-001 is about.
        tokens = {"sent_k": None, "received_k": None}

    invoked = model_invoked(args.harness, log_text)

    if args.json:
        print(json.dumps({**counts, "tokens_sent_k": tokens["sent_k"],
                          "tokens_received_k": tokens["received_k"],
                          "model_invoked": invoked}, separators=(",", ":")))
        return

    # The runners read exactly these 7 lines. "null" and true/false are JSON
    # literals: the runner interpolates them straight into result.json.
    print(counts["passed"])
    print(counts["failed"])
    print(counts["errors"])
    print(counts["total"])
    print("null" if tokens["sent_k"] is None else tokens["sent_k"])
    print("null" if tokens["received_k"] is None else tokens["received_k"])
    print("true" if invoked else "false")


if __name__ == "__main__":
    sys.exit(main())
