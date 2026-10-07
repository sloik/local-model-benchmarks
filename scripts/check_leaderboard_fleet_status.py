#!/usr/bin/env python3
"""
check_leaderboard_fleet_status.py — AC1 (SPEC-003-021): assert no model marked
`fleet` in LEADERBOARD.md section 2 is absent from the live `lms ls` inventory.

Re-runnable: exits 0 (clean) or 1 (listing every offending row) each time.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).parent.parent
LEADERBOARD_MD = BENCH_DIR / "LEADERBOARD.md"

# Matches one section-2 table data row: | model | ... | status |
ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|(?:[^|]*\|){10}\s*(\S[^|]*?)\s*\|\s*$")


def fetch_installed_display_names() -> set[str]:
    proc = subprocess.run(["lms", "ls"], capture_output=True, text=True, timeout=30)
    proc.check_returncode()
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def extract_section_2_rows(text: str) -> list[tuple[str, str]]:
    """Return (model, status) for every data row between the section-2 header
    row and the next `## ` heading (i.e. stops before section 2a)."""
    start = text.index("## 2. Full model leaderboard")
    end = text.index("\n## ", start + 1)
    section = text[start:end]

    rows = []
    in_table = False
    for line in section.splitlines():
        if line.startswith("| model |"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table and not line.startswith("|"):
            break
        if in_table:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) >= 13:
                rows.append((cells[0], cells[-1]))
    return rows


def check(leaderboard_path: Path = LEADERBOARD_MD) -> list[str]:
    """Return a list of violation descriptions (empty = clean)."""
    text = leaderboard_path.read_text(encoding="utf-8")
    rows = extract_section_2_rows(text)

    try:
        installed = fetch_installed_display_names()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return [f"cannot query live `lms ls` inventory: {exc}"]

    installed_blob = " ".join(installed).lower()

    violations = []
    for model, status in rows:
        if "fleet" not in status.lower():
            continue
        if "native" in status.lower():
            continue  # non-LM-Studio system (e.g. Apple FM) — lms ls does not apply
        model_key = model.replace("**", "").strip()
        if model_key.lower() not in installed_blob:
            violations.append(f"{model_key!r} marked {status!r} but not found in live `lms ls`")
    return violations


def main() -> int:
    violations = check()
    if violations:
        print("FAIL — fleet-status claims not matching live inventory:")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("OK — every `fleet`-labelled model is in the live `lms ls` inventory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
