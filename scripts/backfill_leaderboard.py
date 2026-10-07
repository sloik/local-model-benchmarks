#!/usr/bin/env python3
"""
backfill_leaderboard.py — Regenerate leaderboard.csv from every attributed
historical run (SPEC-003-019 R4-R6).

Uses model_identity's resolver for attribution — never a directory-name
guess. Reads only results/, writes only leaderboard.csv (R5: results/ is
frozen evidence). Rebuilds the whole file deterministically each run, so
running it twice produces byte-identical output (R6): the row set is
computed in memory from the current results/ tree and results/ never
changes between two consecutive runs, so there is nothing for a second run
to add or remove.

Usage:
    python3 scripts/backfill_leaderboard.py [--write] [--results-dir DIR]
                                             [--leaderboard PATH]

Without --write, prints the row count and exits without touching
leaderboard.csv (dry run).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import model_identity  # noqa: E402
import report as report_module  # noqa: E402

BENCH_DIR = Path(__file__).parent.parent
RESULTS_DIR = BENCH_DIR / "results"
LEADERBOARD = BENCH_DIR / "leaderboard.csv"

FIELDNAMES = [
    "run_id", "model_id", "eval_date", "evaluator", "evaluator_type",
    "CAT-01_mean", "CAT-02_mean", "CAT-03_mean", "CAT-04_mean", "CAT-05_mean",
    "overall_mean", "category_assignment", "notes",
]


def _is_score_file(data: dict) -> bool:
    """Duck-type a parsed JSON payload as a report.py-compatible score artifact.

    Deliberately excludes the wide variety of other JSON files that live
    beside scores.json in a run directory (argo_eval_verdicts-*.json,
    cat07-scores.json, *.bak, etc.) by requiring exactly the fields
    report.py's own writer depends on — no filename allowlist to maintain.
    """
    return (
        isinstance(data, dict)
        and isinstance(data.get("summary"), dict)
        and "model_id" in data
        and "run_id" in data
        and "evaluator" in data
    )


def find_score_files(run_dir: Path) -> list[Path]:
    candidates = []
    for path in sorted(run_dir.glob("*.json")):
        if path.name.endswith(".bak"):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if _is_score_file(data):
            candidates.append(path)
    return candidates


def build_rows(results_dir: Path = RESULTS_DIR) -> list[dict]:
    """Build the full, deterministic leaderboard row set from every attributed run."""
    registry = model_identity.load_registry()
    attribution = model_identity.attribute_all(registry, results_dir=results_dir)

    rows_by_key: dict[tuple, dict] = {}
    for canonical, run_names in attribution["attributed"].items():
        for run_name in run_names:
            run_dir = results_dir / run_name
            for score_path in find_score_files(run_dir):
                data = json.loads(score_path.read_text(encoding="utf-8"))
                summary = data["summary"]
                labels = report_module.determine_labels(summary)
                row = {
                    "run_id":              data["run_id"],
                    "model_id":            canonical,
                    "eval_date":           data.get("eval_date", ""),
                    "evaluator":           data["evaluator"],
                    "evaluator_type":      data.get("evaluator_type", ""),
                    "CAT-01_mean":         summary.get("CAT-01", {}).get("mean", ""),
                    "CAT-02_mean":         summary.get("CAT-02", {}).get("mean", ""),
                    "CAT-03_mean":         summary.get("CAT-03", {}).get("mean", ""),
                    "CAT-04_mean":         summary.get("CAT-04", {}).get("mean", ""),
                    "CAT-05_mean":         summary.get("CAT-05", {}).get("mean", ""),
                    "overall_mean":        summary.get("overall", {}).get("mean", ""),
                    "category_assignment": " ".join(labels),
                    "notes":               "",
                }
                # Judge identity keeps two judges on the same run distinguishable
                # (R3/AC4) — same dedup key as report.py's update_leaderboard().
                key = (row["run_id"], row["evaluator_type"], row["evaluator"])
                rows_by_key[key] = row

    return [rows_by_key[k] for k in sorted(rows_by_key)]


def write_leaderboard(rows: list[dict], leaderboard: Path = LEADERBOARD) -> None:
    with open(leaderboard, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate leaderboard.csv from results/")
    parser.add_argument("--write", action="store_true", help="Write leaderboard.csv (default: dry run)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--leaderboard", type=Path, default=LEADERBOARD)
    args = parser.parse_args()

    rows = build_rows(args.results_dir)
    print(f"backfill: {len(rows)} row(s) from {args.results_dir}")

    if args.write:
        write_leaderboard(rows, args.leaderboard)
        print(f"wrote {args.leaderboard}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
