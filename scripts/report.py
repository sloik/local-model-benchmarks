#!/usr/bin/env python3
"""
report.py — Report generation and leaderboard.csv update.

Loads scores.json from a given run_dir, generates human-readable REPORT.md
and updates (or creates) leaderboard.csv in the main benchmark folder.

Usage:
    python report.py --run-dir results/2026-03-15_qwen2.5-72b--mlx-4bit/
    python report.py --run-dir results/... --compare results/2026-03-10_.../

Requirements: no external packages (stdlib only).
"""

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import model_identity  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
BENCH_DIR    = SCRIPT_DIR.parent
TESTS_DIR    = BENCH_DIR / "tests"
LEADERBOARD  = BENCH_DIR / "leaderboard.csv"
CALIB_CSV    = BENCH_DIR / "evaluator_calibration.csv"

# Label thresholds (from SPEC.md section 6)
LABEL_THRESHOLDS = {
    "triage":      ("CAT-05", 4.0),
    "extraction":  ("CAT-01", 4.0),
    "distillation":("CAT-02", 4.0),
    "analyst":     ("CAT-03", 4.0),
    "coder":       ("CAT-04", 4.0),
}
GENERAL_THRESHOLD = 3.8

# Warning thresholds
def score_status(score: float) -> str:
    if score < 2.5:   return "❌ Don't delegate"
    if score < 3.5:   return "⚠️  With verification"
    if score < 4.0:   return "✅ Acceptable"
    return "✅✅ Delegate freely"


def determine_labels(summary: dict) -> list[str]:
    """Determine category labels based on results."""
    labels = []
    for label, (cat, threshold) in LABEL_THRESHOLDS.items():
        if cat in summary and summary[cat]["mean"] >= threshold:
            labels.append(label)
    # general: overall ≥ 3.8 and no categories below 2.5
    if "overall" in summary and summary["overall"]["mean"] >= GENERAL_THRESHOLD:
        low_cats = [c for c, s in summary.items()
                    if c.startswith("CAT") and s["mean"] < 2.5]
        if not low_cats:
            labels.append("general")
    return labels or ["none"]


def render_score_bar(score: float, max_score: float = 5.0, width: int = 20) -> str:
    filled = int(round(score / max_score * width))
    return "█" * filled + "░" * (width - filled)


def load_scores(run_dir: Path) -> dict:
    scores_file = run_dir / "scores.json"
    if not scores_file.exists():
        print(f"❌ Missing scores.json in {run_dir}")
        sys.exit(1)
    return json.loads(scores_file.read_text(encoding="utf-8"))


def generate_report(scores: dict, run_dir: Path, compare_scores: dict | None = None) -> str:
    """Generate REPORT.md content."""
    model_id   = scores["model_id"]
    eval_date  = scores["eval_date"]
    evaluator  = scores["evaluator"]
    summary    = scores["summary"]
    labels     = determine_labels(summary)
    test_results = scores.get("test_results", [])

    lines = [
        f"# Benchmark Report — {model_id}",
        f"",
        f"> **Date:** {eval_date}  ",
        f"> **Evaluator:** {evaluator}  ",
        f"> **Run ID:** {scores['run_id']}  ",
        f"> **Labels:** {', '.join(labels)}",
        f"",
        f"---",
        f"",
        f"## Results Summary",
        f"",
    ]

    # Category table
    lines += [
        f"| Category | Score | Bar | Status | Tests |",
        f"|----------|------:|-----|--------|------:|",
    ]
    cat_map = {
        "CAT-01": "Extraction",
        "CAT-02": "Distillation",
        "CAT-03": "Reasoning",
        "CAT-04": "Code",
        "CAT-05": "Classification",
    }
    for cat in ["CAT-01", "CAT-02", "CAT-03", "CAT-04", "CAT-05"]:
        if cat not in summary:
            lines.append(f"| {cat_map.get(cat, cat)} | — | — | — | 0 |")
            continue
        s = summary[cat]
        bar    = render_score_bar(s["mean"], width=10)
        status = score_status(s["mean"])
        comp   = ""
        if compare_scores and cat in compare_scores.get("summary", {}):
            prev_mean = compare_scores["summary"][cat]["mean"]
            diff      = s["mean"] - prev_mean
            comp      = f" ({'+' if diff >= 0 else ''}{diff:.2f} vs previous)"
        lines.append(f"| {cat_map.get(cat, cat)} | {s['mean']:.2f}/5 {comp}| {bar} | {status} | {s['n']} |")

    if "overall" in summary:
        s   = summary["overall"]
        bar = render_score_bar(s["mean"], width=10)
        lines += [
            f"",
            f"**Overall:** {s['mean']:.2f}/5  {bar}  (n={s['n']} tests)",
        ]

    # The mean is only honest next to its denominator — say what went unmeasured.
    unchecked = summary.get("unchecked", {})
    if unchecked.get("prompts"):
        excluded = unchecked.get("tests_excluded_ids", [])
        note = (f"> ⚠️ **{unchecked['prompts']} unchecked prompt(s)** — the judge returned no "
                f"parseable verdict. {unchecked['tests_scored']}/{unchecked['tests_total']} "
                f"tests scored")
        note += f"; excluded from the mean: {', '.join(excluded)}." if excluded else "."
        lines += [f"", note]

    # Per-test details
    if test_results:
        lines += [
            f"",
            f"---",
            f"",
            f"## Per-Test Results",
            f"",
            f"| Test ID | Test Score | Prompt Scores | Scoring |",
            f"|---------|:-----------:|----------------|---------|",
        ]
        for tr in sorted(test_results, key=lambda x: x["test_id"]):
            parts = [str(s) for s in tr["prompt_scores"]]
            parts += [f"{pid}=UNCHECKED" for pid in tr.get("unchecked_prompt_ids", [])]
            scores_str = ", ".join(parts)
            score_str = ("⚠️ UNCHECKED" if tr["test_score"] is None
                         else f"**{tr['test_score']}/5**")
            lines.append(
                f"| {tr['test_id']} | {score_str} | [{scores_str}] | {tr['scoring']} |"
            )

    # Evaluator justifications
    prompt_results = scores.get("prompt_results", [])
    if prompt_results:
        lines += [
            f"",
            f"---",
            f"",
            f"## Evaluator Justifications",
            f"",
        ]
        for pr in sorted(prompt_results, key=lambda x: (x["test_id"], x["prompt_id"])):
            if pr.get("score") is None:
                heading = f"### {pr['test_id']} / {pr['prompt_id']}  ⚠️ UNCHECKED — no verdict"
            else:
                score_emoji = ["", "❌", "⚠️", "✅", "✅", "✅✅"][min(pr["score"], 5)]
                heading = f"### {pr['test_id']} / {pr['prompt_id']}  {score_emoji} {pr['score']}/5"
            lines += [
                heading,
                f"",
                f"{pr['justification']}",
                f"",
            ]

    # Errors
    errors = scores.get("errors", [])
    if errors:
        lines += [
            f"---",
            f"",
            f"## Errors ({len(errors)})",
            f"",
        ]
        for err in errors:
            lines.append(f"- `{err.get('file', '?')}`: {err.get('error', '?')}")

    # Conclusions
    lines += [
        f"",
        f"---",
        f"",
        f"## Conclusions",
        f"",
        f"Model **{model_id}** obtained labels: **{', '.join(labels)}**",
        f"",
    ]
    for label, (cat, thresh) in LABEL_THRESHOLDS.items():
        if label in labels:
            lines.append(f"- ✅ `{label}`: {cat} = {summary.get(cat, {}).get('mean', 0):.2f} ≥ {thresh}")
        else:
            actual = summary.get(cat, {}).get("mean")
            if actual is not None:
                lines.append(f"- ❌ `{label}`: {cat} = {actual:.2f} < {thresh}")
            else:
                lines.append(f"- ❌ `{label}`: {cat} = no data")

    lines += [
        f"",
        f"---",
        f"",
        f"*Auto-generated by report.py at {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
    ]

    return "\n".join(lines)


def update_leaderboard(scores: dict, labels: list[str]) -> None:
    """Add row to leaderboard.csv (or create if it doesn't exist)."""
    summary    = scores["summary"]
    fieldnames = [
        "run_id", "model_id", "eval_date", "evaluator", "evaluator_type",
        "CAT-01_mean", "CAT-02_mean", "CAT-03_mean", "CAT-04_mean", "CAT-05_mean",
        "overall_mean", "category_assignment", "notes",
    ]

    try:
        canonical_model_id = model_identity.resolve(scores["model_id"])
    except model_identity.ModelIdentityError as exc:
        print(f"  ⚠️  model_identity: {exc} — writing raw model_id as-is", file=sys.stderr)
        canonical_model_id = scores["model_id"]

    row = {
        "run_id":              scores["run_id"],
        "model_id":            canonical_model_id,
        "eval_date":           scores["eval_date"],
        "evaluator":           scores["evaluator"],
        "evaluator_type":      scores.get("evaluator_type", ""),
        "CAT-01_mean":         summary.get("CAT-01", {}).get("mean", ""),
        "CAT-02_mean":         summary.get("CAT-02", {}).get("mean", ""),
        "CAT-03_mean":         summary.get("CAT-03", {}).get("mean", ""),
        "CAT-04_mean":         summary.get("CAT-04", {}).get("mean", ""),
        "CAT-05_mean":         summary.get("CAT-05", {}).get("mean", ""),
        "overall_mean":        summary.get("overall", {}).get("mean", ""),
        "category_assignment": " ".join(labels),
        "notes":               "",
    }

    # Load existing rows and remove duplicates of the same (run_id, evaluator_type,
    # evaluator) — keyed by judge identity too, so a run scored by two different
    # judges produces two distinguishable rows instead of one overwriting the
    # other (SPEC-003-019 R3/AC4).
    existing_rows: list[dict] = []
    if LEADERBOARD.exists():
        with open(LEADERBOARD, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not (r.get("run_id") == row["run_id"] and
                        r.get("evaluator_type") == row["evaluator_type"] and
                        r.get("evaluator") == row["evaluator"]):
                    existing_rows.append(r)

    all_rows = existing_rows + [row]
    with open(LEADERBOARD, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"📊 Leaderboard updated: {LEADERBOARD}")


def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark — Report Generator")
    parser.add_argument("--run-dir",  required=True,
                        help="Results folder (contains scores.json)")
    parser.add_argument("--compare",
                        help="Previous run-dir to compare (optional)")
    parser.add_argument("--no-leaderboard", action="store_true",
                        help="Do not update leaderboard.csv")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    scores  = load_scores(run_dir)

    compare_scores = None
    if args.compare:
        compare_scores = load_scores(Path(args.compare))

    labels      = determine_labels(scores["summary"])
    report_text = generate_report(scores, run_dir, compare_scores)

    # Save REPORT.md
    report_file = run_dir / "REPORT.md"
    report_file.write_text(report_text, encoding="utf-8")
    print(f"📄 Report saved: {report_file}")

    # Update leaderboard
    if not args.no_leaderboard:
        update_leaderboard(scores, labels)

    # Console summary
    print(f"\n{'='*50}")
    print(f"Model: {scores['model_id']}")
    print(f"Labels: {', '.join(labels)}")
    if "overall" in scores["summary"]:
        ov = scores["summary"]["overall"]
        print(f"Overall: {ov['mean']:.2f}/5  (n={ov['n']})")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
