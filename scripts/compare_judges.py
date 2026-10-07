#!/usr/bin/env python3
"""Compare benchmark judge artifacts against a named reference adjudication."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def load_scores(path: Path) -> dict[str, int]:
    """Test scores keyed by test id.

    A test with no verdict (SPEC-B06 unchecked) is omitted rather than read as a
    number, so compare() names it instead of averaging a non-measurement.
    Artifacts predating that change recorded such tests as 0 and are read as
    written — this does not reinterpret already-published comparisons.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        item["test_id"]: int(item["test_score"])
        for item in data["test_results"]
        if item.get("test_score") is not None and not item.get("unchecked")
    }


def pearson(left: list[int], right: list[int]) -> float | None:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (left_item - left_mean) * (right_item - right_mean)
        for left_item, right_item in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((item - left_mean) ** 2 for item in left)
        * sum((item - right_mean) ** 2 for item in right)
    )
    return numerator / denominator if denominator else None


def compare(candidate: dict[str, int], reference: dict[str, int]) -> dict[str, object]:
    test_ids = list(reference)
    missing = sorted(set(test_ids) - set(candidate))
    if missing:
        raise ValueError(
            f"candidate is missing tests: {missing} — if these are unchecked verdicts, "
            f"repair them first with evaluate.py --repair-unchecked"
        )
    candidate_values = [candidate[test_id] for test_id in test_ids]
    reference_values = [reference[test_id] for test_id in test_ids]
    deltas = [
        candidate_score - reference_score
        for candidate_score, reference_score in zip(
            candidate_values, reference_values, strict=True
        )
    ]
    count = len(test_ids)
    return {
        "n": count,
        "mean": round(sum(candidate_values) / count, 4),
        "mae_vs_reference": round(sum(abs(delta) for delta in deltas) / count, 4),
        "exact_agreement": round(sum(delta == 0 for delta in deltas) / count, 4),
        "within_one": round(sum(abs(delta) <= 1 for delta in deltas) / count, 4),
        "signed_bias": round(sum(deltas) / count, 4),
        "pearson": None if (value := pearson(candidate_values, reference_values)) is None else round(value, 4),
        "ceiling_rate": round(sum(score == 5 for score in candidate_values) / count, 4),
        "score_histogram": dict(sorted(Counter(candidate_values).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", action="append", nargs=2, metavar=("LABEL", "PATH"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = load_scores(args.reference)
    report: dict[str, object] = {
        "reference": str(args.reference),
        "candidates": {},
    }
    for label, raw_path in args.candidate:
        path = Path(raw_path)
        scores = load_scores(path)
        overall = compare(scores, reference)
        categories = {}
        for category in sorted({test_id[:6] for test_id in reference}):
            category_reference = {
                test_id: score for test_id, score in reference.items() if test_id.startswith(category)
            }
            category_scores = {test_id: scores[test_id] for test_id in category_reference}
            categories[category] = compare(category_scores, category_reference)
        report["candidates"][label] = {
            "path": str(path),
            "overall": overall,
            "categories": categories,
        }

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
