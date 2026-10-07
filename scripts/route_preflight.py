#!/usr/bin/env python3
"""Routing preflight report (A1) — run the Verifiability Router over a batch and summarise.

Spec: specs/SPEC-B02-router-detector-rate.md (A1 section). This is the thin wiring that puts the
already-built `verifiability_router.route()` to work: given a JSONL/JSON list of
`{question, field}` items, it reports each item's tier + route + reason and a summary of how many
would run V0-local vs V2-frontier. It makes NO model calls — pure routing, safe to run any time.

Usage:
    route_preflight.py exercises/route_preflight_example.json
    route_preflight.py items.jsonl          # one {"question":..., "field":...} object per line
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from verifiability_router import route  # noqa: E402


def load_items(path):
    """Load items from a JSON array file or a JSONL file (one object per line)."""
    text = pathlib.Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Whole-file JSON first (a top-level array, or a single object).
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    # Fall back to JSONL: one object per non-blank line.
    items = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def route_items(items):
    """Route each item and return (rows, summary). Each row folds the item in with the decision."""
    rows = []
    for item in items:
        if "question" not in item:
            raise ValueError("each item needs a 'question': %r" % item)
        field = item.get("field")
        decision = route(item["question"], field)
        rows.append({"question": item["question"], "field": field, **decision})
    return rows, summarize(rows)


def summarize(rows):
    """Count decisions by route and tier. The router only emits V0-local / V2-frontier."""
    local = sum(1 for r in rows if r["route"] == "local")
    frontier = sum(1 for r in rows if r["route"] == "frontier")
    return {
        "n": len(rows),
        "local": local,
        "frontier": frontier,
        "v0_local": sum(1 for r in rows if r["tier"] == "V0" and r["route"] == "local"),
        "v2_frontier": sum(1 for r in rows if r["tier"] == "V2" and r["route"] == "frontier"),
    }


def format_report(rows, summary):
    lines = []
    for r in rows:
        field = r["field"] if r["field"] is not None else "—"
        lines.append("[%s/%s] %s  (field=%s)" % (r["tier"], r["route"], r["question"], field))
        lines.append("    %s" % r["reason"])
    lines.append("")
    lines.append(
        "Summary: %d item(s) | V0-local: %d | V2-frontier: %d"
        % (summary["n"], summary["v0_local"], summary["v2_frontier"])
    )
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Verifiability routing preflight (A1)")
    ap.add_argument("file", help="JSON array or JSONL file of {question, field} items")
    a = ap.parse_args(argv)
    rows, summary = route_items(load_items(a.file))
    print(format_report(rows, summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
