"""Tests for the routing preflight (A1 wiring). Pure logic — NO model calls anywhere.

Verifies the three canonical DIA cases route as expected and the summary counts are right.
Spec: specs/SPEC-B02-router-detector-rate.md (A1 section).
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from route_preflight import load_items, route_items, summarize  # noqa: E402

FIXTURE = (
    pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "route_preflight_example.json"
)


def test_fixture_three_items_route_as_expected():
    rows, _ = route_items(load_items(FIXTURE))
    assert len(rows) == 3

    # 1) DIA attribution question -> V2/frontier (causal lexicon hit).
    assert rows[0]["tier"] == "V2"
    assert rows[0]["route"] == "frontier"

    # 2) clean extraction over a whitelisted field -> V0/local.
    assert rows[1]["tier"] == "V0"
    assert rows[1]["route"] == "local"

    # 3) ex-item / one-off question -> V2/frontier despite a whitelisted field.
    assert rows[2]["tier"] == "V2"
    assert rows[2]["route"] == "frontier"


def test_summary_counts_are_right():
    rows, summary = route_items(load_items(FIXTURE))
    assert summary == {
        "n": 3,
        "local": 1,
        "frontier": 2,
        "v0_local": 1,
        "v2_frontier": 2,
    }


def test_summarize_matches_route_items():
    rows, summary = route_items(load_items(FIXTURE))
    assert summarize(rows) == summary


def test_load_items_supports_jsonl(tmp_path):
    p = tmp_path / "items.jsonl"
    p.write_text(
        '{"question": "Ile wyniosly przychody w Q1\'26?", "field": "przychody"}\n'
        '{"question": "Czy to tlumaczy spadek?"}\n',
        encoding="utf-8",
    )
    rows, summary = route_items(load_items(p))
    assert summary["n"] == 2
    assert rows[0]["tier"] == "V0"
    assert rows[1]["tier"] == "V2"
