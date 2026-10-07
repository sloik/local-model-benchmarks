"""SPEC-B06 — an unparseable judge verdict is UNCHECKED, not a score of zero.

Covers the aggregation rules (R1-R3) and the targeted re-judge merge (R4). No
model calls: the end-to-end tests drive evaluate.main() with the evaluator
function monkeypatched, and the merge tests exercise pure functions.

Run: uv run --with openai --with pytest python -m pytest tests/test_evaluate_unchecked.py -v
"""
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

import evaluate  # noqa: E402
from evaluate import (  # noqa: E402
    aggregate_results,
    is_unchecked,
    merge_repaired_results,
    parse_eval_response,
    select_repair_targets,
)

# Every test in the suite scores "min", so a single fabricated 0 would drag a
# whole test to the floor. That is the amplification this spec exists to stop.
ALL_TESTS_MIN = {
    "CAT-05-004": {"scoring": "min"},
    "CAT-03-010": {"scoring": "min"},
}

# A judge that spent its budget reasoning and returned nothing usable — the
# observed real failure mode (finish_reason=length, empty content).
UNPARSEABLE = "<think>The response is quite detailed and I need to weigh"


def prompt_result(test_id, prompt_id, score, **extra):
    entry = {
        "test_id": test_id,
        "prompt_id": prompt_id,
        "category": test_id[:6],
        "score": score,
        "justification": "ok",
        "evaluator": "judge-x",
    }
    entry.update(extra)
    return entry


def unchecked_result(test_id, prompt_id):
    return prompt_result(test_id, prompt_id, None, unchecked=True,
                         justification="[UNCHECKED] no parseable verdict; raw: ...")


# ── The parse itself ─────────────────────────────────────────────────────────

def test_unparseable_judge_response_yields_no_score():
    score, _ = parse_eval_response(UNPARSEABLE)
    assert score is None


def test_empty_judge_response_yields_no_score():
    assert parse_eval_response("")[0] is None


# ── R1: unchecked is its own category, never a number ────────────────────────

def test_is_unchecked_covers_new_and_legacy_shapes():
    assert is_unchecked(unchecked_result("CAT-03-010", "p1"))
    assert is_unchecked(prompt_result("CAT-03-010", "p1", None))
    # Pre-SPEC-B06 artifacts recorded a failed measurement as 0.
    assert is_unchecked(prompt_result("CAT-03-010", "p1", 0))
    for real_score in (1, 2, 3, 4, 5):
        assert not is_unchecked(prompt_result("CAT-03-010", "p1", real_score))


def test_unchecked_prompt_enters_no_aggregate():
    results = [
        prompt_result("CAT-05-004", "p1", 5),
        unchecked_result("CAT-05-004", "p2"),
    ]
    tests_summary, summary = aggregate_results(results, ALL_TESTS_MIN)

    entry = tests_summary[0]
    assert entry["prompt_scores"] == [5]          # the non-verdict is not in here
    assert entry["test_score"] == 5               # not min(5, 0) == 0
    assert entry["unchecked_prompt_ids"] == ["p2"]
    assert summary["CAT-05"]["mean"] == 5
    assert summary["overall"]["mean"] == 5


def test_legacy_zero_is_excluded_the_same_way():
    """R5: historical artifacts stay readable, and read honestly."""
    results = [
        prompt_result("CAT-05-004", "p1", 5),
        prompt_result("CAT-05-004", "p2", 0, justification="[PARSE ERROR] raw: ..."),
    ]
    _, summary = aggregate_results(results, ALL_TESTS_MIN)
    assert summary["overall"]["mean"] == 5
    assert summary["unchecked"]["prompts"] == 1


# ── R2: the count is visible in summary ──────────────────────────────────────

def test_summary_reports_unchecked_count_and_denominator():
    results = [
        prompt_result("CAT-05-004", "p1", 4),
        unchecked_result("CAT-05-004", "p2"),
        prompt_result("CAT-03-010", "p1", 4),
    ]
    _, summary = aggregate_results(results, ALL_TESTS_MIN)

    unchecked = summary["unchecked"]
    assert unchecked["prompts"] == 1
    assert unchecked["tests_affected"] == 1
    assert unchecked["tests_affected_ids"] == ["CAT-05-004"]
    assert unchecked["tests_total"] == 2
    assert unchecked["tests_scored"] == 2      # neither test lost every prompt
    assert unchecked["tests_excluded"] == 0


def test_summary_reports_zero_unchecked_for_a_clean_run():
    results = [
        prompt_result("CAT-05-004", "p1", 4),
        prompt_result("CAT-05-004", "p2", 3),
        prompt_result("CAT-03-010", "p1", 5),
    ]
    _, summary = aggregate_results(results, ALL_TESTS_MIN)

    assert summary["unchecked"]["prompts"] == 0
    assert summary["unchecked"]["tests_excluded"] == 0
    assert summary["unchecked"]["tests_scored"] == summary["unchecked"]["tests_total"] == 2
    assert summary["overall"]["n"] == 2


# ── R3: a fully unchecked test leaves the aggregate ──────────────────────────

def test_fully_unchecked_test_is_excluded_not_scored_zero():
    results = [
        prompt_result("CAT-03-010", "p1", 4),
        unchecked_result("CAT-05-004", "p1"),
        unchecked_result("CAT-05-004", "p2"),
    ]
    tests_summary, summary = aggregate_results(results, ALL_TESTS_MIN)

    dead = next(t for t in tests_summary if t["test_id"] == "CAT-05-004")
    assert dead["test_score"] is None
    assert dead["unchecked"] is True
    assert dead["prompt_scores"] == []

    # Excluded rather than contributing a floor value: the mean is 4, not 2.
    assert summary["overall"]["mean"] == 4
    assert "CAT-05" not in summary
    # ...and the reported test count reflects the exclusion.
    assert summary["overall"]["n"] == 1
    assert summary["unchecked"]["tests_total"] == 2
    assert summary["unchecked"]["tests_scored"] == 1
    assert summary["unchecked"]["tests_excluded"] == 1
    assert summary["unchecked"]["tests_excluded_ids"] == ["CAT-05-004"]


def test_every_prompt_unchecked_produces_no_overall_mean():
    results = [unchecked_result("CAT-03-010", "p1")]
    _, summary = aggregate_results(results, ALL_TESTS_MIN)
    assert "overall" not in summary
    assert summary["unchecked"]["tests_scored"] == 0


# ── R4: targeted re-judge ────────────────────────────────────────────────────

def test_select_repair_targets_picks_unchecked_and_missing_only():
    artifact = {"prompt_results": [
        prompt_result("CAT-05-004", "p1", 5),
        unchecked_result("CAT-05-004", "p2"),
        prompt_result("CAT-03-010", "p1", 0),        # legacy shape
    ]}
    available = [("CAT-05-004", "p1"), ("CAT-05-004", "p2"),
                 ("CAT-03-010", "p1"), ("CAT-03-011", "p1")]  # last never judged

    assert select_repair_targets(artifact, available) == {
        ("CAT-05-004", "p2"),
        ("CAT-03-010", "p1"),
        ("CAT-03-011", "p1"),
    }


def test_select_repair_targets_skips_prompts_with_no_raw_file():
    artifact = {"prompt_results": [unchecked_result("CAT-05-004", "p2")]}
    assert select_repair_targets(artifact, []) == set()


def test_clean_artifact_has_nothing_to_repair():
    artifact = {"prompt_results": [prompt_result("CAT-05-004", "p1", 4)]}
    assert select_repair_targets(artifact, [("CAT-05-004", "p1")]) == set()


def test_merge_repairs_only_the_targeted_prompts():
    keeper = prompt_result("CAT-05-004", "p1", 5)
    broken = unchecked_result("CAT-05-004", "p2")
    other  = prompt_result("CAT-03-010", "p1", 3)
    before = [json.dumps(e, sort_keys=True) for e in (keeper, other)]

    fixed = prompt_result("CAT-05-004", "p2", 4)
    merged, errors = merge_repaired_results(
        [keeper, broken, other],
        [{"test_id": "CAT-05-004", "prompt_id": "p2", "error": "unparseable judge verdict"}],
        [fixed],
        [],
    )

    assert [(m["test_id"], m["prompt_id"], m["score"]) for m in merged] == [
        ("CAT-05-004", "p1", 5),
        ("CAT-05-004", "p2", 4),
        ("CAT-03-010", "p1", 3),
    ]
    # Every already-valid verdict is byte-identical and un-reordered.
    assert [json.dumps(m, sort_keys=True) for m in merged if m is not fixed] == before
    assert merged[0] is keeper and merged[2] is other
    # The error described a verdict that now exists.
    assert errors == []


def test_merge_appends_prompts_absent_from_the_artifact():
    existing = prompt_result("CAT-05-004", "p1", 5)
    added    = prompt_result("CAT-05-004", "p2", 2)
    merged, _ = merge_repaired_results([existing], [], [added], [])
    assert merged == [existing, added]


def test_merge_keeps_errors_unrelated_to_the_repair():
    stale = {"file": "CAT-09-001_p1.txt", "error": "test not found"}
    merged, errors = merge_repaired_results(
        [unchecked_result("CAT-05-004", "p2")],
        [stale, {"test_id": "CAT-05-004", "prompt_id": "p2", "error": "unparseable judge verdict"}],
        [prompt_result("CAT-05-004", "p2", 4)],
        [],
    )
    assert errors == [stale]
    assert merged[0]["score"] == 4


def test_merge_drops_a_file_keyed_error_for_a_repaired_prompt():
    """The exception path records only a filename, with no test/prompt id."""
    _, errors = merge_repaired_results(
        [],
        [{"file": "CAT-05-004_p2.txt", "error": "connection reset"}],
        [prompt_result("CAT-05-004", "p2", 4)],
        [],
    )
    assert errors == []


def test_repair_that_still_fails_stays_unchecked():
    merged, errors = merge_repaired_results(
        [unchecked_result("CAT-05-004", "p1")],
        [{"test_id": "CAT-05-004", "prompt_id": "p1", "error": "unparseable judge verdict"}],
        [unchecked_result("CAT-05-004", "p1")],
        [{"test_id": "CAT-05-004", "prompt_id": "p1", "error": "unparseable judge verdict"}],
    )
    assert is_unchecked(merged[0])
    assert len(errors) == 1


# ── End to end through main(), with the judge stubbed ────────────────────────

def build_run_dir(tmp_path, prompts):
    run_dir = tmp_path / "2026-08-17_stub-model_test"
    (run_dir / "raw").mkdir(parents=True)
    for test_id, prompt_id in prompts:
        (run_dir / "raw" / f"{test_id}_{prompt_id}.txt").write_text(
            "A plausible-looking subject response.", encoding="utf-8")
    (run_dir / "run_info.json").write_text(
        json.dumps({"model_id": "stub-model"}), encoding="utf-8")
    return run_dir


def run_evaluate(monkeypatch, run_dir, judge, extra_args=()):
    monkeypatch.setattr(evaluate, "call_evaluator_claude",
                        lambda messages, model=None, max_tokens=None: judge(messages))
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--run-dir", str(run_dir), "--evaluator-type", "claude",
        *extra_args,
    ])
    evaluate.main()
    return json.loads((run_dir / "scores.json").read_text(encoding="utf-8"))


def test_unparseable_verdict_is_recorded_as_unchecked_end_to_end(tmp_path, monkeypatch):
    """AC1: in errors, no numeric score, absent from every aggregate."""
    run_dir = build_run_dir(tmp_path, [("CAT-05-004", "p1"), ("CAT-05-004", "p2")])
    responses = iter(["SCORE: 5\nJUSTIFICATION: solid", UNPARSEABLE])
    artifact = run_evaluate(monkeypatch, run_dir, lambda _: next(responses))

    broken = next(r for r in artifact["prompt_results"] if r["prompt_id"] == "p2")
    assert broken["score"] is None
    assert broken["unchecked"] is True

    error = next(e for e in artifact["errors"] if e.get("prompt_id") == "p2")
    assert error["test_id"] == "CAT-05-004"
    assert error["error"] == "unparseable judge verdict"
    assert UNPARSEABLE[:40] in error["raw_response_head"]   # raw text preserved

    assert artifact["test_results"][0]["prompt_scores"] == [5]
    assert artifact["summary"]["overall"]["mean"] == 5
    assert artifact["summary"]["unchecked"]["prompts"] == 1


def test_a_wholly_unparseable_test_is_excluded_end_to_end(tmp_path, monkeypatch):
    """AC3, through the real evaluator path."""
    run_dir = build_run_dir(tmp_path, [("CAT-05-004", "p1"), ("CAT-05-004", "p2"),
                                       ("CAT-03-010", "p1")])
    # Raw files are judged in sorted filename order, so CAT-03-010 comes first.
    responses = iter(["SCORE: 4\nJUSTIFICATION: fine", UNPARSEABLE, UNPARSEABLE])
    artifact = run_evaluate(monkeypatch, run_dir, lambda _: next(responses))

    dead = next(t for t in artifact["test_results"] if t["test_id"] == "CAT-05-004")
    assert dead["test_score"] is None and dead["unchecked"] is True
    assert artifact["summary"]["overall"] == {"mean": 4, "n": 1}
    assert artifact["summary"]["unchecked"]["tests_excluded"] == 1
    assert artifact["summary"]["unchecked"]["tests_total"] == 2


def test_repair_fixes_only_the_unchecked_prompt(tmp_path, monkeypatch):
    """AC4 end to end: valid verdicts survive the repair byte-identical."""
    run_dir = build_run_dir(tmp_path, [("CAT-05-004", "p1"), ("CAT-05-004", "p2")])
    responses = iter(["SCORE: 5\nJUSTIFICATION: solid", UNPARSEABLE])
    first = run_evaluate(monkeypatch, run_dir, lambda _: next(responses))
    keeper_before = json.dumps(
        next(r for r in first["prompt_results"] if r["prompt_id"] == "p1"), sort_keys=True)
    assert first["summary"]["unchecked"]["prompts"] == 1

    calls = []
    def judge(messages):
        calls.append(messages)
        return "SCORE: 3\nJUSTIFICATION: adequate"

    repaired = run_evaluate(monkeypatch, run_dir, judge,
                            extra_args=("--repair-unchecked",))

    assert len(calls) == 1, "repair must re-judge only the unchecked prompt"
    keeper_after = json.dumps(
        next(r for r in repaired["prompt_results"] if r["prompt_id"] == "p1"), sort_keys=True)
    assert keeper_after == keeper_before

    fixed = next(r for r in repaired["prompt_results"] if r["prompt_id"] == "p2")
    assert fixed["score"] == 3 and not fixed.get("unchecked")
    assert repaired["summary"]["unchecked"]["prompts"] == 0
    assert repaired["errors"] == []
    assert repaired["summary"]["overall"]["mean"] == 3     # min(5, 3)


def test_repair_on_a_clean_artifact_makes_no_judge_calls(tmp_path, monkeypatch):
    run_dir = build_run_dir(tmp_path, [("CAT-03-010", "p1")])
    run_evaluate(monkeypatch, run_dir, lambda _: "SCORE: 4\nJUSTIFICATION: fine")
    before = (run_dir / "scores.json").read_text(encoding="utf-8")

    calls = []
    def judge(messages):
        calls.append(messages)
        return "SCORE: 1\nJUSTIFICATION: should never run"

    monkeypatch.setattr(evaluate, "call_evaluator_claude",
                        lambda messages, model=None, max_tokens=None: judge(messages))
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--run-dir", str(run_dir), "--evaluator-type", "claude",
        "--repair-unchecked",
    ])
    evaluate.main()

    assert calls == []
    assert (run_dir / "scores.json").read_text(encoding="utf-8") == before


def test_repair_ignores_raw_files_the_loop_can_never_judge(tmp_path, monkeypatch):
    """A deprecated or unknown test yields no prompt_result, so it must not be
    offered as a repair target — otherwise it is a permanent one and "nothing to
    repair" becomes unreachable."""
    run_dir = build_run_dir(tmp_path, [("CAT-03-010", "p1")])
    run_evaluate(monkeypatch, run_dir, lambda _: "SCORE: 4\nJUSTIFICATION: fine")

    # CAT-05-001 is deprecated; CAT-09-999 is not in any test JSON.
    for name in ("CAT-05-001_p1.txt", "CAT-09-999_p1.txt"):
        (run_dir / "raw" / name).write_text("orphan response", encoding="utf-8")
    before = (run_dir / "scores.json").read_text(encoding="utf-8")

    calls = []
    monkeypatch.setattr(evaluate, "call_evaluator_claude",
                        lambda messages, model=None, max_tokens=None: calls.append(messages))
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--run-dir", str(run_dir), "--evaluator-type", "claude",
        "--repair-unchecked",
    ])
    evaluate.main()

    assert calls == []
    assert (run_dir / "scores.json").read_text(encoding="utf-8") == before


def test_repair_without_an_existing_artifact_exits(tmp_path, monkeypatch):
    run_dir = build_run_dir(tmp_path, [("CAT-03-010", "p1")])
    monkeypatch.setattr(sys, "argv", [
        "evaluate.py", "--run-dir", str(run_dir), "--evaluator-type", "claude",
        "--repair-unchecked",
    ])
    with pytest.raises(SystemExit) as exc:
        evaluate.main()
    assert exc.value.code == 1


@pytest.fixture(autouse=True)
def isolate_test_leaderboard(tmp_path, monkeypatch):
    """Evaluator integration checks write only into their private test directory."""
    import report
    monkeypatch.setattr(report, "LEADERBOARD", tmp_path / "leaderboard.csv")
