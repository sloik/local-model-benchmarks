#!/usr/bin/env python3
"""Tests for scripts/score_cat05_deterministic.py (B04). NO model calls.

Answers are injected as fixtures or driven by the constant-`null` path, so the whole suite
runs offline. Coverage is checked against the REAL tests/cat-05-klasyfikacja.json.

Run: python -m pytest tests/test_score_cat05_deterministic.py -v
"""
import importlib.util
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_MOD_PATH = _HERE.parent / "scripts" / "score_cat05_deterministic.py"
_spec = importlib.util.spec_from_file_location("score_cat05_deterministic", _MOD_PATH)
mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = mod  # register before exec (DevKB python.md)
_spec.loader.exec_module(mod)

ITEMS = json.loads((_HERE.parent / "tests" / "cat-05-klasyfikacja.json").read_text(encoding="utf-8"))


def _test(tid):
    return next(t for t in ITEMS["tests"] if t["id"] == tid)


# ── label parsing / extraction ──────────────────────────────────────────────────
def test_parse_label_set_reads_taxonomy_from_system_prompt():
    p1 = _test("CAT-05-001")["prompts"][0]  # 4-class
    labels = mod.parse_label_set(p1["messages"][0]["content"])
    assert labels == ["raport_kwartalny", "news_aktualizacja", "analiza_analityczna", "inne"]
    p_ext = _test("CAT-05-011")["prompts"][0]  # 6-class
    ext = mod.parse_label_set(p_ext["messages"][0]["content"])
    assert "raport_ratingowy" in ext and "strategia_prezentacja" in ext


def test_normalize_tolerates_space_and_hyphen_forms():
    assert mod.normalize_label("Raport Kwartalny") == "raport_kwartalny"
    assert mod.normalize_label("raport-kwartalny") == "raport_kwartalny"


def test_extract_label_takes_the_answer_not_a_mentioned_forbidden_label():
    labels = ["raport_kwartalny", "news_aktualizacja", "analiza_analityczna", "inne"]
    # answer-first; a forbidden label named in the justification must NOT win
    ans = "raport_kwartalny\nThis is not an analiza_analityczna document."
    assert mod.extract_label(ans, labels) == "raport_kwartalny"


def test_extract_label_word_bounded_inne_not_matched_inside_innych():
    labels = ["raport_kwartalny", "inne"]
    # 'inne' must not be spuriously found inside the Polish word 'innych'
    assert mod.extract_label("raport_kwartalny, w odróżnieniu od innych.", labels) == "raport_kwartalny"


def test_extract_label_none_when_no_label_present():
    assert mod.extract_label("I cannot classify this document.", ["inne", "raport_kwartalny"]) is None


# ── verdict: correct / wrong / acceptable / unparseable ──────────────────────────
def test_correct_label_passes():
    p1 = _test("CAT-05-002")["prompts"][0]  # required: raport_kwartalny
    assert mod.verdict("raport_kwartalny\nInterim IFRS statement.", p1) == "CORRECT"


def test_wrong_label_fails():
    p1 = _test("CAT-05-002")["prompts"][0]  # required: raport_kwartalny, forbidden: analiza
    assert mod.verdict("analiza_analityczna\nHas a target price.", p1) == "WRONG"


def test_space_form_of_correct_label_still_passes():
    p1 = _test("CAT-05-002")["prompts"][0]
    assert mod.verdict("raport kwartalny — sprawozdanie śródroczne", p1) == "CORRECT"


def test_acceptable_secondary_label_counts_as_pass_not_wrong():
    p1 = _test("CAT-05-003")["prompts"][0]  # required inne, acceptable news_aktualizacja
    assert mod.verdict("news_aktualizacja\nIt is new information.", p1) == "ACCEPTABLE"
    assert mod.verdict("inne\nESPI current report.", p1) == "CORRECT"


def test_unparseable_answer_is_not_a_false_pass():
    p1 = _test("CAT-05-003")["prompts"][0]
    assert mod.verdict("Trudno powiedzieć.", p1) == "UNPARSEABLE"


# ── gradeability routing: open-ended -> needs-judge (not a forced score) ──────────
def test_batch_item_is_needs_judge_not_scored():
    batch = _test("CAT-05-010")  # required_per_document — genuinely open-ended for label-match
    assert mod.is_deterministic(batch) is False
    res = mod.score(ITEMS, injected={})
    row = next(r for r in res["detail"] if r["id"] == "CAT-05-010")
    assert row["gradeable"] == "needs-judge"
    assert row["item_score"] is None  # NOT 0.0 and NOT a false pass


def test_coverage_over_real_items():
    res = mod.score(ITEMS, injected={})
    # 10 active items (001 deprecated); 9 deterministic, 1 needs-judge (010 batch)
    assert res["items_active"] == 10
    assert res["items_deterministic"] == 9
    assert res["items_needs_judge"] == 1
    assert res["needs_judge_ids"] == ["CAT-05-010"]
    assert res["coverage"] == 0.9
    # null-resistant contrastive (empty PASS-set intersection): 002, 005, 007, 011.
    # 004 has 2 required labels but both prompts accept raport_kwartalny -> NOT null-resistant.
    assert res["items_contrastive"] == 4
    assert res["items_not_null_resistant"] == 5  # 003, 004, 006, 008, 009


# ── a real classifier scores well (positive control) ─────────────────────────────
def _perfect_answers():
    """Inject each prompt's own gold `sample_answer` (or the required label) as the answer."""
    inj = {}
    for t in ITEMS["tests"]:
        if t.get("deprecated"):
            continue
        for p in t["prompts"]:
            g = p.get("gold", {})
            correct = g.get("required") or g.get("required_one_of")
            if correct:
                inj[(t["id"], p["id"])] = g.get("sample_answer") or correct[0]
    return inj


def test_gold_answers_score_perfect_on_contrastive():
    res = mod.score(ITEMS, injected=_perfect_answers())
    assert res["cat05_label_mean_contrastive"] == 1.0
    assert res["cat05_label_mean_all"] == 1.0


def test_load_run_answers_reads_only_saved_cat05_outputs(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "CAT-05-002_p1.txt").write_text("raport_kwartalny", encoding="utf-8")
    (raw / "CAT-04-001_p1.txt").write_text("ignored", encoding="utf-8")

    answers = mod.load_run_answers(tmp_path)

    assert answers == {("CAT-05-002", "p1"): "raport_kwartalny"}


# ── ANTI-GAMING: constant-output null must not score well ─────────────────────────
def test_constant_quarterly_report_null_scores_at_or_below_chance():
    # The task's canonical null: always "raport_kwartalny".
    # chance = random single-label guess over the smallest taxonomy (4 classes) = 0.25.
    chance = 0.25
    all_det = mod.constant_null_score(ITEMS, "raport_kwartalny", subset="all")
    contr = mod.constant_null_score(ITEMS, "raport_kwartalny", subset="contrastive")
    # On the full deterministic set it wins only CAT-05-004 (both prompts accept kwartalny) =
    # 1/9 = 0.111 — still <= chance. The authoritative contrastive headline is fully null-proof.
    assert all_det <= chance
    assert contr == 0.0


def test_every_constant_null_scores_zero_on_contrastive_subset():
    # The gaming-resistant guarantee: min-scoring + within-item label contrast means NO
    # constant label can win a contrastive item.
    for lab in ["raport_kwartalny", "news_aktualizacja", "analiza_analityczna", "inne",
                "raport_roczny", "raport_ratingowy", "strategia_prezentacja", "espi_biezacy"]:
        assert mod.constant_null_score(ITEMS, lab, subset="contrastive") == 0.0


def test_no_constant_null_can_perfect_game_the_full_set():
    # Over ALL deterministic items the ceiling is the single-prompt-'inne' base rate (4/9),
    # never a perfect score. This is the measurement-integrity finding SPEC-B04 acts on.
    label, ceiling = mod.null_ceiling(ITEMS, subset="all")
    assert ceiling < 1.0
    assert label == "inne"
    assert ceiling <= 0.5  # base rate of unanimously-'inne' items, not a real classification
    # And the contrastive headline is fully null-proof:
    _, ceiling_c = mod.null_ceiling(ITEMS, subset="contrastive")
    assert ceiling_c == 0.0
