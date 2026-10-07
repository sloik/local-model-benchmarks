"""Tests for the leaderboard writer re-chain (SPEC-003-019)."""

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import backfill_leaderboard as bl  # noqa: E402
import evaluate  # noqa: E402
import report as report_module  # noqa: E402

BENCH_DIR = Path(__file__).parent.parent


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _score_payload(run_id, model_id, evaluator, overall=4.5):
    return {
        "run_id": run_id,
        "model_id": model_id,
        "eval_date": "2026-09-16",
        "evaluator": evaluator,
        "evaluator_type": "local",
        "summary": {
            "CAT-01": {"mean": overall}, "CAT-02": {"mean": overall},
            "CAT-03": {"mean": overall}, "CAT-04": {"mean": overall},
            "CAT-05": {"mean": overall}, "overall": {"mean": overall},
            "unchecked": {"prompts": 0},
        },
    }


# ── AC1: the scoring path writes the leaderboard with no manual report.py step ──

def test_update_leaderboard_from_score_output_writes_row_directly(tmp_path, monkeypatch):
    fake_leaderboard = tmp_path / "leaderboard.csv"
    monkeypatch.setattr(report_module, "LEADERBOARD", fake_leaderboard)

    output = _score_payload("run-ac1", "qwen3.8-27b", "qwen/qwen3.6-27b")
    evaluate.update_leaderboard_from_score_output(output)

    rows = _read_rows(fake_leaderboard)
    assert len(rows) == 1
    assert rows[0]["run_id"] == "run-ac1"
    assert rows[0]["model_id"] == "qwen3.8-27b"


def test_update_leaderboard_from_score_output_never_raises_on_bad_data(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(report_module, "LEADERBOARD", tmp_path / "leaderboard.csv")
    evaluate.update_leaderboard_from_score_output({"summary": {}})  # missing required keys
    assert "leaderboard update failed" in capsys.readouterr().err


# ── AC2/AC3: backfill covers every installed mid-size model with canonical keys ──

def test_backfill_covers_all_installed_mid_size_models_including_qwen38_4_70():
    rows = bl.build_rows()
    model_ids = {r["model_id"] for r in rows}
    for expected in (
        "qwen3.8-27b", "gemma-4-31b-it", "muse-glimmer-30b",
        "qwen/qwen3.6-27b", "ornith-1.5-35b-a3b-mlx",
    ):
        assert expected in model_ids, f"missing {expected} in backfilled leaderboard"

    qwen38_rows = [r for r in rows if r["model_id"] == "qwen3.8-27b"]
    assert any(abs(float(r["overall_mean"]) - 4.70) < 0.01 for r in qwen38_rows)


def test_backfill_never_carries_a_stale_key():
    import model_identity
    registry = model_identity.load_registry()
    rows = bl.build_rows()
    for row in rows:
        # Every row's model_id must itself be a canonical registry key.
        assert model_identity.resolve(row["model_id"], registry) == row["model_id"]


def test_backfill_never_conflates_qwen38_27b_with_the_rejected_2bit_variant():
    rows = bl.build_rows()
    for row in rows:
        if row["run_id"] == "2026-08-22_qwen3.8-27b-mlx_mlx2bit-ctx246784-gated":
            assert row["model_id"] == "qwen3.8-27b-mlx"


# ── AC4: two judges on one run produce two distinguishable rows ──────────────

def test_two_judges_on_one_run_do_not_collapse_to_one_row(tmp_path, monkeypatch):
    fake_leaderboard = tmp_path / "leaderboard.csv"
    monkeypatch.setattr(report_module, "LEADERBOARD", fake_leaderboard)

    labels = report_module.determine_labels(_score_payload("run-x", "gemma-4-31b-it", "judge-a")["summary"])
    report_module.update_leaderboard(_score_payload("run-x", "gemma-4-31b-it", "judge-a"), labels)
    report_module.update_leaderboard(_score_payload("run-x", "gemma-4-31b-it", "judge-b"), labels)

    rows = _read_rows(fake_leaderboard)
    assert len(rows) == 2
    assert {r["evaluator"] for r in rows} == {"judge-a", "judge-b"}


# ── AC5: backfill is idempotent ───────────────────────────────────────────────

def test_backfill_run_twice_is_byte_identical(tmp_path):
    out1 = tmp_path / "leaderboard-1.csv"
    out2 = tmp_path / "leaderboard-2.csv"
    rows = bl.build_rows()
    bl.write_leaderboard(rows, out1)
    bl.write_leaderboard(bl.build_rows(), out2)
    assert out1.read_bytes() == out2.read_bytes()


# ── AC6: backfill never touches results/ ──────────────────────────────────────

def test_backfill_leaves_results_directory_untouched():
    before = subprocess.run(
        ["git", "status", "--porcelain", "results/"],
        cwd=BENCH_DIR, capture_output=True, text=True,
    ).stdout
    bl.build_rows()  # dry run (no --write) — must not touch results/ regardless
    after = subprocess.run(
        ["git", "status", "--porcelain", "results/"],
        cwd=BENCH_DIR, capture_output=True, text=True,
    ).stdout
    assert before == after


# ── R5: find_score_files only picks report.py-compatible payloads ────────────

def test_find_score_files_excludes_incompatible_and_backup_json(tmp_path):
    run_dir = tmp_path / "run-fixture"
    run_dir.mkdir()
    (run_dir / "scores.json").write_text(json.dumps(_score_payload("r", "m", "j")))
    (run_dir / "argo_eval_verdicts-x.json").write_text(json.dumps({"verdicts": []}))
    (run_dir / "scores-judge-x.json.bak").write_text(json.dumps(_score_payload("r", "m", "j")))

    found = bl.find_score_files(run_dir)
    assert [p.name for p in found] == ["scores.json"]
