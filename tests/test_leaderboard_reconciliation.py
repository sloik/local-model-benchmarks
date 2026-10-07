"""Tests for the LEADERBOARD.md fleet-status reconciliation (SPEC-003-021)."""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import check_leaderboard_fleet_status as check_mod  # noqa: E402

BENCH_DIR = Path(__file__).parent.parent
LEADERBOARD_MD = BENCH_DIR / "LEADERBOARD.md"


# ── AC1: no `fleet`-labelled model is absent from live `lms ls` ──────────────

def test_real_leaderboard_has_no_fleet_status_violations(monkeypatch):
    # Use an explicit historical inventory fixture, not the test host's installed models.
    monkeypatch.setattr(check_mod, "fetch_installed_display_names", lambda: {
        "qwen3.8-27b", "qwen/qwen3.6-27b", "muse-glimmer-30b",
        "gemma-4-31b-it", "ornith-1.5-35b-a3b-mlx",
    })
    violations = check_mod.check()
    assert violations == []


def test_check_flags_a_fleet_model_not_in_live_inventory(tmp_path, monkeypatch):
    fixture = tmp_path / "LEADERBOARD.md"
    fixture.write_text(
        "## 2. Full model leaderboard (all tested, with metadata)\n\n"
        "| model | params | arch | quant | max ctx | disk | tool-trained | vision | serving | "
        "quality (CAT) | aider code | Hermes code | status |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| Totally-Fake-Model-9000 | 1B | x | x | 1 | 1 | x | x | LM Studio | 1.0 | — | — | **fleet** |\n"
        "\n## 3. Next section\n"
    )
    monkeypatch.setattr(check_mod, "fetch_installed_display_names", lambda: {"gemma-4-31b-it"})
    violations = check_mod.check(fixture)
    assert len(violations) == 1
    assert "Totally-Fake-Model-9000" in violations[0]


def test_apple_fm_native_status_is_exempt_from_lms_ls_check(tmp_path, monkeypatch):
    fixture = tmp_path / "LEADERBOARD.md"
    fixture.write_text(
        "## 2. Full model leaderboard (all tested, with metadata)\n\n"
        "| model | params | arch | quant | max ctx | disk | tool-trained | vision | serving | "
        "quality (CAT) | aider code | Hermes code | status |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
        "| Apple FM (system) | ~3B | Apple FM | — | ~4096 | on-device | ✓ | ✗ | fm serve | "
        "3.6 | N/A | N/A | **fleet (native)** |\n"
        "\n## 3. Next section\n"
    )
    monkeypatch.setattr(check_mod, "fetch_installed_display_names", lambda: set())
    assert check_mod.check(fixture) == []


# ── AC2: the three currently-installed models each have a §2 row ────────────

@pytest.mark.parametrize("model", ["qwen3.8-27b", "qwen/qwen3.6-27b", "muse-glimmer-30b"])
def test_section_2_has_a_row_for_each_previously_missing_installed_model(model):
    rows = check_mod.extract_section_2_rows(LEADERBOARD_MD.read_text(encoding="utf-8"))
    assert any(r[0] == model for r in rows), f"no §2 row for {model}"


# ── AC3: the retired pass/5 bar is not published as current anywhere ─────────

def test_no_file_publishes_the_retired_gemma_5_of_5_bar_as_current():
    # The retirement note itself may *mention* "5/5" as explicitly-historical
    # context (that's what the AC allows) — it must not appear as a live data
    # cell. leaderboard-summary.csv's only data row is the retirement note.
    with open(BENCH_DIR / "leaderboard-summary.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["status"] == "retired"

    # LEADERBOARD.md's own section 2 table must show the corrected 3/3, not 5/5.
    leaderboard_text = LEADERBOARD_MD.read_text(encoding="utf-8")
    assert "| Gemma-4-31b | 31B | gemma4 | Q4_K_M | 262144 | 19.9 GB | ✓ | ✓ | LM Studio | 4.10 | **3/3** | **3/3** |" in leaderboard_text


# ── AC4: leaderboard-summary.csv parses with consistent field counts ────────

def test_leaderboard_summary_csv_loads_with_consistent_field_counts():
    with open(BENCH_DIR / "leaderboard-summary.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows  # non-empty
    field_count = len(rows[0])
    for row in rows:
        assert len(row) == field_count


# ── AC5: no results.csv/quarterly.csv duplicate pair in the root ────────────

def test_no_results_csv_in_benchmarks_root():
    assert not (BENCH_DIR / "results.csv").exists()


def test_relocated_fixture_still_has_its_original_content():
    relocated = BENCH_DIR / "tests" / "fixtures" / "cat04-quarterly-sample.csv"
    assert relocated.is_file()
    assert "kwartal;przychody_mln;ebit_mln" in relocated.read_text(encoding="utf-8")


# ── AC6: results/ untouched ───────────────────────────────────────────────────

def test_results_directory_has_no_pending_git_changes():
    out = subprocess.run(
        ["git", "status", "--porcelain", "results/"],
        cwd=BENCH_DIR, capture_output=True, text=True,
    ).stdout
    assert out == ""


# ── AC7: every historical score value survives unchanged ────────────────────

@pytest.mark.parametrize("expected_fragment", [
    "| Ornith-1.0-35B | 35B MoE | qwen35moe | Q6_K | 262144 | 28.5 GB | ✓ | ✗ | LM Studio | 4.22 | **2/3** | **3/3** |",
    "| Ornith-1.0-9B | 9B | qwen35 | Q8_0 | 262144 | 9.5 GB | ✓ | ✗ | LM Studio | 4.10 | 1/3 | 2/3 |",
    "| Mistral-small-3.2 | 24B | mistral3 | 4bit | 131072 | 13.5 GB | ✗ | ✓ | LM Studio | 4.00 | 1/3 | 1/3 |",
    "| Apple FM (system) | ~3B | Apple FM | — | ~4096 | on-device | ✓ (guided) | ✗ | `fm serve` (native) | 3.60 | N/A | N/A |",
])
def test_historical_score_rows_are_unchanged(expected_fragment):
    text = LEADERBOARD_MD.read_text(encoding="utf-8")
    assert expected_fragment in text


def test_previously_fleet_labelled_deleted_models_now_say_deleted():
    rows = check_mod.extract_section_2_rows(LEADERBOARD_MD.read_text(encoding="utf-8"))
    by_model = {model: status for model, status in rows}
    for model in ("Ornith-1.0-35B", "Ornith-1.0-9B", "Mistral-small-3.2"):
        assert "deleted" in by_model[model]
        assert "fleet" not in by_model[model].lower()
