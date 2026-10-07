"""Portable tier-2 source input; all checks run without model calls."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import score_cat06_t2 as scorer


def test_report_preserves_order_labels_and_original_whitespace_normalization(tmp_path):
    (tmp_path / "b.txt").write_text("  second   page\n\n\n\nend  ")
    (tmp_path / "a.txt").write_text("first\t\tpage")
    assert scorer.load_full_report(tmp_path) == (
        "=== SOURCE: a.txt ===\nfirst page\n\n"
        "=== SOURCE: b.txt ===\nsecond page\n\nend"
    )


@pytest.mark.parametrize("kind", ["absent", "no_text", "blank"])
def test_report_rejects_missing_or_empty_inputs(tmp_path, kind):
    path = tmp_path
    if kind == "absent":
        path = tmp_path / "absent"
    elif kind == "no_text":
        (tmp_path / "report.md").write_text("not a .txt extract")
    else:
        (tmp_path / "blank.txt").write_text(" \n\t")
    with pytest.raises(ValueError):
        scorer.load_full_report(path)


def test_cli_fails_before_inference_when_source_not_supplied(monkeypatch, capsys):
    monkeypatch.delenv("BENCHMARK_REPORT_EXTRACTS_DIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["score_cat06_t2.py", "--model", "example"])
    monkeypatch.setattr(scorer, "ask", lambda *a, **k: pytest.fail("unexpected inference"))
    with pytest.raises(SystemExit) as error:
        scorer.main()
    assert error.value.code == 2
    assert "need --extracts-dir" in capsys.readouterr().err


@pytest.mark.parametrize("explicit", [False, True])
def test_cli_reads_environment_input_and_explicit_override(tmp_path, monkeypatch, explicit):
    extracts = tmp_path / "extracts"
    extracts.mkdir()
    (extracts / "report.txt").write_text("A non-empty source")
    output = tmp_path / "out.json"
    argv = ["score_cat06_t2.py", "--null", "always_answer", "--only", "CAT-06-001",
            "--out", str(output)]
    if explicit:
        monkeypatch.setenv("BENCHMARK_REPORT_EXTRACTS_DIR", str(tmp_path / "wrong"))
        argv.extend(["--extracts-dir", str(extracts)])
    else:
        monkeypatch.setenv("BENCHMARK_REPORT_EXTRACTS_DIR", str(extracts))
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(scorer, "ask", lambda *a, **k: pytest.fail("unexpected inference"))
    scorer.main()
    assert output.is_file()


@pytest.mark.parametrize("policy", ["always_refuse", "always_answer"])
def test_full_report_null_policy_never_calls_a_model(tmp_path, monkeypatch, policy):
    report = tmp_path / "report"
    report.mkdir()
    (report / "source.txt").write_text("An original fictional report.")
    output = tmp_path / "score.json"
    monkeypatch.setattr(sys, "argv", ["score_cat06_t2.py", "--null", policy,
                        "--only", "CAT-06-007", "--extracts-dir", str(report),
                        "--out", str(output)])
    monkeypatch.setattr(scorer, "ask", lambda *a, **k: pytest.fail("null policy invoked a model"))
    monkeypatch.setattr(scorer, "detect", lambda *a, **k: pytest.fail("null policy invoked a detector"))
    scorer.main()
    assert output.is_file()
