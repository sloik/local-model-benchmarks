"""Regression tests for the live benchmark settings gate; no model calls."""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

import run_benchmark  # noqa: E402
from run_benchmark import validate_model_settings  # noqa: E402


def model_info(context=131072, maximum=262144):
    return {
        "key": "test-model",
        "max_context_length": maximum,
        "loaded_instances": [{"config": {"context_length": context}}],
    }


def test_settings_gate_accepts_minimum_context_and_equal_reasoning_ceilings():
    result = validate_model_settings(model_info(), 131072, 32768, 81920)

    assert result == {
        "policy": "minimum_context_and_equal_generation_ceilings",
        "minimum_context_length": 131072,
        "actual_context_length": 131072,
        "model_max_context_length": 262144,
        "max_tokens": 32768,
        "retry_max_tokens": 81920,
    }


def test_settings_gate_rejects_missing_loaded_instance():
    with pytest.raises(ValueError, match="no loaded instance"):
        validate_model_settings({"key": "test-model", "loaded_instances": []}, 131072, 32768, 81920)


def test_settings_gate_rejects_context_below_minimum():
    with pytest.raises(ValueError, match="below required minimum"):
        validate_model_settings(model_info(context=65536), 131072, 32768, 81920)


def test_settings_gate_accepts_larger_context_and_records_it():
    result = validate_model_settings(model_info(context=246784), 131072, 32768, 81920)

    assert result["minimum_context_length"] == 131072
    assert result["actual_context_length"] == 246784


def test_settings_gate_rejects_context_above_model_maximum():
    with pytest.raises(ValueError, match="exceeds model maximum"):
        validate_model_settings(model_info(context=131072, maximum=65536), 131072, 32768, 81920)


@pytest.mark.parametrize("standard,retry", [(0, 81920), (32768, 0), (32768, 16384)])
def test_settings_gate_rejects_invalid_generation_ceilings(standard, retry):
    with pytest.raises(ValueError, match="ceiling"):
        validate_model_settings(model_info(), 131072, standard, retry)


def test_keep_awake_allows_display_sleep_but_prevents_system_sleep(monkeypatch):
    captured = {}

    class FakeProcess:
        pass

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(run_benchmark.sys, "platform", "darwin")
    monkeypatch.setattr(run_benchmark.shutil, "which", lambda name: "/usr/bin/caffeinate")
    monkeypatch.setattr(run_benchmark.subprocess, "Popen", fake_popen)

    result = run_benchmark.acquire_keep_awake()

    assert isinstance(result, FakeProcess)
    assert captured["command"][:2] == ["/usr/bin/caffeinate", "-ims"]
    assert "d" not in captured["command"][1]
    assert captured["command"][2] == "-w"
    assert captured["kwargs"]["stdout"] is run_benchmark.subprocess.DEVNULL
    assert captured["kwargs"]["stderr"] is run_benchmark.subprocess.DEVNULL
