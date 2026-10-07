"""Tests for scripts/run_provenance.py (SPEC-003-020)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import evaluate  # noqa: E402
import run_provenance as rp  # noqa: E402

BENCH_DIR = Path(__file__).parent.parent

FAKE_MODEL_INFO = {
    "publisher": "mlx-community",
    "key": "qwen3.8-27b",
    "quantization": {"name": "4bit"},
    "loaded_instances": [{"config": {"context_length": 131072}}],
}


@pytest.fixture(autouse=True)
def _stub_runtime_version(monkeypatch):
    monkeypatch.setattr(rp, "fetch_runtime_version", lambda: "07b7252")


# ── R1/AC1/AC7: schema shape and validator ────────────────────────────────────

def test_build_provenance_block_has_all_required_fields_and_non_null_runtime():
    block = rp.build_provenance_block(FAKE_MODEL_INFO, settings_gate=None)
    for field in rp.REQUIRED_FIELDS:
        assert field in block
    assert block["runtime_name"] == "lmstudio"
    assert block["runtime_version"]  # non-null/non-empty


def test_valid_block_passes_validation():
    block = rp.build_provenance_block(FAKE_MODEL_INFO)
    rp.validate_provenance(block)  # must not raise


def test_validator_rejects_missing_required_field():
    block = rp.build_provenance_block(FAKE_MODEL_INFO)
    del block["quantization"]
    with pytest.raises(rp.ProvenanceSchemaError, match="quantization"):
        rp.validate_provenance(block)


def test_validator_rejects_wrong_type_for_context_length():
    block = rp.build_provenance_block(FAKE_MODEL_INFO)
    block["actual_context_length"] = "131072"  # string of digits, not int or the sentinel enum
    with pytest.raises(rp.ProvenanceSchemaError, match="actual_context_length"):
        rp.validate_provenance(block)


def test_validator_rejects_unexpected_field():
    block = rp.build_provenance_block(FAKE_MODEL_INFO)
    block["extra_field"] = "nope"
    with pytest.raises(rp.ProvenanceSchemaError, match="unexpected"):
        rp.validate_provenance(block)


def test_schema_file_matches_the_validator_shape():
    schema = json.loads((BENCH_DIR / "suites/schemas/run-info.schema.json").read_text())
    assert set(schema["required"]) == set(rp.REQUIRED_FIELDS)


# ── R2/AC3: undeterminable field is 'unknown', never omitted ─────────────────

def test_missing_model_info_still_returns_every_field_as_unknown():
    block = rp.build_provenance_block(None, settings_gate=None)
    assert block["model_artifact"] == "unknown"
    assert block["quantization"] == "unknown"
    assert block["actual_context_length"] == "unknown"
    assert block["speculative_decoding"] == "unknown"
    rp.validate_provenance(block)  # 'unknown' strings must still be schema-valid


def test_context_length_prefers_settings_gate_over_loaded_instance():
    block = rp.build_provenance_block(FAKE_MODEL_INFO, settings_gate={"actual_context_length": 246784})
    assert block["actual_context_length"] == 246784


# ── R3/AC2: one shared builder for both scoring paths ─────────────────────────

def test_same_inputs_from_both_call_sites_produce_byte_identical_structure():
    run_info_with_provenance = {"provenance": rp.build_provenance_block(FAKE_MODEL_INFO)}
    block_a = evaluate.provenance_for_scoring(run_info_with_provenance, "qwen3.8-27b", "http://x/v1")
    block_b = evaluate.provenance_for_scoring(run_info_with_provenance, "qwen3.8-27b", "http://x/v1")
    assert json.dumps(block_a, sort_keys=True) == json.dumps(block_b, sort_keys=True)


def test_provenance_for_scoring_reuses_the_runs_own_recorded_block():
    recorded = rp.build_provenance_block(FAKE_MODEL_INFO)
    result = evaluate.provenance_for_scoring({"provenance": recorded}, "qwen3.8-27b", "http://x/v1")
    assert result == recorded


def test_provenance_for_scoring_falls_back_on_unreachable_lm_studio(monkeypatch):
    def _raise(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(evaluate, "fetch_native_local_model_metadata", _raise)
    result = evaluate.provenance_for_scoring({}, "some-model", "http://unreachable/v1")
    rp.validate_provenance(result)  # never crashes; still schema-valid


# ── R4/R5/AC4: historical run-index matches SPEC-003-018 attribution ─────────

def test_build_run_index_covers_every_directory_with_matching_attribution():
    import model_identity
    registry = model_identity.load_registry()
    attribution = model_identity.attribute_all(registry)
    index = rp.build_run_index(registry=registry)

    all_dirs = {p.name for p in rp.RESULTS_DIR.iterdir() if p.is_dir()}
    assert set(index.keys()) == all_dirs

    attributed_dirs = {name for names in attribution["attributed"].values() for name in names}
    for name, entry in index.items():
        expected = "attributed" if name in attributed_dirs else "unattributable"
        assert entry["classification"] == expected


# ── AC5: no historical field is populated by inference ───────────────────────

def test_historical_provenance_never_infers_from_directory_name():
    block = rp.historical_provenance_from_run_info({})
    assert block == {
        "runtime_name": "unavailable",
        "runtime_version": "unavailable",
        "model_artifact": "unavailable",
        "quantization": "unavailable",
        "speculative_decoding": "unavailable",
        "actual_context_length": "unavailable",
    }


def test_run_index_entries_without_run_info_are_fully_unavailable():
    index = rp.build_run_index()
    for name, entry in index.items():
        run_info_path = rp.RESULTS_DIR / name / "run_info.json"
        if not run_info_path.is_file():
            assert entry["provenance"]["runtime_name"] == "unavailable"
            assert entry["provenance"]["actual_context_length"] == "unavailable"


def test_gated_historical_runs_keep_their_recorded_context_length():
    index = rp.build_run_index()
    gated = index.get("2026-08-22_qwen3.8-27b-mlx_mlx2bit-ctx246784-gated")
    assert gated is not None
    assert gated["provenance"]["actual_context_length"] == 246784
    # everything else about a historical run stays unavailable, never inferred
    assert gated["provenance"]["runtime_name"] == "unavailable"


# ── AC6/R4: results/ is never modified by index building ─────────────────────

def test_build_run_index_leaves_results_directory_untouched():
    before = subprocess.run(
        ["git", "status", "--porcelain", "results/"],
        cwd=BENCH_DIR, capture_output=True, text=True,
    ).stdout
    rp.build_run_index()
    after = subprocess.run(
        ["git", "status", "--porcelain", "results/"],
        cwd=BENCH_DIR, capture_output=True, text=True,
    ).stdout
    assert before == after
