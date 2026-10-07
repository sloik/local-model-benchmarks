"""Tests for scripts/model_identity.py (SPEC-003-018)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import model_identity as mi  # noqa: E402


@pytest.fixture
def registry():
    return mi.load_registry()


@pytest.fixture
def tmp_results(tmp_path):
    """A small synthetic results/ tree with one ambiguity fixture."""
    root = tmp_path / "results"
    root.mkdir()

    def make(name, model_id=None, valid_json=True):
        d = root / name
        d.mkdir()
        if model_id is not None:
            payload = {"run_id": name, "model_id": model_id}
            if valid_json:
                (d / "run_info.json").write_text(json.dumps(payload))
            else:
                (d / "run_info.json").write_text("{not json")

    make("run-a", model_id="alpha")
    make("run-b", model_id="alpha-alias")
    make("run-c", model_id=None)          # no run_info.json at all
    make("run-d", model_id="alpha", valid_json=False)  # unparsable
    make("run-e", model_id="beta")
    return root


@pytest.fixture
def tmp_registry():
    return {
        "alpha": {
            "modelKey": "alpha", "artifact": "x/alpha", "quantization": "4bit",
            "lifecycle_state": "installed", "aliases": ["alpha", "alpha-alias"],
        },
        "beta": {
            "modelKey": "beta", "artifact": "x/beta", "quantization": "4bit",
            "lifecycle_state": "installed", "aliases": ["beta"],
        },
    }


# ── AC1: qwen3.8-27b vs qwen3.8-27b-mlx must never conflate ──────────────────

def test_resolve_qwen38_27b_is_the_retained_4bit_model(registry):
    assert mi.resolve("qwen3.8-27b", registry) == "qwen3.8-27b"


def test_qwen38_27b_runs_never_include_the_rejected_2bit_directory(registry):
    runs = mi.runs_for("qwen3.8-27b", registry)
    assert "2026-08-22_qwen3.8-27b-mlx_mlx2bit-ctx246784-gated" not in runs
    assert len(runs) > 0


def test_rejected_2bit_model_resolves_to_its_own_distinct_key(registry):
    assert mi.resolve("qwen3.8-27b-mlx", registry) == "qwen3.8-27b-mlx"
    assert mi.resolve("qwen3.8-27b-mlx", registry) != mi.resolve("qwen3.8-27b", registry)


def test_no_substring_or_prefix_matching(registry):
    # "qwen3.8-27b" is a prefix of "qwen3.8-27b-mlx" but must not fuzzy-match.
    with pytest.raises(mi.UnknownModelKeyError):
        mi.resolve("qwen3.8-27b-does-not-exist", registry)


# ── AC2: muse-glimmer-30b has three historical identities ────────────────────

def test_muse_glimmer_runs_cover_both_historical_prefixes(registry):
    runs = mi.runs_for("muse-glimmer-30b", registry)
    assert any("lmstudio-community--muse-glimmer-30b" in r for r in runs)
    assert any("argo--muse-glimmer-30b" in r for r in runs)


# ── AC3: qwen3.6-27b has three historical keys ────────────────────────────────

def test_qwen36_runs_cover_all_three_historical_keys(registry):
    runs = mi.runs_for("qwen/qwen3.6-27b", registry)
    assert len(runs) == 8  # 4 native + 3 ctx32k alias + 1 5bit alias


# ── AC4: ambiguous key raises naming both candidates ──────────────────────────

def test_ambiguous_key_raises_with_both_candidates():
    ambiguous_registry = {
        "canonical-a": {"aliases": ["shared-key"]},
        "canonical-b": {"aliases": ["shared-key"]},
    }
    with pytest.raises(mi.AmbiguousModelKeyError) as exc_info:
        mi.resolve("shared-key", ambiguous_registry)
    assert set(exc_info.value.candidates) == {"canonical-a", "canonical-b"}


# ── AC5: full-tree attribution matches the audit's measured split ────────────

def test_attribute_all_matches_audit_baseline_split(registry):
    result = mi.attribute_all(registry)
    attributed_count = sum(len(v) for v in result["attributed"].values())
    assert attributed_count == 38
    assert len(result["unattributable"]) == 59


def test_attribute_all_never_attributes_by_directory_name(registry):
    result = mi.attribute_all(registry)
    # The rejected 2-bit directory's own name contains "qwen3.8-27b" as a
    # substring, but its run_info.json model_id is qwen3.8-27b-mlx — it must
    # land under that distinct canonical key, never under qwen3.8-27b.
    assert (
        "2026-08-22_qwen3.8-27b-mlx_mlx2bit-ctx246784-gated"
        in result["attributed"]["qwen3.8-27b-mlx"]
    )
    assert (
        "2026-08-22_qwen3.8-27b-mlx_mlx2bit-ctx246784-gated"
        not in result["attributed"].get("qwen3.8-27b", [])
    )


def test_attribute_all_buckets_missing_and_unparsable_run_info(tmp_registry, tmp_results):
    result = mi.attribute_all(tmp_registry, results_dir=tmp_results)
    assert result["attributed"]["alpha"] == ["run-a", "run-b"]
    assert result["attributed"]["beta"] == ["run-e"]
    assert set(result["unattributable"]) == {"run-c", "run-d"}


def test_attribute_all_raises_on_unmapped_model_id(tmp_registry, tmp_results):
    d = tmp_results / "run-unmapped"
    d.mkdir()
    (d / "run_info.json").write_text(json.dumps({"run_id": "run-unmapped", "model_id": "gamma"}))
    with pytest.raises(mi.UnknownModelKeyError):
        mi.attribute_all(tmp_registry, results_dir=tmp_results)


# ── AC6/R6: frozen results/ — a full test run leaves it untouched ────────────

def test_results_tree_is_never_modified_by_attribution(registry):
    # attribute_all/runs_for only ever open files for reading.
    before = {
        p: p.stat().st_mtime
        for p in mi.RESULTS_DIR.rglob("run_info.json")
    }
    mi.attribute_all(registry)
    for name in ("qwen3.8-27b", "qwen/qwen3.6-27b"):
        mi.runs_for(name, registry)
    after = {
        p: p.stat().st_mtime
        for p in mi.RESULTS_DIR.rglob("run_info.json")
    }
    assert before == after


# ── Registry integrity ────────────────────────────────────────────────────────

def test_every_registry_entry_has_a_valid_lifecycle_state(registry):
    for canonical, entry in registry.items():
        state = entry.get("lifecycle_state")
        assert state in mi.LIFECYCLE_STATES or state == "external-non-lmstudio", (
            f"{canonical}: unexpected lifecycle_state {state!r}"
        )


def test_no_alias_is_shared_across_two_canonical_entries(registry):
    seen: dict[str, str] = {}
    for canonical, entry in registry.items():
        for alias in entry.get("aliases", []):
            assert alias not in seen, (
                f"alias {alias!r} claimed by both {seen.get(alias)!r} and {canonical!r}"
            )
            seen[alias] = canonical
