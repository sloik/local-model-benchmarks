"""Publication privacy and fixture-integrity checks supplement functional tests."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("release_audit", ROOT / "scripts/audit_public_release.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.mark.parametrize("payload,kind", [
    (b"/" + b"Users/" + b"reviewer/private.txt", "personal-machine-path"),
    (b"Drop" + b"box/private-report", "private-storage-reference"),
    (b"sk-" + b"a" * 40, "provider-key"),
    (b"ghp_" + b"b" * 40, "github-token"),
    (b"contact@" + b"private.invalid", "non-public-contact-address"),
])
def test_release_gate_rejects_actual_leak_shapes(payload, kind):
    assert kind in audit.content_findings(payload)


def test_release_gate_accepts_placeholders_and_public_commit_identity():
    assert not audit.content_findings(b"LM_STUDIO_TOKEN=replace-with-local-token")
    assert not audit.content_findings(b"1531955+sloik@users.noreply.github.com")


def test_publication_fixture_hashes_are_exact():
    manifest = json.loads((ROOT / "PUBLICATION.json").read_text())
    assert manifest["corpus_id"] == "public-synthetic-v1"
    assert manifest["fixture_hashes"]
    for name, digest in manifest["fixture_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name


def test_replaced_cases_remain_present_with_original_prompt_counts():
    manifest = json.loads((ROOT / "PUBLICATION.json").read_text())
    for row in manifest["canonical_case_changes"]:
        if "case_id" not in row:
            continue
        case = next(t for t in json.loads((ROOT / row["path"]).read_text())["tests"]
                    if t["id"] == row["case_id"])
        assert case["source_note"]
        assert len(case["prompts"]) == row.get("prompt_count", 2)


def test_caption_stand_ins_are_explicit_and_self_consistent():
    for video in ("8vUCjYsWeSU", "R1TNGOZAOZs"):
        fixture = json.loads((ROOT / f"test_data/cat07/transcripts/{video}.json").read_text())
        assert "synthetic" in fixture["source_note"].lower()
        assert hashlib.sha256(fixture["transcript_excerpt"].encode()).hexdigest() == fixture["transcript_excerpt_fingerprint_sha256"]


def test_historical_payloads_are_closed_field_summaries():
    for path in (ROOT / "results").glob("*/*.json"):
        data = json.loads(path.read_text())
        assert "public_release_note" in data
        assert not any(key in data for key in ("raw", "rationale", "prompts", "base_url", "command"))
        if "summary" in data:
            assert all(isinstance(value, (int, float, bool)) or value is None
                       for metrics in data["summary"].values() for value in metrics.values())
