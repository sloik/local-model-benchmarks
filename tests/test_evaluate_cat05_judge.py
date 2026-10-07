"""Tests for calibrated local-judge routing in scripts/evaluate.py. No model calls."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "scripts"))

from evaluate import (  # noqa: E402
    CANONICAL_LOCAL_EVALUATORS,
    DEFAULT_LOCAL_EVALUATOR_MODEL,
    resolve_evaluator_model_map,
    resolve_local_evaluator_selection,
    validate_local_evaluator_preflight,
)


def test_local_default_is_calibrated_qwen_without_category_override():
    assert DEFAULT_LOCAL_EVALUATOR_MODEL == "qwen/qwen3.6-27b"
    assert resolve_evaluator_model_map("", "local") == {}


def test_cli_map_overrides_the_cat05_default():
    m = resolve_evaluator_model_map("CAT-05:gemma-4-31b-it", "local")
    assert m["CAT-05"] == "gemma-4-31b-it"


def test_cli_other_category_does_not_invent_cat05_override():
    m = resolve_evaluator_model_map("CAT-04:gemma-4-31b-it", "local")
    assert m == {"CAT-04": "gemma-4-31b-it"}


def test_default_applies_only_to_local_evaluator():
    assert resolve_evaluator_model_map("", "claude") == {}
    assert resolve_evaluator_model_map("", "argo") == {}


def test_qwen_subject_swaps_to_installed_muse():
    evaluator, model_map = resolve_local_evaluator_selection(
        DEFAULT_LOCAL_EVALUATOR_MODEL,
        {},
        "qwen/qwen3.6-27b",
        {
            "qwen/qwen3.6-27b",
            "muse-glimmer-30b",
            "gemma-4-31b-it",
        },
        allow_self_evaluation=False,
    )
    assert evaluator == "muse-glimmer-30b"
    assert model_map == {}


def test_missing_explicit_evaluator_fails_instead_of_silent_substitution():
    import pytest

    with pytest.raises(ValueError, match="not installed"):
        resolve_local_evaluator_selection(
            "deleted/model",
            {},
            "gemma-4-31b-it",
            {"qwen/qwen3.6-27b", "gemma-4-31b-it"},
            allow_self_evaluation=False,
        )


def test_no_independent_installed_judge_fails_clearly():
    import pytest

    with pytest.raises(ValueError, match="No installed independent evaluator"):
        resolve_local_evaluator_selection(
            "qwen/qwen3.6-27b",
            {},
            "qwen/qwen3.6-27b",
            {"qwen/qwen3.6-27b"},
            allow_self_evaluation=False,
        )


def native_model(context=131072, maximum=262144, *, loaded=True):
    return {
        "max_context_length": maximum,
        "loaded_instances": [{"config": {"context_length": context}}] if loaded else [],
    }


def test_local_preflight_accepts_canonical_judge_at_or_above_baseline():
    result = validate_local_evaluator_preflight(
        {DEFAULT_LOCAL_EVALUATOR_MODEL},
        {DEFAULT_LOCAL_EVALUATOR_MODEL: native_model(context=188928)},
        131072,
        allow_experimental_evaluator=False,
    )

    assert result["minimum_context_length"] == 131072
    assert result["evaluators"][0]["active_context_length"] == 188928
    assert DEFAULT_LOCAL_EVALUATOR_MODEL in CANONICAL_LOCAL_EVALUATORS


def test_local_preflight_rejects_unloaded_or_under_context_judge():
    import pytest

    with pytest.raises(ValueError, match="not loaded"):
        validate_local_evaluator_preflight(
            {DEFAULT_LOCAL_EVALUATOR_MODEL},
            {DEFAULT_LOCAL_EVALUATOR_MODEL: native_model(loaded=False)},
            131072,
            allow_experimental_evaluator=False,
        )
    with pytest.raises(ValueError, match="below required baseline"):
        validate_local_evaluator_preflight(
            {DEFAULT_LOCAL_EVALUATOR_MODEL},
            {DEFAULT_LOCAL_EVALUATOR_MODEL: native_model(context=8192)},
            131072,
            allow_experimental_evaluator=False,
        )


def test_local_preflight_rejects_insufficient_model_maximum():
    import pytest

    with pytest.raises(ValueError, match="maximum context"):
        validate_local_evaluator_preflight(
            {DEFAULT_LOCAL_EVALUATOR_MODEL},
            {DEFAULT_LOCAL_EVALUATOR_MODEL: native_model(maximum=65536)},
            131072,
            allow_experimental_evaluator=False,
        )


def test_local_preflight_requires_opt_in_for_experimental_judge():
    import pytest

    ornith = "ornith-1.5-35b-a3b-mlx"
    metadata = {ornith: native_model()}
    with pytest.raises(ValueError, match="non-canonical"):
        validate_local_evaluator_preflight(
            {ornith}, metadata, 131072, allow_experimental_evaluator=False
        )

    result = validate_local_evaluator_preflight(
        {ornith}, metadata, 131072, allow_experimental_evaluator=True
    )
    assert result["allow_experimental_evaluator"] is True
