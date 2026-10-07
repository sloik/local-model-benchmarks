#!/usr/bin/env python3
"""Tests for scripts/cat07_harness.py + scripts/score_cat07.py (SPEC-001). NO live model calls.

The --null adversarial-guard tests below DO exercise cat07_harness.run_case(), but only its
offline null-model path (_null_turn) plus the hermetic frozen-corpus tool executor -- no network,
no LM Studio. Everything else uses hand-crafted transcripts injected directly into score_case().

Run: python -m pytest tests/test_score_cat07.py -v
"""
import importlib.util
import json
import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent / "scripts"


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # register before exec (DevKB python.md)
    spec.loader.exec_module(mod)
    return mod


harness = _load("cat07_harness", "cat07_harness.py")
scorer = _load("score_cat07", "score_cat07.py")

ITEMS = json.loads((_HERE.parent / "tests" / "cat-07-tool-use.json").read_text(encoding="utf-8"))
CORPUS = harness.load_corpus()


def _case(tid):
    return next(t for t in ITEMS["tests"] if t["id"] == tid)


# ── corpus integrity ──────────────────────────────────────────────────────────────
def test_at_least_five_transcript_and_three_financial_cases():
    types = [t["task_type"] for t in ITEMS["tests"]]
    assert types.count("transcript_retrieval") >= 5
    assert types.count("financial_report_retrieval") >= 3


def test_every_case_frozen_source_key_resolves_in_the_corpus():
    for case in ITEMS["tests"]:
        key = case["frozen_source_key"]
        if case["task_type"] == "transcript_retrieval":
            assert key in CORPUS["transcripts"], "missing frozen transcript: %s" % key
        else:
            assert any(key.startswith(c[0].split()[0].lower())
                       for c in CORPUS["financial"]), "no financial entry matches key %s" % key


def test_every_case_has_a_deterministic_check():
    for case in ITEMS["tests"]:
        assert case["checks"].get("final_answer_must_contain_any"), case["id"]


# ── hermetic tool execution ────────────────────────────────────────────────────────
def test_execute_yt_dlp_metadata_mode_returns_frozen_title():
    result, video_id = harness.execute_yt_dlp(
        {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "mode": "metadata"}, CORPUS)
    assert video_id == "dQw4w9WgXcQ"
    assert "Never Gonna Give You Up" in result["title"]


def test_execute_yt_dlp_unknown_video_is_not_found():
    result, video_id = harness.execute_yt_dlp(
        {"url": "https://www.youtube.com/watch?v=AAAAAAAAAAA", "mode": "metadata"}, CORPUS)
    assert video_id is None
    assert result["error"] == "not_found"


def test_execute_financial_exact_date_is_exact_match():
    result, key, exact = harness.execute_financial(
        {"company": "PolCorp SA", "report_date": "2025-11-15"}, CORPUS)
    assert exact is True
    assert "450" in result["text"]


def test_execute_financial_wrong_date_same_company_is_stale_not_exact():
    result, key, exact = harness.execute_financial(
        {"company": "PolCorp SA", "report_date": "1999-01-01"}, CORPUS)
    assert exact is False
    assert result["company"] == "PolCorp SA"  # a REAL snapshot, just not the requested date


def test_execute_financial_ticker_only_company_arg_resolves_same_as_full_name():
    # SPEC-003-002: expected_args_schema across CAT-07-F-001..006 accepts either the company name
    # or its ticker at the schema layer (must_contain_any). A model that reasonably passes just the
    # ticker (e.g. because the prompt gave it "ticker: PCO" for the distractor-tool trap in
    # CAT-07-F-006) must resolve to the exact same corpus entry as the full company name -- not
    # silently fall through to bad_args for using a schema-legal identifier.
    result, key, exact = harness.execute_financial({"company": "PCO", "report_date": "2025-11-15"}, CORPUS)
    assert exact is True
    assert result["company"] == "PolCorp SA"
    assert "450" in result["text"]
    full_name_result, full_name_key, full_name_exact = harness.execute_financial(
        {"company": "PolCorp SA", "report_date": "2025-11-15"}, CORPUS)
    assert key == full_name_key
    assert exact == full_name_exact


def test_execute_financial_unknown_company_is_not_found():
    result, key, exact = harness.execute_financial(
        {"company": "Totally Unknown Corp", "report_date": "2025-11-15"}, CORPUS)
    assert key is None
    assert result["error"] == "not_found"


# ── score_case verdicts (hand-crafted transcripts, no live calls) ──────────────────
def _transcript(tool_calls_made=(), final_answer="", no_call=None, error=None):
    return {
        "tool_calls_made": list(tool_calls_made),
        "final_answer": final_answer,
        "no_call": no_call if no_call is not None else len(tool_calls_made) == 0,
        "error": error,
    }


def test_no_call_at_all_is_wrong_tool():
    case = _case("CAT-07-T-003")
    row = scorer.score_case(case, _transcript(final_answer="It's about 3.5 minutes long."))
    assert row["verdict"] == "wrong_tool"
    assert row["case_score"] == 0.0


def test_calling_a_different_tool_is_wrong_tool():
    case = _case("CAT-07-F-001")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "x", "mode": "metadata"}, "raw_result": {}},
    ]))
    assert row["verdict"] == "wrong_tool"


def test_wrong_mode_is_bad_args():
    case = _case("CAT-07-T-001")  # requires mode=transcript
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=8vUCjYsWeSU",
                                     "mode": "metadata"},
         "raw_result": {"title": "..."}},
    ], final_answer="harnesses are becoming more important than the models"))
    assert row["verdict"] == "bad_args"


def test_wrong_video_id_in_url_is_bad_args():
    case = _case("CAT-07-T-001")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=WRONGIDWRONG",
                                     "mode": "transcript"},
         "raw_result": {"transcript": "..."}},
    ], final_answer="harnesses are becoming more important than the models"))
    assert row["verdict"] == "bad_args"


def test_resolved_to_no_corpus_entry_despite_schema_valid_args_is_bad_args():
    case = _case("CAT-07-F-001")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "get_financial_report_snapshot",
         "args": {"company": "PolCorp SA", "report_date": "2025-11-15"},
         "raw_result": {"error": "not_found"}, "exact_match": False},
    ]))
    assert row["verdict"] == "bad_args"


def test_same_company_wrong_date_is_stale_snapshot_mismatch():
    case = _case("CAT-07-F-002")  # requires report_date 2025-08-14
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "get_financial_report_snapshot",
         "args": {"company": "PolCorp SA", "report_date": "2025-08-14"},
         "raw_result": {"company": "PolCorp SA", "report_date": "2025-11-15", "text": "..."},
         "exact_match": False, "resolved_key": ["polcorp", "2025-11-15"]},
    ], final_answer="EPS was 1.24 PLN"))
    assert row["verdict"] == "stale_snapshot_mismatch"


def test_correct_fetch_but_answer_missing_the_fact_is_hallucinated_content():
    case = _case("CAT-07-T-003")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                     "mode": "metadata"},
         "raw_result": {"title": "Never Gonna Give You Up"}, "exact_match": True},
    ], final_answer="It is a classic 1987 pop song, about 3 and a half minutes long."))
    assert row["verdict"] == "hallucinated_content"


def test_correct_fetch_and_forbidden_content_present_is_hallucinated_content():
    case = _case("CAT-07-F-002")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "get_financial_report_snapshot",
         "args": {"company": "PolCorp SA", "report_date": "2025-08-14"},
         "raw_result": {"text": "..."}, "exact_match": True},
    ], final_answer="EPS was 1.24 PLN"))  # 1.24 is the Q3 figure, forbidden for the Q2 case
    assert row["verdict"] == "hallucinated_content"


def test_markdown_emphasis_does_not_defeat_a_correct_content_match():
    # Regression: live fleet run (qwen/qwen3.6-27b, CAT-07-T-002) answered "build a **graph**
    # based on memory" -- factually correct, but the un-stripped "**" defeated a substring check
    # for "a graph" and produced a false hallucinated_content verdict.
    case = _case("CAT-07-T-002")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=R1TNGOZAOZs",
                                     "mode": "transcript"},
         "raw_result": {"transcript": "..."}, "exact_match": True},
    ], final_answer=(
        "Based on the opening of the video, the speaker states that the agent is able to build "
        "a **graph** based on memory."
    )))
    assert row["verdict"] == "pass"


def test_correct_tool_valid_args_faithful_answer_is_pass():
    case = _case("CAT-07-T-001")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=8vUCjYsWeSU",
                                     "mode": "transcript"},
         "raw_result": {"transcript": "..."}, "exact_match": True},
    ], final_answer="The speaker says harnesses are becoming more important than the models."))
    assert row["verdict"] == "pass"
    assert row["case_score"] == 1.0


def test_harness_error_is_unscored_never_zero_credited():
    case = _case("CAT-07-T-001")
    row = scorer.score_case(case, _transcript(error="EMPTY_ALL_REASONING(reasoning_tokens=8192)"))
    assert row["verdict"] == "UNSCORED"
    assert row["case_score"] is None


# ── aggregation math ────────────────────────────────────────────────────────────────
def test_score_aggregation_excludes_unscored_from_coverage_but_zero_credits_real_failures():
    items = {"tests": [_case("CAT-07-T-001"), _case("CAT-07-T-003")]}
    injected = {
        "CAT-07-T-001": _transcript(error="EMPTY_RESPONSE"),
        "CAT-07-T-003": _transcript(final_answer="I don't know."),  # wrong_tool -> 0.0, scored
    }
    res = scorer.score(items, CORPUS, injected=injected)
    assert res["cases_total"] == 2
    assert res["cases_scored"] == 1
    assert res["coverage"] == 0.5
    assert res["cat07_mean"] == 0.0  # the unscored case is excluded, the wrong_tool case is 0
    assert res["cat07_mean_of_scored"] == 0.0


# ── adversarial null-model guards (offline; exercises the real harness + corpus) ────
def test_null_never_call_scores_zero_on_the_full_corpus():
    res = scorer.score(ITEMS, CORPUS, null="never_call")
    assert res["cat07_mean"] == 0.0
    assert res["verdict_counts"].get("wrong_tool") == len(ITEMS["tests"])


def test_null_always_call_fixed_args_scores_at_most_one_case():
    res = scorer.score(ITEMS, CORPUS, null="always_call_fixed")
    # fixed args match at most CAT-07-T-003 (the one case whose target IS the fixed url/mode);
    # every other case must be bad_args, stale_snapshot_mismatch, or wrong_tool -- never pass.
    passing = [r for r in res["detail"] if r["verdict"] == "pass"]
    assert len(passing) <= 1
    assert res["cat07_mean"] <= (1.0 / len(ITEMS["tests"])) + 1e-9


# ── SPEC-003 corpus expansion: corpus integrity ─────────────────────────────────────
def test_corpus_grew_by_at_least_four_discriminating_cases():
    ids = {t["id"] for t in ITEMS["tests"]}
    assert len(ITEMS["tests"]) >= 12
    for new_id in ("CAT-07-T-006", "CAT-07-F-004", "CAT-07-T-007", "CAT-07-F-005"):
        assert new_id in ids


def test_new_r1_and_r2_corpus_files_document_source_basis_per_nfr001():
    # AC5: zero new cases lacking a documented source basis.
    near_dup = json.loads(
        (_HERE.parent / "test_data" / "cat07" / "transcripts" / "7FwDP17XPlk.json")
        .read_text(encoding="utf-8")
    )
    assert near_dup["source_note"]
    q4 = json.loads(
        (_HERE.parent / "test_data" / "cat07" / "financial" / "polcorp-20260213.json")
        .read_text(encoding="utf-8")
    )
    assert q4["source_note"]


# ── R1: near-duplicate video resolution (CAT-07-T-006) ──────────────────────────────
def test_near_duplicate_video_correct_resolution_is_pass():
    case = _case("CAT-07-T-006")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                     "mode": "metadata"},
         "raw_result": {"duration": 213}, "exact_match": True},
    ], final_answer="It is 213 seconds long."))
    assert row["verdict"] == "pass"


def test_near_duplicate_video_wrong_id_is_bad_args():
    case = _case("CAT-07-T-006")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=7FwDP17XPlk",
                                     "mode": "metadata"},
         "raw_result": {"duration": 214}, "exact_match": True},
    ], final_answer="It is 214 seconds long."))
    assert row["verdict"] == "bad_args"


def test_near_duplicate_video_correct_call_but_near_dup_duration_reported_is_hallucinated_content():
    case = _case("CAT-07-T-006")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                     "mode": "metadata"},
         "raw_result": {"duration": 213}, "exact_match": True},
    ], final_answer="It is 214 seconds long."))  # confused with the near-dup's duration
    assert row["verdict"] == "hallucinated_content"


# ── R2: near-miss financial report date (CAT-07-F-004) ──────────────────────────────
def test_near_miss_date_correct_request_is_pass():
    case = _case("CAT-07-F-004")
    result, key, exact = harness.execute_financial(
        {"company": "PolCorp SA", "report_date": "2026-02-13"}, CORPUS)
    assert exact is True
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "get_financial_report_snapshot",
         "args": {"company": "PolCorp SA", "report_date": "2026-02-13"},
         "raw_result": result, "exact_match": exact, "resolved_key": list(key)},
    ], final_answer="EPS was 1.46 PLN."))
    assert row["verdict"] == "pass"


def test_near_miss_date_guessed_nonexistent_date_is_stale_snapshot_mismatch():
    # A model that (wrongly) guesses the fiscal quarter-end instead of the actual publication
    # date -- this date has no corpus entry, so the harness's real fallback logic serves the
    # closest snapshot it actually holds (Q2, candidates[0]), not the requested Q4 report.
    case = _case("CAT-07-F-004")
    result, key, exact = harness.execute_financial(
        {"company": "PolCorp SA", "report_date": "2025-12-31"}, CORPUS)
    assert exact is False
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "get_financial_report_snapshot",
         "args": {"company": "PolCorp SA", "report_date": "2025-12-31"},
         "raw_result": result, "exact_match": exact, "resolved_key": list(key)},
    ], final_answer="EPS was 1.01 PLN."))
    assert row["verdict"] == "stale_snapshot_mismatch"


def test_near_miss_date_wrong_but_real_adjacent_date_is_hallucinated_content():
    # The model asks for an EXISTING but wrong quarter (Q3) -- a real exact_match, just for the
    # wrong report. The scorer catches this via the case's forbidden-content check, not
    # stale_snapshot_mismatch (that verdict is reserved for a nonexistent-date fallback).
    case = _case("CAT-07-F-004")
    result, key, exact = harness.execute_financial(
        {"company": "PolCorp SA", "report_date": "2025-11-15"}, CORPUS)
    assert exact is True
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "get_financial_report_snapshot",
         "args": {"company": "PolCorp SA", "report_date": "2025-11-15"},
         "raw_result": result, "exact_match": exact, "resolved_key": list(key)},
    ], final_answer="EPS was 1.24 PLN."))
    assert row["verdict"] == "hallucinated_content"


# ── R3: distractor tool (CAT-07-T-007) ───────────────────────────────────────────────
def test_distractor_tool_calling_search_youtube_instead_is_wrong_tool():
    case = _case("CAT-07-T-007")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "search_youtube", "args": {"query": "Prime-Agent AI harness"},
         "raw_result": {"results": [{"video_id": "8vUCjYsWeSU"}]}},
    ], final_answer="The uploader is Prompt Engineering."))
    assert row["verdict"] == "wrong_tool"


def test_distractor_tool_correct_yt_dlp_call_is_pass():
    case = _case("CAT-07-T-007")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=8vUCjYsWeSU",
                                     "mode": "metadata"},
         "raw_result": {"uploader": "Prompt Engineering"}, "exact_match": True},
    ], final_answer="The uploader/channel is Prompt Engineering."))
    assert row["verdict"] == "pass"


def test_search_youtube_dispatch_is_real_not_unknown_tool():
    # The distractor tool must have a real dispatch (not "unknown_tool") -- it needs to look like
    # a genuinely available tool for the wrong_tool choice to be a real trap, per R3.
    result, matched = harness.execute_search_youtube({"query": "Prime-Agent"}, CORPUS)
    assert "results" in result
    assert matched is True


# ── SPEC-003-004: tokenized/fuzzy match (fixes literal-substring gap from SPEC-003-001) ──────
def test_search_youtube_natural_language_query_matches_despite_punctuation_and_word_order():
    # SPEC-003-001's fleet run: a realistic phrasing reorders/adds words relative to the frozen
    # title and drops the exact parenthesization, so this string is NOT a contiguous substring of
    # "Rick Astley - Never Gonna Give You Up (Official Video) (4K Remaster)" -- the old literal
    # substring matcher returned zero hits on queries shaped like this one. A tokenized/fuzzy
    # matcher must still find it.
    result, matched = harness.execute_search_youtube(
        {"query": "Rick Astley Never Gonna Give You Up official video 4k"}, CORPUS)
    assert matched is True
    video_ids = {h["video_id"] for h in result["results"]}
    assert "dQw4w9WgXcQ" in video_ids  # the 4K remaster upload (exact title carries all query terms)


def test_search_youtube_casing_and_punctuation_differences_still_match():
    # Mixed casing and trailing punctuation the frozen title doesn't have.
    result, matched = harness.execute_search_youtube(
        {"query": "never GONNA give you up!!!"}, CORPUS)
    assert matched is True
    video_ids = {h["video_id"] for h in result["results"]}
    assert "dQw4w9WgXcQ" in video_ids


def test_search_youtube_matches_the_corpus_non_latin_title():
    # SPEC-003-009: dedicated regression test for SPEC-003-004's fix -- _SEARCH_TOKEN_RE must stay
    # \w+ (Unicode-aware), not regress to an ASCII-only [a-z0-9]+ pattern, which would silently stop
    # matching the frozen corpus's one non-Latin title ("PSY - GANGNAM STYLE(강남스타일) M/V",
    # video_id 9bZkp7q19f0). AC2: confirmed by hand that reverting _SEARCH_TOKEN_RE to
    # re.compile(r"[a-z0-9]+") locally makes this test fail -- \w+.findall() on the Korean query
    # "강남스타일" returns zero tokens under an ASCII-only pattern (query_tokens is empty, so
    # execute_search_youtube's `if query_tokens:` guard short-circuits to `hits=[]`), producing
    # `matched=False` and failing the assertion below. Reverted back immediately after confirming;
    # not committed.
    result, matched = harness.execute_search_youtube({"query": "강남스타일"}, CORPUS)
    assert matched is True
    video_ids = {h["video_id"] for h in result["results"]}
    assert "9bZkp7q19f0" in video_ids


def test_search_youtube_matches_the_non_latin_title_via_its_english_portion_too():
    # The English half of the same title must independently match -- not just the Korean tokens.
    result, matched = harness.execute_search_youtube({"query": "PSY Gangnam Style M/V"}, CORPUS)
    assert matched is True
    video_ids = {h["video_id"] for h in result["results"]}
    assert "9bZkp7q19f0" in video_ids


def test_search_youtube_unrelated_query_returns_zero_results():
    # R2/AC1 negative case: a genuine filter, not match-anything. None of these words appear in
    # any frozen video title (Rick Astley, Prime-Agent, "Me at the zoo", GANGNAM STYLE, Hermes).
    result, matched = harness.execute_search_youtube(
        {"query": "quarterly earnings report for a fictional semiconductor company"}, CORPUS)
    assert matched is False
    assert result["results"] == []


def test_search_youtube_does_not_become_match_everything():
    # A second, differently-unrelated negative case, to guard against a threshold so loose it
    # degenerates into "return every video regardless of query" -- must not match a video whose
    # title shares no meaningful tokens with the query.
    result, matched = harness.execute_search_youtube(
        {"query": "underwater basket weaving tutorial"}, CORPUS)
    assert matched is False
    video_ids = {h["video_id"] for h in result["results"]}
    assert "dQw4w9WgXcQ" not in video_ids
    assert "9bZkp7q19f0" not in video_ids  # GANGNAM STYLE


# ── R4: multi-hop retrieval (CAT-07-F-005) ───────────────────────────────────────────
def test_multi_hop_case_declares_extra_max_turns():
    case = _case("CAT-07-F-005")
    assert case.get("max_turns", 3) >= 5


def test_resolve_report_id_returns_identifier_without_figures():
    result, resolved = harness.execute_resolve_report_id({"company": "NovaTech SA"}, CORPUS)
    assert resolved is True
    assert result["report_date"] == "2026-02-20"
    assert "text" not in result and "fields" not in result  # no figures, R4: separate hop


def test_multi_hop_hop1_never_reaches_hop2_is_wrong_tool():
    case = _case("CAT-07-F-005")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "resolve_report_id", "args": {"company": "Totally Unknown Corp"},
         "raw_result": {"error": "not_found"}},
    ], final_answer="I could not find the report."))
    assert row["verdict"] == "wrong_tool"


def test_multi_hop_hop2_uses_guessed_date_instead_of_resolved_is_bad_args():
    # hop1 (resolve_report_id) succeeds correctly, but hop2 ignores the resolved date and guesses
    # a different one -- proves the scorer attributes the failure to hop2's args, not hop1. F-005's
    # expected_args_schema enum-restricts report_date (only one real NovaTech snapshot exists), so
    # an off-schema guess is caught at the schema layer (bad_args) before corpus resolution -- the
    # harness-level resolution itself is exact_match=False (verified below), matching F-004's
    # stale_snapshot_mismatch mechanism; F-005's stricter schema just surfaces it one layer earlier.
    case = _case("CAT-07-F-005")
    hop1 = harness.execute_resolve_report_id({"company": "NovaTech SA"}, CORPUS)[0]
    assert hop1["report_date"] == "2026-02-20"
    result, key, exact = harness.execute_financial(
        {"company": "NovaTech SA", "report_date": "2025-01-01"}, CORPUS)  # guessed, not resolved
    assert exact is False
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "resolve_report_id", "args": {"company": "NovaTech SA"}, "raw_result": hop1},
        {"name": "get_financial_report_snapshot",
         "args": {"company": "NovaTech SA", "report_date": "2025-01-01"},
         "raw_result": result, "exact_match": exact, "resolved_key": list(key)},
    ], final_answer="Net margin was 8.9%."))
    assert row["verdict"] == "bad_args"


# ── SPEC-003-001: implicit disambiguation (CAT-07-T-008) ────────────────────────────
def test_implicit_disambiguation_case_prompt_never_names_the_correct_video():
    # AC1: a human-reviewable regression guard -- the prompt text must not contain the literal
    # video_id, "official", or "reupload", the exact phrases T-006 uses to hand the model the
    # resolving detail directly. The correct answer must come from search_youtube + world
    # knowledge / recall, not from parsing the prompt.
    case = _case("CAT-07-T-008")
    prompt_text = " ".join(m["content"] for m in case["prompt"]["messages"]).lower()
    assert "dqw4w9wgxcq" not in prompt_text
    assert "official" not in prompt_text
    assert "reupload" not in prompt_text
    assert case.get("max_turns", 3) >= 5  # search hop + decisive yt_dlp hop + final answer


def test_implicit_disambiguation_search_then_correct_call_is_pass():
    # Gold path (a): search_youtube surfaces both candidates, model picks the famous one by world
    # knowledge, then makes exactly one decisive yt_dlp call on the correct video_id.
    case = _case("CAT-07-T-008")
    search_result, matched = harness.execute_search_youtube(
        {"query": "never gonna give you up"}, CORPUS)
    assert matched is True
    assert {"dQw4w9WgXcQ", "7FwDP17XPlk"} == {h["video_id"] for h in search_result["results"]}
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "search_youtube", "args": {"query": "never gonna give you up"},
         "raw_result": search_result},
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                     "mode": "metadata"},
         "raw_result": {"uploader": "Rick Astley"}, "exact_match": True},
    ], final_answer="The uploader/channel is Rick Astley."))
    assert row["verdict"] == "pass"


def test_implicit_disambiguation_direct_recall_without_search_is_also_pass():
    # Gold path (b): a model with strong world knowledge skips search_youtube entirely and calls
    # yt_dlp directly on the video_id it already recalls -- explicitly allowed by R1's "or from
    # world knowledge combined with the returned data" clause. The scorer does not require any
    # specific tool sequence, only that the DECISIVE call and final answer are correct.
    case = _case("CAT-07-T-008")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                     "mode": "metadata"},
         "raw_result": {"uploader": "Rick Astley"}, "exact_match": True},
    ], final_answer="Rick Astley."))
    assert row["verdict"] == "pass"


def test_implicit_disambiguation_wrong_candidate_first_is_bad_args_not_stale_snapshot():
    # Documents the _find_call() first-match dependency this case's design deliberately avoids
    # relying on: a model that explores the WRONG candidate (7FwDP17XPlk) before ever reaching the
    # correct one is scored on that first yt_dlp call alone, even if it never actually gets a
    # chance to self-correct in this hand-crafted transcript. execute_yt_dlp resolves ANY known
    # corpus video_id (resolved_key is not None), so exact_match is True on the wrong candidate too
    # -- the failure surfaces as bad_args (url must_contain fails), never stale_snapshot_mismatch
    # (that verdict is reserved for a same-company/different-key financial fallback, not a
    # different-video wrong pick). This is why the case's gold path (above) must reach the correct
    # video_id on its first yt_dlp call, not disambiguate via repeated yt_dlp calls.
    case = _case("CAT-07-T-008")
    result, video_id = harness.execute_yt_dlp(
        {"url": "https://www.youtube.com/watch?v=7FwDP17XPlk", "mode": "metadata"}, CORPUS)
    assert video_id == "7FwDP17XPlk"
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=7FwDP17XPlk",
                                     "mode": "metadata"},
         "raw_result": result, "exact_match": True},
    ], final_answer="The uploader/channel is Amazing Lyrics."))
    assert row["verdict"] == "bad_args"


def test_find_call_credits_the_first_name_match_not_a_later_valid_one():
    # SPEC-003-006 Decision (2026-09-16): _find_call()'s first-by-name-match
    # semantics are intentional, not a bug — this test locks that decision in.
    # A model that calls the right tool with BAD args first, then calls it
    # again with fully valid args, is still scored on the first (bad) call.
    # Crediting the later valid call would silently redefine "the decisive
    # call" for every existing and future CAT-07 case, including cases (like
    # CAT-07-T-008 above) whose gold path was deliberately authored around
    # the current behavior.
    case = {
        "id": "SYNTHETIC-find-call-regression",
        "task_type": "transcript_retrieval",
        "difficulty": "easy",
        "expected_tool": "some_tool",
        "expected_args_schema": {"key": {"type": "string", "must_contain": "right"}},
    }
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "some_tool", "args": {"key": "wrong-value"}, "raw_result": {}, "exact_match": False},
        {"name": "some_tool", "args": {"key": "right-value"}, "raw_result": {}, "exact_match": True},
    ], final_answer="whatever"))
    assert row["verdict"] == "bad_args"


def test_implicit_disambiguation_correct_call_but_wrong_uploader_named_is_hallucinated_content():
    case = _case("CAT-07-T-008")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                     "mode": "metadata"},
         "raw_result": {"uploader": "Rick Astley"}, "exact_match": True},
    ], final_answer="The uploader/channel is Amazing Lyrics."))
    assert row["verdict"] == "hallucinated_content"


# ── SPEC-003-005: implicit disambiguation with an OBSCURE near-duplicate pair (CAT-07-T-009) ──
def test_obscure_pair_case_prompt_never_names_the_correct_video_or_studio():
    # AC1/AC3 regression guard, mirroring CAT-07-T-008's own prompt-purity test: the prompt must
    # not name the video_id, "official", "Blender" (the disambiguating uploader field itself), or
    # the reupload channel's name -- only search_youtube's returned title text (or recall) can
    # supply that.
    case = _case("CAT-07-T-009")
    prompt_text = " ".join(m["content"] for m in case["prompt"]["messages"]).lower()
    assert "aqz-ke-bpkq" not in prompt_text
    assert "official" not in prompt_text
    assert "blender" not in prompt_text
    assert "animalsreview" not in prompt_text
    assert case.get("max_turns", 3) >= 5


def test_obscure_pair_search_youtube_surfaces_both_candidates_for_generic_queries():
    # AC5 (CORRECTED, see the case's own "notes" field): a query built from the two titles'
    # SHARED descriptor vocabulary (big/buck/bunny/short/film/4k) surfaces both candidates under
    # search_youtube's 0.6 query-token-recall threshold -- verified here across a range of query
    # lengths, including the verbatim official title text itself. Two of the five live fleet
    # transcripts hit exactly this branch (qwen/qwen3.6-27b: "Big Buck Bunny"; gemma-4-31b-it:
    # "Big Buck Bunny official studio") and both then disambiguated correctly by reading the
    # returned title text -- real AC2 evidence, not pure recall.
    queries = [
        "Big Buck Bunny",
        "Big Buck Bunny short film",
        "Big Buck Bunny 4K short film",
        "Big Buck Bunny official studio",
        "Big Buck Bunny Blender Foundation short film",
        "Big Buck Bunny 60fps 4K Blender Foundation official short film",
        "Big Buck Bunny 60fps 4K - Official Blender Foundation Short Film",
    ]
    for q in queries:
        result, matched = harness.execute_search_youtube({"query": q}, CORPUS)
        assert matched is True, q
        ids = {h["video_id"] for h in result["results"]}
        assert {"aqz-KE-bpKQ", "f7NwyBnIRTE"} == ids, (q, ids)


def test_obscure_pair_search_youtube_prunes_to_the_official_candidate_for_studio_specific_queries():
    # AC5 (CORRECTED): the addendum's OTHER accepted branch -- a query that leans on the
    # DISAMBIGUATING words themselves ("Blender", "Foundation", "official") without also repeating
    # a shared descriptor word prunes to the single correct candidate, because those words appear
    # only in aqz-KE-bpKQ's title. This is real, observed live in 2 of 5 fleet transcripts
    # (ornith-1.0-35b's first search; muse-glimmer-30b's 2nd/3rd searches). It is accepted tool
    # realism (SPEC-003-005 Problem addendum) precisely because in the live run it never
    # determined a model's PASS/FAIL outcome by itself -- both models had independently resolved
    # or already seen both candidates by the time pruning happened; see the run report.
    queries = [
        "Big Buck Bunny Blender official",  # boundary case: 3/5 = 0.6, still matches both
        "Big Buck Bunny Blender Foundation official",  # 3/6 = 0.5, prunes to one
        "Big Buck Bunny Blender Foundation 2008 original upload",
    ]
    expected = [
        {"aqz-KE-bpKQ", "f7NwyBnIRTE"},
        {"aqz-KE-bpKQ"},
        {"aqz-KE-bpKQ"},
    ]
    for q, exp in zip(queries, expected):
        result, matched = harness.execute_search_youtube({"query": q}, CORPUS)
        assert matched is True, q
        ids = {h["video_id"] for h in result["results"]}
        assert ids == exp, (q, ids, exp)


def test_obscure_pair_search_then_correct_call_is_pass():
    # Gold path: search_youtube surfaces both real candidates; the model reads the returned title
    # text (one names "Official Blender Foundation", the other does not) and makes exactly one
    # decisive yt_dlp call on the correct video_id.
    case = _case("CAT-07-T-009")
    search_result, matched = harness.execute_search_youtube(
        {"query": "Big Buck Bunny short film"}, CORPUS)
    assert matched is True
    assert {"aqz-KE-bpKQ", "f7NwyBnIRTE"} == {h["video_id"] for h in search_result["results"]}
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "search_youtube", "args": {"query": "Big Buck Bunny short film"},
         "raw_result": search_result},
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=aqz-KE-bpKQ",
                                     "mode": "metadata"},
         "raw_result": {"uploader": "Blender"}, "exact_match": True},
    ], final_answer="The uploader/channel is Blender."))
    assert row["verdict"] == "pass"


def test_obscure_pair_wrong_candidate_first_is_bad_args_not_stale_snapshot():
    # Documents the same _find_call() first-match dependency as CAT-07-T-008: a model that probes
    # the WRONG candidate (f7NwyBnIRTE) on its first yt_dlp call is scored on that call alone.
    case = _case("CAT-07-T-009")
    result, video_id = harness.execute_yt_dlp(
        {"url": "https://www.youtube.com/watch?v=f7NwyBnIRTE", "mode": "metadata"}, CORPUS)
    assert video_id == "f7NwyBnIRTE"
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=f7NwyBnIRTE",
                                     "mode": "metadata"},
         "raw_result": result, "exact_match": True},
    ], final_answer="The uploader/channel is AnimalsReview."))
    assert row["verdict"] == "bad_args"


def test_obscure_pair_correct_call_but_wrong_uploader_named_is_hallucinated_content():
    case = _case("CAT-07-T-009")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=aqz-KE-bpKQ",
                                     "mode": "metadata"},
         "raw_result": {"uploader": "Blender"}, "exact_match": True},
    ], final_answer="The uploader/channel is AnimalsReview."))
    assert row["verdict"] == "hallucinated_content"


def test_multi_hop_both_hops_correct_is_pass():
    case = _case("CAT-07-F-005")
    hop1 = harness.execute_resolve_report_id({"company": "NovaTech SA"}, CORPUS)[0]
    result, key, exact = harness.execute_financial(
        {"company": "NovaTech SA", "report_date": hop1["report_date"]}, CORPUS)
    assert exact is True
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "resolve_report_id", "args": {"company": "NovaTech SA"}, "raw_result": hop1},
        {"name": "get_financial_report_snapshot",
         "args": {"company": "NovaTech SA", "report_date": hop1["report_date"]},
         "raw_result": result, "exact_match": exact, "resolved_key": list(key)},
    ], final_answer="Net margin was 8.9%."))
    assert row["verdict"] == "pass"


# ── SPEC-003-002 R1/R2/R4: financial-side distractor tool (CAT-07-F-006) ────────────
def test_financial_distractor_case_reuses_existing_frozen_source_key():
    # AC2/R3: no new frozen corpus file -- reuses the SPEC-001 polcorp-20251115.json entry that
    # CAT-07-F-001 already reads from.
    case = _case("CAT-07-F-006")
    assert case["frozen_source_key"] == "polcorp-20251115"
    assert case["frozen_source_key"] == _case("CAT-07-F-001")["frozen_source_key"]


def test_get_stock_quote_dispatch_is_real_not_unknown_tool():
    # The distractor tool must have a real dispatch (not "unknown_tool") -- it needs to look like a
    # genuinely available tool for the wrong_tool choice to be a real trap, per R1.
    result, matched = harness.execute_get_stock_quote({"ticker": "PCO"}, CORPUS)
    assert matched is True
    assert result["ticker"] == "PCO"
    assert "text" not in result and "fields" not in result  # never report figures, just a quote


def test_get_stock_quote_unknown_ticker_is_not_found():
    result, matched = harness.execute_get_stock_quote({"ticker": "ZZZZ"}, CORPUS)
    assert matched is False
    assert result["error"] == "not_found"


def test_financial_distractor_calling_get_stock_quote_instead_is_wrong_tool():
    case = _case("CAT-07-F-006")
    quote, matched = harness.execute_get_stock_quote({"ticker": "PCO"}, CORPUS)
    assert matched is True
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "get_stock_quote", "args": {"ticker": "PCO"}, "raw_result": quote},
    ], final_answer="PCO is currently trading at 84.20 PLN."))
    assert row["verdict"] == "wrong_tool"
    assert row["case_score"] == 0.0


def test_financial_distractor_correct_get_financial_report_snapshot_call_is_pass():
    case = _case("CAT-07-F-006")
    result, key, exact = harness.execute_financial(
        {"company": "PolCorp SA", "report_date": "2025-11-15"}, CORPUS)
    assert exact is True
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "get_financial_report_snapshot",
         "args": {"company": "PolCorp SA", "report_date": "2025-11-15"},
         "raw_result": result, "exact_match": exact, "resolved_key": list(key)},
    ], final_answer="Net income was 38 mln PLN."))
    assert row["verdict"] == "pass"
    assert row["case_score"] == 1.0


def test_implicit_tool_choice_prompt_names_neither_tool():
    # SPEC-003-007 R1/AC1: the correct tool must be inferable only from the
    # system message's tool descriptions, never from the user prompt itself.
    case = _case("CAT-07-F-007")
    user_msg = next(m["content"] for m in case["prompt"]["messages"] if m["role"] == "user")
    assert "get_financial_report_snapshot" not in user_msg
    assert "get_stock_quote" not in user_msg


def test_implicit_tool_choice_calling_get_stock_quote_instead_is_wrong_tool():
    case = _case("CAT-07-F-007")
    quote, matched = harness.execute_get_stock_quote({"ticker": "NVT"}, CORPUS)
    assert matched is True
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "get_stock_quote", "args": {"ticker": "NVT"}, "raw_result": quote},
    ], final_answer="NVT is currently trading at 156.75 PLN."))
    assert row["verdict"] == "wrong_tool"
    assert row["case_score"] == 0.0


def test_implicit_tool_choice_correct_get_financial_report_snapshot_call_is_pass():
    case = _case("CAT-07-F-007")
    result, key, exact = harness.execute_financial(
        {"company": "NovaTech SA", "report_date": "2026-02-20"}, CORPUS)
    assert exact is True
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "get_financial_report_snapshot",
         "args": {"company": "NovaTech SA", "report_date": "2026-02-20"},
         "raw_result": result, "exact_match": exact, "resolved_key": list(key)},
    ], final_answer="Net income was 19 mln PLN."))
    assert row["verdict"] == "pass"
    assert row["case_score"] == 1.0


def test_verbatim_formatting_clean_extraction_is_pass():
    case = _case("CAT-07-F-008")
    result, key, exact = harness.execute_financial(
        {"company": "PCO", "report_date": "2025-11-15"}, CORPUS)
    assert exact is True
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "get_financial_report_snapshot",
         "args": {"company": "PCO", "report_date": "2025-11-15"},
         "raw_result": result, "exact_match": exact, "resolved_key": list(key)},
    ], final_answer="Net income was 38 mln PLN."))
    assert row["verdict"] == "pass"
    assert row["case_score"] == 1.0


def test_verbatim_formatting_bracket_copy_is_bad_args_with_distinct_reason():
    # SPEC-003-008 R2/AC1: schema-valid (must_contain_any still matches the substring "PCO"
    # inside "[PCO]") but fails at corpus resolution -- a DIFFERENT bad_args reason than a
    # schema-level failure, per score_cat07.py's existing two-stage bad_args logic.
    case = _case("CAT-07-F-008")
    args = {"company": "[PCO]", "report_date": "2025-11-15"}
    assert scorer._args_valid(args, case["expected_args_schema"]) is True  # schema passes
    result, key, exact = harness.execute_financial(args, CORPUS)
    assert result.get("error") == "not_found"  # but resolution fails
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "get_financial_report_snapshot", "args": args, "raw_result": result},
    ], final_answer="I could not find that report."))
    assert row["verdict"] == "bad_args"
    assert "resolved to no corpus entry" in row["reason"]


def test_verbatim_formatting_schema_level_bad_args_has_a_different_reason():
    # Contrast: an args dict that fails the SCHEMA itself (no "PolCorp"/"PCO" substring at all)
    # must be distinguishable from the resolution-failure case above.
    case = _case("CAT-07-F-008")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "get_financial_report_snapshot",
         "args": {"company": "Some Other Company", "report_date": "2025-11-15"},
         "raw_result": {}},
    ], final_answer="whatever"))
    assert row["verdict"] == "bad_args"
    assert "fail schema" in row["reason"]
    assert "resolved to no corpus entry" not in row["reason"]


# ── SPEC-003-010: contrastive-decoy tolerance in final_answer_must_not_contain ───────────────────

def test_contrastive_mention_of_decoy_now_passes():
    # AC4: the exact CAT-07-T-009 qwen3.8-27b scenario (SPEC-003-005 finding) -- correct decisive
    # yt_dlp call, correct required content ("Blender") stated, decoy channel ("AnimalsReview")
    # named only in an explicit contrastive aside. Previously scored hallucinated_content; must now
    # pass.
    case = _case("CAT-07-T-009")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=aqz-KE-bpKQ",
                                     "mode": "metadata"},
         "raw_result": {"uploader": "Blender"}, "exact_match": True},
    ], final_answer=(
        "The uploader/channel is Blender, unlike the AnimalsReview reupload which is a separate "
        "channel."
    )))
    assert row["verdict"] == "pass"
    assert row["case_score"] == 1.0


def test_asserting_the_decoy_as_the_answer_still_fails():
    # Contrast: the decoy named with NO negation/contrast marker and NO required content stated
    # (the pre-existing regression test above) must still fail -- unchanged behavior.
    case = _case("CAT-07-T-009")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=aqz-KE-bpKQ",
                                     "mode": "metadata"},
         "raw_result": {"uploader": "Blender"}, "exact_match": True},
    ], final_answer="The uploader/channel is AnimalsReview."))
    assert row["verdict"] == "hallucinated_content"


def test_decoy_named_without_contrast_marker_still_fails_even_with_required_content():
    # A model that states BOTH names flatly, with no contrast/negation cue at all, has not
    # demonstrated disambiguation -- must still fail. Isolates the forbidden-check specifically
    # (unlike the case above, required content ("Blender") IS present here).
    case = _case("CAT-07-T-009")
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=aqz-KE-bpKQ",
                                     "mode": "metadata"},
         "raw_result": {"uploader": "Blender"}, "exact_match": True},
    ], final_answer="The uploader/channel is Blender. AnimalsReview also has an upload of this film."))
    assert row["verdict"] == "hallucinated_content"


def test_forbidden_asserted_helper_directly():
    assert scorer._forbidden_asserted("the answer is animalsreview.", "animalsreview") is True
    assert scorer._forbidden_asserted(
        "the answer is blender, not animalsreview.", "animalsreview") is False
    assert scorer._forbidden_asserted(
        "unlike animalsreview, this is blender.", "animalsreview") is False
    assert scorer._forbidden_asserted(
        "blender is correct. animalsreview also exists.", "animalsreview") is True


# ── SPEC-003-012/SPEC-003-011 Decision: resolved_but_uncommitted verdict ──────────────────────────

def test_resolved_but_uncommitted_when_decisive_call_correct_but_error_and_no_answer():
    case = _case("CAT-07-T-008")
    transcript = _transcript(
        tool_calls_made=[
            {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                         "mode": "metadata"},
             "raw_result": {"uploader": "Rick Astley"}, "exact_match": True},
            {"name": "search_youtube", "args": {"query": "redundant re-check"},
             "raw_result": {"results": []}, "exact_match": None},
        ],
        final_answer="",
        error="max_turns_exhausted_without_final_answer",
    )
    row = scorer.score_case(case, transcript)
    assert row["verdict"] == "resolved_but_uncommitted"
    assert row["case_score"] is None  # never 0-credited, same convention as UNSCORED


def test_generic_unscored_when_error_but_no_decisive_call_ever_made():
    case = _case("CAT-07-T-008")
    transcript = _transcript(tool_calls_made=[], final_answer="",
                              error="max_turns_exhausted_without_final_answer")
    row = scorer.score_case(case, transcript)
    assert row["verdict"] == "UNSCORED"


def test_generic_unscored_when_error_and_only_call_was_wrong():
    # SPEC-003-006's first-call-wins semantics apply here too: a malformed FIRST call followed by a
    # correct SECOND call still resolves UNSCORED, not resolved_but_uncommitted -- _find_call()
    # never sees the later, correct call. Matches the real qwen3.8-27b/CAT-07-F-006 historical
    # transcript's re-scored verdict exactly (see SPEC-003-012 report).
    case = _case("CAT-07-F-006")
    transcript = _transcript(
        tool_calls_made=[
            {"name": "get_financial_report_snapshot", "args": {"company": "PolCorp SA (PCO)",
                                                                 "report_date": "2025-11-15"},
             "raw_result": {"error": "not_found"}},
            {"name": "resolve_report_id", "args": {"company": "PolCorp SA"},
             "raw_result": {"report_date": "2025-11-15"}, "resolved": True},
            {"name": "get_financial_report_snapshot", "args": {"company": "PCO",
                                                                 "report_date": "2025-11-15"},
             "raw_result": {"text": "..."}, "exact_match": True},
        ],
        final_answer="",
        error="max_turns_exhausted_without_final_answer",
    )
    row = scorer.score_case(case, transcript)
    assert row["verdict"] == "UNSCORED"


def test_resolved_but_uncommitted_re_scores_the_real_historical_transcripts():
    # Re-score the actual stored raw07/ transcripts from the 3 SPEC-003-011 occurrences this new
    # verdict was designed for -- not synthetic fixtures.
    fixtures = [
        ("results/2026-08-21_muse-glimmer-30b_cat07-t008-implicit/raw07/CAT-07-T-008.json", "CAT-07-T-008"),
        ("results/2026-08-21_muse-glimmer-30b_cat07-t009-obscure-pair/raw07/CAT-07-T-009.json", "CAT-07-T-009"),
        ("results/2026-08-21_ornith-1.0-35b_cat07-t009-obscure-pair/raw07/CAT-07-T-009.json", "CAT-07-T-009"),
    ]
    for path, case_id in fixtures:
        p = _HERE.parent / path
        assert p.is_file(), f"required regression fixture missing: {path}"
        transcript = json.loads(p.read_text())
        case = _case(case_id)
        row = scorer.score_case(case, transcript)
        assert row["verdict"] == "resolved_but_uncommitted", f"{path}: {row['verdict']}"

    # The compounding-mistake subtype stays UNSCORED -- confirmed against its own real transcript.
    f006 = _HERE.parent / "results/2026-08-21_qwen3.8-27b_cat07-f006-distractor/raw07/CAT-07-F-006.json"
    assert f006.is_file(), "required compounding-mistake regression fixture missing"
    transcript = json.loads(f006.read_text())
    row = scorer.score_case(_case("CAT-07-F-006"), transcript)
    assert row["verdict"] == "UNSCORED"


# ── CAT-07-T-010: commitment-discipline isolation case ───────────────────────────────────────────

def test_commitment_discipline_case_declares_generous_max_turns():
    case = _case("CAT-07-T-010")
    assert case["max_turns"] == 10  # double CAT-07-T-008/T-009's 5-turn precedent (R2)


def test_commitment_discipline_case_reuses_existing_frozen_resource():
    # AC5/R3: no new corpus content -- dQw4w9WgXcQ is already frozen (CAT-07-T-002/003/006/008).
    case = _case("CAT-07-T-010")
    assert case["frozen_source_key"] == "dQw4w9WgXcQ"
    assert case["frozen_source_key"] in CORPUS["transcripts"]


def test_commitment_discipline_case_quick_resolution_and_answer_is_pass():
    case = _case("CAT-07-T-010")
    result, video_id = harness.execute_yt_dlp(
        {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "mode": "metadata"}, CORPUS)
    assert video_id == "dQw4w9WgXcQ"
    row = scorer.score_case(case, _transcript(tool_calls_made=[
        {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                     "mode": "metadata"},
         "raw_result": result, "exact_match": True},
    ], final_answer="The upload date is 2009-10-25."))
    assert row["verdict"] == "pass"


def test_commitment_discipline_case_resolved_then_looping_is_resolved_but_uncommitted():
    case = _case("CAT-07-T-010")
    result, video_id = harness.execute_yt_dlp(
        {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ", "mode": "metadata"}, CORPUS)
    row = scorer.score_case(case, _transcript(
        tool_calls_made=[
            {"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                         "mode": "metadata"},
             "raw_result": result, "exact_match": True},
        ] + [{"name": "yt_dlp", "args": {"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                          "mode": "metadata"},
              "raw_result": result, "exact_match": True}] * 8,  # 8 redundant re-checks, budget used
        final_answer="",
        error="max_turns_exhausted_without_final_answer",
    ))
    assert row["verdict"] == "resolved_but_uncommitted"
