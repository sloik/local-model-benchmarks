#!/usr/bin/env python3
"""CAT-07 · Tool Use / Agentic Retrieval -- tool-call harness shim.

Spec: SPEC.md (R2, AC2)

WHAT THIS DOES
    Drives a real multi-turn OpenAI-style tool-calling conversation against a subject model
    served by LM Studio: sends `tools=[...]`, records exactly what the model asked for (name,
    args, timestamp), executes the tool call against a FROZEN local corpus (never a live network
    fetch at scoring time -- see WHY HERMETIC below), feeds the tool's raw result back to the
    model as a `role: tool` message, and captures the model's final answer. The full call/result
    transcript is returned so `score_cat07.py` can verify tool selection, argument validity, and
    whether the final answer reflects what the tool actually returned -- without needing the
    model's own self-report to be trustworthy (AC2).

WHY HERMETIC (frozen corpus, not a live yt-dlp/network call, at scoring time)
    A live YouTube fetch drifts -- auto-captions regenerate, formats change, rate limits hit,
    network fails -- so a gold fingerprint captured today would stop matching tomorrow and AC4
    ("re-running the same case twice against the same tool output produces the same score") would
    break. The frozen corpus under test_data/cat07/ was built ONCE, offline, with the real yt-dlp
    CLI (see test_data/cat07/transcripts/*.json `captured` field for provenance) and is served
    back verbatim here. This mirrors CAT-05/CAT-06's own precedent: `--run-dir` / injected-fixture
    replay is the reproducible path; live calls are for building corpora, not for scoring them.

TOOL SURFACE (mirrors the real MCP tool, not a harness invention)
    `yt_dlp(url, mode, language, output_dir)` mirrors the live `mcp__shipyard__lmac-run__yt_dlp`
    tool's declared parameters exactly (see commands.yaml in lmac-run-mcp) so the benchmark
    exercises the same tool surface a real agentic session would use, per SPEC-001's Context
    section. `get_financial_report_snapshot(company, report_date)` has no existing MCP/CLI
    precedent in this repo (financial-report retrieval is not yet a wired tool) -- it is a
    harness-local synthetic tool over a frozen, fictional-company corpus (SPEC-001 R4/out-of-scope:
    no live financial-data API). `get_stock_quote(ticker)` (SPEC-003-002 R1) is the financial-side
    distractor tool, mirroring SPEC-003 R3's `search_youtube` pattern: a real, dispatched tool that
    is thematically tempting for a financial-domain question but wrong for a report-retrieval task
    -- it returns a live-style price quote, never report figures, over a small harness-local
    synthetic table (no live market-data API, same out-of-scope boundary as R4).

Usage (library):
    from cat07_harness import load_corpus, run_case
    corpus = load_corpus()
    transcript = run_case(case, corpus, model="qwen/qwen3.6-27b")
"""
import json
import os
import re
import time
import urllib.request
from pathlib import Path

LM_URL = "http://127.0.0.1:1234/v1/chat/completions"
HERE = Path(__file__).resolve().parent
CORPUS_DIR = HERE.parent / "test_data" / "cat07"

YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_youtube",
            "description": (
                "Search a local video index by keywords/title fragments and return approximate "
                "matches (title, video_id). Results are approximate -- for exact title, duration, "
                "uploader, or transcript text of a KNOWN url, call yt_dlp directly instead; this "
                "search index is not authoritative for exact metadata."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_report_id",
            "description": (
                "Resolve a company name to its most recent report identifier/date on file. Does "
                "NOT return report content or figures -- call get_financial_report_snapshot "
                "afterward with the resolved report_date to fetch the actual snapshot."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "Company name or ticker"},
                },
                "required": ["company"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "yt_dlp",
            "description": (
                "Acquire media on the Mac with yt-dlp. Modes: metadata returns JSON without "
                "downloading; transcript returns a clean English YouTube transcript; audio writes "
                "one M4A file for later transcription. Public or user-authorized URLs only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Public or user-authorized media URL"},
                    "mode": {
                        "type": "string",
                        "enum": ["metadata", "transcript", "audio"],
                        "description": "Operation: metadata, transcript, or audio",
                    },
                    "language": {
                        "type": "string",
                        "description": "Transcript language; currently only en is supported",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Mac-local destination directory used by audio mode",
                    },
                },
                "required": ["url", "mode"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_financial_report_snapshot",
            "description": (
                "Fetch a specific, dated financial report snapshot for a company. Returns the "
                "report text for the exact report_date requested, or the closest snapshot this "
                "source actually holds for that company if the exact date is not available."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "Company name or ticker"},
                    "report_date": {
                        "type": "string",
                        "description": "ISO date (YYYY-MM-DD) of the specific report requested",
                    },
                },
                "required": ["company", "report_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_quote",
            "description": (
                "Fetch a live-style price quote for a ticker -- current price, currency, and "
                "as-of timestamp. Does NOT return report figures (revenue, net income, EPS, "
                "margins) -- for a specific dated report's content, call "
                "get_financial_report_snapshot instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker symbol"},
                },
                "required": ["ticker"],
            },
        },
    },
]


def _normalize_company(s):
    s = (s or "").strip().lower()
    for suffix in (" sa", " s.a.", " s.a", " inc", " inc."):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s.strip()


def load_corpus(corpus_dir=CORPUS_DIR):
    """Load the frozen CAT-07 evidence into lookup dicts.

    Returns {"transcripts": {video_id: record}, "financial": {(company_norm, date): record},
    "financial_by_company": {company_norm: [record, ...]},
    "financial_ticker_to_company": {ticker_lower: company_norm}}.
    """
    transcripts = {}
    tdir = corpus_dir / "transcripts"
    if tdir.is_dir():
        for f in sorted(tdir.glob("*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            transcripts[rec["video_id"]] = rec

    financial = {}
    financial_by_company = {}
    financial_ticker_to_company = {}
    fdir = corpus_dir / "financial"
    if fdir.is_dir():
        for f in sorted(fdir.glob("*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            key = (_normalize_company(rec["company"]), rec["report_date"])
            financial[key] = rec
            financial_by_company.setdefault(key[0], []).append(rec)
            # SPEC-003-002: every frozen financial record already carries a `ticker` field.
            # expected_args_schema across CAT-07-F-001..006 accepts either the company name or
            # the ticker at the schema layer (must_contain_any: [company, TICKER]) -- index the
            # ticker here too so a schema-valid ticker-only company arg actually resolves in the
            # corpus, instead of a schema-legal identifier silently falling through to bad_args.
            ticker = (rec.get("ticker") or "").strip().lower()
            if ticker:
                financial_ticker_to_company.setdefault(ticker, key[0])

    return {
        "transcripts": transcripts,
        "financial": financial,
        "financial_by_company": financial_by_company,
        "financial_ticker_to_company": financial_ticker_to_company,
    }


def execute_yt_dlp(args, corpus):
    """Hermetic execution of a yt_dlp tool call against the frozen transcript corpus.

    Returns (raw_result: dict, resolved_video_id: str|None).
    """
    url = args.get("url") or ""
    mode = args.get("mode") or ""
    m = YOUTUBE_ID_RE.search(url)
    video_id = m.group(1) if m else None
    if video_id is None or video_id not in corpus["transcripts"]:
        return {"error": "not_found", "detail": "no frozen corpus entry for url=%r" % url}, None

    rec = corpus["transcripts"][video_id]
    if mode == "metadata":
        return {
            "id": rec["video_id"], "title": rec["title"], "uploader": rec["uploader"],
            "duration": rec["duration_s"], "upload_date": rec["upload_date"],
        }, video_id
    if mode == "transcript":
        if rec.get("transcript_excerpt") is None:
            return {"error": "no_transcript_available_for_this_source"}, video_id
        return {"transcript": rec["transcript_excerpt"]}, video_id
    if mode == "audio":
        return {"error": "audio_mode_not_supported_in_this_harness"}, video_id
    return {"error": "invalid_mode", "detail": mode}, video_id


def execute_financial(args, corpus):
    """Hermetic execution of a get_financial_report_snapshot call against the frozen corpus.

    Returns (raw_result: dict, resolved_key: (company_norm, date)|None, exact_match: bool).
    exact_match distinguishes a correct-date fetch from a same-company-wrong-date fetch (the
    stale_snapshot_mismatch trap, R3d) from a company the corpus has never heard of (bad_args).
    """
    company_norm = _normalize_company(args.get("company"))
    if company_norm not in corpus["financial_by_company"]:
        # SPEC-003-002: resolve a schema-valid ticker-only company arg (e.g. "PCO") to the same
        # company the full name would resolve to -- see load_corpus's financial_ticker_to_company.
        ticker_company = corpus.get("financial_ticker_to_company", {}).get(company_norm)
        if ticker_company:
            company_norm = ticker_company
    date = (args.get("report_date") or "").strip()
    key = (company_norm, date)
    if key in corpus["financial"]:
        rec = corpus["financial"][key]
        return {
            "company": rec["company"], "report_date": rec["report_date"],
            "period_label": rec["period_label"], "text": rec["full_text"],
        }, key, True

    candidates = corpus["financial_by_company"].get(company_norm)
    if candidates:
        # Company is real to this source, but not that exact date -- serve the closest snapshot
        # this source actually holds (mirrors a real retrieval API returning the nearest filing),
        # not an invented value. This is the mechanism that produces stale_snapshot_mismatch.
        rec = candidates[0]
        resolved_key = (company_norm, rec["report_date"])
        return {
            "company": rec["company"], "report_date": rec["report_date"],
            "period_label": rec["period_label"], "text": rec["full_text"],
        }, resolved_key, False

    return {"error": "not_found", "detail": "no frozen corpus entry for company=%r" % args.get("company")}, None, False


_SEARCH_TOKEN_RE = re.compile(r"\w+", re.UNICODE)  # not [a-z0-9]+: the frozen corpus has a
# non-Latin title ("... GANGNAM STYLE(강남스타일) M/V") and the OLD literal-substring matcher could
# match it verbatim -- an ASCII-only token pattern would silently narrow (not fix) that case.
# Stopwords excluded from the match ratio so a query that is ENTIRELY common words (e.g. "the")
# cannot spuriously satisfy the overlap threshold against every title that also contains "the".
_SEARCH_STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "for", "with", "by", "is", "are",
}
# SPEC-003-004 R1: fraction of the query's meaningful tokens that must appear among the title's
# tokens for a hit. Recall-oriented (denominator is the QUERY's token count, not the title's) --
# a short, specific query fully contained in a longer real title should match even though the
# title carries extra words (uploader tag, "(Official Video)", resolution suffix, etc.) that the
# query never mentioned. R2 requires this still be a genuine filter: an unrelated query shares no
# tokens with an unrelated title and scores 0.0, well under threshold.
_SEARCH_MATCH_THRESHOLD = 0.6


def _search_tokenize(text):
    return set(_SEARCH_TOKEN_RE.findall((text or "").lower())) - _SEARCH_STOPWORDS


def execute_search_youtube(args, corpus):
    """SPEC-003 R3 -- distractor tool. Hermetic, approximate title search over the frozen
    transcript corpus. Deliberately NOT authoritative for exact metadata (see TOOLS description) --
    a model that uses this instead of yt_dlp on a known url is exercising a real wrong_tool choice,
    not a no-tool-available baseline. Returns (raw_result: dict, matched: bool).

    SPEC-003-004: matching is tokenized/fuzzy (word-set overlap, case- and punctuation-insensitive),
    not a literal contiguous substring -- see module docstring's TOOLS description, which already
    promises "approximate matches". A natural-language query like "Rick Astley Never Gonna Give You
    Up official video 4k" must match the frozen title "Rick Astley - Never Gonna Give You Up
    (Official Video) (4K Remaster)" even though it is reordered and missing the exact
    parenthesization; an unrelated query must still return zero hits (R2 -- genuine filter, not
    match-anything).
    """
    query_tokens = _search_tokenize(args.get("query"))
    hits = []
    if query_tokens:
        for rec in corpus["transcripts"].values():
            title_tokens = _search_tokenize(rec["title"])
            overlap = query_tokens & title_tokens
            if not overlap:
                continue
            if len(overlap) / len(query_tokens) >= _SEARCH_MATCH_THRESHOLD:
                hits.append({"video_id": rec["video_id"], "title": rec["title"]})
    return {"results": hits, "note": "approximate index -- use yt_dlp for exact metadata/transcript"}, bool(hits)


def execute_resolve_report_id(args, corpus):
    """SPEC-003 R4 -- multi-hop lookup tool. Resolves a company name to the MOST RECENT report on
    file (by report_date, descending) without returning any report content/figures -- the model
    must make a second call (get_financial_report_snapshot) with the resolved report_date to get
    the actual snapshot. Returns (raw_result: dict, resolved: bool).
    """
    company_norm = _normalize_company(args.get("company"))
    candidates = corpus["financial_by_company"].get(company_norm)
    if not candidates:
        return {"error": "not_found", "detail": "no frozen corpus entry for company=%r" % args.get("company")}, False
    rec = sorted(candidates, key=lambda r: r["report_date"], reverse=True)[0]
    return {
        "company": rec["company"], "report_id": "%s-%s" % (company_norm, rec["report_date"].replace("-", "")),
        "report_date": rec["report_date"], "period_label": rec["period_label"],
    }, True


# SPEC-003-002 R1 -- small, harness-local synthetic quote table (NOT the frozen financial corpus).
# Deliberately a separate, tiny, hardcoded dict rather than a new test_data/cat07/ file: this is a
# distractor tool's data, never the gold source for any case (AC2 -- no new frozen corpus file
# required), and its values (price/currency) never overlap with any report figure a case checks
# for, so a model that wrongly calls this tool can never accidentally satisfy a case's
# final_answer_must_contain_any check.
_STOCK_QUOTES = {
    "pco": {"ticker": "PCO", "company": "PolCorp SA", "price": 84.20, "currency": "PLN"},
    "nvt": {"ticker": "NVT", "company": "NovaTech SA", "price": 156.75, "currency": "PLN"},
}


def execute_get_stock_quote(args, corpus):
    """SPEC-003-002 R1 -- distractor tool. Hermetic, synthetic live-style price quote, NOT report
    figures. Deliberately NOT authoritative for report content (see TOOLS description) -- a model
    that calls this instead of get_financial_report_snapshot is exercising a real wrong_tool
    choice, not a no-tool-available baseline. Returns (raw_result: dict, matched: bool).
    """
    ticker = (args.get("ticker") or "").strip().lower()
    quote = _STOCK_QUOTES.get(ticker)
    if quote is None:
        return {"error": "not_found", "detail": "no quote for ticker=%r" % args.get("ticker")}, False
    return {
        "ticker": quote["ticker"], "price": quote["price"], "currency": quote["currency"],
        "as_of": "synthetic/fictional, harness-local -- no live market-data API (SPEC-003-002)",
        "note": "live-style price quote only -- does not include report figures",
    }, True


def _execute_tool(name, args, corpus):
    """Dispatch + timestamp a single tool call. Returns a recorded-call dict (R2/AC2 shape)."""
    ts = time.time()
    if name == "yt_dlp":
        raw_result, resolved_key = execute_yt_dlp(args, corpus)
        exact = resolved_key is not None
    elif name == "get_financial_report_snapshot":
        raw_result, resolved_key, exact = execute_financial(args, corpus)
    elif name == "search_youtube":
        raw_result, matched = execute_search_youtube(args, corpus)
        resolved_key, exact = None, matched
    elif name == "resolve_report_id":
        raw_result, resolved = execute_resolve_report_id(args, corpus)
        resolved_key, exact = None, resolved
    elif name == "get_stock_quote":
        raw_result, matched = execute_get_stock_quote(args, corpus)
        resolved_key, exact = None, matched
    else:
        raw_result, resolved_key, exact = {"error": "unknown_tool", "detail": name}, None, False
    return {
        "name": name, "args": args, "raw_result": raw_result,
        "resolved_key": list(resolved_key) if isinstance(resolved_key, tuple) else resolved_key,
        "exact_match": exact, "timestamp": ts,
    }


# ── model calling ──────────────────────────────────────────────────────────────
# max_tokens=8192, not 32768: matches score_cat05/score_cat06 precedent. A <think> model spends
# budget on reasoning FIRST (recorded benchmark reliability finding) -- EMPTY_ALL_REASONING must not be scored as "chose not
# to call the tool"; it is a harness/budget failure and is reported separately from wrong_tool.
def _call_model(model, messages, tools, api_key, max_tokens=8192, timeout=180):
    body = json.dumps({
        "model": model, "messages": messages, "tools": tools, "tool_choice": "auto",
        "max_tokens": max_tokens, "temperature": 0, "stream": False,
    }).encode("utf-8")
    hdr = {"Content-Type": "application/json"}
    if api_key:
        hdr["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(LM_URL, body, hdr)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    msg = payload["choices"][0]["message"]
    usage = payload.get("usage", {})
    reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    content = (msg.get("content") or "").strip()
    tool_calls = msg.get("tool_calls") or []
    if not content and not tool_calls and reasoning:
        return None, "EMPTY_ALL_REASONING(reasoning_tokens=%d)" % reasoning
    return {"content": content, "tool_calls": tool_calls}, None


def _null_turn(null, case, turn_index):
    """Simulate a null model's assistant message. Adversarial guards (no ACs require these; the
    CAT-05/CAT-06 scorers both treat their absence as a real bug they had to fix -- see their
    module docstrings). Runs entirely offline, no LM Studio call.
    """
    if turn_index > 0:
        return {"content": "Here is the answer based on the tool result.", "tool_calls": []}
    if null == "never_call":
        return {"content": "I recall the answer without needing to look it up.", "tool_calls": []}
    if null == "always_call_fixed":
        # Fixed, hardcoded args regardless of the case -- correct for at most one case in the
        # whole corpus, bad_args/stale_snapshot_mismatch for every other case.
        return {
            "content": "",
            "tool_calls": [{
                "id": "null-fixed-1", "type": "function",
                "function": {
                    "name": "yt_dlp",
                    "arguments": json.dumps({"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                                              "mode": "metadata"}),
                },
            }],
        }
    raise ValueError("unknown null kind: %r" % null)


def run_case(case, corpus, model=None, null=None, api_key=None, max_tokens=8192, max_turns=3):
    """Run one CAT-07 case end to end. Returns the recorded transcript (R2/AC2).

    {case_id, no_call, tool_calls_made: [...], final_answer, turns_used, error}
    Exactly one of `model` (live LM Studio) or `null` (offline adversarial guard) must be given.
    """
    if (model is None) == (null is None):
        raise ValueError("run_case needs exactly one of model= or null=")
    api_key = api_key if api_key is not None else os.environ.get("LM_API_KEY", "")

    messages = [dict(m) for m in case["prompt"]["messages"]]
    tool_calls_made = []
    final_answer = None
    error = None

    for turn in range(max_turns):
        if null is not None:
            turn_result, err = _null_turn(null, case, turn), None
        else:
            turn_result, err = _call_model(model, messages, TOOLS, api_key, max_tokens=max_tokens)
        if err:
            error = err
            break

        tool_calls = turn_result["tool_calls"]
        if not tool_calls:
            final_answer = turn_result["content"]
            break

        messages.append({"role": "assistant", "content": turn_result["content"] or None,
                          "tool_calls": tool_calls})
        for tc in tool_calls:
            fn = tc["function"]
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            recorded = _execute_tool(fn["name"], args, corpus)
            recorded["tool_call_id"] = tc.get("id")
            tool_calls_made.append(recorded)
            messages.append({
                "role": "tool", "tool_call_id": tc.get("id"), "name": fn["name"],
                "content": json.dumps(recorded["raw_result"], ensure_ascii=False),
            })
        # loop again so the model can read the tool result(s) and produce a final answer
    else:
        error = error or "max_turns_exhausted_without_final_answer"

    return {
        "case_id": case["id"],
        "subject": null and ("null:" + null) or model,
        "no_call": len(tool_calls_made) == 0,
        "tool_calls_made": tool_calls_made,
        "final_answer": final_answer,
        "error": error,
    }
