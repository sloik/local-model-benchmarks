---
id: EVAL-002
priority: 2
layer: 1
type: eval
status: ready
after: []
prior_attempts: []
created: 2026-03-16
---

# RFC 4180 CSV Parser

## Problem

The project needs a lightweight CSV parser that handles the full RFC 4180 specification — including quoted fields, embedded commas, embedded newlines, and escaped quotes. This tests the agent's ability to handle edge cases systematically and write thorough tests from detailed acceptance criteria.

## Requirements

- [ ] Implement `parse_csv(text: str) -> list[list[str]]` in a single module
- [ ] Handle RFC 4180 rules: comma-separated, CRLF or LF line endings, optional quoting, doubled-quote escaping
- [ ] No external dependencies (no `csv` stdlib, no pandas) — implement from scratch
- [ ] Raise `ValueError` with a descriptive message on malformed input (unclosed quotes)

## Acceptance Criteria

- [ ] AC 1: Simple row — `"a,b,c"` → `[["a", "b", "c"]]`
- [ ] AC 2: Multiple rows — `"a,b\nc,d"` → `[["a", "b"], ["c", "d"]]`
- [ ] AC 3: Quoted field — `'"hello, world",b'` → `[["hello, world", "b"]]`
- [ ] AC 4: Escaped quote — `'"he said ""hi""",b'` → `[["he said \"hi\"", "b"]]`
- [ ] AC 5: Embedded newline in quoted field — `'"line1\nline2",b'` → `[["line1\nline2", "b"]]`
- [ ] AC 6: CRLF line endings — `"a,b\r\nc,d"` → `[["a", "b"], ["c", "d"]]`
- [ ] AC 7: Empty fields — `",,"` → `[["", "", ""]]`
- [ ] AC 8: Single field — `"hello"` → `[["hello"]]`
- [ ] AC 9: Empty input — `""` → `[]`
- [ ] AC 10: Trailing newline — `"a,b\n"` → `[["a", "b"]]` (no empty trailing row)
- [ ] AC 11: Unclosed quote raises `ValueError`
- [ ] AC 12: Mixed quoted and unquoted fields in same row

## Context

- Target: `src/csv_parser.py`
- Tests: `tests/test_csv_parser.py`
- Framework: pytest
- Constraint: Must NOT use Python's `csv` module — the point is to implement parsing logic
- RFC 4180: https://tools.ietf.org/html/rfc4180

## Out of Scope

- Writing/serializing CSV (read only)
- Streaming/large file support
- Header detection or dict output
- Custom delimiters

## Scoring Notes

This spec tests **thorough AC coverage**. Key scoring dimensions:
- Does every AC have at least one dedicated test?
- Are edge cases handled correctly (not just happy path)?
- Expected review cycles: 1-2 (quoting edge cases may need iteration)
- Duration: <30 minutes
