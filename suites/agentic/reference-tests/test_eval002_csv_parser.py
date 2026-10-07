"""Reference suite for EVAL-002 — one test per acceptance criterion.

Authored from eval-specs/EVAL-002-csv-parser.md. The model under test never sees
this file. Contract: parse_csv(text: str) -> list[list[str]], RFC 4180, no stdlib csv.
"""

import pytest

parse_csv = pytest.importorskip("csv_parser").parse_csv


def test_ac1_simple_row():
    assert parse_csv("a,b,c") == [["a", "b", "c"]]


def test_ac2_multiple_rows():
    assert parse_csv("a,b\nc,d") == [["a", "b"], ["c", "d"]]


def test_ac3_quoted_field_with_comma():
    assert parse_csv('"hello, world",b') == [["hello, world", "b"]]


def test_ac4_escaped_doubled_quote():
    assert parse_csv('"he said ""hi""",b') == [['he said "hi"', "b"]]


def test_ac5_embedded_newline_in_quoted_field():
    assert parse_csv('"line1\nline2",b') == [["line1\nline2", "b"]]


def test_ac6_crlf_line_endings():
    assert parse_csv("a,b\r\nc,d") == [["a", "b"], ["c", "d"]]


def test_ac7_empty_fields():
    assert parse_csv(",,") == [["", "", ""]]


def test_ac8_single_field():
    assert parse_csv("hello") == [["hello"]]


def test_ac9_empty_input():
    assert parse_csv("") == []


def test_ac10_trailing_newline_yields_no_empty_row():
    assert parse_csv("a,b\n") == [["a", "b"]]


def test_ac11_unclosed_quote_raises_valueerror():
    with pytest.raises(ValueError):
        parse_csv('"unterminated,b')


def test_ac12_mixed_quoted_and_unquoted_in_same_row():
    assert parse_csv('a,"b,c",d') == [["a", "b,c", "d"]]
