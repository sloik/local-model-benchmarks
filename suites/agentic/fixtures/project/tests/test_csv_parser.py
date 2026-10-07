import pytest
from src.csv_parser import parse_csv

def test_ac1_simple_row():
    # AC 1: Simple row — "a,b,c" → [["a", "b", "c"]]
    assert parse_csv("a,b,c") == [["a", "b", "c"]]

def test_ac2_multiple_rows():
    # AC 2: Multiple rows — "a,b\nc,d" → [["a", "b"], ["c", "d"]]
    assert parse_csv("a,b\nc,d") == [["a", "b"], ["c", "d"]]

def test_ac3_quoted_field():
    # AC 3: Quoted field — '"hello, world",b' → [["hello, world", "b"]]
    assert parse_csv('"hello, world",b') == [["hello, world", "b"]]

def test_ac4_escaped_quote():
    # AC 4: Escaped quote — '"he said ""hi""",b' → [["he said \"hi\"", "b"]]
    assert parse_csv('"he said ""hi""",b') == [["he said \"hi\"", "b"]]

def test_ac5_embedded_newline():
    # AC 5: Embedded newline in quoted field — '"line1\nline2",b' → [["line1\nline2", "b"]]
    assert parse_csv('"line1\nline2",b') == [["line1\nline2", "b"]]

def test_ac6_crlf_line_endings():
    # AC 6: CRLF line endings — "a,b\r\nc,d" → [["a", "b"], ["c", "d"]]
    assert parse_csv("a,b\r\nc,d") == [["a", "b"], ["c", "d"]]

def test_ac7_empty_fields():
    # AC 7: Empty fields — ",," → [["", "", ""]]
    assert parse_csv(",,") == [["", "", ""]]

def test_ac8_single_field():
    # AC 8: Single field — "hello" → [["hello"]]
    assert parse_csv("hello") == [["hello"]]

def test_ac9_empty_input():
    # AC 9: Empty input — "" → []
    assert parse_csv("") == []

def test_ac10_trailing_newline():
    # AC 10: Trailing newline — "a,b\n" → [["a", "b"]] (no empty trailing row)
    assert parse_csv("a,b\n") == [["a", "b"]]

def test_ac11_unclosed_quote():
    # AC 11: Unclosed quote raises ValueError
    with pytest.raises(ValueError, match="Unclosed quote"):
        parse_csv('"unclosed quote')
    with pytest.raises(ValueError, match="Unclosed quote"):
        parse_csv('a,"unclosed quote')

def test_ac12_mixed_quoted_unquoted():
    # AC 12: Mixed quoted and unquoted fields in same row
    assert parse_csv('a,"b",c') == [["a", "b", "c"]]
    assert parse_csv('"a",b,"c"') == [["a", "b", "c"]]

def test_complex_mixed():
    # Extra: Combining several ACs
    input_text = 'a,"b ""quoted"" c",d\r\n"e\nf",g,h'
    expected = [
        ["a", 'b "quoted" c', "d"],
        ["e\nf", "g", "h"]
    ]
    assert parse_csv(input_text) == expected

def test_empty_row_at_end():
    # Test that two newlines at the end actually produce an empty row
    assert parse_csv("a,b\n\n") == [["a", "b"], [""]]
