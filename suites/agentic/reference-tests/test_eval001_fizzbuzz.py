"""Reference suite for EVAL-001 — one test per acceptance criterion.

Authored from eval-specs/EVAL-001-fizzbuzz-custom.md. The model under test never
sees this file. Contract: fizzbuzz(n, rules) -> list[str].
"""

import pytest

fizzbuzz = pytest.importorskip("fizzbuzz").fizzbuzz


def test_ac1_classic_sequence_ends_with_fizzbuzz():
    assert fizzbuzz(15, [(3, "Fizz"), (5, "Buzz")]) == [
        "1", "2", "Fizz", "4", "Buzz", "Fizz", "7", "8",
        "Fizz", "Buzz", "11", "Fizz", "13", "14", "FizzBuzz",
    ]


def test_ac2_single_number_no_match():
    assert fizzbuzz(1, [(3, "Fizz")]) == ["1"]


def test_ac3_labels_concatenate_in_rule_order():
    assert fizzbuzz(6, [(2, "Even"), (3, "Tri")]) == [
        "1", "Even", "Tri", "Even", "5", "EvenTri",
    ]


def test_ac4_empty_rules_returns_every_number_as_string():
    assert fizzbuzz(3, []) == ["1", "2", "3"]


def test_ac5_n_zero_returns_empty_list():
    assert fizzbuzz(0, [(3, "Fizz")]) == []


def test_ac6_negative_n_returns_empty_list():
    assert fizzbuzz(-5, [(3, "Fizz")]) == []


def test_ac7_divisor_one_labels_every_number():
    assert fizzbuzz(3, [(1, "X")]) == ["X", "X", "X"]
