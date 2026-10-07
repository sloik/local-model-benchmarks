"""Reference suite for EVAL-005 — one test per acceptance criterion.

Authored from eval-specs/EVAL-005-refactor-extract-class.md. The model under test
never sees this file. It scores the model's src/calculator.py (via EVAL_SRC_DIR),
not the model's own tests.

EVAL-005 is a *refactor* task. After BUG-004 the fixture ships a working inline
calculate() with NO Calculator class, so "the 48 tests still pass" no longer
distinguishes a no-op from a real refactor. A pass here requires the model to have
extracted a Calculator class with independently-usable parse() and evaluate()
methods (AC3/AC4/AC6) while keeping calculate() backward-compatible (AC1/AC2/AC7)
and the 48 held-out tests green (AC5). A no-op leaves no Calculator class, so the
import below skips and every structural AC fails.
"""

import inspect
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

calculator = pytest.importorskip("calculator")

# The 48 held-out behavioral+structural tests the refactor must keep green (AC5).
# BUG-006: score against the PRISTINE copy under reference-tests/, not the in-tree
# eval-project/tests/test_calculator.py. The in-tree file sits in the model's
# writable working tree, so a model told to "keep the tests green" could weaken or
# delete them to pass; the pristine copy is out of the model's reach and is the
# graded contract. A checksum guard (score_reference.py) catches any edit to the
# in-tree copy separately. The suite runs the pristine tests against whatever src
# is under test (EVAL_SRC_DIR).
HELD_OUT_TESTS = (
    Path(__file__).resolve().parent / "eval005_grader" / "test_calculator.py"
)


def test_ac1_calculate_addition_backward_compatible():
    """AC1: calculate("2 + 3") returns 5.0 (backward compatible)."""
    assert calculator.calculate("2 + 3") == 5.0


def test_ac2_calculate_float_division():
    """AC2: calculate("10 / 3") returns 3.333... (float division)."""
    assert abs(calculator.calculate("10 / 3") - 3.3333333333) < 1e-6


def test_ac3_calculator_parse_returns_token_list():
    """AC3: Calculator().parse("2 + 3") returns a non-empty token list."""
    tokens = calculator.Calculator().parse("2 + 3")
    assert isinstance(tokens, list)
    assert tokens, "parse must return a non-empty token list"


def test_ac4_calculator_evaluate_of_parse_is_five():
    """AC4: Calculator().evaluate(Calculator().parse("2 + 3")) returns 5.0."""
    calc = calculator.Calculator()
    assert calc.evaluate(calc.parse("2 + 3")) == 5.0


def test_ac5_all_pre_existing_tests_pass_unmodified():
    """AC5: the 48 pre-existing held-out tests pass against this src, unmodified.

    This is the "refactor did not break behavior" half of the grade and encodes
    BUG-004 AC4's "the 48 tests pass AND the class exists" condition directly.

    The held-out test file is copied to an isolated temp dir so pytest imports the
    src under test (EVAL_SRC_DIR) and never the eval-project/src fixture that its own
    rootdir/prepend logic would otherwise force onto sys.path.
    """
    src = os.environ["EVAL_SRC_DIR"]
    with tempfile.TemporaryDirectory() as td:
        shutil.copy(HELD_OUT_TESTS, Path(td) / HELD_OUT_TESTS.name)
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", HELD_OUT_TESTS.name, "-q",
             "-p", "no:cacheprovider", "--import-mode=importlib"],
            cwd=td,
            env={**os.environ, "PYTHONPATH": src},
            capture_output=True, text=True, timeout=120,
        )
    assert proc.returncode == 0, (
        f"the 48 held-out tests must pass unmodified:\n{proc.stdout}\n{proc.stderr}"
    )


def test_ac6_parse_and_evaluate_are_independently_usable():
    """AC6: parse() and evaluate() are separable — parse tokenizes without
    evaluating, evaluate consumes a hand-built token list without calling parse."""
    calc = calculator.Calculator()
    tokens = calc.parse("10 - 4")
    assert isinstance(tokens, list)
    # evaluate works on a token list the test built itself, not one from parse()
    assert calc.evaluate(["3", "*", "7"]) == 21.0
    # and the two compose end-to-end
    assert calc.evaluate(tokens) == 6.0


def test_ac7_calculate_is_module_level_callable():
    """AC7: calculate remains a module-level function, not only a class method."""
    assert inspect.isfunction(calculator.calculate)
    assert calculator.calculate("1 + 2 + 3") == 6.0
