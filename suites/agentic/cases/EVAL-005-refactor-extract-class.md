---
id: EVAL-005
priority: 5
layer: 2
type: eval
status: ready
after: [EVAL-001]
prior_attempts: []
created: 2026-03-16
---

# Refactor: Extract Calculator Class from Inline Logic

## Problem

The file `src/calculator.py` contains a single function `calculate(expression: str)` that parses and evaluates simple arithmetic expressions inline. The logic mixes parsing and evaluation. This should be refactored into a `Calculator` class with separate `parse()` and `evaluate()` methods — without changing any external behavior.

**Pre-existing code and tests will be provided in the eval scaffold.** The agent must refactor without breaking the existing test suite.

## Requirements

- [ ] Extract a `Calculator` class from the existing `calculate()` function
- [ ] Class has `parse(expression: str) -> list` that returns a token list
- [ ] Class has `evaluate(tokens: list) -> float` that evaluates the token list
- [ ] Keep the module-level `calculate(expression)` function as a convenience wrapper that delegates to `Calculator`
- [ ] All existing tests in `tests/test_calculator.py` must continue to pass without modification
- [ ] Add new tests for `Calculator.parse()` and `Calculator.evaluate()` individually

## Acceptance Criteria

- [ ] AC 1: Existing `calculate("2 + 3")` returns `5.0` (backward compatible)
- [ ] AC 2: Existing `calculate("10 / 3")` returns `3.333...` (float division)
- [ ] AC 3: `Calculator().parse("2 + 3")` returns a token list (structure agent's choice)
- [ ] AC 4: `Calculator().evaluate(Calculator().parse("2 + 3"))` returns `5.0`
- [ ] AC 5: All pre-existing tests pass without modification
- [ ] AC 6: New tests cover `parse()` and `evaluate()` independently
- [ ] AC 7: `calculate` function still works as a module-level import

## Context

- Target: `src/calculator.py` (modify existing file)
- Tests: `tests/test_calculator.py` (add to, don't modify existing tests)
- Framework: pytest
- The existing code and tests are provided in the eval scaffold — read them before planning
- This is a refactor spec: type=eval but pattern is type=refactor

## Out of Scope

- Adding new operations (only +, -, *, / are in scope)
- Parentheses or operator precedence changes
- Error handling improvements (preserve existing behavior)

## Scoring Notes

This spec tests **refactoring discipline**. Key dimensions:
- Did all pre-existing tests pass without modification? (Critical — any test edit is a failure)
- Was the refactor clean (no behavior changes, just structure)?
- Were new tests added for the extracted methods?
- Expected review cycles: 1-2
- Duration: <20 minutes
