---
id: EVAL-001
priority: 1
layer: 0
type: eval
status: ready
after: []
prior_attempts: []
created: 2026-03-16
---

# Custom FizzBuzz with Configurable Divisors

## Problem

The eval test project needs a basic utility function that replaces numbers with labels based on configurable divisor rules. This is the simplest possible spec — a baseline that any competent loop execution should complete with zero stalls and minimal review cycles.

## Requirements

- [ ] Implement a function `fizzbuzz(n, rules)` that takes an integer and a list of (divisor, label) tuples
- [ ] For each number 1..n, apply all matching rules; if multiple match, concatenate labels in rule order
- [ ] If no rule matches, return the number as a string
- [ ] Return a list of strings

## Acceptance Criteria

- [ ] AC 1: `fizzbuzz(15, [(3, "Fizz"), (5, "Buzz")])` returns the classic sequence ending with "FizzBuzz"
- [ ] AC 2: `fizzbuzz(1, [(3, "Fizz")])` returns `["1"]`
- [ ] AC 3: `fizzbuzz(6, [(2, "Even"), (3, "Tri")])` returns `["1", "Even", "Tri", "Even", "5", "EvenTri"]`
- [ ] AC 4: Empty rules list → every number returned as string
- [ ] AC 5: `n=0` returns empty list
- [ ] AC 6: Negative `n` returns empty list
- [ ] AC 7: Single-rule with divisor 1 → every number gets the label

## Context

- Target: `src/fizzbuzz.py`
- Tests: `tests/test_fizzbuzz.py`
- Framework: pytest
- This is an eval spec — used to benchmark the loop, not to deliver user value

## Out of Scope

- CLI interface
- File I/O
- Performance optimization (this is a correctness exercise)

## Scoring Notes

This spec is the **baseline**. Expected outcomes:
- Completion: YES (anything less is a loop failure)
- Review cycles: 0-1
- Stalls: 0
- Test pass rate: 100%
- Duration: <15 minutes
