---
id: EVAL-003
priority: 3
layer: 1
type: eval
status: ready
after: []
prior_attempts: []
created: 2026-03-16
---

# Retry Decorator with Exponential Backoff

## Problem

The project needs a reusable retry decorator that wraps flaky operations with exponential backoff and jitter. This tests the agent's ability to write testable time-dependent code using dependency injection and mocking — a common real-world pattern that exercises TDD discipline.

## Requirements

- [ ] Implement a `@retry` decorator with configurable: max_retries, base_delay, max_delay, backoff_factor, jitter
- [ ] Exponential backoff: delay = min(base_delay * backoff_factor^attempt, max_delay)
- [ ] Jitter: add random value in [0, delay * jitter_factor) to prevent thundering herd
- [ ] Accept a `sleep_fn` parameter (default: `time.sleep`) for testability — tests MUST NOT actually sleep
- [ ] On final failure, raise the last exception with original traceback preserved
- [ ] Return the successful result if any attempt succeeds

## Acceptance Criteria

- [ ] AC 1: Function succeeds on first try → no retries, no delay, result returned
- [ ] AC 2: Function fails once then succeeds → 1 retry, 1 delay call, result returned
- [ ] AC 3: Function fails max_retries times → raises last exception
- [ ] AC 4: Backoff doubles each attempt (base=1, factor=2 → delays: 1, 2, 4, 8...)
- [ ] AC 5: max_delay caps the backoff (base=1, factor=2, max_delay=5 → delays: 1, 2, 4, 5, 5...)
- [ ] AC 6: jitter=0 produces deterministic delays (testable without randomness)
- [ ] AC 7: jitter>0 adds randomness within expected range
- [ ] AC 8: Decorated function preserves original function's name and docstring
- [ ] AC 9: Works with both sync functions and methods
- [ ] AC 10: sleep_fn receives the computed delay value (verifiable via mock)

## Context

- Target: `src/retry.py`
- Tests: `tests/test_retry.py`
- Framework: pytest
- Tests must use `sleep_fn` injection — never call `time.sleep` in tests
- Use `unittest.mock` for tracking calls, not for patching globals

## Out of Scope

- Async support
- Circuit breaker pattern
- Retry-specific exceptions (retry all exceptions)
- Logging

## Scoring Notes

This spec tests **testability and mocking discipline**. Key dimensions:
- Does the agent use `sleep_fn` injection rather than patching `time.sleep`?
- Are delays verified precisely via the mock?
- Is jitter tested with a seeded random or range assertion?
- Expected review cycles: 1-3
- Duration: <30 minutes
