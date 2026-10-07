---
id: EVAL-004
priority: 4
layer: 2
type: eval
status: ready
after: []
prior_attempts: []
created: 2026-03-16
---

# User Profile Validation

## Problem

Users submit profile data that needs validation before saving. Some profiles are invalid and should be rejected.

## Requirements

- [ ] Validate user profiles
- [ ] Reject bad data
- [ ] Return helpful error messages

## Acceptance Criteria

- [ ] AC 1: Valid profiles pass validation
- [ ] AC 2: Invalid profiles are rejected
- [ ] AC 3: Error messages are helpful

## Context

- Target: `src/validator.py`
- Tests: `tests/test_validator.py`

## Out of Scope

- Database storage
- Authentication
