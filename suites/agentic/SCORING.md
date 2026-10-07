# Eval Scoring Criteria

> How to score a Nightshift Kit loop run against eval specs.
> Each eval run produces a metrics YAML file — this document defines how to interpret it.

---

## Dimensions

Every eval run is scored on 7 dimensions. Each dimension is 0-100.

### 1. Completion (binary: 0 or 100)

Did the spec reach `status: done`?

| Outcome | Score |
|---------|-------|
| Spec completed, all AC met | 100 |
| Spec blocked (intentionally, e.g. EVAL-004) | 100 (if blocking was correct) |
| Spec blocked (unintentionally — agent got stuck) | 0 |
| Spec abandoned / circuit breaker fired | 0 |

### 2. Test Pass Rate (from metrics)

```
score = (tests_passed / tests_total) * 100
```

Read from `metrics.validation.test_results.passed` / `total`.
Final state only — intermediate red-phase failures don't count against the score.

### 3. TDD Compliance (0, 50, or 100)

Did the agent follow the Red → Green → Refactor cycle?

| Behavior | Score |
|----------|-------|
| Tests written before implementation, committed red, then green | 100 |
| Tests written before implementation but not committed separately | 50 |
| Implementation written first, tests added after | 0 |

Evidence: check git log for commit order (`test(SPEC-XXX): add tests (red)` before `feat(SPEC-XXX): implement`).

### 4. Review Efficiency (0-100)

How many review cycles did it take?

```
score = max(0, 100 - (review_cycles - 1) * 25)
```

| Cycles | Score |
|--------|-------|
| 0-1 | 100 |
| 2 | 75 |
| 3 | 50 |
| 4 | 25 |
| 5+ | 0 |

Read from `metrics.review.cycles`.

### 5. Duration Efficiency (relative to expected)

Each eval spec declares an expected duration in its "Scoring Notes" section.

```
ratio = actual_duration / expected_duration
score = 100 if ratio <= 1.0
score = max(0, 100 - (ratio - 1.0) * 50) if ratio > 1.0
```

| Ratio | Score |
|-------|-------|
| ≤1.0x | 100 |
| 1.5x | 75 |
| 2.0x | 50 |
| 3.0x+ | 0 |

Read from `metrics.timing.total_duration_min` vs spec's expected duration.

### 6. Stall Freedom (binary: 0 or 100)

Did any circuit breaker trigger?

| Outcome | Score |
|---------|-------|
| No stalls, no circuit breakers | 100 |
| Stall detected but recovered | 50 |
| Circuit breaker fired (spec abandoned) | 0 |

Read from `metrics.stall`.

### 7. Metrics Completeness (0-100)

Did the agent produce a valid, complete metrics file?

```
score = (fields_present / fields_expected) * 100
```

Required fields (per _SCHEMA.md): task_id, status, loop_version, model, harness, timing, preflight, context_load, test_planning, test_writing, implementation, review, validation.

---

## Composite Score

```
composite = weights · scores

weights:
  completion:    0.25
  test_pass:     0.20
  tdd_compliance: 0.15
  review_eff:    0.10
  duration_eff:  0.10
  stall_freedom: 0.10
  metrics_comp:  0.10
```

### Interpretation

| Composite | Rating | Meaning |
|-----------|--------|---------|
| 90-100 | Excellent | Loop executed cleanly, production-ready |
| 75-89 | Good | Minor inefficiencies, loop works |
| 50-74 | Needs Work | Loop completed but with significant friction |
| <50 | Failing | Loop has fundamental issues to address |

---

## Per-Spec Expected Outcomes

| Spec | Expected Completion | Expected Cycles | Expected Duration | Special |
|------|-------------------|-----------------|-------------------|---------|
| EVAL-001 | done | 0-1 | <15 min | Baseline — must score 90+ |
| EVAL-002 | done | 1-2 | <30 min | Tests edge case thoroughness |
| EVAL-003 | done | 1-3 | <30 min | Tests mocking discipline |
| EVAL-004 | blocked | 0 | <5 min | Must detect ambiguity, NOT implement |
| EVAL-005 | done | 1-2 | <20 min | Must not modify existing tests |

---

## Running an Eval

1. Set up the eval project scaffold (see `_eval-project/`)
2. Fill `config.yaml` for the eval project
3. Point the harness at `BOOTSTRAP.md`
4. Let it process all `ready` eval specs
5. Collect metrics YAML files
6. Run `analyze_metrics.py` on the results
7. Score each dimension per this document
8. Compute composite score
9. Record in `reports/` with date, model, harness, loop version

### Comparing Loop Versions

Run the same eval specs with different versions of LOOP.md. Keep the model and harness constant. Compare composite scores. This is the primary mechanism for data-driven loop improvement.

### Comparing Models

Run the same eval specs + same LOOP.md with different models (e.g., Claude Opus vs Sonnet vs local via LM Studio). Compare composite scores. This reveals which models can follow the protocol effectively.

---

## Adding New Eval Specs

When adding an eval spec:
1. Use `type: eval` in frontmatter
2. Place in `specs/_eval/`
3. Include a "Scoring Notes" section with expected outcomes
4. Keep it small (ideally <30 min for the strongest model)
5. Test one loop capability per spec — avoid compound evaluations
6. If the spec requires pre-existing code (like EVAL-005), document it in the scaffold
