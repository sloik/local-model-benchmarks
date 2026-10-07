#!/usr/bin/env python3
"""B1 · Rate-with-confidence-bound scoring (AC12 done right).

Plan: ../Improvement-Plan-Agent-Panel-2026-07-17.md (item B1)
Spec: ../specs/SPEC-B02-router-detector-rate.md (B1 section, AC-B1-1..4)
Findings: ../Local-Model-Reliability-Findings.md ("~1 in 11 (~9%)")

WHAT THIS IS (and is NOT)
    PLUMBING. It fixes RARITY, not INVISIBILITY. The measured failure is rare,
    stochastic, and confident (~1 in 11 identical temp-0 runs). A point estimate
    over a few runs is misleading: 5/5 clean looks perfect yet only bounds the
    true rate below ~a half. So we gate on the CONFIDENCE-INTERVAL UPPER BOUND,
    never on the point estimate. This module is pure arithmetic + orchestration:
    it takes an injected per-run scorer (`run_fn`) and NEVER calls a model.

THE d WARNING (do not silence it)
    Every rate reported here is precision-WITHOUT-accuracy until the detector's
    sensitivity `d` (TPR on KNOWN fabrications) is measured. N=60 on an instance
    the detector cannot see yields a confident, precise measurement of ZERO. So
    the result dict carries `rate/d` (with `d` shown) and a loud warning whenever
    `d` is unknown.

Usage:
    python scripts/score_rate.py --demo   # Bernoulli(0.09) sim run_fn at N=20
"""
import argparse
import math
import random
import sys


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for k fabrications in n runs. Pure stdlib `math`.

    Returns (lo, hi), both clamped to [0, 1]. Handles the degenerate cases
    without ZeroDivision:
      - n == 0 -> (0.0, 1.0)  (no information; the whole unit interval)
      - k == 0 -> lo ~ 0
      - k == n -> hi ~ 1
    """
    if not isinstance(k, int) or not isinstance(n, int):
        raise ValueError("k and n must be ints; got k=%r n=%r" % (k, n))
    if n < 0:
        raise ValueError("n must be >= 0; got %d" % n)
    if k < 0:
        raise ValueError("k must be >= 0; got %d" % k)
    if k > n:
        raise ValueError("k must be <= n; got k=%d n=%d" % (k, n))
    if n == 0:
        return (0.0, 1.0)

    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lo = center - margin
    hi = center + margin
    # Float slop near the boundaries can push these a hair outside [0, 1].
    return (max(0.0, lo), min(1.0, hi))


def rate_score(run_fn, n=20, reload_fn=None, reload_every=None, d=None):
    """Score a fabrication RATE over n injected temp-0 runs.

    `run_fn()` is called n times and must return a bool `fabricated`
    (True = the bad event). NO model call happens in this module — `run_fn`
    is the injected seam.

    Spread-across-reloads (AC-B1-3): Metal nondeterminism is correlated within
    one loaded session, so N runs in one session are NOT i.i.d. If `reload_fn`
    and `reload_every` are given, `reload_fn()` is called before every block of
    `reload_every` runs (i.e. at run indices 0, reload_every, 2*reload_every, ...)
    so N is spread across separate model reloads. This module does not load
    models itself; it only exposes the hook point.

    `d` (AC-B1-4) is the detector's measured sensitivity (TPR). When given,
    `rate/d` is reported; when unknown, a loud precision-without-accuracy warning
    is emitted and `rate_over_d` is None.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be an int >= 1; got %r" % (n,))
    if reload_every is not None and (not isinstance(reload_every, int) or reload_every < 1):
        raise ValueError("reload_every must be an int >= 1 or None; got %r" % (reload_every,))
    if d is not None and not (0.0 < d <= 1.0):
        raise ValueError("d must be in (0, 1] or None; got %r" % (d,))

    k = 0
    for i in range(n):
        if reload_fn is not None and reload_every is not None and i % reload_every == 0:
            reload_fn()
        if run_fn():
            k += 1

    rate = k / n
    lo, hi = wilson_interval(k, n)
    if d is None:
        warning = (
            "WARNING: detector sensitivity d is UNKNOWN — this rate is "
            "PRECISION-WITHOUT-ACCURACY. A confident, precise measurement of ZERO "
            "is meaningless if the detector cannot see the failure. rate/d is "
            "uncomputable. Measure d (TPR on KNOWN fabrications) before trusting "
            "this number (Improvement-Plan B1 / Local-Model-Reliability-Findings)."
        )
        rate_over_d = None
    else:
        warning = None
        rate_over_d = rate / d

    return {
        "n": n,
        "fabrications": k,
        "rate": rate,
        "wilson95": [lo, hi],
        "min_detectable_rate": 3 / n,
        "rate_over_d": rate_over_d,
        "d": d,
        "warning": warning,
    }


def passes(result, threshold):
    """Gate on the CI UPPER BOUND, not the point estimate.

    Returns True iff the Wilson 95% upper bound is at or below `threshold`.
    Rule of three: a 5/5-clean run (0 fabrications in 5) does NOT certify a low
    rate — its upper bound is ~52%, so it only certifies "true rate < ~52%",
    nowhere near a 10% bar. Certifying a 10% rate needs ~N=60 clean runs; 5%
    needs ~N=120. This is why a point-estimate pass is forbidden.
    """
    return result["wilson95"][1] <= threshold


def screen(run_fn, n=5):
    """5-run SCREEN: may only REJECT, never certify (AC-B1-2).

    Any fabrication in n runs -> REJECT. Zero fabrications -> INCONCLUSIVE
    (NOT a pass — the upper bound is far too wide at n=5 to certify anything).
    Use rate_score with N>=20 (and passes()) to certify.
    """
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be an int >= 1; got %r" % (n,))
    k = sum(1 for _ in range(n) if run_fn())
    return {
        "n": n,
        "fabrications": k,
        "verdict": "REJECT" if k > 0 else "INCONCLUSIVE",
        "note": "A screen can only reject. It never certifies — use rate_score(N>=20).",
    }


def _demo():
    """Bernoulli(0.09) simulated run_fn at N=20 so the math is visible."""
    rng = random.Random(20260717)
    run_fn = lambda: rng.random() < 0.09
    return rate_score(run_fn, n=20)


def main(argv=None):
    ap = argparse.ArgumentParser(description="B1 rate-with-CI scoring (no model calls)")
    ap.add_argument("--demo", action="store_true",
                    help="run a built-in Bernoulli(0.09) sim run_fn at N=20 and print the dict")
    a = ap.parse_args(argv)
    if not a.demo:
        ap.error("nothing to do: pass --demo (this module scores an INJECTED run_fn)")
    import json
    print(json.dumps(_demo(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
