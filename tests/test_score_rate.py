#!/usr/bin/env python3
"""Tests for scripts/score_rate.py (B1). NO model calls — inject deterministic run_fns.

Run: python -m pytest tests/test_score_rate.py -v
"""
import importlib.util
import pathlib
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_MOD_PATH = _HERE.parent / "scripts" / "score_rate.py"
_spec = importlib.util.spec_from_file_location("score_rate", _MOD_PATH)
score_rate_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = score_rate_mod
_spec.loader.exec_module(score_rate_mod)

wilson_interval = score_rate_mod.wilson_interval
rate_score = score_rate_mod.rate_score
passes = score_rate_mod.passes
screen = score_rate_mod.screen
main = score_rate_mod.main


# --- deterministic injectable run_fns (NO model) ---

def never():
    return False


def always():
    return True


def make_sequence(bools):
    """run_fn that yields the given bools in order (then raises if over-called)."""
    it = iter(bools)
    return lambda: next(it)


# --- Wilson interval ---

def test_wilson_k0_n5_rule_of_three_ballpark():
    lo, hi = wilson_interval(0, 5)
    assert lo == pytest.approx(0.0, abs=1e-6)
    assert 0.4 < hi < 0.6  # rule-of-three ballpark for a 0/5 clean run


def test_wilson_k_equals_n():
    lo, hi = wilson_interval(5, 5)
    assert hi == pytest.approx(1.0, abs=1e-6)
    assert lo > 0.5  # a 5/5-fabrication run bounds the true rate well above 0


def test_wilson_n0_returns_unit_interval_no_crash():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_bounds_within_unit_interval():
    for k, n in [(0, 5), (1, 20), (3, 20), (20, 20), (0, 60)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_wilson_rejects_bad_inputs():
    with pytest.raises(ValueError):
        wilson_interval(-1, 5)
    with pytest.raises(ValueError):
        wilson_interval(0, -5)
    with pytest.raises(ValueError):
        wilson_interval(6, 5)  # k > n


# --- rate_score core ---

def test_min_detectable_rate_is_three_over_n():
    r = rate_score(never, n=20)
    assert r["min_detectable_rate"] == 3 / 20
    r2 = rate_score(never, n=60)
    assert r2["min_detectable_rate"] == 3 / 60


def test_rate_and_fabrication_count():
    # 2 fabrications in 20 runs
    seq = [True, True] + [False] * 18
    r = rate_score(make_sequence(seq), n=20)
    assert r["fabrications"] == 2
    assert r["rate"] == pytest.approx(2 / 20)
    assert r["n"] == 20


# --- gate uses the UPPER bound, not the point estimate ---

def test_gate_0of5_does_not_pass_10pct():
    r = rate_score(never, n=5)
    # point estimate is 0.0, but the Wilson upper bound (~0.4-0.5) exceeds 10%
    assert r["rate"] == 0.0
    assert not passes(r, 0.10)


def test_gate_0of60_does_pass_10pct():
    r = rate_score(never, n=60)
    assert r["rate"] == 0.0
    assert passes(r, 0.10)  # upper bound ~6% <= 10%


# --- rate_over_d and the loud warning ---

def test_rate_over_d_none_with_warning_when_d_unknown():
    r = rate_score(make_sequence([True] + [False] * 19), n=20)
    assert r["d"] is None
    assert r["rate_over_d"] is None
    assert r["warning"] is not None
    assert "PRECISION-WITHOUT-ACCURACY" in r["warning"]


def test_rate_over_d_computed_when_d_given():
    # 2/20 = 0.1 rate, d=0.5 -> rate/d = 0.2
    seq = [True, True] + [False] * 18
    r = rate_score(make_sequence(seq), n=20, d=0.5)
    assert r["d"] == 0.5
    assert r["rate_over_d"] == pytest.approx(0.2)
    assert r["warning"] is None


# --- screen: reject-only, never certify ---

def test_screen_rejects_on_any_fabrication():
    r = screen(make_sequence([False, False, True, False, False]))
    assert r["verdict"] == "REJECT"


def test_screen_never_certifies_on_clean_run():
    r = screen(never)  # 0/5 clean
    assert r["verdict"] == "INCONCLUSIVE"
    assert r["verdict"] != "PASS"


# --- reload hook: called the right number of times ---

def test_reload_fn_called_per_block():
    counter = {"n": 0}

    def reload_fn():
        counter["n"] += 1

    rate_score(never, n=20, reload_fn=reload_fn, reload_every=5)
    # reloads at run indices 0, 5, 10, 15 -> 4 separate reloads
    assert counter["n"] == 4


def test_reload_fn_not_called_without_reload_every():
    counter = {"n": 0}

    def reload_fn():
        counter["n"] += 1

    rate_score(never, n=20, reload_fn=reload_fn)  # reload_every is None
    assert counter["n"] == 0


# --- invalid inputs raise ValueError (clear path) ---

def test_rate_score_rejects_negative_n():
    with pytest.raises(ValueError):
        rate_score(never, n=-5)


def test_rate_score_rejects_zero_n():
    with pytest.raises(ValueError):
        rate_score(never, n=0)


def test_rate_score_rejects_bad_d():
    with pytest.raises(ValueError):
        rate_score(never, n=20, d=0.0)
    with pytest.raises(ValueError):
        rate_score(never, n=20, d=1.5)


def test_rate_score_rejects_bad_reload_every():
    with pytest.raises(ValueError):
        rate_score(never, n=20, reload_fn=lambda: None, reload_every=0)


# --- CLI ---

def test_cli_without_demo_exits():
    with pytest.raises(SystemExit):
        main([])  # argparse.error -> SystemExit


def test_cli_demo_runs(capsys):
    rc = main(["--demo"])
    assert rc == 0
    out = capsys.readouterr().out
    assert '"wilson95"' in out
    assert '"min_detectable_rate"' in out
