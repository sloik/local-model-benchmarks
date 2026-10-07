"""Reference suite for EVAL-003 — one test per acceptance criterion.

Authored from eval-specs/EVAL-003-retry-backoff.md. The model under test never
sees this file. Contract per the spec's Requirements:

    @retry(max_retries=, base_delay=, max_delay=, backoff_factor=, jitter=, sleep_fn=)
    delay = min(base_delay * backoff_factor**attempt, max_delay)

sleep_fn is injected so no test ever really sleeps.
"""

import pytest

retry = pytest.importorskip("retry").retry


class Boom(Exception):
    pass


def _recorder():
    """Return (sleep_fn, delays) capturing every delay handed to sleep_fn."""
    delays = []
    return (lambda d: delays.append(d)), delays


def test_ac1_success_first_try_no_retry_no_delay():
    sleep_fn, delays = _recorder()
    calls = []

    @retry(max_retries=3, base_delay=1, backoff_factor=2, jitter=0, sleep_fn=sleep_fn)
    def ok():
        calls.append(1)
        return "result"

    assert ok() == "result"
    assert len(calls) == 1
    assert delays == []


def test_ac2_one_failure_then_success():
    sleep_fn, delays = _recorder()
    calls = []

    @retry(max_retries=3, base_delay=1, backoff_factor=2, jitter=0, sleep_fn=sleep_fn)
    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise Boom("first")
        return "result"

    assert flaky() == "result"
    assert len(calls) == 2
    assert len(delays) == 1


def test_ac3_exhausted_retries_raises_last_exception():
    sleep_fn, _ = _recorder()

    @retry(max_retries=2, base_delay=1, backoff_factor=2, jitter=0, sleep_fn=sleep_fn)
    def always_fails():
        raise Boom("last")

    with pytest.raises(Boom, match="last"):
        always_fails()


def test_ac4_backoff_doubles_each_attempt():
    sleep_fn, delays = _recorder()

    @retry(max_retries=4, base_delay=1, backoff_factor=2, jitter=0, sleep_fn=sleep_fn)
    def always_fails():
        raise Boom("x")

    with pytest.raises(Boom):
        always_fails()
    assert delays[:4] == [1, 2, 4, 8]


def test_ac5_max_delay_caps_the_backoff():
    sleep_fn, delays = _recorder()

    @retry(max_retries=5, base_delay=1, backoff_factor=2, max_delay=5, jitter=0, sleep_fn=sleep_fn)
    def always_fails():
        raise Boom("x")

    with pytest.raises(Boom):
        always_fails()
    assert delays[:5] == [1, 2, 4, 5, 5]


def test_ac6_jitter_zero_is_deterministic():
    runs = []
    for _ in range(2):
        sleep_fn, delays = _recorder()

        @retry(max_retries=3, base_delay=1, backoff_factor=2, jitter=0, sleep_fn=sleep_fn)
        def always_fails():
            raise Boom("x")

        with pytest.raises(Boom):
            always_fails()
        runs.append(list(delays))
    assert runs[0] == runs[1]


def test_ac7_jitter_adds_randomness_within_range():
    sleep_fn, delays = _recorder()

    @retry(max_retries=4, base_delay=1, backoff_factor=2, jitter=0.5, sleep_fn=sleep_fn)
    def always_fails():
        raise Boom("x")

    with pytest.raises(Boom):
        always_fails()
    base = [1, 2, 4, 8]
    assert len(delays) >= 4
    for actual, expected in zip(delays[:4], base):
        # jitter adds a random value in [0, delay * jitter_factor)
        assert expected <= actual <= expected * 1.5


def test_ac8_preserves_name_and_docstring():
    sleep_fn, _ = _recorder()

    @retry(max_retries=1, base_delay=1, backoff_factor=2, jitter=0, sleep_fn=sleep_fn)
    def documented():
        """Original docstring."""
        return 1

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Original docstring."


def test_ac9_works_on_methods():
    sleep_fn, delays = _recorder()

    class Service:
        def __init__(self):
            self.calls = 0

        @retry(max_retries=3, base_delay=1, backoff_factor=2, jitter=0, sleep_fn=sleep_fn)
        def fetch(self):
            self.calls += 1
            if self.calls == 1:
                raise Boom("first")
            return "ok"

    assert Service().fetch() == "ok"
    assert len(delays) == 1


def test_ac10_sleep_fn_receives_computed_delay():
    sleep_fn, delays = _recorder()

    @retry(max_retries=1, base_delay=3, backoff_factor=2, jitter=0, sleep_fn=sleep_fn)
    def always_fails():
        raise Boom("x")

    with pytest.raises(Boom):
        always_fails()
    assert delays == [3]
