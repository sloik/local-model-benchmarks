import pytest
from unittest.mock import Mock, call
from src.retry import retry

def test_ac1_success_first_try():
    """AC 1: Function succeeds on first try -> no retries, no delay, result returned"""
    mock_sleep = Mock()
    
    @retry(sleep_fn=mock_sleep)
    def succeed():
        return "success"
    
    assert succeed() == "success"
    mock_sleep.assert_not_called()

def test_ac2_fail_once_then_succeed():
    """AC 2: Function fails once then succeeds -> 1 retry, 1 delay call, result returned"""
    mock_sleep = Mock()
    counter = 0
    
    @retry(max_retries=3, sleep_fn=mock_sleep)
    def fail_once():
        nonlocal counter
        counter += 1
        if counter == 1:
            raise ValueError("fail")
        return "success"
    
    assert fail_once() == "success"
    assert counter == 2
    assert mock_sleep.call_count == 1

def test_ac3_fail_max_retries():
    """AC 3: Function fails max_retries times -> raises last exception"""
    mock_sleep = Mock()
    
    @retry(max_retries=2, sleep_fn=mock_sleep)
    def always_fail():
        raise TypeError("permanent fail")
    
    with pytest.raises(TypeError) as excinfo:
        always_fail()
    assert str(excinfo.value) == "permanent fail"
    # Total attempts = 1 (initial) + 2 (retries) = 3. Delays occur between them, so 2 delays.
    assert mock_sleep.call_count == 2

def test_ac4_backoff_doubles():
    """AC 4: Backoff doubles each attempt (base=1, factor=2 -> delays: 1, 2, 4, 8...)"""
    mock_sleep = Mock()
    
    @retry(max_retries=4, base_delay=1.0, backoff_factor=2.0, jitter=0, sleep_fn=mock_sleep)
    def always_fail():
        raise RuntimeError("fail")
    
    with pytest.raises(RuntimeError):
        always_fail()
    
    expected_calls = [call(1.0), call(2.0), call(4.0), call(8.0)]
    mock_sleep.assert_has_calls(expected_calls)

def test_ac5_max_delay_caps():
    """AC 5: max_delay caps the backoff (base=1, factor=2, max_delay=5 -> delays: 1, 2, 4, 5, 5...)"""
    mock_sleep = Mock()
    
    @retry(max_retries=5, base_delay=1.0, backoff_factor=2.0, max_delay=5.0, jitter=0, sleep_fn=mock_sleep)
    def always_fail():
        raise RuntimeError("fail")
    
    with pytest.raises(RuntimeError):
        always_fail()
    
    expected_calls = [call(1.0), call(2.0), call(4.0), call(5.0), call(5.0)]
    mock_sleep.assert_has_calls(expected_calls)

def test_ac6_deterministic_jitter_zero():
    """AC 6: jitter=0 produces deterministic delays (testable without randomness)"""
    mock_sleep = Mock()
    
    @retry(max_retries=2, base_delay=1.0, backoff_factor=3.0, jitter=0, sleep_fn=mock_sleep)
    def always_fail():
        raise RuntimeError("fail")
    
    with pytest.raises(RuntimeError):
        always_fail()
    
    # Delays should be: 1 * 3^0 = 1, 1 * 3^1 = 3
    mock_sleep.assert_has_calls([call(1.0), call(3.0)])

def test_ac7_jitter_randomness():
    """AC 7: jitter>0 adds randomness within expected range"""
    mock_sleep = Mock()
    base_delay = 1.0
    jitter_factor = 0.5 # Adds [0, delay * 0.5)
    
    @retry(max_retries=1, base_delay=base_delay, jitter=jitter_factor, sleep_fn=mock_sleep)
    def fail_once():
        raise RuntimeError("fail")
    
    with pytest.raises(RuntimeError):
        fail_once()
    
    # delay = min(1 * 2^0, max) = 1.0. Jitter adds [0, 1.0 * 0.5) = [0, 0.5).
    # Total range: [1.0, 1.5)
    actual_delay = mock_sleep.call_args[0][0]
    assert base_delay <= actual_delay < base_delay + (base_delay * jitter_factor)

def test_ac8_preserves_metadata():
    """AC 8: Decorated function preserves original function's name and docstring"""
    def my_func():
        """Original Docstring"""
        pass
    
    decorated = retry()(my_func)
    assert decorated.__name__ == "my_func"
    assert decorated.__doc__ == "Original Docstring"

def test_ac9_works_with_methods():
    """AC 9: Works with both sync functions and methods"""
    mock_sleep = Mock()
    
    class Service:
        def __init__(self):
            self.count = 0
        
        @retry(max_retries=2, sleep_fn=mock_sleep)
        def flaky_method(self, x):
            self.count += 1
            if self.count < 2:
                raise ValueError("fail")
            return x * 2

    service = Service()
    assert service.flaky_method(10) == 20
    assert service.count == 2
    assert mock_sleep.call_count == 1

def test_ac10_sleep_fn_receives_value():
    """AC 10: sleep_fn receives the computed delay value (verifiable via mock)"""
    mock_sleep = Mock()
    
    @retry(max_retries=1, base_delay=42.0, jitter=0, sleep_fn=mock_sleep)
    def fail():
        raise RuntimeError("fail")
    
    with pytest.raises(RuntimeError):
        fail()
    
    # The delay should be exactly 42.0
    mock_sleep.assert_called_once_with(42.0)
