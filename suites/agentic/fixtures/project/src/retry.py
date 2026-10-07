import time
import random
import functools
from typing import Callable, Any, Optional

def retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.0,
    sleep_fn: Callable[[float], Any] = time.sleep
):
    """
    Retry decorator with exponential backoff and jitter.
    
    Args:
        max_retries: Maximum number of retry attempts after the first failure.
        base_delay: Initial delay between retries.
        max_delay: Maximum cap for the delay.
        backoff_factor: Factor by which the delay grows each attempt.
        jitter: Jitter factor. Adds a random value in [0, current_delay * jitter).
        sleep_fn: Function used to sleep (default: time.sleep). Injectable for testing.
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception: Optional[Exception] = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt == max_retries:
                        break
                    
                    delay = min(base_delay * (backoff_factor ** attempt), max_delay)
                    if jitter > 0:
                        delay += random.uniform(0, delay * jitter)
                    
                    sleep_fn(delay)
            
            if last_exception is None:
                raise RuntimeError("Retry loop failed without capturing an exception")
            raise last_exception
        return wrapper
    return decorator
