"""Rate limiting and retry logic with exponential backoff.

Provides configurable rate limiting and automatic retry with
exponential backoff for API calls.
"""

import logging
import time
from dataclasses import dataclass
from functools import wraps
from typing import Callable, Optional, TypeVar, Any
from enum import Enum

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryStrategy(Enum):
    """Retry strategies for failed requests."""
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"  # Linear backoff
    FIXED = "fixed"  # Fixed delay


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting and retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    exponential_base: float = 2.0
    jitter: bool = True  # Add random jitter to prevent thundering herd
    rate_limit_per_minute: Optional[int] = None  # Requests per minute limit
    rate_limit_per_second: Optional[int] = None  # Requests per second limit


class RateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, requests_per_second: Optional[float] = None, requests_per_minute: Optional[float] = None):
        self.requests_per_second = requests_per_second
        self.requests_per_minute = requests_per_minute
        self._last_request_time: Optional[float] = None
        self._request_count_minute = 0
        self._minute_start: Optional[float] = None
        self._lock = None  # Would use threading.Lock in production
    
    def acquire(self) -> bool:
        """Acquire permission to make a request. Returns True if allowed."""
        now = time.time()
        
        # Per-second rate limiting
        if self.requests_per_second:
            if self._last_request_time:
                elapsed = now - self._last_request_time
                min_interval = 1.0 / self.requests_per_second
                if elapsed < min_interval:
                    sleep_time = min_interval - elapsed
                    logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
        
        # Per-minute rate limiting
        if self.requests_per_minute:
            if self._minute_start is None or now - self._minute_start >= 60:
                self._minute_start = now
                self._request_count_minute = 0
            
            if self._request_count_minute >= self.requests_per_minute:
                sleep_time = 60 - (now - self._minute_start)
                if sleep_time > 0:
                    logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s (minute limit)")
                    time.sleep(sleep_time)
                # Reset after waiting
                self._minute_start = time.time()
                self._request_count_minute = 0
            
            self._request_count_minute += 1
        
        self._last_request_time = time.time()
        return True


class RetryWithBackoff:
    """Decorator for retrying functions with exponential backoff."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self.rate_limiter = RateLimiter(
            requests_per_second=self.config.rate_limit_per_second,
            requests_per_minute=self.config.rate_limit_per_minute,
        )
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(self.config.max_retries + 1):
                # Apply rate limiting
                self.rate_limiter.acquire()
                
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Don't retry on certain errors
                    if self._is_non_retryable(e):
                        raise
                    
                    if attempt < self.config.max_retries:
                        delay = self._calculate_delay(attempt)
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}. "
                            f"Retrying in {delay:.2f}s..."
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {self.config.max_retries + 1} attempts")
            
            # All retries exhausted
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected state: no exception but all retries exhausted")
        
        return wrapper
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the next retry attempt."""
        if self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.exponential_base ** attempt)
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * (attempt + 1)
        else:  # FIXED
            delay = self.config.base_delay
        
        # Cap at max delay
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random())  # 0.5x to 1.5x
        
        return delay
    
    def _is_non_retryable(self, exception: Exception) -> bool:
        """Check if an exception should not trigger a retry."""
        # Don't retry on authentication errors
        error_str = str(exception).lower()
        non_retryable = [
            "authentication",
            "unauthorized",
            "api key invalid",
            "permission denied",
            "not found",  # 404s usually won't fix themselves
        ]
        return any(err in error_str for err in non_retryable)


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    rate_limit_per_minute: Optional[int] = None,
):
    """Decorator factory for adding retry logic to functions.
    
    Example:
        >>> @with_retry(max_retries=3, base_delay=1.0)
        ... def call_api():
        ...     return requests.get("https://api.example.com/data")
    """
    config = RateLimitConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        exponential_base=exponential_base,
        jitter=jitter,
        rate_limit_per_minute=rate_limit_per_minute,
    )
    return RetryWithBackoff(config)


class CircuitBreaker:
    """Circuit breaker pattern for failing services."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self._failures = 0
        self._last_failure_time: Optional[float] = None
        self._state = "closed"  # closed, open, half-open
    
    def can_execute(self) -> bool:
        """Check if operation is allowed."""
        if self._state == "closed":
            return True
        
        if self._state == "open":
            if self._last_failure_time and time.time() - self._last_failure_time >= self.recovery_timeout:
                self._state = "half-open"
                return True
            return False
        
        return True  # half-open
    
    def record_success(self) -> None:
        """Record a successful execution."""
        self._failures = 0
        self._state = "closed"
    
    def record_failure(self) -> None:
        """Record a failed execution."""
        self._failures += 1
        self._last_failure_time = time.time()
        
        if self._failures >= self.failure_threshold:
            self._state = "open"
            logger.warning(f"Circuit breaker opened after {self._failures} failures")
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            if not self.can_execute():
                raise RuntimeError("Circuit breaker is open - service unavailable")
            
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except self.expected_exception as e:
                self.record_failure()
                raise
        
        return wrapper


# Provider-specific rate limit configurations
PROVIDER_RATE_LIMITS = {
    "openai": RateLimitConfig(
        max_retries=3,
        base_delay=1.0,
        rate_limit_per_minute=60,  # 60 RPM for most tiers
    ),
    "anthropic": RateLimitConfig(
        max_retries=3,
        base_delay=1.0,
        rate_limit_per_minute=40,  # 40 RPM for most tiers
    ),
    "gemini": RateLimitConfig(
        max_retries=3,
        base_delay=1.0,
        rate_limit_per_minute=60,
    ),
    "openrouter": RateLimitConfig(
        max_retries=3,
        base_delay=1.0,
        rate_limit_per_minute=120,
    ),
    "ollama": RateLimitConfig(
        max_retries=2,
        base_delay=0.5,
        rate_limit_per_second=10,  # Local inference is faster
    ),
    "vllm": RateLimitConfig(
        max_retries=2,
        base_delay=0.5,
        rate_limit_per_second=20,  # Self-hosted, can be aggressive
    ),
}


def get_rate_limiter(provider: str) -> RetryWithBackoff:
    """Get rate limiter configuration for a specific provider."""
    config = PROVIDER_RATE_LIMITS.get(provider, RateLimitConfig())
    return RetryWithBackoff(config)
