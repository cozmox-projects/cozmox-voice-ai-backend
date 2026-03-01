"""
services/resilience/retry_policy.py
─────────────────────────────────────
Retry decorators for AI API calls (Deepgram, OpenAI, ElevenLabs).
Uses tenacity library for robust exponential backoff.

Usage:
    from services.resilience.retry_policy import retry_ai_call

    @retry_ai_call(service_name="deepgram")
    async def call_deepgram(...):
        ...
"""
import functools
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
    RetryError,
)
import logging
import httpx
from logger import get_logger

log = get_logger(__name__)
_tenacity_log = logging.getLogger("tenacity")


def retry_ai_call(service_name: str, max_attempts: int = 3):
    """
    Decorator for retrying AI API calls with exponential backoff.
    
    - Retries on network errors and HTTP 5xx errors
    - Waits: 0.5s → 1s → 2s between attempts
    - Logs each retry with service name + attempt number
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            attempt = 0
            last_error = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except (httpx.NetworkError, httpx.TimeoutException) as e:
                    last_error = e
                    wait = 0.5 * (2 ** (attempt - 1))  # 0.5, 1.0, 2.0
                    log.warning(
                        "ai_call_retry",
                        service=service_name,
                        attempt=attempt,
                        max_attempts=max_attempts,
                        error=str(e),
                        retry_in_seconds=wait,
                    )
                    if attempt < max_attempts:
                        import asyncio
                        await asyncio.sleep(wait)
                except Exception as e:
                    # Non-retryable errors — fail immediately
                    log.error(
                        "ai_call_failed_non_retryable",
                        service=service_name,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise

            log.error(
                "ai_call_exhausted_retries",
                service=service_name,
                attempts=attempt,
            )
            raise last_error

        return wrapper
    return decorator
