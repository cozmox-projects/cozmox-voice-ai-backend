"""
services/resilience/circuit_breaker.py
────────────────────────────────────────
Circuit breaker for Azure OpenAI (the most likely service to get overloaded).

States:
  CLOSED  → Normal operation. Requests go through.
  OPEN    → Too many failures. Requests blocked, fallback returned immediately.
  HALF    → Recovery probe. One test request allowed through.

Thresholds:
  - Opens after 5 failures in 30 seconds
  - Tries recovery after 60 seconds
"""
import time
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Optional
from logger import get_logger
from services.metrics.collector import llm_errors_total

log = get_logger(__name__)

FALLBACK_RESPONSE = (
    "I'm having a little trouble right now. Could you give me just a moment? "
    "I'll be right with you."
)


class CircuitState(Enum):
    CLOSED = "closed"       # Normal — requests flow through
    OPEN = "open"           # Failing — requests blocked
    HALF_OPEN = "half_open" # Testing recovery


@dataclass
class CircuitBreaker:
    service_name: str
    failure_threshold: int = 5       # failures before opening
    window_seconds: int = 30         # window to count failures in
    recovery_timeout: int = 60       # seconds before trying half-open

    state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    failure_count: int = field(default=0, init=False)
    last_failure_time: float = field(default=0.0, init=False)
    opened_at: float = field(default=0.0, init=False)

    def _reset_if_window_expired(self):
        """Reset counter if we're outside the failure window."""
        if time.time() - self.last_failure_time > self.window_seconds:
            self.failure_count = 0

    def record_success(self):
        if self.state == CircuitState.HALF_OPEN:
            log.info("circuit_breaker_recovered", service=self.service_name)
            self.state = CircuitState.CLOSED
            self.failure_count = 0

    def record_failure(self):
        self._reset_if_window_expired()
        self.failure_count += 1
        self.last_failure_time = time.time()
        llm_errors_total.inc()

        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                log.error(
                    "circuit_breaker_opened",
                    service=self.service_name,
                    failures=self.failure_count,
                )
                self.state = CircuitState.OPEN
                self.opened_at = time.time()

    def allow_request(self) -> bool:
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if time.time() - self.opened_at > self.recovery_timeout:
                log.info("circuit_breaker_half_open", service=self.service_name)
                self.state = CircuitState.HALF_OPEN
                return True  # Allow one probe request
            return False  # Still open, block request

        if self.state == CircuitState.HALF_OPEN:
            return True  # Allow probe

        return True

    async def call(self, func: Callable, *args, **kwargs):
        """
        Wraps an async function call with circuit breaker logic.
        
        Usage:
            result = await breaker.call(azure_openai_fn, prompt=prompt)
        """
        if not self.allow_request():
            log.warning(
                "circuit_breaker_blocked_request",
                service=self.service_name,
                state=self.state.value,
            )
            return FALLBACK_RESPONSE

        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            log.error(
                "circuit_breaker_call_failed",
                service=self.service_name,
                error=str(e),
                new_failure_count=self.failure_count,
                state=self.state.value,
            )
            raise


# Singleton breaker for Azure OpenAI
llm_circuit_breaker = CircuitBreaker(
    service_name="azure_openai",
    failure_threshold=5,
    window_seconds=30,
    recovery_timeout=60,
)
