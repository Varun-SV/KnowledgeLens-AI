from __future__ import annotations

from dataclasses import dataclass

MAX_CONSECUTIVE_REQUEST_FAILURES = 3


@dataclass(slots=True)
class RequestFailureCircuit:
    """Bound repeated request failures while allowing recovery after a success."""

    limit: int = MAX_CONSECUTIVE_REQUEST_FAILURES
    consecutive_failures: int = 0

    def __post_init__(self) -> None:
        if self.limit < 1:
            raise ValueError("Request failure circuit limit must be positive.")

    def record_success(self) -> None:
        self.consecutive_failures = 0

    def record_failure(self) -> bool:
        """Record one failed request and return whether the circuit is now open."""
        self.consecutive_failures += 1
        return self.consecutive_failures >= self.limit
