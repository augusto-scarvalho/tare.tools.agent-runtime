"""Minimal cross-boundary contracts selected by the ADR-057 F0 receipt."""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from enum import Enum
from threading import Event
from typing import Callable, Protocol, runtime_checkable

from .grounding import GroundingEnvelope
from .prompt import PromptEnvelope
from .work import ProgressSignal, WorkEnvelope


def _require_text(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


@dataclass(frozen=True)
class BindingRef:
    """Opaque reference to authority owned and fenced by tare.tools.os."""

    work_id: str
    binding_id: str
    owner_id: str
    fence: int
    token: str = field(repr=False)

    def __post_init__(self) -> None:
        _require_text(self.work_id, "work_id")
        _require_text(self.binding_id, "binding_id")
        _require_text(self.owner_id, "owner_id")
        _require_text(self.token, "token")
        if self.fence < 1:
            raise ValueError("fence must be a positive monotonic integer")


@dataclass(frozen=True)
class RunBudget:
    max_turns: int | None = None
    max_retries_per_turn: int | None = 0
    wall_clock_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_turns is not None and (type(self.max_turns) is not int or self.max_turns < 1):
            raise ValueError("max_turns must be positive")
        if self.max_retries_per_turn is not None and (type(self.max_retries_per_turn) is not int or self.max_retries_per_turn < 0):
            raise ValueError("max_retries_per_turn cannot be negative")
        if self.wall_clock_seconds is not None and (isinstance(self.wall_clock_seconds, bool) or not isinstance(self.wall_clock_seconds, (int, float)) or not math.isfinite(self.wall_clock_seconds) or self.wall_clock_seconds <= 0):
            raise ValueError("wall_clock_seconds must be positive")


class CancellationToken:
    """Thread-safe cooperative cancellation signal supplied by the host."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


@dataclass(frozen=True)
class ExecutionControl:
    """Live cancellation and deadline information propagated to the adapter."""

    binding: BindingRef
    cancellation: CancellationToken
    deadline_monotonic: float
    monotonic: Callable[[], float] = field(default=time.monotonic, repr=False, compare=False)
    binding_is_current: Callable[[], bool] = field(
        default=lambda: True,
        repr=False,
        compare=False,
    )

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_monotonic - self.monotonic())

    def should_stop(self) -> bool:
        return (
            self.cancellation.is_cancelled()
            or self.remaining_seconds() <= 0
            or not self.binding_is_current()
        )


class TurnStatus(str, Enum):
    CONTINUE = "CONTINUE"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    RETRYABLE_FAILURE = "RETRYABLE_FAILURE"
    BLOCKED = "BLOCKED"


class OutcomeStatus(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    EXHAUSTED = "EXHAUSTED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class TurnRequest:
    binding: BindingRef
    instruction: str = field(repr=False)
    turn_number: int
    attempt_number: int
    tactical_context: tuple[str, ...]
    prompt: PromptEnvelope | None = field(default=None, repr=False)
    grounding: GroundingEnvelope | None = field(default=None, repr=False)
    groundings: tuple[GroundingEnvelope, ...] = field(default=(), repr=False)
    work: WorkEnvelope | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        envelope = self.prompt or PromptEnvelope(self.instruction)
        if envelope.text != self.instruction:
            raise ValueError("instruction must match the admitted prompt envelope")
        if self.grounding is not None and not isinstance(
            self.grounding, GroundingEnvelope
        ):
            raise TypeError("grounding must be GroundingEnvelope")
        if not isinstance(self.groundings, tuple) or not all(
            isinstance(item, GroundingEnvelope) for item in self.groundings
        ):
            raise TypeError("groundings must be a tuple of GroundingEnvelope")
        if self.grounding is not None and self.groundings not in (
            (),
            (self.grounding,),
        ):
            raise ValueError("singular grounding must match admitted groundings")
        if self.work is not None and not isinstance(self.work, WorkEnvelope):
            raise TypeError("work must be WorkEnvelope")
        object.__setattr__(self, "prompt", envelope)


@dataclass(frozen=True)
class TurnResult:
    status: TurnStatus
    output: str = ""
    context_delta: tuple[str, ...] = ()
    progress: ProgressSignal | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if self.progress is not None and not isinstance(self.progress, ProgressSignal):
            raise TypeError("progress must be ProgressSignal")
        if self.status is TurnStatus.BLOCKED:
            if self.reason_code is None:
                raise ValueError("blocked turns require a reason_code")
            _require_text(self.reason_code, "reason_code")
        elif self.reason_code is not None:
            raise ValueError("reason_code is reserved for blocked turns")


@dataclass(frozen=True)
class TerminalClaim:
    binding: BindingRef
    status: OutcomeStatus
    output: str
    turns: int
    attempts: int


@dataclass(frozen=True)
class RuntimeOutcome:
    status: OutcomeStatus
    output: str
    turns: int
    attempts: int
    claim_accepted: bool
    reason_code: str | None = None
    prompt_bytes: int | None = None
    prompt_sha256: str | None = None


@runtime_checkable
class TurnExecutor(Protocol):
    def execute(self, request: TurnRequest, control: ExecutionControl) -> TurnResult:
        """Execute one bounded turn and propagate control to mediated effects."""


@runtime_checkable
class AuthorityGate(Protocol):
    def is_current(self, binding: BindingRef) -> bool:
        """Return whether this exact fenced binding may still emit canonically."""

    def submit_terminal_claim(self, claim: TerminalClaim) -> bool:
        """Atomically accept only a claim from the current fenced binding."""
