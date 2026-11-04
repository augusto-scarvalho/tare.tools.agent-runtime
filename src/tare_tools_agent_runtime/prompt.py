"""Provider-neutral prompt admission at the agent-runtime ingress boundary."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


DEFAULT_MAX_PROMPT_BYTES = 16_384


class PromptDecision(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


@dataclass(frozen=True, init=False)
class PromptEnvelope:
    """One immutable UTF-8 materialization shared by policy and provider ports."""

    text: str = field(repr=False)
    canonical_bytes: bytes = field(repr=False)
    sha256: str

    def __init__(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError("prompt must be a string")
        canonical_bytes = text.encode("utf-8", errors="strict")
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "canonical_bytes", canonical_bytes)
        object.__setattr__(self, "sha256", hashlib.sha256(canonical_bytes).hexdigest())

    @property
    def byte_count(self) -> int:
        return len(self.canonical_bytes)


@dataclass(frozen=True)
class PromptAdmission:
    decision: PromptDecision
    reason_code: str
    prompt_bytes: int
    prompt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.decision, PromptDecision):
            raise TypeError("decision must be PromptDecision")
        if not isinstance(self.reason_code, str) or not self.reason_code.strip():
            raise ValueError("reason_code must be non-empty")
        if (
            not isinstance(self.prompt_bytes, int)
            or isinstance(self.prompt_bytes, bool)
            or self.prompt_bytes < 0
        ):
            raise ValueError("prompt_bytes cannot be negative")
        if (
            not isinstance(self.prompt_sha256, str)
            or len(self.prompt_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.prompt_sha256)
        ):
            raise ValueError("prompt_sha256 must be a lowercase SHA-256 digest")


@runtime_checkable
class PromptGate(Protocol):
    def evaluate(self, prompt: PromptEnvelope) -> PromptAdmission:
        """Return a decision without mutating or replacing the prompt."""


@dataclass(frozen=True)
class PromptBudgetGate:
    """Pure fail-closed size policy applied before every provider adapter."""

    max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_prompt_bytes, int)
            or isinstance(self.max_prompt_bytes, bool)
            or self.max_prompt_bytes < 1
        ):
            raise ValueError("max_prompt_bytes must be positive")

    def evaluate(self, prompt: PromptEnvelope) -> PromptAdmission:
        if not isinstance(prompt, PromptEnvelope):
            raise TypeError("prompt must be PromptEnvelope")
        if not prompt.text.strip():
            return self._decision(prompt, PromptDecision.DENY, "EMPTY_PROMPT")
        if prompt.byte_count > self.max_prompt_bytes:
            return self._decision(prompt, PromptDecision.DENY, "PROMPT_TOO_LARGE")
        return self._decision(prompt, PromptDecision.ALLOW, "WITHIN_PROMPT_BUDGET")

    @staticmethod
    def _decision(
        prompt: PromptEnvelope,
        decision: PromptDecision,
        reason_code: str,
    ) -> PromptAdmission:
        return PromptAdmission(
            decision=decision,
            reason_code=reason_code,
            prompt_bytes=prompt.byte_count,
            prompt_sha256=prompt.sha256,
        )
