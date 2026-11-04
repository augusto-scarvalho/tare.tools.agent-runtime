from __future__ import annotations

import hashlib

import pytest

from tare_tools_agent_runtime import (
    DEFAULT_MAX_PROMPT_BYTES,
    PromptBudgetGate,
    PromptDecision,
    PromptEnvelope,
)


def test_prompt_envelope_materializes_utf8_once_and_hides_content_from_repr() -> None:
    prompt = PromptEnvelope("ação")

    assert prompt.canonical_bytes == "ação".encode("utf-8")
    assert prompt.byte_count == 6
    assert prompt.sha256 == hashlib.sha256(prompt.canonical_bytes).hexdigest()
    assert "ação" not in repr(prompt)


def test_prompt_budget_is_utf8_byte_exact_at_the_boundary() -> None:
    gate = PromptBudgetGate(max_prompt_bytes=4)

    exact = gate.evaluate(PromptEnvelope("éé"))
    oversized = gate.evaluate(PromptEnvelope("ééé"))

    assert exact.decision is PromptDecision.ALLOW
    assert exact.prompt_bytes == 4
    assert oversized.decision is PromptDecision.DENY
    assert oversized.reason_code == "PROMPT_TOO_LARGE"
    assert oversized.prompt_bytes == 6


def test_prompt_budget_denies_empty_and_whitespace_only_prompts() -> None:
    gate = PromptBudgetGate()

    for value in ("", " \t\r\n"):
        outcome = gate.evaluate(PromptEnvelope(value))
        assert outcome.decision is PromptDecision.DENY
        assert outcome.reason_code == "EMPTY_PROMPT"


def test_default_prompt_budget_constant_is_bounded_and_exact() -> None:
    gate = PromptBudgetGate()

    assert gate.max_prompt_bytes == DEFAULT_MAX_PROMPT_BYTES
    assert gate.evaluate(PromptEnvelope("x" * DEFAULT_MAX_PROMPT_BYTES)).decision is PromptDecision.ALLOW
    assert gate.evaluate(PromptEnvelope("x" * (DEFAULT_MAX_PROMPT_BYTES + 1))).decision is PromptDecision.DENY


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_prompt_budget_rejects_invalid_limits(value: object) -> None:
    with pytest.raises(ValueError, match="max_prompt_bytes"):
        PromptBudgetGate(value)  # type: ignore[arg-type]
