"""Deterministic tactical agent-loop state machine."""

from __future__ import annotations

import time
from itertools import count
from typing import Callable

from .contracts import (
    AuthorityGate,
    BindingRef,
    CancellationToken,
    ExecutionControl,
    OutcomeStatus,
    RunBudget,
    RuntimeOutcome,
    TerminalClaim,
    TurnExecutor,
    TurnRequest,
    TurnResult,
    TurnStatus,
)
from .prompt import (
    PromptAdmission,
    PromptBudgetGate,
    PromptDecision,
    PromptEnvelope,
    PromptGate,
)
from .grounding import GroundingEnvelope
from .work import (
    ProgressSignal,
    StrictWorkControl,
    WorkEnvelope,
    WorkGuard,
    WorkGuardDecision,
    WorkGuardResult,
)


class AgentRuntime:
    """Advance tactical turns without owning Work or execution infrastructure."""

    def __init__(
        self,
        turn_executor: TurnExecutor,
        authority_gate: AuthorityGate,
        *,
        prompt_gate: PromptGate | None = None,
        work_guard: WorkGuard | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._turn_executor = turn_executor
        self._authority_gate = authority_gate
        self._prompt_gate = PromptBudgetGate() if prompt_gate is None else prompt_gate
        if not isinstance(self._prompt_gate, PromptGate):
            raise TypeError("prompt_gate must implement PromptGate")
        if work_guard is not None and not isinstance(work_guard, WorkGuard):
            raise TypeError("work_guard must implement WorkGuard")
        self._work_guard = work_guard
        self._monotonic = monotonic

    def run(
        self,
        *,
        binding: BindingRef,
        instruction: str,
        budget: RunBudget | None = None,
        cancellation: CancellationToken | None = None,
        grounding: GroundingEnvelope | None = None,
        groundings: tuple[GroundingEnvelope, ...] = (),
        work: WorkEnvelope | None = None,
        strict_work: StrictWorkControl | None = None,
    ) -> RuntimeOutcome:
        if grounding is not None and not isinstance(grounding, GroundingEnvelope):
            raise TypeError("grounding must be GroundingEnvelope")
        if not isinstance(groundings, tuple) or not all(
            isinstance(item, GroundingEnvelope) for item in groundings
        ):
            raise TypeError("groundings must be a tuple of GroundingEnvelope")
        if grounding is not None and groundings:
            raise ValueError("use grounding or groundings, not both")
        admitted_groundings = groundings or ((grounding,) if grounding is not None else ())
        admitted_groundings = tuple(
            sorted(admitted_groundings, key=lambda item: (item.repository.casefold(), item.repository))
        )
        if work is not None and not isinstance(work, WorkEnvelope):
            raise TypeError("work must be WorkEnvelope")
        if strict_work is not None and not isinstance(strict_work, StrictWorkControl):
            raise TypeError("strict_work must be StrictWorkControl")
        if strict_work is not None and work is None:
            return self._blocked_control("WORK_CONTEXT_REQUIRED", 0, 0, "")
        if strict_work is not None and self._work_guard is None:
            return self._blocked_control("WORK_GUARD_REQUIRED", 0, 0, "")
        context_error = self._managed_context_error(work, admitted_groundings, strict_work)
        if context_error is not None:
            return self._blocked_control(context_error, 0, 0, "")
        budget = budget or RunBudget()
        cancel = cancellation or CancellationToken()
        deadline = (self._monotonic() + budget.wall_clock_seconds
                    if budget.wall_clock_seconds is not None else float("inf"))
        control = ExecutionControl(
            binding=binding,
            cancellation=cancel,
            deadline_monotonic=deadline,
            monotonic=self._monotonic,
            binding_is_current=lambda: self._authority_gate.is_current(binding),
        )
        context: tuple[str, ...] = ()
        attempts = 0
        completed_turns = 0
        last_output = ""
        previous_progress: ProgressSignal | None = None
        stagnant_turns = 0

        stopped = self._stop_outcome(
            control,
            completed_turns,
            attempts,
            last_output,
            work,
            strict_work,
        )
        if stopped is not None:
            return stopped
        if not self._authority_gate.is_current(binding):
            return self._rejected(completed_turns, attempts, last_output)

        prompt, admission = self._admit_prompt(instruction)
        if admission.decision is PromptDecision.DENY:
            return self._blocked(admission, prompt)

        for turn_number in count(1):
            if budget.max_turns is not None and turn_number > budget.max_turns:
                break
            retry_number = 0
            while True:
                stopped = self._stop_outcome(
                    control,
                    completed_turns,
                    attempts,
                    last_output,
                    work,
                    strict_work,
                )
                if stopped is not None:
                    return stopped
                if not self._authority_gate.is_current(binding):
                    return self._rejected(completed_turns, attempts, last_output)
                guarded = self._guard_work(
                    work, strict_work, completed_turns, attempts, last_output
                )
                if guarded is not None:
                    return guarded

                attempts += 1
                request = TurnRequest(
                    binding=binding,
                    instruction=instruction,
                    turn_number=turn_number,
                    attempt_number=attempts,
                    tactical_context=context,
                    prompt=prompt,
                    grounding=admitted_groundings[0] if len(admitted_groundings) == 1 else None,
                    groundings=admitted_groundings,
                    work=work,
                )
                result = self._turn_executor.execute(request, control)
                if not isinstance(result, TurnResult):
                    raise TypeError("TurnExecutor.execute must return TurnResult")

                stopped = self._stop_outcome(
                    control,
                    completed_turns,
                    attempts,
                    result.output,
                    work,
                    strict_work,
                )
                if stopped is not None:
                    return stopped
                if not self._authority_gate.is_current(binding):
                    return self._rejected(completed_turns, attempts, result.output)
                guarded = self._guard_work(
                    work, strict_work, completed_turns, attempts, result.output
                )
                if guarded is not None:
                    return guarded

                if result.status is TurnStatus.BLOCKED:
                    return self._blocked_control(
                        result.reason_code or "TURN_EXECUTION_BLOCKED",
                        completed_turns,
                        attempts,
                        result.output,
                    )

                if result.status is TurnStatus.RETRYABLE_FAILURE:
                    if budget.max_retries_per_turn is None or retry_number < budget.max_retries_per_turn:
                        retry_number += 1
                        continue
                    return self._finish(
                        binding, OutcomeStatus.EXHAUSTED, result.output,
                        completed_turns, attempts, control, work, strict_work,
                    )

                completed_turns = turn_number
                last_output = result.output
                context += result.context_delta

                if strict_work is not None and result.status in {
                    TurnStatus.CONTINUE,
                    TurnStatus.SUCCEEDED,
                }:
                    if result.progress is None:
                        return self._blocked_control(
                            "PROGRESS_SIGNAL_REQUIRED",
                            completed_turns,
                            attempts,
                            result.output,
                        )
                    if previous_progress is not None:
                        if result.progress.regresses(previous_progress):
                            return self._blocked_control(
                                "PROGRESS_REGRESSION",
                                completed_turns,
                                attempts,
                                result.output,
                            )
                        if result.progress.advances(previous_progress):
                            stagnant_turns = 0
                        else:
                            stagnant_turns += 1
                            if stagnant_turns >= strict_work.max_stagnant_turns:
                                return self._blocked_control(
                                    "STAGNATION_DETECTED",
                                    completed_turns,
                                    attempts,
                                    result.output,
                                )
                    previous_progress = result.progress
                    if (
                        result.status is TurnStatus.SUCCEEDED
                        and result.progress.remaining_obligations != 0
                    ):
                        return self._blocked_control(
                            "EXIT_CRITERIA_UNSATISFIED",
                            completed_turns,
                            attempts,
                            result.output,
                        )

                if result.status is TurnStatus.CONTINUE:
                    break
                if result.status is TurnStatus.SUCCEEDED:
                    return self._finish(
                        binding, OutcomeStatus.SUCCEEDED, result.output,
                        completed_turns, attempts, control, work, strict_work,
                    )
                if result.status is TurnStatus.FAILED:
                    return self._finish(
                        binding, OutcomeStatus.FAILED, result.output,
                        completed_turns, attempts, control, work, strict_work,
                    )
                raise ValueError(f"unsupported turn status: {result.status!r}")

        return self._finish(
            binding, OutcomeStatus.EXHAUSTED, last_output,
            completed_turns, attempts, control, work, strict_work,
        )

    def _stop_outcome(
        self,
        control: ExecutionControl,
        turns: int,
        attempts: int,
        output: str,
        work: WorkEnvelope | None,
        strict_work: StrictWorkControl | None,
    ) -> RuntimeOutcome | None:
        if control.cancellation.is_cancelled():
            return RuntimeOutcome(OutcomeStatus.CANCELLED, output, turns, attempts, False)
        if control.remaining_seconds() <= 0:
            return self._finish(
                control.binding, OutcomeStatus.EXHAUSTED, output,
                turns, attempts, control, work, strict_work,
            )
        return None

    def _finish(
        self,
        binding: BindingRef,
        status: OutcomeStatus,
        output: str,
        turns: int,
        attempts: int,
        control: ExecutionControl,
        work: WorkEnvelope | None,
        strict_work: StrictWorkControl | None,
    ) -> RuntimeOutcome:
        if control.cancellation.is_cancelled():
            return RuntimeOutcome(OutcomeStatus.CANCELLED, output, turns, attempts, False)
        if not self._authority_gate.is_current(binding):
            return self._rejected(turns, attempts, output)
        guarded = self._guard_work(work, strict_work, turns, attempts, output)
        if guarded is not None:
            return guarded
        claim = TerminalClaim(binding, status, output, turns, attempts)
        accepted = self._authority_gate.submit_terminal_claim(claim)
        if not accepted:
            return self._rejected(turns, attempts, output)
        return RuntimeOutcome(status, output, turns, attempts, True)

    def _guard_work(
        self,
        work: WorkEnvelope | None,
        strict_work: StrictWorkControl | None,
        turns: int,
        attempts: int,
        output: str,
    ) -> RuntimeOutcome | None:
        if strict_work is None:
            return None
        if work is None or self._work_guard is None:
            return self._blocked_control("WORK_GUARD_REQUIRED", turns, attempts, output)
        try:
            result = self._work_guard.evaluate(work)
        except Exception:
            return self._blocked_control("WORK_GUARD_ERROR", turns, attempts, output)
        if not isinstance(result, WorkGuardResult):
            return self._blocked_control("WORK_GUARD_ERROR", turns, attempts, output)
        if result.decision is WorkGuardDecision.CURRENT:
            if (
                result.observed_scope_sha256 is not None
                and result.observed_scope_sha256 != work.execution_scope_sha256
            ):
                return self._blocked_control("WORK_ITEM_DRIFT", turns, attempts, output)
            return None
        if result.decision is WorkGuardDecision.DRIFTED:
            reason = (
                "GROUNDING_CONTEXT_DRIFT"
                if result.reason_code.startswith("GROUNDING_")
                else "WORK_ITEM_DRIFT"
            )
            return self._blocked_control(reason, turns, attempts, output)
        reason = (
            "GROUNDING_CONTEXT_UNAVAILABLE"
            if result.reason_code.startswith("GROUNDING_")
            else "WORK_GUARD_UNAVAILABLE"
        )
        return self._blocked_control(reason, turns, attempts, output)

    @staticmethod
    def _managed_context_error(
        work: WorkEnvelope | None,
        groundings: tuple[GroundingEnvelope, ...],
        strict_work: StrictWorkControl | None,
    ) -> str | None:
        if strict_work is None or work is None:
            return None
        selected = bool(work.repository_scopes)
        if selected and not groundings:
            return "GROUNDING_CONTEXT_REQUIRED"
        if not groundings:
            return None
        if not selected:
            return "GROUNDING_CONTEXT_UNBOUND"
        if len({item.repository for item in groundings}) != len(groundings):
            return "GROUNDING_CONTEXT_MISMATCH"
        by_repository = {item.repository: item for item in groundings}
        if set(by_repository) != {scope.repository for scope in work.repository_scopes}:
            return "GROUNDING_CONTEXT_MISMATCH"
        for scope in work.repository_scopes:
            if by_repository[scope.repository].requested_ids != scope.grounding_refs:
                return "GROUNDING_CONTEXT_MISMATCH"
        return None

    @staticmethod
    def _rejected(turns: int, attempts: int, output: str) -> RuntimeOutcome:
        return RuntimeOutcome(OutcomeStatus.REJECTED, output, turns, attempts, False)

    @staticmethod
    def _blocked_control(
        reason_code: str,
        turns: int,
        attempts: int,
        output: str,
    ) -> RuntimeOutcome:
        return RuntimeOutcome(
            OutcomeStatus.BLOCKED,
            output,
            turns,
            attempts,
            False,
            reason_code=reason_code,
        )

    def _admit_prompt(
        self,
        instruction: str,
    ) -> tuple[PromptEnvelope | None, PromptAdmission]:
        try:
            prompt = PromptEnvelope(instruction)
        except (TypeError, UnicodeError):
            return None, PromptAdmission(
                PromptDecision.DENY,
                "INVALID_PROMPT",
                0,
                "0" * 64,
            )
        try:
            admission = self._prompt_gate.evaluate(prompt)
        except Exception:
            return prompt, self._deny_prompt(prompt, "PROMPT_GATE_ERROR")
        if not isinstance(admission, PromptAdmission):
            return prompt, self._deny_prompt(prompt, "INVALID_PROMPT_GATE_RESULT")
        if (
            admission.prompt_bytes != prompt.byte_count
            or admission.prompt_sha256 != prompt.sha256
        ):
            return prompt, self._deny_prompt(prompt, "INVALID_PROMPT_GATE_RESULT")
        return prompt, admission

    @staticmethod
    def _deny_prompt(prompt: PromptEnvelope, reason_code: str) -> PromptAdmission:
        return PromptAdmission(
            PromptDecision.DENY,
            reason_code,
            prompt.byte_count,
            prompt.sha256,
        )

    @staticmethod
    def _blocked(
        admission: PromptAdmission,
        prompt: PromptEnvelope | None,
    ) -> RuntimeOutcome:
        return RuntimeOutcome(
            OutcomeStatus.BLOCKED,
            "",
            0,
            0,
            False,
            reason_code=admission.reason_code,
            prompt_bytes=prompt.byte_count if prompt is not None else None,
            prompt_sha256=prompt.sha256 if prompt is not None else None,
        )
