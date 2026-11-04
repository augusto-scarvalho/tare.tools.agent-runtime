from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from tare_tools_agent_runtime import (
    AgentRuntime,
    BindingRef,
    CancellationToken,
    GroundingEnvelope,
    OutcomeStatus,
    PromptAdmission,
    PromptBudgetGate,
    PromptDecision,
    PromptEnvelope,
    ProgressSignal,
    RunBudget,
    StrictWorkControl,
    TerminalClaim,
    TurnRequest,
    TurnResult,
    TurnStatus,
    WorkEnvelope,
    WorkGuardDecision,
    WorkGuardResult,
)


def grounding(
    *,
    repository: str = "tare.tools.kernel",
    requested_ids: tuple[str, ...] = (),
) -> GroundingEnvelope:
    payload = {
        "byte_count": 0,
        "entries": [],
        "grounding_scope_sha256": "d" * 64,
        "manifest_sha256": "a" * 64,
        "manifest_version": "2.1.0",
        "matched_count": 0,
        "omitted_count": 0,
        "reason_code": "NO_TARGET_MAPPING",
        "repository": repository,
        "requested_ids": list(requested_ids),
        "returned_count": 0,
        "schema": "tare.tools/repository-grounding/1",
        "status": "NOT_MAPPED",
    }
    import json

    for _ in range(8):
        raw = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if payload["byte_count"] == len(raw):
            return GroundingEnvelope(raw)
        payload["byte_count"] = len(raw)
    raise AssertionError("fixture byte count did not converge")


def work(
    *,
    target_repositories: tuple[str, ...] = (),
    grounding_refs: tuple[str, ...] = (),
    repository_scopes: tuple[dict[str, object], ...] | None = None,
) -> WorkEnvelope:
    import json

    payload = {
        "authority": "NONE / READ_ONLY PROJECTION",
        "byte_count": 0,
        "schema": "tare.tools/work-grounding/1",
        "downstream": [],
        "execution_scope_sha256": "c" * 64,
        "feasibility": {
            "ready": True,
            "reasons": [],
            "unresolved_prerequisites": [],
        },
        "graph_revision": "projection-7",
        "graph_sha256": "a" * 64,
        "omitted_downstream": 0,
        "profile": "operational",
        "provenance": {"source_claim_ids": [], "source_refs": []},
        "reason_codes": ["READY_FOR_MANAGED_EXECUTION"],
        "status": "READY",
        "work": {
            "id": "WORK-1",
            "exit_criteria": ["Tests pass"],
            "target_repositories": list(target_repositories),
            "grounding_refs": list(grounding_refs),
            **(
                {"repository_scopes": list(repository_scopes)}
                if repository_scopes is not None
                else {}
            ),
        },
        "work_id": "WORK-1",
        "work_item_sha256": "b" * 64,
    }
    for _ in range(8):
        raw = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if payload["byte_count"] == len(raw):
            return WorkEnvelope(raw)
        payload["byte_count"] = len(raw)
    raise AssertionError("fixture byte count did not converge")


def progress(remaining: int, evidence: int) -> ProgressSignal:
    return ProgressSignal(
        state_sha256=f"{remaining + 1:064x}",
        evidence_sha256=f"{evidence + 100:064x}",
        remaining_obligations=remaining,
        verified_evidence_count=evidence,
    )


class SequenceWorkGuard:
    def __init__(self, *decisions: WorkGuardDecision) -> None:
        self.decisions = list(decisions)
        self.calls = 0

    def evaluate(self, envelope: WorkEnvelope) -> WorkGuardResult:
        self.calls += 1
        decision = self.decisions.pop(0) if self.decisions else WorkGuardDecision.CURRENT
        return WorkGuardResult(
            decision,
            f"TEST_{decision.value}",
            envelope.execution_scope_sha256
            if decision is WorkGuardDecision.CURRENT
            else None,
        )


def binding(fence: int = 1) -> BindingRef:
    return BindingRef("WORK-1", f"binding-{fence}", "worker-a", fence, "secret-token")


@dataclass
class FakeAuthority:
    current_fence: int = 1
    claims: list[TerminalClaim] = field(default_factory=list)
    reject_submit: bool = False

    def is_current(self, candidate: BindingRef) -> bool:
        return candidate.fence == self.current_fence

    def submit_terminal_claim(self, claim: TerminalClaim) -> bool:
        if self.reject_submit or not self.is_current(claim.binding):
            return False
        self.claims.append(claim)
        return True


class SequenceExecutor:
    def __init__(self, *results: TurnResult) -> None:
        self.results = list(results)
        self.requests: list[TurnRequest] = []
        self.controls = []

    def execute(self, request, control):
        self.requests.append(request)
        self.controls.append(control)
        return self.results.pop(0)


def test_omitted_task_limits_allow_progress_and_keep_cancellation():
    executor = SequenceExecutor(*([TurnResult(TurnStatus.CONTINUE, context_delta=("evidence",))] * 75),
                                TurnResult(TurnStatus.SUCCEEDED, output="done"))
    outcome = AgentRuntime(executor, FakeAuthority()).run(binding=binding(), instruction="Finish")
    assert outcome.status is OutcomeStatus.SUCCEEDED and outcome.turns == 76
    assert len(executor.requests[-1].tactical_context) == 75
    token = CancellationToken()
    class CancelExecutor:
        def execute(self, request, control):
            assert control.remaining_seconds() == float("inf")
            token.cancel()
            return TurnResult(TurnStatus.CONTINUE)
    result = AgentRuntime(CancelExecutor(), FakeAuthority()).run(binding=binding(), instruction="Work", cancellation=token)
    assert result.status is OutcomeStatus.CANCELLED


def test_nonfinite_task_deadlines_are_rejected():
    for value in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            RunBudget(wall_clock_seconds=value)


def test_success_submits_one_fenced_terminal_claim() -> None:
    gate = FakeAuthority()
    executor = SequenceExecutor(TurnResult(TurnStatus.SUCCEEDED, "done"))
    outcome = AgentRuntime(executor, gate).run(
        binding=binding(), instruction="do work", budget=RunBudget(2, 0, 10)
    )

    assert outcome == outcome.__class__(OutcomeStatus.SUCCEEDED, "done", 1, 1, True)
    assert len(gate.claims) == 1
    assert gate.claims[0].binding.fence == 1
    assert executor.requests[0].binding == binding()


def test_continue_carries_only_tactical_context_to_next_turn() -> None:
    gate = FakeAuthority()
    executor = SequenceExecutor(
        TurnResult(TurnStatus.CONTINUE, "observed", ("tool-result",)),
        TurnResult(TurnStatus.SUCCEEDED, "done"),
    )
    outcome = AgentRuntime(executor, gate).run(
        binding=binding(), instruction="do work", budget=RunBudget(2, 0, 10)
    )

    assert outcome.status is OutcomeStatus.SUCCEEDED
    assert executor.requests[1].tactical_context == ("tool-result",)
    assert executor.requests[1].binding.work_id == "WORK-1"


def test_explicit_retry_is_bounded_per_turn() -> None:
    gate = FakeAuthority()
    executor = SequenceExecutor(
        TurnResult(TurnStatus.RETRYABLE_FAILURE, "transient-1"),
        TurnResult(TurnStatus.RETRYABLE_FAILURE, "transient-2"),
    )
    outcome = AgentRuntime(executor, gate).run(
        binding=binding(), instruction="do work", budget=RunBudget(1, 1, 10)
    )

    assert outcome.status is OutcomeStatus.EXHAUSTED
    assert outcome.attempts == 2
    assert outcome.claim_accepted


def test_stale_result_after_rebind_cannot_submit_claim() -> None:
    gate = FakeAuthority()

    class RebindingExecutor:
        def execute(self, request, control):
            assert not control.should_stop()
            gate.current_fence = 2
            assert control.should_stop()
            return TurnResult(TurnStatus.SUCCEEDED, "late")

    outcome = AgentRuntime(RebindingExecutor(), gate).run(
        binding=binding(1), instruction="do work", budget=RunBudget(1, 0, 10)
    )

    assert outcome.status is OutcomeStatus.REJECTED
    assert not outcome.claim_accepted
    assert gate.claims == []


def test_cancellation_during_turn_is_propagated_and_never_claimed() -> None:
    gate = FakeAuthority()
    token = CancellationToken()

    class CancellingExecutor:
        def execute(self, request, control):
            assert control.cancellation is token
            assert control.binding.fence == request.binding.fence
            token.cancel()
            return TurnResult(TurnStatus.SUCCEEDED, "too late")

    outcome = AgentRuntime(CancellingExecutor(), gate).run(
        binding=binding(), instruction="do work", budget=RunBudget(1, 0, 10), cancellation=token
    )

    assert outcome.status is OutcomeStatus.CANCELLED
    assert not outcome.claim_accepted
    assert gate.claims == []


def test_deadline_exhaustion_is_fenced_before_claim() -> None:
    now = [10.0]
    gate = FakeAuthority()

    class SlowExecutor:
        def execute(self, request, control):
            assert control.remaining_seconds() == 1.0
            now[0] = 12.0
            return TurnResult(TurnStatus.CONTINUE, "late")

    outcome = AgentRuntime(SlowExecutor(), gate, monotonic=lambda: now[0]).run(
        binding=binding(), instruction="do work", budget=RunBudget(2, 0, 1)
    )

    assert outcome.status is OutcomeStatus.EXHAUSTED
    assert outcome.claim_accepted
    assert len(gate.claims) == 1


def test_gate_rejection_fails_closed() -> None:
    gate = FakeAuthority(reject_submit=True)
    executor = SequenceExecutor(TurnResult(TurnStatus.SUCCEEDED, "done"))
    outcome = AgentRuntime(executor, gate).run(
        binding=binding(), instruction="do work", budget=RunBudget(1, 0, 10)
    )
    assert outcome.status is OutcomeStatus.REJECTED
    assert not outcome.claim_accepted


def test_binding_secret_is_not_exposed_by_repr() -> None:
    assert "secret-token" not in repr(binding())


def test_invalid_inputs_fail_early() -> None:
    with pytest.raises(ValueError, match="fence"):
        binding(0)
    with pytest.raises(ValueError, match="max_turns"):
        RunBudget(0, 0, 1)
    gate = FakeAuthority()
    executor = SequenceExecutor(TurnResult(TurnStatus.SUCCEEDED))
    outcome = AgentRuntime(executor, gate).run(
        binding=binding(), instruction=" ", budget=RunBudget(1, 0, 1)
    )
    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == "EMPTY_PROMPT"
    assert executor.requests == []
    assert gate.claims == []


def test_oversized_prompt_is_blocked_before_provider_or_claim() -> None:
    gate = FakeAuthority()
    executor = SequenceExecutor(TurnResult(TurnStatus.SUCCEEDED, "must not run"))
    prompt = "é" * 5

    outcome = AgentRuntime(
        executor,
        gate,
        prompt_gate=PromptBudgetGate(max_prompt_bytes=9),
    ).run(binding=binding(), instruction=prompt, budget=RunBudget(1, 0, 10))

    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == "PROMPT_TOO_LARGE"
    assert outcome.prompt_bytes == 10
    assert outcome.prompt_sha256 == PromptEnvelope(prompt).sha256
    assert outcome.attempts == 0
    assert outcome.turns == 0
    assert executor.requests == []
    assert gate.claims == []


def test_admitted_prompt_envelope_is_the_same_object_seen_by_provider() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(TurnResult(TurnStatus.SUCCEEDED, "done"))

    class RecordingGate:
        def __init__(self) -> None:
            self.seen: list[PromptEnvelope] = []

        def evaluate(self, prompt: PromptEnvelope) -> PromptAdmission:
            self.seen.append(prompt)
            return PromptAdmission(
                PromptDecision.ALLOW,
                "TEST_ALLOW",
                prompt.byte_count,
                prompt.sha256,
            )

    prompt_gate = RecordingGate()
    instruction = "inspect sem reescrever"
    outcome = AgentRuntime(executor, authority, prompt_gate=prompt_gate).run(
        binding=binding(), instruction=instruction, budget=RunBudget(1, 0, 10)
    )

    assert outcome.status is OutcomeStatus.SUCCEEDED
    assert len(prompt_gate.seen) == 1
    assert executor.requests[0].prompt is prompt_gate.seen[0]
    assert executor.requests[0].instruction == instruction
    assert executor.requests[0].prompt.canonical_bytes == instruction.encode("utf-8")


def test_turn_request_rejects_instruction_envelope_divergence() -> None:
    with pytest.raises(ValueError, match="instruction must match"):
        TurnRequest(
            binding=binding(),
            instruction="original",
            turn_number=1,
            attempt_number=1,
            tactical_context=(),
            prompt=PromptEnvelope("rewritten"),
        )


def test_prompt_gate_runs_once_across_provider_retry() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(
        TurnResult(TurnStatus.RETRYABLE_FAILURE, "retry"),
        TurnResult(TurnStatus.SUCCEEDED, "done"),
    )

    class CountingGate:
        calls = 0

        def evaluate(self, prompt: PromptEnvelope) -> PromptAdmission:
            self.calls += 1
            return PromptAdmission(
                PromptDecision.ALLOW,
                "TEST_ALLOW",
                prompt.byte_count,
                prompt.sha256,
            )

    prompt_gate = CountingGate()
    outcome = AgentRuntime(executor, authority, prompt_gate=prompt_gate).run(
        binding=binding(), instruction="do work", budget=RunBudget(1, 1, 10)
    )

    assert outcome.status is OutcomeStatus.SUCCEEDED
    assert prompt_gate.calls == 1
    assert len(executor.requests) == 2
    assert executor.requests[0].prompt is executor.requests[1].prompt


def test_grounding_envelope_is_injected_unchanged_across_retry_and_turn() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(
        TurnResult(TurnStatus.RETRYABLE_FAILURE, "retry"),
        TurnResult(TurnStatus.CONTINUE, "continue", ("tool-result",)),
        TurnResult(TurnStatus.SUCCEEDED, "done"),
    )
    envelope = grounding()

    outcome = AgentRuntime(executor, authority).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(2, 1, 10),
        grounding=envelope,
    )

    assert outcome.status is OutcomeStatus.SUCCEEDED
    assert len(executor.requests) == 3
    assert all(request.grounding is envelope for request in executor.requests)
    assert executor.requests[-1].tactical_context == ("tool-result",)


def test_invalid_grounding_type_never_reaches_provider() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(TurnResult(TurnStatus.SUCCEEDED, "must not run"))

    with pytest.raises(TypeError, match="GroundingEnvelope"):
        AgentRuntime(executor, authority).run(
            binding=binding(),
            instruction="do work",
            budget=RunBudget(1, 0, 10),
            grounding=b"unadmitted",  # type: ignore[arg-type]
        )

    assert executor.requests == []
    assert authority.claims == []


@pytest.mark.parametrize(
    ("work_envelope", "grounding_envelope", "reason"),
    [
        (
            work(target_repositories=("tare.tools.kernel",)),
            None,
            "GROUNDING_CONTEXT_REQUIRED",
        ),
        (
            work(target_repositories=("tare.tools.kernel",)),
            grounding(repository="other-repo"),
            "GROUNDING_CONTEXT_MISMATCH",
        ),
        (
            work(
                target_repositories=("tare.tools.kernel",),
                grounding_refs=("SPEC-A",),
            ),
            grounding(requested_ids=()),
            "GROUNDING_CONTEXT_MISMATCH",
        ),
        (
            work(),
            grounding(),
            "GROUNDING_CONTEXT_UNBOUND",
        ),
    ],
)
def test_strict_work_grounding_binding_fails_before_provider(
    work_envelope: WorkEnvelope,
    grounding_envelope: GroundingEnvelope | None,
    reason: str,
) -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(TurnResult(TurnStatus.SUCCEEDED, "must not run"))

    outcome = AgentRuntime(
        executor, authority, work_guard=SequenceWorkGuard()
    ).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(1, 0, 10),
        work=work_envelope,
        grounding=grounding_envelope,
        strict_work=StrictWorkControl(),
    )

    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == reason
    assert outcome.attempts == 0
    assert executor.requests == []


def test_strict_work_accepts_exact_work_selected_grounding() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(
        TurnResult(TurnStatus.SUCCEEDED, "done", progress=progress(0, 1))
    )
    work_envelope = work(
        target_repositories=("tare.tools.kernel",),
        grounding_refs=("ADR-A", "SPEC-B"),
    )
    grounding_envelope = grounding(requested_ids=("ADR-A", "SPEC-B"))

    outcome = AgentRuntime(
        executor, authority, work_guard=SequenceWorkGuard()
    ).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(1, 0, 10),
        work=work_envelope,
        grounding=grounding_envelope,
        strict_work=StrictWorkControl(),
    )

    assert outcome.status is OutcomeStatus.SUCCEEDED
    assert executor.requests[0].work is work_envelope
    assert executor.requests[0].grounding is grounding_envelope
    assert executor.requests[0].groundings == (grounding_envelope,)


def test_strict_work_accepts_exact_multirepository_groundings() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(
        TurnResult(TurnStatus.SUCCEEDED, "done", progress=progress(0, 1))
    )
    scopes = (
        {
            "repository": "repo-a",
            "grounding_refs": ["A"],
            "target_paths": ["src/a.py"],
            "target_symbols": [],
        },
        {
            "repository": "repo-b",
            "grounding_refs": ["B"],
            "target_paths": ["src/b.py"],
            "target_symbols": [],
        },
    )
    work_envelope = work(
        target_repositories=("repo-a", "repo-b"), repository_scopes=scopes
    )
    repo_b = grounding(repository="repo-b", requested_ids=("B",))
    repo_a = grounding(repository="repo-a", requested_ids=("A",))

    outcome = AgentRuntime(
        executor, authority, work_guard=SequenceWorkGuard()
    ).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(1, 0, 10),
        work=work_envelope,
        groundings=(repo_b, repo_a),
        strict_work=StrictWorkControl(),
    )

    assert outcome.status is OutcomeStatus.SUCCEEDED
    assert executor.requests[0].grounding is None
    assert executor.requests[0].groundings == (repo_a, repo_b)


def test_multirepository_grounding_set_must_be_complete_before_provider() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(TurnResult(TurnStatus.SUCCEEDED, "must not run"))
    scopes = (
        {"repository": "repo-a", "grounding_refs": [], "target_paths": [], "target_symbols": []},
        {"repository": "repo-b", "grounding_refs": [], "target_paths": [], "target_symbols": []},
    )

    outcome = AgentRuntime(
        executor, authority, work_guard=SequenceWorkGuard()
    ).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(1, 0, 10),
        work=work(target_repositories=("repo-a", "repo-b"), repository_scopes=scopes),
        groundings=(grounding(repository="repo-a"),),
        strict_work=StrictWorkControl(),
    )

    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == "GROUNDING_CONTEXT_MISMATCH"
    assert executor.requests == []


def test_strict_work_drift_before_dispatch_blocks_provider() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(TurnResult(TurnStatus.SUCCEEDED, "must not run"))
    guard = SequenceWorkGuard(WorkGuardDecision.DRIFTED)

    outcome = AgentRuntime(executor, authority, work_guard=guard).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(1, 0, 10),
        work=work(),
        strict_work=StrictWorkControl(),
    )

    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == "WORK_ITEM_DRIFT"
    assert outcome.attempts == 0
    assert executor.requests == []
    assert authority.claims == []


@pytest.mark.parametrize(
    ("decision", "guard_reason", "outcome_reason"),
    [
        (
            WorkGuardDecision.DRIFTED,
            "GROUNDING_SCOPE_CHANGED",
            "GROUNDING_CONTEXT_DRIFT",
        ),
        (
            WorkGuardDecision.UNAVAILABLE,
            "GROUNDING_SOURCE_TIMEOUT",
            "GROUNDING_CONTEXT_UNAVAILABLE",
        ),
    ],
)
def test_grounding_guard_failures_keep_distinct_stable_runtime_reasons(
    decision: WorkGuardDecision,
    guard_reason: str,
    outcome_reason: str,
) -> None:
    class GroundingGuard:
        def evaluate(self, _work):
            return WorkGuardResult(decision, guard_reason)

    executor = SequenceExecutor(TurnResult(TurnStatus.SUCCEEDED, "must not run"))
    outcome = AgentRuntime(
        executor, FakeAuthority(), work_guard=GroundingGuard()
    ).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(1, 0, 10),
        work=work(
            target_repositories=("tare.tools.kernel",),
            grounding_refs=("SPEC-A",),
        ),
        grounding=grounding(requested_ids=("SPEC-A",)),
        strict_work=StrictWorkControl(),
    )

    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == outcome_reason
    assert executor.requests == []


def test_turn_adapter_can_block_without_terminal_claim() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(
        TurnResult(
            TurnStatus.BLOCKED,
            "observer unavailable",
            reason_code="PROGRESS_OBSERVATION_FAILED",
        )
    )

    outcome = AgentRuntime(executor, authority).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(1, 0, 10),
    )

    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == "PROGRESS_OBSERVATION_FAILED"
    assert outcome.attempts == 1
    assert authority.claims == []


def test_strict_work_drift_after_result_blocks_terminal_claim() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(
        TurnResult(TurnStatus.SUCCEEDED, "late", progress=progress(0, 1))
    )
    guard = SequenceWorkGuard(
        WorkGuardDecision.CURRENT,
        WorkGuardDecision.DRIFTED,
    )

    outcome = AgentRuntime(executor, authority, work_guard=guard).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(1, 0, 10),
        work=work(),
        strict_work=StrictWorkControl(),
    )

    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == "WORK_ITEM_DRIFT"
    assert outcome.attempts == 1
    assert authority.claims == []


def test_strict_work_is_rechecked_immediately_before_terminal_claim() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(
        TurnResult(TurnStatus.SUCCEEDED, "late", progress=progress(0, 1))
    )
    guard = SequenceWorkGuard(
        WorkGuardDecision.CURRENT,
        WorkGuardDecision.CURRENT,
        WorkGuardDecision.DRIFTED,
    )

    outcome = AgentRuntime(executor, authority, work_guard=guard).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(1, 0, 10),
        work=work(),
        strict_work=StrictWorkControl(),
    )

    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == "WORK_ITEM_DRIFT"
    assert authority.claims == []


def test_strict_mode_blocks_zeno_stagnation_without_agent_self_report() -> None:
    authority = FakeAuthority()
    same_progress = progress(3, 0)
    executor = SequenceExecutor(
        TurnResult(TurnStatus.CONTINUE, "slice-a", progress=same_progress),
        TurnResult(TurnStatus.CONTINUE, "slice-b", progress=same_progress),
        TurnResult(TurnStatus.CONTINUE, "slice-c", progress=same_progress),
    )
    guard = SequenceWorkGuard()

    outcome = AgentRuntime(executor, authority, work_guard=guard).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(4, 0, 10),
        work=work(),
        strict_work=StrictWorkControl(max_stagnant_turns=2),
    )

    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == "STAGNATION_DETECTED"
    assert outcome.attempts == 3
    assert authority.claims == []


def test_strict_mode_blocks_progress_regression_immediately() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(
        TurnResult(TurnStatus.CONTINUE, "baseline", progress=progress(2, 1)),
        TurnResult(TurnStatus.CONTINUE, "worse", progress=progress(3, 1)),
    )

    outcome = AgentRuntime(
        executor, authority, work_guard=SequenceWorkGuard()
    ).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(3, 0, 10),
        work=work(),
        strict_work=StrictWorkControl(),
    )

    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == "PROGRESS_REGRESSION"
    assert outcome.attempts == 2


def test_strict_mode_rejects_success_with_open_obligations() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(
        TurnResult(TurnStatus.SUCCEEDED, "premature", progress=progress(1, 3))
    )

    outcome = AgentRuntime(
        executor, authority, work_guard=SequenceWorkGuard()
    ).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(1, 0, 10),
        work=work(),
        strict_work=StrictWorkControl(),
    )

    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == "EXIT_CRITERIA_UNSATISFIED"
    assert authority.claims == []


def test_strict_mode_accepts_verified_monotonic_progress() -> None:
    authority = FakeAuthority()
    envelope = work()
    executor = SequenceExecutor(
        TurnResult(TurnStatus.CONTINUE, "first", progress=progress(2, 1)),
        TurnResult(TurnStatus.CONTINUE, "second", progress=progress(1, 2)),
        TurnResult(TurnStatus.SUCCEEDED, "done", progress=progress(0, 3)),
    )

    outcome = AgentRuntime(
        executor, authority, work_guard=SequenceWorkGuard()
    ).run(
        binding=binding(),
        instruction="do work",
        budget=RunBudget(3, 0, 10),
        work=envelope,
        strict_work=StrictWorkControl(),
    )

    assert outcome.status is OutcomeStatus.SUCCEEDED
    assert outcome.claim_accepted
    assert all(request.work is envelope for request in executor.requests)


@pytest.mark.parametrize(
    ("prompt_gate", "reason_code"),
    [
        (
            type(
                "RaisingGate",
                (),
                {"evaluate": lambda self, prompt: (_ for _ in ()).throw(RuntimeError("boom"))},
            )(),
            "PROMPT_GATE_ERROR",
        ),
        (
            type("WrongTypeGate", (), {"evaluate": lambda self, prompt: object()})(),
            "INVALID_PROMPT_GATE_RESULT",
        ),
        (
            type(
                "LyingGate",
                (),
                {
                    "evaluate": lambda self, prompt: PromptAdmission(
                        PromptDecision.ALLOW,
                        "TEST_ALLOW",
                        prompt.byte_count + 1,
                        prompt.sha256,
                    )
                },
            )(),
            "INVALID_PROMPT_GATE_RESULT",
        ),
    ],
)
def test_prompt_gate_failures_block_without_provider_execution(
    prompt_gate: object,
    reason_code: str,
) -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(TurnResult(TurnStatus.SUCCEEDED, "must not run"))

    outcome = AgentRuntime(executor, authority, prompt_gate=prompt_gate).run(  # type: ignore[arg-type]
        binding=binding(), instruction="do work", budget=RunBudget(1, 0, 10)
    )

    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == reason_code
    assert executor.requests == []
    assert authority.claims == []


def test_non_utf8_prompt_is_blocked_without_leaking_or_calling_provider() -> None:
    authority = FakeAuthority()
    executor = SequenceExecutor(TurnResult(TurnStatus.SUCCEEDED, "must not run"))

    outcome = AgentRuntime(executor, authority).run(
        binding=binding(), instruction="\ud800", budget=RunBudget(1, 0, 10)
    )

    assert outcome.status is OutcomeStatus.BLOCKED
    assert outcome.reason_code == "INVALID_PROMPT"
    assert outcome.prompt_sha256 is None
    assert executor.requests == []
