import hashlib
import json
from dataclasses import replace

import pytest

from tare_tools_agent_runtime import (
    BindingRef, CancellationToken, ConversationLog, EffectResult, ExecutionControl,
    ModelFailure, ModelReply, NativeBudget, NativeSession, ToolCall,
)
from tare_tools_agent_runtime.native_context import ContextPolicy, NativeContext


POLICY = ContextPolicy(1000, 100, 20)


class EstimatedCounter:
    def count(self, request, execution):
        row = Counter().count(request, execution)
        return row | {'status': 'estimated', 'input_tokens': None,
                      'estimated_input_tokens': row['input_tokens'], 'limitation': 'fixture estimate only'}


def test_estimate_requires_opt_in_and_preserves_compaction_and_measured_unknowns():
    strict_model = Model()
    result = session(ConversationLog(), strict_model, EstimatedCounter()).run(control())
    assert result.reason == 'CONTEXT_COUNT_UNAVAILABLE' and not strict_model.requests
    log, model = ConversationLog(), Model()
    result = session(log, model, EstimatedCounter(),
                     policy=replace(POLICY, allow_estimated_counts=True)).run(control())
    assert result.status == 'proposed' and result.model_calls == 4
    assert [r.purpose for r in model.requests] == ['model', 'model', 'summary', 'model']
    counts = [e['count'] for e in events(log) if e['kind'] == 'context_counted']
    assert counts and all(c['status'] == 'estimated' and c['input_tokens'] is None for c in counts)


def test_measured_session_header_stays_compatible_and_opt_in_cannot_change_on_resume():
    log, model = ConversationLog(), Model()
    first = session(log, model)
    assert first.run(control()).status == 'proposed'
    assert 'allow_estimated_counts' not in events(log)[0]['context_policy']
    assert session(log, Model()).binding_identity() == first.binding_identity()
    with pytest.raises(ValueError, match='identity/configuration mismatch'):
        session(log, Model(), policy=replace(POLICY, allow_estimated_counts=True)).binding_identity()


@pytest.mark.parametrize('changes', [{'estimated_input_tokens': -1}, {'estimated_input_tokens': True},
    {'estimated_input_tokens': None}, {'limitation': ''}, {'input_tokens': 1}, {'status': 'unavailable'}])
def test_opt_in_does_not_accept_malformed_estimates(changes):
    class Invalid(EstimatedCounter):
        def count(self, request, execution):
            return super().count(request, execution) | changes
    model = Model()
    result = session(ConversationLog(), model, Invalid(),
                     policy=replace(POLICY, allow_estimated_counts=True)).run(control())
    assert result.reason == 'CONTEXT_COUNT_UNAVAILABLE' and not model.requests


def control():
    return ExecutionControl(BindingRef("w", "b", "o", 1, "secret"), CancellationToken(), 10, lambda: 0)


def events(log):
    return [r.event() for r in log.snapshot().records]


def identity(request):
    items = request.context.events()
    items = [e for e in items if e["kind"] == "context_activated"] + [e for e in items if e["kind"] != "context_activated"]
    payload = [request.instruction, request.required_context, request.purpose, request.suffix,
               request.remaining_model_calls, items]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


class Counter:
    status = "observed"
    summary_size = 400
    candidate_size = 300
    initial_size = 100

    def count(self, request, execution):
        items = request.context.events()
        if request.purpose == "summary":
            size = self.summary_size
        elif any(e["kind"] == "context_activated" for e in items):
            size = self.candidate_size
        else:
            size = 850 if sum(e["kind"] == "model_returned" for e in items) >= 2 else self.initial_size
        return {"status": self.status, "input_tokens": size, "method": "fixture-wire",
                "context_sha256": request.context.sha256, "request_sha256": identity(request),
                "profile_digest": "a" * 64, "context_window": 1000, "output_tokens": 100,
                "request_purpose": request.purpose}


class Model:
    def __init__(self, summary=None):
        self.requests = []
        self.main_calls = 0
        self.summary = summary or ModelReply(text="Earlier observed facts and source refs",
                                             metadata_json='{"usage":{"input":10,"output":5}}')

    def generate(self, request, execution):
        self.requests.append(request)
        assert request.admitted_request_sha256 == identity(request)
        if request.purpose == "summary":
            if isinstance(self.summary, BaseException):
                raise self.summary
            if callable(self.summary):
                return self.summary(request, execution)
            return self.summary
        self.main_calls += 1
        if self.main_calls <= 2:
            return ModelReply(tool_calls=(ToolCall(str(self.main_calls), "read", '{"path":"a.py"}'),))
        return ModelReply(text="proposal", proposes_delivery=True)


class Tools:
    def __init__(self):
        self.calls = 0

    def execute(self, request, execution):
        self.calls += 1
        return EffectResult(request.effect_id, "succeeded", json.dumps({"source": "a.py", "text": "x" * 1000}), True)


def session(log, model, counter=None, *, policy=POLICY, budget=None):
    return NativeSession(session_id="s", instruction="Keep objective exact", capabilities=("read",),
                         required_context=("SPEC: immutable required source",), model=model,
                         tools=Tools(), journal=log, budget=budget or NativeBudget(),
                         context=NativeContext(policy, counter=(counter or Counter()).count, summarizer=model))


def test_compaction_counts_summary_preserves_sources_tail_originals_and_budget():
    log, model = ConversationLog(), Model()
    result = session(log, model).run(control())
    assert result.status == "proposed" and result.model_calls == 4
    assert [r.purpose for r in model.requests] == ["model", "model", "summary", "model"]
    assert all(r.instruction == "Keep objective exact" and r.required_context == ("SPEC: immutable required source",)
               for r in model.requests)
    final = model.requests[-1].context.events()
    assert sum(e["kind"] == "model_returned" for e in final) == 1
    assert sum(e["kind"] == "effect_returned" for e in final) == 1
    active = next(e for e in events(log) if e["kind"] == "context_activated")
    original = {r.sequence: r for r in log.snapshot().records}
    assert all(original[x["sequence"]].sha256 == x["sha256"] for x in active["covered"])
    assert len([e for e in events(log) if e["kind"] == "effect_returned"]) == 2
    auxiliary = next(e for e in events(log) if e["kind"] == "context_summary_returned")
    assert json.loads(auxiliary["reply"]["metadata_json"])["usage"]["output"] == 5


def test_model_sees_remaining_budget_including_compaction_and_resume():
    log, model = ConversationLog(), Model()
    runtime = session(log, model, budget=NativeBudget(max_model_calls=5))
    assert runtime.run(control()).status == 'proposed'
    assert [r.remaining_model_calls for r in model.requests] == [5, 4, None, 2]
    assert session(log, model, budget=NativeBudget(max_model_calls=5)).run(control()).status == 'proposed'
    assert len(model.requests) == 4


@pytest.mark.parametrize("trigger,output,expected", [(0.8,100,800),(0.5,100,500),(0.8,300,680)])
def test_threshold_uses_fraction_and_shared_window_reserve(trigger,output,expected):
    p = ContextPolicy(1000, output, 20, trigger=trigger, target=0.4)
    assert p.trigger_limit == expected


@pytest.mark.parametrize('size', [800, 801, 850, 880, 881])
def test_disabled_compaction_preserves_exact_history_up_to_the_hard_limit(size):
    log, model, counter = ConversationLog(), Model(), Counter()
    counter.initial_size = size
    policy = replace(POLICY, max_compactions=0)
    result = session(log, model, counter, policy=policy).run(control())
    if size > policy.hard_limit:
        assert result.reason == 'COMPACTION_BUDGET' and not model.requests
    else:
        assert result.status == 'proposed' and result.model_calls == 3
        assert all(r.purpose == 'model' for r in model.requests)
        assert [len(r.context.events()) for r in model.requests] == [0, 2, 4]
        for before, after in zip(model.requests, model.requests[1:]):
            assert before.context.events() == after.context.events()[:len(before.context.events())]
            assert before.instruction == after.instruction and before.required_context == after.required_context
    assert not any(e['kind'].startswith('context_summary') or e['kind'] == 'context_activated' for e in events(log))


@pytest.mark.parametrize("cause", ["missing", "wrong_identity", "wrong_profile_window", "estimated"])
def test_no_generation_without_usable_count_for_the_actual_request(cause):
    class BadCounter(Counter):
        def count(self, request, execution):
            row = super().count(request, execution)
            row.update({"missing": {"input_tokens": None}, "wrong_identity": {"context_sha256": "wrong"},
                        "wrong_profile_window": {"context_window": 999}, "estimated": {"status": "estimated"}}[cause])
            return row
    log, model = ConversationLog(), Model()
    assert session(log, model, BadCounter()).run(control()).status == "blocked"
    assert not model.requests


def test_large_system_or_tools_with_no_history_cannot_trigger_useless_summary():
    counter = Counter()
    counter.initial_size = 900
    log, model = ConversationLog(), Model()
    assert session(log, model, counter).run(control()).reason == "NO_COMPACTABLE_HISTORY"
    assert not model.requests


@pytest.mark.parametrize("failure,reason", [(ModelFailure("429"), "SUMMARY_FAILED"),
    (ModelReply(text="partial", complete=False), "INVALID_SUMMARY"),
    (ModelReply(tool_calls=(ToolCall("oops", "read", '{}'),)), "INVALID_SUMMARY")])
def test_failed_or_invalid_summary_never_replaces_original_projection(failure, reason):
    log, model = ConversationLog(), Model(failure)
    result = session(log, model).run(control())
    assert result.reason == reason and result.model_calls == 3
    assert not any(e["kind"] == "context_activated" for e in events(log))
    projection = NativeContext.project(log)
    assert sum(e["kind"] == "model_returned" for e in projection.events()) == 2
    assert session(log, model).run(control()).reason == "COMPACTION_BUDGET"
    assert len(model.requests) == 3


@pytest.mark.parametrize("which,reason", [("summary", "SUMMARY_INPUT_OVERFLOW"),
                                        ("candidate", "COMPACTION_TARGET_NOT_REACHED")])
def test_overflow_before_or_after_summary_is_bounded(which, reason):
    log, model, counter = ConversationLog(), Model(), Counter()
    if which == "summary":
        counter.summary_size = 900
    else:
        counter.candidate_size = 801
    result = session(log, model, counter).run(control())
    assert result.reason == reason
    assert len(model.requests) == (2 if which == "summary" else 3)
    assert not any(e["kind"] == "context_activated" for e in events(log))


@pytest.mark.parametrize('fits', [True, False])
def test_final_call_preserves_prefix_below_hard_limit_without_spending_a_summary(fits):
    log, model = ConversationLog(), Model()
    class LastCounter(Counter):
        def count(self, request, execution):
            row = super().count(request, execution)
            if not fits and row['input_tokens'] == 850:
                row['input_tokens'] = POLICY.hard_limit+1
            return row
    result = session(log, model, LastCounter(), budget=NativeBudget(max_model_calls=3)).run(control())
    assert result.reason == ('DELIVERY_PROPOSED' if fits else 'MODEL_CALL_BUDGET')
    assert result.model_calls == (3 if fits else 2)
    assert len(model.requests) == (3 if fits else 2)
    assert not any(e['kind'] == 'context_summary_requested' for e in events(log))
    if fits:
        assert model.requests[-1].remaining_model_calls == 1
        assert model.requests[-1].instruction == 'Keep objective exact'
        assert model.requests[-1].required_context == ('SPEC: immutable required source',)


def test_cancelled_summary_retains_response_but_never_activates_it_until_resume():
    def cancel(request, execution):
        execution.cancellation.cancel()
        return ModelReply(text="saved summary")
    log, model = ConversationLog(), Model(cancel)
    assert session(log, model).run(control()).status == "cancelled"
    assert not any(e["kind"] == "context_activated" for e in events(log))
    assert session(log, model).run(control()).status == "proposed"
    assert sum(r.purpose == "summary" for r in model.requests) == 1


class Crash(BaseException):
    pass


class FaultJournal(ConversationLog):
    def __init__(self, kind, after):
        super().__init__()
        self.kind, self.after, self.armed = kind, after, True

    def append(self, event, **kwargs):
        fault = self.armed and event["kind"] == self.kind
        if fault:
            self.armed = False
            if not self.after:
                raise Crash()
        result = super().append(event, **kwargs)
        if fault:
            raise Crash()
        return result


@pytest.mark.parametrize("kind,after,status", [
    ("context_summary_requested", True, "reconciliation_required"),
    ("context_summary_returned", False, "reconciliation_required"),
    ("context_summary_returned", True, "proposed"),
    ("context_activated", False, "proposed"),
    ("context_activated", True, "proposed"),
])
def test_crash_before_or_after_activation_never_repeats_an_unconfirmed_summary(kind, after, status):
    log, model = FaultJournal(kind, after), Model()
    with pytest.raises(Crash):
        session(log, model).run(control())
    result = session(log, model).run(control())
    assert result.status == status
    assert sum(r.purpose == "summary" for r in model.requests) <= 1
    if status == 'proposed':
        assert model.requests[-1].remaining_model_calls == 13


def test_unknown_summary_failure_requires_reconciliation_and_keeps_previous_context():
    log, model = ConversationLog(), Model(TimeoutError())
    result = session(log, model).run(control())
    assert result.status == "reconciliation_required" and result.pending_id == "s/summary/1"
    assert session(log, model).run(control()).status == "reconciliation_required"
    assert len(model.requests) == 3
    assert not any(e["kind"] == "context_activated" for e in events(log))


def test_two_compactions_keep_whole_recent_turns_and_count_both_auxiliary_calls():
    class GrowingCounter(Counter):
        def count(self, request, execution):
            result = super().count(request, execution)
            items = request.context.events()
            if request.purpose == "model" and any(e["kind"] == "context_activated" for e in items):
                result["input_tokens"] = 850 if sum(e["kind"] == "model_returned" for e in items) >= 2 else 300
            return result
    class LongerModel(Model):
        def generate(self, request, execution):
            if request.purpose == "summary" or self.main_calls < 2:
                return super().generate(request, execution)
            self.requests.append(request)
            assert request.admitted_request_sha256 == identity(request)
            self.main_calls += 1
            if self.main_calls <= 3:
                return ModelReply(tool_calls=(ToolCall(str(self.main_calls), "read", '{}'),))
            return ModelReply(text="proposal", proposes_delivery=True)
    log, model = ConversationLog(), LongerModel()
    result = session(log, model, GrowingCounter(), policy=replace(POLICY, max_compactions=2)).run(control())
    assert result.status == "proposed" and result.model_calls == 6
    assert sum(e["kind"] == "context_activated" for e in events(log)) == 2
    tail = model.requests[-1].context.events()
    assert sum(e["kind"] == "model_returned" for e in tail) == 1
    assert sum(e["kind"] == "effect_returned" for e in tail) == 1


def test_invalid_summary_saved_before_crash_is_still_rejected_on_resume():
    log, model = FaultJournal("context_summary_returned", True), Model(ModelReply(text="partial", complete=False))
    with pytest.raises(Crash):
        session(log, model).run(control())
    assert session(log, model).run(control()).reason == "INVALID_SUMMARY"
    assert not any(e["kind"] == "context_activated" for e in events(log))
    assert sum(r.purpose == "summary" for r in model.requests) == 1


def test_required_source_or_policy_change_on_resume_is_not_silently_admitted():
    log, model = ConversationLog(), Model()
    assert session(log, model).run(control()).status == "proposed"
    with pytest.raises(ValueError, match="identity/configuration"):
        session(log, model, policy=replace(POLICY, trigger=0.7)).run(control())
    changed = session(log, model)
    changed.required_context = ("SPEC changed",)
    with pytest.raises(ValueError, match="identity/configuration"):
        changed.run(control())


def test_unlimited_model_and_compaction_budgets_keep_summary_accounting():
    log, model = ConversationLog(), Model()
    result = session(log, model, policy=replace(POLICY, max_compactions=None),
                     budget=NativeBudget(max_model_calls=None)).run(control())
    assert result.status == 'proposed' and result.model_calls == 4
    assert len([e for e in events(log) if e['kind'] == 'context_activated']) == 1
    assert events(log)[0]['context_policy']['max_compactions'] is None


def test_operational_facts_survive_conflicting_summary_and_crash_without_recollection():
    log = FaultJournal('context_summary_returned', True)
    model = Model(ModelReply(text='Model claim: all tests passed'))
    calls = []
    def checkpoint(snapshot):
        calls.append(snapshot.revision)
        return 'Recorded verifier: returncode=2, SyntaxError, receipt=7'
    first = session(log, model)
    first.context.checkpoint = checkpoint
    with pytest.raises(Crash):
        first.run(control())
    resumed = session(log, model)
    resumed.context.checkpoint = lambda snapshot: pytest.fail('must reuse saved checkpoint')
    assert resumed.run(control()).status == 'proposed'
    assert len(calls) == 1
    activation = next(e for e in events(log) if e['kind'] == 'context_activated')
    assert activation['observed_state'] == 'Recorded verifier: returncode=2, SyntaxError, receipt=7'
    assert activation['summary'] == 'Model claim: all tests passed'
    assert activation in model.requests[-1].context.events()


def test_checkpoint_is_included_in_candidate_count_before_activation():
    class CountsFacts(Counter):
        def count(self, request, execution):
            row = super().count(request, execution)
            if any(e.get('observed_state') == 'large checkpoint' for e in request.context.events()):
                row['input_tokens'] = 801
            return row
    log, model = ConversationLog(), Model()
    selected = session(log, model, CountsFacts())
    selected.context.checkpoint = lambda snapshot: 'large checkpoint'
    assert selected.run(control()).reason == 'COMPACTION_TARGET_NOT_REACHED'
    assert not any(e['kind'] == 'context_activated' for e in events(log))


@pytest.mark.parametrize("candidate_size", [501, 600, 800])
def test_compaction_above_target_but_within_admission_limit_continues(candidate_size):
    log, model, counter = ConversationLog(), Model(), Counter()
    counter.candidate_size = candidate_size
    result = session(log, model, counter).run(control())
    assert result.status == "proposed" and result.model_calls == 4
    assert [r.purpose for r in model.requests] == ["model", "model", "summary", "model"]
    assert len([e for e in events(log) if e["kind"] == "context_activated"]) == 1
    assert not any(e["kind"] == "context_summary_rejected" for e in events(log))


@pytest.mark.parametrize("candidate_size", [801, 850, 900])
def test_compaction_still_above_trigger_never_dispatches_candidate(candidate_size):
    log, model, counter = ConversationLog(), Model(), Counter()
    counter.candidate_size = candidate_size
    result = session(log, model, counter).run(control())
    assert result.reason == "COMPACTION_TARGET_NOT_REACHED"
    assert [r.purpose for r in model.requests] == ["model", "model", "summary"]
    assert not any(e["kind"] == "context_activated" for e in events(log))


def test_large_protected_history_preview_uses_admitted_source_envelope():
    from tare_tools_agent_runtime import ConversationLimits
    log, model = ConversationLog(ConversationLimits(max_projection_bytes=4*1024*1024)), Model()
    def messages(ids):
        return (str(len(ids)), 'x'*15000) if len(ids) < 160 else None
    result = session(log, model).run(control(), receive_message=messages)
    assert result.status == 'proposed'
    assert len([e for e in events(log) if e['kind'] == 'context_activated']) == 1
    assert sum(e['kind'] == 'user_message' for e in model.requests[-1].context.events()) == 160


def test_genuine_projection_refusal_blocks_without_model_dispatch_or_masking_other_reasons():
    from tare_tools_agent_runtime import ConversationLimits, ConversationRejected, ConversationRejectReason
    log, model = ConversationLog(ConversationLimits(max_projection_bytes=500)), Model()
    runtime = session(log, model)
    binding = {name: getattr(control().binding, name) for name in ('work_id', 'binding_id', 'owner_id', 'fence')}
    log.append(runtime._header(binding))
    log.append({'kind': 'user_message', 'message_id': 'large', 'after_sequence': 1, 'text': 'x'*1000})
    outcome = runtime.run(control())
    assert outcome.status == 'blocked' and outcome.reason == 'PROJECTION_TOO_LARGE'
    assert not model.requests

    class OtherRefusal(ConversationLog):
        def project(self, spec):
            raise ConversationRejected(ConversationRejectReason.INVALID_PROJECTION, 'different error')
    with pytest.raises(ConversationRejected) as exc:
        session(OtherRefusal(), Model()).run(control())
    assert exc.value.reason == ConversationRejectReason.INVALID_PROJECTION
