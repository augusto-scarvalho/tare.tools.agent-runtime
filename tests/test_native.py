"""Behavioral falsifiers for the native loop using no provider or filesystem tools."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from threading import Event, Thread

import pytest

from tare_tools_agent_runtime import BindingRef, CancellationToken, ConversationLog, ExecutionControl
from tare_tools_agent_runtime.native import (
    EffectResult, ModelFailure, ModelReply, NativeBudget, NativeSession, ToolCall,
)


def control(*, cancel=None, current=lambda: True, clock=lambda: 0, fence=1):
    return ExecutionControl(BindingRef("work", "binding", "owner", fence, "secret"),
                            cancel or CancellationToken(), 10, clock, current)


class ScriptedModel:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.requests = []

    def generate(self, request, execution):
        self.requests.append(request)
        response = next(self.replies)
        if isinstance(response, BaseException):
            raise response
        return response(request, execution) if callable(response) else response


class MemoryTools:
    def __init__(self):
        self.content = "old"
        self.requests = []
        self.verified = False

    def execute(self, request, execution):
        assert not execution.should_stop()
        self.requests.append(request)
        call = request.tool_call
        args = json.loads(call.arguments_json)
        if args.get("path") != "app.py":
            return EffectResult(request.effect_id, "refused", '{"reason":"scope"}')
        if call.name == "read":
            data = {"text": self.content, "version": hashlib.sha256(self.content.encode()).hexdigest()}
        elif call.name == "edit":
            if args["expected_version"] != hashlib.sha256(self.content.encode()).hexdigest():
                return EffectResult(request.effect_id, "refused", '{"reason":"stale"}')
            self.content = args["text"]
            data = {"changed": True}
        elif call.name == "verify":
            self.verified = self.content == "new"
            data = {"passed": self.verified}
        else:
            raise AssertionError("unexpected tool")
        return EffectResult(request.effect_id, "succeeded", json.dumps(data), progressed=True)


def tool(name, call_id="t1", **args):
    return ModelReply(tool_calls=(ToolCall(call_id, name, json.dumps({"path": "app.py", **args})),))


def session(model, tools, journal, **kwargs):
    return NativeSession(session_id="s", instruction="Change old to new and verify",
                         capabilities=("read", "edit", "verify"), model=model,
                         tools=tools, journal=journal, **kwargs)


def delivery():
    return ModelReply(text="Changed app.py; verification attached.", proposes_delivery=True)


@pytest.mark.parametrize('calls', [None, 8])
@pytest.mark.parametrize('clock,reserve,expected', [(6, 3, 'normal'), (7, 3, 1), (8, 3, 1), (8, None, 'normal')])
def test_final_call_time_reserve_uses_existing_notice_without_renewing_limits(calls, clock, reserve, expected):
    log, model = ConversationLog(), ScriptedModel([delivery()])
    runner = session(model, None, log, budget=NativeBudget(max_model_calls=calls),
                     final_call_reserve_seconds=reserve)
    result = runner.run(control(clock=lambda: clock))
    assert result.status == 'proposed' and result.model_calls == 1
    assert model.requests[0].remaining_model_calls == (calls if expected == 'normal' else expected)
    header = log.snapshot().records[0].event()
    assert header['budget']['max_model_calls'] == calls
    assert ('final_call_reserve_seconds' in header) is (reserve is not None)
    assert runner.budget.max_model_calls == calls
    assert session(model, None, log, budget=NativeBudget(max_model_calls=calls),
                   final_call_reserve_seconds=reserve).binding_identity() == runner.binding_identity()
    with pytest.raises(ValueError, match='identity/configuration mismatch'):
        session(model, None, log, budget=NativeBudget(max_model_calls=calls),
                final_call_reserve_seconds=4).binding_identity()


@pytest.mark.parametrize('reserve', [0, -1, True, '3', float('inf'), float('nan')])
def test_invalid_final_call_reserve_is_rejected(reserve):
    with pytest.raises(ValueError):
        session(ScriptedModel([]), None, ConversationLog(), final_call_reserve_seconds=reserve)


def test_final_call_reserve_does_not_admit_an_expired_or_exhausted_request():
    model = ScriptedModel([])
    result = session(model, None, ConversationLog(), final_call_reserve_seconds=3).run(control(clock=lambda:10))
    assert result.status == 'exhausted' and not model.requests and result.model_calls == 0
    log = ConversationLog()
    model = ScriptedModel([ModelReply(text='Still incomplete.')])
    budget = NativeBudget(max_model_calls=1)
    result = session(model, None, log, budget=budget, final_call_reserve_seconds=3).run(control(clock=lambda:8))
    assert result.status == 'exhausted' and result.model_calls == 1 and len(model.requests) == 1
    later = ScriptedModel([])
    result = session(later, None, log, budget=budget, final_call_reserve_seconds=3).run(control(clock=lambda:8))
    assert result.status == 'exhausted' and result.model_calls == 1 and not later.requests


@pytest.mark.parametrize('retries', [0, 1])
def test_transient_undispatched_failure_stops_run_but_explicit_resume_keeps_budget(retries):
    log = ConversationLog()
    model = ScriptedModel([*[ModelFailure('CONNECT_FAILED', not_started=True, retryable=True)
                              for _ in range(retries + 1)], delivery()])
    budget = NativeBudget(max_model_retries=retries, max_model_calls=retries + 2)
    result = session(model, None, log, budget=budget).run(control())
    assert result.status == 'blocked' and result.reason == 'CONNECT_FAILED'
    assert result.pending_id is None and len(model.requests) == retries + 1
    assert log.snapshot().records[-1].event() == {
        'kind': 'invocation_not_started', 'invocation_id': f's/model/{retries + 1}',
        'reason': 'CONNECT_FAILED'}
    # Reconstruct the consumer: no ephemeral reset or new budget authorizes recovery.
    recovered = session(model, None, log, budget=budget).run(control())
    assert recovered.status == 'proposed' and recovered.model_calls == retries + 2
    assert len(model.requests) == retries + 2


def test_undispatched_resume_does_not_renew_exhausted_call_budget():
    log = ConversationLog()
    model = ScriptedModel([ModelFailure('CONNECT_FAILED', not_started=True, retryable=True), delivery()])
    budget = NativeBudget(max_model_calls=1)
    assert session(model, None, log, budget=budget).run(control()).reason == 'CONNECT_FAILED'
    assert session(model, None, log, budget=budget).run(control()).reason == 'MODEL_CALL_BUDGET'
    assert len(model.requests) == 1


def test_nonretryable_undispatched_refusal_remains_terminal():
    log = ConversationLog()
    model = ScriptedModel([ModelFailure('SPEND_LIMIT', not_started=True), delivery()])
    for _ in range(2):
        assert session(model, None, log).run(control()).reason == 'SPEND_LIMIT'
    assert len(model.requests) == 1


def test_read_edit_verify_delivery_with_source_version_and_immutable_context():
    journal, tools = ConversationLog(), MemoryTools()

    def edit_after_read(request, execution):
        events = request.context.events()
        observed = json.loads(events[-1]["result"]["output_json"])
        events[-1]["result"]["output_json"] = "tampered"
        assert request.context.events()[-1]["result"]["output_json"] != "tampered"
        return tool("edit", "t2", expected_version=observed["version"], text="new")

    model = ScriptedModel([tool("read"), edit_after_read, tool("verify", "t3"), delivery()])
    result = session(model, tools, journal).run(control())
    assert (result.status, result.model_calls) == ("proposed", 4)
    assert tools.content == "new" and tools.verified
    assert [r.tool_call.name for r in tools.requests] == ["read", "edit", "verify"]
    assert len({r.effect_id for r in tools.requests}) == 3
    assert "secret" not in str([r.event() for r in journal.snapshot().records])
    assert session(model, tools, journal).run(control()) == result
    assert len(model.requests) == 4  # Terminal replay does not call the model again.


@pytest.mark.parametrize("reply,reason", [
    (ModelReply(text="done", complete=False, tool_calls=tool("edit").tool_calls), "PARTIAL_REPLY"),
    (ModelReply(text="done", proposes_delivery=True, tool_calls=tool("read").tool_calls), "AMBIGUOUS_DELIVERY"),
    (ModelReply(proposes_delivery=True), "AMBIGUOUS_DELIVERY"),
    (ModelReply(tool_calls=(ToolCall("x", "shell", '{}'),)), "TOOL_NOT_ADMITTED_OR_DUPLICATE"),
    (ModelReply(tool_calls=tuple(ToolCall(str(i), "read", '{}') for i in range(33))), "TOOL_BATCH_LIMIT"),
    (ModelReply(tool_calls=(ToolCall("x", "read", '{}'), ToolCall("x", "read", '{}'))), "TOOL_NOT_ADMITTED_OR_DUPLICATE"),
    ({"text": "done", "tool": "edit"}, "INVALID_REPLY"),
])
def test_invalid_or_partial_reply_never_executes_an_effect(reply, reason):
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([reply])
    outcome = session(model, tools, journal).run(control())
    assert (outcome.status, outcome.reason) == ("blocked", reason)
    assert tools.requests == []
    assert session(model, tools, journal).run(control()).status == "blocked"


@pytest.mark.parametrize("raw", ['[]', '{"path":1,"path":2}', '{"value":NaN}', '{'])
def test_tool_arguments_reuse_strict_object_ingress(raw):
    with pytest.raises(Exception):
        ToolCall("id", "edit", raw)


@pytest.mark.parametrize('stop', ['cancel','revoke','unknown'])
def test_serial_batch_checks_control_and_resumes_without_repeating_completed_effect(stop):
    journal, tools = ConversationLog(), MemoryTools()
    token = CancellationToken()
    current = [True]
    first = control(cancel=token, current=lambda:current[0])
    calls = tuple(tool('read',str(i)).tool_calls[0] for i in range(3))
    model = ScriptedModel([ModelReply(tool_calls=calls), delivery()])
    execute = tools.execute
    def stopping(request, execution):
        result = execute(request, execution)
        if len(tools.requests)==1:
            if stop=='cancel':token.cancel()
            elif stop=='revoke':current[0]=False
            else:return EffectResult(request.effect_id,'unknown','{}')
        return result
    tools.execute=stopping
    result=session(model,tools,journal).run(first)
    assert result.status == {'cancel':'cancelled','revoke':'rejected','unknown':'reconciliation_required'}[stop]
    assert len(tools.requests)==len(model.requests)==1
    tools.execute=execute;current[0]=True
    result=session(model,tools,journal).run(control())
    if stop=='unknown':
        assert result.status=='reconciliation_required' and len(tools.requests)==1
    else:
        assert result.status=='proposed' and len(model.requests)==2
        assert [r.tool_call.call_id for r in tools.requests]==['0','1','2']
        assert len({r.effect_id for r in tools.requests})==3


def test_batch_replay_after_reply_or_completed_effect_and_projection_includes_every_result():
    journal, tools = ConversationLog(), MemoryTools()
    token=CancellationToken()
    def batch(request,execution):
        token.cancel()
        return ModelReply(tool_calls=tuple(tool('read',str(i)).tool_calls[0] for i in range(3)))
    def finish(request,execution):
        results=[e for e in request.context.events() if e['kind']=='effect_returned']
        assert len(results)==3
        return delivery()
    model=ScriptedModel([batch,finish])
    assert session(model,tools,journal).run(control(cancel=token)).status=='cancelled'
    assert not tools.requests
    assert session(model,tools,journal).run(control()).status=='proposed'
    assert len(model.requests)==2 and len(tools.requests)==3


def test_prose_is_not_completion_and_stagnation_is_bounded_on_resume():
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([ModelReply(text="I fixed everything") for _ in range(3)])
    budget = NativeBudget(max_stagnant_turns=2)
    result = session(model, tools, journal, budget=budget).run(control())
    assert result.reason == "NO_PROGRESS" and result.model_calls == 2
    assert session(model, tools, journal, budget=budget).run(control()) == result
    assert len(model.requests) == 2 and not tools.requests


def test_model_budget_cannot_be_reset_by_resume_or_changed_configuration():
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([tool("read")])
    budget = NativeBudget(max_model_calls=1)
    first = session(model, tools, journal, budget=budget).run(control())
    assert first.reason == "MODEL_CALL_BUDGET"
    assert session(model, tools, journal, budget=budget).run(control()) == first
    with pytest.raises(ValueError, match="identity/configuration"):
        session(model, tools, journal).run(control())
    with pytest.raises(ValueError, match="identity/configuration"):
        session(model, tools, journal, budget=budget).run(control(fence=2))
    assert len(model.requests) == len(tools.requests) == 1


def successor_control(**kwargs):
    from dataclasses import replace
    return replace(control(**kwargs), binding=BindingRef('work','successor','next-owner',2,'next-secret'))


def test_rebind_consumes_saved_reply_preserving_calls_and_original_header():
    journal, tools = ConversationLog(), MemoryTools()
    stopped = control()
    def reply_then_cancel(request, execution):
        execution.cancellation.cancel()
        return tool('read')
    model = ScriptedModel([reply_then_cancel, delivery()])
    runner = session(model, tools, journal, budget=NativeBudget(max_model_calls=2))
    assert runner.run(stopped).status == 'cancelled'
    original = journal.snapshot().records[0]
    previous = runner.binding_identity()
    runner.rebind(successor_control(), previous_binding=previous, handoff_id='os-receipt-1')
    assert runner.run(successor_control()).status == 'proposed'
    assert len(model.requests) == 2 and len(tools.requests) == 1
    assert journal.snapshot().records[0] == original
    with pytest.raises(ValueError, match='identity/configuration'):
        runner.run(control())  # old owner's stale instance cannot emit even with a permissive callback
    assert 'next-secret' not in str([r.event() for r in journal.snapshot().records])


@pytest.mark.parametrize('state',['pending','terminal','budget'])
def test_rebind_does_not_bypass_uncertainty_terminal_or_budget(state):
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([RuntimeError('lost')] if state=='pending' else [delivery()] if state=='terminal' else [tool('read')])
    runner = session(model, tools, journal, budget=NativeBudget(max_model_calls=1))
    runner.run(control())
    before = journal.snapshot()
    if state != 'budget':
        with pytest.raises(ValueError, match='nonterminal session without pending'):
            runner.rebind(successor_control(), previous_binding=runner.binding_identity(), handoff_id='h')
        assert journal.snapshot() == before
    else:
        runner.rebind(successor_control(), previous_binding=runner.binding_identity(), handoff_id='h')
        assert runner.run(successor_control()).reason == 'MODEL_CALL_BUDGET'
    assert len(model.requests) == 1


@pytest.mark.parametrize('invalid',['work','fence','id','revoked','cancelled'])
def test_invalid_rebind_keeps_journal_unchanged(invalid):
    from dataclasses import replace
    journal, tools = ConversationLog(), MemoryTools()
    runner = session(ScriptedModel([]), tools, journal)
    stopped = control(); stopped.cancellation.cancel(); runner.run(stopped)
    successor = successor_control(current=lambda:invalid!='revoked')
    if invalid=='cancelled':successor.cancellation.cancel()
    elif invalid in {'work','fence','id'}:
        changes={'work':{'work_id':'other'},'fence':{'fence':1},'id':{'binding_id':'binding'}}[invalid]
        successor=replace(successor,binding=replace(successor.binding,**changes))
    before=journal.snapshot()
    with pytest.raises(ValueError):runner.rebind(successor,previous_binding=runner.binding_identity(),handoff_id='h')
    assert journal.snapshot()==before


def test_rebound_event_with_wrong_predecessor_is_not_accepted_on_replay():
    journal, tools = ConversationLog(), MemoryTools()
    runner = session(ScriptedModel([]), tools, journal)
    stopped = control(); stopped.cancellation.cancel();runner.run(stopped)
    previous=runner.binding_identity()
    journal.append({'kind':'binding_rebound','previous':dict(previous,owner_id='wrong'),
        'binding':dict(previous,binding_id='next',fence=2),'handoff_id':'h'})
    with pytest.raises(ValueError,match='predecessor'):runner.run(successor_control())


def test_only_explicit_retryable_failures_retry_within_the_same_call_budget():
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([ModelFailure("throttled", retryable=True), delivery()])
    result = session(model, tools, journal, budget=NativeBudget(max_model_retries=1)).run(control())
    assert result.status == "proposed" and result.model_calls == 2
    events = [r.event() for r in journal.snapshot().records]
    assert any(e["kind"] == "model_failed" and e["code"] == "throttled" for e in events)


def test_the_retry_budget_is_per_call_not_spent_for_the_whole_session():
    # A failure early in a session used up the single retry of every later call: one more slip blocked it.
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([ModelFailure("slip", retryable=True), tool("read"),
                           ModelFailure("slip", retryable=True), delivery()])
    result = session(model, tools, journal, budget=NativeBudget(max_model_retries=1)).run(control())
    assert result.status == "proposed" and result.model_calls == 4
    model = ScriptedModel([ModelFailure("slip", retryable=True), ModelFailure("slip", retryable=True)])
    result = session(model, MemoryTools(), ConversationLog(), budget=NativeBudget(max_model_retries=1)).run(control())
    assert result.status == "blocked" and result.model_calls == 2  # two in a row still exceed one retry


def test_retry_failure_before_terminal_persistence_does_not_reset_on_resume():
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([ModelFailure("throttled", retryable=True)] * 2)
    budget = NativeBudget(max_model_retries=1)
    result = session(model, tools, journal, budget=budget).run(control())
    assert result.status == "blocked" and result.model_calls == 2
    assert session(model, tools, journal, budget=budget).run(control()) == result
    assert len(model.requests) == 2


class Crash(BaseException):
    pass


class FaultJournal(ConversationLog):
    def __init__(self, kind, after=False):
        super().__init__()
        self.kind, self.after, self.armed = kind, after, True

    def append(self, event, **kwargs):
        triggered = self.armed and event["kind"] == self.kind
        if triggered:
            self.armed = False
            if not self.after:
                raise Crash()
        record = super().append(event, **kwargs)
        if triggered:
            raise Crash()
        return record


@pytest.mark.parametrize("kind,after,expected,effects", [
    ("model_requested", False, "proposed", 1),
    ("model_requested", True, "reconciliation_required", 0),
    ("model_returned", False, "reconciliation_required", 0),
    ("model_returned", True, "proposed", 1),
    ("effect_requested", False, "proposed", 1),
    ("effect_requested", True, "reconciliation_required", 0),
    ("effect_returned", False, "reconciliation_required", 1),
    ("effect_returned", True, "proposed", 1),
])
def test_crash_boundaries_never_repeat_an_uncertain_effect(kind, after, expected, effects):
    journal, tools = FaultJournal(kind, after), MemoryTools()
    version = hashlib.sha256(b"old").hexdigest()
    model = ScriptedModel([tool("edit", text="new", expected_version=version), delivery()])
    with pytest.raises(Crash):
        session(model, tools, journal).run(control())
    result = session(model, tools, journal).run(control())
    assert result.status == expected
    assert len(tools.requests) == effects
    if expected == "reconciliation_required":
        assert result.pending_id is not None
        assert session(model, tools, journal).run(control()) == result


def test_unknown_tool_result_stays_pending_instead_of_being_retried():
    class UnknownTools:
        calls = 0

        def execute(self, request, execution):
            self.calls += 1
            return EffectResult(request.effect_id, "unknown")

    journal, tools = ConversationLog(), UnknownTools()
    model = ScriptedModel([tool("edit")])
    outcome = session(model, tools, journal).run(control())
    assert outcome.status == "reconciliation_required"
    assert session(model, tools, journal).run(control()) == outcome
    assert tools.calls == 1


def test_adapter_exception_retains_unfinished_model_invocation():
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([RuntimeError("connection lost")])
    assert session(model, tools, journal).run(control()).status == "reconciliation_required"
    assert session(model, tools, journal).run(control()).status == "reconciliation_required"
    assert len(model.requests) == 1 and not tools.requests
    observation = journal.snapshot().records[-1].event()
    assert observation["error_type"] == "RuntimeError" and observation["usage"] is None


def test_cancel_during_model_wait_propagates_and_prevents_tool_dispatch():
    entered, release = Event(), Event()
    cancel = CancellationToken()
    journal, tools = ConversationLog(), MemoryTools()

    def waiting_model(request, execution):
        entered.set()
        assert release.wait(2)
        assert execution.cancellation.is_cancelled()
        return tool("read")

    model = ScriptedModel([waiting_model, delivery()])
    outcomes = []
    current = session(model, tools, journal)
    thread = Thread(target=lambda: outcomes.append(current.run(control(cancel=cancel))))
    thread.start()
    try:
        assert entered.wait(2)
        with pytest.raises(RuntimeError, match="already running"):
            current.run(control())
        cancel.cancel()
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive()
    assert outcomes[0].status == "cancelled" and tools.requests == []
    # Explicit same-binding resume consumes the complete saved reply once.
    assert session(model, tools, journal).run(control()).status == "proposed"
    assert len(tools.requests) == 1


def test_expired_run_replays_recorded_terminal_without_input_or_dispatch():
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([delivery()])
    runner = session(model, tools, journal)
    completed = runner.run(control())
    received = []

    replay = runner.run(control(clock=lambda: 10),
                        receive_message=lambda ids: received.append(ids))

    assert replay == completed
    assert received == [] and len(model.requests) == 1 and tools.requests == []
    with pytest.raises(ValueError, match='budget exhausted'):
        runner.continue_with(control(clock=lambda: 10), message_id='late', text='Continue')


@pytest.mark.parametrize('stronger', ['cancel', 'revoke'])
def test_cancellation_or_revocation_wins_over_expired_terminal_replay(stronger):
    journal = ConversationLog()
    runner = session(ScriptedModel([delivery()]), MemoryTools(), journal)
    runner.run(control())
    token = CancellationToken()
    if stronger == 'cancel':
        token.cancel()
    outcome = runner.run(control(cancel=token, current=lambda: stronger != 'revoke',
                                 clock=lambda: 10))
    assert (outcome.status, outcome.reason) == (
        ('cancelled', 'CANCEL_REQUESTED') if stronger == 'cancel'
        else ('rejected', 'BINDING_REVOKED'))


def test_expired_pending_effect_keeps_uncertainty_without_reexecution():
    class LostEffect(MemoryTools):
        def execute(self, request, execution):
            self.requests.append(request)
            raise RuntimeError('effect may have run; receipt lost')

    journal, tools = ConversationLog(), LostEffect()
    model = ScriptedModel([tool('read')])
    runner = session(model, tools, journal)
    assert runner.run(control()).status == 'reconciliation_required'
    before = journal.snapshot()
    replay = runner.run(control(clock=lambda: 10))
    assert replay.reason == 'DEADLINE' and replay.pending_id == 's/effect/1'
    assert len(model.requests) == len(tools.requests) == 1
    assert journal.snapshot() == before


def test_expiry_does_not_complete_recorded_reply_or_pending_invocation():
    now = [0]
    def reply_at_deadline(request, execution):
        now[0] = 10
        return delivery()
    journal = ConversationLog()
    model = ScriptedModel([reply_at_deadline])
    runner = session(model, MemoryTools(), journal)
    assert runner.run(control(clock=lambda: now[0])).reason == 'DEADLINE'
    assert runner.run(control(clock=lambda: now[0])).reason == 'DEADLINE'
    assert runner.run(control()).status == 'proposed'

    pending_log = ConversationLog()
    pending = session(ScriptedModel([RuntimeError('lost')]), MemoryTools(), pending_log)
    pending.run(control())
    outcome = pending.run(control(clock=lambda: 10))
    assert outcome.reason == 'DEADLINE' and outcome.pending_id == 's/model/1'


@pytest.mark.parametrize("reason", ["deadline", "revocation"])
def test_control_change_after_model_response_prevents_effect(reason):
    state = {"now": 0, "current": True}
    journal, tools = ConversationLog(), MemoryTools()

    def respond(request, execution):
        state["now" if reason == "deadline" else "current"] = 11 if reason == "deadline" else False
        return tool("read")

    model = ScriptedModel([respond])
    outcome = session(model, tools, journal).run(control(
        clock=lambda: state["now"], current=lambda: state["current"]))
    assert outcome.status == ("exhausted" if reason == "deadline" else "rejected")
    assert not tools.requests


def test_cancel_after_effect_keeps_receipt_and_resume_never_repeats_edit():
    cancel = CancellationToken()

    class CancelTools(MemoryTools):
        def execute(self, request, execution):
            result = super().execute(request, execution)
            cancel.cancel()
            return result

    journal, tools = ConversationLog(), CancelTools()
    model = ScriptedModel([tool("edit", expected_version=hashlib.sha256(b"old").hexdigest(),
                                text="new"), delivery()])
    assert session(model, tools, journal).run(control(cancel=cancel)).status == "cancelled"
    assert session(model, tools, journal).run(control()).status == "proposed"
    assert tools.content == "new" and len(tools.requests) == 1


@pytest.mark.parametrize("args", [{"expected_version": "stale", "text": "new"},
                                {"path": "../escape.py", "expected_version": "x", "text": "new"}])
def test_capability_refusal_is_preserved_as_evidence(args):
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([tool("edit", **args), delivery()])
    assert session(model, tools, journal).run(control()).status == "proposed"
    assert tools.content == "old"  # Proposal is not admission of a successful Work.
    result = next(r.event() for r in journal.snapshot().records if r.event()["kind"] == "effect_returned")
    assert result["result"]["status"] == "refused"


def test_reused_tool_call_id_cannot_dispatch_twice():
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([tool("read"), tool("read")])
    assert session(model, tools, journal).run(control()).reason == "TOOL_NOT_ADMITTED_OR_DUPLICATE"
    assert len(tools.requests) == 1


@pytest.mark.parametrize("failure", ["exception", "wrong_id"])
def test_tool_may_change_state_before_fault_but_no_retry_is_assumed(failure):
    class BrokenReceiptTools(MemoryTools):
        def execute(self, request, execution):
            result = super().execute(request, execution)
            if failure == "exception":
                raise RuntimeError("failed after mutation")
            return EffectResult("another-effect", result.status, result.output_json)

    journal, tools = ConversationLog(), BrokenReceiptTools()
    model = ScriptedModel([tool("edit", text="new", expected_version=hashlib.sha256(b"old").hexdigest())])
    result = session(model, tools, journal).run(control())
    assert result.status == "reconciliation_required"
    assert session(model, tools, journal).run(control()) == result
    assert len(tools.requests) == 1 and tools.content == "new"


def test_log_capacity_failure_after_effect_preserves_pending_intent():
    from tare_tools_agent_runtime import ConversationLimits, ConversationRejected

    journal = ConversationLog(ConversationLimits(max_events=4))
    tools = MemoryTools()
    model = ScriptedModel([tool("edit", text="new", expected_version=hashlib.sha256(b"old").hexdigest())])
    with pytest.raises(ConversationRejected):
        session(model, tools, journal).run(control())
    assert session(model, tools, journal).run(control()).status == "reconciliation_required"
    assert len(tools.requests) == 1 and tools.content == "new"


def test_cancelled_entry_never_calls_model_or_tool():
    cancel = CancellationToken()
    cancel.cancel()
    journal, tools, model = ConversationLog(), MemoryTools(), ScriptedModel([])
    assert session(model, tools, journal).run(control(cancel=cancel)).status == "cancelled"
    assert model.requests == tools.requests == []


@pytest.mark.parametrize("kind", ["model_requested", "effect_requested"])
def test_cancel_between_intent_and_dispatch_records_known_nonexecution(kind):
    cancel = CancellationToken()

    class CancelJournal(ConversationLog):
        armed = True

        def append(self, event, **kwargs):
            result = super().append(event, **kwargs)
            if self.armed and event["kind"] == kind:
                self.armed = False
                cancel.cancel()
            return result

    journal, tools = CancelJournal(), MemoryTools()
    model = ScriptedModel([tool("read"), delivery()])
    paused = session(model, tools, journal).run(control(cancel=cancel))
    assert paused.status == "cancelled" and paused.pending_id is None
    assert journal.snapshot().records[-1].event()["kind"] == "invocation_not_started"
    assert not tools.requests
    assert session(model, tools, journal).run(control()).status == "proposed"
    assert len(tools.requests) == 1 and len(model.requests) == 2


def test_native_cycle_loads_without_site_packages_or_incumbent():
    source = str(Path(__file__).resolve().parents[1] / "src")
    script = '''
import sys
sys.path.insert(0, sys.argv[1])
from tare_tools_agent_runtime import BindingRef, CancellationToken, ConversationLog, ExecutionControl
from tare_tools_agent_runtime.native import NativeSession, ModelReply
class Model:
    def generate(self, request, control): return ModelReply(text="proposal", proposes_delivery=True)
class Tools:
    def execute(self, request, control): raise AssertionError("no tool requested")
session = NativeSession(session_id="isolated", instruction="inspect", capabilities=(),
                        model=Model(), tools=Tools(), journal=ConversationLog())
control = ExecutionControl(BindingRef("w", "b", "o", 1, "token"), CancellationToken(), 1, lambda: 0)
assert session.run(control).status == "proposed"
assert not any(name.split(".")[0] in {"aider", "litellm", "compute_plane", "relay", "strands"}
               for name in sys.modules)
print("PASS: native cycle without site packages")
'''
    result = subprocess.run([sys.executable, "-S", "-c", script, source],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_unlimited_calls_and_stagnation_pass_old_ceiling_and_keep_counts():
    model = ScriptedModel([ModelReply(text='working')] * 20 + [delivery()])
    log = ConversationLog()
    budget = NativeBudget(max_model_calls=None, max_stagnant_turns=None)
    runner = session(model, MemoryTools(), log, budget=budget)
    result = runner.run(control())
    assert (result.status, result.model_calls) == ('proposed', 21)
    assert runner.run(control()) == result and len(model.requests) == 21
    with pytest.raises(ValueError, match='identity/configuration'):
        session(model, MemoryTools(), log).run(control())


def test_unlimited_retry_only_retries_retryable_failures_and_counts_them():
    log = ConversationLog()
    model = ScriptedModel([ModelFailure('busy', retryable=True)] * 4 + [delivery()])
    result = session(model, MemoryTools(), log, budget=NativeBudget(max_model_retries=None)).run(control())
    assert (result.status, result.model_calls) == ('proposed', 5)
    blocked = session(ScriptedModel([ModelFailure('auth')]), MemoryTools(), ConversationLog(),
                      budget=NativeBudget(None, None, None)).run(control())
    assert blocked.status == 'blocked' and blocked.model_calls == 1


def test_unlimited_session_still_cancels_and_checks_authority():
    for revoked in (False, True):
        cancel = CancellationToken()
        def stop(request, execution):
            cancel.cancel()
            return delivery()
        ctl = control(cancel=cancel, current=lambda: not revoked)
        result = session(ScriptedModel([stop]), MemoryTools(), ConversationLog(),
                         budget=NativeBudget(None, None, None)).run(ctl)
        assert result.status == ('rejected' if revoked else 'cancelled')


def test_more_than_64_followups_with_unlimited_calls():
    log, tools = ConversationLog(), MemoryTools()
    budget = NativeBudget(None, None, None)
    runner = session(ScriptedModel([delivery()] * 67), tools, log, budget=budget)
    assert runner.run(control()).status == 'proposed'
    for i in range(66):
        assert runner.continue_with(control(), message_id=str(i), text='Continue')
        assert runner.run(control()).model_calls == i + 2
    replay = session(ScriptedModel([]), tools, log, budget=budget)
    assert replay.run(control()).status == 'proposed'
    assert not replay.continue_with(control(), message_id='65', text='Continue')
