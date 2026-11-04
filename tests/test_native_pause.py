"""Offline pause boundaries and replay; no provider, clock waits or external effects."""
import pytest

from tare_tools_agent_runtime import CancellationToken, ConversationLog
from tare_tools_agent_runtime.native import EffectResult, ModelReply, NativeBudget
from test_native import MemoryTools, ScriptedModel, control, delivery, session, tool


def events(log):
    return [record.event() for record in log.snapshot().records]


def test_pause_before_first_call_checks_once_and_leaves_input_with_host():
    log, tools, checks = ConversationLog(), MemoryTools(), []
    model = ScriptedModel([delivery()])
    def pause():
        checks.append(True)
        assert len(checks) == 1  # No polling/busy wait.
        return True
    def receive(ids):
        pytest.fail('paused loop must not receive queued human input')
    result = session(model, tools, log).run(
        control(), pause_requested=pause, receive_message=receive)
    assert (result.status, result.reason, result.model_calls) == ('paused', 'PAUSE_REQUESTED', 0)
    assert result.pending_id is None and result.output == ''
    assert not model.requests and not tools.requests and checks == [True]
    assert [event['kind'] for event in events(log)] == ['session_started']
    revision = log.snapshot().revision
    assert session(model, tools, log).run(control(), pause_requested=lambda: True) == result
    assert log.snapshot().revision == revision


@pytest.mark.parametrize('raised_in', ['model', 'first_tool'])
def test_complete_original_batch_then_reconstruct_and_resume_without_repeated_effects(raised_in):
    log, tools, requested = ConversationLog(), MemoryTools(), [False]
    calls = tool('read', 'one').tool_calls + tool('read', 'two').tool_calls
    def round_reply(request, execution):
        if raised_in == 'model':
            requested[0] = True
        return ModelReply(tool_calls=calls)
    model = ScriptedModel([round_reply, delivery()])
    execute = tools.execute
    def execute_and_pause(request, execution):
        result = execute(request, execution)
        if raised_in == 'first_tool' and len(tools.requests) == 1:
            requested[0] = True
        return result
    tools.execute = execute_and_pause
    queued = []
    def receive(ids):
        if tools.requests:
            pytest.fail('pause must precede queued input after the full batch')
        return None
    paused = session(model, tools, log).run(
        control(), pause_requested=lambda: requested[0], receive_message=receive)
    assert (paused.status, paused.model_calls) == ('paused', 1)
    assert [r.tool_call.call_id for r in tools.requests] == ['one', 'two']
    assert len(model.requests) == 1
    assert not any(e['kind'] in {'terminal', 'user_message'} for e in events(log))
    # Rebuild both the journal consumer and session, not just a second run on one object.
    restored = ConversationLog()
    for event in events(log):
        restored.append(event)
    before = restored.snapshot().revision
    assert session(model, tools, restored).run(control(), pause_requested=lambda: True) == paused
    assert restored.snapshot().revision == before
    def resumed_receive(ids):
        queued.append(ids)
        return ('human', 'Explain the result.') if 'human' not in ids else None
    outcome = session(model, tools, restored).run(
        control(), pause_requested=lambda: False, receive_message=resumed_receive)
    assert outcome.status == 'proposed' and outcome.model_calls == 2
    assert len(tools.requests) == 2 and len({r.effect_id for r in tools.requests}) == 2
    assert len(model.requests) == 2 and queued
    assert sum(e['kind'] == 'user_message' for e in events(restored)) == 1
    assert model.requests[-1].remaining_model_calls == 15


@pytest.mark.parametrize('stop,status,reason', [
    ('cancel', 'cancelled', 'CANCEL_REQUESTED'),
    ('revoke', 'rejected', 'BINDING_REVOKED'),
    ('deadline', 'exhausted', 'DEADLINE'),
])
@pytest.mark.parametrize('during_callback', [False, True])
def test_control_precedes_pause(stop, status, reason, during_callback):
    log, tools, token = ConversationLog(), MemoryTools(), CancellationToken()
    current, now, checks = [True], [0], []
    def trigger():
        if stop == 'cancel':
            token.cancel()
        elif stop == 'revoke':
            current[0] = False
        else:
            now[0] = 11
    if not during_callback:
        trigger()
    def pause():
        checks.append(True)
        trigger()
        return True
    model = ScriptedModel([])
    outcome = session(model, tools, log).run(
        control(cancel=token, current=lambda: current[0], clock=lambda: now[0]),
        pause_requested=pause)
    assert (outcome.status, outcome.reason, outcome.model_calls) == (status, reason, 0)
    assert len(checks) == int(during_callback)
    assert not model.requests and not tools.requests
    assert not any(e['kind'] == 'terminal' for e in events(log))


@pytest.mark.parametrize('unknown_in', ['model', 'tool'])
def test_unknown_invocation_is_not_a_safe_pause(unknown_in):
    log, tools, requested = ConversationLog(), MemoryTools(), [False]
    def reply(request, execution):
        requested[0] = True
        if unknown_in == 'model':
            raise RuntimeError('lost response')
        return ModelReply(tool_calls=tool('read', 'one').tool_calls + tool('read', 'two').tool_calls)
    model = ScriptedModel([reply])
    execute = tools.execute
    def unknown(request, execution):
        execute(request, execution)
        return EffectResult(request.effect_id, 'unknown')
    tools.execute = unknown
    outcome = session(model, tools, log).run(control(), pause_requested=lambda: requested[0])
    assert (outcome.status, outcome.reason, outcome.model_calls) == (
        'reconciliation_required', 'UNFINISHED_INVOCATION', 1)
    assert outcome.pending_id is not None
    assert len(tools.requests) == (0 if unknown_in == 'model' else 1)
    revision = log.snapshot().revision
    def forbidden_pause():
        pytest.fail('pending invocation must be reconciled before checking pause')
    assert session(model, tools, log).run(control(), pause_requested=forbidden_pause) == outcome
    assert log.snapshot().revision == revision and len(model.requests) == 1


def test_pause_does_not_reset_cumulative_model_call_budget():
    log, tools, requested = ConversationLog(), MemoryTools(), [False]
    def reply(request, execution):
        requested[0] = True
        return tool('read')
    model = ScriptedModel([reply, delivery()])
    budget = NativeBudget(max_model_calls=1)
    paused = session(model, tools, log, budget=budget).run(
        control(), pause_requested=lambda: requested[0])
    assert paused.status == 'paused' and paused.model_calls == 1
    outcome = session(model, tools, log, budget=budget).run(control(), pause_requested=lambda: False)
    assert (outcome.status, outcome.reason, outcome.model_calls) == ('exhausted', 'MODEL_CALL_BUDGET', 1)
    assert len(model.requests) == len(tools.requests) == 1
    with pytest.raises(ValueError, match='identity/configuration mismatch'):
        session(model, tools, log, budget=NativeBudget(max_model_calls=2)).run(
            control(), pause_requested=lambda: True)


def test_proposal_from_running_round_stays_proposed_and_leaves_queued_input():
    log, tools, requested = ConversationLog(), MemoryTools(), [False]
    def final(request, execution):
        requested[0] = True
        return delivery()
    def receive(ids):
        assert not requested[0], 'terminal proposal must not be replaced by queued input'
        return None
    model = ScriptedModel([final])
    outcome = session(model, tools, log).run(
        control(), pause_requested=lambda: requested[0], receive_message=receive)
    assert (outcome.status, outcome.reason, outcome.model_calls) == ('proposed', 'DELIVERY_PROPOSED', 1)
    assert outcome.output == delivery().text
    revision = log.snapshot().revision
    assert session(model, tools, log).run(control(), pause_requested=lambda: True) == outcome
    assert log.snapshot().revision == revision
    assert not any(e['kind'] == 'user_message' for e in events(log))
