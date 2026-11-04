"""Explicit model-response discard cannot resolve or replay uncertain effects."""
import pytest

from tare_tools_agent_runtime import CancellationToken, ConversationLog
from tare_tools_agent_runtime.native import EffectResult, NativeBudget
from test_native import control, delivery, MemoryTools, ScriptedModel, session, tool


@pytest.mark.parametrize('not_started,limit', [(True, 4), (False, 4), (True, 2)])
def test_explicit_preflight_retry_keeps_effects_history_and_budget(not_started, limit):
    from tare_tools_agent_runtime import ModelFailure
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([tool('read'), ModelFailure('CLI_VERSION_CHANGED', not_started=not_started), delivery()])
    runner = session(model, tools, journal, budget=NativeBudget(max_model_calls=limit))
    assert runner.run(control()).status == 'blocked'
    before = journal.snapshot()
    assert runner.run(control()).status == 'blocked'
    assert journal.snapshot() == before and len(model.requests) == 2
    if not not_started or limit == 2:
        with pytest.raises(ValueError):
            runner.retry_not_started_model(control(), call_id='s/model/2', resolution_id='r')
        assert journal.snapshot() == before
        return
    with pytest.raises(ValueError):
        runner.retry_not_started_model(control(fence=2), call_id='s/model/2', resolution_id='r')
    assert runner.retry_not_started_model(control(), call_id='s/model/2', resolution_id='r')
    assert not runner.retry_not_started_model(control(), call_id='s/model/2', resolution_id='r')
    outcome = runner.run(control())
    assert outcome.status == 'proposed' and outcome.model_calls == 3
    assert len(tools.requests) == 1
    assert journal.snapshot().records[:len(before.records)] == before.records


def test_forged_retry_cannot_clear_a_dispatched_failure():
    from tare_tools_agent_runtime import ModelFailure
    journal = ConversationLog()
    runner = session(ScriptedModel([ModelFailure('failure')]), MemoryTools(), journal)
    runner.run(control())
    journal.append({'kind': 'model_retry_authorized', 'call_id': 's/model/1', 'resolution_id': 'r'})
    with pytest.raises(ValueError, match='retry authorization'):
        runner.run(control())


def test_explicit_recovery_retains_calls_and_effects_and_accepts_queued_message():
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([tool('read'), RuntimeError('interrupted'), delivery()])
    runner = session(model, tools, journal)
    assert runner.run(control()).status == 'reconciliation_required'
    with pytest.raises(ValueError):
        runner.abandon_interrupted_model(control(fence=2), call_id='s/model/2', resolution_id='r')
    assert runner.abandon_interrupted_model(control(), call_id='s/model/2', resolution_id='r')
    assert not runner.abandon_interrupted_model(control(), call_id='s/model/2', resolution_id='r')
    with pytest.raises(ValueError):
        runner.abandon_interrupted_model(control(), call_id='s/model/3', resolution_id='r')
    outcome = runner.run(control(), receive_message=lambda ids: None if ids else ('correction', 'Only explain.'))
    assert outcome.status == 'proposed' and outcome.model_calls == 3
    assert len(tools.requests) == 1 and len(model.requests) == 3
    assert model.requests[-1].call_id == 's/model/3'
    assert any(e.get('message_id') == 'correction' for e in model.requests[-1].context.events())
    records = [r.event() for r in journal.snapshot().records]
    abandoned = next(e for e in records if e['kind'] == 'model_abandoned')
    assert abandoned['usage'] is None and abandoned['remote_stop'] == 'unknown'
    assert any(e['kind'] == 'invocation_interrupted' for e in records)


@pytest.mark.parametrize('stopped', ['cancelled','deadline','revoked'])
def test_stopped_control_cannot_authorize_recovery(stopped):
    journal = ConversationLog()
    runner = session(ScriptedModel([RuntimeError()]), MemoryTools(), journal)
    runner.run(control()); before = journal.snapshot()
    cancel = CancellationToken()
    if stopped == 'cancelled': cancel.cancel()
    check = control(cancel=cancel, current=lambda:stopped != 'revoked', clock=lambda:20 if stopped=='deadline' else 0)
    with pytest.raises(ValueError):
        runner.abandon_interrupted_model(check, call_id='s/model/1', resolution_id='r')
    assert journal.snapshot() == before


def test_recovery_cannot_clear_unknown_tool_or_renew_call_budget():
    class UnknownTools(MemoryTools):
        def execute(self, request, execution):
            self.requests.append(request)
            return EffectResult(request.effect_id, 'unknown', '{}')
    journal, tools = ConversationLog(), UnknownTools()
    runner = session(ScriptedModel([tool('read')]), tools, journal)
    assert runner.run(control()).status == 'reconciliation_required'
    before = journal.snapshot()
    with pytest.raises(ValueError):
        runner.abandon_interrupted_model(control(), call_id='s/effect/1', resolution_id='r')
    assert journal.snapshot() == before and len(tools.requests) == 1
    journal = ConversationLog()
    runner = session(ScriptedModel([RuntimeError(), delivery()]), MemoryTools(), journal,
                     budget=NativeBudget(max_model_calls=1))
    runner.run(control())
    runner.abandon_interrupted_model(control(), call_id='s/model/1', resolution_id='r')
    assert runner.run(control()).status == 'exhausted'
    assert len(runner.model.requests) == 1


def test_intent_without_interruption_and_forged_resolution_are_rejected():
    journal = ConversationLog()
    runner = session(ScriptedModel([]), MemoryTools(), journal)
    journal.append(runner._header({'work_id':'work','binding_id':'binding','owner_id':'owner','fence':1}))
    journal.append({'kind':'model_requested','call_id':'s/model/1'})
    with pytest.raises(ValueError):
        runner.abandon_interrupted_model(control(), call_id='s/model/1', resolution_id='r')
    journal.append({'kind':'model_abandoned','call_id':'s/model/1','resolution_id':'r','remote_stop':'unknown','usage':None})
    with pytest.raises(ValueError, match='abandonment'):
        runner.run(control())
