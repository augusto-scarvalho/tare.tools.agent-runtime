"""A proposal can receive new input without losing the prior session's limits."""
import pytest
from tare_tools_agent_runtime import ConversationLog, NativeBudget
from test_native import ScriptedModel, MemoryTools, session, control, delivery, tool


def test_continuation_preserves_history_calls_and_idempotence():
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([delivery(), delivery(), delivery()])
    run = session(model, tools, journal)
    run.run(control())
    assert run.continue_with(control(), message_id='m1', text='Explain the change')
    assert not run.continue_with(control(), message_id='m1', text='Explain the change')
    result = session(model, tools, journal).run(control())
    assert result.model_calls == 2 and result.status == 'proposed'
    assert [e['kind'] for e in model.requests[-1].context.events()] == ['model_returned', 'user_message']
    assert model.requests[-1].instruction == model.requests[0].instruction
    assert not run.continue_with(control(), message_id='m1', text='Explain the change')
    assert run.run(control()).model_calls == 2
    with pytest.raises(ValueError, match='conflicting'):
        run.continue_with(control(), message_id='m1', text='Different')
    run.continue_with(control(), message_id='m2', text='Continue')
    run.run(control())
    with pytest.raises(ValueError, match='stale'):
        run.continue_with(control(), message_id='m1', text='Explain the change')
    assert len(model.requests) == 3


@pytest.mark.parametrize('mode', ['pending', 'blocked', 'calls', 'deadline', 'binding'])
def test_new_input_cannot_clear_a_stop_or_unknown_invocation(mode):
    journal, tools = ConversationLog(), MemoryTools()
    reply = RuntimeError('unknown') if mode=='pending' else tool('not-admitted') if mode=='blocked' else delivery()
    run = session(ScriptedModel([reply]), tools, journal, budget=NativeBudget(max_model_calls=1 if mode=='calls' else 4))
    run.run(control())
    before = journal.snapshot().revision
    execution = control(fence=2) if mode=='binding' else control(clock=lambda:11) if mode=='deadline' else control()
    with pytest.raises(ValueError):
        run.continue_with(execution, message_id='m1', text='Try again')
    assert journal.snapshot().revision == before


def test_reducer_rejects_forged_continuation_and_keeps_legacy_terminal_rule():
    journal, tools = ConversationLog(), MemoryTools()
    run = session(ScriptedModel([delivery()]), tools, journal)
    run.run(control())
    journal.append({'kind':'user_message','message_id':'m','text':'next','after_sequence':999})
    with pytest.raises(ValueError, match='continuation'):
        run.run(control())


def test_compaction_preserves_explicit_input_and_counts_summary():
    from tare_tools_agent_runtime import ContextPolicy, NativeContext
    from test_native_context import Counter, Model
    journal, tools = ConversationLog(), MemoryTools()
    counter, summary = Counter(), Model()
    context = NativeContext(ContextPolicy(1000, 100, 50), counter=counter.count, summarizer=summary)
    model = ScriptedModel([delivery(), delivery(), delivery()])
    run = session(model, tools, journal, context=context)
    run.run(control())
    run.continue_with(control(), message_id='m1', text='Preserve this exact correction')
    run.run(control())
    run.continue_with(control(), message_id='m2', text='And preserve this question')
    result = run.run(control())
    assert result.status == 'proposed' and result.model_calls == 4
    events = model.requests[-1].context.events()
    assert [e['text'] for e in events if e['kind']=='user_message'] == [
        'Preserve this exact correction','And preserve this question']
    assert any(e['kind']=='context_activated' for e in events)
    assert any(e.get('text')=='Preserve this exact correction' for e in summary.requests[0].context.events())
