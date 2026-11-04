"""Explicit host grants extend only the call ceiling and survive journal replay."""
import pytest
from tare_tools_agent_runtime import ConversationLog, NativeBudget
from test_native import ScriptedModel, MemoryTools, session, control, delivery


def test_grant_preserves_initial_identity_and_consumption_across_restart():
    journal,tools=ConversationLog(),MemoryTools()
    model=ScriptedModel([delivery(),delivery(),delivery()])
    budget=NativeBudget(max_model_calls=1)
    first=session(model,tools,journal,budget=budget)
    assert first.run(control()).model_calls==1
    with pytest.raises(ValueError,match='budget exhausted'):
        first.continue_with(control(),message_id='ordinary',text='Continue')
    assert first.continue_with(control(),message_id='grant',text='Authorize three total calls',model_call_limit=3)
    assert not first.continue_with(control(),message_id='grant',text='Authorize three total calls',model_call_limit=3)
    second=session(model,tools,journal,budget=budget)
    assert second.run(control()).model_calls==2
    assert model.requests[-1].remaining_model_calls==2
    second.continue_with(control(),message_id='last',text='One more')
    assert second.run(control()).model_calls==3
    with pytest.raises(ValueError,match='budget exhausted'):
        second.continue_with(control(),message_id='spent',text='No more calls')
    assert journal.snapshot().records[0].event()['budget']['max_model_calls']==1
    assert len(model.requests)==3


@pytest.mark.parametrize('limit',[False,0,-1,1.5,'3'])
def test_invalid_grants_append_nothing(limit):
    journal=ConversationLog();run=session(ScriptedModel([delivery()]),MemoryTools(),journal)
    run.run(control());before=journal.snapshot().revision
    with pytest.raises(ValueError,match='call limit grant'):
        run.continue_with(control(),message_id='bad',text='Continue',model_call_limit=limit)
    assert journal.snapshot().revision==before


@pytest.mark.parametrize('mode',['decrease','deadline','binding','pending','cancel'])
def test_grants_cannot_clear_other_execution_guards(mode):
    journal=ConversationLog()
    run=session(ScriptedModel([RuntimeError('unknown') if mode=='pending' else delivery()]),MemoryTools(),journal,
                budget=NativeBudget(max_model_calls=2))
    run.run(control());before=journal.snapshot().revision
    execution=control(fence=2) if mode=='binding' else control(clock=lambda:11) if mode=='deadline' else control()
    if mode=='cancel':execution.cancellation.cancel()
    with pytest.raises(ValueError):
        run.continue_with(execution,message_id='grant',text='Continue',model_call_limit=1 if mode=='decrease' else 4)
    assert journal.snapshot().revision==before
