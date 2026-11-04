"""A corrective message supersedes unstarted actions without replay or new limits."""
import pytest
from tare_tools_agent_runtime import CancellationToken, ConversationLog, ModelReply, NativeBudget, NativeMessage
from test_native import ScriptedModel, MemoryTools, session, control, delivery, tool


@pytest.mark.parametrize('stop', ['before_tools', 'between_tools', 'known_failure', 'invalid_action'])
def test_explicit_correction_preserves_completed_work_and_skips_remaining_actions(stop):
    from tare_tools_agent_runtime import ModelFailure
    log, tools, cancel = ConversationLog(), MemoryTools(), CancellationToken()
    if stop in {'before_tools','between_tools'}:
        calls=(tool('read','read-1').tool_calls[0],tool('edit','edit-1',expected_version='bad',text='wrong').tool_calls[0])
        def reply(request, execution):
            if stop=='before_tools': cancel.cancel()
            return ModelReply(tool_calls=calls)
        if stop=='between_tools':
            original=tools.execute
            def execute(request, execution):
                result=original(request,execution); cancel.cancel(); return result
            tools.execute=execute
        model=ScriptedModel([reply, delivery()])
    else:
        model=ScriptedModel([ModelFailure('known') if stop=='known_failure' else tool('not-admitted'),delivery()])
    run=session(model,tools,log,budget=NativeBudget(max_model_calls=3))
    first=run.run(control(cancel=cancel))
    assert first.status in {'cancelled','blocked'}
    before=log.snapshot().records
    assert run.continue_with(control(),message_id='correction',text='Stop that. Answer my question instead.',interrupts_turn=True)
    assert log.snapshot().records[:len(before)]==before
    assert not run.continue_with(control(),message_id='correction',text='Stop that. Answer my question instead.',interrupts_turn=True)
    result=session(model,tools,log,budget=NativeBudget(max_model_calls=3)).run(control())
    assert result.status=='proposed' and result.model_calls==2
    assert tools.content=='old' and len(tools.requests)==(1 if stop=='between_tools' else 0)
    assert model.requests[-1].instruction==model.requests[0].instruction
    last=model.requests[-1].context.events()[-1]
    assert last['kind']=='user_message' and last['interrupts_turn'] is True


@pytest.mark.parametrize('stop',['pending_model','pending_effect','calls','deadline','binding','cancelled_control'])
def test_correction_never_bypasses_unknown_effect_binding_or_limits(stop):
    log, tools=ConversationLog(),MemoryTools()
    if stop=='pending_effect':tools.execute=lambda *args: (_ for _ in ()).throw(RuntimeError('unknown'))
    reply=RuntimeError('unknown') if stop=='pending_model' else tool('read') if stop=='pending_effect' else delivery()
    run=session(ScriptedModel([reply]),tools,log,budget=NativeBudget(max_model_calls=1 if stop=='calls' else 3))
    run.run(control()); before=log.snapshot().revision
    cancel=CancellationToken()
    if stop=='cancelled_control':cancel.cancel()
    execution=control(cancel=cancel, fence=2 if stop=='binding' else 1,clock=(lambda:11) if stop=='deadline' else (lambda:0))
    with pytest.raises(ValueError):run.continue_with(execution,message_id='c',text='Correct it',interrupts_turn=True)
    assert log.snapshot().revision==before


def test_interruption_marker_is_canonical_and_legacy_messages_are_unchanged():
    message=NativeMessage('m','Correction',interrupts_turn=True)
    assert NativeMessage.from_event(message.event(1))==message
    assert 'interrupts_turn' not in NativeMessage('m','Ordinary').event(1)
    for invalid in (False,1,'yes'):
        event={**message.event(1),'interrupts_turn':invalid}
        with pytest.raises(ValueError):NativeMessage.from_event(event)
