from dataclasses import asdict, replace
import hashlib

import pytest

from tare_tools_agent_runtime import (
    AttachmentRef, ConversationLog, ContextPolicy, NativeContext, NativeMessage, NativeSession,
)
from tare_tools_agent_runtime.native_content import attachments_from_records
from test_native import MemoryTools, ScriptedModel, control, delivery, session, tool


def attachment(name='error.png', body=b'opaque screenshot snapshot'):
    return AttachmentRef(name, 'image/png', hashlib.sha256(body).hexdigest(), len(body))


def test_initial_attachment_identity_survives_reopen_without_loading_bytes():
    ref = attachment()
    log, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([delivery()])
    run = session(model, tools, log, attachments=(ref,))
    result = run.run(control())
    assert model.requests[0].attachments == (ref,)
    assert log.snapshot().records[0].event()['attachments'] == [asdict(ref)]
    assert session(model, tools, log, attachments=(ref,)).run(control()) == result
    assert len(model.requests) == 1 and not tools.requests
    with pytest.raises(ValueError, match='identity/configuration'):
        session(model, tools, log, attachments=(attachment(body=b'changed'),)).run(control())
    with pytest.raises(ValueError, match='identity/configuration'):
        session(model, tools, log).run(control())
    # A screenshot alone is a valid user input, with no invented user text.
    standalone = NativeSession(session_id='image-only', instruction='', capabilities=(),
        model=ScriptedModel([delivery()]), tools=tools, journal=ConversationLog(), attachments=(ref,))
    assert standalone.run(control()).status == 'proposed'


def test_attachment_only_followup_and_retries_compare_the_complete_message():
    ref = attachment()
    log, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([delivery(), delivery()])
    run = session(model, tools, log)
    run.run(control())
    assert run.continue_with(control(), message_id='screen', text='', attachments=(ref,))
    assert not run.continue_with(control(), message_id='screen', text='', attachments=(ref,))
    assert session(model, tools, log).run(control()).model_calls == 2
    event = next(e for e in model.requests[-1].context.events() if e['kind'] == 'user_message')
    assert event['attachments'] == [asdict(ref)] and event['text'] == ''
    event['attachments'][0]['sha256'] = 'f'*64
    assert NativeMessage.from_event(next(e for e in model.requests[-1].context.events()
                                        if e['kind'] == 'user_message')).attachments == (ref,)
    for changed in (attachment(body=b'changed'), replace(ref, name='other.png'), replace(ref, media_type='image/jpeg')):
        with pytest.raises(ValueError, match='conflicting'):
            run.continue_with(control(), message_id='screen', text='', attachments=(changed,))
    assert not run.continue_with(control(), message_id='screen', text='', attachments=(ref,))
    assert len(model.requests) == 2


def test_live_attachment_waits_for_the_whole_tool_batch_and_replays_once():
    ref = attachment()
    log, tools = ConversationLog(), MemoryTools()
    calls = tool('read', 'one').tool_calls + tool('read', 'two').tool_calls
    from tare_tools_agent_runtime import ModelReply
    def receive(consumed):
        assert len(tools.requests) != 1
        return NativeMessage('screen', 'Use this error', (ref,)) if tools.requests and not consumed else None
    model = ScriptedModel([ModelReply(tool_calls=calls), delivery()])
    result = session(model, tools, log).run(control(), receive_message=receive)
    assert result.status == 'proposed' and len(tools.requests) == 2
    assert any(e.get('attachments') == [asdict(ref)] for e in model.requests[-1].context.events())
    assert session(model, tools, log).run(control(), receive_message=receive) == result
    assert len(model.requests) == 2


def test_crash_after_attachment_append_does_not_repeat_delivery():
    class CrashLog(ConversationLog):
        crash = True
        def append(self, event):
            result = super().append(event)
            if event['kind'] == 'user_message' and self.crash:
                self.crash = False
                raise RuntimeError('crash after durable append')
            return result
    ref = attachment()
    log, tools = CrashLog(), MemoryTools()
    model = ScriptedModel([delivery()])
    receive = lambda consumed: None if 'screen' in consumed else NativeMessage('screen', '', (ref,))
    with pytest.raises(RuntimeError, match='durable'):
        session(model, tools, log).run(control(), receive_message=receive)
    assert session(model, tools, log).run(control(), receive_message=receive).status == 'proposed'
    assert len(model.requests) == 1
    assert sum(r.event()['kind'] == 'user_message' for r in log.snapshot().records) == 1


def test_compaction_preserves_initial_and_message_attachment_identities():
    from test_native_context import Counter, Model
    initial, later = attachment(), attachment('later.png', b'later screenshot')
    log, tools = ConversationLog(), MemoryTools()
    summary, counter = Model(), Counter()
    context = NativeContext(ContextPolicy(1000, 100, 50), counter=counter.count, summarizer=summary)
    model = ScriptedModel([delivery(), delivery(), delivery()])
    run = session(model, tools, log, context=context, attachments=(initial,))
    run.run(control())
    run.continue_with(control(), message_id='screen', text='Compare this', attachments=(later,))
    run.run(control())
    run.continue_with(control(), message_id='next', text='Continue')
    assert run.run(control()).model_calls == 4
    for request in [summary.requests[0], model.requests[-1]]:
        assert request.attachments == (initial,)
        assert any(e.get('attachments') == [asdict(later)] for e in request.context.events())
    assert any(e['kind'] == 'context_activated' for e in model.requests[-1].context.events())


@pytest.mark.parametrize('change', [
    {'sha256': 'invalid'}, {'size_bytes': True}, {'size_bytes': -1}, {'size_bytes': 17*1024*1024},
    {'media_type': 'image/png; charset=utf8'}, {'name': 'bad\x1bname'}, {'path': '/outside/file.png'},
])
def test_malformed_attachment_metadata_is_rejected_before_a_model_call(change):
    log, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([delivery()])
    run = session(model, tools, log)
    run.run(control())
    record = asdict(attachment()) | change
    log.append({'kind': 'user_message', 'message_id': 'bad', 'text': 'Inspect',
                'after_sequence': log.snapshot().records[-1].sequence, 'attachments': [record]})
    with pytest.raises((TypeError, ValueError)):
        run.run(control())
    assert len(model.requests) == 1 and not tools.requests


def test_reference_limits_and_legacy_message_event_remain_explicit():
    ref = attachment()
    with pytest.raises(ValueError, match='duplicate'):
        NativeMessage('m', 'Inspect', (ref, ref))
    with pytest.raises(ValueError):
        attachments_from_records([asdict(replace(ref, name=str(i))) for i in range(17)])
    with pytest.raises(TypeError):
        NativeMessage('m', 'Inspect', [ref])
    assert NativeMessage('m', 'Legacy').event(2) == {
        'kind': 'user_message', 'message_id': 'm', 'text': 'Legacy', 'after_sequence': 2}
    assert 'PRIVATE' not in repr(NativeMessage('m', 'PRIVATE', (ref,)))
