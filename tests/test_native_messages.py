import pytest

from tare_tools_agent_runtime import ConversationLog, CancellationToken
from tare_tools_agent_runtime.native import ModelReply, NativeBudget
from test_native import MemoryTools, ScriptedModel, control, delivery, session, tool


def test_message_enters_after_whole_tool_batch_and_survives_replay():
    log, tools = ConversationLog(), MemoryTools()
    calls = tool('read', 'one').tool_calls + tool('read', 'two').tool_calls
    def receive(consumed):
        assert len(tools.requests) != 1  # Never split a tool batch with user input.
        return ('correction', 'Only explain; do not edit.') if tools.requests and not consumed else None
    model = ScriptedModel([ModelReply(tool_calls=calls), delivery()])
    result = session(model, tools, log).run(control(), receive_message=receive)
    assert result.status == 'proposed' and len(tools.requests) == 2
    assert any(e.get('text') == 'Only explain; do not edit.' for e in model.requests[-1].context.events())
    assert session(model, tools, log).run(control(), receive_message=receive) == result
    assert len(model.requests) == 2
    assert sum(r.event()['kind'] == 'user_message' for r in log.snapshot().records) == 1


def test_message_arriving_during_final_generation_gets_a_new_response():
    log, tools, queued = ConversationLog(), MemoryTools(), []
    def final(request, execution):
        queued.append(('later', 'Now explain the exception.'))
        return delivery()
    def receive(consumed):
        return queued[0] if queued and 'later' not in consumed else None
    model = ScriptedModel([final, delivery()])
    assert session(model, tools, log).run(control(), receive_message=receive).model_calls == 2
    assert model.requests[-1].context.events()[-1]['text'] == 'Now explain the exception.'


def test_crash_after_message_append_does_not_consume_twice():
    class CrashLog(ConversationLog):
        crash = True
        def append(self, event):
            result = super().append(event)
            if event['kind'] == 'user_message' and self.crash:
                self.crash = False
                raise RuntimeError('crash after durable append')
            return result
    log, tools = CrashLog(), MemoryTools()
    receive = lambda ids: None if 'm' in ids else ('m', 'Use the new instruction.')
    model = ScriptedModel([delivery()])
    with pytest.raises(RuntimeError, match='durable'):
        session(model, tools, log).run(control(), receive_message=receive)
    assert session(model, tools, log).run(control(), receive_message=receive).status == 'proposed'
    assert len(model.requests) == 1


def test_exhausted_budget_leaves_new_message_with_host():
    log, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([delivery()])
    budget = NativeBudget(max_model_calls=1)
    first = session(model, tools, log, budget=budget).run(control())
    def receive(ids):
        raise AssertionError('Must not acknowledge input without budget to act on it')
    assert session(model, tools, log, budget=budget).run(control(), receive_message=receive) == first


def test_cancel_after_receive_leaves_message_unacknowledged():
    log, tools, token = ConversationLog(), MemoryTools(), CancellationToken()
    def receive(ids):
        token.cancel()
        return 'm', 'A correction'
    result = session(ScriptedModel([]), tools, log).run(control(cancel=token), receive_message=receive)
    assert result.status == 'cancelled'
    assert not any(r.event()['kind'] == 'user_message' for r in log.snapshot().records)
