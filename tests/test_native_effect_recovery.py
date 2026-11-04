"""Proved effect outcomes continue the journal without repeating the tool."""
from dataclasses import replace
import pytest
from tare_tools_agent_runtime import ConversationLog, EffectResult
from test_native import control, delivery, MemoryTools, ScriptedModel, session, tool


def test_reconcile_effect_requires_exact_request_and_live_binding():
    class Interrupted(MemoryTools):
        def execute(self, request, control):
            self.requests.append(request)
            raise RuntimeError('lost checkpoint after effect')
    tools, journal = Interrupted(), ConversationLog()
    runner = session(ScriptedModel([tool('read'), delivery()]), tools, journal)
    assert runner.run(control()).status == 'reconciliation_required'
    request = tools.requests[0]
    result = EffectResult(request.effect_id, 'succeeded', '{"saved":true}', True)
    for check, req, outcome in [
        (control(fence=2), request, result),
        (control(), replace(request, session_id='foreign'), result),
        (control(), replace(request, tool_call=replace(request.tool_call, arguments_json='{"other":1}')), result),
        (control(), request, replace(result, status='unknown', progressed=False))]:
        with pytest.raises(ValueError):
            runner.reconcile_effect(check, req, outcome)
    assert runner.reconcile_effect(control(), request, result)
    assert runner.run(control()).status == 'proposed'
    assert len(tools.requests) == 1
    assert len(runner.model.requests) == 2
    with pytest.raises(ValueError):
        runner.reconcile_effect(control(), request, result)
