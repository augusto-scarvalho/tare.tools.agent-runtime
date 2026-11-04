"""A bounded delivery fits its host journal without relaxing tool arguments."""
import json

import pytest

from tare_tools_agent_runtime import ConversationLimits, ConversationLog, ConversationRejected
from tare_tools_agent_runtime.native import ModelReply, NativeBudget, ToolCall
from test_native import ScriptedModel, control, session


def compact(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def test_bounded_delivery_metadata_survives_the_real_session_unchanged():
    payload = {'summary': 'A bounded delivery', 'criteria': ['\\"'*500 for _ in range(11)]}
    assert len(compact(payload).encode()) < 32768
    arguments = compact(payload)
    metadata = {'assistant': {'role': 'assistant', 'content': '', 'tool_calls': [
        {'id': 'delivery', 'type': 'function', 'function': {'name': 'deliver', 'arguments': arguments}}]},
        'delivery_payload': payload}
    raw = compact(metadata)
    assert 64*1024 < len(raw.encode()) < 128*1024
    reply = ModelReply(text=payload['summary'], proposes_delivery=True, metadata_json=raw)
    journal = ConversationLog(ConversationLimits(max_event_bytes=256*1024))
    model = ScriptedModel([reply])
    result = session(model, None, journal, budget=NativeBudget(max_model_calls=1)).run(control())
    assert result.status == 'proposed' and result.model_calls == 1
    returned = next(r.event() for r in journal.snapshot().records if r.event()['kind'] == 'model_returned')
    assert returned['reply']['metadata_json'] == raw
    assert json.loads(returned['reply']['metadata_json'])['assistant']['tool_calls'][0]['function']['arguments'] == arguments


def test_reply_metadata_still_has_a_finite_ingress_bound():
    with pytest.raises(ConversationRejected) as error:
        ModelReply(metadata_json=compact({'value': 'x'*(128*1024)}))
    assert error.value.reason.value == 'EVENT_TOO_LARGE'


def test_tool_argument_bound_is_not_lifted_with_reply_metadata():
    with pytest.raises(ConversationRejected) as error:
        ToolCall('call', 'edit', compact({'value': 'x'*(64*1024)}))
    assert error.value.reason.value == 'EVENT_TOO_LARGE'


def test_reply_metadata_still_rejects_duplicate_keys():
    with pytest.raises(ConversationRejected):
        ModelReply(metadata_json='{"value":1,"value":2}')
