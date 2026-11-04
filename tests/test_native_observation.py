"""Tactical continuation with host observation at a completed proposal boundary."""

import pytest
from tare_tools_agent_runtime import CancellationToken, ConversationLog, NativeBudget
from test_native import ScriptedModel, MemoryTools, session, control, delivery, tool


def test_same_session_continuation_and_cumulative_calls():
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([delivery(), delivery()])
    run = session(model, tools, journal)
    res1 = run.run(control())
    assert res1.status == 'proposed' and res1.model_calls == 1

    last = journal.snapshot().records[-1]
    assert run.continue_after_observation(
        control(),
        observation_id='obs-1',
        text='Citation verified: RFC 9110 section 8.2',
        after_sequence=last.sequence,
        after_sha256=last.sha256,
    )

    res2 = run.run(control())
    assert res2.status == 'proposed' and res2.model_calls == 2

    # Verify context projection includes host_observation separately from user_message
    events = model.requests[-1].context.events()
    obs_events = [e for e in events if e['kind'] == 'host_observation']
    assert len(obs_events) == 1
    assert obs_events[0] == {
        'kind': 'host_observation',
        'observation_id': 'obs-1',
        'text': 'Citation verified: RFC 9110 section 8.2',
        'after_sequence': last.sequence,
        'after_sha256': last.sha256,
    }
    assert all(e['kind'] != 'user_message' for e in events)


def test_exact_duplicate_noop_and_completed_replay():
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([delivery(), delivery()])
    run = session(model, tools, journal, budget=NativeBudget(max_model_calls=2))
    run.run(control())

    last = journal.snapshot().records[-1]
    # First admission succeeds
    assert run.continue_after_observation(
        control(), observation_id='obs-1', text='Citation verified',
        after_sequence=last.sequence, after_sha256=last.sha256,
    )
    rev_after_first = journal.snapshot().revision

    # Duplicate before running next round returns False without mutating
    assert not run.continue_after_observation(
        control(), observation_id='obs-1', text='Citation verified',
        after_sequence=last.sequence, after_sha256=last.sha256,
    )
    assert journal.snapshot().revision == rev_after_first

    # Continuation completes; cumulative budget (2 calls) is now exhausted
    res = run.run(control())
    assert res.status == 'proposed' and res.model_calls == 2

    # Duplicate replay after continuation completed is still a no-op returning False
    # without requiring unused call capacity
    assert not run.continue_after_observation(
        control(), observation_id='obs-1', text='Citation verified',
        after_sequence=last.sequence, after_sha256=last.sha256,
    )


def test_second_observation_rejected():
    journal, tools = ConversationLog(), MemoryTools()
    model = ScriptedModel([delivery(), delivery(), delivery()])
    run = session(model, tools, journal)
    run.run(control())

    last = journal.snapshot().records[-1]
    assert run.continue_after_observation(
        control(), observation_id='obs-1', text='First citation',
        after_sequence=last.sequence, after_sha256=last.sha256,
    )
    run.run(control())

    # Second observation with different id/payload is rejected
    last2 = journal.snapshot().records[-1]
    with pytest.raises(ValueError):
        run.continue_after_observation(
            control(), observation_id='obs-2', text='Second citation',
            after_sequence=last2.sequence, after_sha256=last2.sha256,
        )

    # Different text with same observation_id is also rejected
    with pytest.raises(ValueError):
        run.continue_after_observation(
            control(), observation_id='obs-1', text='Conflicting text',
            after_sequence=last.sequence, after_sha256=last.sha256,
        )


def test_stale_predecessor_and_hash_rejection():
    journal, tools = ConversationLog(), MemoryTools()
    run = session(ScriptedModel([delivery()]), tools, journal)
    run.run(control())
    last = journal.snapshot().records[-1]
    rev = journal.snapshot().revision

    # Wrong sequence
    with pytest.raises(ValueError):
        run.continue_after_observation(
            control(), observation_id='obs-1', text='Citation',
            after_sequence=last.sequence - 1, after_sha256=last.sha256,
        )

    # Wrong hash
    with pytest.raises(ValueError):
        run.continue_after_observation(
            control(), observation_id='obs-1', text='Citation',
            after_sequence=last.sequence, after_sha256='0' * 64,
        )
    assert journal.snapshot().revision == rev


def test_invalid_binding_rejection():
    journal, tools = ConversationLog(), MemoryTools()
    run = session(ScriptedModel([delivery()]), tools, journal)
    run.run(control())
    last = journal.snapshot().records[-1]

    with pytest.raises(ValueError):
        run.continue_after_observation(
            control(fence=2), observation_id='obs-1', text='Citation',
            after_sequence=last.sequence, after_sha256=last.sha256,
        )


@pytest.mark.parametrize('mode', ['cancelled', 'deadline', 'exhausted'])
def test_cancellation_deadline_and_exhausted_budget(mode):
    journal, tools = ConversationLog(), MemoryTools()
    budget = NativeBudget(max_model_calls=1 if mode == 'exhausted' else 4)
    run = session(ScriptedModel([delivery()]), tools, journal, budget=budget)
    run.run(control())
    last = journal.snapshot().records[-1]
    rev = journal.snapshot().revision

    if mode == 'cancelled':
        ctrl = control(cancel=CancellationToken())
        ctrl.cancellation.cancel()
    elif mode == 'deadline':
        ctrl = control(clock=lambda: 11)
    else:
        ctrl = control()

    with pytest.raises(ValueError):
        run.continue_after_observation(
            ctrl, observation_id='obs-1', text='Citation',
            after_sequence=last.sequence, after_sha256=last.sha256,
        )
    assert journal.snapshot().revision == rev


@pytest.mark.parametrize('mode', ['pending', 'blocked'])
def test_pending_effect_or_blocked_terminal_rejection(mode):
    journal, tools = ConversationLog(), MemoryTools()
    reply = RuntimeError('interrupted') if mode == 'pending' else tool('not-admitted')
    run = session(ScriptedModel([reply]), tools, journal)
    run.run(control())
    last = journal.snapshot().records[-1]
    rev = journal.snapshot().revision

    with pytest.raises(ValueError):
        run.continue_after_observation(
            control(), observation_id='obs-1', text='Citation',
            after_sequence=last.sequence, after_sha256=last.sha256,
        )
    assert journal.snapshot().revision == rev


@pytest.mark.parametrize('bad_event', [
    {'kind': 'host_observation', 'observation_id': 'o1', 'text': 'txt'},  # missing fields
    {'kind': 'host_observation', 'observation_id': '', 'text': 'txt', 'after_sequence': 4, 'after_sha256': 'a'*64},  # blank id
    {'kind': 'host_observation', 'observation_id': 'x'*129, 'text': 'txt', 'after_sequence': 4, 'after_sha256': 'a'*64},  # id > 128
    {'kind': 'host_observation', 'observation_id': 'o1', 'text': '', 'after_sequence': 4, 'after_sha256': 'a'*64},  # blank text
    {'kind': 'host_observation', 'observation_id': 'o1', 'text': 'x'*16385, 'after_sequence': 4, 'after_sha256': 'a'*64},  # text > 16384
    {'kind': 'host_observation', 'observation_id': 'o1', 'text': 'txt', 'after_sequence': True, 'after_sha256': 'a'*64},  # bool seq
    {'kind': 'host_observation', 'observation_id': 'o1', 'text': 'txt', 'after_sequence': 0, 'after_sha256': 'a'*64},  # non-positive seq
    {'kind': 'host_observation', 'observation_id': 'o1', 'text': 'txt', 'after_sequence': 4, 'after_sha256': 'A'*64},  # uppercase hash
    {'kind': 'host_observation', 'observation_id': 'o1', 'text': 'txt', 'after_sequence': 4, 'after_sha256': 'a'*63},  # short hash
    {'kind': 'host_observation', 'observation_id': 'o1', 'text': 'txt', 'after_sequence': 4, 'after_sha256': 'a'*64, 'extra': 1},  # extra field
])
def test_reducer_rejects_malformed_event(bad_event):
    journal, tools = ConversationLog(), MemoryTools()
    run = session(ScriptedModel([delivery()]), tools, journal)
    run.run(control())
    journal.append(bad_event)
    with pytest.raises(ValueError):
        run.run(control())


def test_reducer_rejects_second_observation_and_cross_boundary():
    journal, tools = ConversationLog(), MemoryTools()
    run = session(ScriptedModel([delivery(), delivery()]), tools, journal)
    run.run(control())
    last = journal.snapshot().records[-1]

    # Forged sequence mismatch
    journal.append({
        'kind': 'host_observation',
        'observation_id': 'obs-forged',
        'text': 'Forged citation',
        'after_sequence': 999,
        'after_sha256': last.sha256,
    })
    with pytest.raises(ValueError):
        run.run(control())

    # Forged second observation ceiling
    journal2, tools2 = ConversationLog(), MemoryTools()
    run2 = session(ScriptedModel([delivery(), delivery()]), tools2, journal2)
    run2.run(control())
    last2 = journal2.snapshot().records[-1]
    run2.continue_after_observation(
        control(), observation_id='obs-1', text='First citation',
        after_sequence=last2.sequence, after_sha256=last2.sha256,
    )
    run2.run(control())
    last3 = journal2.snapshot().records[-1]
    journal2.append({
        'kind': 'host_observation',
        'observation_id': 'obs-2',
        'text': 'Second citation',
        'after_sequence': last3.sequence,
        'after_sha256': last3.sha256,
    })
    with pytest.raises(ValueError):
        run2.run(control())


def test_preservation_through_native_context_project_and_compaction():
    from tare_tools_agent_runtime import ContextPolicy, NativeContext
    from test_native_context import Counter, Model

    journal, tools = ConversationLog(), MemoryTools()
    counter, summary = Counter(), Model()
    context = NativeContext(ContextPolicy(1000, 100, 50), counter=counter.count, summarizer=summary)
    model = ScriptedModel([delivery(), delivery(), delivery()])
    run = session(model, tools, journal, context=context)
    run.run(control())

    last = journal.snapshot().records[-1]
    assert run.continue_after_observation(
        control(),
        observation_id='obs-1',
        text='Preserve this citation across compaction',
        after_sequence=last.sequence,
        after_sha256=last.sha256,
    )
    # First turn after observation
    run.run(control())

    # Add user message to trigger further turns leading to compaction
    run.continue_with(control(), message_id='m1', text='Next turn instruction')
    result = run.run(control())
    assert result.status == 'proposed' and result.model_calls == 4

    # Check projection events in model request
    events = model.requests[-1].context.events()
    assert any(e.get('kind') == 'host_observation' and e.get('text') == 'Preserve this citation across compaction' for e in events)
    assert any(e.get('kind') == 'user_message' and e.get('text') == 'Next turn instruction' for e in events)
    assert any(e['kind'] == 'context_activated' for e in events)
    # Summarizer context also received the preserved observation
    assert any(e.get('kind') == 'host_observation' and e.get('text') == 'Preserve this citation across compaction' for e in summary.requests[0].context.events())


def test_duplicate_observation_does_not_hide_revoked_control():
    journal = ConversationLog()
    run = session(ScriptedModel([delivery()]), MemoryTools(), journal)
    run.run(control())
    last = journal.snapshot().records[-1]
    fields = dict(observation_id='one', text='Observed source', after_sequence=last.sequence, after_sha256=last.sha256)
    run.continue_after_observation(control(), **fields)
    with pytest.raises(ValueError):
        run.continue_after_observation(control(current=lambda: False), **fields)
