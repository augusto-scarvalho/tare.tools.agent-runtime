from __future__ import annotations

import hashlib
import threading

import pytest

import tare_tools_agent_runtime as agent_runtime
from tare_tools_agent_runtime import (
    CANONICALIZER_VERSION,
    CancellationToken,
    ConversationLimits,
    ConversationLog,
    ConversationRejectReason,
    ConversationRejected,
    ProjectionSpec,
)


def event(text: str, **extra):
    return {"kind": "message", "text": text, **extra}


def test_pc01_committed_bytes_and_hashes_never_change() -> None:
    log = ConversationLog()
    first = log.append(event("one"))
    before = (first.canonical_bytes, first.sha256)

    log.append(event("two"))
    log.project(ProjectionSpec(1, "projector-v1"))
    log.clear_projection_cache()

    stored = log.snapshot().records[0]
    assert (stored.canonical_bytes, stored.sha256) == before
    assert stored.sha256 == hashlib.sha256(stored.canonical_bytes).hexdigest()


def test_pc02_concurrent_writers_publish_contiguous_sequences() -> None:
    log = ConversationLog()
    writers = 24
    start = threading.Barrier(writers)
    records = []
    errors = []

    def append_one(number: int) -> None:
        try:
            start.wait()
            records.append(log.append(event(str(number))))
        except BaseException as exc:  # Test captures worker failures for the main thread.
            errors.append(exc)

    threads = [threading.Thread(target=append_one, args=(number,)) for number in range(writers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    assert errors == []
    assert all(not thread.is_alive() for thread in threads)
    snapshot = log.snapshot()
    assert snapshot.revision == writers
    assert snapshot.next_sequence == writers + 1
    assert [record.sequence for record in snapshot.records] == list(range(1, writers + 1))
    assert sorted(record.sequence for record in records) == list(range(1, writers + 1))


def test_pc02_rejections_and_cancellation_consume_no_sequence() -> None:
    log = ConversationLog(ConversationLimits(max_events=1))
    cancelled = CancellationToken()
    cancelled.cancel()

    with pytest.raises(ConversationRejected) as cancel_error:
        log.append(event("cancelled"), cancellation=cancelled)
    assert cancel_error.value.reason is ConversationRejectReason.CANCELLED
    assert log.snapshot().next_sequence == 1

    accepted = log.append(event("accepted"))
    assert accepted.sequence == 1
    with pytest.raises(ConversationRejected) as full_error:
        log.append(event("too many"))
    assert full_error.value.reason is ConversationRejectReason.LOG_FULL
    assert log.snapshot().next_sequence == 2
    assert log.snapshot().revision == 1


def test_pc02_writer_wait_is_bounded_and_lock_is_released() -> None:
    log = ConversationLog(ConversationLimits(writer_timeout_seconds=0.01))
    assert log._writer_lock.acquire(timeout=0.01)  # Deliberate white-box liveness falsifier.
    try:
        with pytest.raises(ConversationRejected) as busy_error:
            log.append(event("blocked"))
        assert busy_error.value.reason is ConversationRejectReason.WRITER_BUSY
    finally:
        log._writer_lock.release()

    assert log.append(event("after-release")).sequence == 1


def test_pc03_ingress_and_egress_aliasing_cannot_mutate_history() -> None:
    log = ConversationLog()
    original = event("hello", nested={"items": [1, 2]})
    record = log.append(original)
    original["nested"]["items"].append(3)
    returned = record.event()
    returned["nested"]["items"].append(4)
    projection = log.project(ProjectionSpec(1, "projector-v1"))
    projected_events = projection.events()
    projected_events[0]["nested"]["items"].append(5)

    stored = log.snapshot().records[0]
    assert stored.event()["nested"]["items"] == [1, 2]
    assert stored.sha256 == hashlib.sha256(stored.canonical_bytes).hexdigest()
    assert projection.events()[0]["nested"]["items"] == [1, 2]


def test_pc04_pc06_projection_is_deterministic_and_cache_is_disposable() -> None:
    log = ConversationLog()
    first = log.append(event("one"))
    second = log.append(event("two"))
    spec = ProjectionSpec(1, "projector-v1")

    warm = log.project(spec)
    assert log.projection_cache_entries == 1
    assert warm.projected_bytes == (
        b"[" + first.canonical_bytes + b"," + second.canonical_bytes + b"]"
    )

    log.clear_projection_cache()
    assert log.projection_cache_entries == 0
    cold = log.project(spec)
    assert cold is not warm
    assert cold == warm


def test_pc05_selection_never_rewrites_source_events() -> None:
    log = ConversationLog()
    first = log.append(event("hidden"))
    second = log.append(event("visible"))
    source_hashes = tuple(record.sha256 for record in log.snapshot().records)

    projection = log.project(ProjectionSpec(1, "window-v1", (2,)))

    assert projection.included_sequences == (2,)
    assert projection.projected_bytes == b"[" + second.canonical_bytes + b"]"
    assert tuple(record.sha256 for record in log.snapshot().records) == source_hashes
    assert log.snapshot().records[0] == first


def test_pc06_projection_cache_is_bounded() -> None:
    log = ConversationLog(ConversationLimits(max_projection_cache_entries=1))
    log.append(event("one"))
    log.project(ProjectionSpec(1, "projector-v1"))
    log.project(ProjectionSpec(2, "projector-v1"))
    assert log.projection_cache_entries == 1


def test_pc10_stale_projection_fails_closed_at_admission() -> None:
    log = ConversationLog()
    log.append(event("one"))
    projection = log.project(ProjectionSpec(1, "projector-v1"))
    log.append(event("two"))

    with pytest.raises(ConversationRejected) as stale_error:
        log.admit_current(projection)
    assert stale_error.value.reason is ConversationRejectReason.STALE_PROJECTION


def test_pc10_equal_revision_from_another_log_fails_closed() -> None:
    first_log = ConversationLog()
    second_log = ConversationLog()
    first_log.append(event("first"))
    second_log.append(event("second"))
    foreign = first_log.project(ProjectionSpec(1, "projector-v1"))

    with pytest.raises(ConversationRejected) as stale_error:
        second_log.admit_current(foreign)
    assert stale_error.value.reason is ConversationRejectReason.STALE_PROJECTION


def test_pc10_pc17_append_during_projection_cannot_publish_mixed_state(monkeypatch) -> None:
    log = ConversationLog()
    log.append(event("one"))
    started = threading.Event()
    resume = threading.Event()
    original_builder = log._build_projection
    result = []

    def blocked_builder(state, spec):
        started.set()
        assert resume.wait(timeout=2)
        return original_builder(state, spec)

    monkeypatch.setattr(log, "_build_projection", blocked_builder)

    def project() -> None:
        try:
            result.append(log.project(ProjectionSpec(1, "projector-v1")))
        except BaseException as exc:  # Test captures worker result for assertion.
            result.append(exc)

    thread = threading.Thread(target=project)
    thread.start()
    assert started.wait(timeout=2)
    log.append(event("two"))
    resume.set()
    thread.join(timeout=2)

    assert len(result) == 1
    assert isinstance(result[0], ConversationRejected)
    assert result[0].reason is ConversationRejectReason.STALE_PROJECTION
    assert log.projection_cache_entries == 0


def test_pc17_pc19_every_visible_snapshot_is_complete() -> None:
    log = ConversationLog()
    writers = 8
    start = threading.Barrier(writers + 1)

    def append_one(number: int) -> None:
        start.wait()
        log.append(event(str(number)))

    threads = [threading.Thread(target=append_one, args=(number,)) for number in range(writers)]
    for thread in threads:
        thread.start()
    start.wait()

    observed = []
    while any(thread.is_alive() for thread in threads):
        observed.append(log.snapshot())
    for thread in threads:
        thread.join(timeout=2)
    observed.append(log.snapshot())

    for snapshot in observed:
        assert len(snapshot.records) == snapshot.revision
        assert snapshot.next_sequence == snapshot.revision + 1
        assert [record.sequence for record in snapshot.records] == list(
            range(1, snapshot.revision + 1)
        )


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b'{"text":"\xff"}', ConversationRejectReason.INVALID_EVENT),
        ('{"key":1,"key":2}', ConversationRejectReason.INVALID_EVENT),
        ('{"value":NaN}', ConversationRejectReason.INVALID_EVENT),
    ],
)
def test_pc18_strict_json_ingress_rejects_ambiguous_identity(raw, reason) -> None:
    log = ConversationLog()
    with pytest.raises(ConversationRejected) as rejected:
        log.append_json(raw)
    assert rejected.value.reason is reason
    assert log.revision == 0


def test_pc18_canonical_golden_bytes_pin_order_and_unicode_policy() -> None:
    log = ConversationLog()
    record = log.append({"b": 1, "a": "é"})
    assert CANONICALIZER_VERSION == "tare-json-v1"
    assert record.canonical_bytes == b'{"a":"\xc3\xa9","b":1}'


def test_pc18_depth_and_materialized_byte_limits_fail_closed() -> None:
    depth_limited = ConversationLog(ConversationLimits(max_json_depth=2))
    with pytest.raises(ConversationRejected) as depth_error:
        depth_limited.append({"outer": {"inner": "too deep"}})
    assert depth_error.value.reason is ConversationRejectReason.JSON_TOO_DEEP
    assert depth_limited.revision == 0

    byte_limited = ConversationLog(ConversationLimits(max_event_bytes=16))
    with pytest.raises(ConversationRejected) as size_error:
        byte_limited.append({"x": "\n" * 8})
    assert size_error.value.reason is ConversationRejectReason.EVENT_TOO_LARGE
    assert byte_limited.revision == 0


def test_projection_rejects_missing_sequences_and_oversize_output() -> None:
    log = ConversationLog(ConversationLimits(max_projection_bytes=2))
    log.append({})
    with pytest.raises(ConversationRejected) as missing_error:
        log.project(ProjectionSpec(1, "projector-v1", (2,)))
    assert missing_error.value.reason is ConversationRejectReason.INVALID_PROJECTION

    with pytest.raises(ConversationRejected) as size_error:
        log.project(ProjectionSpec(1, "projector-v1"))
    assert size_error.value.reason is ConversationRejectReason.PROJECTION_TOO_LARGE


def test_p1_exposes_no_provider_cache_plan_or_production_entrypoint() -> None:
    assert not hasattr(agent_runtime, "PromptCachePlan")
    assert not hasattr(agent_runtime, "ProviderCache")


def test_projection_spec_is_bounded_and_deterministic() -> None:
    with pytest.raises(ValueError, match="generation"):
        ProjectionSpec(0, "v1")
    with pytest.raises(ValueError, match="projector_version"):
        ProjectionSpec(1, " ")
    with pytest.raises(ValueError, match="unique and ascending"):
        ProjectionSpec(1, "v1", (2, 1))
    with pytest.raises(ValueError, match="unique and ascending"):
        ProjectionSpec(1, "v1", (1, 1))
    with pytest.raises(ValueError, match="must be a tuple"):
        ProjectionSpec(1, "v1", [1])
    with pytest.raises(ValueError, match="128"):
        ProjectionSpec(1, "v" * 129)


def test_limits_and_raw_ingress_types_fail_early() -> None:
    with pytest.raises(ValueError, match="max_event_bytes"):
        ConversationLimits(max_event_bytes=True)
    with pytest.raises(ValueError, match="writer_timeout_seconds"):
        ConversationLimits(writer_timeout_seconds=float("nan"))
    with pytest.raises(ConversationRejected) as raw_type_error:
        ConversationLog().append_json(42)
    assert raw_type_error.value.reason is ConversationRejectReason.INVALID_EVENT
