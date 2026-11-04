from __future__ import annotations

import hashlib
import json

import pytest

from tare_tools_agent_runtime import GroundingEnvelope


def grounding_bytes(**overrides: object) -> bytes:
    payload: dict[str, object] = {
        "byte_count": 0,
        "entries": [],
        "grounding_scope_sha256": hashlib.sha256(b"scope").hexdigest(),
        "manifest_sha256": hashlib.sha256(b"manifest").hexdigest(),
        "manifest_version": "2.1.0",
        "matched_count": 0,
        "omitted_count": 0,
        "reason_code": "NO_TARGET_MAPPING",
        "repository": "tare.tools.kernel",
        "requested_ids": [],
        "returned_count": 0,
        "schema": "tare.tools/repository-grounding/1",
        "status": "NOT_MAPPED",
    }
    payload.update(overrides)
    for _ in range(8):
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if payload["byte_count"] == len(encoded):
            return encoded
        payload["byte_count"] = len(encoded)
    raise AssertionError("fixture byte count did not converge")


def wrong_byte_count() -> bytes:
    payload = json.loads(grounding_bytes())
    payload["byte_count"] = 1
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def test_exact_specgraph_grounding_bytes_are_immutable_and_identified() -> None:
    raw = grounding_bytes()

    envelope = GroundingEnvelope(raw)

    assert envelope.canonical_bytes is raw
    assert envelope.byte_count == len(raw)
    assert envelope.sha256 == hashlib.sha256(raw).hexdigest()
    assert envelope.repository == "tare.tools.kernel"
    assert envelope.grounding_scope_sha256 == hashlib.sha256(b"scope").hexdigest()
    assert envelope.requested_ids == ()
    assert raw not in repr(envelope).encode("utf-8")


@pytest.mark.parametrize(
    "raw",
    [
        b"not-json",
        b'{"status": "NOT_MAPPED"}',
        grounding_bytes(status="SOURCE_INVALID"),
        wrong_byte_count(),
        grounding_bytes(matched_count=1),
    ],
)
def test_invalid_or_noncanonical_grounding_fails_closed(raw: bytes) -> None:
    with pytest.raises(ValueError):
        GroundingEnvelope(raw)


def test_grounding_budget_is_checked_before_runtime_dispatch() -> None:
    raw = grounding_bytes()

    with pytest.raises(ValueError, match="byte budget"):
        GroundingEnvelope(raw, max_bytes=len(raw) - 1)
