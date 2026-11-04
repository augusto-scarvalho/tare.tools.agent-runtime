from __future__ import annotations

import hashlib
import json

import pytest

from tare_tools_agent_runtime import WorkEnvelope


def work_bytes(**overrides: object) -> bytes:
    payload: dict[str, object] = {
        "authority": "NONE / READ_ONLY PROJECTION",
        "byte_count": 0,
        "schema": "tare.tools/work-grounding/1",
        "downstream": [],
        "execution_scope_sha256": "c" * 64,
        "feasibility": {
            "ready": True,
            "reasons": [],
            "unresolved_prerequisites": [],
        },
        "graph_revision": "projection-7",
        "graph_sha256": "a" * 64,
        "omitted_downstream": 0,
        "profile": "operational",
        "provenance": {"source_claim_ids": [], "source_refs": []},
        "reason_codes": ["READY_FOR_MANAGED_EXECUTION"],
        "status": "READY",
        "work": {"id": "WORK-1", "exit_criteria": ["Tests pass"]},
        "work_id": "WORK-1",
        "work_item_sha256": "b" * 64,
    }
    payload.update(overrides)
    for _ in range(8):
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if payload["byte_count"] == len(encoded):
            return encoded
        payload["byte_count"] = len(encoded)
    raise AssertionError("fixture byte_count did not converge")


def test_work_envelope_preserves_exact_bytes_and_scope_identity() -> None:
    raw = work_bytes()

    envelope = WorkEnvelope(raw)

    assert envelope.canonical_bytes is raw
    assert envelope.sha256 == hashlib.sha256(raw).hexdigest()
    assert envelope.work_id == "WORK-1"
    assert envelope.execution_scope_sha256 == "c" * 64
    assert envelope.exit_criteria_count == 1
    assert raw not in repr(envelope).encode("utf-8")


def test_work_envelope_exposes_explicit_grounding_selection() -> None:
    raw_payload = json.loads(work_bytes())
    raw_payload["work"].update(
        {
            "target_repositories": ["tare.tools.kernel"],
            "grounding_refs": ["SPEC-B", "ADR-A"],
            "target_paths": ["compute_plane/specgraph_grounding.py"],
            "target_symbols": ["check_specgraph_scope"],
        }
    )
    raw_payload["byte_count"] = 0
    for _ in range(8):
        raw = json.dumps(
            raw_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        if raw_payload["byte_count"] == len(raw):
            break
        raw_payload["byte_count"] = len(raw)
    envelope = WorkEnvelope(raw)

    assert envelope.target_repositories == ("tare.tools.kernel",)
    assert envelope.grounding_refs == ("ADR-A", "SPEC-B")
    assert envelope.target_paths == ("compute_plane/specgraph_grounding.py",)
    assert envelope.target_symbols == ("check_specgraph_scope",)
    assert envelope.repository_scopes[0].repository == "tare.tools.kernel"
    assert envelope.repository_scopes[0].grounding_refs == ("ADR-A", "SPEC-B")


def test_work_envelope_admits_explicit_multirepository_scopes() -> None:
    raw_payload = json.loads(work_bytes())
    raw_payload["work"].update(
        {
            "target_repositories": ["repo-a", "repo-b"],
            "repository_scopes": [
                {
                    "repository": "repo-b",
                    "grounding_refs": ["B"],
                    "target_paths": ["src/b.py"],
                    "target_symbols": [],
                },
                {
                    "repository": "repo-a",
                    "grounding_refs": ["A"],
                    "target_paths": ["src/a.py"],
                    "target_symbols": [],
                },
            ],
        }
    )
    raw_payload["byte_count"] = 0
    for _ in range(8):
        raw = json.dumps(
            raw_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        if raw_payload["byte_count"] == len(raw):
            break
        raw_payload["byte_count"] = len(raw)

    envelope = WorkEnvelope(raw)

    assert [scope.repository for scope in envelope.repository_scopes] == [
        "repo-a",
        "repo-b",
    ]


@pytest.mark.parametrize(
    "raw",
    [
        work_bytes(status="DRIFT"),
        work_bytes(profile="planning"),
        work_bytes(authority="SELF_GRANTED"),
        work_bytes(schema="tare.tools/work-grounding/0"),
        work_bytes(feasibility={"ready": False}),
        work_bytes(work={"id": "WORK-1", "exit_criteria": []}),
    ],
)
def test_non_executable_work_grounding_fails_closed(raw: bytes) -> None:
    with pytest.raises(ValueError):
        WorkEnvelope(raw)
