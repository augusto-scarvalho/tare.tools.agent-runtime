"""Managed-work fencing and verified progress contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


DEFAULT_MAX_WORK_BYTES = 65_536


@dataclass(frozen=True)
class RepositoryScope:
    repository: str
    grounding_refs: tuple[str, ...]
    target_paths: tuple[str, ...]
    target_symbols: tuple[str, ...]


@dataclass(frozen=True, init=False)
class WorkEnvelope:
    """One exact, execution-eligible ``backlog ground`` materialization."""

    canonical_bytes: bytes = field(repr=False)
    sha256: str
    byte_count: int
    work_id: str
    graph_sha256: str
    work_item_sha256: str
    execution_scope_sha256: str
    exit_criteria_count: int
    repository_scopes: tuple[RepositoryScope, ...]
    target_repositories: tuple[str, ...]
    grounding_refs: tuple[str, ...]
    target_paths: tuple[str, ...]
    target_symbols: tuple[str, ...]

    def __init__(
        self,
        canonical_bytes: bytes,
        *,
        max_bytes: int = DEFAULT_MAX_WORK_BYTES,
    ) -> None:
        if not isinstance(canonical_bytes, bytes):
            raise TypeError("work grounding must be exact bytes")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if not canonical_bytes or len(canonical_bytes) > max_bytes:
            raise ValueError("work grounding exceeds its byte budget")
        try:
            payload = json.loads(canonical_bytes.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("work grounding must be valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("work grounding root must be an object")
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if canonical != canonical_bytes:
            raise ValueError("work grounding must use canonical compact JSON bytes")
        if payload.get("schema") != "tare.tools/work-grounding/1":
            raise ValueError("unsupported work grounding schema")
        if payload.get("status") != "READY":
            raise ValueError("work grounding is not execution-eligible")
        if payload.get("profile") != "operational":
            raise ValueError("managed work grounding must use the operational profile")
        if payload.get("authority") != "NONE / READ_ONLY PROJECTION":
            raise ValueError("work grounding cannot grant Authority")

        byte_count = _non_negative_int(payload.get("byte_count"), "byte_count")
        if byte_count != len(canonical_bytes):
            raise ValueError("work grounding byte_count does not match exact bytes")
        work_id = _required_text(payload.get("work_id"), "work_id")
        graph_sha256 = _sha256(payload.get("graph_sha256"), "graph_sha256")
        work_item_sha256 = _sha256(
            payload.get("work_item_sha256"), "work_item_sha256"
        )
        execution_scope_sha256 = _sha256(
            payload.get("execution_scope_sha256"), "execution_scope_sha256"
        )
        work = payload.get("work")
        feasibility = payload.get("feasibility")
        if not isinstance(work, dict) or work.get("id") != work_id:
            raise ValueError("work grounding identity is inconsistent")
        if not isinstance(feasibility, dict) or feasibility.get("ready") is not True:
            raise ValueError("work grounding feasibility is not ready")
        exit_criteria = work.get("exit_criteria")
        if not isinstance(exit_criteria, list) or not exit_criteria or not all(
            isinstance(item, str) and item.strip() for item in exit_criteria
        ):
            raise ValueError("work grounding requires explicit exit criteria")
        target_repositories = _bounded_string_tuple(
            work.get("target_repositories", []), "target_repositories", 16
        )
        grounding_refs = _bounded_string_tuple(
            work.get("grounding_refs", []), "grounding_refs", 128
        )
        target_paths = _bounded_string_tuple(
            work.get("target_paths", []), "target_paths", 256
        )
        target_symbols = _bounded_string_tuple(
            work.get("target_symbols", []), "target_symbols", 256
        )
        repository_scopes = _repository_scopes(
            work.get("repository_scopes"),
            target_repositories=target_repositories,
            grounding_refs=grounding_refs,
            target_paths=target_paths,
            target_symbols=target_symbols,
        )
        scoped_repositories = tuple(scope.repository for scope in repository_scopes)
        if target_repositories != scoped_repositories:
            raise ValueError("work target_repositories must match repository_scopes")

        object.__setattr__(self, "canonical_bytes", canonical_bytes)
        object.__setattr__(self, "sha256", hashlib.sha256(canonical_bytes).hexdigest())
        object.__setattr__(self, "byte_count", byte_count)
        object.__setattr__(self, "work_id", work_id)
        object.__setattr__(self, "graph_sha256", graph_sha256)
        object.__setattr__(self, "work_item_sha256", work_item_sha256)
        object.__setattr__(self, "execution_scope_sha256", execution_scope_sha256)
        object.__setattr__(self, "exit_criteria_count", len(exit_criteria))
        object.__setattr__(self, "repository_scopes", repository_scopes)
        object.__setattr__(self, "target_repositories", target_repositories)
        object.__setattr__(
            self,
            "grounding_refs",
            tuple(sorted(grounding_refs, key=lambda item: (item.casefold(), item))),
        )
        object.__setattr__(self, "target_paths", target_paths)
        object.__setattr__(self, "target_symbols", target_symbols)


class WorkGuardDecision(str, Enum):
    CURRENT = "CURRENT"
    DRIFTED = "DRIFTED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class WorkGuardResult:
    decision: WorkGuardDecision
    reason_code: str
    observed_scope_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.decision, WorkGuardDecision):
            raise TypeError("decision must be WorkGuardDecision")
        _required_text(self.reason_code, "reason_code")
        if self.observed_scope_sha256 is not None:
            _sha256(self.observed_scope_sha256, "observed_scope_sha256")


@runtime_checkable
class WorkGuard(Protocol):
    def evaluate(self, work: WorkEnvelope) -> WorkGuardResult:
        """Revalidate the exact work scope without mutating canonical state."""


@dataclass(frozen=True)
class StrictWorkControl:
    """Opt-in fail-closed drift and stagnation policy."""

    max_stagnant_turns: int = 2

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_stagnant_turns, int)
            or isinstance(self.max_stagnant_turns, bool)
            or self.max_stagnant_turns < 1
        ):
            raise ValueError("max_stagnant_turns must be positive")


@dataclass(frozen=True)
class ProgressSignal:
    """Kernel-observed progress vector; never a provider self-attestation."""

    state_sha256: str
    evidence_sha256: str
    remaining_obligations: int
    verified_evidence_count: int

    def __post_init__(self) -> None:
        _sha256(self.state_sha256, "state_sha256")
        _sha256(self.evidence_sha256, "evidence_sha256")
        _non_negative_int(self.remaining_obligations, "remaining_obligations")
        _non_negative_int(self.verified_evidence_count, "verified_evidence_count")

    def advances(self, previous: "ProgressSignal") -> bool:
        obligations_not_worse = self.remaining_obligations <= previous.remaining_obligations
        evidence_not_worse = (
            self.verified_evidence_count >= previous.verified_evidence_count
        )
        strictly_better = (
            self.remaining_obligations < previous.remaining_obligations
            or self.verified_evidence_count > previous.verified_evidence_count
        )
        return obligations_not_worse and evidence_not_worse and strictly_better

    def regresses(self, previous: "ProgressSignal") -> bool:
        return (
            self.remaining_obligations > previous.remaining_obligations
            or self.verified_evidence_count < previous.verified_evidence_count
        )


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _sha256(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


def _non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _bounded_string_tuple(
    value: Any, field_name: str, maximum: int
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{field_name} must be a bounded string list")
    items = tuple(_required_text(item, field_name) for item in value)
    if any(len(item.encode("utf-8")) > 2_048 for item in items):
        raise ValueError(f"{field_name} contains an oversized item")
    if len(set(items)) != len(items):
        raise ValueError(f"{field_name} must not contain duplicates")
    return items


def _repository_scopes(
    value: Any,
    *,
    target_repositories: tuple[str, ...],
    grounding_refs: tuple[str, ...],
    target_paths: tuple[str, ...],
    target_symbols: tuple[str, ...],
) -> tuple[RepositoryScope, ...]:
    if value is None:
        if not target_repositories:
            if grounding_refs or target_paths or target_symbols:
                raise ValueError("flat work selection requires exactly one repository")
            return ()
        if len(target_repositories) != 1:
            raise ValueError("multiple repositories require repository_scopes")
        return (
            RepositoryScope(
                repository=target_repositories[0],
                grounding_refs=tuple(sorted(grounding_refs, key=lambda item: (item.casefold(), item))),
                target_paths=target_paths,
                target_symbols=target_symbols,
            ),
        )
    if not isinstance(value, list) or len(value) > 16:
        raise ValueError("repository_scopes must be a bounded list")
    scopes: list[RepositoryScope] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "repository", "grounding_refs", "target_paths", "target_symbols"
        }:
            raise ValueError("repository scope has an invalid shape")
        refs = _bounded_string_tuple(item["grounding_refs"], "grounding_refs", 128)
        scopes.append(
            RepositoryScope(
                repository=_required_text(item["repository"], "repository"),
                grounding_refs=tuple(sorted(refs, key=lambda entry: (entry.casefold(), entry))),
                target_paths=_bounded_string_tuple(item["target_paths"], "target_paths", 256),
                target_symbols=_bounded_string_tuple(item["target_symbols"], "target_symbols", 256),
            )
        )
    scopes.sort(key=lambda scope: (scope.repository.casefold(), scope.repository))
    if len({scope.repository for scope in scopes}) != len(scopes):
        raise ValueError("repository scopes must have unique repositories")
    return tuple(scopes)
