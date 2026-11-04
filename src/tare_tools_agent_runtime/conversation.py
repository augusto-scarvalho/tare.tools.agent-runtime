"""Byte-immutable in-memory tactical conversation primitives.

This module implements only the P1 laboratory approved by
CASE-2026-08-24-AGENT-RUNTIME-PROMPT-CACHING. It performs no filesystem or
network I/O and contains no provider-cache behavior.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Protocol


CANONICALIZER_VERSION = "tare-json-v1"


class CancellationSignal(Protocol):
    def is_cancelled(self) -> bool:
        """Return whether the pending append must fail closed."""


class ConversationRejectReason(str, Enum):
    INVALID_EVENT = "INVALID_EVENT"
    EVENT_TOO_LARGE = "EVENT_TOO_LARGE"
    JSON_TOO_DEEP = "JSON_TOO_DEEP"
    LOG_FULL = "LOG_FULL"
    CANCELLED = "CANCELLED"
    WRITER_BUSY = "WRITER_BUSY"
    INVALID_PROJECTION = "INVALID_PROJECTION"
    PROJECTION_TOO_LARGE = "PROJECTION_TOO_LARGE"
    STALE_PROJECTION = "STALE_PROJECTION"


class ConversationRejected(RuntimeError):
    """Structured fail-closed rejection without partial state publication."""

    def __init__(self, reason: ConversationRejectReason, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ConversationLimits:
    max_event_bytes: int = 64 * 1024
    max_json_depth: int = 32
    max_events: int = 4096
    max_projection_bytes: int = 2 * 1024 * 1024
    max_projection_cache_entries: int = 32
    writer_timeout_seconds: float = 1.0

    def __post_init__(self) -> None:
        integer_limits = {
            "max_event_bytes": self.max_event_bytes,
            "max_json_depth": self.max_json_depth,
            "max_events": self.max_events,
            "max_projection_bytes": self.max_projection_bytes,
            "max_projection_cache_entries": self.max_projection_cache_entries,
        }
        for name, value in integer_limits.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be positive")
        if (
            not isinstance(self.writer_timeout_seconds, (int, float))
            or isinstance(self.writer_timeout_seconds, bool)
            or not math.isfinite(self.writer_timeout_seconds)
            or self.writer_timeout_seconds <= 0
        ):
            raise ValueError("writer_timeout_seconds must be positive")


@dataclass(frozen=True)
class ConversationRecord:
    sequence: int
    canonical_bytes: bytes
    sha256: str

    def event(self) -> dict[str, Any]:
        """Return a fresh detached object; callers never receive stored aliases."""

        value = json.loads(self.canonical_bytes)
        if not isinstance(value, dict):  # Defensive assertion for corrupted internals.
            raise RuntimeError("stored conversation event is not a JSON object")
        return value


@dataclass(frozen=True)
class ConversationSnapshot:
    revision: int
    next_sequence: int
    records: tuple[ConversationRecord, ...]


@dataclass(frozen=True)
class ProjectionSpec:
    generation: int
    projector_version: str
    included_sequences: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.generation, int)
            or isinstance(self.generation, bool)
            or self.generation < 1
        ):
            raise ValueError("generation must be positive")
        if not isinstance(self.projector_version, str) or not self.projector_version.strip():
            raise ValueError("projector_version must be non-empty")
        try:
            version_bytes = self.projector_version.encode("utf-8", errors="strict")
        except UnicodeError as exc:
            raise ValueError("projector_version must be valid UTF-8") from exc
        if len(version_bytes) > 128:
            raise ValueError("projector_version cannot exceed 128 UTF-8 bytes")
        if self.included_sequences is not None:
            if not isinstance(self.included_sequences, tuple):
                raise ValueError("included sequences must be a tuple")
            if any(
                not isinstance(sequence, int)
                or isinstance(sequence, bool)
                or sequence < 1
                for sequence in self.included_sequences
            ):
                raise ValueError("included sequences must be positive")
            if any(
                current <= previous
                for previous, current in zip(
                    self.included_sequences,
                    self.included_sequences[1:],
                    strict=False,
                )
            ):
                raise ValueError("included sequences must be unique and ascending")


@dataclass(frozen=True)
class ContextProjection:
    source_revision: int
    source_snapshot_sha256: str
    projected_bytes: bytes
    projection_generation: int
    projector_version: str
    included_sequences: tuple[int, ...]
    source_event_hashes: tuple[str, ...]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.projected_bytes).hexdigest()

    def events(self) -> list[dict[str, Any]]:
        """Return fresh detached projection values."""

        value = json.loads(self.projected_bytes)
        if not isinstance(value, list):  # Defensive assertion for corrupted internals.
            raise RuntimeError("stored context projection is not a JSON array")
        return value


@dataclass(frozen=True)
class _LogState:
    revision: int = 0
    next_sequence: int = 1
    records: tuple[ConversationRecord, ...] = ()


class ConversationLog:
    """Bounded append-only log with atomic snapshots and disposable projections."""

    def __init__(self, limits: ConversationLimits | None = None) -> None:
        self._limits = limits or ConversationLimits()
        self._writer_lock = Lock()
        self._state = _LogState()
        self._projection_cache: dict[tuple[int, ProjectionSpec], ContextProjection] = {}

    @property
    def limits(self) -> ConversationLimits:
        return self._limits

    @property
    def revision(self) -> int:
        return self.snapshot().revision

    def append(
        self,
        event: Mapping[str, Any],
        *,
        cancellation: CancellationSignal | None = None,
    ) -> ConversationRecord:
        """Materialize once, then publish record/sequence/revision atomically."""

        self._reject_if_cancelled(cancellation)
        canonical_bytes = self._materialize(event)
        event_hash = hashlib.sha256(canonical_bytes).hexdigest()
        self._reject_if_cancelled(cancellation)

        self._acquire_writer()
        try:
            self._reject_if_cancelled(cancellation)
            current = self._state
            if len(current.records) >= self._limits.max_events:
                raise ConversationRejected(
                    ConversationRejectReason.LOG_FULL,
                    "conversation event limit reached",
                )
            record = ConversationRecord(
                sequence=current.next_sequence,
                canonical_bytes=canonical_bytes,
                sha256=event_hash,
            )
            committed = _LogState(
                revision=current.revision + 1,
                next_sequence=current.next_sequence + 1,
                records=current.records + (record,),
            )
            self._state = committed
            self._projection_cache = {}
            return record
        finally:
            self._writer_lock.release()

    def append_json(
        self,
        raw_event: str | bytes,
        *,
        cancellation: CancellationSignal | None = None,
    ) -> ConversationRecord:
        """Strict raw-JSON ingress with duplicate-key and UTF-8 rejection."""

        self._reject_if_cancelled(cancellation)
        if not isinstance(raw_event, (str, bytes)):
            raise ConversationRejected(
                ConversationRejectReason.INVALID_EVENT,
                "raw JSON event must be str or bytes",
            )
        try:
            text = (
                raw_event.decode("utf-8", errors="strict")
                if isinstance(raw_event, bytes)
                else raw_event.encode("utf-8", errors="strict").decode("utf-8")
            )
            value = json.loads(
                text,
                object_pairs_hook=_unique_object,
                parse_constant=_reject_json_constant,
            )
        except (UnicodeError, json.JSONDecodeError, RecursionError, TypeError) as exc:
            raise ConversationRejected(
                ConversationRejectReason.INVALID_EVENT,
                f"invalid JSON event: {exc}",
            ) from exc
        if not isinstance(value, dict):
            raise ConversationRejected(
                ConversationRejectReason.INVALID_EVENT,
                "conversation event must be a JSON object",
            )
        return self.append(value, cancellation=cancellation)

    def snapshot(self) -> ConversationSnapshot:
        self._acquire_writer()
        try:
            state = self._state
            return ConversationSnapshot(
                revision=state.revision,
                next_sequence=state.next_sequence,
                records=state.records,
            )
        finally:
            self._writer_lock.release()

    def project(self, spec: ProjectionSpec) -> ContextProjection:
        """Build or reuse one deterministic projection of the current snapshot."""

        if not isinstance(spec, ProjectionSpec):
            raise TypeError("spec must be ProjectionSpec")
        self._acquire_writer()
        try:
            source_state = self._state
            cache_key = (source_state.revision, spec)
            cached = self._projection_cache.get(cache_key)
            if cached is not None:
                return cached
        finally:
            self._writer_lock.release()

        projection = self._build_projection(source_state, spec)

        self._acquire_writer()
        try:
            if self._state is not source_state:
                raise ConversationRejected(
                    ConversationRejectReason.STALE_PROJECTION,
                    "conversation advanced while projection was being built",
                )
            cached = self._projection_cache.get(cache_key)
            if cached is not None:
                return cached
            if len(self._projection_cache) >= self._limits.max_projection_cache_entries:
                oldest_key = next(iter(self._projection_cache))
                del self._projection_cache[oldest_key]
            self._projection_cache[cache_key] = projection
            return projection
        finally:
            self._writer_lock.release()

    def admit_current(self, projection: ContextProjection) -> ContextProjection:
        """Linearize admission and reject a ticket already stale at that point."""

        if not isinstance(projection, ContextProjection):
            raise TypeError("projection must be ContextProjection")
        self._acquire_writer()
        try:
            if projection.source_revision != self._state.revision:
                raise ConversationRejected(
                    ConversationRejectReason.STALE_PROJECTION,
                    "projection source revision is no longer current",
                )
            if projection.source_snapshot_sha256 != _snapshot_sha256(self._state):
                raise ConversationRejected(
                    ConversationRejectReason.STALE_PROJECTION,
                    "projection belongs to a different conversation snapshot",
                )
            return projection
        finally:
            self._writer_lock.release()

    def clear_projection_cache(self) -> None:
        """Physically replace the disposable cache structure."""

        self._acquire_writer()
        try:
            self._projection_cache = {}
        finally:
            self._writer_lock.release()

    @property
    def projection_cache_entries(self) -> int:
        self._acquire_writer()
        try:
            return len(self._projection_cache)
        finally:
            self._writer_lock.release()

    def _build_projection(
        self,
        state: _LogState,
        spec: ProjectionSpec,
    ) -> ContextProjection:
        if spec.included_sequences is None:
            records = state.records
        else:
            if len(spec.included_sequences) > self._limits.max_events:
                raise ConversationRejected(
                    ConversationRejectReason.INVALID_PROJECTION,
                    "projection sequence count exceeds max_events",
                )
            missing = [
                sequence
                for sequence in spec.included_sequences
                if sequence >= state.next_sequence
            ]
            if missing:
                raise ConversationRejected(
                    ConversationRejectReason.INVALID_PROJECTION,
                    f"projection references missing sequences: {missing}",
                )
            records = tuple(state.records[sequence - 1] for sequence in spec.included_sequences)

        projected_bytes = b"[" + b",".join(record.canonical_bytes for record in records) + b"]"
        if len(projected_bytes) > self._limits.max_projection_bytes:
            raise ConversationRejected(
                ConversationRejectReason.PROJECTION_TOO_LARGE,
                "projected canonical bytes exceed max_projection_bytes",
            )
        return ContextProjection(
            source_revision=state.revision,
            source_snapshot_sha256=_snapshot_sha256(state),
            projected_bytes=projected_bytes,
            projection_generation=spec.generation,
            projector_version=spec.projector_version,
            included_sequences=tuple(record.sequence for record in records),
            source_event_hashes=tuple(record.sha256 for record in records),
        )

    def _materialize(self, event: Mapping[str, Any]) -> bytes:
        if not isinstance(event, Mapping):
            raise ConversationRejected(
                ConversationRejectReason.INVALID_EVENT,
                "conversation event must be a mapping",
            )
        try:
            normalized = _normalize_json(event, depth=1, max_depth=self._limits.max_json_depth)
            text = json.dumps(
                normalized,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            canonical_bytes = text.encode("utf-8", errors="strict")
        except ConversationRejected:
            raise
        except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
            raise ConversationRejected(
                ConversationRejectReason.INVALID_EVENT,
                f"event cannot be canonicalized: {exc}",
            ) from exc
        if len(canonical_bytes) > self._limits.max_event_bytes:
            raise ConversationRejected(
                ConversationRejectReason.EVENT_TOO_LARGE,
                "canonical event bytes exceed max_event_bytes",
            )
        return canonical_bytes

    def _acquire_writer(self) -> None:
        if not self._writer_lock.acquire(timeout=self._limits.writer_timeout_seconds):
            raise ConversationRejected(
                ConversationRejectReason.WRITER_BUSY,
                "conversation writer did not become available within the bound",
            )

    @staticmethod
    def _reject_if_cancelled(cancellation: CancellationSignal | None) -> None:
        if cancellation is not None and cancellation.is_cancelled():
            raise ConversationRejected(
                ConversationRejectReason.CANCELLED,
                "conversation append was cancelled",
            )


def _normalize_json(value: Any, *, depth: int, max_depth: int) -> Any:
    if depth > max_depth:
        raise ConversationRejected(
            ConversationRejectReason.JSON_TOO_DEEP,
            "conversation event exceeds max_json_depth",
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConversationRejected(
                ConversationRejectReason.INVALID_EVENT,
                "non-finite JSON numbers are forbidden",
            )
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ConversationRejected(
                    ConversationRejectReason.INVALID_EVENT,
                    "JSON object keys must be strings",
                )
            normalized[key] = _normalize_json(
                nested,
                depth=depth + 1,
                max_depth=max_depth,
            )
        return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [
            _normalize_json(nested, depth=depth + 1, max_depth=max_depth)
            for nested in value
        ]
    raise ConversationRejected(
        ConversationRejectReason.INVALID_EVENT,
        f"unsupported JSON value type: {type(value).__name__}",
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, nested in pairs:
        if key in value:
            raise ConversationRejected(
                ConversationRejectReason.INVALID_EVENT,
                f"duplicate JSON object key: {key}",
            )
        value[key] = nested
    return value


def _reject_json_constant(value: str) -> None:
    raise ConversationRejected(
        ConversationRejectReason.INVALID_EVENT,
        f"non-finite JSON number is forbidden: {value}",
    )


def _snapshot_sha256(state: _LogState) -> str:
    digest = hashlib.sha256(b"tare-conversation-snapshot-v1\x00")
    digest.update(state.revision.to_bytes(8, "big"))
    digest.update(state.next_sequence.to_bytes(8, "big"))
    for record in state.records:
        digest.update(record.sequence.to_bytes(8, "big"))
        digest.update(bytes.fromhex(record.sha256))
    return digest.hexdigest()
