"""Immutable SpecGraph grounding admitted at the managed-runtime ingress."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


DEFAULT_MAX_GROUNDING_BYTES = 65_536
_USABLE_STATUSES = {"READY", "READY_TRUNCATED", "NOT_MAPPED"}


@dataclass(frozen=True, init=False)
class GroundingEnvelope:
    """One exact, bounded ``specgraph ground`` JSON materialization."""

    canonical_bytes: bytes = field(repr=False)
    sha256: str
    repository: str
    manifest_sha256: str
    grounding_scope_sha256: str
    requested_ids: tuple[str, ...]
    status: str
    byte_count: int

    def __init__(
        self,
        canonical_bytes: bytes,
        *,
        max_bytes: int = DEFAULT_MAX_GROUNDING_BYTES,
    ) -> None:
        if not isinstance(canonical_bytes, bytes):
            raise TypeError("grounding must be exact bytes")
        if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        if not canonical_bytes or len(canonical_bytes) > max_bytes:
            raise ValueError("grounding exceeds its byte budget")

        try:
            text = canonical_bytes.decode("utf-8", errors="strict")
            payload = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("grounding must be valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("grounding root must be an object")
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if canonical != canonical_bytes:
            raise ValueError("grounding must use canonical compact JSON bytes")
        if payload.get("schema") != "tare.tools/repository-grounding/1":
            raise ValueError("unsupported grounding schema")

        status = _required_text(payload.get("status"), "status")
        if status not in _USABLE_STATUSES:
            raise ValueError("grounding status is not usable")
        repository = _required_text(payload.get("repository"), "repository")
        manifest_sha256 = _sha256(payload.get("manifest_sha256"), "manifest_sha256")
        grounding_scope_sha256 = _sha256(
            payload.get("grounding_scope_sha256"), "grounding_scope_sha256"
        )
        requested_ids = _bounded_string_tuple(
            payload.get("requested_ids"), "requested_ids", 128
        )
        if requested_ids != tuple(
            sorted(requested_ids, key=lambda item: (item.casefold(), item))
        ) or len(set(requested_ids)) != len(requested_ids):
            raise ValueError("grounding requested_ids must be unique and sorted")
        declared_bytes = _non_negative_int(payload.get("byte_count"), "byte_count")
        if declared_bytes != len(canonical_bytes):
            raise ValueError("grounding byte_count does not match exact bytes")

        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("grounding entries must be a list")
        matched = _non_negative_int(payload.get("matched_count"), "matched_count")
        returned = _non_negative_int(payload.get("returned_count"), "returned_count")
        omitted = _non_negative_int(payload.get("omitted_count"), "omitted_count")
        if returned != len(entries) or matched != returned + omitted:
            raise ValueError("grounding entry counts are inconsistent")
        if status == "READY" and omitted != 0:
            raise ValueError("READY grounding cannot omit entries")
        if status == "READY_TRUNCATED" and omitted == 0:
            raise ValueError("READY_TRUNCATED grounding must omit entries")
        if status == "NOT_MAPPED" and (matched != 0 or entries):
            raise ValueError("NOT_MAPPED grounding cannot contain entries")

        object.__setattr__(self, "canonical_bytes", canonical_bytes)
        object.__setattr__(self, "sha256", hashlib.sha256(canonical_bytes).hexdigest())
        object.__setattr__(self, "repository", repository)
        object.__setattr__(self, "manifest_sha256", manifest_sha256)
        object.__setattr__(self, "grounding_scope_sha256", grounding_scope_sha256)
        object.__setattr__(self, "requested_ids", requested_ids)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "byte_count", declared_bytes)


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"grounding {field_name} must be non-empty")
    return value


def _sha256(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ValueError(f"grounding {field_name} must be a lowercase SHA-256 digest")
    return text


def _non_negative_int(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"grounding {field_name} must be a non-negative integer")
    return value


def _bounded_string_tuple(
    value: Any, field_name: str, maximum: int
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"grounding {field_name} must be a bounded string list")
    items = tuple(_required_text(item, field_name) for item in value)
    if any(len(item.encode("utf-8")) > 256 for item in items):
        raise ValueError(f"grounding {field_name} contains an oversized item")
    return items
