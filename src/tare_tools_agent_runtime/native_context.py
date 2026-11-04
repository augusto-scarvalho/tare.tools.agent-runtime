"""Request admission and transactional context replacement for the native loop.

All I/O is injected. The journal retains original events and auxiliary responses;
activation is one atomic append after the candidate request has been counted.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, replace
from typing import Callable, TYPE_CHECKING

from .conversation import ConversationLimits, ConversationLog, ProjectionSpec

if TYPE_CHECKING:
    from .native import ModelRequest, ModelPort, NativeJournal
    from .contracts import ExecutionControl


class ContextBlocked(RuntimeError):
    pass


@dataclass(frozen=True)
class ContextPolicy:
    window: int
    output_reserve: int
    margin: int
    trigger: float = 0.8
    target: float = 0.5
    keep_recent_turns: int = 1
    max_compactions: int | None = 1
    allow_estimated_counts: bool = False

    def __post_init__(self):
        if type(self.allow_estimated_counts) is not bool:
            raise ValueError('allow_estimated_counts must be boolean')
        for name in ("window", "output_reserve", "margin", "keep_recent_turns", "max_compactions"):
            value = getattr(self, name)
            if name == "max_compactions" and value is None:
                continue
            minimum = 0 if name in {"margin", "max_compactions"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"invalid {name}")
        for value in (self.trigger, self.target):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("finite context fractions required")
        if not 0 < self.target < self.trigger < 1 or self.hard_limit <= 0:
            raise ValueError("target < trigger < 1 and positive input space required")

    @property
    def hard_limit(self):
        return self.window - self.output_reserve - self.margin

    @property
    def trigger_limit(self):
        return min(math.floor(self.trigger * self.window), self.hard_limit)

    @property
    def target_limit(self):
        return min(math.floor(self.target * self.window), self.hard_limit)


class NativeContext:
    def __init__(self, policy: ContextPolicy, *, counter: Callable, summarizer: ModelPort,
                 checkpoint: Callable | None = None):
        if not isinstance(policy, ContextPolicy):
            raise TypeError("policy must be ContextPolicy")
        self.policy, self.counter, self.summarizer = policy, counter, summarizer
        self.checkpoint = checkpoint

    @staticmethod
    def project(journal: NativeJournal):
        snapshot = journal.snapshot()
        active = next((r for r in reversed(snapshot.records) if r.event()["kind"] == "context_activated"), None)
        keep = set(active.event()["retained_sequences"]) if active else set()
        included = tuple(r.sequence for r in snapshot.records if (
            active is not None and r.sequence == active.sequence
            or r.event()["kind"] in {'user_message', 'host_observation'}
            or r.event()["kind"] in {"model_returned", "effect_returned"}
            and (active is None or r.sequence > active.sequence or r.sequence in keep)))
        return journal.project(ProjectionSpec(1, "native-context-v1", included))

    @staticmethod
    def _check(control, journal, revision):
        if control.should_stop():
            raise ContextBlocked("CONTEXT_CONTROL_STOPPED")
        if journal.snapshot().revision != revision:
            raise ContextBlocked("CONTEXT_CHANGED_DURING_PREPARATION")

    def _count(self, request, journal, control):
        revision = journal.snapshot().revision
        self._check(control, journal, revision)
        try:
            row = self.counter(request, control)
        except Exception as exc:
            raise ContextBlocked("CONTEXT_COUNT_ERROR") from exc
        self._check(control, journal, revision)
        # Count observations must identify the same projection and deployment limits.
        if not isinstance(row, dict):
            raise ContextBlocked("INVALID_CONTEXT_COUNT")
        if (row.get("context_sha256") != request.context.sha256
                or row.get("context_window") != self.policy.window
                or row.get("output_tokens") != self.policy.output_reserve
                or row.get("request_purpose") != request.purpose
                or not isinstance(row.get("request_sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["request_sha256"])
                or not isinstance(row.get("profile_digest"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["profile_digest"])):
            raise ContextBlocked("CONTEXT_COUNT_IDENTITY_MISMATCH")
        journal.append({"kind": "context_counted", "call_id": request.call_id, "count": row})
        if row.get('status') == 'estimated' and self.policy.allow_estimated_counts:
            if (not row.get('limitation') or row.get('input_tokens') is not None
                    or not row.get('method') or type(row.get('estimated_input_tokens')) is not int
                    or row['estimated_input_tokens'] < 0):
                raise ContextBlocked('CONTEXT_COUNT_UNAVAILABLE')
            # Internal admission value only. Journal and usage retain the estimate
            # separately from measured tokens; compaction uses the same policy.
            return row | {'input_tokens': row['estimated_input_tokens']}
        if (row.get("status") != "observed" or not row.get("method")
                or type(row.get("input_tokens")) is not int or row["input_tokens"] < 0):
            raise ContextBlocked("CONTEXT_COUNT_UNAVAILABLE")
        return row

    def prepare(self, request: ModelRequest, journal: NativeJournal,
                control: ExecutionControl, remaining_calls: int | None) -> ModelRequest:
        from .native import ModelReply, ModelFailure, ToolCall

        count = self._count(request, journal, control)
        # Disabled compaction and the final call use the hard admission ceiling.
        # A soft trigger must not spend the last delivery opportunity on a summary.
        limit = (self.policy.hard_limit if self.policy.max_compactions == 0 or remaining_calls == 1
                 else self.policy.trigger_limit)
        if count["input_tokens"] <= limit:
            return replace(request, admitted_request_sha256=count["request_sha256"])

        events = [r.event() for r in journal.snapshot().records]
        attempts = [e for e in events if e["kind"] == "context_summary_requested"]
        closed = {e["call_id"] for e in events if e["kind"] in
                  {"context_summary_rejected", "context_activated", "context_summary_failed"}}
        candidate = None
        intent = None
        summary_call_spent = False
        # Resume a completed, unactivated summary without invoking another model.
        for e in reversed(events):
            if e["kind"] == "context_summary_returned" and e["call_id"] not in closed:
                intent = next(i for i in attempts if i["call_id"] == e["call_id"])
                if intent["source_context_sha256"] == request.context.sha256:
                    data = dict(e["reply"])
                    data["tool_calls"] = tuple(ToolCall(**c) for c in data["tool_calls"])
                    candidate = ModelReply(**data)
                    break
        if candidate is None:
            if self.policy.max_compactions is not None and len(attempts) >= self.policy.max_compactions:
                raise ContextBlocked("COMPACTION_BUDGET")
            if remaining_calls is not None and remaining_calls < 2:  # One summary plus the next implementation call.
                raise ContextBlocked("MODEL_CALL_BUDGET")
            records = {r.sequence: r for r in journal.snapshot().records}
            selected = [records[n] for n in request.context.included_sequences]
            prior = [r for r in selected if r.event()["kind"] == "context_activated"]
            turns = []
            for record in selected:
                kind = record.event()["kind"]
                if kind == "model_returned":
                    turns.append([record])
                elif kind == "effect_returned":
                    if not turns:
                        raise ContextBlocked("INCOMPLETE_CONTEXT_TURN")
                    turns[-1].append(record)
            prefix = prior + [r for turn in turns[:-self.policy.keep_recent_turns] for r in turn]
            # Explicit user input is protected, like the initial instruction.
            # Keep it verbatim across compaction; overflow remains a visible limit.
            retained = sorted([r.sequence for turn in turns[-self.policy.keep_recent_turns:] for r in turn]
                              + [r.sequence for r in selected if r.event()['kind'] in {'user_message', 'host_observation'}])
            if not prefix:
                raise ContextBlocked("NO_COMPACTABLE_HISTORY")
            prefix_projection = journal.project(ProjectionSpec(
                1, "native-summary-source-v1", tuple(sorted(
                    [r.sequence for r in prefix]
                    + [r.sequence for r in selected if r.event()['kind'] in {'user_message', 'host_observation'}]))))
            summary_id = f"{request.session_id}/summary/{len(attempts) + 1}"
            summary_request = replace(request, call_id=summary_id, context=prefix_projection,
                purpose="summary", remaining_model_calls=None, suffix=(
                    "Summarize the earlier task history as a concise continuation checkpoint, aiming for "
                    "200-400 words; expand only for "
                    "essential details. Use: Objective; Completed changes; Last verification; Remaining work; "
                    "Unconfirmed hypotheses. Preserve exact paths and necessary references. Report the latest "
                    "tool-observed outcome, including failed collection or syntax errors; distinguish facts "
                    "from claims. A successful edit means the requested replacement was applied, not that "
                    "the code is correct. Do not turn suspected tool corruption into fact. Omit source dumps, "
                    "repeated output and abandoned repair narratives. Keep concrete next actions and avoid "
                    "inventing new obligations. Recent complete turns remain available separately. "
                    "Return summary text only; do not call tools or propose delivery."))
            summary_count = self._count(summary_request, journal, control)
            if summary_count["profile_digest"] != count["profile_digest"]:
                raise ContextBlocked("SUMMARY_PROFILE_MISMATCH")
            if summary_count["input_tokens"] > self.policy.hard_limit:
                raise ContextBlocked("SUMMARY_INPUT_OVERFLOW")
            intent = {"kind": "context_summary_requested", "call_id": summary_id,
                      "source_context_sha256": request.context.sha256,
                      "profile_digest": count["profile_digest"],
                      "retained_sequences": retained,
                      "covered": [{"sequence": r.sequence, "sha256": r.sha256} for r in prefix]}
            if self.checkpoint is not None:
                # The host projects tool facts; the summarizer cannot rewrite them.
                # Store once with the intent, so interrupted activation reuses it.
                observed = self.checkpoint(journal.snapshot())
                if not isinstance(observed, str):
                    raise ContextBlocked("INVALID_OPERATIONAL_CHECKPOINT")
                intent["observed_state"] = observed
            revision = journal.snapshot().revision
            self._check(control, journal, revision)
            journal.append(intent)
            if control.should_stop():
                journal.append({"kind": "context_summary_failed", "call_id": summary_id, "reason": "NOT_STARTED"})
                raise ContextBlocked("CONTEXT_CONTROL_STOPPED")
            revision = journal.snapshot().revision
            try:
                summary_call_spent = True
                candidate = self.summarizer.generate(replace(summary_request,
                    admitted_request_sha256=summary_count["request_sha256"]), control)
            except ModelFailure as exc:
                journal.append({"kind": "context_summary_failed", "call_id": summary_id, "reason": exc.code,
                                "usage": None, **({'not_started': True} if exc.not_started else {})})
                raise ContextBlocked(exc.code if exc.not_started else "SUMMARY_FAILED") from None
            except Exception as exc:
                journal.append({"kind": "context_summary_interrupted", "call_id": summary_id,
                                "error_type": type(exc).__name__, "usage": None})
                raise ContextBlocked("SUMMARY_UNFINISHED") from None
            if not isinstance(candidate, ModelReply):
                journal.append({"kind": "context_summary_failed", "call_id": summary_id, "reason": "INVALID_SUMMARY"})
                raise ContextBlocked("INVALID_SUMMARY")
            changed = journal.snapshot().revision != revision
            journal.append({"kind": "context_summary_returned", "call_id": summary_id, "reply": asdict(candidate)})
            if changed:
                journal.append({"kind": "context_summary_rejected", "call_id": summary_id, "reason": "SOURCE_CHANGED"})
                raise ContextBlocked("CONTEXT_CHANGED_DURING_SUMMARY")

        assert intent is not None and candidate is not None
        if not candidate.complete or candidate.tool_calls or candidate.proposes_delivery or not candidate.text.strip():
            journal.append({"kind": "context_summary_rejected", "call_id": intent["call_id"], "reason": "INVALID_SUMMARY"})
            raise ContextBlocked("INVALID_SUMMARY")
        activation = {"kind": "context_activated", "call_id": intent["call_id"],
                      "source_context_sha256": intent["source_context_sha256"],
                      "profile_digest": intent["profile_digest"], "summary": candidate.text,
                      "retained_sequences": intent["retained_sequences"], "covered": intent["covered"]}
        if "observed_state" in intent:
            activation["observed_state"] = intent["observed_state"]
        # A disposable preview log is not the canonical session. Its source refs
        # point back to the real journal; only the request bytes are admitted.
        # The durable journal's bounded envelope already admitted this source.
        # Do not silently reapply ConversationLog's unrelated 2 MiB default.
        preview = ConversationLog(ConversationLimits(max_projection_bytes=max(
            ConversationLimits().max_projection_bytes,
            len(request.context.projected_bytes) + ConversationLimits().max_event_bytes)))
        records = {r.sequence: r for r in journal.snapshot().records}
        for seq in intent["retained_sequences"]:
            preview.append(records[seq].event())
        preview.append(activation)
        projection = preview.project(ProjectionSpec(1, "native-preview-v1"))
        remaining = request.remaining_model_calls
        if summary_call_spent and remaining is not None:
            remaining -= 1
        proposed = replace(request, context=projection, remaining_model_calls=remaining)
        candidate_count = self._count(proposed, journal, control)
        if candidate_count["profile_digest"] != intent["profile_digest"]:
            raise ContextBlocked("SUMMARY_PROFILE_MISMATCH")
        # The target is a compression goal, not the admission ceiling. A
        # counted candidate below the trigger can continue safely even when
        # protected input and the retained turn prevent reaching the target.
        if candidate_count["input_tokens"] > self.policy.trigger_limit:
            journal.append({"kind": "context_summary_rejected", "call_id": intent["call_id"], "reason": "TARGET_NOT_REACHED"})
            raise ContextBlocked("COMPACTION_TARGET_NOT_REACHED")
        revision = journal.snapshot().revision
        self._check(control, journal, revision)
        journal.append(activation)
        return replace(proposed, context=self.project(journal), admitted_request_sha256=candidate_count["request_sha256"])
