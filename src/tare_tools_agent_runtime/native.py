"""Single-writer native session. No provider, filesystem or settlement ownership.

The journal must publish each append atomically before returning. The host owns
exclusive custody across session instances/processes; the lock here prevents
reentry on one instance only. An unfinished external invocation is never replayed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from threading import Lock
from typing import Any, Callable, Mapping, Protocol

from .contracts import ExecutionControl
from .conversation import (
    ContextProjection, ConversationLimits, ConversationLog, ConversationRecord,
    ConversationSnapshot, ConversationRejected, ConversationRejectReason, ProjectionSpec,
)
from .native_context import ContextBlocked, NativeContext
from .native_content import (
    AttachmentRef, NativeMessage, attachment_records, validate_attachments, validate_user_text,
)


def _text(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("non-empty text required")


def _object(raw: str, *, max_bytes: int | None = None) -> str:
    # Share the existing strict, bounded JSON ingress (including duplicate keys).
    limits = ConversationLimits() if max_bytes is None else ConversationLimits(max_event_bytes=max_bytes)
    return ConversationLog(limits).append_json(raw).canonical_bytes.decode("utf-8")


def _validate_observation_fields(
    observation_id: str, text: str, after_sequence: int, after_sha256: str
) -> None:
    if (not isinstance(observation_id, str) or not observation_id.strip()
            or len(observation_id.encode("utf-8")) > 128):
        raise ValueError("invalid host observation identity")
    if (not isinstance(text, str) or not text.strip()
            or len(text.encode("utf-8")) > 16384):
        raise ValueError("invalid host observation text")
    if type(after_sequence) is not int or after_sequence < 1:
        raise ValueError("invalid host observation sequence")
    if not isinstance(after_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", after_sha256):
        raise ValueError("invalid host observation hash")


@dataclass(frozen=True)
class NativeBudget:
    max_model_calls: int | None = 16
    max_stagnant_turns: int | None = 3
    max_model_retries: int | None = 0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            minimum = 0 if name == "max_model_retries" else 1
            if value is not None and (type(value) is not int or value < minimum):
                raise ValueError(f"invalid {name}")


@dataclass(frozen=True)
class ToolCall:
    call_id: str
    name: str
    arguments_json: str = field(repr=False)

    def __post_init__(self) -> None:
        _text(self.call_id)
        _text(self.name)
        object.__setattr__(self, "arguments_json", _object(self.arguments_json))


@dataclass(frozen=True)
class ModelReply:
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    complete: bool = True
    proposes_delivery: bool = False
    metadata_json: str = field(default="{}", repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("text must be a string")
        if type(self.complete) is not bool or type(self.proposes_delivery) is not bool:
            raise TypeError("completion flags must be booleans")
        if not isinstance(self.tool_calls, tuple) or not all(
            isinstance(call, ToolCall) for call in self.tool_calls
        ):
            raise TypeError("tool_calls must be a tuple of ToolCall")
        # A typed delivery may occur both in the provider envelope and in its
        # validated payload. Give metadata its own finite ingress bound; tool
        # arguments and the host journal retain their independent limits.
        object.__setattr__(self, "metadata_json", _object(self.metadata_json, max_bytes=128 * 1024))


@dataclass(frozen=True)
class ModelRequest:
    session_id: str
    call_id: str
    instruction: str = field(repr=False)
    capabilities: tuple[str, ...]
    context: ContextProjection = field(repr=False)
    purpose: str = "model"
    suffix: str = field(default="", repr=False)
    required_context: tuple[str, ...] = field(default=(), repr=False)
    admitted_request_sha256: str | None = None
    attachments: tuple[AttachmentRef, ...] = field(default=(), repr=False)
    remaining_model_calls: int | None = None

    def __post_init__(self):
        validate_attachments(self.attachments)
        if self.remaining_model_calls is not None and (type(self.remaining_model_calls) is not int
                                                       or self.remaining_model_calls < 1):
            raise ValueError('remaining_model_calls must be positive or None')


@dataclass(frozen=True)
class EffectRequest:
    session_id: str
    effect_id: str
    tool_call: ToolCall


@dataclass(frozen=True)
class EffectResult:
    effect_id: str
    status: str  # succeeded, refused, failed, unknown
    output_json: str = field(default="{}", repr=False)
    progressed: bool = False  # Observed by the capability adapter, not the model.

    def __post_init__(self) -> None:
        _text(self.effect_id)
        if self.status not in {"succeeded", "refused", "failed", "unknown"}:
            raise ValueError("invalid effect status")
        if type(self.progressed) is not bool:
            raise TypeError("progressed must be boolean")
        if self.progressed and self.status != "succeeded":
            raise ValueError("only a confirmed success can advance progress")
        object.__setattr__(self, "output_json", _object(self.output_json))


class ModelFailure(Exception):
    """A completed failed call; only an explicit retryable failure can be retried."""

    def __init__(self, code: str, *, retryable: bool = False, not_started: bool = False) -> None:
        _text(code)
        if type(retryable) is not bool:
            raise TypeError("retryable must be boolean")
        if type(not_started) is not bool:
            raise TypeError('not_started must be boolean')
        super().__init__(code)
        self.code = code
        self.retryable = retryable
        self.not_started = not_started


class ModelPort(Protocol):
    def generate(self, request: ModelRequest, control: ExecutionControl) -> ModelReply: ...


class CapabilityPort(Protocol):
    def execute(self, request: EffectRequest, control: ExecutionControl) -> EffectResult: ...


class NativeJournal(Protocol):
    """The existing conversation surface, supplied by the host; no storage discovery."""

    def append(self, event: Mapping[str, Any]) -> ConversationRecord: ...
    def snapshot(self) -> ConversationSnapshot: ...
    def project(self, spec: ProjectionSpec) -> ContextProjection: ...


@dataclass(frozen=True)
class NativeOutcome:
    status: str  # proposed, paused, blocked, exhausted, cancelled, rejected, reconciliation_required
    reason: str
    model_calls: int
    output: str = ""
    pending_id: str | None = None


@dataclass
class _State:
    binding: dict[str, Any] | None = None
    model_call_limit: int | None = None
    calls: int = 0
    auxiliary_calls: int = 0
    failures: int = 0
    stagnant: int = 0
    pending: str | None = None
    reply: ModelReply | None = None
    effect_index: int = 0
    reply_progressed: bool = False
    terminal: dict[str, Any] | None = None
    tool_ids: set[str] = field(default_factory=set)

    @property
    def total_calls(self):
        return self.calls + self.auxiliary_calls


class NativeSession:
    """A native loop with independently optional limits, selected instead of an aggregate TurnExecutor loop.

    Initial limits and identity are fixed in the first journal event. A host-authorized
    input may raise the model-call ceiling explicitly; other limits remain fixed. A resumed session
    counts previous model calls and retries. Deadlines remain host-supplied via
    ExecutionControl; the host must preserve its execution deadline on resume.
    """

    def __init__(
        self, *, session_id: str, instruction: str, capabilities: tuple[str, ...],
        model: ModelPort, tools: CapabilityPort, journal: NativeJournal,
        budget: NativeBudget = NativeBudget(),
        context: NativeContext | None = None,
        required_context: tuple[str, ...] = (),
        attachments: tuple[AttachmentRef, ...] = (),
        final_call_reserve_seconds: float | None = None,
    ) -> None:
        _text(session_id)
        validate_user_text(instruction, attachments)
        if not isinstance(capabilities, tuple) or len(set(capabilities)) != len(capabilities):
            raise ValueError("capabilities must be a unique tuple")
        for name in capabilities:
            _text(name)
        if not isinstance(budget, NativeBudget):
            raise TypeError("budget must be NativeBudget")
        self.session_id, self.instruction = session_id, instruction
        self.capabilities, self.budget = capabilities, budget
        self.model, self.tools, self.journal = model, tools, journal
        if context is not None and not isinstance(context, NativeContext):
            raise TypeError("context must be NativeContext")
        if not isinstance(required_context, tuple):
            raise TypeError("required_context must be a tuple")
        for source in required_context:
            _text(source)
        self.context, self.required_context = context, required_context
        self.attachments = attachments
        if final_call_reserve_seconds is not None and (
                isinstance(final_call_reserve_seconds, bool) or not isinstance(final_call_reserve_seconds, (int, float))
                or not math.isfinite(final_call_reserve_seconds) or final_call_reserve_seconds <= 0):
            raise ValueError('final_call_reserve_seconds must be positive and finite or None')
        self.final_call_reserve_seconds = final_call_reserve_seconds
        self._lock = Lock()

    def _header(self, binding):
        header = {"kind": "session_started", "schema": "tare-native-session/1",
                  "session_id": self.session_id, "instruction": self.instruction,
                  "capabilities": list(self.capabilities), "budget": asdict(self.budget),
                  "binding": dict(binding)}
        if self.context is not None:
            header["context_policy"] = asdict(self.context.policy)
            if not self.context.policy.allow_estimated_counts:
                # Preserve the identity of existing measured-count sessions.
                header['context_policy'].pop('allow_estimated_counts')
        if self.required_context:
            header["required_context"] = list(self.required_context)
        if self.attachments:
            header['attachments'] = attachment_records(self.attachments)
        if self.final_call_reserve_seconds is not None:
            header['final_call_reserve_seconds'] = self.final_call_reserve_seconds
        return header

    def binding_identity(self):
        """Current public binding from validated history; host still owns authority."""
        records = self.journal.snapshot().records
        if not records:
            return None
        first = records[0].event()
        if first != self._header(first["binding"]):
            raise ValueError("native session identity/configuration mismatch")
        return dict(self._read().binding)

    def rebind(self, control: ExecutionControl, *, previous_binding, handoff_id: str):
        """Record a host-authorized handoff under exclusive custody, without a call.

        Host authenticates the causal handoff and supplies live successor control.
        Pending invocations and terminal sessions cannot transfer execution.
        """
        _text(handoff_id)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("session already running")
        try:
            current = self.binding_identity()
            successor = {name: getattr(control.binding, name) for name in
                         ("work_id", "binding_id", "owner_id", "fence")}
            if current != previous_binding or current is None:
                raise ValueError("native rebind predecessor mismatch")
            state = self._read()
            if state.pending or state.terminal:
                raise ValueError("native rebind requires a nonterminal session without pending invocation")
            self._check_rebind(current, successor)
            if control.should_stop():
                raise ValueError("native rebind successor control stopped")
            self.journal.append({"kind": "binding_rebound", "previous": current,
                                 "binding": successor, "handoff_id": handoff_id})
        finally:
            self._lock.release()

    @staticmethod
    def _check_rebind(previous, successor):
        if (set(successor) != {"work_id", "binding_id", "owner_id", "fence"}
                or successor["work_id"] != previous["work_id"]
                or successor["binding_id"] == previous["binding_id"]
                or type(successor["fence"]) is not int or successor["fence"] <= previous["fence"]):
            raise ValueError("invalid native binding succession")
        for name in ("work_id", "binding_id", "owner_id"):
            _text(successor[name])

    def run(self, control: ExecutionControl, *,
            receive_message: Callable[[tuple[str, ...]], NativeMessage | tuple[str, str] | None] | None = None,
            pause_requested: Callable[[], bool] | None = None) -> NativeOutcome:
        """Consume optional host input between complete rounds, under journal custody.

        The host retains messages until their IDs appear in the journal. Delivery
        acknowledges context insertion, not model understanding. Running calls and
        partially executed tool batches are never interrupted or discarded.
        A host pause returns only at a complete boundary, before receiving input;
        it writes no pause/terminal event and never supersedes a terminal result.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("session already running")
        try:
            binding = {name: getattr(control.binding, name) for name in
                       ("work_id", "binding_id", "owner_id", "fence")}
            records = self.journal.snapshot().records
            if not records:
                self.journal.append(self._header(binding))
            elif self.binding_identity() != binding:
                raise ValueError("native session identity/configuration mismatch")
            while True:
                state = self._read()
                stopped = self._stop(control, state)
                if stopped:
                    if (stopped.reason == "DEADLINE" and state.pending is None
                            and state.terminal is not None):
                        return NativeOutcome(state.terminal["status"], state.terminal["reason"],
                                             state.total_calls, state.terminal["output"])
                    return stopped
                if state.pending:
                    return NativeOutcome("reconciliation_required", "UNFINISHED_INVOCATION",
                                         state.total_calls, pending_id=state.pending)
                if ((state.reply is None or state.terminal is not None)
                        and pause_requested is not None and pause_requested()):
                    # The callback may race with cancellation, revocation or the
                    # deadline. Those controls still take precedence over pause.
                    stopped = self._stop(control, state)
                    if stopped:
                        if not (stopped.reason == "DEADLINE" and state.terminal is not None):
                            return stopped
                    if state.terminal is not None:
                        return NativeOutcome(state.terminal["status"], state.terminal["reason"],
                                             state.total_calls, state.terminal["output"])
                    return NativeOutcome("paused", "PAUSE_REQUESTED", state.total_calls)
                ready = (state.reply is None and state.terminal is None or
                         state.terminal is not None and state.terminal['status'] == 'proposed')
                if (receive_message is not None and ready and
                        (state.model_call_limit is None or state.total_calls < state.model_call_limit)):
                    records = self.journal.snapshot().records
                    ids = tuple(r.event()['message_id'] for r in records if r.event()['kind'] == 'user_message')
                    message = receive_message(ids)
                    if message is not None:
                        if not isinstance(message, NativeMessage):
                            if not isinstance(message, tuple) or len(message) != 2:
                                raise ValueError('invalid native live message')
                            message = NativeMessage(*message)
                        if message.message_id in ids:
                            raise ValueError('invalid native live message identity or size')
                        if control.should_stop():
                            return self._stop(control, state)
                        self.journal.append(message.event(records[-1].sequence))
                        continue
                if state.terminal:
                    return NativeOutcome(state.terminal["status"], state.terminal["reason"],
                                         state.total_calls, state.terminal["output"])
                if state.reply is not None:
                    self._consume(state, control)
                    continue
                if self.budget.max_stagnant_turns is not None and state.stagnant >= self.budget.max_stagnant_turns:
                    return NativeOutcome("blocked", "NO_PROGRESS", state.total_calls)
                if (state.model_call_limit is not None and state.total_calls >= state.model_call_limit):
                    return NativeOutcome("exhausted", "MODEL_CALL_BUDGET", state.total_calls)
                prepared = self._generate(state, control)
                if prepared is not None:
                    return prepared
        finally:
            self._lock.release()

    def reconcile_effect(self, control: ExecutionControl, request: EffectRequest,
                         result: EffectResult) -> bool:
        """Record a host-proven outcome under custody, without executing a tool.

        The host owns durable evidence and workspace checks. Recovery cannot
        change the pending request, renew limits or settle an unknown result.
        """
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('session already running')
        try:
            binding = {name: getattr(control.binding, name) for name in
                       ('work_id', 'binding_id', 'owner_id', 'fence')}
            if self.binding_identity() != binding or control.should_stop():
                raise ValueError('native effect recovery control rejected')
            if (request.session_id != self.session_id or result.effect_id != request.effect_id
                    or result.status == 'unknown'):
                raise ValueError('native effect recovery identity or outcome rejected')
            state = self._read()
            if (state.reply is None or state.pending != request.effect_id
                    or state.reply.tool_calls[state.effect_index] != request.tool_call):
                raise ValueError('only the pending effect can be reconciled')
            self.journal.append({'kind': 'effect_returned', 'result': asdict(result)})
            return True
        finally:
            self._lock.release()

    def abandon_interrupted_model(self, control: ExecutionControl, *, call_id: str,
                                  resolution_id: str, owner_interrupted: bool = False) -> bool:
        """Host-authorized discard of an interrupted model response, never a tool.

        The host must confirm local invocation cleanup, or attest that the owner
        process ended under exclusive custody, and authorize another call.
        Remote completion and usage remain unknown; admitted calls stay counted.
        """
        _text(call_id)
        _text(resolution_id)
        if type(owner_interrupted) is not bool:
            raise TypeError('owner_interrupted must be boolean')
        if len(resolution_id.encode()) > 128:
            raise ValueError('native resolution identity too long')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('session already running')
        try:
            binding = {name: getattr(control.binding, name) for name in
                       ('work_id', 'binding_id', 'owner_id', 'fence')}
            if self.binding_identity() != binding or control.should_stop():
                raise ValueError('native model recovery control rejected')
            state = self._read()
            records = self.journal.snapshot().records
            for record in records:
                event = record.event()
                if event['kind'] == 'model_abandoned' and event['resolution_id'] == resolution_id:
                    if event['call_id'] != call_id:
                        raise ValueError('native resolution identity conflict')
                    return False
            last = records[-1].event() if records else {}
            observed = (last.get('kind') == 'invocation_interrupted' and last.get('invocation_id') == call_id)
            crashed = (owner_interrupted and last.get('kind') == 'model_requested' and last.get('call_id') == call_id)
            if (state.reply is not None or state.pending != call_id
                    or call_id != f'{self.session_id}/model/{state.calls}'
                    or not (observed or crashed)):
                raise ValueError('only an observed interrupted model can be abandoned')
            self.journal.append({'kind': 'model_abandoned', 'call_id': call_id,
                                 'resolution_id': resolution_id, 'remote_stop': 'unknown', 'usage': None,
                                 **({'owner_interrupted': True} if crashed else {})})
            return True
        finally:
            self._lock.release()

    def retry_not_started_model(self, control: ExecutionControl, *, call_id: str, resolution_id: str) -> bool:
        """Authorize one explicit retry of a recorded pre-dispatch failure.

        Keep the failed call, history and cumulative budgets. Never use this for
        a request which may have reached the provider or for a tool effect.
        """
        _text(resolution_id)
        if len(resolution_id.encode()) > 128:
            raise ValueError('native resolution identity too long')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('session already running')
        try:
            binding = {name: getattr(control.binding, name) for name in
                       ('work_id', 'binding_id', 'owner_id', 'fence')}
            if self.binding_identity() != binding or control.should_stop():
                raise ValueError('native model recovery control rejected')
            state = self._read()
            records = self.journal.snapshot().records
            for record in records:
                event = record.event()
                if event['kind'] == 'model_retry_authorized' and event['resolution_id'] == resolution_id:
                    if event['call_id'] != call_id:
                        raise ValueError('native resolution identity conflict')
                    return False
            last = records[-1].event() if records else {}
            if (state.pending or state.reply is not None
                    or last.get('kind') != 'model_failed' or last.get('not_started') is not True
                    or last.get('call_id') != call_id or call_id != f'{self.session_id}/model/{state.calls}'):
                raise ValueError('only a model failure recorded as not started can be retried')
            if state.model_call_limit is not None and state.total_calls >= state.model_call_limit:
                raise ValueError('native model recovery execution budget exhausted')
            self.journal.append({'kind': 'model_retry_authorized', 'call_id': call_id,
                                 'resolution_id': resolution_id})
            return True
        finally:
            self._lock.release()

    def continue_with(self, control: ExecutionControl, *, message_id: str, text: str,
                      attachments: tuple[AttachmentRef, ...] = (), interrupts_turn: bool = False,
                      model_call_limit: int | None = None) -> bool:
        """Append host-supplied input; explicit interruption never renews execution limits.

        interrupts_turn lets the host deliver a correction after cancellation or a
        known failure. It supersedes unstarted actions, never an unknown invocation.

        The last message ID is retryable with identical text, even after a crash
        or its completed response. Older IDs and conflicting retries are refused.
        Host holds exclusive journal custody and authenticates the user input.
        Only an explicitly supplied model_call_limit raises the cumulative call
        ceiling. It cannot decrease it, reset consumption or change other limits.
        """
        message = NativeMessage(message_id, text, attachments, interrupts_turn, model_call_limit)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('session already running')
        try:
            current = self.binding_identity()
            binding = {name: getattr(control.binding, name) for name in
                       ('work_id', 'binding_id', 'owner_id', 'fence')}
            if current is None or current != binding:
                raise ValueError('native continuation binding mismatch')
            state = self._read()
            records = self.journal.snapshot().records
            messages = [r.event() for r in records if r.event()['kind'] == 'user_message']
            matches = [m for m in messages if m['message_id'] == message_id]
            if matches:
                if matches[-1] != messages[-1] or NativeMessage.from_event(matches[-1]) != message:
                    raise ValueError('native continuation conflicting or stale message')
                return False
            completed = (state.terminal and state.terminal['status'] == 'proposed'
                         and records[-1].event()['kind'] == 'terminal')
            if state.pending or not (completed or interrupts_turn and records):
                raise ValueError('native continuation requires a completed proposal')
            ceiling = state.model_call_limit
            if model_call_limit is not None:
                if ceiling is None or model_call_limit < ceiling:
                    raise ValueError('native model call limit grant cannot reduce the ceiling')
                ceiling = model_call_limit
            if control.should_stop() or (ceiling is not None and state.total_calls >= ceiling):
                raise ValueError('native continuation execution budget exhausted')
            self.journal.append(message.event(records[-1].sequence))
            return True
        finally:
            self._lock.release()

    def continue_after_observation(self, control: ExecutionControl, *,
                                   observation_id: str, text: str,
                                   after_sequence: int, after_sha256: str) -> bool:
        """Append a host-authorized observation after a proposal; never renew execution limits.

        At most one host observation is admitted in the entire session. Exact duplicate
        replay is an idempotent no-op (returns False) even after continuation completed.
        """
        _validate_observation_fields(observation_id, text, after_sequence, after_sha256)
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('session already running')
        try:
            current = self.binding_identity()
            binding = {name: getattr(control.binding, name) for name in
                       ('work_id', 'binding_id', 'owner_id', 'fence')}
            if current is None or current != binding:
                raise ValueError('native continuation binding mismatch')
            if control.should_stop():
                raise ValueError('native observation control rejected')
            state = self._read()
            records = self.journal.snapshot().records
            observations = [r.event() for r in records if r.event()['kind'] == 'host_observation']
            candidate = {
                'kind': 'host_observation',
                'observation_id': observation_id,
                'text': text,
                'after_sequence': after_sequence,
                'after_sha256': after_sha256,
            }
            if observations:
                if observations[0] != candidate:
                    raise ValueError('native continuation conflicting observation')
                return False
            if (state.pending or not state.terminal or state.terminal['status'] != 'proposed'
                    or not records or records[-1].event()['kind'] != 'terminal'):
                raise ValueError('native continuation requires a completed proposal')
            last = records[-1]
            if last.sequence != after_sequence or last.sha256 != after_sha256:
                raise ValueError('native continuation predecessor mismatch')
            if control.should_stop() or (state.model_call_limit is not None and state.total_calls >= state.model_call_limit):
                raise ValueError('native continuation execution budget exhausted')
            self.journal.append(candidate)
            return True
        finally:
            self._lock.release()

    @staticmethod
    def _stop(control: ExecutionControl, state: _State) -> NativeOutcome | None:
        if control.cancellation.is_cancelled():
            return NativeOutcome("cancelled", "CANCEL_REQUESTED", state.total_calls,
                                 pending_id=state.pending)
        if not control.binding_is_current():
            return NativeOutcome("rejected", "BINDING_REVOKED", state.total_calls,
                                 pending_id=state.pending)
        if control.remaining_seconds() <= 0:
            return NativeOutcome("exhausted", "DEADLINE", state.total_calls,
                                 pending_id=state.pending)
        return None

    def _projection_blocked(self, control: ExecutionControl) -> NativeOutcome:
        current = self._read()
        stopped = self._stop(control, current)
        if stopped:
            return stopped
        return NativeOutcome("reconciliation_required" if current.pending else "blocked",
                             "PROJECTION_TOO_LARGE", current.total_calls, pending_id=current.pending)

    def _generate(self, state: _State, control: ExecutionControl) -> NativeOutcome | None:
        call_id = f"{self.session_id}/model/{state.calls + 1}"
        try:
            if self.context is not None:
                context = self.context.project(self.journal)
            else:
                snapshot = self.journal.snapshot()
                included = tuple(record.sequence for record in snapshot.records
                                 if record.event()["kind"] in {"model_returned", "effect_returned", "user_message", "host_observation"})
                context = self.journal.project(ProjectionSpec(
                    generation=1, projector_version="native-full-v1", included_sequences=included))
        except ConversationRejected as exc:
            if exc.reason != ConversationRejectReason.PROJECTION_TOO_LARGE:
                raise
            return self._projection_blocked(control)
        remaining_calls = None if state.model_call_limit is None else state.model_call_limit - state.total_calls
        if (self.final_call_reserve_seconds is not None
                and control.remaining_seconds() <= self.final_call_reserve_seconds):
            remaining_calls = 1  # Host-selected final-call notice; never extends calls or deadline.
        request = ModelRequest(self.session_id, call_id, self.instruction, self.capabilities,
                               context, required_context=self.required_context, attachments=self.attachments,
                               remaining_model_calls=remaining_calls)
        if self.context is not None:
            try:
                request = self.context.prepare(request, self.journal, control,
                                               remaining_calls)
            except ConversationRejected as exc:
                if exc.reason != ConversationRejectReason.PROJECTION_TOO_LARGE:
                    raise
                return self._projection_blocked(control)
            except ContextBlocked as exc:
                current = self._read()
                stopped = self._stop(control, current)
                if stopped:
                    return stopped
                return NativeOutcome("reconciliation_required" if current.pending else "blocked",
                                     str(exc), current.total_calls, pending_id=current.pending)
        if control.should_stop():
            return self._stop(control, self._read())
        self.journal.append({"kind": "model_requested", "call_id": call_id})
        # We still know dispatch did not happen; persist that fact when possible.
        if control.should_stop():
            self.journal.append({"kind": "invocation_not_started", "invocation_id": call_id})
            return
        try:
            reply = self.model.generate(request, control)
        except ModelFailure as exc:
            if (exc.not_started and exc.retryable and self.budget.max_model_retries is not None
                    and state.failures >= self.budget.max_model_retries):
                # No external invocation occurred. Stop this run when automatic
                # retries are exhausted, but keep explicit resume possible under
                # the original call budget, like cancellation before dispatch.
                self.journal.append({'kind': 'invocation_not_started', 'invocation_id': call_id,
                                     'reason': exc.code})
                return NativeOutcome('blocked', exc.code, state.total_calls + 1)
            self.journal.append({"kind": "model_failed", "call_id": call_id,
                                 "code": exc.code, "retryable": exc.retryable,
                                 **({'not_started': True} if exc.not_started else {})})
            return
        except Exception as exc:
            self._interrupted(call_id, exc)
            return
        if not isinstance(reply, ModelReply):
            self.journal.append({"kind": "model_failed", "call_id": call_id,
                                 "code": "INVALID_REPLY", "retryable": False})
            return
        self.journal.append({"kind": "model_returned", "call_id": call_id,
                             "reply": asdict(reply)})

    def _consume(self, state: _State, control: ExecutionControl) -> None:
        reply = state.reply
        assert reply is not None
        if not reply.complete:
            self._terminal("blocked", "PARTIAL_REPLY", reply.text)
        elif reply.proposes_delivery:
            if reply.tool_calls or not reply.text.strip():
                self._terminal("blocked", "AMBIGUOUS_DELIVERY", reply.text)
            else:
                self._terminal("proposed", "DELIVERY_PROPOSED", reply.text)
        elif not reply.tool_calls:
            self.journal.append({"kind": "reply_consumed"})
        elif len(reply.tool_calls) > 32:
            self._terminal("blocked", "TOOL_BATCH_LIMIT", reply.text)
        else:
            remaining = reply.tool_calls[state.effect_index:]
            ids = [call.call_id for call in remaining]
            if (len(ids) != len(set(ids)) or any(call.name not in self.capabilities
                    or call.call_id in state.tool_ids for call in remaining)):
                self._terminal("blocked", "TOOL_NOT_ADMITTED_OR_DUPLICATE", reply.text)
                return
            call = remaining[0]
            effect_id = self._effect_id(state)
            self.journal.append({"kind": "effect_requested", "effect_id": effect_id,
                                 "tool_call": asdict(call)})
            if control.should_stop():
                self.journal.append({"kind": "invocation_not_started", "invocation_id": effect_id})
                return
            try:
                result = self.tools.execute(EffectRequest(self.session_id, effect_id, call), control)
            except Exception as exc:
                self._interrupted(effect_id, exc)
                return
            if not isinstance(result, EffectResult) or result.effect_id != effect_id:
                # A broken adapter cannot close a pending effect with unrelated evidence.
                return
            self.journal.append({"kind": "effect_returned", "result": asdict(result)})

    def _effect_id(self, state):
        base = f"{self.session_id}/effect/{state.calls}"
        # Preserve identities in existing single-tool journals. Batches are
        # executed serially, with a durable intent/control check for each item.
        return base if len(state.reply.tool_calls) == 1 else f"{base}/{state.effect_index + 1}"

    def _interrupted(self, invocation_id: str, exc: Exception) -> None:
        # Retain the diagnostic category, not potentially secret exception text.
        # Failure to journal this observation still leaves the original intent.
        self.journal.append({"kind": "invocation_interrupted", "invocation_id": invocation_id,
                             "error_type": type(exc).__name__, "usage": None})

    def _terminal(self, status: str, reason: str, output: str = "") -> None:
        self.journal.append({"kind": "terminal", "status": status,
                             "reason": reason, "output": output})

    def _read(self) -> _State:
        """Reduce the journal, rejecting incompatible transitions rather than guessing."""
        records = self.journal.snapshot().records
        state = _State(binding=records[0].event()["binding"] if records else None,
                       model_call_limit=self.budget.max_model_calls)
        message_ids = set()
        resolution_ids = set()
        has_observation = False
        previous = records[0] if records else None
        for record in records[1:]:
            event = record.event()
            kind = event["kind"]
            if kind == 'user_message':
                message = NativeMessage.from_event(event)
                boundary = (message.interrupts_turn or state.terminal is None and state.reply is None or
                            state.terminal is not None and state.terminal['status'] == 'proposed'
                            and previous.event()['kind'] == 'terminal')
                if (not boundary or state.pending
                        or event.get('after_sequence') != previous.sequence):
                    raise ValueError('invalid native continuation transition')
                if message.message_id in message_ids:
                    raise ValueError('invalid native continuation identity or size')
                message_ids.add(event['message_id'])
                if message.model_call_limit is not None:
                    if state.model_call_limit is None or message.model_call_limit < state.model_call_limit:
                        raise ValueError('native model call limit grant cannot reduce the ceiling')
                    state.model_call_limit = message.model_call_limit
                state.terminal = state.reply = None
                state.stagnant = state.effect_index = 0
                state.reply_progressed = False
                previous = record
                continue
            if kind == 'host_observation':
                if has_observation:
                    raise ValueError('multiple host observations not permitted')
                if set(event) != {'kind', 'observation_id', 'text', 'after_sequence', 'after_sha256'}:
                    raise ValueError('invalid host observation event')
                _validate_observation_fields(
                    event['observation_id'], event['text'], event['after_sequence'], event['after_sha256']
                )
                if (state.model_call_limit is not None and state.total_calls >= state.model_call_limit
                        or state.pending or not state.terminal or state.terminal.get('status') != 'proposed'
                        or previous is None or previous.event().get('kind') != 'terminal'
                        or event['after_sequence'] != previous.sequence or event['after_sha256'] != previous.sha256):
                    raise ValueError('invalid native observation transition')
                has_observation = True
                state.terminal = state.reply = None
                state.stagnant = state.effect_index = 0
                state.reply_progressed = False
                previous = record
                continue
            if state.terminal and kind != 'model_retry_authorized':
                raise ValueError("event after terminal")
            if kind == "binding_rebound" and not state.pending:
                if event["previous"] != state.binding:
                    raise ValueError("native rebind predecessor mismatch")
                self._check_rebind(state.binding, event["binding"])
                _text(event["handoff_id"])
                state.binding = event["binding"]
            elif kind == "context_summary_requested" and not state.pending and state.reply is None:
                state.auxiliary_calls += 1
                if event["call_id"] != f"{self.session_id}/summary/{state.auxiliary_calls}":
                    raise ValueError("summary call identity mismatch")
                state.pending = event["call_id"]
            elif kind in {"context_summary_returned", "context_summary_failed"} and (
                state.pending == event["call_id"] == f"{self.session_id}/summary/{state.auxiliary_calls}"
            ):
                state.pending = None
            elif kind == "context_summary_interrupted" and state.pending == event["call_id"]:
                pass
            elif kind in {"context_counted", "context_activated", "context_summary_rejected"} and not state.pending:
                pass
            elif kind == "model_requested" and not state.pending and state.reply is None:
                state.calls += 1
                expected = f"{self.session_id}/model/{state.calls}"
                if event["call_id"] != expected:
                    raise ValueError("model call identity mismatch")
                state.pending = expected
            elif (kind in {"model_returned", "model_failed"} and state.reply is None
                  and state.pending == event["call_id"] == f"{self.session_id}/model/{state.calls}"):
                state.pending = None
                if kind == "model_returned":
                    data = dict(event["reply"])
                    data["tool_calls"] = tuple(ToolCall(**call) for call in data["tool_calls"])
                    state.reply = ModelReply(**data)
                    state.effect_index = 0
                    state.reply_progressed = False
                    # Retries are per call: one failure early in a long session no longer spends the retry of
                    # every later call (2026-11-09, a blocked TUI session).
                    state.failures = 0
                else:
                    state.failures += 1
                    # Failure recorded before a crash must not bypass the retry policy.
                    if not event["retryable"] or (self.budget.max_model_retries is not None and state.failures > self.budget.max_model_retries):
                        state.terminal = {"status": "blocked", "reason": event["code"]
                                          if event["code"] == "INVALID_REPLY" or event.get('not_started') else "MODEL_FAILURE",
                                          "output": ""}
            elif kind == 'model_retry_authorized' and state.reply is None and not state.pending:
                resolution = event.get('resolution_id')
                _text(resolution)
                prior = previous.event()
                if (prior.get('kind') != 'model_failed' or prior.get('not_started') is not True
                        or event.get('call_id') != prior.get('call_id')
                        or event.get('call_id') != f'{self.session_id}/model/{state.calls}'
                        or len(resolution.encode()) > 128 or resolution in resolution_ids):
                    raise ValueError('invalid native model retry authorization')
                resolution_ids.add(resolution)
                state.terminal = None
            elif (kind == 'model_abandoned' and state.reply is None
                  and state.pending == event['call_id'] == f'{self.session_id}/model/{state.calls}'):
                resolution = event.get('resolution_id')
                _text(resolution)
                prior = previous.event()
                observed = prior['kind'] == 'invocation_interrupted' and prior['invocation_id'] == state.pending
                crashed = (event.get('owner_interrupted') is True and prior['kind'] == 'model_requested'
                           and prior['call_id'] == state.pending)
                if (not (observed or crashed)
                        or len(resolution.encode()) > 128 or resolution in resolution_ids
                        or event.get('remote_stop') != 'unknown' or event.get('usage') is not None):
                    raise ValueError('invalid native model abandonment')
                resolution_ids.add(resolution)
                state.pending = None
            elif kind == "reply_consumed" and state.reply and not state.reply.tool_calls and not state.pending:
                state.reply = None
                state.stagnant += 1
            elif kind == "effect_requested" and state.reply and not state.pending:
                call = ToolCall(**event["tool_call"])
                if (not state.reply.complete or state.reply.proposes_delivery
                        or len(state.reply.tool_calls) > 32
                        or state.reply.tool_calls[state.effect_index:state.effect_index+1] != (call,)
                        or call.name not in self.capabilities
                        or call.call_id in state.tool_ids
                        or event["effect_id"] != self._effect_id(state)):
                    raise ValueError("invalid effect intent")
                state.tool_ids.add(call.call_id)
                state.pending = event["effect_id"]
            elif kind == "effect_returned" and state.reply and state.pending:
                result = EffectResult(**event["result"])
                if result.effect_id != state.pending:
                    raise ValueError("effect identity mismatch")
                if result.status == "unknown":
                    # Observation is preserved but cannot clear the uncertainty.
                    continue
                state.pending = None
                state.effect_index += 1
                state.reply_progressed = state.reply_progressed or result.progressed
                if state.effect_index == len(state.reply.tool_calls):
                    state.reply = None
                    state.stagnant = 0 if state.reply_progressed else state.stagnant + 1
            elif kind == "terminal" and not state.pending:
                if event["status"] not in {"blocked", "proposed"}:
                    raise ValueError("invalid native terminal status")
                if event["status"] == "proposed" and (
                    state.reply is None or not state.reply.complete
                    or not state.reply.proposes_delivery or state.reply.tool_calls
                    or not state.reply.text.strip() or event["output"] != state.reply.text
                ):
                    raise ValueError("delivery requires a complete proposal")
                state.terminal = event
            elif kind == "invocation_interrupted" and state.pending == event["invocation_id"]:
                pass
            elif kind == "invocation_not_started" and state.pending == event["invocation_id"]:
                if state.reply is not None:
                    state.tool_ids.remove(state.reply.tool_calls[state.effect_index].call_id)
                state.pending = None
            else:
                raise ValueError(f"invalid native journal transition: {kind}")
            previous = record
        return state
