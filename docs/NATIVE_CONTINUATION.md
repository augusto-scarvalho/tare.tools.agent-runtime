# Native continuation — 0.13.0

Since 0.14.5, an authenticated host continuation may explicitly supply
`model_call_limit=<cumulative ceiling>`. The grant is recorded on that user
message and survives restart with the original session header. It cannot lower
the ceiling, reset model/summary consumption, bypass a pending invocation,
renew time or change other limits. Ordinary input without a grant retains the
previous ceiling. Hosts own authorization and any aggregate worker/spend limits;
Runtime adds no Work or provider policy. Journals containing this optional grant
require Runtime 0.14.5 or newer.

`NativeSession.continue_with(control, message_id=..., text=...)` accepts
host-authenticated user input after a completed proposal. Host holds exclusive
custody and preserves the original deadline, model/configuration and budgets.
Runtime owns the single atomic `user_message` append, linked by `after_sequence`
to the preceding terminal. It does not read stdin or import providers/OS.

Returns True for a new append and False for an identical retry of the latest
message. A stale ID or conflicting retry raises ValueError. Pending effects,
unknown invocations, blocked terminals, expired control and exhausted calls
cannot be cleared with a new instruction. Replaying the journal rejects invalid
transitions. Calls/retries and effect identities remain cumulative; old proposals
remain in history. The host invokes `run` separately after the append, allowing
recovery at that boundary without duplicate input.

The active context and summary source include explicit user messages. Those
messages are protected verbatim during compaction, so a correction cannot be
removed by the summarizer. Protected-input overflow remains a visible bound.
Initial instruction/header identity is unchanged. No implicit time/budget
extension, mid-invocation steering or Work authority is introduced.

Existing `tare-native-session/1` journals remain compatible; histories containing
the new event require 0.13.0.
