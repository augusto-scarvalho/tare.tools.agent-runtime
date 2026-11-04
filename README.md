# tare.tools.agent-runtime

*Versão em português: [README.pt-BR.md](README.pt-BR.md).*

The canonical bounded context for tactical agent-loop execution in the tare.tools ecosystem.

Status: **tactical core, isolated native-session increment, immutable conversation, provider-neutral prompt admission and bounded SpecGraph grounding ingress**. OS hosts the current agent runtime through Kernel's provider and process ports; this package has no provider prompt cache, filesystem persistence, daemon, database, registry, UI or plugin system.

`tare.tools.agent-runtime` is the current tactical engine. Older Aider references
describe retired host integrations. This core remains provider-neutral. Task
turn and wall-clock limits can be omitted through
`RunBudget()`; cancellation and configured strict Work checks remain active.

## How it fits in

[`tare.tools.os`](https://github.com/augusto-scarvalho/tare.tools.os) is the host.
It pins this repository as a Git submodule at `satellites/tare.tools.agent-runtime`
and supplies the model and tool ports built on
[`tare.tools.kernel`](https://github.com/augusto-scarvalho/tare.tools.kernel).
This package imports neither of them: everything it needs is injected.

## Install and test

You need Python 3.11 or newer and Git. The package has no runtime dependencies;
the tests are offline and use synthetic fixtures.

```bash
git clone https://github.com/augusto-scarvalho/tare.tools.agent-runtime.git
cd tare.tools.agent-runtime
python -m venv .venv
. .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q src
```

CI runs these same commands on GitHub-hosted Linux and Windows runners. To run them on your
own machines instead, set the repository variables `CI_RUNNER_LINUX` and `CI_RUNNER_WINDOWS`
to a runner label list (for example `["self-hosted","Linux"]`).

## Charter

This repository owns:

- tactical turn progression;
- tactical context carried between turns;
- provider-neutral prompt admission before any turn executor is invoked;
- immutable repository grounding injected before provider execution;
- immutable Work grounding with optional strict drift and progress fencing;
- byte-immutable in-memory tactical conversation events;
- atomic conversation snapshots, deterministic projections and a disposable
  projection cache;
- turn, retry and wall-clock budgets;
- cooperative cancellation state;
- construction and submission of a fenced terminal claim.

It does not own:

- Work, Workflow, strategic Routing, Authority, Evidence or settlement (`tare.tools.os`);
- providers, subprocesses, sandboxes or mediated capability execution (`tare.tools.kernel`);
- Aider internals or the frozen historical `tare.tools.harness`.

The core consumes an injected `TurnExecutor`, `AuthorityGate`, optional
`PromptGate`, and an optional already-materialized `GroundingEnvelope`. When no custom prompt gate is supplied, the bounded
`PromptBudgetGate` is active. Every turn receives the opaque Work binding and
its monotonic fence. The Authority gate is rechecked after an invocation and
before terminal submission, so a late result cannot become canonical after
rebind.

## Native session, context and handoff increments (0.11.0)

`NativeSession` owns model/tool progression through injected `ModelPort` and
`CapabilityPort`. Select it for native execution; do not nest it inside the
aggregate `AgentRuntime` loop. Both modes reuse the existing control and
conversation contracts. OS supplies Work composition and durable attempts;
native proposal-to-settlement integration remains pending.

```python
from tare_tools_agent_runtime import NativeSession, NativeBudget, ConversationLog

session = NativeSession(
    session_id="session-1",
    instruction="Apply the scoped change and verify it.",
    capabilities=("read", "edit", "verify"),
    model=model_adapter,
    tools=capability_adapter,
    journal=ConversationLog(),
    budget=NativeBudget(max_model_calls=8, max_stagnant_turns=3),
)
outcome = session.run(execution_control)
```

Since 0.13.5, hosts may supply `session.run(control, receive_message=receive)`.
The callback receives journaled message IDs and returns the next `(id, text)`
or `None`. It runs only between complete rounds or after a proposal, with execution
budget available. The host retains input until the journal contains its ID.
The durable acknowledgement means context insertion, not model understanding;
running invocations and tool batches finish before the input is consumed. Storage,
operator authentication and UI remain host responsibilities.

`ModelReply` separates text, complete tool requests and an explicit delivery
proposal. A reply may carry up to 32 tool requests, executed one at a time in the
supplied order ([serial tool batches](docs/NATIVE_BATCHES.md)).
`ToolCall.arguments_json` and adapter metadata/results are bounded, strict JSON
objects materialized into immutable strings. Tool call IDs must be normalized
by the adapter to be unique within the session. The capability adapter checks
arguments, expected source version and scope and receives the same live control.
Only adapter-observed progress resets stagnation; prose does not.

The journal records intent before an invocation and observation after it. A
crash or exception between them returns `reconciliation_required` on resume;
neither a model request nor an uncertain effect is automatically repeated.
Hosts that confirm local model-invocation cleanup may explicitly call
`abandon_interrupted_model(control, call_id=..., resolution_id=...)` before
resuming. This records abandonment of an observed interrupted model response;
remote completion/usage remain unknown and the next generation gets a new ID.
It preserves the original request count and refuses pending effects, summaries,
unobserved interruptions and revoked/cancelled/expired control. Resolution IDs
are idempotent only for the same call. This does not automatically recover a
crashed host or undo any effects.

Explicit completed `ModelFailure(retryable=True)` may retry within the fixed
budget. Failure observations remain in the log; missing usage is not zero.
Persisting a complete reply before a crash permits consuming that reply without
calling the model again. Persisting a tool receipt prevents repeating the tool.

For simulated resume, construct a new session against the same journal with the
same identity, instruction, capabilities, budget and binding. With `NativeContext`,
the policy and protected sources are also bound on resume. Used calls/retries
remain counted, including auxiliary summary intents. `model_calls` counts
admitted intents (including interrupted or undispatched reservations), not
confirmed remote executions or billable calls. Metadata is retained for future
normalization, not represented as a complete cost ledger.

For an explicit handoff, the OS authenticates the causal transfer using its
authority ledger, holds exclusive custody and calls `session.rebind(control,
previous_binding=session.binding_identity(), handoff_id=receipt_id)`. The new
control must represent the current successor. This appends `binding_rebound`;
it never rewrites the original header or grants authority itself. Pending
invocations and terminal sessions cannot transfer execution. Construct the
successor's capability adapter with the new binding, then run the same journal.
All earlier model/summary calls, duplicate tool IDs and context remain counted.
The OS wrapper handles a lost append acknowledgement by reading the current
binding before attempting the transition again. Older runtimes must not open
a journal containing the new transition; pre-handoff journals retain their format.

The optional [native context component](docs/NATIVE_CONTEXT.md) admits the full
serialized request against explicit window/output limits and configurable
percentages. Compaction preserves original records and activates a new projection
only after validation. Failed or uncertain summaries cannot silently replace
the active context. Without this component, the M1 example above does not enforce
the provider context window. Real deployment and cost qualification remain open.

The host must retain exclusive custody of the journal across instances/processes
and preserve its execution deadline on resume. The local lock only prevents
reentry on one instance. Physical persistence, external-call reconciliation,
cross-process fencing, provider cancellation and independent Work checks remain
integration work. The adapter must cooperate with cancellation; this core cannot
interrupt a blocked OS call. `proposed` is a delivery proposal, never a test-pass,
terminal claim, commit or admission of a Work. A proposal may contain an incorrect
claim; the OS evaluates it against original evidence.

This increment introduces no Strands or Aider dependency.

## Immutable conversation P1

`ConversationLog` materializes each accepted JSON event once, stores its exact
canonical UTF-8 bytes and SHA-256, and publishes the record, next sequence and
revision under one bounded in-memory writer lock. Snapshots and records are
immutable; read APIs return detached values.

`ProjectionSpec` selects an ordered subset of source events. The resulting
`ContextProjection` is a detached ticket bound to its source revision, snapshot
fingerprint, generation and projector version. The fingerprint is local
integrity metadata, never Work or Authority. Projection bytes are built from the
stored event bytes, cached only as a disposable optimization, and reproducible
after the cache structure is replaced.

```python
from tare_tools_agent_runtime import ConversationLog, ProjectionSpec

conversation = ConversationLog()
conversation.append({"kind": "user", "text": "Inspect the failing test."})
conversation.append({"kind": "assistant", "text": "I found the race."})

ticket = conversation.project(
    ProjectionSpec(generation=1, projector_version="tactical-context-v1")
)
conversation.admit_current(ticket)
```

This is not provider prompt caching. Provider request serialization, cache-point
placement, prefix digests, TTL, persistence and cache telemetry are owned
outside this core by the provider/Kernel integration boundary.

## Pre-provider prompt admission

`AgentRuntime.run()` materializes the submitted instruction once as an immutable
`PromptEnvelope`, including exact UTF-8 bytes and SHA-256 identity. The
provider-neutral `PromptGate` sees that envelope before the `TurnExecutor`.
Denied, malformed or gate-error inputs return `OutcomeStatus.BLOCKED` with zero
turns and zero attempts; no provider adapter runs and no terminal claim is
submitted.

The digest supports identity checks without placing plaintext in outcomes or
representations; it is not a secrecy boundary for low-entropy prompts.

The default `PromptBudgetGate` admits non-blank prompts up to 16,384 UTF-8 bytes.
The limit is configurable by injection. A gate returns only `ALLOW` or `DENY`;
there is deliberately no prompt-rewrite field. On admission, the exact same
`PromptEnvelope` object is attached to every retry and turn request.

```python
from tare_tools_agent_runtime import AgentRuntime, PromptBudgetGate

runtime = AgentRuntime(
    turn_executor=my_kernel_adapter,
    authority_gate=my_os_gate,
    prompt_gate=PromptBudgetGate(max_prompt_bytes=32_768),
)
```

This boundary applies to every provider invoked through the managed runtime.
Vendor-native prompt hooks remain relevant only when a user operates a provider
CLI directly, outside this composition path. Provider-internal tool and model
lifecycle events remain adapter-specific.

## Repository-local canonical tool hooks

This repository is the second opt-in dogfood target for the canonical Kernel
hook policy. Its checked-in Claude Code, Codex and Antigravity manifests are
thin adapters to the installed `tare-hook`; they contain no local policy.

At `PreToolUse`, native grep tools and shell `rg`/`grep` are blocked in favor of
`specgraph resolve <query> --format json`. SpecGraph owns the bounded
`MISS`/`STALE`/`TIMEOUT`/`UNAVAILABLE` recovery, external disposable evidence
and privacy-preserving telemetry. Before and after recognized edits, the same
Kernel policy supplies advisory impact and staleness context where the vendor
contract permits it.

The local `specgraph.yaml` keeps source and tests searchable while classifying
ordinary repository documents only as neutral `search_paths`. It deliberately
does not invent formal specs, schemas, APIs or ADRs. Install `tare-kernel` and
`tare-tools-specgraph` in the environment that launches the vendor CLI; a
missing hook command is not silently replaced by repository code.

## SpecGraph grounding ingress

The host or Kernel adapter runs `specgraph ground` against an explicitly chosen
Library manifest and supplies its exact stdout bytes to `GroundingEnvelope`.
The runtime validates canonical compact JSON, usable status, source digest,
entry counts and the declared byte length. It then attaches the same immutable
object to every `TurnRequest`; it does not merge grounding into the instruction
or tactical conversation, so provider serialization and prompt-cache placement
remain Kernel responsibilities.

```python
from tare_tools_agent_runtime import GroundingEnvelope

grounding = GroundingEnvelope(specgraph_completed_process.stdout)
outcome = runtime.run(
    binding=binding,
    instruction="Implement the accepted packet.",
    budget=budget,
    grounding=grounding,
)
```

There is no Library or SpecGraph import, manifest discovery, network access, or
source-document read in this package. Non-usable SpecGraph outcomes are rejected
before a provider executes.

## Strict managed-work control

`StrictWorkControl` is an opt-in fail-closed mode. The host supplies an exact
`backlog ground` `WorkEnvelope` and an injected `WorkGuard`. The runtime
revalidates that work scope before dispatch, after every result and immediately
before a terminal claim. Drift, unavailable revalidation or a guard error stops
the run with no canonical claim.

Every `CONTINUE` or `SUCCEEDED` result in strict mode must also carry a
Kernel-observed `ProgressSignal`: remaining obligations, verified evidence
count, and state/evidence identities. Regression blocks immediately. Repeated
windows with no reduction in obligations and no evidence gain trigger
`STAGNATION_DETECTED`; changing prose or subdividing the task is not progress.
A `SUCCEEDED` result is blocked while any obligation remains open.
Adapters may also return a `BLOCKED` turn with a stable reason code; Runtime
then emits no terminal claim, including outside strict Work mode.

```python
runtime = AgentRuntime(
    turn_executor=my_kernel_adapter,
    authority_gate=my_os_gate,
    work_guard=my_backlog_guard,
)
outcome = runtime.run(
    binding=binding,
    instruction=instruction,
    budget=budget,
    work=work_envelope,
    strict_work=StrictWorkControl(max_stagnant_turns=2),
)
```

## Development

Install and test as shown in [Install and test](#install-and-test). The package
has no runtime dependencies; Strands is deliberately absent and removable by
construction. Contributor rules are in [AGENTS.md](AGENTS.md).

## Minimal example

```python
from tare_tools_agent_runtime import AgentRuntime, BindingRef, RunBudget

runtime = AgentRuntime(turn_executor=my_kernel_adapter, authority_gate=my_os_gate)
outcome = runtime.run(
    binding=BindingRef(
        work_id="WORK-42",
        binding_id="binding-7",
        owner_id="worker-01",
        fence=7,
        token="opaque-authority-token",
    ),
    instruction="Implement the accepted packet.",
    budget=RunBudget(max_turns=4, max_retries_per_turn=1, wall_clock_seconds=300),
)
```

The host composes these dependencies. The core never discovers or imports them.

Native execution ceilings are independent options. `NativeBudget(max_model_calls=None,
max_stagnant_turns=None, max_model_retries=None)` explicitly removes those ceilings;
omitted fields retain 16/3/0 defaults. `ContextPolicy(max_compactions=None, ...)`
removes only the number-of-compactions ceiling. Calls, retries and summaries remain
observed; context admission, cancellation, binding and pending-effect checks remain
active. The host supplies and persists any deadline (or no deadline). Continuation
has no separate fixed 64-message cap; existing message/journal size bounds remain.

With `max_compactions=0`, requests keep their original history and are admitted up
to `window - output_reserve - margin`. The soft trigger only requests compaction;
it does not discard otherwise usable context when compaction is disabled.

### Host observations (0.14.0)

A host holding exclusive journal custody can call
`continue_after_observation(control, observation_id=..., text=...,
after_sequence=..., after_sha256=...)` after a completed proposal. It records
one explicitly attributed host observation, separately from authenticated user
input. The predecessor must match the exact terminal record. Text is limited to
16 KiB. An identical replay is a no-op; conflicting or second observations fail.
Original control, model identity, deadline and cumulative call ceilings remain.
Compaction preserves the observation. OS owns eligibility and source/effect
reconciliation; this port does not authorize reads, select models or retry tools.
Paired Kernel adapters must render `host_observation` before a host enables it.

### Host pause (0.14.1)

`NativeSession.run(control, pause_requested=host_pause_check)` accepts an optional
zero-argument callback. A true request returns `NativeOutcome(status="paused",
reason="PAUSE_REQUESTED", model_calls=...)` only after the complete model reply
and its original tool batch have been consumed, with no unfinished invocation or
terminal outcome pending. It makes no further model call, does not poll or wait,
and does not receive queued human messages. A completed terminal proposal remains
`proposed`, even when pause is requested, so the host can checkpoint its next stage.
Cancellation, binding revocation, deadlines and unknown-invocation reconciliation
retain precedence. Pause writes no journal event and confers no authority.

Under exclusive journal custody, reconstruct the session with the same validated
journal and configuration, then run with pause false (or omitted). Completed tools
are not repeated; cumulative calls, budgets, identity and binding stay unchanged.
The host must preserve the original deadline. Workflow checkpointing and resumption
policy belong to the host, not this tactical port.

## License

Agent Runtime code is licensed under [Apache-2.0](LICENSE). Injected providers,
tools and other host dependencies retain their own licenses.
