# Native context admission and compaction — M2b

Local implementation, 2026-09-10, Runtime 0.10.0. No live inference,
production composition, package adoption or federation pin change.

`NativeContext` is injected into `NativeSession`. The Runtime decides admission
and replacement; the injected counter and model adapter own serialization and
remote operations. No OS, Kernel, Aider or frozen harness imports are required.

```python
policy = ContextPolicy(
    window=deployment_window,
    output_reserve=deployment_output_limit,
    margin=64,
    trigger=0.8,
    target=0.5,
    keep_recent_turns=1,
    max_compactions=1,
)
context = NativeContext(policy, counter=model_adapter.count, summarizer=model_adapter)
# Supply context=context and required_context=(original_source_text, ...) to
# NativeSession alongside its model, tools, journal and execution budget.
```

These numbers illustrate configuration, not qualified deployment defaults.
The host supplies the effective limits and expands protected source references.

## Admission

For a shared input/output window W, output reserve O and margin M, the hard
input ceiling is `W - O - M`. The trigger is `min(floor(trigger * W), ceiling)`;
the post-compaction target uses the same formula with `target`. A request at
the trigger is admitted; one above it normally requires compaction. On the last
available model call, preserve the original request up to the hard ceiling:
the soft trigger must not consume the final delivery opportunity or rewrite its
cache prefix. Requests above the hard ceiling still fail closed. O includes reasoning
when the provider accounts for it within output; it must not be subtracted twice.
Profiles with separate input/output limits need their own qualification.

Counting covers the generated request, including system, tool schemas, original
instruction and required sources. Observations identify the projection, purpose,
profile, window, output reserve and request digest. Missing or estimated counts
block this increment explicitly; there is no silent character-count fallback.
The Kernel adapter checks that generation uses the admitted request digest.

The original instruction and `required_context` strings remain byte-preserved
inputs to every request. The journal retains all original records and receipts;
compaction replaces only the derived model projection.

## Transactional compaction

The continuation prompt requests a concise checkpoint (200–400 words as guidance,
not an output or session ceiling): objective, changes, last verification, remaining
work and explicitly unconfirmed hypotheses. Existing model/thinking parameters
and the cache-friendly prefix are retained.

Hosts may inject `checkpoint(snapshot) -> str` into `NativeContext`. The resulting
operational facts are stored with the summary intent and copied unchanged into
activation, including crash recovery. They are part of the counted candidate
request. The Kernel projector supplies bounded edit/verification receipts with
source sequences/hashes and explicit omissions. It does not prove current disk
state, semantic correctness, or execution of tests not present in those receipts.

The newest configured number of complete model/tool turns stays in the next
projection. Older turns and any prior summary form the summary input, alongside
the original instruction and protected sources. Summary requests use the same
profile and tools, with an explicit text-only summarization instruction; returned
tool calls or delivery proposals are rejected, never executed.

The summary request itself must fit the hard ceiling. Its intent, response,
metadata and raw usage are recorded separately. It consumes the session's model
call budget as well as the compaction budget; enough room for the subsequent
main call is required before starting it. There is no hidden second model.

A detached preview combines the summary and retained turns. Only after counting
the full candidate request and bringing it at or below the admission trigger
does one journal append activate it. The lower target is a compression goal,
not a reason to reject a usable candidate: protected input and retained turns
can keep the candidate above that goal. The trigger still reserves output space
and margin, and candidates above it remain blocked. A failed count, oversized
candidate, invalid summary or cancellation leaves the previous projection active. Activation records covered source
sequences and hashes so original evidence remains recoverable.

A completed summary saved before interruption can be reused on resume without
another generation. An intent whose execution is uncertain requires
reconciliation; the Runtime does not repeat it blindly. Resume binds the same
policy and protected sources. Physical durable storage and exclusive custody
across processes remain host responsibilities.

## Qualification and limits

The context tests in `tests/test_native_context.py` cover thresholds, count
provenance, immutable sources, failed transformations, global call budgets,
cancellation, crashes around receipts/activation, resumption and successive
compactions.

Kernel integration separately checks both Chat and Converse, comparing counted
and generated request digests and preserving original sources, thinking blocks
and raw auxiliary usage. See the Kernel guide
[`docs/guides/NATIVE_MODEL_ADAPTERS.md`](https://github.com/augusto-scarvalho/tare.tools.kernel/blob/main/docs/guides/NATIVE_MODEL_ADAPTERS.md).

These tests establish local transaction/protocol behavior. They do not establish
the semantic quality of a real model's summary or savings across an entire task.
Chunked summaries, deterministic deduplication, different summarizer profiles,
normalized usage/cost accounting and physical cancellation are not implemented
here. Oversized fixed input or a summary source that cannot fit produces an
explicit limit. OS source composition and real Work settlement remain M3.

`NativeSession` without `context` retains the M1 primitive behavior; callers must
not mistake that mode for window-qualified production execution. Raw metadata
and admitted model intents are not a complete financial ledger. No production
default has been switched to this implementation.
