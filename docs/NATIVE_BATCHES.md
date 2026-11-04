# Serial tool batches — Runtime 0.12

A real native Qwen Work requested three reads in one complete response. Runtime
0.11 stopped with SERIAL_TOOL_POLICY before executing them. This was a supported
provider response encountering an implementation restriction, not an invalid
model action. The original response remains in the M3 qualification evidence.

Runtime now admits up to 32 calls in a reply and executes them in the supplied
order, one at a time. It validates the remaining names/IDs before dispatch,
journals each intent before its effect, and checks live cancellation, binding
and deadline between effects. A repeated ID or unsupported capability blocks;
an uncertain invocation stops the remainder for reconciliation. Completed
operations are not repeated on resume. Stagnation is measured per reply, with
any confirmed progress in the batch preserving progress.

Single-tool event identities remain compatible. Multiple calls receive an
ordinal suffix on the effect ID. Kernel Chat/Converse codecs correlate every
result to its provider tool ID, preserve reasoning blocks, and refuse incomplete
or reordered projections. Usage joins all effects to the originating model
call rather than treating the ordinal as a generation number.

This adds no parallel writer, automatic extra generation, storage schema or
admission authority. Old terminal refusals remain refusals; upgrading the code
does not reinterpret a completed session as runnable. Unknown effects still
require an explicit reconciliation mechanism outside this increment.
