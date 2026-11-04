# Native attachments

`AttachmentRef(name, media_type, sha256, size_bytes)` identifies a host-owned
content snapshot. Its lower-case SHA-256 and byte length describe the original
bytes. The display name is not a path or a capability. Runtime performs no file
reads, MIME detection, content extraction, upload or provider conversion.

Pass a tuple of references as `NativeSession(..., attachments=...)` for initial
input, or to `continue_with(..., attachments=...)` for a follow-up. The optional
live-message port accepts `NativeMessage(message_id, text, attachments)` as well
as the existing `(message_id, text)` pair. Blank text is valid only when at least
one attachment is present. Each message admits at most 16 distinct references;
each reference describes at most 16 MiB. Hosts and providers may enforce stricter
content and combined-request limits.

Initial references are part of the saved session header and every corresponding
`ModelRequest`. Later references accompany protected `user_message` events.
Compaction preserves both forms, including in summary requests. Reusing a
message ID with different text or references is rejected; reopening a completed
session does not repeat model calls or content delivery. Messages enter context
only between complete model/tool rounds.

The model port must resolve and verify every referenced snapshot before
dispatch, include it in request counting and encoding, or explicitly refuse the
unsupported input. A missing or changed blob must never become an omitted
attachment or a newly read version of the source file. OS owns per-session
retention and Kernel owns content loading and provider conversion.

Text-only headers and messages retain their existing serialized shapes. The
attachment field is absent for an empty tuple. Older readers may reject messages
containing attachments; this extension is not permission to discard them.

This contract does not by itself add a TUI picker, image understanding or
document parsing.
