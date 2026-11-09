# Changelog

Notable changes to `tare.tools.agent-runtime` are recorded here, newest first.

## Unreleased

- `max_model_retries` counts per call: a model call that returns resets the failure count, so a failure early in a long session no longer spends the retry of every later call. Two failures in a row still exceed one retry, and a resume still never resets the count (2026-11-09: a second slip of the AGY CLI blocked a TUI session because its first slip, calls earlier, had used the session's only retry).
- CI runs on GitHub-hosted Linux and Windows runners (`Tests (linux)`, `Tests (windows)`) with the README's install and test commands and a pip cache. The repository variables `CI_RUNNER_LINUX` and `CI_RUNNER_WINDOWS` move it to your own runners, and pull requests from forks then skip it.
- First public release. The repository starts from a single commit; earlier history is not published. It contains the
  bounded tactical agent-loop runtime that tare.tools.os hosts through the kernel's provider and process ports, and
  its test suite.
