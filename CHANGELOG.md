# Changelog

Notable changes to `tare.tools.agent-runtime` are recorded here, newest first.

## Unreleased

- CI runs on GitHub-hosted Linux and Windows runners (`Tests (linux)`, `Tests (windows)`) with the README's install and test commands and a pip cache. The repository variables `CI_RUNNER_LINUX` and `CI_RUNNER_WINDOWS` move it to your own runners, and pull requests from forks then skip it.
- First public release. The repository starts from a single commit; earlier history is not published. It contains the
  bounded tactical agent-loop runtime that tare.tools.os hosts through the kernel's provider and process ports, and
  its test suite.
