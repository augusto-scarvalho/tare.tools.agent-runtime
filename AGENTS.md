# Agent instructions

This repository owns only tactical agent-loop execution.

- Keep Work, Workflow, Routing, Authority, Evidence and settlement in `tare.tools.os`.
- Keep providers, processes, sandboxes and mediated capability execution in `tare.tools.kernel`.
- Inject ports; never import private OS or Kernel modules and never auto-discover adapters.
- Do not import, copy or revive `tare.tools.harness`.
- Do not add a daemon, database, UI, registry, plugin framework or public CLI without a successor ADR.
- Write code, comments, prompts and documentation in English; agents still reply in the user's language.
  The Portuguese version for Brazilian readers lives only in `README.pt-BR.md`: update it with its English source.
- Update `CHANGELOG.md` with every material change.
- After changing `src`, refresh the SHA-256 entries in `CONTRACT_INVENTORY.json`
  from UTF-8 source bytes with CRLF normalized to LF. Package version,
  `CORE_CONTRACT_VERSION`, inventory version and their assertions move together.
- Run `python -m pytest -q` and `python -m compileall -q src` before commit.
