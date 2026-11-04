from __future__ import annotations

import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("relative_path", "namespace", "matcher", "command"),
    [
        (
            ".claude/settings.json",
            "hooks",
            "Bash|Grep|Edit|Write",
            "tare-hook claude",
        ),
        (
            ".codex/hooks.json",
            "hooks",
            "Bash|Grep|apply_patch|Edit|Write",
            "tare-hook codex",
        ),
        (
            ".agents/hooks.json",
            "tare-specgraph-first",
            "run_command|grep_search|write_to_file|replace_file_content|multi_replace_file_content",
            "tare-hook agy --event pre",
        ),
    ],
)
def test_repository_hooks_are_thin_canonical_kernel_adapters(
    relative_path: str,
    namespace: str,
    matcher: str,
    command: str,
) -> None:
    payload = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))

    assert payload[namespace]["PreToolUse"] == [
        {
            "matcher": matcher,
            "hooks": [{"type": "command", "command": command, "timeout": 5}],
        }
    ]


@pytest.mark.parametrize(
    ("relative_path", "namespace", "matcher", "command"),
    [
        (
            ".claude/settings.json",
            "hooks",
            "Edit|Write",
            "tare-hook claude",
        ),
        (
            ".codex/hooks.json",
            "hooks",
            "apply_patch|Edit|Write",
            "tare-hook codex",
        ),
        (
            ".agents/hooks.json",
            "tare-specgraph-first",
            "write_to_file|replace_file_content|multi_replace_file_content",
            "tare-hook agy --event post",
        ),
    ],
)
def test_repository_post_edit_hooks_reuse_the_same_kernel_policy(
    relative_path: str,
    namespace: str,
    matcher: str,
    command: str,
) -> None:
    payload = json.loads((ROOT / relative_path).read_text(encoding="utf-8"))

    assert payload[namespace]["PostToolUse"] == [
        {
            "matcher": matcher,
            "hooks": [{"type": "command", "command": command, "timeout": 5}],
        }
    ]


def test_specgraph_configuration_separates_search_from_governance() -> None:
    config = (ROOT / "specgraph.yaml").read_text(encoding="utf-8")

    assert 'project_name: "tare.tools.agent-runtime"' in config
    assert "governed_paths:\n  - src\n" in config
    assert "test_paths:\n  - tests\n" in config
    assert "spec_paths: []\n" in config
    assert "schema_paths: []\n" in config
    assert "api_paths: []\n" in config
    assert "adr_paths: []\n" in config
    assert "strict_orphan_check: false\n" in config
    for path in (
        "README.md",
        "CHANGELOG.md",
        "AGENTS.md",
        "pyproject.toml",
        "CONTRACT_INVENTORY.json",
        "BOOTSTRAP_PROVENANCE.json",
        "specgraph.yaml",
    ):
        assert f"  - {path}\n" in config
