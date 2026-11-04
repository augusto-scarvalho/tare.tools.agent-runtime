from __future__ import annotations

import ast
import hashlib
import json
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
FORBIDDEN_IMPORT_ROOTS = {
    "compute_plane",
    "control_plane",
    "data_plane",
    "experience_plane",
    "kernel",
    "relay",
    "scripts",
    "tare_tools_harness",
    "tare_tools_kernel",
    "tare_tools_os",
}
FORBIDDEN_SURFACES = {"daemon", "database", "plugins", "registry", "ui"}


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_core_has_no_private_ecosystem_or_legacy_imports() -> None:
    imported = set().union(*(imported_roots(path) for path in SRC.rglob("*.py")))
    assert imported.isdisjoint(FORBIDDEN_IMPORT_ROOTS)


def test_bootstrap_has_no_forbidden_platform_surfaces() -> None:
    top_level_names = {path.name.lower() for path in ROOT.iterdir()}
    assert top_level_names.isdisjoint(FORBIDDEN_SURFACES)


def test_no_executable_entrypoint_is_declared() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project.scripts]" not in pyproject
    assert "if __name__ ==" not in "\n".join(
        path.read_text(encoding="utf-8") for path in SRC.rglob("*.py")
    )


def test_bootstrap_provenance_is_clean_history_and_legacy_free() -> None:
    receipt = json.loads((ROOT / "BOOTSTRAP_PROVENANCE.json").read_text(encoding="utf-8"))
    assert receipt["lineage"] == {
        "git_history": "new",
        "source_origin": "clean implementation from ADR-057 F0 contracts",
        "tare.tools.harness_source_read": False,
        "tare.tools.harness_source_copied": False,
        "tare.tools.harness_imported": False,
    }
    assert receipt["production_wiring"] == "HELD_PENDING_INTEGRATION_GATES"


def test_contract_inventory_matches_exact_source_bytes() -> None:
    inventory = json.loads((ROOT / "CONTRACT_INVENTORY.json").read_text(encoding="utf-8"))
    assert inventory["core_contract_version"] == "0.14.7"
    assert inventory["hash_mode"] == "canonical_lf_utf8"
    expected_sources = {
        path.relative_to(ROOT).as_posix()
        for path in SRC.rglob("*.py")
    }
    assert set(inventory["sources"]) == expected_sources
    for relative_path, expected in inventory["sources"].items():
        canonical_bytes = (ROOT / relative_path).read_bytes().replace(b"\r\n", b"\n")
        actual = hashlib.sha256(canonical_bytes).hexdigest()
        assert actual == expected


def test_package_version_matches_core_contract() -> None:
    from tare_tools_agent_runtime import CORE_CONTRACT_VERSION

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    inventory = json.loads((ROOT / "CONTRACT_INVENTORY.json").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == inventory["core_contract_version"]
    assert CORE_CONTRACT_VERSION == inventory["core_contract_version"]


def test_core_has_no_strands_dependency() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "strands" not in pyproject
