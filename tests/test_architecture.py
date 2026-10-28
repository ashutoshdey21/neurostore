"""Keeps the layers separate: domain imports nothing below it, infrastructure never
imports the Store, and only the infrastructure layer touches sqlite3."""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "src" / "neurostore"


def imported_modules(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def imports_any(path: Path, prefixes: tuple[str, ...]) -> list[str]:
    return sorted(m for m in imported_modules(path) if m.startswith(prefixes))


def files_in(layer: str) -> list[Path]:
    return sorted((PACKAGE / layer).glob("*.py"))


@pytest.mark.parametrize("path", files_in("domain"), ids=lambda p: p.name)
def test_domain_has_no_storage_dependencies(path: Path) -> None:
    assert imports_any(path, ("sqlite3", "neurostore.infrastructure", "neurostore.store")) == []


@pytest.mark.parametrize("path", files_in("infrastructure"), ids=lambda p: p.name)
def test_infrastructure_does_not_depend_on_store(path: Path) -> None:
    assert imports_any(path, ("neurostore.store",)) == []


def test_store_leaves_sql_and_file_formats_to_infrastructure() -> None:
    assert imports_any(PACKAGE / "store.py", ("sqlite3", "io", "hashlib", "json")) == []
