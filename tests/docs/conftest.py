"""Docs tests (TDD §9.1 "Docs" layer, CI job ``docs``): README code, paper tables, paper text.

Every test here carries the ``docs`` marker, so it runs only in the ``docs`` job, in a fresh venv with
``pip install .``.
"""

from __future__ import annotations

import importlib.util
import sys
from types import ModuleType

import pytest

from tests.conftest import ROOT


@pytest.fixture(scope="session")
def paper_tables() -> ModuleType:
    """``scripts/paper_tables.py`` as a module (registered, so its dataclasses resolve their annotations)."""
    name = "paper_tables"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "paper_tables.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
