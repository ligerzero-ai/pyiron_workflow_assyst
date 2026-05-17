"""Shared pytest fixtures for downstream users of pyiron_workflow_assyst.

Importable directly into a consumer's ``conftest.py``::

    from pyiron_workflow_assyst.testing.fixtures import cu_fcc, emt_engine

Both fixtures are session-cheap — ``cu_fcc`` builds a 4-atom FCC Cu cell;
``emt_engine`` returns an :class:`ASEEngine` wired with the EMT calculator
and ``working_directory`` set to pytest's ``tmp_path``.
"""

from __future__ import annotations

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic


@pytest.fixture
def cu_fcc():
    """Tiny 4-atom FCC Cu cell at the EMT equilibrium lattice parameter."""
    return bulk("Cu", "fcc", a=3.6, cubic=True)


@pytest.fixture
def emt_engine(tmp_path):
    """Static-mode ASE engine with EMT, working_directory under ``tmp_path``."""
    return ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
