"""Tiny function-node helpers for fanning out engines / concatenating
channel-bound lists inside a pyiron_workflow macro graph.

These are private because they exist purely to bridge channel-bound values
into ordinary Python lists at graph-execution time — plain list
comprehensions over channels don't dispatch through ``__call__`` correctly
inside macro bodies.
"""

from __future__ import annotations

import pyiron_workflow as pwf

from pyiron_workflow_atomistics.engine import Engine


@pwf.as_function_node("subengines")
def _build_subengines(engine: Engine, names: list[str]) -> list[Engine]:
    """One sub-engine per name, composed via ``engine.with_working_directory``.

    The parent ``engine`` is not mutated (Engine.with_working_directory is
    contractually pure).
    """
    subengines = [engine.with_working_directory(n) for n in names]
    return subengines


@pwf.as_function_node("concatenated")
def _concat(a: list, b: list) -> list:
    """List concatenation. Wraps the channel boundary."""
    concatenated = list(a) + list(b)
    return concatenated
