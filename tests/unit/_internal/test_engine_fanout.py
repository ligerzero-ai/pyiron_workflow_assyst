"""_build_subengines composes per-name working_directories purely; _concat
concatenates two lists."""

import os

from ase.calculators.emt import EMT
from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic

from pyiron_workflow_assyst._internal.engine_fanout import (
    _build_subengines,
    _concat,
)


def test_build_subengines_one_per_name(tmp_path):
    parent = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    subs = _build_subengines.node_function(engine=parent, names=["a", "b", "c"])
    assert [s.working_directory for s in subs] == [
        os.path.join(str(tmp_path), "a"),
        os.path.join(str(tmp_path), "b"),
        os.path.join(str(tmp_path), "c"),
    ]
    # Parent untouched.
    assert parent.working_directory == str(tmp_path)


def test_build_subengines_empty_names(tmp_path):
    parent = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    subs = _build_subengines.node_function(engine=parent, names=[])
    assert subs == []


def test_concat_two_lists():
    assert _concat.node_function(a=[1, 2], b=[3, 4]) == [1, 2, 3, 4]


def test_concat_empty_left():
    assert _concat.node_function(a=[], b=["x"]) == ["x"]


def test_concat_empty_right():
    assert _concat.node_function(a=["x"], b=[]) == ["x"]
