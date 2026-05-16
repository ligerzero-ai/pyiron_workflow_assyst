"""Multi-stage structural relaxation macro.

Default chain mirrors the ASSYST methodology -- volume relax -> shape relax ->
ion relax (ISIF=7 / ISIF=5 / ISIF=2 under VaspEngine). Callers may override
``stages`` and ``stage_names`` for any other cascade.

The macro feeds each stage's ``final_structure`` into the next stage's
input automatically, and runs each stage inside its own subdirectory under
``engine.working_directory``.

Limitation
----------
pyiron_workflow does not support variable-length unrolling inside a macro body;
``len(stages)`` must be 1 or 3. Refactor when pwf gains dynamic fan-out.
``@as_macro_node`` also performs AST-level validation that rejects multiple
``return`` statements, so the branching logic lives outside the graph-creator
functions via the ``pwf.macro_node`` instance-factory API.
"""

from __future__ import annotations

from dataclasses import replace

import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import (
    CalcInputMinimize,
    Engine,
    EngineOutput,
    calculate,
)

_OUTPUT_LABELS = ("engine_outputs", "final_structure", "converged")


def _default_stages() -> list[CalcInputMinimize]:
    return [
        CalcInputMinimize(cell_relaxation="volume"),
        CalcInputMinimize(cell_relaxation="shape"),
        CalcInputMinimize(cell_relaxation="none"),
    ]


def _default_stage_names() -> list[str]:
    return ["ISIF7", "ISIF5", "ISIF2"]


@pwf.as_function_node("sub_engine")
def _replace_engine_input_and_subdir(
    engine: Engine, stage: CalcInputMinimize, subdir: str
) -> Engine:
    """Return a copy of ``engine`` with ``EngineInput=stage`` and
    ``working_directory`` extended by ``subdir``."""
    return replace(engine, EngineInput=stage).with_working_directory(subdir)


@pwf.as_function_node("engine_outputs")
def _pack1(a: EngineOutput) -> list:
    return [a]


@pwf.as_function_node("engine_outputs")
def _pack3(a: EngineOutput, b: EngineOutput, c: EngineOutput) -> list:
    return [a, b, c]


@pwf.as_function_node("final_structure", "converged")
def _final(last_engine_output: EngineOutput):
    return last_engine_output.final_structure, last_engine_output.converged


def _build_graph_1stage(
    wf,
    structure: Atoms,
    engine: Engine,
    stages: list,
    stage_names: list,
):
    wf.engine_0 = _replace_engine_input_and_subdir(
        engine=engine, stage=stages[0], subdir=stage_names[0]
    )
    wf.calc_0 = calculate(structure=structure, engine=wf.engine_0.outputs.sub_engine)
    wf.pack = _pack1(a=wf.calc_0.outputs.engine_output)
    wf.final = _final(last_engine_output=wf.calc_0.outputs.engine_output)
    return (
        wf.pack.outputs.engine_outputs,
        wf.final.outputs.final_structure,
        wf.final.outputs.converged,
    )


def _build_graph_3stage(
    wf,
    structure: Atoms,
    engine: Engine,
    stages: list,
    stage_names: list,
):
    wf.engine_0 = _replace_engine_input_and_subdir(
        engine=engine, stage=stages[0], subdir=stage_names[0]
    )
    wf.calc_0 = calculate(structure=structure, engine=wf.engine_0.outputs.sub_engine)

    wf.engine_1 = _replace_engine_input_and_subdir(
        engine=engine, stage=stages[1], subdir=stage_names[1]
    )
    wf.calc_1 = calculate(
        structure=wf.calc_0.outputs.engine_output.final_structure,
        engine=wf.engine_1.outputs.sub_engine,
    )

    wf.engine_2 = _replace_engine_input_and_subdir(
        engine=engine, stage=stages[2], subdir=stage_names[2]
    )
    wf.calc_2 = calculate(
        structure=wf.calc_1.outputs.engine_output.final_structure,
        engine=wf.engine_2.outputs.sub_engine,
    )

    wf.pack = _pack3(
        a=wf.calc_0.outputs.engine_output,
        b=wf.calc_1.outputs.engine_output,
        c=wf.calc_2.outputs.engine_output,
    )
    wf.final = _final(last_engine_output=wf.calc_2.outputs.engine_output)
    return (
        wf.pack.outputs.engine_outputs,
        wf.final.outputs.final_structure,
        wf.final.outputs.converged,
    )


def multistage_relax(
    structure: Atoms,
    engine: Engine,
    stages: list[CalcInputMinimize] | None = None,
    stage_names: list[str] | None = None,
):
    """Chain ``len(stages)`` ``calculate()`` calls; feed forward final_structure.

    Returns a :class:`pyiron_workflow.nodes.macro.Macro` instance ready to
    ``.run()``.

    Parameters
    ----------
    structure:
        Initial structure passed to the first stage.
    engine:
        Engine template; each stage gets a copy with its own ``EngineInput``
        and subdirectory.
    stages:
        Sequence of :class:`CalcInputMinimize` objects, one per stage.
        Defaults to the three-stage ASSYST chain (ISIF=7/5/2).
        Currently only ``len(stages) in {1, 3}`` is supported.
    stage_names:
        Directory labels for each stage under ``engine.working_directory``.
        Defaults to ``["ISIF7", "ISIF5", "ISIF2"]``.

    Returns
    -------
    macro : Macro
        Runnable macro with outputs ``engine_outputs``, ``final_structure``,
        and ``converged``.
    """
    if stages is None:
        stages = _default_stages()
    if stage_names is None:
        stage_names = _default_stage_names()
    if len(stages) != len(stage_names):
        raise ValueError(
            f"stages ({len(stages)}) and stage_names ({len(stage_names)}) "
            "must have the same length"
        )

    if len(stages) == 1:
        graph_creator = _build_graph_1stage
    elif len(stages) == 3:
        graph_creator = _build_graph_3stage
    else:
        raise NotImplementedError(
            f"multistage_relax currently supports len(stages) in {{1, 3}}; "
            f"got {len(stages)}. Add a graph-creator function for other lengths."
        )

    return pwf.macro_node(
        graph_creator,
        output_labels=_OUTPUT_LABELS,
        structure=structure,
        engine=engine,
        stages=stages,
        stage_names=stage_names,
    )
