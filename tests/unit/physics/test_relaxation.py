"""multistage_relax: defaults match ASSYST ISIF=7/5/2; runs end-to-end on ASE."""

from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize

from pyiron_workflow_assyst.physics.relaxation import (
    _default_stage_names,
    _default_stages,
    multistage_relax,
)


def test_default_stages_are_volume_shape_none():
    stages = _default_stages()
    names = _default_stage_names()
    assert [s.cell_relaxation for s in stages] == ["volume", "shape", "none"]
    assert names == ["ISIF7", "ISIF5", "ISIF2"]
    assert all(isinstance(s, CalcInputMinimize) for s in stages)


def test_runs_one_stage_with_ase_full(tmp_path):
    """ASE only supports 'full' and 'none'; exercise the len(stages)==1 branch."""
    engine = ASEEngine(
        EngineInput=CalcInputMinimize(cell_relaxation="full", max_iterations=3),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    macro = multistage_relax(
        structure=bulk("Cu", "fcc", a=3.6, cubic=True),
        engine=engine,
        stages=[CalcInputMinimize(cell_relaxation="full", max_iterations=3)],
        stage_names=["fullrelax"],
    )
    macro.run()
    # Output channel names from the macro's @as_macro_node decorator:
    #   engine_outputs, final_structure, converged
    out_list = macro.outputs.engine_outputs.value
    assert len(out_list) == 1
    assert macro.outputs.final_structure.value is not None
    assert macro.outputs.converged.value in (True, False)
