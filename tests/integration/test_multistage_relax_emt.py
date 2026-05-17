"""multistage_relax with the ASE-friendly single-stage default."""

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize

from pyiron_workflow_assyst.physics.relaxation import multistage_relax


@pytest.mark.integration
def test_full_relax_lowers_energy(tmp_path):
    """An under-strained Cu cell should relax to lower energy under EMT."""
    engine = ASEEngine(
        EngineInput=CalcInputMinimize(cell_relaxation="full", max_iterations=20),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    cu = bulk("Cu", "fcc", a=3.4, cubic=True)
    initial_energy = EMT().get_potential_energy(cu)

    macro = multistage_relax(
        structure=cu,
        engine=engine,
        stages=[CalcInputMinimize(cell_relaxation="full", max_iterations=20)],
        stage_names=["full"],
    )
    macro.run()
    final = macro.outputs.final_structure.value
    final.calc = EMT()
    final_energy = final.get_potential_energy()
    assert final_energy < initial_energy, (
        f"relax did not lower energy: initial={initial_energy:.4f}, "
        f"final={final_energy:.4f}"
    )
