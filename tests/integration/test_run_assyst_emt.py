"""End-to-end ASSYST run on Cu+EMT — proves the engine-agnostic macro works
without VASP. Small enough to ship in CI."""

import pandas as pd
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import (
    ASEEngine,
    CalcInputMinimize,
    CalcInputStatic,
)

from pyiron_workflow_assyst.physics.assyst import run_assyst


@pytest.mark.integration
def test_run_assyst_emt_endtoend(tmp_path):
    relax_engine = ASEEngine(
        EngineInput=CalcInputMinimize(cell_relaxation="full", max_iterations=20),
        calculator=EMT(),
        working_directory=str(tmp_path / "relax"),
    )
    static_engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path / "static"),
    )
    pickle_path = tmp_path / "train.pkl"

    macro = run_assyst(
        structure=bulk("Cu", "fcc", a=3.6, cubic=True),
        relax_engine=relax_engine,
        static_engine=static_engine,
        base_name="cu0",
        relax_stages=[CalcInputMinimize(cell_relaxation="full", max_iterations=20)],
        relax_stage_names=["fullrelax"],
        n_rattle=1,
        n_triaxial=1,
        n_shear=1,
        rattle_displacement=0.05,
        rattle_cell_strain=0.02,
        triaxial_strain=0.05,
        shear_strain=0.05,
        seed=42,
        training_path=str(pickle_path),
    )
    macro.run()

    assert pickle_path.exists(), f"pickle not created at {pickle_path}"

    df = pd.read_pickle(pickle_path)
    # 1 base (last frame of single-stage relax) + 3 perms (1 rattle + 1 triax + 1 shear) = 4
    assert len(df) == 4, f"expected 4 rows, got {len(df)}\n{df}"
    assert set(df.columns) == {
        "name",
        "energy",
        "volume",
        "converged",
        "structure",
        "forces",
        "stress",
    }, f"unexpected columns: {df.columns.tolist()}"
    assert all(
        df["converged"]
    ), f"some frames did not converge:\n{df[['name','converged']]}"
    # Names should contain one base entry + three perms
    names = df["name"].tolist()
    assert any(
        "relaxstep" in n for n in names
    ), f"no base frame found in names: {names}"
    assert any("rattle" in n for n in names)
    assert any("triax" in n for n in names)
    assert any("shear" in n for n in names)
