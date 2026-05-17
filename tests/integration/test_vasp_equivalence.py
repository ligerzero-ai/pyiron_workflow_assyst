"""Compare the legacy VASP-only ``run_ASSYST_on_structure`` against the new
engine-agnostic ``run_assyst`` driven with VaspEngine.

Skipped unless ``VASP_TEST=1`` is set and ``vasp_std`` is on ``PATH``. This
test is the ground-truth equivalence check — content-level, not
schema-level (the two pipelines produce different DataFrame layouts; we
compare exploded frames per name).

To run:

    VASP_TEST=1 FE_POTCAR_PATH=/path/to/Fe/POTCAR pixi run python -m pytest \\
        tests/integration/test_vasp_equivalence.py -v
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk

VASP_AVAILABLE = (
    bool(os.environ.get("VASP_TEST")) and shutil.which("vasp_std") is not None
)
pytestmark = pytest.mark.skipif(
    not VASP_AVAILABLE,
    reason="set VASP_TEST=1 and ensure vasp_std is on PATH to run this test",
)


def _load_legacy_module():
    """Import the frozen legacy workflow.py snapshot.

    The snapshot lives at ``tests/_legacy_assyst/workflow.py`` and was
    copied from the pre-rewrite ``pyiron_workflow_assyst/workflow.py``.
    Loading requires the legacy package's runtime dependencies to be
    importable in the active env: ``pyiron_workflow_vasp.vasp`` and a
    handful of pyiron_workflow helpers. We import them by absolute path
    via importlib rather than as a relative sub-package.
    """
    here = pathlib.Path(__file__).resolve()
    snap = here.parents[1] / "_legacy_assyst" / "workflow.py"
    spec = importlib.util.spec_from_file_location("_legacy_assyst_workflow", snap)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_legacy_assyst_workflow"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.vasp
def test_assyst_vasp_equivalence(tmp_path):
    """Run legacy and new pipelines on identical Fe BCC + INCAR + RNG seed."""
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.vasp.inputs import Incar

    from pyiron_workflow_atomistics.engine import CalcInputMinimize, CalcInputStatic
    from pyiron_workflow_vasp.engine import VaspEngine

    from pyiron_workflow_assyst.physics.assyst import run_assyst

    legacy = _load_legacy_module()

    potcar_path = os.environ.get("FE_POTCAR_PATH")
    if not potcar_path:
        pytest.skip("set FE_POTCAR_PATH to point at an Fe POTCAR")

    fe = bulk("Fe", "bcc", a=2.86, cubic=True)
    incar_dict = {
        "ENCUT": 300,
        "ISIF": 7,
        "NSW": 5,
        "EDIFF": 1e-4,
        "EDIFFG": -0.05,
        "PREC": "Low",
        "ISMEAR": 1,
        "SIGMA": 0.2,
        "ALGO": "Fast",
    }
    seed = 123

    # --- Legacy run ---
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy_pickle = legacy_dir / "df_ASSYST_jobs.pkl"
    legacy_pmg = AseAtomsAdaptor.get_structure(fe)
    np.random.seed(seed)
    legacy_macro = legacy.run_ASSYST_on_structure(
        legacy_pmg,
        Incar.from_dict(incar_dict),
        potcar_paths=[potcar_path],
        ionic_steps=5,
        n_stretch_permutations=1,
        n_rattle_permutations=1,
        shear_strain=0.05,
        triaxial_strain=0.05,
        rattle_displacement=0.05,
        rattle_strain=0.02,
        job_name=str(legacy_dir / "struct"),
        train_df_filename=str(legacy_pickle),
    )
    legacy_macro.run()

    # --- New run ---
    new_dir = tmp_path / "new"
    new_dir.mkdir()
    new_pickle = new_dir / "df_ASSYST_jobs.pkl"
    relax_engine = VaspEngine(
        EngineInput=CalcInputMinimize(cell_relaxation="volume", max_iterations=5),
        working_directory=str(new_dir / "struct"),
        potcar_config_file=potcar_path,
        encut=300,
        kpoints_density=0.30,
        command="vasp_std",
    )
    static_engine = VaspEngine(
        EngineInput=CalcInputStatic(),
        working_directory=str(new_dir / "struct"),
        potcar_config_file=potcar_path,
        encut=300,
        kpoints_density=0.25,
        ediff=1e-5,
        lreal=False,
        command="vasp_std",
    )
    new_macro = run_assyst(
        structure=fe,
        relax_engine=relax_engine,
        static_engine=static_engine,
        base_name="struct",
        n_rattle=1,
        n_triaxial=1,
        n_shear=1,
        rattle_displacement=0.05,
        rattle_cell_strain=0.02,
        triaxial_strain=0.05,
        shear_strain=0.05,
        seed=seed,
        training_path=str(new_pickle),
    )
    new_macro.run()

    # --- Compare ---
    assert legacy_pickle.exists()
    assert new_pickle.exists()

    legacy_df = pd.read_pickle(legacy_pickle)
    new_df = pd.read_pickle(new_pickle)

    # Legacy: one row per VASP job, list-valued ``structures`` / ``energy`` cells.
    # New: one row per frame, scalar cells. Explode the legacy first.
    legacy_exploded = legacy_df.explode(["structures", "energy"]).reset_index(drop=True)

    assert len(legacy_exploded) == len(new_df), (
        f"frame count mismatch: legacy_exploded={len(legacy_exploded)} "
        f"new={len(new_df)}"
    )

    # Compare per-name energy drift (≤ 1e-3 eV/atom is the spec acceptance bound).
    legacy_energies: dict[str, float] = {}
    legacy_n_atoms: dict[str, int] = {}
    for _, row in legacy_exploded.iterrows():
        name = row.get("name") or row.get("job_name")
        if name is None:
            pytest.skip(
                "legacy DataFrame has no 'name' / 'job_name' column to align on"
            )
        legacy_energies[name] = float(row["energy"])
        legacy_n_atoms[name] = len(row["structures"])

    new_energies = dict(zip(new_df["name"], new_df["energy"]))

    assert set(legacy_energies) == set(new_energies), (
        f"frame-name sets differ:\n"
        f"  legacy: {sorted(legacy_energies)}\n"
        f"  new:    {sorted(new_energies)}"
    )

    for name in legacy_energies:
        n_atoms = legacy_n_atoms[name]
        diff_per_atom = abs(legacy_energies[name] - new_energies[name]) / n_atoms
        assert diff_per_atom < 1e-3, (
            f"{name}: |Δenergy|/atom = {diff_per_atom:.6f} eV/atom "
            f"(legacy={legacy_energies[name]:.6f}, new={new_energies[name]:.6f})"
        )
