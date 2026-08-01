import numpy as np
import pandas as pd
import pyiron_workflow as pwf
import pytest
from pymatgen.core import Lattice, Structure

from pyiron_workflow_assyst.workflow import NoConvergedImagesError, collect_structures


def _row(lattice_params, n_images, energies, scf, start_time):
    structures = np.array(
        [
            Structure(Lattice.cubic(a), ["Fe"], [[0, 0, 0]]).to_json()
            for a in lattice_params
        ]
    )
    assert len(structures) == n_images
    return {
        "structures": structures,
        "energy": np.array(energies),
        "scf_convergence": np.array(scf),
        "calc_start_time": start_time,
    }


def test_collect_takes_last_image_by_default():
    df = pd.DataFrame([_row([4.0, 3.5, 3.0], 3, [-1.0, -2.0, -3.0], [True] * 3, 1)])
    run = pwf.node(collect_structures).run(vasp_output=df, job_name="ISIF2")
    assert len(run.outputs.structures) == 1
    assert run.outputs.structures[0].get_cell()[0][0] == pytest.approx(3.0)
    assert run.outputs.names == ["ISIF2_accur_relaxstep2"]


def test_collect_drops_unconverged_images():
    df = pd.DataFrame([_row([4.0, 3.0], 2, [-1.0, -2.0], [True, False], 1)])
    run = pwf.node(collect_structures).run(
        vasp_output=df, job_name="ISIF2", image_selection_eVatom_threshold=0.001
    )
    assert all(name.endswith("relaxstep0") for name in run.outputs.names)
    assert len(run.outputs.structures) == 1


def test_collect_uses_the_most_recent_row_not_the_first():
    """With a custodian error archive present the parser returns several rows,
    ascending by start time. The crashed earliest run must NOT be chosen."""
    crashed = _row([9.9], 1, [-0.1], [True], 1)
    good = _row([3.0], 1, [-3.0], [True], 2)
    df = pd.DataFrame([crashed, good])
    run = pwf.node(collect_structures).run(vasp_output=df, job_name="ISIF2")
    assert run.outputs.structures[0].get_cell()[0][0] == pytest.approx(3.0), (
        "picked the crashed earliest row"
    )


def test_collect_threshold_selects_multiple_images():
    df = pd.DataFrame(
        [_row([4.0, 3.9, 3.0], 3, [-1.0, -1.0005, -3.0], [True] * 3, 1)]
    )
    run = pwf.node(collect_structures).run(
        vasp_output=df, job_name="ISIF2", image_selection_eVatom_threshold=0.1
    )
    # Threshold selection compares each candidate to the last SELECTED value
    # (0.0005 eV/atom from index 0 to 1 -> index 1 skipped; 2.0 eV/atom from
    # index 0 to 2 -> index 2 selected), so indices 0 and 2 are the only ones
    # that can come out. `len(structures) == len(names)` alone cannot fail --
    # both are built in the same loop, appended together -- so it is replaced
    # with assertions on the actual selected indices/energies, which can.
    assert run.outputs.names == ["ISIF2_accur_relaxstep0", "ISIF2_accur_relaxstep2"]
    assert len(run.outputs.structures) == 2
    assert run.outputs.energies == [-1.0, -3.0]


def test_collect_raises_a_clear_error_when_nothing_converges():
    """If every candidate image fails the SCF-convergence filter, collect_structures must
    raise a clear, named error rather than silently returning empty lists.
    An empty structures/names pair would otherwise feed a length-0 zip-loop
    in run_ASSYST_on_structure, which dies deep inside flowrep with an opaque
    `IndexError: list index out of range` -- the n=0 sibling of the n=1
    ForEach bug worked around elsewhere in this package (see the module
    docstring of pyiron_workflow_assyst.workflow)."""
    df = pd.DataFrame([_row([4.0], 1, [-1.0], [False], 1)])
    with pytest.raises(NoConvergedImagesError, match="ISIF2"):
        pwf.node(collect_structures).run(vasp_output=df, job_name="ISIF2")
