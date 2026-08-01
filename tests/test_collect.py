import pathlib

import numpy as np
import pandas as pd
import pyiron_workflow as pwf
import pytest
from ase import Atoms as AseAtoms
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Incar
from vaspparser.vasp.output import parse_vasp_output as external_parse_vasp_output

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
    assert run.outputs.structures[0].get_cell()[0][0] == pytest.approx(
        3.0
    ), "picked the crashed earliest row"


def test_collect_threshold_selects_multiple_images():
    df = pd.DataFrame([_row([4.0, 3.9, 3.0], 3, [-1.0, -1.0005, -3.0], [True] * 3, 1)])
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


# --- collect_structures: the DEFAULT parser's dict shape -------------------
#
# vasp_job's DEFAULT parser is the external vaspparser.vasp.output.parse_vasp
# _output, which returns a dict, not the bundled parser's DataFrame. Every
# test above hand-builds a DataFrame, which is exactly why this shape
# mismatch (AttributeError: 'dict' object has no attribute 'iloc') survived
# 37 passing tests: nothing ever fed collect_structures what vasp_job's
# default parser actually produces.

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "vasp_isif2_fe2"


def _dict_output(cells, energies, scf_lens, numbers=(26,)):
    """Hand-built stand-in for the external parser's dict shape, used only to
    exercise the SCF-convergence-from-NELM derivation and threshold/selection
    logic directly (independent of the real fixture below, which happens to
    have every ionic step converged, so it alone couldn't prove the drop
    path)."""
    n_steps = len(energies)
    n_atoms = len(numbers)
    positions = np.zeros((n_steps, n_atoms, 3))
    cells_arr = np.array(cells, dtype=float)
    return {
        "generic": {
            "positions": positions,
            "cells": cells_arr,
            "energy_tot": np.array(energies, dtype=float),
            "dft": {"scf_energy_free": [[0.0] * n for n in scf_lens]},
        },
        "structure": {
            "numbers": np.array(numbers),
            "positions": positions[-1],
            "cell": cells_arr[-1],
            "pbc": np.array([True, True, True]),
        },
    }


def test_collect_structures_accepts_real_external_parser_dict():
    """Uses a REAL captured dict, parsed from a genuine 2-atom Fe,
    2-ionic-step ISIF2 relaxation fixture (tests/fixtures/vasp_isif2_fe2/,
    committed OUTCAR/vasprun.xml/INCAR/POSCAR/CONTCAR, POTCAR omitted for
    licensing -- same convention as pyiron_workflow_vasp's
    vasp_isif7_fe2/), rather than a hand-written stand-in, since a
    hand-written dict is exactly what let this class of bug through before
    (see pyiron_workflow_vasp commit 3e8a89c)."""
    real_output = external_parse_vasp_output(working_directory=str(FIXTURE_DIR))
    assert isinstance(real_output, dict), (
        "fixture-capture assumption broke upstream: vaspparser no longer "
        "returns a dict"
    )
    assert set(real_output["structure"].keys()) >= {
        "numbers",
        "positions",
        "cell",
        "pbc",
    }, "fixture-capture assumption broke upstream: structure dict shape changed"
    assert set(real_output["generic"].keys()) >= {
        "positions",
        "cells",
        "energy_tot",
    }, "fixture-capture assumption broke upstream: generic dict shape changed"

    # NELM is read off this fixture's own INCAR (12), exactly the way
    # run_ASSYST_on_structure derives it (incar2.get("NELM")) -- not guessed.
    incar = Incar.from_file(FIXTURE_DIR / "INCAR")
    run = pwf.node(collect_structures).run(
        vasp_output=real_output, job_name="ISIF2", nelm=incar.get("NELM")
    )

    # Default threshold (-1) keeps only the final image -- exactly one
    # structure, from the LAST ionic step (index 1 of 2).
    assert len(run.outputs.structures) == 1
    assert run.outputs.names == ["ISIF2_accur_relaxstep1"]

    result = run.outputs.structures[0]
    assert isinstance(result, AseAtoms)
    np.testing.assert_allclose(
        np.array(result.get_cell()),
        real_output["generic"]["cells"][-1],
        err_msg="returned structure's cell must match the LAST ionic step's cell",
    )
    np.testing.assert_allclose(
        result.get_positions(),
        real_output["generic"]["positions"][-1],
        err_msg="returned structure's positions must match the LAST ionic step's positions",
    )
    assert run.outputs.energies == pytest.approx(
        [real_output["generic"]["energy_tot"][-1]]
    )
    # Sanity-check against the exact values quoted for this fixture.
    assert run.outputs.energies == pytest.approx([-13.19624727], abs=1e-5)


def test_collect_structures_dict_path_drops_unconverged_using_nelm():
    """SCF convergence isn't carried explicitly in the dict shape -- it must
    be derived from len(scf_energy_free[i]) < nelm (VASP runs the full NELM
    electronic steps only when it FAILS to converge within NELM). Step 0 has
    5 electronic steps (< nelm=12 -> converged); step 1 has 12 (not < 12 ->
    NOT converged, VASP hit the cap). threshold=0.001 with a huge energy jump
    selects both indices (0 and 1), so only the drop-on-non-convergence logic
    decides what survives -- this fails if convergence is derived wrong (e.g.
    <= instead of <, or the wrong step is kept)."""
    cell = np.eye(3) * 3.0
    out = _dict_output(cells=[cell, cell], energies=[-1.0, -50.0], scf_lens=[5, 12])
    run = pwf.node(collect_structures).run(
        vasp_output=out,
        job_name="JOB",
        image_selection_eVatom_threshold=0.001,
        nelm=12,
    )
    assert run.outputs.names == ["JOB_accur_relaxstep0"]
    assert len(run.outputs.structures) == 1
    assert run.outputs.energies == [-1.0]


def test_collect_structures_dict_path_warns_and_assumes_default_nelm_when_missing():
    """Dropping unconverged images is a real filter that protects training
    data; silently treating every dict-path image as converged with no
    signal when nelm is omitted would be an invisible degradation. Instead,
    collect_structures must warn AND still apply VASP's own built-in default
    (60) rather than warning and then ignoring it: scf_lens=[65] > 60 must
    still be correctly rejected as non-converged."""
    cell = np.eye(3) * 3.0
    out = _dict_output(cells=[cell], energies=[-1.0], scf_lens=[65])
    with pytest.warns(UserWarning, match="nelm not provided"), pytest.raises(
        NoConvergedImagesError
    ):
        pwf.node(collect_structures).run(vasp_output=out, job_name="JOB")


def test_collect_structures_rejects_unsupported_type():
    """Anything that is neither a dict nor a DataFrame must fail loudly with
    a clear TypeError naming both supported shapes, not silently return None
    or raise some unrelated AttributeError deep inside pandas/numpy."""
    with pytest.raises(TypeError, match="unsupported vasp_output type"):
        pwf.node(collect_structures).run(
            vasp_output=["not", "a", "supported", "shape"], job_name="ISIF2"
        )
