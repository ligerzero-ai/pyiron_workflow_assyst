import os

import numpy as np
import pandas as pd
import pyiron_workflow as pwf
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Incar, Poscar

from pyiron_workflow_assyst.workflow import (
    NoPermutationsGeneratedError,
    run_ASSYST_on_structure,
)


def _fake_parser(directory):
    """Stand in for parse_vasp_directory with a realistically-shaped frame."""
    blobs = np.array(
        [Structure(Lattice.cubic(a), ["Fe"], [[0, 0, 0]]).to_json() for a in (3.0, 2.9)]
    )
    return pd.DataFrame(
        [
            {
                "structures": blobs,
                "energy": np.array([-8.0, -8.2]),
                "scf_convergence": np.array([True, True]),
                "calc_start_time": 1,
                "job_name": os.path.basename(str(directory)),
            }
        ]
    )


# _fake_parser always returns the SAME two-image frame regardless of what was
# actually written to/run in a given directory, so the last of those two
# images (a=2.9) is exactly what construct_sequential_vasp_input must feed
# forward into the NEXT relaxation stage's POSCAR -- for every stage, not
# just the first. Named here so the feed-forward assertions below don't
# silently duplicate this magic number.
_FED_FORWARD_LATTICE_A = 2.9
_INITIAL_LATTICE_A = 2.83

# collect_structures("ISIF2", ...) with the default threshold (-1) keeps only
# the LAST of _fake_parser's two images -> index 1.
_BASE_IMAGE_NAME = "ISIF2_accur_relaxstep1"


def _run_graph(tmp_path, **overrides):
    """Run the graph against a real (scratch) directory tree and return both
    the ``Run`` and the absolute job directory the artefacts were written
    under, so callers can read back INCAR/POSCAR files afterwards."""
    job_dir = str((tmp_path / "struct_0").resolve())
    # NSW deliberately DIFFERS from ionic_steps below (mirrors production,
    # where the raw INCAR carried NSW=500 while ionic_steps was 100/200).
    # If they were equal, `_assert_relax_incar`'s `expected_nsw` check would
    # pass under both the current uniform-`ionic_steps` semantics AND the
    # old ISIF7-only semantics this port deliberately replaced -- i.e. it
    # would not actually be testing which code path ran. ISIF is included
    # too, so the statics-INCAR assertions exercise the code that strips it
    # rather than a fixture that never had it to begin with.
    incar = Incar.from_dict({"ENCUT": 300, "ISPIN": 1, "NSW": 500, "ISIF": 7})

    kwargs = dict(
        structure=Structure(Lattice.cubic(_INITIAL_LATTICE_A), ["Fe"], [[0, 0, 0]]),
        incar=incar,
        potcar_paths=None,
        job_name=job_dir,
        vasp_command=(
            "echo 'reached required accuracy - stopping structural energy "
            "minimisation' > vasp.log"
        ),
        ionic_steps=5,
        n_stretch_permutations=1,
        n_rattle_permutations=1,
        remove_calc_dirs=False,
        compress_dirs=False,
        train_df_filename=str(tmp_path / "df_ASSYST_jobs.pkl"),
        seed=42,
        vasp_parser_function=_fake_parser,
    )
    kwargs.update(overrides)
    node = pwf.node(run_ASSYST_on_structure)
    run = node.run(**kwargs)
    return run, job_dir


def _read_incar(directory):
    return Incar.from_file(os.path.join(directory, "INCAR"))


def _read_poscar_lattice_a(directory):
    return Poscar.from_file(os.path.join(directory, "POSCAR")).structure.lattice.a


def _assert_relax_incar(directory, expected_isif, expected_nsw):
    """A relaxation-stage (ISIF7/ISIF5/ISIF2) directory's INCAR must carry
    the ISIF value for that stage and the requested ionic_steps as NSW."""
    incar = _read_incar(directory)
    assert incar["ISIF"] == expected_isif, f"{directory}: wrong ISIF"
    assert incar["NSW"] == expected_nsw, f"{directory}: wrong NSW (ionic_steps)"


def _assert_static_incar(directory):
    """An accurate-statics directory's INCAR must carry the full override
    dict from run_ASSYST_on_structure (KSPACING/EDIFF/EDIFFG/LREAL/NSW=0)
    and must NOT carry an ISIF tag (statics do not relax anything)."""
    incar = _read_incar(directory)
    assert incar["KSPACING"] == pytest.approx(0.25), f"{directory}: wrong KSPACING"
    assert incar["EDIFF"] == pytest.approx(1e-5), f"{directory}: wrong EDIFF"
    assert incar["EDIFFG"] == pytest.approx(1e-4), f"{directory}: wrong EDIFFG"
    assert incar["LREAL"] is False, f"{directory}: wrong LREAL"
    assert incar["NSW"] == 0, f"{directory}: static job must not relax ions"
    assert "ISIF" not in incar, f"{directory}: static job must not carry ISIF"


def _assert_base_static_structure(directory):
    """The base statics job's POSCAR must be the fed-forward RELAXED
    geometry (ISIF2's parsed output, a=2.9 under `_fake_parser`) -- not the
    workflow's raw unrelaxed input (a=2.83). Getting this wrong means the
    energy/force labels in `train_df` would describe the input cell while
    being recorded as belonging to the relaxed one -- silently corrupting
    the training data with no crash to flag it.
    """
    a = _read_poscar_lattice_a(directory)
    assert a == pytest.approx(_FED_FORWARD_LATTICE_A), (
        f"{directory}: base statics structure must be the fed-forward "
        f"relaxed lattice (a={_FED_FORWARD_LATTICE_A}), got a={a} -- "
        f"statics may be running on the unrelaxed input instead"
    )


def _assert_perm_static_structure(directory):
    """A permutation statics job's POSCAR must be a genuine deformation of
    the base image: neither the workflow's raw unrelaxed input (a=2.83,
    which is what it would be if the permutation loop fed `structure`
    instead of its own `perm_structure`) nor an exact copy of the base
    image's own lattice (which would mean no deformation was applied at
    all). Checking only "differs from the raw input" would miss a bug that
    swaps in the base structure unperturbed; checking only "differs from
    base" would miss a bug that swaps in the raw input (a=2.83 != a=2.9
    happens to differ from the base too) -- both checks are needed to pin
    down which structure is actually feeding these jobs.
    """
    a = _read_poscar_lattice_a(directory)
    assert a != pytest.approx(_INITIAL_LATTICE_A), (
        f"{directory}: permutation statics structure equals the workflow's "
        f"raw unrelaxed input (a={_INITIAL_LATTICE_A}) -- permutations may "
        f"not be feeding their own deformed structure"
    )
    assert a != pytest.approx(_FED_FORWARD_LATTICE_A), (
        f"{directory}: permutation statics structure is an exact copy of "
        f"the base image (a={_FED_FORWARD_LATTICE_A}) -- no deformation "
        f"appears to have been applied"
    )


def _assert_isif_chain(job_dir, ionic_steps):
    """INCAR content and structure feed-forward for the ISIF7 -> 5 -> 2 chain.

    ISIF7's POSCAR must be the workflow's own initial structure (a=2.83).
    ISIF5's POSCAR must be built from ISIF7's *parsed output*, not from the
    original structure again -- with `_fake_parser` that parsed output is
    fixed at a=2.9 regardless of what ISIF7's own POSCAR/run actually held,
    so this also catches construct_sequential_vasp_input being swapped for a
    fresh generate_vasp_input(structure, ...) call (which would silently
    keep feeding the ORIGINAL structure into every stage instead). ISIF2's
    POSCAR must equal ISIF5's for the identical reason, one stage on.
    """
    isif7_dir = os.path.join(job_dir, "ISIF7")
    isif5_dir = os.path.join(job_dir, "ISIF5")
    isif2_dir = os.path.join(job_dir, "ISIF2")

    _assert_relax_incar(isif7_dir, expected_isif=7, expected_nsw=ionic_steps)
    _assert_relax_incar(isif5_dir, expected_isif=5, expected_nsw=ionic_steps)
    _assert_relax_incar(isif2_dir, expected_isif=2, expected_nsw=ionic_steps)

    a7 = _read_poscar_lattice_a(isif7_dir)
    a5 = _read_poscar_lattice_a(isif5_dir)
    a2 = _read_poscar_lattice_a(isif2_dir)

    assert a7 == pytest.approx(_INITIAL_LATTICE_A), (
        "ISIF7 POSCAR must be the workflow's input structure"
    )
    assert a5 == pytest.approx(_FED_FORWARD_LATTICE_A), (
        "ISIF5 POSCAR must be built from ISIF7's parsed output, not the "
        "original structure (feed-forward broken)"
    )
    assert a2 == pytest.approx(a5), (
        "ISIF2 POSCAR must be built from ISIF5's parsed output "
        "(feed-forward broken)"
    )


@pytest.mark.slow
def test_assyst_graph_runs_end_to_end_default_threshold(tmp_path):
    """Exercises the single-image PRODUCTION path, AND the actual physics.

    ``image_selection_eVatom_threshold`` is intentionally NOT passed here, so
    it takes the workflow's own default of -1 ("keep only the final
    relaxation image"). Every real ASSYST campaign runs with this default, so
    ``collect_structures`` routinely returns exactly ONE base structure, and
    both `for`-loops below iterate a length-1 collection.

    This is precisely the configuration that trips ``pyiron_workflow==0.19.0``'s
    ``ForEach``/``Transform1toN`` bug (see the module docstring of
    ``pyiron_workflow_assyst.workflow`` and the ``unwrap_singleton`` node): a
    length-1 zipped loop variable arrives wrapped in a spurious 1-tuple. Without
    the ``unwrap_singleton`` workaround this test fails with:

        TypeError: join() argument must be str, bytes, or os.PathLike object,
        not 'tuple'

    A test that instead avoids this configuration (e.g. by always passing a
    threshold that selects >1 image) would never catch a regression here --
    see ``test_assyst_graph_runs_end_to_end_multi_image_threshold`` below for
    the complementary length>=2 case, which is NOT affected by the bug.

    Beyond just "did it run": ``remove_calc_dirs=False`` leaves every written
    INCAR/POSCAR on disk, so this also asserts the graph computed the RIGHT
    thing -- exact output row count and job names, correct ISIF/NSW on every
    relaxation stage, correct KSPACING/EDIFF/EDIFFG/LREAL/NSW=0/no-ISIF on
    every accurate-statics job, and the ISIF7->5->2 structure feed-forward --
    not just that ``status == "finished"``.
    """
    run, job_dir = _run_graph(tmp_path)

    assert run.status == "finished", f"graph failed: {run.exception}"
    df = run.outputs.train_df
    assert isinstance(df, pd.DataFrame)
    assert os.path.exists(tmp_path / "df_ASSYST_jobs.pkl")

    # Exact coverage: 1 base image + rattle_1 + shear_1 + triax_1
    # permutations, no more, no fewer. `len(df) > 0` cannot distinguish "half
    # the dataset is missing" from "all of it is there" -- this can.
    expected_job_names = sorted(
        [
            _BASE_IMAGE_NAME,
            f"{_BASE_IMAGE_NAME}_rattle_1",
            f"{_BASE_IMAGE_NAME}_shear_1",
            f"{_BASE_IMAGE_NAME}_triax_1",
        ]
    )
    assert len(df) == 4
    assert sorted(df["job_name"].tolist()) == expected_job_names

    # Relaxation convergence must be surfaced on every row, not silently
    # discarded: `vasp_command` here writes VASP's own "reached required
    # accuracy" convergence line, so all three relaxation stages converge
    # and every row must carry True for all three flags.
    for col in ("isif7_converged", "isif5_converged", "isif2_converged"):
        assert col in df.columns, f"train_df missing convergence column {col!r}"
        assert df[col].tolist() == [True] * len(df), (
            f"{col}: expected every row True (the fake relaxations all "
            f"'converge'), got {df[col].tolist()}"
        )

    # Physics guards, read back from the actual written INCARs/POSCARs.
    _assert_isif_chain(job_dir, ionic_steps=5)
    for name in expected_job_names:
        _assert_static_incar(os.path.join(job_dir, name))
        if name == _BASE_IMAGE_NAME:
            _assert_base_static_structure(os.path.join(job_dir, name))
        else:
            _assert_perm_static_structure(os.path.join(job_dir, name))


@pytest.mark.slow
def test_assyst_graph_runs_end_to_end_multi_image_threshold(tmp_path):
    """Multi-image path: ``image_selection_eVatom_threshold=0.1``.

    ``_fake_parser``'s two energies (-8.0, -8.2 eV, single-atom cell) differ
    by 0.2 eV/atom > 0.1, so both images are selected and every zipped loop
    in the graph has length >= 2 -- the case pyiron_workflow's length-1
    ``ForEach`` bug does NOT affect. Kept alongside the default-threshold
    test above specifically because the difference between the two is the
    bug's exact boundary: both must pass, in both the presence and absence
    of the ``unwrap_singleton`` workaround for THIS test, while the
    default-threshold test above must fail without it.
    """
    run, job_dir = _run_graph(tmp_path, image_selection_eVatom_threshold=0.1)

    assert run.status == "finished", f"graph failed: {run.exception}"
    df = run.outputs.train_df
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert os.path.exists(tmp_path / "df_ASSYST_jobs.pkl")
    _assert_isif_chain(job_dir, ionic_steps=5)


@pytest.mark.slow
def test_assyst_graph_survives_singleton_permutation_loop(tmp_path):
    """The permutation loop's ``unwrap_singleton`` calls are independently
    load-bearing, not merely incidental coverage of the base loop's.

    With ``n_rattle_permutations=1, n_stretch_permutations=0``,
    ``get_ASSYST_deformed_structures`` produces exactly ONE permutation (a
    single rattle variant; zero triax, zero shear), so ``perm_structures``/
    ``perm_names`` are length-1 lists and the permutation loop hits
    pyiron_workflow's length-1 ``ForEach`` bug on its own -- independent of
    the base loop (which stays length-1 too here, default threshold, but
    keeps its own unwrap calls intact). Deleting ONLY the permutation loop's
    unwrap calls is invisible under the default test configuration above
    (whose permutation loop has length 3, unaffected), so this case needs
    its own test to be caught at all.
    """
    run, job_dir = _run_graph(
        tmp_path, n_rattle_permutations=1, n_stretch_permutations=0
    )

    assert run.status == "finished", f"graph failed: {run.exception}"
    df = run.outputs.train_df
    expected_job_names = sorted([_BASE_IMAGE_NAME, f"{_BASE_IMAGE_NAME}_rattle_1"])
    assert len(df) == 2
    assert sorted(df["job_name"].tolist()) == expected_job_names


@pytest.mark.slow
def test_zero_permutations_raises_a_clear_error(tmp_path):
    """``n_rattle_permutations=0, n_stretch_permutations=0`` is a plausible
    "relax and statics only, no permutations" CLI configuration -- but it
    makes ``get_ASSYST_deformed_structures`` return empty ``perm_structures``/
    ``perm_names`` lists, which would otherwise reach the permutation
    for-loop and hit pyiron_workflow's n=0 ``ForEach`` bug (see the module
    docstring): a bare ``IndexError: list index out of range`` with no
    mention of permutations. ``require_permutations`` guards against that,
    symmetrically with ``collect_structures``'s ``NoConvergedImagesError``,
    and must raise a clear, named, actionable error instead.
    """
    with pytest.raises(NoPermutationsGeneratedError):
        _run_graph(tmp_path, n_rattle_permutations=0, n_stretch_permutations=0)


def test_assyst_graph_has_the_expected_isif_chain():
    """Three relaxation-stage `vasp_job`s and two accurate-statics fan-outs
    (base images, permutations) must all be present as distinct children.

    This checks presence/count of labels in the STATIC graph only -- it does
    not run anything, so it cannot check execution order or feed-forward.
    The actual 7 -> 5 -> 2 order and structure feed-forward is verified for
    real, from written POSCARs, by ``_assert_isif_chain`` inside
    ``test_assyst_graph_runs_end_to_end_default_threshold`` above -- a
    stronger check than label inspection could ever give here.
    """
    node = pwf.node(run_ASSYST_on_structure)
    labels = list(node.nodes.keys())
    assert sum("vasp_job" in label for label in labels) == 3
    for_each_labels = [label for label in labels if "for_each" in label]
    assert len(for_each_labels) == 2, (
        f"expected exactly 2 for_each fan-outs (base images, permutations), "
        f"got {for_each_labels}"
    )
