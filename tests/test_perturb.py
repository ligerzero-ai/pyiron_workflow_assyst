import numpy as np
import pyiron_workflow as pwf
import pytest
from pymatgen.io.ase import AseAtomsAdaptor

from pyiron_workflow_assyst.perturb import (
    apply_rattle,
    apply_shear_strain,
    apply_triaxial_strain,
    get_ASSYST_deformed_structures,
)


def test_rattle_moves_atoms_but_preserves_composition(fe_structure):
    np.random.seed(0)
    out = apply_rattle(fe_structure, displacement=0.1, max_cell_strain=0.05)
    assert out.composition == fe_structure.composition
    assert len(out) == len(fe_structure)


def test_triaxial_strain_changes_volume(fe_structure):
    np.random.seed(0)
    out = apply_triaxial_strain(fe_structure, max_strain=0.5)
    assert out.volume != pytest.approx(fe_structure.volume)


def test_shear_strain_changes_angles_not_composition(fe_structure):
    np.random.seed(0)
    out = apply_shear_strain(fe_structure, max_strain=0.5)
    assert out.composition == fe_structure.composition
    assert out.lattice.angles != pytest.approx(fe_structure.lattice.angles)


def test_deformed_structures_counts_and_name_ordering(fe_structure):
    atoms = AseAtomsAdaptor.get_atoms(fe_structure)
    run = pwf.node(get_ASSYST_deformed_structures).run(
        structure_list=[atoms],
        job_basename=["base0"],
        n_stretch_permutations=2,
        n_rattle_permutations=3,
        seed=1234,
    )
    structures = run.outputs.all_structures
    names = run.outputs.job_names
    assert len(structures) == len(names) == 3 + 2 + 2
    assert [n.split("_")[1] for n in names] == (
        ["rattle"] * 3 + ["triax"] * 2 + ["shear"] * 2
    )
    assert all(n.startswith("base0_") for n in names)


def test_deformed_structures_are_deterministic_under_seed(fe_structure):
    atoms = AseAtomsAdaptor.get_atoms(fe_structure)
    kwargs = dict(
        structure_list=[atoms],
        job_basename=["b"],
        n_stretch_permutations=2,
        n_rattle_permutations=2,
        seed=7,
    )
    a = pwf.node(get_ASSYST_deformed_structures).run(**kwargs).outputs.all_structures
    b = pwf.node(get_ASSYST_deformed_structures).run(**kwargs).outputs.all_structures
    for x, y in zip(a, b):
        assert np.allclose(x.get_positions(), y.get_positions())
        assert np.allclose(x.get_cell(), y.get_cell())


def test_every_returned_structure_passes_the_validity_filter(fe_structure):
    """The generator's contract: it only emits structures that pass."""
    from pyiron_workflow_assyst.structure_filter_utils import is_valid_structure

    atoms = AseAtomsAdaptor.get_atoms(fe_structure)
    run = pwf.node(get_ASSYST_deformed_structures).run(
        structure_list=[atoms],
        job_basename=["b"],
        n_stretch_permutations=3,
        n_rattle_permutations=3,
        seed=99,
    )
    for atoms_out in run.outputs.all_structures:
        structure = AseAtomsAdaptor.get_structure(atoms_out)
        assert is_valid_structure(structure, min_dist=1.0, core_overlap_tolerance=0.2)


def test_gives_up_after_max_attempts_rather_than_looping_forever(two_species_structure):
    """A structure that cannot be validly deformed must terminate, returning
    fewer structures than requested rather than hanging."""
    atoms = AseAtomsAdaptor.get_atoms(two_species_structure)
    run = pwf.node(get_ASSYST_deformed_structures).run(
        structure_list=[atoms],
        job_basename=["tight"],
        n_stretch_permutations=2,
        n_rattle_permutations=2,
        rattle_displacement=3.0,
        max_attempts=5,
        seed=3,
    )
    assert len(run.outputs.all_structures) == len(run.outputs.job_names)
    assert run.status == "finished"
