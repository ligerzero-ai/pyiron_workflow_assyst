"""Rattle/triaxial/shear deformation generators for ASSYST structure sampling.

The `apply_*` functions are plain, pure functions of a pymatgen `Structure`
(moved verbatim out of `workflow.py`, where they were mixed in with graph
plumbing) so they can be unit tested directly. `get_ASSYST_deformed_structures`
is the flowrep node that drives them: for each input structure it emits
`n_rattle_permutations` rattled variants, then `n_stretch_permutations`
triaxially-strained variants, then `n_stretch_permutations` sheared variants,
retrying each until it finds one that passes `is_valid_structure` or gives up
after `max_attempts` tries for that variant.
"""

import numpy as np
import flowrep as fr
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)

from .structure_filter_utils import is_valid_structure


def apply_triaxial_strain(structure, max_strain=0.8):
    """Apply random triaxial strain up to max_strain."""
    # Generating a diagonal strain matrix for triaxial strain
    strain_values = 1 + np.random.uniform(-max_strain, max_strain, 3)
    strain_matrix = np.diag(strain_values)
    transformation = DeformStructureTransformation(strain_matrix)
    return transformation.apply_transformation(structure)


def apply_shear_strain(structure, max_strain=0.8):
    """Apply random shear strain up to max_strain."""
    # For shear, we need a deformation matrix that includes off-diagonal shear components.
    shear_matrix = np.identity(3) + np.random.uniform(-max_strain, max_strain, (3, 3))
    np.fill_diagonal(shear_matrix, 1)  # Keeping the volume roughly the same
    transformation = DeformStructureTransformation(shear_matrix)
    return transformation.apply_transformation(structure)


def apply_rattle(structure, displacement=0.5, max_cell_strain=0.05):
    """Apply random displacement (RATTLE) to atoms and a small strain."""
    new_struct = structure.copy()
    # Random displacement
    for site in new_struct:
        displacement_vector = np.random.normal(0, displacement, 3)
        site.coords += displacement_vector

    # Random small strain
    strain_values = 1 + np.random.uniform(-max_cell_strain, max_cell_strain, 3)
    strain_matrix = np.diag(strain_values)
    transformation = DeformStructureTransformation(strain_matrix)
    return transformation.apply_transformation(new_struct)


@fr.atomic("all_structures", "job_names")
def get_ASSYST_deformed_structures(
    structure_list,
    job_basename,
    n_stretch_permutations=5,
    n_rattle_permutations=5,
    shear_strain=0.8,
    triaxial_strain=0.8,
    rattle_displacement=0.1,
    rattle_strain=0.05,
    min_dist=1.0,
    core_overlap_tolerance=0.2,
    potcar_paths=None,
    min_dist_backend="neighbor_list",
    max_attempts=100,
    seed=None,
):
    """Generate validity-filtered rattle/triaxial/shear variants of each input.

    Output order per input structure is rattle, then triaxial, then shear;
    ``job_names`` is parallel to ``all_structures``. Generation retries until
    ``max_attempts`` per requested variant, so a structure that cannot be
    validly deformed yields fewer variants rather than looping forever.
    """
    if seed is not None:
        np.random.seed(seed)
    all_structures, job_names = [], []
    for idx, atoms in enumerate(structure_list):
        structure = AseAtomsAdaptor.get_structure(atoms)
        base = job_basename[idx]
        for kind, count, generate in (
            ("rattle", n_rattle_permutations,
             lambda s: apply_rattle(s, rattle_displacement, rattle_strain)),
            ("triax", n_stretch_permutations,
             lambda s: apply_triaxial_strain(s, triaxial_strain)),
            ("shear", n_stretch_permutations,
             lambda s: apply_shear_strain(s, shear_strain)),
        ):
            made = 0
            attempts = 0
            while made < count and attempts < max_attempts * count:
                attempts += 1
                candidate = generate(structure)
                if is_valid_structure(
                    candidate,
                    min_dist=min_dist,
                    core_overlap_tolerance=core_overlap_tolerance,
                    potcar_paths=potcar_paths,
                    min_dist_backend=min_dist_backend,
                ):
                    made += 1
                    all_structures.append(AseAtomsAdaptor.get_atoms(candidate))
                    job_names.append(f"{base}_{kind}_{made}")
    return all_structures, job_names
