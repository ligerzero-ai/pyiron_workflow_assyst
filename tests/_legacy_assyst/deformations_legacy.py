"""Standalone snapshot of the legacy apply_rattle / apply_triaxial_strain /
apply_shear_strain functions from pyiron_workflow_assyst/workflow.py
(pre-rewrite).

Carved out so they can be imported without pulling the full legacy module's
relative imports / pyiron_workflow dependencies. Used only by
tests/unit/structure/test_deformations.py::TestDistributionMatchesLegacy
to verify the new implementations match the legacy distributions
statistically.

DO NOT EDIT — this is the frozen reference.
"""

import numpy as np
from pymatgen.transformations.standard_transformations import (
    DeformStructureTransformation,
)


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
