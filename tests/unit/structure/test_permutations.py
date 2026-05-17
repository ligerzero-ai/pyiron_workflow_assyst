"""generate_assyst_permutations: correct shape, names, validity, determinism."""

import numpy as np
import pytest
from ase.build import bulk

from pyiron_workflow_assyst.structure.filters import is_valid_structure
from pyiron_workflow_assyst.structure.permutations import generate_assyst_permutations


@pytest.fixture
def two_bases():
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    return [cu, cu.copy()], ["pyxtal_0", "pyxtal_1"]


class TestShape:
    def test_count_per_category(self, two_bases):
        bases, names = two_bases
        atoms, perm_names = generate_assyst_permutations.node_function(
            base_structures=bases,
            base_names=names,
            n_rattle=2,
            n_triaxial=2,
            n_shear=2,
            rattle_displacement=0.05,
            rattle_cell_strain=0.02,
            triaxial_strain=0.05,
            shear_strain=0.05,
            seed=7,
        )
        # 2 bases × (2 rattle + 2 triax + 2 shear) = 12
        assert len(atoms) == 12
        assert len(perm_names) == 12


class TestNames:
    def test_name_scheme(self, two_bases):
        bases, names = two_bases
        _, perm_names = generate_assyst_permutations.node_function(
            base_structures=bases,
            base_names=names,
            n_rattle=1,
            n_triaxial=1,
            n_shear=1,
            rattle_displacement=0.05,
            rattle_cell_strain=0.02,
            triaxial_strain=0.05,
            shear_strain=0.05,
            seed=7,
        )
        # For each base, rattle then triax then shear; index is 1-based
        assert perm_names == [
            "pyxtal_0_rattle_1",
            "pyxtal_0_triax_1",
            "pyxtal_0_shear_1",
            "pyxtal_1_rattle_1",
            "pyxtal_1_triax_1",
            "pyxtal_1_shear_1",
        ]


class TestValidity:
    def test_all_outputs_pass_filter(self, two_bases):
        bases, names = two_bases
        atoms, _ = generate_assyst_permutations.node_function(
            base_structures=bases,
            base_names=names,
            n_rattle=2,
            n_triaxial=2,
            n_shear=2,
            rattle_displacement=0.05,
            rattle_cell_strain=0.02,
            triaxial_strain=0.05,
            shear_strain=0.05,
            min_dist=1.0,
            core_overlap_tolerance=0.2,
            seed=7,
        )
        for a in atoms:
            assert is_valid_structure(a, min_dist=1.0, core_overlap_tolerance=0.2)


class TestDeterminism:
    def test_same_seed_same_output(self, two_bases):
        bases, names = two_bases
        a1, n1 = generate_assyst_permutations.node_function(
            base_structures=bases,
            base_names=names,
            n_rattle=1,
            n_triaxial=1,
            n_shear=1,
            seed=99,
            rattle_displacement=0.05,
            rattle_cell_strain=0.02,
            triaxial_strain=0.05,
            shear_strain=0.05,
        )
        a2, n2 = generate_assyst_permutations.node_function(
            base_structures=bases,
            base_names=names,
            n_rattle=1,
            n_triaxial=1,
            n_shear=1,
            seed=99,
            rattle_displacement=0.05,
            rattle_cell_strain=0.02,
            triaxial_strain=0.05,
            shear_strain=0.05,
        )
        assert n1 == n2
        for x, y in zip(a1, a2):
            np.testing.assert_array_equal(x.get_positions(), y.get_positions())
            np.testing.assert_array_equal(x.cell.array, y.cell.array)
