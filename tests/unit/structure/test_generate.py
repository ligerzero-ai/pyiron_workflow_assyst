"""pyxtal_random_crystals: optional-dep node, lazy import, friendly error."""

import pytest

pyxtal = pytest.importorskip("pyxtal")

from pyiron_workflow_assyst.structure.generate import pyxtal_random_crystals  # noqa: E402


class TestPyXtal:
    def test_generates_requested_count(self):
        atoms_list, names = pyxtal_random_crystals.node_function(
            composition={"Cu": 4},
            n_structures=2,
            space_groups=[225],  # Fm-3m FCC
            name_prefix="cu",
            seed=1,
        )
        assert len(atoms_list) == 2
        assert names == ["cu_0", "cu_1"]
        for a in atoms_list:
            # Should be ASE Atoms (not pymatgen Structure)
            assert hasattr(a, "get_positions")
            assert len(a) == 4


class TestMissingPyXtal:
    def test_missing_pyxtal_raises_friendly(self, monkeypatch):
        from pyiron_workflow_assyst.structure import generate

        # Simulate pyxtal-not-installed at call time.
        monkeypatch.setattr(generate, "_PYXTAL_AVAILABLE", False)
        with pytest.raises(ImportError, match=r"\[pyxtal\]"):
            pyxtal_random_crystals.node_function(composition={"Cu": 4}, n_structures=1)
