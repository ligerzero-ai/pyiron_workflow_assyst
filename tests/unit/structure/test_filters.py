"""Filter functions preserve the legacy ASSYST numerics."""

import importlib.util
import pathlib

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from pyiron_workflow_assyst.structure.filters import (
    RCORE,
    filter_distance_by_species,
    get_minimum_distance,
    is_valid_structure,
)


def _load_legacy_filters():
    """Import the frozen legacy module by absolute path to compare numerics."""
    p = (
        pathlib.Path(__file__).resolve().parents[2]
        / "_legacy_assyst"
        / "structure_filter_utils.py"
    )
    spec = importlib.util.spec_from_file_location("_legacy_filters", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRCORE:
    def test_table_bytes_identical_to_legacy(self):
        legacy = _load_legacy_filters()
        assert RCORE == legacy.RCORE


class TestMinimumDistance:
    def test_isolated_dimer(self):
        atoms = Atoms(
            "H2", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[10, 10, 10], pbc=True
        )
        assert get_minimum_distance(atoms) == pytest.approx(0.8, rel=1e-6)

    def test_bulk_copper(self):
        cu = bulk("Cu", "fcc", a=3.6, cubic=True)
        d = get_minimum_distance(cu)
        assert d == pytest.approx(3.6 / np.sqrt(2), rel=1e-6)


class TestIsValidStructure:
    def test_relaxed_bulk_is_valid(self):
        cu = bulk("Cu", "fcc", a=3.6, cubic=True)
        assert is_valid_structure(cu, min_dist=1.0, core_overlap_tolerance=0.2) is True

    def test_too_close_is_invalid(self):
        atoms = Atoms(
            "H2",
            positions=[[0, 0, 0], [0.1, 0, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        assert is_valid_structure(atoms, min_dist=1.0) is False

    def test_core_overlap_threshold(self):
        # Two Cu atoms at 0.79 · 2·RCORE_Cu — overlap tolerance 0.2
        # (allowed = 0.8 · sum) means this is invalid.
        rcu = RCORE["Cu"]
        atoms = Atoms(
            "Cu2",
            positions=[[0, 0, 0], [0.79 * 2 * rcu, 0, 0]],
            cell=[10, 10, 10],
            pbc=True,
        )
        assert filter_distance_by_species(atoms, core_overlap_tolerance=0.2) is False
