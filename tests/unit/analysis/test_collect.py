"""collect_relaxation_frames threshold + convergence semantics."""

import numpy as np
import pytest
from ase.build import bulk

from pyiron_workflow_atomistics.engine import EngineOutput

from pyiron_workflow_assyst.analysis.collect import (
    _select_indices_by_threshold,
    collect_relaxation_frames,
)


@pytest.fixture
def trajectory_4_frames():
    """Synthetic EngineOutput with 4 frames; descending energy by 0.05 eV per
    frame on a 2-atom cell ⇒ 0.025 eV/atom per frame."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)[:2]
    energies = [-3.50, -3.55, -3.60, -3.65]
    structures = [atoms.copy() for _ in range(4)]
    return EngineOutput(
        final_structure=structures[-1],
        final_energy=energies[-1],
        converged=True,
        energies=energies,
        forces=[np.zeros((2, 3))] * 4,
        structures=structures,
        n_ionic_steps=4,
    )


class TestSelectIndicesByThreshold:
    def test_negative_one_returns_last_only(self):
        assert _select_indices_by_threshold([0.0, 1.0, 2.0, 3.0], threshold=-1.0) == [3]

    def test_positive_threshold_selects_first_last_and_jumps(self):
        assert _select_indices_by_threshold([0.0, 1.0, 2.0, 3.0], threshold=0.5) == [
            0,
            1,
            2,
            3,
        ]

    def test_threshold_skips_small_jumps(self):
        # 0 → 0.1 (skip, < 0.5) → 1.0 (keep) → 1.05 (skip) — first/last always in
        assert _select_indices_by_threshold([0.0, 0.1, 1.0, 1.05], threshold=0.5) == [
            0,
            2,
            3,
        ]

    def test_empty_returns_empty(self):
        assert _select_indices_by_threshold([], threshold=0.5) == []


class TestCollectFrames:
    def test_last_only_default(self, trajectory_4_frames):
        atoms, energies, names, indices = collect_relaxation_frames.node_function(
            engine_output=trajectory_4_frames,
            base_name="ISIF2",
        )
        assert len(atoms) == 1
        assert energies == [-3.65]
        assert names == ["ISIF2_accur_relaxstep3"]
        assert indices == [3]

    def test_threshold_picks_all_frames(self, trajectory_4_frames):
        atoms, energies, names, indices = collect_relaxation_frames.node_function(
            engine_output=trajectory_4_frames,
            base_name="ISIF2",
            image_selection_eV_atom_threshold=0.01,
        )
        # 0.025 eV/atom delta per frame > 0.01 ⇒ all 4
        assert indices == [0, 1, 2, 3]
        assert names == [f"ISIF2_accur_relaxstep{i}" for i in [0, 1, 2, 3]]


class TestConvergenceFilter:
    def test_drops_unconverged_when_required(self, trajectory_4_frames):
        trajectory_4_frames.converged = False
        atoms, energies, names, indices = collect_relaxation_frames.node_function(
            engine_output=trajectory_4_frames,
            base_name="ISIF2",
            require_converged=True,
        )
        assert atoms == []
        assert energies == []
        assert names == []
        assert indices == []

    def test_keeps_unconverged_when_not_required(self, trajectory_4_frames):
        trajectory_4_frames.converged = False
        atoms, _, _, _ = collect_relaxation_frames.node_function(
            engine_output=trajectory_4_frames,
            base_name="ISIF2",
            require_converged=False,
        )
        assert len(atoms) == 1
