"""Pure-ASE structural deformations for ASSYST permutations.

All functions accept and return ``ase.Atoms``. Distributions match the
legacy ASSYST code at the statistical level (verified by
``tests/unit/structure/test_deformations.py::TestDistributionMatchesLegacy``);
bit-for-bit identity within the new code is guaranteed by the golden-value
tests given the same ``rng``.

The ``rng: numpy.random.Generator | None`` parameter is the explicit way
to seed these functions for deterministic output — pass
``numpy.random.default_rng(42)`` (or any other seed) for reproducible
runs.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms


def _resolve_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def apply_rattle(
    atoms: Atoms,
    *,
    displacement: float = 0.1,
    max_cell_strain: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Gaussian per-atom displacements + uniform diagonal cell strain."""
    rng = _resolve_rng(rng)
    out = atoms.copy()
    positions = out.get_positions()
    positions += rng.normal(0.0, displacement, size=positions.shape)
    out.set_positions(positions)
    strain_vec = 1.0 + rng.uniform(-max_cell_strain, max_cell_strain, size=3)
    cell = out.cell.array @ np.diag(strain_vec)
    out.set_cell(cell, scale_atoms=True)
    return out


def apply_triaxial_strain(
    atoms: Atoms,
    *,
    max_strain: float = 0.8,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Uniform-random diagonal strain in U(-max_strain, +max_strain)."""
    rng = _resolve_rng(rng)
    out = atoms.copy()
    strain_vec = 1.0 + rng.uniform(-max_strain, max_strain, size=3)
    cell = out.cell.array @ np.diag(strain_vec)
    out.set_cell(cell, scale_atoms=True)
    return out


def apply_shear_strain(
    atoms: Atoms,
    *,
    max_strain: float = 0.8,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Uniform-random full 3x3 strain with diagonal forced to 1 (shear-only)."""
    rng = _resolve_rng(rng)
    out = atoms.copy()
    shear = np.eye(3) + rng.uniform(-max_strain, max_strain, size=(3, 3))
    np.fill_diagonal(shear, 1.0)
    cell = out.cell.array @ shear
    out.set_cell(cell, scale_atoms=True)
    return out
