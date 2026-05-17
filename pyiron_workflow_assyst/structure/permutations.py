"""ASSYST permutation generator: rattle + triaxial + shear, each validated."""

from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms

from .deformations import apply_rattle, apply_shear_strain, apply_triaxial_strain
from .filters import is_valid_structure


@pwf.as_function_node("structures", "names")
def generate_assyst_permutations(
    base_structures: list[Atoms],
    base_names: list[str],
    *,
    n_rattle: int = 5,
    n_triaxial: int = 5,
    n_shear: int = 5,
    rattle_displacement: float = 0.1,
    rattle_cell_strain: float = 0.05,
    triaxial_strain: float = 0.8,
    shear_strain: float = 0.8,
    min_dist: float = 1.0,
    core_overlap_tolerance: float = 0.2,
    max_attempts_per_perm: int = 100,
    seed: int | None = None,
) -> tuple[list[Atoms], list[str]]:
    """For each base structure, generate ``n_rattle`` + ``n_triaxial`` +
    ``n_shear`` valid permutations.

    Each candidate is rejected if ``is_valid_structure`` (min-distance
    + RCORE core-overlap) returns False; up to ``max_attempts_per_perm``
    retries per slot before falling short.

    Output order for each base is ``rattle -> triax -> shear``; names follow
    ``f"{base_name}_{tag}_{slot}"`` with ``slot`` 1-indexed.
    """
    rng = np.random.default_rng(seed)
    out_atoms: list[Atoms] = []
    out_names: list[str] = []
    for atoms, base_name in zip(base_structures, base_names):
        _fill(
            out_atoms,
            out_names,
            atoms,
            n_rattle,
            rng,
            "rattle",
            base_name,
            apply_rattle,
            dict(displacement=rattle_displacement, max_cell_strain=rattle_cell_strain),
            min_dist,
            core_overlap_tolerance,
            max_attempts_per_perm,
        )
        _fill(
            out_atoms,
            out_names,
            atoms,
            n_triaxial,
            rng,
            "triax",
            base_name,
            apply_triaxial_strain,
            dict(max_strain=triaxial_strain),
            min_dist,
            core_overlap_tolerance,
            max_attempts_per_perm,
        )
        _fill(
            out_atoms,
            out_names,
            atoms,
            n_shear,
            rng,
            "shear",
            base_name,
            apply_shear_strain,
            dict(max_strain=shear_strain),
            min_dist,
            core_overlap_tolerance,
            max_attempts_per_perm,
        )
    return out_atoms, out_names


def _fill(
    out_atoms: list,
    out_names: list,
    base: Atoms,
    n: int,
    rng: np.random.Generator,
    tag: str,
    base_name: str,
    fn,
    kwargs: dict,
    min_dist: float,
    tol: float,
    max_attempts: int,
) -> None:
    accepted = 0
    attempts = 0
    cap = max_attempts * n
    while accepted < n and attempts < cap:
        candidate = fn(base, rng=rng, **kwargs)
        if is_valid_structure(candidate, min_dist=min_dist, core_overlap_tolerance=tol):
            out_atoms.append(candidate)
            out_names.append(f"{base_name}_{tag}_{accepted + 1}")
            accepted += 1
        attempts += 1
