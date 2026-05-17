"""Distance-based validity filters for ASSYST structure generation.

Public surface accepts ``ase.Atoms`` for consistency with the rest of the
package. Internally, the existing pymatgen-based ``get_all_neighbors``
algorithm is preserved bit-for-bit to keep numerics stable.
"""

from collections import defaultdict
from itertools import combinations_with_replacement

import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

RCORE = {
    # RCORE as per POTCAR * bohr to angtrom factor
    "H": 1.100000 * 0.5291773,
    "He": 1.100000 * 0.5291773,
    "Li": 2.050000 * 0.5291773,
    "Be": 1.900000 * 0.5291773,
    "B": 1.700000 * 0.5291773,
    "C": 1.500000 * 0.5291773,
    "N": 1.500000 * 0.5291773,
    "O": 1.520000 * 0.5291773,
    "F": 1.520000 * 0.5291773,
    "Ne": 1.700000 * 0.5291773,
    "Na": 2.200000 * 0.5291773,
    "Mg": 2.000000 * 0.5291773,
    "Al": 1.900000 * 0.5291773,
    "Si": 1.900000 * 0.5291773,
    "P": 1.900000 * 0.5291773,
    "S": 1.900000 * 0.5291773,
    "Cl": 1.900000 * 0.5291773,
    "Ar": 1.900000 * 0.5291773,
    "K": 2.300000 * 0.5291773,
    "Ca": 2.300000 * 0.5291773,
    "Sc": 2.500000 * 0.5291773,
    "Ti": 2.800000 * 0.5291773,
    "V": 2.700000 * 0.5291773,
    "Cr": 2.500000 * 0.5291773,
    "Mn": 2.300000 * 0.5291773,
    "Fe": 2.300000 * 0.5291773,
    "Co": 2.300000 * 0.5291773,
    "Ni": 2.300000 * 0.5291773,
    "Cu": 2.300000 * 0.5291773,
    "Zn": 2.300000 * 0.5291773,
    "Ga": 2.600000 * 0.5291773,
    "Ge": 2.300000 * 0.5291773,
    "As": 2.100000 * 0.5291773,
    "Se": 2.100000 * 0.5291773,
    "Br": 2.100000 * 0.5291773,
    "Kr": 2.300000 * 0.5291773,
    "Rb": 2.500000 * 0.5291773,
    "Sr": 2.500000 * 0.5291773,
    "Y": 2.800000 * 0.5291773,
    "Zr": 3.000000 * 0.5291773,
    "Nb": 2.400000 * 0.5291773,
    "Mo": 2.750000 * 0.5291773,
    "Tc": 2.800000 * 0.5291773,
    "Ru": 2.700000 * 0.5291773,
    "Rh": 2.700000 * 0.5291773,
    "Pd": 2.600000 * 0.5291773,
    "Ag": 2.500000 * 0.5291773,
    "Cd": 2.300000 * 0.5291773,
    "In": 3.100000 * 0.5291773,
    "Sn": 3.000000 * 0.5291773,
    "Sb": 2.300000 * 0.5291773,
    "Te": 2.300000 * 0.5291773,
    "I": 2.300000 * 0.5291773,
    "Xe": 2.500000 * 0.5291773,
    "Cs": 2.500000 * 0.5291773,
    "Ba": 2.800000 * 0.5291773,
    "La": 2.800000 * 0.5291773,
    "Hf": 3.000000 * 0.5291773,
    "Ta": 2.900000 * 0.5291773,
    "W": 2.750000 * 0.5291773,
    "Re": 2.700000 * 0.5291773,
    "Os": 2.700000 * 0.5291773,
    "Ir": 2.600000 * 0.5291773,
    "Pt": 2.600000 * 0.5291773,
    "Au": 2.500000 * 0.5291773,
    "Hg": 2.500000 * 0.5291773,
    "Tl": 3.200000 * 0.5291773,
    "Pb": 3.100000 * 0.5291773,
    "Bi": 3.000000 * 0.5291773,
}


def _to_pymatgen(atoms: Atoms):
    return AseAtomsAdaptor.get_structure(atoms)


def _element_wise_dist(structure):
    pair = defaultdict(lambda: np.inf)
    neighbors = structure.get_all_neighbors(r=5.0, include_index=True)
    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            j, d = neighbor.index, neighbor.nn_distance
            ei, ej = sorted((structure[i].specie.symbol, structure[j].specie.symbol))
            pair[ei, ej] = min(d, pair[ei, ej])
    return pair


def get_minimum_distance(atoms: Atoms) -> float:
    """Minimum pair distance (Angstrom) in the cell, excluding self-distance."""
    pmg = _to_pymatgen(atoms)
    dm = pmg.distance_matrix
    np.fill_diagonal(dm, np.inf)
    return float(np.min(dm))


def filter_distance_by_species(
    atoms: Atoms,
    *,
    rcore: dict = RCORE,
    core_overlap_tolerance: float = 0.2,
) -> bool:
    """True iff every species-pair distance >= (1 - tol) * (RCORE_i + RCORE_j)."""
    pmg = _to_pymatgen(atoms)
    if len(pmg) == 1:
        pmg = pmg * [2, 2, 2]
    pair = _element_wise_dist(pmg)
    species = sorted({site.specie.symbol for site in pmg})
    for ei, ej in combinations_with_replacement(species, 2):
        allowed = (1 - core_overlap_tolerance) * (rcore[ei] + rcore[ej])
        if pair[ei, ej] < allowed:
            return False
    return True


def is_valid_structure(
    atoms: Atoms,
    *,
    min_dist: float = 1.0,
    core_overlap_tolerance: float = 0.2,
) -> bool:
    """Combine min-dist and RCORE filters."""
    if get_minimum_distance(atoms) < min_dist:
        return False
    return filter_distance_by_species(
        atoms, core_overlap_tolerance=core_overlap_tolerance
    )
