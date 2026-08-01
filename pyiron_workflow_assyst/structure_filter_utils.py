"""Structure-acceptance filters for ASSYST structure generation.

Core-radius (RCORE) values used by the overlap filter must match the POTCAR
variant a VASP run actually writes -- see `resolve_rcore`/`rcore_from_potcar`
for why these are read from the POTCAR by default rather than trusted to a
hardcoded table.
"""

import re
import warnings
from collections import defaultdict
from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np

BOHR_TO_ANGSTROM = 0.5291773

# Fallback core radii, in Angstrom, keyed by bare element symbol. Used only
# when no POTCAR is available to read RCORE from directly.
#
# These are NOT the plain-POTCAR radii from the original hardcoded RCORE
# table in this module's history -- they are the radii of the POTCAR
# *variants* `pyiron_workflow_vasp.vasp.get_default_POTCAR_paths` actually
# selects by default (e.g. Ti_sv, Cr_pv, Zr_sv, V_sv, Mo_sv, ...), read
# directly from the installed POTCAR files under the configured
# `pyiron_vasp_resources` library. Every past campaign ran with these
# semicore values, not the plain-element radii the old table held, so this
# fallback reproduces what actually ran rather than being stricter than any
# campaign to date.
RCORE_FALLBACK = {
    "H": 1.100 * BOHR_TO_ANGSTROM,  # H
    "He": 1.100 * BOHR_TO_ANGSTROM,  # He
    "Li": 1.700 * BOHR_TO_ANGSTROM,  # Li_sv
    "Be": 1.900 * BOHR_TO_ANGSTROM,  # Be
    "B": 1.700 * BOHR_TO_ANGSTROM,  # B
    "C": 1.500 * BOHR_TO_ANGSTROM,  # C
    "N": 1.500 * BOHR_TO_ANGSTROM,  # N
    "O": 1.520 * BOHR_TO_ANGSTROM,  # O
    "F": 1.520 * BOHR_TO_ANGSTROM,  # F
    "Ne": 1.700 * BOHR_TO_ANGSTROM,  # Ne
    "Na": 2.200 * BOHR_TO_ANGSTROM,  # Na_pv
    "Mg": 2.000 * BOHR_TO_ANGSTROM,  # Mg
    "Al": 1.900 * BOHR_TO_ANGSTROM,  # Al
    "Si": 1.900 * BOHR_TO_ANGSTROM,  # Si
    "P": 1.900 * BOHR_TO_ANGSTROM,  # P
    "S": 1.900 * BOHR_TO_ANGSTROM,  # S
    "Cl": 1.900 * BOHR_TO_ANGSTROM,  # Cl
    "Ar": 1.900 * BOHR_TO_ANGSTROM,  # Ar
    "K": 2.300 * BOHR_TO_ANGSTROM,  # K_sv
    "Ca": 2.300 * BOHR_TO_ANGSTROM,  # Ca_sv
    "Sc": 2.500 * BOHR_TO_ANGSTROM,  # Sc_sv
    "Ti": 2.300 * BOHR_TO_ANGSTROM,  # Ti_sv
    "V": 2.300 * BOHR_TO_ANGSTROM,  # V_sv
    "Cr": 2.300 * BOHR_TO_ANGSTROM,  # Cr_pv
    "Mn": 2.300 * BOHR_TO_ANGSTROM,  # Mn_pv
    "Fe": 2.300 * BOHR_TO_ANGSTROM,  # Fe
    "Co": 2.300 * BOHR_TO_ANGSTROM,  # Co
    "Ni": 2.300 * BOHR_TO_ANGSTROM,  # Ni
    "Cu": 2.300 * BOHR_TO_ANGSTROM,  # Cu
    "Zn": 2.300 * BOHR_TO_ANGSTROM,  # Zn
    "Ga": 2.300 * BOHR_TO_ANGSTROM,  # Ga_d
    "Ge": 2.300 * BOHR_TO_ANGSTROM,  # Ge_d
    "As": 2.100 * BOHR_TO_ANGSTROM,  # As
    "Se": 2.100 * BOHR_TO_ANGSTROM,  # Se
    "Br": 2.100 * BOHR_TO_ANGSTROM,  # Br
    "Kr": 2.300 * BOHR_TO_ANGSTROM,  # Kr
    "Rb": 2.500 * BOHR_TO_ANGSTROM,  # Rb_sv
    "Sr": 2.500 * BOHR_TO_ANGSTROM,  # Sr_sv
    "Y": 2.800 * BOHR_TO_ANGSTROM,  # Y_sv
    "Zr": 2.500 * BOHR_TO_ANGSTROM,  # Zr_sv
    "Nb": 2.400 * BOHR_TO_ANGSTROM,  # Nb_sv
    "Mo": 2.500 * BOHR_TO_ANGSTROM,  # Mo_sv
    "Tc": 2.600 * BOHR_TO_ANGSTROM,  # Tc_pv
    "Ru": 2.500 * BOHR_TO_ANGSTROM,  # Ru_pv
    "Rh": 2.400 * BOHR_TO_ANGSTROM,  # Rh_pv
    "Pd": 2.600 * BOHR_TO_ANGSTROM,  # Pd
    "Ag": 2.500 * BOHR_TO_ANGSTROM,  # Ag
    "Cd": 2.300 * BOHR_TO_ANGSTROM,  # Cd
    "In": 2.500 * BOHR_TO_ANGSTROM,  # In_d
    "Sn": 2.500 * BOHR_TO_ANGSTROM,  # Sn_d
    "Sb": 2.300 * BOHR_TO_ANGSTROM,  # Sb
    "Te": 2.300 * BOHR_TO_ANGSTROM,  # Te
    "I": 2.300 * BOHR_TO_ANGSTROM,  # I
    "Xe": 2.500 * BOHR_TO_ANGSTROM,  # Xe
    "Cs": 2.500 * BOHR_TO_ANGSTROM,  # Cs_sv
    "Ba": 2.800 * BOHR_TO_ANGSTROM,  # Ba_sv
    "La": 2.800 * BOHR_TO_ANGSTROM,  # La
    "Hf": 2.600 * BOHR_TO_ANGSTROM,  # Hf_pv
    "Ta": 2.500 * BOHR_TO_ANGSTROM,  # Ta_pv
    "W": 2.500 * BOHR_TO_ANGSTROM,  # W_sv
    "Re": 2.700 * BOHR_TO_ANGSTROM,  # Re
    "Os": 2.700 * BOHR_TO_ANGSTROM,  # Os
    "Ir": 2.600 * BOHR_TO_ANGSTROM,  # Ir
    "Pt": 2.600 * BOHR_TO_ANGSTROM,  # Pt
    "Au": 2.500 * BOHR_TO_ANGSTROM,  # Au
    "Hg": 2.500 * BOHR_TO_ANGSTROM,  # Hg
    "Tl": 2.500 * BOHR_TO_ANGSTROM,  # Tl_d
    "Pb": 2.500 * BOHR_TO_ANGSTROM,  # Pb_d
    "Bi": 2.500 * BOHR_TO_ANGSTROM,  # Bi_d
}

_TITEL_RE = re.compile(r"TITEL\s*=\s*\S+\s+(\S+)")
_RCORE_RE = re.compile(r"RCORE\s*=\s*([0-9.]+)")


@lru_cache(maxsize=256)
def rcore_from_potcar(potcar_path: str) -> dict[str, float]:
    """Map element symbol -> core radius in Angstrom, read from a POTCAR.

    A POTCAR may concatenate several elements; every TITEL/RCORE pair is
    read. The TITEL carries the variant name (e.g. ``Ti_sv``), but the
    returned key is the bare element symbol, because the overlap filter
    looks species up by symbol, not by POTCAR variant.
    """
    with open(potcar_path, errors="ignore") as handle:
        text = handle.read()
    titles = _TITEL_RE.findall(text)
    radii = _RCORE_RE.findall(text)
    if not titles or len(titles) != len(radii):
        raise ValueError(
            f"Could not read matching TITEL/RCORE pairs from {potcar_path}: "
            f"{len(titles)} titles, {len(radii)} radii"
        )
    return {
        title.split("_")[0]: float(radius) * BOHR_TO_ANGSTROM
        for title, radius in zip(titles, radii)
    }


def _rcore_fallback_or_raise(symbol: str) -> float:
    """``RCORE_FALLBACK[symbol]``, or a clear error naming the element if it
    has neither a resolved POTCAR reading nor a fallback table entry (e.g.
    ``Ce``, which is off-table). Without this, a missing key surfaces as a
    bare, confusing ``KeyError: 'Ce'`` chained from inside an ``except``
    block that is already handling an unrelated POTCAR-resolution failure.
    """
    try:
        return RCORE_FALLBACK[symbol]
    except KeyError:
        raise KeyError(
            f"No RCORE available for element {symbol!r}: it could not be "
            f"read from a POTCAR and has no RCORE_FALLBACK table entry."
        ) from None


def resolve_rcore(structure, potcar_paths=None) -> dict[str, float]:
    """Core radii for every element in ``structure``, from the POTCARs in use.

    Reading the POTCAR rather than consulting a static table matters because
    the default POTCAR selection is semicore (``Ti_sv``, ``Cr_pv``, ...),
    whose radii differ from the plain-element ones; a table hardcoded to
    either variant is wrong for the other. When ``potcar_paths`` is given
    (mirroring a VASP run's explicit override), those are read instead of
    the default selection. If a POTCAR cannot be read, falls back to
    ``RCORE_FALLBACK`` (with a warning) for the affected element(s) only -
    elements that *were* read successfully keep their genuine value.
    """
    symbols = sorted({site.specie.symbol for site in structure})

    if potcar_paths:
        merged: dict[str, float] = {}
        for path in potcar_paths:
            try:
                resolved = rcore_from_potcar(path)
            except (OSError, ValueError) as exc:
                warnings.warn(
                    f"Could not read RCORE from POTCAR {path!r} ({exc}); "
                    f"falling back to the built-in table.",
                    stacklevel=2,
                )
                continue
            for sym, radius in resolved.items():
                if sym in merged and merged[sym] != radius:
                    warnings.warn(
                        f"Element {sym!r} appears in more than one entry of "
                        f"potcar_paths with conflicting RCORE values "
                        f"({merged[sym]} vs {radius} Angstrom, the latter "
                        f"from {path!r}); the later path wins.",
                        stacklevel=2,
                    )
                merged[sym] = radius
        missing = [s for s in symbols if s not in merged]
        if missing:
            warnings.warn(
                f"No POTCAR RCORE for {missing}; falling back to the built-in table.",
                stacklevel=2,
            )
        return {
            sym: merged[sym] if sym in merged else _rcore_fallback_or_raise(sym)
            for sym in symbols
        }

    try:
        from ase import Atoms
        from pyiron_workflow_vasp.vasp import (
            _default_functional,
            _default_POTCAR_library_path,
            get_default_POTCAR_paths,
        )

        # get_default_POTCAR_paths only needs the element sequence (it reads
        # `atom.symbol` per site); a throwaway, uncelled Atoms is enough to
        # ask "what would the default POTCAR selection be for these
        # elements". The leading-underscore helpers are pyiron_workflow_vasp's
        # own way of lazily resolving the user's ``.pyiron_vasp_config`` (so
        # importing this module never requires one to exist) - there is no
        # public, non-underscore equivalent as of pyiron_workflow_vasp 0.2.x.
        atoms = Atoms(symbols, positions=[[float(i), 0.0, 0.0] for i in range(len(symbols))])
        paths = get_default_POTCAR_paths(
            atoms, _default_POTCAR_library_path(), _default_functional()
        )
    except Exception as exc:
        # Resolving *which* POTCARs to read failed entirely (missing
        # pyiron_workflow_vasp, misconfigured POTCAR library, unknown
        # element, ...) - every element falls back.
        warnings.warn(
            f"Could not resolve default POTCAR paths ({exc}); "
            f"falling back to the built-in RCORE table for all elements.",
            stacklevel=2,
        )
        return {sym: _rcore_fallback_or_raise(sym) for sym in symbols}

    merged = {}
    for path in paths:
        try:
            merged.update(rcore_from_potcar(path))
        except (OSError, ValueError) as exc:
            # A single unreadable/unparseable POTCAR must not poison every
            # other element that resolved fine - only the elements it would
            # have covered fall back (via the `missing` handling below).
            warnings.warn(
                f"Could not read RCORE from default POTCAR {path!r} ({exc}); "
                f"the affected element(s) will fall back to the built-in table.",
                stacklevel=2,
            )

    missing = [sym for sym in symbols if sym not in merged]
    if missing:
        warnings.warn(
            f"No default-POTCAR RCORE resolved for {missing}; "
            f"falling back to the built-in table for just these element(s).",
            stacklevel=2,
        )
    return {
        sym: merged[sym] if sym in merged else _rcore_fallback_or_raise(sym)
        for sym in symbols
    }


def _element_wise_dist(structure):
    """
    Computes the minimum distance between each pair of elements in the structure using pymatgen.

    Parameters:
    structure (Structure): A pymatgen Structure object.

    Returns:
    dict: A dictionary with element pairs as keys and their minimum distance as values.
    """
    pair = defaultdict(lambda: np.inf)

    # get_all_neighbors handles periodicity (including self-images) correctly
    # on its own; no manual supercell expansion is needed here.
    neighbors = structure.get_all_neighbors(
        r=5.0, include_index=True
    )  # Adjust the radius as needed

    # Loop through each atom and its neighbors
    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            j, d = neighbor.index, neighbor.nn_distance
            ei, ej = sorted((structure[i].specie.symbol, structure[j].specie.symbol))
            pair[ei, ej] = min(d, pair[ei, ej])

    return pair


def is_valid_structure(
    structure,
    min_dist=1.0,
    core_overlap_tolerance=0.2,
    potcar_paths=None,
    min_dist_backend="neighbor_list",
):
    # Validate both the minimum distance and RCORE-based constraints
    if get_minimum_distance(structure, backend=min_dist_backend) < min_dist:
        return False
    rcore = resolve_rcore(structure, potcar_paths=potcar_paths)
    return filter_distance_by_species(
        structure, rcore, core_overlap_tolerance=core_overlap_tolerance
    )


def get_minimum_distance(structure, backend: str = "neighbor_list") -> float:
    """Smallest interatomic distance in ``structure``.

    ``neighbor_list`` (default) performs a real periodic neighbour search,
    so it correctly counts an atom's distance to its own periodic image.

    ``mic`` uses pymatgen's ``distance_matrix`` with its diagonal (the
    self-image terms) excluded via ``np.fill_diagonal(..., np.inf)``. That
    diagonal exclusion throws away every atom's distance to its own
    periodic image, not just a numerically-degenerate self-term: for a
    single-atom cell the whole matrix is diagonal, so the "mic" backend
    always returns inf regardless of how compressed the cell is, and for
    any cell with a short lattice vector it can miss a genuine short
    contact along that vector. It is kept only to reproduce campaigns run
    before this was fixed.
    """
    if backend == "mic":
        distance_matrix = structure.distance_matrix.copy()
        np.fill_diagonal(distance_matrix, np.inf)  # Exclude self-site distances
        return float(np.min(distance_matrix))
    if backend == "neighbor_list":
        from ase.neighborlist import neighbor_list
        from pymatgen.io.ase import AseAtomsAdaptor

        atoms = AseAtomsAdaptor.get_atoms(structure)
        cutoff = 5.0
        for _ in range(4):
            distances = neighbor_list("d", atoms, cutoff)
            if len(distances):
                return float(distances.min())
            cutoff *= 2
        return float("inf")
    raise ValueError(
        f"Unknown backend {backend!r}; expected 'neighbor_list' or 'mic'."
    )


def filter_distance_by_species(structure, rcore, core_overlap_tolerance=0.2):
    """
    Checks if all interatomic distances in the structure are greater than the sum of
    the core radii for the involved element pairs, allowing for a percentage overlap tolerance.

    Parameters:
    structure (Structure): A pymatgen Structure object.
    rcore (dict): Dictionary of core radii for elements, e.g. from `resolve_rcore`.
    core_overlap_tolerance (float): Fractional overlap tolerance allowed (e.g., 0.2 for 20%).

    Returns:
    bool: True if all interatomic distances are within the allowed overlap tolerance, False otherwise.
    """
    pair = _element_wise_dist(structure)
    species_list = sorted({site.specie.symbol for site in structure})

    for ei, ej in combinations_with_replacement(species_list, 2):
        # Allow for overlap tolerance
        allowed_distance = (1 - core_overlap_tolerance) * (rcore[ei] + rcore[ej])
        if pair[ei, ej] < allowed_distance:
            return False
    return True
