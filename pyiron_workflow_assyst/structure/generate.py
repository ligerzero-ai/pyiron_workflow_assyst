"""PyXtal random crystal generation (optional extra)."""

from __future__ import annotations

import pyiron_workflow as pwf
from ase import Atoms

try:
    from pyxtal import pyxtal as _PyXtal  # type: ignore[import-untyped]

    _PYXTAL_AVAILABLE = True
except ImportError:
    _PYXTAL_AVAILABLE = False
    _PyXtal = None  # type: ignore[assignment]


@pwf.as_function_node("structures", "names")
def pyxtal_random_crystals(
    composition: dict[str, int],
    n_structures: int = 100,
    space_groups: list[int] | range = range(1, 231),
    volume_factor: float = 1.0,
    name_prefix: str = "pyxtal",
    seed: int | None = None,
) -> tuple[list[Atoms], list[str]]:
    """Sample ``n_structures`` random crystal structures across the requested
    space groups using PyXtal.

    Requires the optional ``[pyxtal]`` extra. ``composition`` maps element
    symbols to atom counts in the unit cell (e.g. ``{"Cu": 4}``).

    Each generated structure satisfies PyXtal's ``.valid`` flag — invalid
    sampling attempts are silently retried until ``n_structures`` candidates
    have been produced. There is no hard cap on retries; if your composition
    is incompatible with the requested space groups, this loop will hang.
    """
    if not _PYXTAL_AVAILABLE:
        raise ImportError(
            "PyXtal is required for random crystal generation. "
            "Install with: pip install pyiron_workflow_assyst[pyxtal]"
        )

    import random

    if seed is not None:
        random.seed(seed)

    species = list(composition.keys())
    numIons = list(composition.values())
    atoms_list: list[Atoms] = []
    names: list[str] = []
    sg_list = list(space_groups)

    while len(atoms_list) < n_structures:
        sg = random.choice(sg_list)
        struct = _PyXtal()
        try:
            struct.from_random(
                dim=3,
                group=sg,
                species=species,
                numIons=numIons,
                factor=volume_factor,
            )
        except Exception:
            continue
        if not struct.valid:
            continue
        atoms_list.append(struct.to_ase())
        names.append(f"{name_prefix}_{len(atoms_list) - 1}")

    return atoms_list, names
