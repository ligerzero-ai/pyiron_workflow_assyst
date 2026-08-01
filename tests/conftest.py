import numpy as np
import pytest
from pymatgen.core import Lattice, Structure


@pytest.fixture
def fe_structure() -> Structure:
    return Structure(Lattice.cubic(2.83), ["Fe"], [[0.0, 0.0, 0.0]])


@pytest.fixture
def two_species_structure() -> Structure:
    """Fe and H at 0.9 A separation - inside any sane core-overlap limit."""
    return Structure(
        Lattice.cubic(6.0),
        ["Fe", "H"],
        [[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]],
    )


@pytest.fixture
def sheared_cell() -> Structure:
    """A heavily sheared cell where the minimum-image convention LIES.

    The true nearest-neighbour distance (via a real neighbour search) is well
    under 1 A, but pymatgen's distance_matrix, which applies the minimum-image
    convention, reports a comfortably larger value. This is the fixture that
    distinguishes the two min_dist backends in Task 3; if a candidate cell does
    not show that discrepancy, it is the wrong fixture and the test proves
    nothing.

    Empirically verified (see task-1-report.md for the full derivation): with
    this lattice/basis, pymatgen's own pairwise search for the H-H cross term
    is numerically correct on its own - the discrepancy instead comes from
    `get_minimum_distance`'s "mic" backend excluding the distance_matrix
    diagonal (`np.fill_diagonal(..., np.inf)`), which throws away the fact
    that atom 0 sits only 0.9 A from its own periodic image along the short,
    heavily-tilted `a` lattice vector (abc = 0.9, 3.93, 3.88 A; angles =
    147.4, 52.4, 140.5 deg). A real neighbour search (the "neighbor_list"
    backend) correctly counts that self-image contact; the "mic" backend does
    not, and reports only the larger H-H cross-term distance instead:

        mic (pymatgen distance_matrix, diagonal excluded) = 1.4824 A
        d_true (ase.neighborlist.neighbor_list, self-images included) = 0.9000 A
    """
    lattice = Lattice(
        [
            [0.9, 0.0, 0.0],
            [-3.0286, 2.5009, 0.0],
            [2.3675, -2.2701, 2.0747],
        ]
    )
    return Structure(
        lattice,
        ["H", "H"],
        [[0.0, 0.0, 0.0], [0.469285, 0.859888, 0.448072]],
    )
