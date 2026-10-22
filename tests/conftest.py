import os
from pathlib import Path

import pytest
from pymatgen.core import Lattice, Structure

# RCORE (bohr) written into the fake POTCARs below -- the real potpaw_64 GGA
# values for each element's default variant, so code reading them behaves
# exactly as it would against a real library.
FAKE_POTCAR_RCORE_BOHR = {"Fe": 2.300, "H": 1.100, "Ce": 2.700}


@pytest.fixture
def fake_potcar_library(tmp_path, monkeypatch):
    """Point pyiron_workflow_vasp at a throwaway config and POTCAR tree.

    Anything that resolves default POTCARs (``VaspInput``/``write_POTCAR``,
    ``resolve_rcore`` without ``potcar_paths``) otherwise needs the user's
    real ``~/.pyiron_vasp_config`` and licensed POTCAR files. Each fake
    POTCAR carries just the TITEL/RCORE lines this package reads, for the
    default variant of every element in ``FAKE_POTCAR_RCORE_BOHR``.
    pyiron_workflow_vasp resolves its config path at import and caches the
    parsed config, so both are reset here (monkeypatch restores the path).
    """
    from ase import Atoms
    from pyiron_workflow_vasp import vasp as _vasp

    resources = tmp_path / "pyiron_vasp_resources"
    cfg = tmp_path / ".pyiron_vasp_config"
    cfg.write_text(
        "default_POTCAR_set = potpaw64\n"
        "default_functional = GGA\n"
        f"pyiron_vasp_resources = {resources}\n"
        "vasp_POTCAR_path_potpaw64 = {pyiron_vasp_resources}/potpaw_64\n"
    )
    monkeypatch.setenv("PYIRON_VASP_CONFIG", str(cfg))
    monkeypatch.setattr(_vasp, "DEFAULT_CONFIG_PATH", cfg)
    _vasp._get_potcar_config.cache_clear()

    symbols = list(FAKE_POTCAR_RCORE_BOHR)
    paths = _vasp.get_default_POTCAR_paths(
        Atoms(symbols, positions=[[float(i), 0.0, 0.0] for i in range(len(symbols))]),
        _vasp._default_POTCAR_library_path(),
        _vasp._default_functional(),
    )
    for symbol, path in zip(symbols, paths):
        variant = Path(path).parent.name
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Path(path).write_text(
            f"  PAW_PBE {variant} 01Jan2000\n"
            f"   TITEL  = PAW_PBE {variant} 01Jan2000\n"
            f"   RCORE  =    {FAKE_POTCAR_RCORE_BOHR[symbol]:.3f}"
            "    outmost cutoff radius\n"
        )
    try:
        yield resources
    finally:
        _vasp._get_potcar_config.cache_clear()


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
