import warnings

import pytest

from pyiron_workflow_assyst.structure_filter_utils import (
    RCORE_FALLBACK,
    rcore_from_potcar,
    resolve_rcore,
)

BOHR = 0.5291773
FE_POTCAR = (
    "/cmmc/u/hmai/pyiron-resources-cmmc/vasp/potentials/"
    "pyiron_nodes/potpaw_64/GGA/Fe/POTCAR"
)
TI_SV_POTCAR = (
    "/cmmc/u/hmai/pyiron-resources-cmmc/vasp/potentials/"
    "pyiron_nodes/potpaw_64/GGA/Ti_sv/POTCAR"
)


def test_rcore_from_potcar_reads_fe():
    rc = rcore_from_potcar(FE_POTCAR)
    assert set(rc) == {"Fe"}
    assert rc["Fe"] == pytest.approx(2.300 * BOHR, abs=1e-4)


def test_rcore_from_potcar_maps_sv_variant_to_bare_element():
    """Ti_sv must key as 'Ti' - the filter looks up by element symbol."""
    rc = rcore_from_potcar(TI_SV_POTCAR)
    assert set(rc) == {"Ti"}
    assert rc["Ti"] == pytest.approx(2.300 * BOHR, abs=1e-4)


def test_resolve_rcore_uses_semicore_defaults_matching_past_campaigns(fe_structure):
    """The default selection must reproduce the values every campaign ran with."""
    from pymatgen.core import Lattice, Structure

    ti = Structure(Lattice.cubic(3.3), ["Ti"], [[0.0, 0.0, 0.0]])
    assert resolve_rcore(ti)["Ti"] == pytest.approx(1.2171, abs=1e-3)
    assert resolve_rcore(fe_structure)["Fe"] == pytest.approx(1.2171, abs=1e-3)


def test_resolve_rcore_honours_explicit_potcar_paths():
    """An explicit plain-Ti POTCAR must give the larger plain radius, not the
    semicore default - this is the case a hardcoded table gets wrong."""
    from pymatgen.core import Lattice, Structure

    plain_ti = (
        "/cmmc/u/hmai/pyiron-resources-cmmc/vasp/potentials/"
        "pyiron_nodes/potpaw_64/GGA/Ti/POTCAR"
    )
    ti = Structure(Lattice.cubic(3.3), ["Ti"], [[0.0, 0.0, 0.0]])
    rc = resolve_rcore(ti, potcar_paths=[plain_ti])
    assert rc["Ti"] == pytest.approx(2.800 * BOHR, abs=1e-3)
    assert rc["Ti"] > resolve_rcore(ti)["Ti"], "plain must exceed semicore"


def test_resolve_rcore_falls_back_and_warns_on_unreadable_potcar():
    from pymatgen.core import Lattice, Structure

    fe = Structure(Lattice.cubic(2.83), ["Fe"], [[0.0, 0.0, 0.0]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        rc = resolve_rcore(fe, potcar_paths=["/nonexistent/POTCAR"])
    assert rc["Fe"] == pytest.approx(RCORE_FALLBACK["Fe"], abs=1e-6)
    assert any("POTCAR" in str(w.message) for w in caught), "must warn on fallback"
