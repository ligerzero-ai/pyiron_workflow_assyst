import warnings

import numpy as np
import pytest

from pyiron_workflow_assyst.structure_filter_utils import (
    RCORE_FALLBACK,
    get_minimum_distance,
    is_valid_structure,
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


def test_mic_backend_overestimates_distance_in_sheared_cell(sheared_cell):
    """Documents the bug the default backend exists to avoid.

    If this test ever fails, the fixture has stopped exhibiting the
    discrepancy and every other test in this group is proving nothing.
    """
    mic = get_minimum_distance(sheared_cell, backend="mic")
    true = get_minimum_distance(sheared_cell, backend="neighbor_list")
    assert true < 1.0, f"fixture must have a genuine sub-A contact, got {true}"
    assert mic > true, f"mic ({mic}) should overestimate vs true ({true})"


def test_mic_backend_is_inert_for_single_atom_cells():
    """The most damaging form of the bug, and the one that reached production.

    With one atom the distance matrix is all-diagonal, so excluding the
    diagonal leaves nothing and the minimum is inf - the min_dist floor can
    never reject a one-atom cell however compressed it is. ASSYST unary
    generation runs with min_atoms=1, so these are produced routinely.
    """
    from pymatgen.core import Lattice, Structure

    tiny = Structure(Lattice.cubic(0.9), ["H"], [[0.0, 0.0, 0.0]])
    assert get_minimum_distance(tiny, backend="mic") == float("inf")
    assert get_minimum_distance(tiny, backend="neighbor_list") == pytest.approx(0.9)

    # Isolate the min_dist floor's inertness from the (separate, and always
    # correct - it uses get_all_neighbors, which sees self-images) RCORE
    # overlap check. At a=0.9 the H-H self-image distance is already below
    # the RCORE-derived overlap threshold for H
    # ((1 - 0.2) * 2 * 0.582 A = 0.931 A), so filter_distance_by_species
    # rejects `tiny` on its own, regardless of min_dist backend - that would
    # mask the floor bug in an is_valid_structure()-level check. At a=0.95
    # the self-image distance clears that RCORE threshold (0.95 A > 0.931 A)
    # while still sitting below min_dist=1.0, so only the min_dist floor's
    # own behaviour determines the outcome below.
    isolating_the_floor = Structure(Lattice.cubic(0.95), ["H"], [[0.0, 0.0, 0.0]])
    assert (
        is_valid_structure(isolating_the_floor, min_dist=1.0, min_dist_backend="mic")
        is True
    ), "mic backend's inert floor must let a 0.95 A self-image contact through"
    assert (
        is_valid_structure(isolating_the_floor, min_dist=1.0) is False
    ), "neighbor_list backend must reject that same contact"


def test_short_lattice_vector_hides_contact_from_mic_backend():
    """A multi-atom cell can also hide a contact along a short lattice vector."""
    from pymatgen.core import Lattice, Structure

    s = Structure(
        Lattice([[0.95, 0, 0], [0, 6, 0], [0, 0, 6]]),
        ["H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.5, 0.5]],
    )
    assert get_minimum_distance(s, backend="mic") > 4.0
    assert get_minimum_distance(s, backend="neighbor_list") == pytest.approx(0.95)


def test_default_backend_rejects_the_sub_angstrom_contact():
    """The default `min_dist_backend` must be 'neighbor_list', not the inert
    'mic' backend. `sheared_cell` cannot prove that on its own: its H-H
    contact is close enough that the RCORE overlap check rejects it
    independently of which min_dist backend runs (see
    `test_mic_backend_still_rejects_sheared_cell_via_the_rcore_check`), so a
    regression that silently defaulted to "mic" would go undetected there -
    confirmed by direct simulation, where `is_valid_structure(sheared_cell)`
    stayed False even with the default forced to "mic". This single-atom H
    cell at a=0.95 clears the RCORE threshold (0.931 A, so RCORE alone would
    accept it) but sits below min_dist=1.0, so only a correctly-wired
    default backend rejects it - the same simulation flips this one to True.
    """
    from pymatgen.core import Lattice, Structure

    s = Structure(Lattice.cubic(0.95), ["H"], [[0.0, 0.0, 0.0]])
    assert is_valid_structure(s, min_dist=1.0) is False


def test_mic_backend_still_rejects_sheared_cell_via_the_rcore_check(sheared_cell):
    """The mic backend's min_dist floor is inert here too (proven above by
    `test_mic_backend_overestimates_distance_in_sheared_cell`), but for
    `sheared_cell` specifically the overall verdict does not flip: the RCORE
    overlap check (via get_all_neighbors, which sees self-images and is
    unaffected by min_dist_backend) independently rejects the same 0.9 A
    H-H contact, since it is below the RCORE-derived allowed threshold for
    H ((1 - 0.2) * 2 * 0.582 A = 0.931 A). This also matches what pre-fix
    code did: filter_distance_by_species's overlap logic was never broken,
    so a pre-2026 campaign would have rejected this exact structure too -
    it is only the min_dist floor, isolated above, that was inert.
    """
    assert (
        is_valid_structure(sheared_cell, min_dist=1.0, min_dist_backend="mic") is False
    )


def test_backends_agree_on_an_undistorted_cell(two_species_structure):
    """`fe_structure` cannot serve this purpose: it is single-atom, so the
    mic backend is structurally inert (always inf) regardless of distortion.
    `two_species_structure` is a genuine cross-atom pair in an undistorted
    cubic cell, where pymatgen's distance_matrix (the "mic" backend) has
    already been verified numerically robust - so both backends should
    agree here.
    """
    assert get_minimum_distance(
        two_species_structure, backend="mic"
    ) == pytest.approx(
        get_minimum_distance(two_species_structure, backend="neighbor_list"),
        rel=1e-6,
    )


def test_unknown_backend_raises():
    from pymatgen.core import Lattice, Structure

    s = Structure(Lattice.cubic(3.0), ["Fe"], [[0, 0, 0]])
    with pytest.raises(ValueError, match="backend"):
        get_minimum_distance(s, backend="telepathy")
