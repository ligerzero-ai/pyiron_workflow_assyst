"""Engine-agnostic structure operations for ASSYST."""

from __future__ import annotations

_DEFERRED = {
    "apply_rattle": ".deformations",
    "apply_shear_strain": ".deformations",
    "apply_triaxial_strain": ".deformations",
    "generate_assyst_permutations": ".permutations",
    "pyxtal_random_crystals": ".generate",
}


def __getattr__(name: str):
    if name in _DEFERRED:
        import importlib

        mod = importlib.import_module(_DEFERRED[name], package=__name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .filters import (  # noqa: E402
    RCORE,
    filter_distance_by_species,
    get_minimum_distance,
    is_valid_structure,
)

__all__ = [
    "RCORE",
    "apply_rattle",
    "apply_shear_strain",
    "apply_triaxial_strain",
    "filter_distance_by_species",
    "generate_assyst_permutations",
    "get_minimum_distance",
    "is_valid_structure",
    "pyxtal_random_crystals",
]
