"""pyiron_workflow_assyst — ASSYST training-set generation workflows for pyiron.

The package follows ``pyiron_workflow_atomistics`` design principles: physics
workflows are engine-agnostic and route through the ``Engine`` Protocol;
structure operations are pure ASE-in / ASE-out; post-processing operates on
``EngineOutput``.

Subpackages
-----------
structure : engine-agnostic builders, deformations, filters, permutations
physics   : ``multistage_relax``, ``run_assyst`` macros
analysis  : trajectory frame collection, training-set export
testing   : shared pytest fixtures

Import the workflows per topic::

    from pyiron_workflow_assyst.physics.assyst import run_assyst
"""

from . import _version

__version__ = _version.get_versions()["version"]

__all__ = ["__version__"]


def __getattr__(name):
    """PEP 562 lazy loader — keeps build-time imports cheap.

    Without this, versioneer's ``[tool.setuptools.dynamic.version] attr =
    "pyiron_workflow_assyst.__version__"`` triggers a full package import
    at build time, cascading through ``from . import structure`` and
    pulling ``ase`` / ``pymatgen`` into the build-isolation env. See
    ``pyiron_workflow_atomistics``' historical fix for the same issue.
    """
    if name in {"structure", "physics", "analysis", "testing"}:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
