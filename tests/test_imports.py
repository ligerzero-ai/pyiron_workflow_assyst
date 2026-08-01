import pyiron_workflow as pwf


def test_pyiron_workflow_is_019():
    assert pwf.__version__.startswith("0.19")


def test_vasp_dependency_is_ported_version():
    """"Ported version" is judged by API shape, not a version-string floor.

    pyiron_workflow_vasp uses versioneer (git-describe-derived dev versions
    off its last tag, currently ``pyiron_workflow_vasp-0.1.0``), so its
    ``__version__`` can read e.g. ``"0.1.0.post0.dev8"`` even though the
    0.19 port is fully landed -- a hardcoded ``>= "0.2.0"`` floor would be
    comparing version strings lexicographically (wrong for semver in
    general) against a tag this package does not control and that upstream
    has not cut. The names below are the actual, load-bearing signal: they
    only exist on the flowrep-based 0.19 port, not the pre-port
    ``generate_VaspInput``/``construct_sequential_VaspInput_from_vaspoutput_structure``-era
    API, so this still fails against an unported dependency.
    """
    import pyiron_workflow_vasp as pwv

    for name in ("vasp_job", "generate_vasp_input", "construct_sequential_vasp_input"):
        assert hasattr(pwv, name), f"{name} missing from pyiron_workflow_vasp"


def test_assyst_exports_present():
    import pyiron_workflow_assyst as pwa

    for name in ("run_ASSYST_on_structure", "is_valid_structure", "resolve_rcore"):
        assert hasattr(pwa, name), f"{name} missing"


def test_all_advertised_exports_resolve():
    """Every name in __all__ must actually be importable from the package.

    Regression guard: a previous package in this project shipped an __all__
    advertising a name that did not exist, so `from pkg import *` raised
    AttributeError at the first star-import. Iterate __all__ explicitly
    rather than trusting it.
    """
    import pyiron_workflow_assyst as pwa

    assert len(pwa.__all__) > 0
    for name in pwa.__all__:
        assert hasattr(pwa, name), f"__all__ advertises {name!r}, but it does not resolve"
        assert getattr(pwa, name) is not None
