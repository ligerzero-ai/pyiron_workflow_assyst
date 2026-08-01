import pyiron_workflow as pwf


def test_pyiron_workflow_is_019():
    assert pwf.__version__.startswith("0.19")


def test_vasp_dependency_is_ported_version():
    import pyiron_workflow_vasp as pwv

    assert pwv.__version__ >= "0.2.0"
    for name in ("vasp_job", "generate_vasp_input", "construct_sequential_vasp_input"):
        assert hasattr(pwv, name), f"{name} missing from pyiron_workflow_vasp"


def test_assyst_exports_present():
    import pyiron_workflow_assyst as pwa

    for name in ("run_ASSYST_on_structure", "is_valid_structure", "resolve_rcore"):
        assert hasattr(pwa, name), f"{name} missing"
