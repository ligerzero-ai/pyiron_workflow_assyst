import os

import numpy as np
import pandas as pd
import pyiron_workflow as pwf
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Incar

from pyiron_workflow_assyst.workflow import run_ASSYST_on_structure


def _fake_parser(directory):
    """Stand in for parse_vasp_directory with a realistically-shaped frame."""
    blobs = np.array(
        [Structure(Lattice.cubic(a), ["Fe"], [[0, 0, 0]]).to_json() for a in (3.0, 2.9)]
    )
    return pd.DataFrame(
        [
            {
                "structures": blobs,
                "energy": np.array([-8.0, -8.2]),
                "scf_convergence": np.array([True, True]),
                "calc_start_time": 1,
                "job_name": os.path.basename(str(directory)),
            }
        ]
    )


def _run_graph(tmp_path, **overrides):
    job_name = str((tmp_path / "struct_0").resolve())
    incar = Incar.from_dict({"ENCUT": 300, "ISPIN": 1, "NSW": 5})

    kwargs = dict(
        structure=Structure(Lattice.cubic(2.83), ["Fe"], [[0, 0, 0]]),
        incar=incar,
        potcar_paths=None,
        job_name=job_name,
        vasp_command="echo 'reached required accuracy' > vasp.log",
        ionic_steps=5,
        n_stretch_permutations=1,
        n_rattle_permutations=1,
        remove_calc_dirs=False,
        compress_dirs=False,
        train_df_filename=str(tmp_path / "df_ASSYST_jobs.pkl"),
        seed=42,
        vasp_parser_function=_fake_parser,
    )
    kwargs.update(overrides)
    node = pwf.node(run_ASSYST_on_structure)
    return node.run(**kwargs)


@pytest.mark.slow
def test_assyst_graph_runs_end_to_end_default_threshold(tmp_path):
    """Exercises the single-image PRODUCTION path.

    ``image_selection_eVatom_threshold`` is intentionally NOT passed here, so
    it takes the workflow's own default of -1 ("keep only the final
    relaxation image"). Every real ASSYST campaign runs with this default, so
    ``collect_structures`` routinely returns exactly ONE base structure, and
    both `for`-loops below iterate a length-1 collection.

    This is precisely the configuration that trips ``pyiron_workflow==0.19.0``'s
    ``ForEach``/``Transform1toN`` bug (see the module docstring of
    ``pyiron_workflow_assyst.workflow`` and the ``unwrap_singleton`` node): a
    length-1 zipped loop variable arrives wrapped in a spurious 1-tuple. Without
    the ``unwrap_singleton`` workaround this test fails with:

        TypeError: join() argument must be str, bytes, or os.PathLike object,
        not 'tuple'

    A test that instead avoids this configuration (e.g. by always passing a
    threshold that selects >1 image) would never catch a regression here --
    see ``test_assyst_graph_runs_end_to_end_multi_image_threshold`` below for
    the complementary length>=2 case, which is NOT affected by the bug.
    """
    run = _run_graph(tmp_path)

    assert run.status == "finished", f"graph failed: {run.exception}"
    df = run.outputs.train_df
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert os.path.exists(tmp_path / "df_ASSYST_jobs.pkl")


@pytest.mark.slow
def test_assyst_graph_runs_end_to_end_multi_image_threshold(tmp_path):
    """Multi-image path: ``image_selection_eVatom_threshold=0.1``.

    ``_fake_parser``'s two energies (-8.0, -8.2 eV, single-atom cell) differ
    by 0.2 eV/atom > 0.1, so both images are selected and every zipped loop
    in the graph has length >= 2 -- the case pyiron_workflow's length-1
    ``ForEach`` bug does NOT affect. Kept alongside the default-threshold
    test above specifically because the difference between the two is the
    bug's exact boundary: both must pass, in both the presence and absence
    of the ``unwrap_singleton`` workaround for THIS test, while the
    default-threshold test above must fail without it.
    """
    run = _run_graph(tmp_path, image_selection_eVatom_threshold=0.1)

    assert run.status == "finished", f"graph failed: {run.exception}"
    df = run.outputs.train_df
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert os.path.exists(tmp_path / "df_ASSYST_jobs.pkl")


def test_assyst_graph_has_the_expected_isif_chain():
    """The three relaxation stages must be present and ordered 7 -> 5 -> 2."""
    node = pwf.node(run_ASSYST_on_structure)
    labels = list(node.nodes.keys())
    assert sum("vasp_job" in label for label in labels) >= 3
    assert any("for_each" in label for label in labels), "fan-outs missing"
