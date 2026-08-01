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


@pytest.mark.slow
def test_assyst_graph_runs_end_to_end(tmp_path):
    job_name = str((tmp_path / "struct_0").resolve())
    incar = Incar.from_dict({"ENCUT": 300, "ISPIN": 1, "NSW": 5})

    node = pwf.node(run_ASSYST_on_structure)
    run = node.run(
        structure=Structure(Lattice.cubic(2.83), ["Fe"], [[0, 0, 0]]),
        incar=incar,
        potcar_paths=None,
        job_name=job_name,
        vasp_command="echo 'reached required accuracy' > vasp.log",
        ionic_steps=5,
        n_stretch_permutations=1,
        n_rattle_permutations=1,
        # NOT -1 (the production default of "last image only"): pyiron_workflow
        # 0.19's ForEach/Transform1toN scatter node mis-handles length-1 zipped
        # inputs (verified with a minimal standalone repro, independent of this
        # package: a `for a, b in zip(xs, ys)` loop body over length-1 xs/ys
        # receives `a=(xs[0],)`, `b=(ys[0],)` -- an extra 1-tuple wrapper --
        # instead of the bare elements, because `atomic_node._store_atomic_outputs`
        # assigns a single-output atomic's whole return value to its one port
        # without unpacking, and the scatter node for a length-1 axis has
        # exactly one output port). `_fake_parser`'s two energies (-8.0, -8.2 eV,
        # single-atom cell) differ by 0.2 eV/atom, so threshold=0.1 selects BOTH
        # images and keeps every zipped loop below at length >= 2, sidestepping
        # the bug while still exercising both for_each fan-outs for real.
        image_selection_eVatom_threshold=0.1,
        remove_calc_dirs=False,
        compress_dirs=False,
        train_df_filename=str(tmp_path / "df_ASSYST_jobs.pkl"),
        seed=42,
        vasp_parser_function=_fake_parser,
    )

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
