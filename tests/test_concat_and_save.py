import pathlib

import numpy as np
import pandas as pd
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Incar
from vaspparser.vasp.output import parse_vasp_output as external_parse_vasp_output

from pyiron_workflow_assyst.workflow import _vasp_output_to_dataframe, concat_and_save

# --- concat_and_save / _vasp_output_to_dataframe: the DEFAULT parser's dict
# shape ------------------------------------------------------------------
#
# Discovered running the real ASSYST chain end to end against real VASP
# output, one step further than collect_structures: base_results/
# perm_results are the raw vasp_job outputs for each accurate-statics job,
# and concat_and_save fed them straight into pd.concat -- fine for the
# bundled DataFrame parser, but vasp_job's DEFAULT parser returns a dict for
# EVERY job it runs, including the statics jobs, not just the relaxation
# chain. Every existing end-to-end test injects a DataFrame-returning fake
# parser for the whole graph (relaxation AND statics), which is exactly why
# this shape mismatch (TypeError: cannot concatenate object of type
# 'class dict') survived until a real run reached this far.

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "vasp_isif2_fe2"


def _dict_output(cells, energies, scf_lens, numbers=(26,)):
    """Hand-built stand-in for the external parser's dict shape -- used only
    to exercise the NELM-derived SCF-convergence path and the missing-nelm
    warning/fallback directly, independent of the real fixture below (which
    happens to have every step converged, so can't prove the drop path)."""
    n_steps = len(energies)
    n_atoms = len(numbers)
    positions = np.zeros((n_steps, n_atoms, 3))
    cells_arr = np.array(cells, dtype=float)
    return {
        "generic": {
            "positions": positions,
            "cells": cells_arr,
            "energy_tot": np.array(energies, dtype=float),
            "dft": {"scf_energy_free": [[0.0] * n for n in scf_lens]},
        },
        "structure": {
            "numbers": np.array(numbers),
            "positions": positions[-1],
            "cell": cells_arr[-1],
            "pbc": np.array([True, True, True]),
        },
    }


def _fake_dataframe_row(a, energy, start_time=1):
    """Stand-in for a bundled-legacy-parser statics result: one row, one
    ionic step (accurate statics run with NSW=0)."""
    blob = np.array([Structure(Lattice.cubic(a), ["Fe"], [[0, 0, 0]]).to_json()])
    return pd.DataFrame(
        [
            {
                "structures": blob,
                "energy": np.array([energy]),
                "scf_convergence": np.array([True]),
                "calc_start_time": start_time,
            }
        ]
    )


def test_vasp_output_to_dataframe_accepts_real_external_parser_dict():
    """Uses a REAL captured dict (the same fixture collect_structures's dict
    test uses -- tests/fixtures/vasp_isif2_fe2/), not a hand-written
    stand-in, since a hand-written dict is exactly what let this class of
    bug through before."""
    real_output = external_parse_vasp_output(working_directory=str(FIXTURE_DIR))
    assert isinstance(real_output, dict), (
        "fixture-capture assumption broke upstream: vaspparser no longer "
        "returns a dict"
    )
    incar = Incar.from_file(FIXTURE_DIR / "INCAR")

    row = _vasp_output_to_dataframe(real_output, nelm=incar.get("NELM"))

    assert isinstance(row, pd.DataFrame)
    assert len(row) == 1
    n_steps = len(real_output["generic"]["energy_tot"])
    assert len(row["structures"].iloc[0]) == n_steps
    np.testing.assert_allclose(
        row["energy"].iloc[0], real_output["generic"]["energy_tot"]
    )
    assert list(row["scf_convergence"].iloc[0]) == [True] * n_steps

    # Each per-step structure must round-trip to the MATCHING ionic step --
    # not e.g. every entry silently built from step 0. The cell is IDENTICAL
    # across both steps for this ISIF2 fixture (ISIF=2 relaxes ions only,
    # not cell shape), so it cannot distinguish a wrong-index bug; positions
    # DO change step to step (energy went -11.06 -> -13.20 eV), so check
    # those, for every step, not just the last.
    for i in range(n_steps):
        step_structure = Structure.from_str(
            str(row["structures"].iloc[0][i]), fmt="json"
        )
        np.testing.assert_allclose(
            step_structure.cart_coords,
            real_output["generic"]["positions"][i],
            err_msg=f"row's step-{i} structure must match ionic step {i}'s positions",
        )


def test_vasp_output_to_dataframe_passes_dataframe_through_unchanged():
    """The bundled legacy parser's DataFrame output must be returned as-is
    -- no re-wrapping, no row duplication."""
    df = _fake_dataframe_row(3.5, -8.0)
    result = _vasp_output_to_dataframe(df)
    assert result is df


def test_vasp_output_to_dataframe_warns_and_assumes_default_nelm_when_missing():
    """Dropping unconverged images from training data is a real filter;
    silently treating every dict-path statics result as converged with no
    signal when nelm is omitted would be an invisible degradation. Must
    warn AND still apply VASP's built-in default (60) rather than warning
    and then ignoring it: scf_lens=[65] > 60 must be recorded as NOT
    converged."""
    cell = np.eye(3) * 3.0
    out = _dict_output(cells=[cell], energies=[-1.0], scf_lens=[65])
    with pytest.warns(UserWarning, match="nelm not provided"):
        row = _vasp_output_to_dataframe(out)
    assert list(row["scf_convergence"].iloc[0]) == [False]


def test_vasp_output_to_dataframe_rejects_unsupported_type():
    """Anything that is neither a dict nor a DataFrame must fail loudly with
    a clear TypeError naming both supported shapes."""
    with pytest.raises(TypeError, match="unsupported vasp_job result type"):
        _vasp_output_to_dataframe(["not", "supported"])


def test_concat_and_save_handles_mixed_dict_and_dataframe_results(tmp_path):
    """base_results/perm_results can be a MIX of dict (default parser) and
    DataFrame (bundled parser) entries -- e.g. base images collected from a
    dict-returning vasp_job but a custom vasp_parser_function used
    elsewhere -- and concat_and_save must concatenate both into one frame,
    still broadcasting the isif7/5/2_converged flags onto every row."""
    real_output = external_parse_vasp_output(working_directory=str(FIXTURE_DIR))
    df_result = _fake_dataframe_row(3.5, -8.0)

    out_path = tmp_path / "df_ASSYST_jobs.pkl"
    combined = concat_and_save(
        base_results=[real_output],
        perm_results=[df_result],
        filename=str(out_path),
        isif7_converged=True,
        isif5_converged=False,
        isif2_converged=True,
        nelm=12,
    )

    assert len(combined) == 2
    assert list(combined["isif7_converged"]) == [True, True]
    assert list(combined["isif5_converged"]) == [False, False]
    assert list(combined["isif2_converged"]) == [True, True]
    assert out_path.is_file()
    reloaded = pd.read_pickle(out_path)
    assert len(reloaded) == 2
