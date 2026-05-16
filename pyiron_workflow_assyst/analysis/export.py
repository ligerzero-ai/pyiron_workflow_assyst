"""Training-set export — pickle DataFrame, extended XYZ, or ASE-DB.

All formats produce one record per ``EngineOutput`` (one frame per row).
The ``pickle_df`` schema is intentionally a clean rewrite of the legacy
one-row-per-job list-valued schema — see the spec §10. Downstream
consumers of the legacy ``df_ASSYST_jobs.pkl`` will need a one-time
migration.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd
import pyiron_workflow as pwf
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write as ase_write

from pyiron_workflow_atomistics.engine import EngineOutput


@pwf.as_function_node("path")
def export_training_set(
    engine_outputs: list[EngineOutput],
    names: list[str],
    *,
    path: str = "df_ASSYST_jobs.pkl",
    format: Literal["pickle_df", "extxyz", "ase_db"] = "pickle_df",
) -> str:
    """Serialise the ASSYST training set to disk.

    Parameters
    ----------
    engine_outputs
        One EngineOutput per training frame.
    names
        Per-frame names, same length as ``engine_outputs``.
    path
        Output path. Extension is not enforced — caller's responsibility
        to match the format.
    format
        ``"pickle_df"`` -> pandas DataFrame with one row per frame and
        columns ``name, energy, volume, converged, structure, forces,
        stress``. ``"extxyz"`` -> ASE extended XYZ via ``ase.io.write``.
        ``"ase_db"`` -> SQLite via ``ase.db.connect``.

    Returns
    -------
    path
        The same ``path`` that was passed in (so the node has a useful
        output channel for downstream graph nodes).
    """
    if len(engine_outputs) != len(names):
        raise ValueError(
            f"engine_outputs ({len(engine_outputs)}) and names "
            f"({len(names)}) must have the same length"
        )

    if format == "pickle_df":
        rows = []
        for name, out in zip(names, engine_outputs):
            rows.append(
                {
                    "name": name,
                    "energy": out.final_energy,
                    "volume": out.final_volume,
                    "converged": out.converged,
                    "structure": out.final_structure,
                    "forces": out.final_forces,
                    "stress": out.final_stress_voigt,
                }
            )
        df = pd.DataFrame(
            rows,
            columns=[
                "name",
                "energy",
                "volume",
                "converged",
                "structure",
                "forces",
                "stress",
            ],
        )
        df.to_pickle(path)
    elif format == "extxyz":
        atoms_list = []
        for name, out in zip(names, engine_outputs):
            a = out.final_structure.copy()
            a.info["name"] = name
            calc_kwargs = {"energy": out.final_energy}
            if out.final_forces is not None:
                calc_kwargs["forces"] = out.final_forces
            a.calc = SinglePointCalculator(a, **calc_kwargs)
            atoms_list.append(a)
        ase_write(path, atoms_list, format="extxyz")
    elif format == "ase_db":
        from ase.db import connect

        with connect(path) as db:
            for name, out in zip(names, engine_outputs):
                a = out.final_structure.copy()
                a.calc = SinglePointCalculator(a, energy=out.final_energy)
                db.write(
                    a,
                    name=name,
                    converged=out.converged,
                )
    else:
        raise ValueError(f"Unknown export format: {format!r}")

    return path
