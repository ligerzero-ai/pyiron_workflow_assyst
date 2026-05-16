"""Harvest training frames from a relaxation EngineOutput.

Replaces the legacy ``collect_structures`` + ``select_indices_by_threshold``
helpers from ``pyiron_workflow_assyst/workflow.py``. Operates on the
canonical ``EngineOutput`` instead of a VASP-parser DataFrame.
"""

from __future__ import annotations

import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import EngineOutput


def _select_indices_by_threshold(values: list[float], threshold: float) -> list[int]:
    """Algorithm preserved verbatim from legacy ``select_indices_by_threshold``.

    ``threshold == -1`` → last index only.
    Otherwise: always include first and last, plus every index whose value
    differs from the previously selected value by more than ``threshold``.
    """
    if not values:
        return []
    if threshold == -1:
        return [len(values) - 1]

    selected = [0]
    for i in range(1, len(values)):
        if abs(values[i] - values[selected[-1]]) > threshold:
            selected.append(i)
    if len(values) - 1 not in selected:
        selected.append(len(values) - 1)
    return selected


@pwf.as_function_node("structures", "energies", "names", "frame_indices")
def collect_relaxation_frames(
    engine_output: EngineOutput,
    base_name: str,
    *,
    image_selection_eV_atom_threshold: float = -1.0,
    require_converged: bool = True,
) -> tuple[list[Atoms], list[float], list[str], list[int]]:
    """Sub-select frames from a relaxation trajectory.

    ``image_selection_eV_atom_threshold == -1`` returns only the last
    frame (matches legacy default). Positive values select first/last
    plus any intermediate frame whose eV/atom differs from the previously
    selected frame by more than the threshold.

    If ``require_converged`` is True and the engine reports
    ``converged=False``, returns four empty lists (matches legacy
    SCF-filter behavior).
    """
    out_atoms: list[Atoms] = []
    out_energies: list[float] = []
    out_names: list[str] = []
    out_indices: list[int] = []

    if not (require_converged and not engine_output.converged):
        if not engine_output.structures or not engine_output.energies:
            # Static / no trajectory — final values only.
            out_atoms = [engine_output.final_structure]
            out_energies = [engine_output.final_energy]
            out_names = [f"{base_name}_accur_relaxstep0"]
            out_indices = [0]
        else:
            energies = engine_output.energies
            structures = engine_output.structures
            n_atoms = len(structures[0])
            eV_atom = [e / n_atoms for e in energies]
            indices = _select_indices_by_threshold(
                eV_atom, image_selection_eV_atom_threshold
            )
            out_atoms = [structures[i] for i in indices]
            out_energies = [energies[i] for i in indices]
            out_names = [f"{base_name}_accur_relaxstep{i}" for i in indices]
            out_indices = indices

    return out_atoms, out_energies, out_names, out_indices
