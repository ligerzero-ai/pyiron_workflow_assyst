"""Top-level ASSYST data-generation macro: relax → harvest → static base →
permute → static perm → export."""

from __future__ import annotations

from typing import Literal

import pyiron_workflow as pwf
from ase import Atoms
from pyiron_workflow.api import for_node

from pyiron_workflow_atomistics.engine import (
    CalcInputMinimize,
    Engine,
    EngineOutput,
    calculate,
)

from pyiron_workflow_assyst._internal.engine_fanout import _build_subengines, _concat
from pyiron_workflow_assyst.analysis.collect import collect_relaxation_frames
from pyiron_workflow_assyst.analysis.export import export_training_set
from pyiron_workflow_assyst.physics.relaxation import multistage_relax
from pyiron_workflow_assyst.structure.permutations import generate_assyst_permutations

_OUTPUT_LABELS = ("training_path", "base_outputs", "perm_outputs")


@pwf.as_function_node("final_relax_output")
def _last_engine_output(engine_outputs: list) -> EngineOutput:
    """Extract the last EngineOutput from a list (final relax stage)."""
    final_relax_output = engine_outputs[-1]
    return final_relax_output


@pwf.as_function_node("engine_outputs")
def _df_to_engine_outputs(df) -> list[EngineOutput]:
    """Extract the engine_output column from a for_node DataFrame as a list."""
    engine_outputs = df["engine_output"].tolist()
    return engine_outputs


def _make_assyst_graph_creator(relax_stages: list, relax_stage_names: list):
    """Return a graph-creator function with ``relax_stages`` and
    ``relax_stage_names`` captured in closure.

    This is necessary because ``multistage_relax`` calls ``len(stages)`` at
    graph-construction time; if those args were passed as pwf graph-creator
    parameters they would arrive as ``UserInput`` channel objects (not lists).
    Capturing them in a closure ensures they remain plain Python values.
    """

    def _build_assyst_graph(
        wf,
        structure: Atoms,
        relax_engine: Engine,
        static_engine: Engine,
        base_name: str,
        image_selection_eV_atom_threshold: float,
        n_rattle: int,
        n_triaxial: int,
        n_shear: int,
        rattle_displacement: float,
        rattle_cell_strain: float,
        triaxial_strain: float,
        shear_strain: float,
        min_dist: float,
        core_overlap_tolerance: float,
        seed,
        training_path: str,
        export_format: Literal["pickle_df", "extxyz", "ase_db"],
    ):
        # 1. Multi-stage relaxation (stages/stage_names come from the closure)
        wf.relax = multistage_relax(
            structure=structure,
            engine=relax_engine,
            stages=relax_stages,
            stage_names=relax_stage_names,
        )

        # 2. Extract the last relax engine output (final stage)
        wf.last_relax = _last_engine_output(wf.relax.outputs.engine_outputs)

        # 3. Harvest base structures from the final relax output
        wf.harvest = collect_relaxation_frames(
            engine_output=wf.last_relax.outputs.final_relax_output,
            base_name=base_name,
            image_selection_eV_atom_threshold=image_selection_eV_atom_threshold,
        )

        # 4. Fan out static engines for base structures
        wf.base_engines = _build_subengines(
            engine=static_engine,
            names=wf.harvest.outputs.names,
        )

        # 5. Static calculations for base structures
        wf.base_static = for_node(
            calculate,
            zip_on=("structure", "engine"),
            structure=wf.harvest.outputs.structures,
            engine=wf.base_engines.outputs.subengines,
        )

        # 6. Extract flat list of base EngineOutputs from for_node DataFrame
        wf.base_eo = _df_to_engine_outputs(df=wf.base_static.outputs.df)

        # 7. Generate permutations
        wf.perms = generate_assyst_permutations(
            base_structures=wf.harvest.outputs.structures,
            base_names=wf.harvest.outputs.names,
            n_rattle=n_rattle,
            n_triaxial=n_triaxial,
            n_shear=n_shear,
            rattle_displacement=rattle_displacement,
            rattle_cell_strain=rattle_cell_strain,
            triaxial_strain=triaxial_strain,
            shear_strain=shear_strain,
            min_dist=min_dist,
            core_overlap_tolerance=core_overlap_tolerance,
            seed=seed,
        )

        # 8. Fan out static engines for permutation structures
        wf.perm_engines = _build_subengines(
            engine=static_engine,
            names=wf.perms.outputs.names,
        )

        # 9. Static calculations for permutation structures
        wf.perm_static = for_node(
            calculate,
            zip_on=("structure", "engine"),
            structure=wf.perms.outputs.structures,
            engine=wf.perm_engines.outputs.subengines,
        )

        # 10. Extract flat list of permutation EngineOutputs from for_node DataFrame
        wf.perm_eo = _df_to_engine_outputs(df=wf.perm_static.outputs.df)

        # 11. Concatenate base + perm outputs and names
        wf.all_outputs = _concat(
            a=wf.base_eo.outputs.engine_outputs,
            b=wf.perm_eo.outputs.engine_outputs,
        )
        wf.all_names = _concat(
            a=wf.harvest.outputs.names,
            b=wf.perms.outputs.names,
        )

        # 12. Export training set
        wf.export = export_training_set(
            engine_outputs=wf.all_outputs.outputs.concatenated,
            names=wf.all_names.outputs.concatenated,
            path=training_path,
            format=export_format,
        )

        return (
            wf.export.outputs.path,
            wf.base_eo.outputs.engine_outputs,
            wf.perm_eo.outputs.engine_outputs,
        )

    return _build_assyst_graph


def _default_relax_stages() -> list[CalcInputMinimize]:
    return [
        CalcInputMinimize(cell_relaxation="volume"),
        CalcInputMinimize(cell_relaxation="shape"),
        CalcInputMinimize(cell_relaxation="none"),
    ]


def _default_relax_stage_names() -> list[str]:
    return ["ISIF7", "ISIF5", "ISIF2"]


def run_assyst(
    structure: Atoms,
    relax_engine: Engine,
    static_engine: Engine,
    base_name: str = "pyxtal",
    *,
    relax_stages: list[CalcInputMinimize] | None = None,
    relax_stage_names: list[str] | None = None,
    image_selection_eV_atom_threshold: float = -1.0,
    n_rattle: int = 5,
    n_triaxial: int = 5,
    n_shear: int = 5,
    rattle_displacement: float = 0.1,
    rattle_cell_strain: float = 0.05,
    triaxial_strain: float = 0.8,
    shear_strain: float = 0.8,
    min_dist: float = 1.0,
    core_overlap_tolerance: float = 0.2,
    seed: int | None = None,
    training_path: str = "df_ASSYST_jobs.pkl",
    export_format: Literal["pickle_df", "extxyz", "ase_db"] = "pickle_df",
):
    """Chain relax → harvest → static base → permute → static perm → export.

    Returns a :class:`pyiron_workflow.nodes.macro.Macro` instance ready to
    ``.run()``.

    Parameters
    ----------
    structure:
        Input structure.
    relax_engine:
        Engine template for the relaxation stages.
    static_engine:
        Engine template for single-point static calculations. Per-job
        subdirectories are derived automatically from the job names.
    base_name:
        Prefix for harvested frame names.
    relax_stages:
        Sequence of CalcInputMinimize for each relaxation stage. Defaults
        to the three-stage ASSYST chain (ISIF=7/5/2). len must be 1 or 3.
    relax_stage_names:
        Directory labels for each relax stage.
    image_selection_eV_atom_threshold:
        Frame-selection threshold (eV/atom). -1 → last frame only.
    n_rattle, n_triaxial, n_shear:
        Number of permutations of each type per base structure.
    rattle_displacement, rattle_cell_strain, triaxial_strain, shear_strain:
        Amplitude parameters for each perturbation kind.
    min_dist, core_overlap_tolerance:
        Structure validity filters.
    seed:
        RNG seed for permutation reproducibility.
    training_path:
        Output file path for the exported training set.
    export_format:
        Serialisation format: ``"pickle_df"``, ``"extxyz"``, or ``"ase_db"``.

    Returns
    -------
    macro : Macro
        Runnable macro with outputs ``training_path``, ``base_outputs``,
        and ``perm_outputs``.
    """
    if relax_stages is None:
        relax_stages = _default_relax_stages()
    if relax_stage_names is None:
        relax_stage_names = _default_relax_stage_names()

    graph_creator = _make_assyst_graph_creator(relax_stages, relax_stage_names)

    return pwf.macro_node(
        graph_creator,
        output_labels=_OUTPUT_LABELS,
        structure=structure,
        relax_engine=relax_engine,
        static_engine=static_engine,
        base_name=base_name,
        image_selection_eV_atom_threshold=image_selection_eV_atom_threshold,
        n_rattle=n_rattle,
        n_triaxial=n_triaxial,
        n_shear=n_shear,
        rattle_displacement=rattle_displacement,
        rattle_cell_strain=rattle_cell_strain,
        triaxial_strain=triaxial_strain,
        shear_strain=shear_strain,
        min_dist=min_dist,
        core_overlap_tolerance=core_overlap_tolerance,
        seed=seed,
        training_path=training_path,
        export_format=export_format,
    )
