# pyiron_workflow_assyst rewrite — engine-agnostic, pwa-aligned

Date: 2026-05-16
Status: Approved (design)

## 1. Goal

Rewrite `pyiron_workflow_assyst` as a pyiron-workflow module that follows the
same design principles as `pyiron_workflow_atomistics` (pwa), while preserving
exact equivalence to the current VASP-only ASSYST workflow when driven with
`VaspEngine`.

Three things must hold simultaneously:

1. **Engine-agnostic.** Physics workflows take a `pyiron_workflow_atomistics.engine.Engine`
   and call `calculate()`. No VASP-specific imports leak into `physics/` or
   `structure/`. Any future engine (ASE, VASP, LAMMPS, GRACE, ...) drives the
   pipeline through the Engine Protocol.
2. **VASP-equivalent.** Wired with `VaspEngine`, the new module reproduces the
   current pipeline behavior — ISIF=7→5→2 relaxation cascade, trajectory
   harvesting with SCF-convergence filter, rattle/triaxial/shear permutations,
   accurate-static SCFs, pickle-DataFrame training set — to within ≤ 1e-3 eV/atom
   energy drift on identical inputs.
3. **pwa-shaped.** Subpackage layering, naming, decorator conventions, version
   handling, and dependency pinning match pwa.

## 2. Background

### Current code (clone @ `~/pyiron_workflow_assyst`)

`pyiron_workflow_assyst/__init__.py` re-exports `run_ASSYST_on_structure`,
`get_ASSYST_deformed_structures`, `collect_structures`, `RCORE`,
`is_valid_structure` at the top level. Implementation is two files:

- `workflow.py` (507 LOC) — one giant `@pwf.as_macro_node`
  `run_ASSYST_on_structure` that wires INCAR construction, three sequential
  `vasp_job` calls (ISIF=7/5/2), trajectory collection via
  `collect_structures`, rattle/triax/shear permutations, accurate-static
  `vasp_job` fan-out via `for_node`, and a final pickle of the combined
  DataFrame. Tightly coupled to `pyiron_workflow_vasp.vasp` (`VaspInput`,
  `vasp_job`, `generate_modified_incar`,
  `construct_sequential_VaspInput_from_vaspoutput_structure`,
  `parse_vasp_directory`).
- `structure_filter_utils.py` (155 LOC) — `RCORE` POTCAR core-radius table,
  `_element_wise_dist`, `get_minimum_distance`, `filter_distance_by_species`,
  `is_valid_structure`.

### pwa design principles to inherit

(Verified against `~/pyiron_workflow_atomistics` HEAD on 2026-05-16.)

1. **Engine abstraction.** `pyiron_workflow_atomistics.engine.Engine` is a
   runtime-checkable Protocol with `working_directory`, `get_calculate_fn`,
   `with_working_directory`. Implementations live alongside their backend
   (`ASEEngine` in pwa; `VaspEngine` in pyiron_workflow_vasp).
2. **Physics-level input dataclasses.** `CalcInputStatic`, `CalcInputMinimize`,
   `CalcInputMD` describe *what* in physics language. Engines translate to
   native parameters.
3. **Canonical result type.** `EngineOutput` dataclass with `final_structure`,
   `final_energy`, `converged`, plus optional `forces`, `stress`, trajectory.
   Downstream code reads `EngineOutput`, never engine-specific outputs.
4. **Subpackage layering.** `engine/` · `structure/` (engine-agnostic) ·
   `physics/` (workflows by topic, no cross-export) · `analysis/`
   (post-processing) · `testing/` (conformance mixins) · `_internal/` (private
   plumbing).
5. **Decorator conventions.** Small `@pwf.as_function_node("out_name")` units
   composed into `@pwf.as_macro_node` graphs. `subengine()` / `subdir_path()`
   helpers for nested workdirs. `for_node` for fan-out (e.g.
   `physics/grain_boundary.py:174`).
6. **PEP 562 lazy `__getattr__`** in top-level `__init__.py` (resolves the
   versioneer + build-isolation issue documented in
   `[[pwa-install-quirk]]`).
7. **Versioneer for versions.** Pinned exact dep versions in pyproject.toml.

### VaspEngine status (verified @ `ligerzero-ai/pyiron_workflow_vasp` HEAD)

`pyiron_workflow_vasp/engine.py` implements the Engine Protocol. It exposes
`EngineInput` (`CalcInputStatic | CalcInputMinimize | CalcInputMD`),
`working_directory`, `potcar_config_file`, `functional`, `encut`,
`kpoints_density`, `command`. MD raises `NotImplementedError`.
`pyiron_workflow_vasp/_run.py` maps `params["ISIF"] = 3 if relax_cell else 2`.

## 3. Architecture

### Package layout

```
pyiron_workflow_assyst/
├── __init__.py               # versioneer __version__; PEP 562 lazy __getattr__
├── _version.py               # versioneer-managed
├── py.typed
├── _internal/
│   ├── __init__.py
│   └── engine_fanout.py      # _build_subengines, _concat — node-side list helpers
├── structure/
│   ├── __init__.py           # re-exports the public structure ops
│   ├── filters.py            # RCORE, min-dist + core-overlap validators
│   ├── deformations.py       # apply_rattle / apply_triaxial_strain / apply_shear_strain
│   ├── permutations.py       # @function_node generate_assyst_permutations
│   └── generate.py           # @function_node pyxtal_random_crystals (optional dep)
├── physics/
│   ├── __init__.py           # docstring only — no re-exports
│   ├── relaxation.py         # @macro_node multistage_relax (reusable building block)
│   └── assyst.py             # @macro_node run_assyst (top-level pipeline)
├── analysis/
│   ├── __init__.py
│   ├── collect.py            # @function_node collect_relaxation_frames
│   └── export.py             # @function_node export_training_set
└── testing/
    ├── __init__.py
    └── fixtures.py           # shared pytest fixtures (FCC seed, ASE+EMT engine)

tests/
├── unit/structure/, unit/analysis/   # mirrors source tree
└── integration/              # ASE+EMT end-to-end smoke (VASP gated by marker)

docs/
└── superpowers/specs/2026-05-16-pyiron-workflow-assyst-rewrite-design.md
```

### Layering rules

- `structure/` is engine-agnostic — `ase.Atoms` in, `ase.Atoms` out. No
  `Engine` imports.
- `physics/` is the only place `Engine` enters. Macros take
  `(structure, engine, ...)` and call
  `pyiron_workflow_atomistics.engine.calculate`.
- `analysis/` consumes `list[EngineOutput]` and structures — no Engine, no
  calculator imports.
- `_internal/` and any `_*` module is private; not re-exported.
- `physics/__init__.py` and `analysis/__init__.py` re-export nothing across
  topics — callers import per topic (e.g.
  `from pyiron_workflow_assyst.physics.assyst import run_assyst`).

### Dependencies (`pyproject.toml`)

Drop the existing `pyiron_vasp` / `pyiron_workflow_vasp` runtime deps. Pin
versions in pwa style:

- Required: `pyiron_workflow_atomistics` (Engine, calculate, CalcInputMinimize,
  CalcInputStatic, ASEEngine, subengine, subdir_path), `pyiron-workflow`,
  `ase`, `pymatgen`, `numpy`, `pandas`, `tqdm`. Exact versions chosen to match
  pwa's pins where they overlap.
- `[project.optional-dependencies]`:
  - `pyxtal = ["pyxtal"]` — random crystal generation
  - `vasp = ["pyiron_workflow_vasp"]` — for users wanting VaspEngine
  - `test = ["pytest", "nbformat", "nbclient"]`

Build backend: setuptools + versioneer (mirrors pwa).

## 4. Module contents and node signatures

### `structure/filters.py`

- `RCORE: dict[str, float]` — POTCAR core radii × Bohr→Å. **Bytes-identical**
  to legacy table.
- `get_minimum_distance(atoms: ase.Atoms) -> float`
- `filter_distance_by_species(atoms: ase.Atoms, *, rcore: dict[str, float] = RCORE, core_overlap_tolerance: float = 0.2) -> bool`
- `is_valid_structure(atoms: ase.Atoms, *, min_dist: float = 1.0, core_overlap_tolerance: float = 0.2) -> bool`

Public surface accepts `ase.Atoms` (pwa convention). Internally converts via
`AseAtomsAdaptor.get_structure` and runs the existing pymatgen
`get_all_neighbors(r=5.0)` logic — numerics preserved bit-for-bit.

### `structure/deformations.py`

Pure functions, ASE in / ASE out, optional `rng: numpy.random.Generator | None`
for reproducibility:

- `apply_rattle(atoms, *, displacement=0.1, max_cell_strain=0.05, rng=None) -> Atoms`
- `apply_triaxial_strain(atoms, *, max_strain=0.8, rng=None) -> Atoms`
- `apply_shear_strain(atoms, *, max_strain=0.8, rng=None) -> Atoms`

Same distributions as current code: per-site Gaussian rattle with `stdev =
displacement`, uniform `U(-max_strain, max_strain)` cell strain on the diagonal
for triaxial, full 3×3 with diagonal forced to 1 for shear. Given the same
seed, output matches current code element-wise.

### `structure/permutations.py`

```python
@pwf.as_function_node("structures", "names")
def generate_assyst_permutations(
    base_structures: list[Atoms],
    base_names: list[str],
    *,
    n_rattle: int = 5,
    n_triaxial: int = 5,
    n_shear: int = 5,
    rattle_displacement: float = 0.1,
    rattle_cell_strain: float = 0.05,
    triaxial_strain: float = 0.8,
    shear_strain: float = 0.8,
    min_dist: float = 1.0,
    core_overlap_tolerance: float = 0.2,
    max_attempts_per_perm: int = 100,
    seed: int | None = None,
) -> tuple[list[Atoms], list[str]]:
    ...
```

Wraps the three-category attempt loop from current
`get_ASSYST_deformed_structures` verbatim. Names follow
`f"{base_name}_rattle_{i}"` / `"_triax_{i}"` / `"_shear_{i}"` (matching current
scheme so downstream consumers can keep parsing).

### `structure/generate.py` (PyXtal — optional dep)

```python
@pwf.as_function_node("structures", "names")
def pyxtal_random_crystals(
    composition: dict[str, int],
    n_structures: int = 100,
    space_groups: list[int] | range = range(1, 231),
    volume_factor: float = 1.0,
    name_prefix: str = "pyxtal",
    seed: int | None = None,
) -> tuple[list[Atoms], list[str]]:
    ...
```

Lazy-imports `pyxtal`. Raises a friendly error if the `[pyxtal]` extra is not
installed.

### `physics/relaxation.py`

```python
@pwf.as_macro_node("engine_outputs", "final_structure", "converged")
def multistage_relax(
    wf,
    structure: Atoms,
    engine: Engine,
    stages: list[CalcInputMinimize] | None = None,
    stage_names: list[str] | None = None,
) -> tuple[list[EngineOutput], Atoms, bool]:
    ...
```

Default `stages`:

```python
[
    CalcInputMinimize(cell_relaxation="volume"),  # ISIF=7
    CalcInputMinimize(cell_relaxation="shape"),   # ISIF=5
    CalcInputMinimize(cell_relaxation="none"),    # ISIF=2
]
```

Default `stage_names = ["ISIF7", "ISIF5", "ISIF2"]`.

Per stage, the macro builds `sub_engine = engine` with
`EngineInput=stages[i]` (via `dataclasses.replace`) and
`working_directory=parent/stage_names[i]` (via `with_working_directory`),
then calls `calculate(structure=current, engine=sub_engine)`. The previous
stage's `final_structure` becomes the next stage's input.

The macro returns all three `EngineOutput`s — the third one's trajectory is
what ASSYST harvests.

### `analysis/collect.py`

```python
@pwf.as_function_node("structures", "energies", "names", "frame_indices")
def collect_relaxation_frames(
    engine_output: EngineOutput,
    base_name: str,
    *,
    image_selection_eV_atom_threshold: float = -1.0,
    require_converged: bool = True,
) -> tuple[list[Atoms], list[float], list[str], list[int]]:
    ...
```

Replaces `collect_structures` + `select_indices_by_threshold` from current
code. Operates on `EngineOutput` (no DataFrame interop). Threshold semantics
preserved: `-1.0` selects the last frame only; positive value selects first,
last, plus any frame whose eV/atom differs from the previously selected frame
by more than the threshold. `require_converged=True` drops frames where the
engine reports `converged=False`. The `select_indices_by_threshold` helper
algorithm is preserved verbatim inside `collect.py` as a private function.

### `analysis/export.py`

```python
@pwf.as_function_node("path")
def export_training_set(
    engine_outputs: list[EngineOutput],
    names: list[str],
    *,
    path: str = "df_ASSYST_jobs.pkl",
    format: Literal["pickle_df", "extxyz", "ase_db"] = "pickle_df",
) -> str:
    ...
```

- `pickle_df`: pandas DataFrame, **one row per frame**, with scalar columns
  `name`, `energy`, `volume`, `converged`, and object columns `structure`
  (`ase.Atoms`), `forces` (`np.ndarray`), `stress` (`np.ndarray` or `None`).
  Saved via `df.to_pickle`. This is **a new schema**, not a clone of the
  legacy `df_ASSYST_jobs.pkl` — legacy was one row per VASP job with
  list-valued trajectory cells (`structures`, `energy`, `scf_convergence` as
  lists), which is awkward for downstream MLIP training. The cleanup is
  intentional. See §10 (out of scope) for the legacy-schema rationale.
- `extxyz`: `ase.io.write(path, atoms_list, format="extxyz")` with
  energy/forces/stress attached — drop-in for MACE / GRACE / Allegro
  training.
- `ase_db`: `ase.db.connect(path)` SQLite with one row per frame.

The exporter is the **only** place schema choices are made — all upstream
nodes pass `list[EngineOutput]` around so adding a new export format never
touches the workflow body.

### `physics/assyst.py` — top-level macro

```python
@pwf.as_macro_node("training_path", "base_outputs", "perm_outputs")
def run_assyst(
    wf,
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
) -> tuple[str, list[EngineOutput], list[EngineOutput]]:
    wf.relax = multistage_relax(
        structure, relax_engine,
        stages=relax_stages, stage_names=relax_stage_names,
    )
    wf.harvest = collect_relaxation_frames(
        engine_output=wf.relax.outputs.engine_outputs[-1],
        base_name=base_name,
        image_selection_eV_atom_threshold=image_selection_eV_atom_threshold,
    )
    # Static SCF on base frames — fan-out via for_node.
    wf.base_engines = _build_subengines(static_engine, wf.harvest.outputs.names)
    wf.base_static = for_node(
        calculate,
        zip_on=("structure", "engine"),
        structure=wf.harvest.outputs.structures,
        engine=wf.base_engines,
    )
    # Permute, then static SCF on permutations.
    wf.perms = generate_assyst_permutations(
        base_structures=wf.harvest.outputs.structures,
        base_names=wf.harvest.outputs.names,
        n_rattle=n_rattle, n_triaxial=n_triaxial, n_shear=n_shear,
        rattle_displacement=rattle_displacement,
        rattle_cell_strain=rattle_cell_strain,
        triaxial_strain=triaxial_strain,
        shear_strain=shear_strain,
        min_dist=min_dist, core_overlap_tolerance=core_overlap_tolerance,
        seed=seed,
    )
    wf.perm_engines = _build_subengines(static_engine, wf.perms.outputs.names)
    wf.perm_static = for_node(
        calculate,
        zip_on=("structure", "engine"),
        structure=wf.perms.outputs.structures,
        engine=wf.perm_engines,
    )
    wf.export = export_training_set(
        engine_outputs=_concat(wf.base_static.outputs, wf.perm_static.outputs),
        names=_concat(wf.harvest.outputs.names, wf.perms.outputs.names),
        path=training_path, format=export_format,
    )
    return wf.export.outputs.path, wf.base_static.outputs, wf.perm_static.outputs
```

`_build_subengines(engine, names)` is a small `@pwf.as_function_node` helper
in `_internal/engine_fanout.py` that returns
`[engine.with_working_directory(n) for n in names]`. List comprehensions
can't run on pyiron_workflow channels directly, so this helper exists to
defer the resolution to graph-execution time. `_concat` is similarly a node
that concatenates two channel-bound lists.

## 5. VASP-equivalence

### Upstream gaps and resolutions

**Gap A — `CalcInputMinimize.relax_cell: bool` is too coarse.** Today
`VaspEngine` maps `True → ISIF=3`, `False → ISIF=2`. ISIF=7 (volume-only) and
ISIF=5 (shape-only) are unreachable.

Resolution: add `cell_relaxation: Literal["none", "volume", "shape", "full"]
= "none"` to `pyiron_workflow_atomistics.engine.inputs.CalcInputMinimize`.
Keep `relax_cell: bool` as a backwards-compat alias (`True ⇔ "full"`),
deprecation-warn on construction. Update `pyiron_workflow_vasp/_run.py`'s
ISIF mapping:

```python
ISIF_MAP = {"none": 2, "volume": 7, "shape": 5, "full": 3}
params["ISIF"] = ISIF_MAP[engine_input.cell_relaxation]
```

`ASEEngine` honors `"full"` (variable-cell via `UnitCellFilter` — already
implemented) and `"none"` (vanilla BFGS — already implemented); `"volume"`
and `"shape"` raise `NotImplementedError` (ASE has no native primitive;
ASSYST users on ASE just use `"full"`).

**Gap B — accurate-static INCAR overrides** (`KSPACING=0.25`, `EDIFF=1e-5`,
`LREAL=False`, `NSW=0`). `KSPACING` is reachable via `VaspEngine.kpoints_density`.
`EDIFF`, `LREAL`, `NSW` are not currently exposed.

Resolution: the caller owns engine construction. The ASSYST macro takes
**two engines** — `relax_engine` for stages 1–3 and `static_engine` for the
accurate static SCFs — so the user supplies, e.g.:

```python
relax_engine = VaspEngine(
    EngineInput=CalcInputMinimize(cell_relaxation="volume", max_iterations=100),
    kpoints_density=0.30, encut=400, command=vasp_cmd,
)
static_engine = VaspEngine(
    EngineInput=CalcInputStatic(),
    kpoints_density=0.25, encut=400, ediff=1e-5, lreal=False, command=vasp_cmd,
)
```

`ediff`, `lreal`, `compress_outputs`, `remove_workdir` fields are added to
`VaspEngine` as part of this work (upstream PR #2). On ASE/EMT, the same
engine can be passed twice; the `cell_relaxation` field defaults to a
sensible value for `CalcInputStatic` (NSW=0 is implicit).

### Fidelity matrix — pipeline behaviors

| Current behavior | Current location | New location |
|---|---|---|
| Convert input pymatgen → ASE Atoms | `convert_pymatgen_to_ase` node | Removed; macro accepts `ase.Atoms`. PyXtal generator emits ASE. Caller converts if needed. |
| `NSW` override for relax | `get_ionic_steps_dict` node | `CalcInputMinimize.max_iterations` (already in pwa) |
| Stage 1 ISIF=7 (volume relax) | `ISIF7_job = vasp_job(...)` + `generate_modified_incar({"ISIF":7})` | `multistage_relax` stage 0 = `CalcInputMinimize(cell_relaxation="volume")`; subdir `"ISIF7"` |
| Stage 2 ISIF=5 (shape relax) seeded by stage 1's final structure | `construct_sequential_VaspInput_from_vaspoutput_structure(ISIF7_job.outputs.vasp_output, ...)` + `vasp_job(...)` | `multistage_relax` stage 1 = `CalcInputMinimize(cell_relaxation="shape")`; macro feeds `engine_outputs[0].final_structure` into stage 1 |
| Stage 3 ISIF=2 (atoms relax) seeded by stage 2's final structure | same idiom | `multistage_relax` stage 2 = `CalcInputMinimize(cell_relaxation="none")`; subdir `"ISIF2"` |
| Harvest ISIF=2 trajectory with eV/atom threshold + SCF-conv filter | `collect_structures(df_list=[ISIF2_output], image_selection_eVatom_threshold=...)` | `collect_relaxation_frames(engine_output=relax.engine_outputs[-1], image_selection_eV_atom_threshold=..., require_converged=True)` |
| `select_indices_by_threshold` (first/last + monotone-delta picker, `-1` ⇒ last only) | helper in `workflow.py` | Private helper in `analysis/collect.py`. Algorithm preserved verbatim. |
| Accurate static INCAR (`KSPACING=0.25, EDIFF=1e-5, EDIFFG=1e-4, LREAL=False, NSW=0`) | `accurate_incar = generate_modified_incar(incar, {...})` | Caller constructs `static_engine = VaspEngine(EngineInput=CalcInputStatic(), kpoints_density=0.25, ediff=1e-5, lreal=False, ...)`. New engine knobs added in upstream PR #2. |
| Static SCF on each base frame | `for_node(vasp_job, zip_on=("vasp_input","workdir","vasp_parser_args"), ...)` | `for_node(calculate, zip_on=("structure","engine"), structure=base_frames, engine=_build_subengines(static_engine, names))` |
| Rattle: per-site Gaussian (`stdev=rattle_displacement`) + diagonal cell strain ∈ U(−`rattle_strain`, +`rattle_strain`) | `apply_rattle` in `workflow.py` | `structure/deformations.py::apply_rattle` — distributions preserved, `rng` parameterised |
| Triaxial strain: diagonal strain ∈ U(−`triaxial_strain`, +`triaxial_strain`) | `apply_triaxial_strain` | `structure/deformations.py::apply_triaxial_strain` |
| Shear strain: full 3×3 strain ∈ U(−`shear_strain`, +`shear_strain`) with diagonal forced to 1 | `apply_shear_strain` | `structure/deformations.py::apply_shear_strain` |
| Attempt loop, `max_attempts=100`, validity via `is_valid_structure` (min-dist + RCORE core-overlap) | `get_ASSYST_deformed_structures` inner loops | `structure/permutations.py::generate_assyst_permutations`. Loop structure, retry count, and category ordering (rattle → triax → shear) preserved verbatim. |
| Permutation name scheme `f"{base}_rattle_{i}"` / `_triax_{i}` / `_shear_{i}` | inline in `get_ASSYST_deformed_structures` | `generate_assyst_permutations` — same f-strings |
| RCORE table (POTCAR core radii × Bohr→Å) | `structure_filter_utils.py::RCORE` | `structure/filters.py::RCORE` — bytes-identical |
| `_element_wise_dist` minimum pair distance via pymatgen `get_all_neighbors(r=5.0)` | `structure_filter_utils.py` | `structure/filters.py` — internal helper, pymatgen path preserved |
| `filter_distance_by_species` with `1 − core_overlap_tolerance` rule | `structure_filter_utils.py` | `structure/filters.py` |
| Static SCF on permutations | `for_node(vasp_job, ...)` | `for_node(calculate, ...)` with `_build_subengines(static_engine, perm_names)` |
| Combine base+perm DataFrames | `pwf.api.inputs_to_list(2, ...)` + `get_concat_df` | `export_training_set` concatenates `EngineOutput` lists and assembles the DataFrame inside the exporter |
| Save as pickle `df_ASSYST_jobs.pkl` | `save_df` | `export_training_set(format="pickle_df", path="df_ASSYST_jobs.pkl")` — note: schema is intentionally cleaner (one row per frame, scalar cells). See §4 export.py and §10. |
| `compress` / `compressed_file_in_dir` / `remove_calc_dir` knobs | passed through every `vasp_job` | Engine-level concern. `VaspEngine.compress_outputs` / `.remove_workdir` added in upstream PR #2. |
| `vasp_command` arg | passed through every `vasp_job` | `VaspEngine.command` (already exists) |
| `vasp_parser_function` arg | passed through every `vasp_job` | Internal to VaspEngine. Not exposed on the ASSYST macro. |
| `potcar_paths` arg | passed through every `vasp_job` | `VaspEngine.potcar_config_file` (already exists) |
| Top-level re-exports of `run_ASSYST_on_structure` etc. | `__init__.py` | Dropped (clean break). `__init__.py` exposes only `__version__` with lazy `__getattr__`. |

### Parameter renames at the public surface

| Old parameter | New parameter | Reason |
|---|---|---|
| `incar` + `potcar_paths` + `vasp_command` + `compress_dirs` + `compressed_file_in_dir` + `remove_calc_dirs` + `vasp_parser_function` | `relax_engine: Engine`, `static_engine: Engine` | Engine abstraction |
| `ionic_steps` | `relax_stages[i].max_iterations` | Already on `CalcInputMinimize`; per-stage now |
| `n_stretch_permutations` | `n_triaxial`, `n_shear` (split — old name controlled **both**) | Disambiguation matches the actual loop structure |
| `n_rattle_permutations` | `n_rattle` | Tidier |
| `rattle_strain` | `rattle_cell_strain` | Distinguish from the position-rattle displacement |
| `image_selection_eVatom_threshold` | `image_selection_eV_atom_threshold` | PEP 8 |
| `job_basename` | `base_name` | Plays nicer with `engine.with_working_directory(...)` |
| `train_df_filename` | `training_path` | Generic over output formats |

The default `pickle_df` schema differs from the legacy `df_ASSYST_jobs.pkl`
schema (one frame per row vs one job per row with list-valued cells) — this
is a deliberate cleanup, not a regression. Downstream consumers of the legacy
pickle will need a one-time migration; the VASP-equivalence test in §6
asserts content equivalence (same (name, energy, structure) triples up to
ordering), not schema equivalence.

## 6. Testing strategy

### Unit tests (`tests/unit/`, fast, no calculator)

Mirror the source tree one-to-one.

- `tests/unit/structure/test_filters.py` — `RCORE` table is byte-identical to
  the legacy table (read both modules, assert equal); `get_minimum_distance`
  on a hand-built Atoms; `is_valid_structure` true/false on a tight pair vs a
  relaxed pair.
- `tests/unit/structure/test_deformations.py` — golden-value tests with
  `rng = np.random.default_rng(42)`: after each deformation, assert
  positions / cell against frozen reference arrays. This is the equivalence
  guarantor for the deformations.
- `tests/unit/structure/test_permutations.py` — tiny FCC-Cu seed,
  `generate_assyst_permutations(n_rattle=2, n_triaxial=2, n_shear=2)` →
  `len(structures) == 6`; names match expected pattern; every structure
  passes `is_valid_structure`; re-run with the same seed yields bit-identical
  output.
- `tests/unit/structure/test_generate.py` — skipped if `pyxtal` is not
  installed; otherwise a small smoke test on a 1-element composition.
- `tests/unit/analysis/test_collect.py` — `select_indices_by_threshold`
  golden tests (legacy function copied into the test file as reference);
  `collect_relaxation_frames` with synthetic `EngineOutput` fixtures —
  threshold `-1` returns the last frame only; `require_converged=True` drops
  non-converged frames.
- `tests/unit/analysis/test_export.py` — round-trip: synthetic
  `EngineOutput`s → `pickle_df` → read back → column names and dtypes match
  the legacy schema. `extxyz` round-trip via `ase.io.read`.

### Integration tests (ASE+EMT, CI-runnable)

- `tests/integration/test_multistage_relax_emt.py` — Cu FCC + ASEEngine+EMT,
  default stages → assert final cell is closer to equilibrium than input;
  3 `EngineOutput`s returned; final stage's trajectory is non-empty.
- `tests/integration/test_run_assyst_emt.py` — Cu FCC + ASEEngine+EMT,
  `n_rattle = n_triaxial = n_shear = 1`,
  `image_selection_eV_atom_threshold = -1` → produces a pickle, frame count
  `= 1 base + 3 perms = 4`, all converged, schema matches the legacy
  schema.

### VASP-equivalence test (gated `@pytest.mark.vasp`)

`tests/integration/test_vasp_equivalence.py` — skipped unless `VASP_TEST=1`
is set and `vasp_std` is on `$PATH`. Runs the **legacy**
`run_ASSYST_on_structure` (pinned in a `legacy/` test-only checkout) and the
new `run_assyst` on identical inputs (2-atom Fe BCC, same INCAR, same RNG
seed). Asserts:

- workdir tree has the same `ISIF7/ISIF5/ISIF2/` layout
- both pickles contain the same **set of (name, structure) pairs** — for
  the new schema this means iterating rows; for the legacy schema it means
  exploding list-valued cells into per-frame entries first
- corresponding energies agree to within `1e-3 eV/atom`
- corresponding forces agree to within `1e-3 eV/Å` per component
- the row/frame *count* matches after exploding the legacy schema

This is the ground-truth equivalence check. The legacy module is held in a
test-only `_legacy_assyst/` directory in the repo (frozen snapshot of the
current `workflow.py` + `structure_filter_utils.py`), imported via a
`conftest.py` shim — it does not pollute the runtime package.

### Engine conformance

Not applicable here — this module is a consumer of the Engine, not a
provider. The EMT integration test path doubles as Engine-usage validation;
the VASP-equivalence test validates both engines via the same macro.

## 7. Upstream PRs

Three PRs land in sequence. The implementation plan (`writing-plans` step)
will track them as explicit prerequisites.

1. **`pyiron_workflow_atomistics` — `feat/cell-relaxation-enum`**
   Additive only. Add `CalcInputMinimize.cell_relaxation:
   Literal["none","volume","shape","full"] = "none"`. Keep `relax_cell:
   bool` as a property alias with a `DeprecationWarning` on access. Update
   `ASEEngine` to honor the new field (`"full"` already works via
   `UnitCellFilter`; `"none"` already works; `"volume"`/`"shape"` raise
   `NotImplementedError` with a clear message). Add unit tests.

2. **`pyiron_workflow_vasp` — `feat/vasp-engine-isif-mapping`**
   - Replace `params["ISIF"] = 3 if engine_input.relax_cell else 2` in
     `_run.py` with the `ISIF_MAP[engine_input.cell_relaxation]` lookup.
   - Add `ediff: float | None = None`, `lreal: bool | None = None`,
     `compress_outputs: bool = False`, `remove_workdir: bool = False`
     fields to `VaspEngine`. Pass-through to INCAR / post-run cleanup.
   - Bump pinned `pyiron_workflow_atomistics` dep to the version from PR #1.
   - Update conformance tests.

3. **`pyiron_workflow_assyst` — `feat/engine-agnostic-rewrite`** (this spec)
   - Bumps version to `0.2.0a0`.
   - Pinned deps on the new pwa + pwv from PRs #1 and #2.

PRs #1 and #2 can be developed in parallel; PR #2 merges after PR #1 (deps).
PR #3 blocks on both.

## 8. Delivery plan

| Branch | Repo | Contents | Blocks |
|---|---|---|---|
| `feat/cell-relaxation-enum` | `pyiron_workflow_atomistics` | `CalcInputMinimize.cell_relaxation` field + `relax_cell` shim + `ASEEngine` handling + tests | — |
| `feat/vasp-engine-isif-mapping` | `pyiron_workflow_vasp` | ISIF map + `ediff` / `lreal` / cleanup knobs + tests + dep bump | PR #1 merged |
| `feat/engine-agnostic-rewrite` | `pyiron_workflow_assyst` | full module rewrite per this spec; integration tests use EMT for CI; VASP-equivalence test gated | PRs #1 & #2 merged |

## 9. Acceptance criteria

The work is done when:

1. All three PRs are merged.
2. `pip install pyiron_workflow_assyst[pyxtal,vasp]` succeeds in a clean env
   (verified against the `[[pwa-install-quirk]]` lessons — lazy `__getattr__`
   in `__init__.py`, no eager submodule imports at top level).
3. `pytest tests/unit tests/integration` (the EMT path) passes on CI.
4. The VASP-equivalence test passes on a machine with VASP available, with
   the same input → same set of (name, structure) frames and ≤ 1e-3 eV/atom
   energy drift vs the legacy implementation. (The pickle schema itself is
   intentionally different — equivalence is content-level, not schema-level
   — see §6 and §4 export.py.)
5. The fidelity matrix in §5 is reproduced by the actual code — every row
   of the right column corresponds to real code in the new tree.

## 10. Out of scope (deferred)

- CalPhy / free-energy integration (validate trained MLIPs via thermodynamic
  integration). Would build on `pyiron_workflow_atomistics.physics.free_energy`
  and is a follow-on module.
- MD support in the ASSYST pipeline (`CalcInputMD` is rejected by
  `VaspEngine` today, and ASSYST methodology is relax-and-perturb, not MD).
- LAMMPS-backed ASSYST runs (requires a `LammpsEngine` that doesn't exist
  yet).
- Backwards-compatibility shims for the old top-level API — this is a clean
  break to `0.2.0a0`.
- A `pickle_df_legacy` export format reproducing the old one-row-per-job
  list-valued schema. If a downstream consumer needs it later, add it as an
  extra `Literal` value on `export_training_set.format` — no other code
  changes required (the export module is the only place schemas are
  materialised).
