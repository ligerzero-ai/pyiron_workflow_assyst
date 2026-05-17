# pyiron_workflow_assyst

Engine-agnostic ASSYST (Automated Symmetric Structure Training) workflows
for `pyiron_workflow`. Generates training datasets for machine-learning
interatomic potentials by relaxing seed crystals, harvesting frames from
the relaxation trajectories, perturbing them with rattle / triaxial /
shear deformations, and running accurate single-points on the result.

ASSYST methodology:

> Poul, M., Huber, L., & Neugebauer, J. (2024). Automated Generation of
> Structure Datasets for Machine Learning Potentials and Alloys.
> [Research Square](https://www.researchsquare.com/article/rs-4732459/v1)

## Design

The package follows `pyiron_workflow_atomistics` design principles:

- **Engine-agnostic.** Physics workflows take a
  `pyiron_workflow_atomistics.engine.Engine` and call `calculate()`. Drop
  in `VaspEngine`, `ASEEngine`, or any future backend without touching the
  workflow body.
- **Physics-level inputs.** `CalcInputMinimize.cell_relaxation`
  (`"none"` / `"volume"` / `"shape"` / `"full"`) replaces VASP-specific
  `ISIF` numbers; each engine translates to its native parameters.
- **`EngineOutput` is the canonical result type.** Post-processing
  (`collect_relaxation_frames`, `export_training_set`) consumes
  `EngineOutput`, never engine-specific outputs.
- **Subpackage layering.** `structure/` (engine-agnostic builders +
  filters + deformations + permutations + PyXtal), `physics/`
  (`multistage_relax`, `run_assyst` macros), `analysis/` (frame
  collection + training-set export), `testing/` (shared fixtures).

## Installation

```bash
pip install pyiron_workflow_assyst[vasp]      # with VaspEngine
pip install pyiron_workflow_assyst[pyxtal]    # with random-crystal generator
pip install pyiron_workflow_assyst[vasp,pyxtal,test]
```

Development install:

```bash
git clone git@github.com:ligerzero-ai/pyiron_workflow_assyst.git
cd pyiron_workflow_assyst
pip install -e ".[vasp,pyxtal,test]"
```

## Usage

`run_assyst` takes two `Engine` instances — one for the relaxation cascade
and one for the accurate single-point evaluations. This lets you use a
cheap setup for relaxation and a tighter one for the final training-set
SCFs without writing two macros:

```python
from ase.build import bulk
from pyiron_workflow_atomistics.engine import CalcInputMinimize, CalcInputStatic
from pyiron_workflow_vasp.engine import VaspEngine
from pyiron_workflow_assyst.physics.assyst import run_assyst

structure = bulk("Fe", "bcc", a=2.86, cubic=True)

relax_engine = VaspEngine(
    EngineInput=CalcInputMinimize(cell_relaxation="volume", max_iterations=100),
    working_directory="./run",
    potcar_config_file="/path/to/POTCAR-config.yaml",
    encut=400,
    kpoints_density=0.30,
    command="vasp_std",
)
static_engine = VaspEngine(
    EngineInput=CalcInputStatic(),
    working_directory="./run",
    potcar_config_file="/path/to/POTCAR-config.yaml",
    encut=400,
    kpoints_density=0.25,
    ediff=1e-5,
    lreal=False,
    command="vasp_std",
)

macro = run_assyst(
    structure=structure,
    relax_engine=relax_engine,
    static_engine=static_engine,
    base_name="fe0",
    n_rattle=5,
    n_triaxial=5,
    n_shear=5,
)
macro.run()
# Training-set DataFrame at ./df_ASSYST_jobs.pkl
```

For non-VASP backends (ASE / EMT for tests, future MACE / GRACE / LAMMPS
engines), swap `VaspEngine` for `ASEEngine` and provide the appropriate
calculator. The macro body never changes.

`run_assyst`'s default relax cascade is the ASSYST three-stage chain
(`cell_relaxation="volume"` → `"shape"` → `"none"`, equivalent to ISIF=7
→ ISIF=5 → ISIF=2 on VASP). Override `relax_stages` for any other
sequence — e.g. a single full-cell relax for ASE:

```python
relax_stages = [CalcInputMinimize(cell_relaxation="full", max_iterations=100)]
relax_stage_names = ["fullrelax"]
```

## Dependencies

- `numpy`, `pandas`, `ase`, `pymatgen`, `tqdm`
- `pyiron-workflow`
- `pyiron_workflow_atomistics` — provides `Engine`, `calculate`, `EngineOutput`, `ASEEngine`
- Optional: `pyiron_workflow_vasp` (`[vasp]` extra) — provides `VaspEngine`
- Optional: `pyxtal` (`[pyxtal]` extra) — random crystal generation
- Optional: `pytest`, `nbformat`, `nbclient` (`[test]` extra)

## Contributing

Issues and PRs welcome. The codebase mirrors
[`pyiron_workflow_atomistics`](https://github.com/pyiron/pyiron_workflow_atomistics)'s
layering — please keep `structure/` engine-agnostic and route everything
that talks to a calculator through the `Engine` Protocol.

## License

BSD-3-Clause.

## Citation

If you use this package in your research, please cite:

```bibtex
@article{poul2024automated,
  title={Automated Generation of Structure Datasets for Machine Learning Potentials and Alloys},
  author={Poul, Marvin and Huber, Liam and Neugebauer, J{\"o}rg},
  journal={Research Square},
  year={2024},
  doi={10.21203/rs.3.rs-4732459/v1}
}
```
