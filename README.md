# pyiron_workflow_assyst

A workflow package for ASSYST (Automated Small SYmmetric Structure Training) using VASP calculations, integrated with pyiron_workflow.

## Overview

This package implements the ASSYST methodology as described in:

> Poul, M., Huber, L., & Neugebauer, J. (2024). Automated Generation of Structure Datasets for Machine Learning Potentials and Alloys. [Research Square](https://www.researchsquare.com/article/rs-4732459/v1)

ASSYST is a strategy for generating unbiased and systematically extendable training data for machine learning interatomic potentials (MLIP) for multicomponent alloys. It explores the full space of random crystal structures with space groups, facilitating the construction of training sets for MLIPs in an automatic way without prior knowledge of the material.

## Features

- Automated generation of training structures for MLIPs
- Systematic exploration of crystal structure space
- Integration with VASP for DFT calculations
- Workflow-based implementation using pyiron_workflow
- Support for multicomponent alloys
- Structure validation and filtering

## Installation

```bash
pip install pyiron_workflow_assyst
```

For development installation:

```bash
git clone https://github.com/pyiron/pyiron_workflow_assyst.git
cd pyiron_workflow_assyst
pip install -e .
```

## Dependencies

- numpy>=1.20.0
- pandas>=1.3.0
- pymatgen>=2023.0.0
- ase>=3.22.0
- pyiron_workflow>=0.19,<0.20
- flowrep>=0.6,<0.7
- pyiron_workflow_vasp>=0.2.0

## Usage

`run_ASSYST_on_structure` is a [`flowrep`](https://github.com/pyiron/flowrep)
`@fr.workflow` function, not a callable macro object -- under
`pyiron_workflow>=0.19`, it must be wrapped with `pyiron_workflow.node(...)`
and run with keyword arguments, not called or `.run()`-ed directly:

```python
import pyiron_workflow as pwf
from pymatgen.io.vasp.inputs import Incar
from pyiron_workflow_assyst import run_ASSYST_on_structure

incar = Incar.from_dict({"ENCUT": 400, "ISIF": 7, "NSW": 100, ...})

node = pwf.node(run_ASSYST_on_structure)
run = node.run(
    structure=your_structure,      # pymatgen Structure or ase Atoms
    incar=incar,
    potcar_paths=None,             # or an explicit list of POTCAR paths
    job_name="/absolute/path/to/job_dir",
    vasp_command="mpiexec -n 40 vasp_std",
    ionic_steps=100,
    n_stretch_permutations=2,
    n_rattle_permutations=2,
    rattle_displacement=0.1,
)

train_df = run.outputs.train_df  # pandas.DataFrame of accurate-statics results
```

See `pyiron_workflow_assyst/example.py` for a complete, runnable example, and
`tests/test_workflow.py` for the full set of keyword arguments
`run_ASSYST_on_structure` accepts.

## Documentation

For detailed documentation, please refer to the [documentation](https://github.com/pyiron/pyiron_workflow_assyst).

## Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

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

## Contact

For questions or support, please contact the pyiron team at pyiron@mpie.de 