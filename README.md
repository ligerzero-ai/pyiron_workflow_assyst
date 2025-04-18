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
- pyiron_workflow>=0.1.0
- pyiron_vasp>=0.1.0
- ase>=3.22.0
- pyiron_workflow_vasp

## Usage

```python
import pyiron_workflow_assyst as pwfa

# Create a workflow for ASSYST structure generation
workflow = pwfa.create_assyst_workflow(
    base_structure=your_structure,
    n_structures=100,
    max_strain=0.8,
    rattle_displacement=0.1
)

# Run the workflow
workflow.run()
```

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