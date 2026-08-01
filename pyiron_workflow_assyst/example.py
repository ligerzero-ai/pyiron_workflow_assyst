from ase.build import bulk
import pyiron_workflow as pwf
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar
from pyiron_workflow_assyst.workflow import run_ASSYST_on_structure


vasp_command="module load vasp; module load intel/19.1.0 impi/2019.6; unset I_MPI_HYDRA_BOOTSTRAP; unset I_MPI_PMI_LIBRARY; mpiexec -n 40 vasp_std"
potcar_paths = ["/cmmc/u/hmai/vasp_potentials_54/Fe_sv/POTCAR"]
bulk_Fe = bulk("Fe", cubic=True, a=2.7)
bulk_Fe = AseAtomsAdaptor().get_structure(bulk_Fe)
bulk_Fe.perturb(0.1)
structure_folder = "/cmmc/ptmp/hmai/ASSYST_testrun/struct_pyxtal_4"

incar = Incar.from_dict({
    "ALGO": "Fast",
    "AMIX": 0.01,
    "AMIX_MAG": 0.1,
    "BMIX": 0.0001,
    "BMIX_MAG": 0.0001,
    "EDIFF": 1e-05,
    "EDIFFG": -0.01,
    "ENCUT": 400,
    "GGA": "Pe",
    "IBRION": 2,
    "ISIF": 7,
    "ISMEAR": 1,
    "ISPIN": 2,
    "ISTART": 0,
    "KPAR": 2,
    "LORBIT": 10,
    "LPLANE": False,
    "LREAL": False,
    "MAGMOM": "20*3.0 1*-0.01 27*3.0",
    "NCORE": 4,
    "NELM": 120,
    "NSIM": 1,
    "NSW": 100,
    "PREC": "Accurate",
    "SIGMA": 0.2,
    "SYSTEM": "He-20-d-2.4"
})
incar["MAGMOM"] = "2*3"

# `run_ASSYST_on_structure` is a plain @flowrep.workflow function, not a
# pyiron_workflow 0.11-style macro: calling it directly (or calling `.run()`
# on its return value) would execute its body eagerly instead of building a
# graph. In 0.19 it must be wrapped as a node first, then run with keyword
# arguments -- see also tests/test_workflow.py, which exercises exactly this
# pattern end-to-end.
node = pwf.node(run_ASSYST_on_structure)
run = node.run(
    structure=bulk_Fe,
    incar=incar,
    potcar_paths=potcar_paths,
    ionic_steps=100,
    n_stretch_permutations=2,
    n_rattle_permutations=2,
    shear_strain=0.8,
    triaxial_strain=0.8,
    rattle_displacement=0.1,
    rattle_strain=0.05,
    job_name=structure_folder,
    vasp_command=vasp_command,
)

# `run.outputs.train_df` is the pandas.DataFrame of accurate-statics results
# (also written to `train_df_filename`, which defaults to a CWD-relative
# "df_ASSYST_jobs.pkl" -- pass an absolute, job-specific path for concurrent
# use; see the run_ASSYST_on_structure docstring).
train_df = run.outputs.train_df
print(train_df)
