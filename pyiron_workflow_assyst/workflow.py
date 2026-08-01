"""End-to-end ASSYST DFT workflow: ISIF7->5->2 relaxation, accurate statics on
the relaxed images, and accurate statics on rattle/triaxial/shear permutations
of those images.

All work directories are absolute paths derived from ``job_name`` -- nothing
here calls ``os.chdir``; ``run_shell`` (in ``pyiron_workflow_vasp.generic``)
scopes each VASP invocation with subprocess's ``cwd`` instead.
"""

import os

import pandas as pd
import flowrep as fr
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar

# VASP-specific imports
try:
    from pyiron_workflow_vasp.vasp import (
        vasp_job,
        generate_vasp_input,
        generate_modified_incar,
        construct_sequential_vasp_input,
    )
except ImportError:
    raise ImportError(
        "pyiron_workflow_vasp is required for this package. "
        "Please install it using: pip install pyiron_workflow_vasp"
    )

# Local imports
from .perturb import get_ASSYST_deformed_structures


def select_indices_by_threshold(array, threshold):
    """
    Selects the indices of the first, last, and all values in the array
    that differ from the previous selected value by more than a specified threshold.

    Parameters:
    - array (iterable): The input array or list of values.
    - threshold (float): The minimum difference required to select a value.

    Returns:
    - list: Indices of selected values.
    """
    if len(array) == 0:
        return []

    # Initialize the list with the first index
    selected_indices = [0]

    # Iterate through the array and select indices based on the threshold
    for i in range(1, len(array)):
        if abs(array[i] - array[selected_indices[-1]]) > threshold:
            selected_indices.append(i)

    # Ensure the last index is included
    if len(array) - 1 not in selected_indices:
        selected_indices.append(len(array) - 1)

    return selected_indices


@fr.atomic("structures", "names", "energies")
def collect_structures(vasp_output, job_name, image_selection_eVatom_threshold=-1):
    """Select relaxation images from a parsed VASP directory.

    Takes the MOST RECENT row (``.iloc[-1]``): parse_vasp_directory returns one
    row per OUTCAR plus one per custodian error archive, ascending by start
    time, so the first row is the oldest - the crashed run in a restart chain.

    ``image_selection_eVatom_threshold=-1`` keeps only the final image.
    Images whose SCF did not converge are dropped.
    """
    row = vasp_output.iloc[-1]
    json_blobs = row["structures"]
    energies = list(row["energy"])
    scf = list(row["scf_convergence"])

    first = Structure.from_str(str(json_blobs[0]), fmt="json")
    n_atoms = len(first)
    ev_atom = [e / n_atoms for e in energies]

    if image_selection_eVatom_threshold == -1:
        selected = [len(ev_atom) - 1]
    else:
        selected = select_indices_by_threshold(
            ev_atom, threshold=image_selection_eVatom_threshold
        )

    structures, names, kept_energies = [], [], []
    for i in selected:
        if not scf[i]:
            continue
        structure = Structure.from_str(str(json_blobs[i]), fmt="json")
        structures.append(AseAtomsAdaptor.get_atoms(structure))
        names.append(f"{job_name}_accur_relaxstep{i}")
        kept_energies.append(energies[i])
    return structures, names, kept_energies


@fr.atomic("incar")
def set_ionic_steps(incar, ionic_steps):
    modified = Incar.from_dict(dict(incar))
    modified["NSW"] = ionic_steps
    return modified


@fr.atomic("path")
def join_path(base, leaf):
    return os.path.join(base, leaf)


@fr.atomic("parser_args")
def build_parser_args(directory):
    """Build the ``{"directory": ...}`` kwargs a custom ``vasp_parser_function``
    needs.

    ``vasp_job``'s ``parse_vasp_output`` only auto-supplies ``workdir`` to the
    *default* parser (``parse_vasp_directory``); a caller-supplied function is
    invoked as ``function(**(vasp_parser_args or {}))`` with no implicit
    argument, so a custom parser needs its target directory threaded through
    explicitly. A `@fr.workflow` body cannot spell ``{"directory": path}``
    inline (dict literals may only hold constants, not the dynamic ``path``
    symbol), hence this one-line node. Harmless when the default parser is
    used, since that branch never reads ``parser_args``.
    """
    return {"directory": directory}


@fr.atomic("df")
def concat_and_save(base_results, perm_results, filename):
    frames = [f for f in list(base_results) + list(perm_results) if f is not None]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_pickle(filename)
    return combined


@fr.workflow("train_df")
def run_ASSYST_on_structure(
    structure,
    incar,
    potcar_paths,
    job_name,
    vasp_command,
    ionic_steps=100,
    n_stretch_permutations=2,
    n_rattle_permutations=2,
    image_selection_eVatom_threshold=-1,
    shear_strain=0.8,
    triaxial_strain=0.8,
    rattle_displacement=0.1,
    rattle_strain=0.05,
    core_overlap_tolerance=0.3,
    min_dist_backend="neighbor_list",
    compress_dirs=True,
    compressed_file_in_dir=False,
    remove_calc_dirs=True,
    train_df_filename="df_ASSYST_jobs.pkl",
    seed=None,
    vasp_parser_function=None,
):
    """Relax a structure through ISIF 7 -> 5 -> 2, then run accurate statics on
    the converged images and on rattle/triaxial/shear permutations of them.

    All work directories are absolute paths derived from ``job_name``; nothing
    depends on the process working directory.
    """
    relax_incar = set_ionic_steps(incar, ionic_steps)

    incar7 = generate_modified_incar(relax_incar, {"ISIF": 7})
    input7 = generate_vasp_input(structure, incar7, potcar_paths)
    path7 = join_path(job_name, "ISIF7")
    parser_args7 = build_parser_args(path7)
    out7, conv7 = vasp_job(
        path7, input7, vasp_command, None, compress_dirs,
        compressed_file_in_dir, remove_calc_dirs, vasp_parser_function, parser_args7,
    )

    incar5 = generate_modified_incar(relax_incar, {"ISIF": 5})
    input5 = construct_sequential_vasp_input(out7, incar5, potcar_paths)
    path5 = join_path(job_name, "ISIF5")
    parser_args5 = build_parser_args(path5)
    out5, conv5 = vasp_job(
        path5, input5, vasp_command, None, compress_dirs,
        compressed_file_in_dir, remove_calc_dirs, vasp_parser_function, parser_args5,
    )

    incar2 = generate_modified_incar(relax_incar, {"ISIF": 2})
    input2 = construct_sequential_vasp_input(out5, incar2, potcar_paths)
    path2 = join_path(job_name, "ISIF2")
    parser_args2 = build_parser_args(path2)
    out2, conv2 = vasp_job(
        path2, input2, vasp_command, None, compress_dirs,
        compressed_file_in_dir, remove_calc_dirs, vasp_parser_function, parser_args2,
    )

    base_structures, base_names, base_energies = collect_structures(
        out2, "ISIF2", image_selection_eVatom_threshold
    )

    accurate_incar = generate_modified_incar(
        incar,
        {"KSPACING": 0.25, "EDIFFG": 1e-4, "EDIFF": 1e-5, "LREAL": False, "NSW": 0},
    )

    base_results = []
    for base_structure, base_name in zip(base_structures, base_names):
        base_input = generate_vasp_input(base_structure, accurate_incar, potcar_paths)
        base_path = join_path(job_name, base_name)
        base_parser_args = build_parser_args(base_path)
        base_out, base_conv = vasp_job(
            base_path, base_input, vasp_command, None, compress_dirs,
            compressed_file_in_dir, remove_calc_dirs, vasp_parser_function, base_parser_args,
        )
        base_results.append(base_out)

    perm_structures, perm_names = get_ASSYST_deformed_structures(
        base_structures,
        base_names,
        n_stretch_permutations,
        n_rattle_permutations,
        shear_strain,
        triaxial_strain,
        rattle_displacement,
        rattle_strain,
        1.0,
        core_overlap_tolerance,
        potcar_paths,
        min_dist_backend,
        100,
        seed,
    )

    perm_results = []
    for perm_structure, perm_name in zip(perm_structures, perm_names):
        perm_input = generate_vasp_input(perm_structure, accurate_incar, potcar_paths)
        perm_path = join_path(job_name, perm_name)
        perm_parser_args = build_parser_args(perm_path)
        perm_out, perm_conv = vasp_job(
            perm_path, perm_input, vasp_command, None, compress_dirs,
            compressed_file_in_dir, remove_calc_dirs, vasp_parser_function, perm_parser_args,
        )
        perm_results.append(perm_out)

    train_df = concat_and_save(base_results, perm_results, train_df_filename)
    return train_df
