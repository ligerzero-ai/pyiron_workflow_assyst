"""End-to-end ASSYST DFT workflow: ISIF7->5->2 relaxation, accurate statics on
the relaxed images, and accurate statics on rattle/triaxial/shear permutations
of those images.

All work directories are absolute paths derived from ``job_name`` -- nothing
here calls ``os.chdir``; ``run_shell`` (in ``pyiron_workflow_vasp.generic``)
scopes each VASP invocation with subprocess's ``cwd`` instead.

KNOWN UPSTREAM BUG (temporary workaround in this file -- see ``unwrap_singleton``):
``pyiron_workflow==0.19.0``'s ``ForEach``/``Transform1toN`` machinery mishandles
zipped/nested for-loop iterables whose length is SMALL, in two distinct ways
depending on the exact length (length 2+ is handled correctly in all cases):

  n=1 -> loop body receives {'type': 'tuple', 'value': (0,), 'name': ('n0',)}
         (elements silently wrapped in a spurious 1-tuple). Symptom, e.g. in
         ``join_path``: ``TypeError: join() argument must be str, bytes, or
         os.PathLike object, not 'tuple'``.
  n=2 -> loop body receives {'type': 'int',   'value': 0,    'name': 'n0'}
         (correct)
  n=0 -> the for-node scatter machinery raises ``IndexError: list index out
         of range`` deep inside flowrep/pyiron_workflow internals, with no
         mention of which loop, which port, or why -- confirmed with a
         minimal reproducer independent of this package (an ``@fr.workflow``
         with a single ``for x in xs: ...; results.append(...)`` loop called
         with ``xs=[]``).

Reported upstream: https://github.com/pyiron/pyiron_workflow/issues/915
("ForEach: zip loops over length-1 iterables wrap elements in a spurious
1-tuple (length-0 raises IndexError)"). Check that issue for the fix status
before touching anything below.

The n=1 case matters here because ``image_selection_eVatom_threshold``
defaults to -1 (keep only the final relaxation image), so ``collect_structures``
routinely returns exactly ONE base structure -- the normal production path,
not an edge case. ``unwrap_singleton`` is applied to every iterated loop
variable in both for-loops in ``run_ASSYST_on_structure`` as an explicit,
visible workaround for the n=1 case. Delete ``unwrap_singleton`` and its call
sites once this is fixed upstream in ``pyiron_workflow`` (tracked at the
issue above); re-run
``tests/test_workflow.py::test_assyst_graph_runs_end_to_end_default_threshold``
and ``::test_assyst_graph_survives_singleton_permutation_loop`` with the
workaround removed to confirm the fix landed.

The n=0 case is NOT fixable from this file (there is no element to unwrap --
the loop body never runs at all), so instead both for-loops are guarded
against an empty input before they run: ``collect_structures`` raises a clear
``NoConvergedImagesError`` for the first (base-image) loop, and
``require_permutations`` raises ``NoPermutationsGeneratedError`` for the
second (permutation) loop, if ``get_ASSYST_deformed_structures`` returns zero
permutations (e.g. ``n_rattle_permutations=0, n_stretch_permutations=0``).
Either turns the opaque ``IndexError`` above into an actionable error instead
of a graph crash.
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


class NoPermutationsGeneratedError(RuntimeError):
    """Raised by ``run_ASSYST_on_structure`` when
    ``get_ASSYST_deformed_structures`` returns zero permutations.

    ``n_rattle_permutations=0, n_stretch_permutations=0`` is a legitimate
    "relax and statics only, no permutations" configuration in principle
    (and any combination of permutation counts can also bottom out at zero
    if every generation attempt fails the validity filter). But an empty
    ``perm_structures``/``perm_names`` pair would otherwise reach the
    permutation for-loop below and hit the n=0 sibling of the upstream
    ``pyiron_workflow==0.19.0`` ``ForEach`` bug documented at the top of
    this module: an opaque ``IndexError: list index out of range`` with no
    mention of permutations or which loop. That n=0 case is NOT fixable
    from this file (there is no element to unwrap -- the loop body never
    runs at all; see the module docstring and
    https://github.com/pyiron/pyiron_workflow/issues/915), so until it is
    fixed upstream, this is raised here instead, with a clear message,
    turning a silent crash into an actionable one.
    """


class NoConvergedImagesError(RuntimeError):
    """Raised by ``collect_structures`` when every candidate image fails the
    SCF-convergence filter, leaving nothing to return.

    Returning empty ``structures``/``names`` lists instead would feed a
    length-0 iterable into the next for-loop in ``run_ASSYST_on_structure``,
    which hits the n=0 sibling of the upstream ``pyiron_workflow==0.19.0``
    ``ForEach`` bug documented at the top of this module: an opaque
    ``IndexError: list index out of range`` with no mention of SCF, images,
    or which loop. Raising here, with the job name and image counts, turns
    that into an immediately actionable error instead.
    """


@fr.atomic("structures", "names", "energies")
def collect_structures(vasp_output, job_name, image_selection_eVatom_threshold=-1):
    """Select relaxation images from a parsed VASP directory.

    Takes the MOST RECENT row (``.iloc[-1]``): parse_vasp_directory returns one
    row per OUTCAR plus one per custodian error archive, ascending by start
    time, so the first row is the oldest - the crashed run in a restart chain.

    ``image_selection_eVatom_threshold=-1`` keeps only the final image.
    Images whose SCF did not converge are dropped; if that leaves nothing,
    raises ``NoConvergedImagesError`` (see its docstring for why).
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

    if not structures:
        n_examined = len(selected)
        n_rejected = sum(1 for i in selected if not scf[i])
        raise NoConvergedImagesError(
            f"collect_structures({job_name!r}): no SCF-converged image found; "
            f"examined {n_examined} image(s), {n_rejected} rejected for "
            f"non-convergence."
        )
    return structures, names, kept_energies


@fr.atomic("incar")
def set_ionic_steps(incar, ionic_steps):
    modified = Incar.from_dict(dict(incar))
    modified["NSW"] = ionic_steps
    return modified


@fr.atomic("incar")
def strip_isif(incar):
    """Drop any ``ISIF`` tag left over from the caller's raw INCAR.

    ``generate_modified_incar`` only sets keys, it never removes ones the
    caller didn't ask to override, so an ``ISIF`` carried in over the raw
    INCAR (as every real campaign's does -- it drives the ISIF7 stage)
    would otherwise persist unchanged into the accurate-statics INCAR too.
    With ``NSW=0`` that ``ISIF`` is inert for VASP itself (there is no ionic
    relaxation for it to configure), but leaving it in the written INCAR is
    misleading for provenance: it reads as though the statics job relaxes
    something. Statics jobs never carry ISIF; this makes that true whether
    or not the caller's raw INCAR had one.
    """
    modified = Incar.from_dict(dict(incar))
    modified.pop("ISIF", None)
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


@fr.atomic("structures", "names")
def require_permutations(structures, names):
    """Guard against an empty permutation set reaching the permutation
    for-loop -- see ``NoPermutationsGeneratedError``."""
    if not structures:
        raise NoPermutationsGeneratedError(
            "get_ASSYST_deformed_structures produced zero permutations "
            "(n_rattle_permutations=0 and n_stretch_permutations=0 would "
            "always do this; a nonzero request can still land here if every "
            "generation attempt failed the validity filter). The "
            "permutation for-loop cannot run on an empty list under "
            "pyiron_workflow's current n=0 ForEach bug (see the module "
            "docstring); request at least one permutation, or check "
            "core_overlap_tolerance/min_dist if a nonzero request produced "
            "this."
        )
    return structures, names


@fr.atomic("value")
def unwrap_singleton(value):
    """Undo pyiron_workflow 0.19.0's length-1 ``ForEach`` scatter bug.

    TEMPORARY WORKAROUND -- see the module docstring for the full writeup.
    Reported upstream: https://github.com/pyiron/pyiron_workflow/issues/915
    ("ForEach: zip loops over length-1 iterables wrap elements in a spurious
    1-tuple (length-0 raises IndexError)").
    Affected version: ``pyiron_workflow==0.19.0``. Symptom: inside a
    ``@fr.workflow`` body, a ``for a, b in zip(xs, ys)`` loop's variables
    arrive wrapped in a spurious 1-tuple whenever the iterated collection has
    length EXACTLY 1 (length 2+ is unaffected). Root cause:
    ``pyiron_workflow.transformers.Transform1toN(n=1)`` builds a
    single-output atomic node whose function returns ``tuple(items)`` (a
    1-tuple), and ``atomic_node._store_atomic_outputs`` only unpacks a
    returned tuple across ports when there is MORE than one output port --
    with exactly one port it assigns the whole 1-tuple to that port instead
    of unpacking its lone element. Minimal reproducer, independent of this
    package:

        n=1 -> loop body receives {'type': 'tuple', 'value': (0,), 'name': ('n0',)}
        n=2 -> loop body receives {'type': 'int',   'value': 0,    'name': 'n0'}   (correct)

    Call this on EVERY iterated loop variable, at the top of the loop body,
    before it is used for anything else -- do not bury the unwrap inside a
    downstream node (``join_path``, ``generate_vasp_input``, ...), since a
    caller reading those should not have to discover this workaround.

    DELETE this function and every call site once the upstream bug is fixed
    (check https://github.com/pyiron/pyiron_workflow/issues/915 for status);
    ``tests/test_workflow.py::test_assyst_graph_runs_end_to_end_default_threshold``
    is the regression test that exercises the length-1 path this guards, and
    must keep passing with the workaround removed once the fix lands.
    """
    if isinstance(value, tuple) and len(value) == 1:
        return value[0]
    return value


@fr.atomic("df")
def concat_and_save(
    base_results,
    perm_results,
    filename,
    isif7_converged,
    isif5_converged,
    isif2_converged,
):
    """Concatenate every statics result and stamp each row with whether the
    ISIF7/5/2 relaxation chain that produced its geometry actually
    converged.

    Convergence is a property of the shared relaxation chain, not of any
    individual statics job, so the same three booleans are broadcast across
    every row (base images and permutations alike -- permutations are
    deformations of the base images, so they inherit the same relaxation
    provenance). Without this, a campaign has no way to tell, after the
    fact, whether the geometry it trained on came from a converged
    relaxation or one truncated by ``ionic_steps``/NSW.
    """
    frames = [f for f in list(base_results) + list(perm_results) if f is not None]
    combined = pd.concat(frames, ignore_index=True)
    combined["isif7_converged"] = isif7_converged
    combined["isif5_converged"] = isif5_converged
    combined["isif2_converged"] = isif2_converged
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

    CAVEAT: unlike every work directory above, ``train_df_filename`` defaults
    to a CWD-relative path (``"df_ASSYST_jobs.pkl"``), not an absolute path
    derived from ``job_name``. Concurrent runs for different structures that
    both leave this default (and share a CWD) will collide, each overwriting
    the other's pickle. Pass an absolute, job-specific path explicitly for
    concurrent use; the default is left as-is here for backward compatibility
    with existing callers.

    DELIBERATE SEMANTIC DEVIATION FROM PRIOR CAMPAIGNS -- ``ionic_steps``:
    this port applies ``ionic_steps`` uniformly to all three relaxation
    stages (ISIF7, ISIF5, ISIF2); the pre-port implementation applied it only
    to the ISIF7 stage, and let ISIF5 and ISIF2 run with the raw INCAR's
    ``NSW`` value unchanged. That difference was material in production:
    ISIF7 ran at NSW=100 (200 on reconverge) while ISIF5 and ISIF2 ran at
    NSW=500. Consequently, results from this version are NOT directly
    comparable to existing datasets for the ISIF5 and ISIF2 stages -- a
    structure that previously needed more than ``ionic_steps`` iterations to
    converge its cell shape (ISIF5) or ion positions (ISIF2) will now stop
    early, and it is that geometry, not a fully converged one, that feeds the
    subsequent accurate statics. This was reviewed and deliberately accepted
    by the package owner on 2026-08-01 -- it is not an oversight, and should
    not be "fixed" back to the old per-stage behavior without a fresh
    discussion. Convergence rates for the ISIF5 and ISIF2 stages should be
    checked against the first campaign run under this version.
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
    accurate_incar = strip_isif(accurate_incar)

    base_results = []
    for base_structure, base_name in zip(base_structures, base_names):
        # Workaround for the length-1 ForEach bug -- see module docstring.
        base_structure = unwrap_singleton(base_structure)
        base_name = unwrap_singleton(base_name)
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
    perm_structures, perm_names = require_permutations(perm_structures, perm_names)

    perm_results = []
    for perm_structure, perm_name in zip(perm_structures, perm_names):
        # Workaround for the length-1 ForEach bug -- see module docstring.
        perm_structure = unwrap_singleton(perm_structure)
        perm_name = unwrap_singleton(perm_name)
        perm_input = generate_vasp_input(perm_structure, accurate_incar, potcar_paths)
        perm_path = join_path(job_name, perm_name)
        perm_parser_args = build_parser_args(perm_path)
        perm_out, perm_conv = vasp_job(
            perm_path, perm_input, vasp_command, None, compress_dirs,
            compressed_file_in_dir, remove_calc_dirs, vasp_parser_function, perm_parser_args,
        )
        perm_results.append(perm_out)

    train_df = concat_and_save(
        base_results, perm_results, train_df_filename, conv7, conv5, conv2
    )
    return train_df
