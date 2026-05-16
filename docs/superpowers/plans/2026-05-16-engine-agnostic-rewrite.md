# Engine-agnostic pyiron_workflow_assyst rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `pyiron_workflow_assyst` to follow `pyiron_workflow_atomistics` (pwa) design principles — engine-agnostic via the pwa Engine Protocol — while preserving exact behavioural equivalence to the current VASP workflow when driven with `VaspEngine`.

**Architecture:** Three sequential phases across three repos. Phase A adds a `cell_relaxation` enum to pwa's `CalcInputMinimize` so ISIF=7/5/2 are reachable. Phase B threads that enum through `pyiron_workflow_vasp` and adds the missing accurate-static knobs (`ediff`, `lreal`, `compress_outputs`, `remove_workdir`). Phase C rewrites `pyiron_workflow_assyst` against the Engine Protocol into `structure/`, `physics/`, `analysis/`, `testing/`, `_internal/` subpackages with TDD, finishing with an ASE+EMT integration test for CI and a `@pytest.mark.vasp`-gated equivalence test.

**Tech Stack:**
- Python 3.9–3.12
- `pyiron-workflow==0.15.6`, `pyiron_workflow_atomistics` (HEAD-of-Phase-A)
- `pyiron_workflow_vasp` (HEAD-of-Phase-B)
- `ase==3.28.0`, `pymatgen==2026.5.4`, `numpy==1.26.4`, `pandas==3.0.3`
- `pytest`, `pyxtal` (optional), `versioneer==0.29`
- Build: `setuptools` + `versioneer` (pwa style)
- Env: **`pixi`** for any newly-created env (per user preference); existing `test_pyiron_workflow_atomistics` mamba env may be reused

**Spec:** `docs/superpowers/specs/2026-05-16-pyiron-workflow-assyst-rewrite-design.md` (in the assyst repo)

---

## File Structure

### Phase A — `pyiron_workflow_atomistics` (one-file change + tests)

| File | Action | Responsibility |
|---|---|---|
| `pyiron_workflow_atomistics/engine/inputs.py` | Modify | Add `cell_relaxation` field; keep `relax_cell` as deprecating property |
| `pyiron_workflow_atomistics/engine/ase.py` | Modify | `_minimize` routes on `cell_relaxation` instead of `relax_cell` |
| `tests/unit/engine/test_inputs.py` | Create | Unit tests for new field + backwards-compat property |
| `tests/unit/engine/test_ase_minimize.py` | Modify | Add cases for the new enum values |

### Phase B — `pyiron_workflow_vasp` (engine knob additions)

| File | Action | Responsibility |
|---|---|---|
| `pyiron_workflow_vasp/_run.py` | Modify | Replace ISIF assignment with `ISIF_MAP` lookup; honor new VaspEngine knobs |
| `pyiron_workflow_vasp/engine.py` | Modify | Add `ediff`, `lreal`, `compress_outputs`, `remove_workdir` fields |
| `pyproject.toml` | Modify | Bump pinned `pyiron_workflow_atomistics` |
| `tests/unit/test_engine_isif_mapping.py` | Create | Unit test the ISIF_MAP and INCAR emission |
| `tests/unit/test_engine_knobs.py` | Create | Verify new fields reach INCAR / cleanup |

### Phase C — `pyiron_workflow_assyst` (full rewrite)

| File | Action | Responsibility |
|---|---|---|
| `pyiron_workflow_assyst/__init__.py` | Rewrite | Versioneer `__version__` + PEP 562 lazy `__getattr__` only |
| `pyiron_workflow_assyst/_version.py` | Create (versioneer-managed) | Version string |
| `pyiron_workflow_assyst/py.typed` | Create (empty) | PEP 561 typing marker |
| `pyiron_workflow_assyst/_internal/__init__.py` | Create | Empty package |
| `pyiron_workflow_assyst/_internal/engine_fanout.py` | Create | `_build_subengines`, `_concat` node helpers |
| `pyiron_workflow_assyst/structure/__init__.py` | Create | Re-exports for public structure ops |
| `pyiron_workflow_assyst/structure/filters.py` | Create | `RCORE` table + `is_valid_structure` etc. |
| `pyiron_workflow_assyst/structure/deformations.py` | Create | `apply_rattle` / `apply_triaxial_strain` / `apply_shear_strain` |
| `pyiron_workflow_assyst/structure/permutations.py` | Create | `generate_assyst_permutations` function node |
| `pyiron_workflow_assyst/structure/generate.py` | Create | `pyxtal_random_crystals` function node (lazy pyxtal import) |
| `pyiron_workflow_assyst/physics/__init__.py` | Create | Docstring only |
| `pyiron_workflow_assyst/physics/relaxation.py` | Create | `multistage_relax` macro |
| `pyiron_workflow_assyst/physics/assyst.py` | Create | `run_assyst` top-level macro |
| `pyiron_workflow_assyst/analysis/__init__.py` | Create | Re-exports |
| `pyiron_workflow_assyst/analysis/collect.py` | Create | `collect_relaxation_frames` + `_select_indices_by_threshold` |
| `pyiron_workflow_assyst/analysis/export.py` | Create | `export_training_set` |
| `pyiron_workflow_assyst/testing/__init__.py` | Create | Re-exports |
| `pyiron_workflow_assyst/testing/fixtures.py` | Create | Shared pytest fixtures |
| `pyproject.toml` | Rewrite | Setuptools + versioneer, pinned deps, optional extras |
| `setup.cfg` or `pyproject.toml versioneer table` | Create | Versioneer config |
| `tests/unit/structure/test_filters.py` | Create | RCORE byte-identity + validity edge cases |
| `tests/unit/structure/test_deformations.py` | Create | Golden-value tests with seeded RNG |
| `tests/unit/structure/test_permutations.py` | Create | Generator determinism + name scheme |
| `tests/unit/structure/test_generate.py` | Create | PyXtal smoke (skipped if missing) |
| `tests/unit/analysis/test_collect.py` | Create | Threshold-selection golden + EngineOutput filter |
| `tests/unit/analysis/test_export.py` | Create | Pickle + extxyz round-trip |
| `tests/integration/test_multistage_relax_emt.py` | Create | EMT integration test |
| `tests/integration/test_run_assyst_emt.py` | Create | End-to-end EMT integration test |
| `tests/integration/test_vasp_equivalence.py` | Create | Legacy-vs-new VASP equivalence (gated) |
| `tests/_legacy_assyst/` | Create (test-only) | Frozen snapshot of old `workflow.py` + filter utils for equivalence test |
| `pyiron_workflow_assyst/example.py` | Delete | Replaced by example notebook (out of scope) |
| `pyiron_workflow_assyst/workflow.py` | Delete | Replaced by `physics/` |
| `pyiron_workflow_assyst/structure_filter_utils.py` | Delete | Replaced by `structure/filters.py` |
| `README.md` | Update | Engine-agnostic usage, dual-engine pattern |

---

## Phase A — pwa upstream: `feat/cell-relaxation-enum`

**Repo:** `~/pyiron_workflow_atomistics`
**Branch:** `feat/cell-relaxation-enum`
**Outcome:** `CalcInputMinimize` exposes a `cell_relaxation: Literal["none","volume","shape","full"]` field with a `relax_cell` deprecation shim.

### Task A0: Branch and verify env

**Files:** none

- [ ] **Step 1: Branch from main**

Run:
```bash
cd ~/pyiron_workflow_atomistics
git fetch origin
git checkout main && git pull --ff-only
git checkout -b feat/cell-relaxation-enum
```

Expected: `Switched to a new branch 'feat/cell-relaxation-enum'`

- [ ] **Step 2: Verify the existing test env can import pwa**

Run:
```bash
PYIRON_PY=/home/liger/miniforge3/envs/test_pyiron_workflow_atomistics/bin/python
"$PYIRON_PY" -c "from pyiron_workflow_atomistics.engine.inputs import CalcInputMinimize; print(CalcInputMinimize())"
```

Expected: `CalcInputMinimize(force_convergence_tolerance=0.01, energy_convergence_tolerance=1e-05, max_iterations=1000000, relax_cell=False)`

- [ ] **Step 3: Run existing tests to establish baseline**

Run:
```bash
"$PYIRON_PY" -m pytest tests/unit/engine -q
```

Expected: all green (record the count for sanity).

### Task A1: Add `cell_relaxation` field with failing test

**Files:**
- Create: `tests/unit/engine/test_inputs.py`
- Modify: `pyiron_workflow_atomistics/engine/inputs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/engine/test_inputs.py`:
```python
"""Unit tests for CalcInput* dataclasses, focused on the cell_relaxation
enum added in feat/cell-relaxation-enum."""

import warnings

import pytest

from pyiron_workflow_atomistics.engine.inputs import CalcInputMinimize


class TestCellRelaxationField:
    def test_default_is_none(self):
        ci = CalcInputMinimize()
        assert ci.cell_relaxation == "none"

    @pytest.mark.parametrize(
        "value", ["none", "volume", "shape", "full"]
    )
    def test_accepts_all_four_modes(self, value):
        ci = CalcInputMinimize(cell_relaxation=value)
        assert ci.cell_relaxation == value


class TestRelaxCellShim:
    def test_relax_cell_true_maps_to_full(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ci = CalcInputMinimize(relax_cell=True)
        assert ci.cell_relaxation == "full"

    def test_relax_cell_false_maps_to_none(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            ci = CalcInputMinimize(relax_cell=False)
        assert ci.cell_relaxation == "none"

    def test_relax_cell_property_reflects_cell_relaxation(self):
        assert CalcInputMinimize(cell_relaxation="full").relax_cell is True
        assert CalcInputMinimize(cell_relaxation="none").relax_cell is False
        assert CalcInputMinimize(cell_relaxation="volume").relax_cell is True
        assert CalcInputMinimize(cell_relaxation="shape").relax_cell is True

    def test_relax_cell_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="cell_relaxation"):
            CalcInputMinimize(relax_cell=True)

    def test_relax_cell_and_cell_relaxation_together_is_an_error(self):
        # Both provided explicitly: ambiguous, reject.
        with pytest.raises(ValueError, match="both"):
            CalcInputMinimize(relax_cell=True, cell_relaxation="shape")
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
"$PYIRON_PY" -m pytest tests/unit/engine/test_inputs.py -q
```

Expected: all tests in `TestCellRelaxationField` and `TestRelaxCellShim` FAIL — `CalcInputMinimize` has no `cell_relaxation` attribute, `relax_cell` is a plain field.

- [ ] **Step 3: Implement the field + shim**

Rewrite `pyiron_workflow_atomistics/engine/inputs.py` `CalcInputMinimize` only — leave `CalcInputStatic` and `CalcInputMD` untouched:

```python
@dataclass
class CalcInputMinimize:
    """Structural relaxation parameters.

    Attributes
    ----------
    force_convergence_tolerance
        Max allowed force component on any atom, in eV/Å. Default 1e-2.
    energy_convergence_tolerance
        Energy change between consecutive steps, in eV. Default 1e-5.
    max_iterations
        Hard cap on optimiser steps.
    cell_relaxation
        Which cell degrees of freedom to relax. ``"none"`` (atoms only, fixed
        cell), ``"volume"`` (cell volume only, shape and atoms fixed),
        ``"shape"`` (cell shape only, volume and atoms fixed), or ``"full"``
        (cell + atoms). Default ``"none"``.

    Notes
    -----
    The legacy ``relax_cell: bool`` argument is still accepted but
    deprecated — ``relax_cell=True`` is an alias for
    ``cell_relaxation="full"`` and ``relax_cell=False`` for
    ``cell_relaxation="none"``. Specifying both raises ``ValueError``.
    """

    force_convergence_tolerance: float = 1e-2
    energy_convergence_tolerance: float = 1e-5
    max_iterations: int = 1_000_000
    cell_relaxation: Literal["none", "volume", "shape", "full"] = "none"

    # Constructor-only legacy alias; never stored on the instance.
    def __init__(
        self,
        force_convergence_tolerance: float = 1e-2,
        energy_convergence_tolerance: float = 1e-5,
        max_iterations: int = 1_000_000,
        cell_relaxation: Literal["none", "volume", "shape", "full"] | None = None,
        *,
        relax_cell: bool | None = None,
    ) -> None:
        if cell_relaxation is not None and relax_cell is not None:
            raise ValueError(
                "Specify cell_relaxation OR relax_cell, not both. "
                "relax_cell is deprecated; prefer cell_relaxation."
            )
        if relax_cell is not None:
            import warnings as _warnings

            _warnings.warn(
                "relax_cell is deprecated; use cell_relaxation='full' or "
                "cell_relaxation='none' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            cell_relaxation = "full" if relax_cell else "none"
        if cell_relaxation is None:
            cell_relaxation = "none"
        object.__setattr__(self, "force_convergence_tolerance", force_convergence_tolerance)
        object.__setattr__(self, "energy_convergence_tolerance", energy_convergence_tolerance)
        object.__setattr__(self, "max_iterations", max_iterations)
        object.__setattr__(self, "cell_relaxation", cell_relaxation)

    @property
    def relax_cell(self) -> bool:
        """Deprecated. Backwards-compat property: True iff cell_relaxation != 'none'."""
        return self.cell_relaxation != "none"
```

Also add `from typing import Literal` to the imports at the top if not present.

- [ ] **Step 4: Run to verify it passes**

Run:
```bash
"$PYIRON_PY" -m pytest tests/unit/engine/test_inputs.py -q
```

Expected: all 10 tests pass.

- [ ] **Step 5: Re-run the whole engine test suite to check for regressions**

Run:
```bash
"$PYIRON_PY" -m pytest tests/unit/engine -q
```

Expected: same green count as the A0 baseline, plus the new test_inputs tests.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/engine/test_inputs.py pyiron_workflow_atomistics/engine/inputs.py
git commit -m "feat(engine): add cell_relaxation enum + relax_cell deprecation shim"
```

### Task A2: ASEEngine honors `cell_relaxation` enum

**Files:**
- Modify: `pyiron_workflow_atomistics/engine/ase.py`
- Create: `tests/unit/engine/test_ase_cell_relaxation.py`

- [ ] **Step 1: Locate the current relax_cell branch**

Run:
```bash
grep -n "relax_cell" pyiron_workflow_atomistics/engine/ase.py
```

Expected: one or two lines showing where the optimiser chooses between fixed-cell BFGS and `UnitCellFilter`-wrapped BFGS. Note the line numbers — Step 3 patches in-place.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/engine/test_ase_cell_relaxation.py`:
```python
"""ASEEngine routes the new cell_relaxation enum to the right optimiser."""

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import (
    ASEEngine,
    CalcInputMinimize,
    calculate,
)


def _make_engine(tmp_path, mode):
    return ASEEngine(
        EngineInput=CalcInputMinimize(cell_relaxation=mode, max_iterations=5),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )


class TestASEMinimizeRouting:
    def test_none_runs(self, tmp_path):
        eng = _make_engine(tmp_path, "none")
        out = calculate.node_function(structure=bulk("Cu", "fcc", a=3.6, cubic=True), engine=eng)
        assert out.final_energy is not None

    def test_full_runs(self, tmp_path):
        eng = _make_engine(tmp_path, "full")
        out = calculate.node_function(structure=bulk("Cu", "fcc", a=3.6, cubic=True), engine=eng)
        assert out.final_energy is not None

    @pytest.mark.parametrize("mode", ["volume", "shape"])
    def test_volume_and_shape_raise(self, tmp_path, mode):
        eng = _make_engine(tmp_path, mode)
        with pytest.raises(NotImplementedError, match=mode):
            calculate.node_function(
                structure=bulk("Cu", "fcc", a=3.6, cubic=True), engine=eng
            )
```

- [ ] **Step 3: Run to verify it fails**

Run:
```bash
"$PYIRON_PY" -m pytest tests/unit/engine/test_ase_cell_relaxation.py -q
```

Expected: at minimum, the `test_volume_and_shape_raise` cases fail (ASE silently treats them as `True`). `test_none` / `test_full` may already pass — that's fine.

- [ ] **Step 4: Implement the routing**

In `pyiron_workflow_atomistics/engine/ase.py`, find the minimisation function and replace its `relax_cell` branch with:

```python
def _route_minimize(self, atoms: Atoms) -> Any:
    mode = self.EngineInput.cell_relaxation
    if mode == "none":
        return BFGS(atoms, ...)  # existing fixed-cell call kept as-is
    if mode == "full":
        from ase.constraints import UnitCellFilter
        return BFGS(UnitCellFilter(atoms), ...)  # existing variable-cell call kept as-is
    raise NotImplementedError(
        f"ASEEngine does not support cell_relaxation={mode!r}; "
        "ASE has no native volume-only or shape-only relaxation primitive. "
        "Use cell_relaxation='full' or switch to a backend that supports it (e.g. VaspEngine)."
    )
```

Adapt the exact integration to the file's current structure — the key change is reading `self.EngineInput.cell_relaxation` instead of `self.EngineInput.relax_cell`, and raising `NotImplementedError` for `"volume"`/`"shape"`.

- [ ] **Step 5: Run to verify it passes**

Run:
```bash
"$PYIRON_PY" -m pytest tests/unit/engine/test_ase_cell_relaxation.py -q
```

Expected: all 4 tests pass.

- [ ] **Step 6: Re-run the whole engine test suite**

Run:
```bash
"$PYIRON_PY" -m pytest tests/unit/engine -q
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add tests/unit/engine/test_ase_cell_relaxation.py pyiron_workflow_atomistics/engine/ase.py
git commit -m "feat(engine/ase): route minimise on cell_relaxation enum"
```

### Task A3: Open PR, await merge

**Files:** none

- [ ] **Step 1: Push branch**

```bash
git push -u origin feat/cell-relaxation-enum
```

- [ ] **Step 2: Open PR**

```bash
gh pr create --title "feat(engine): cell_relaxation enum on CalcInputMinimize" --body "$(cat <<'EOF'
## Summary
- Adds `cell_relaxation: Literal["none","volume","shape","full"]` to `CalcInputMinimize`.
- `relax_cell: bool` becomes a deprecating property — `True ⇔ "full"`, `False ⇔ "none"`.
- `ASEEngine` routes minimisation on the new enum; raises `NotImplementedError` for `"volume"` / `"shape"` (ASE has no native primitive for these).
- Unblocks downstream packages that need ISIF=7 / ISIF=5 semantics (notably the engine-agnostic `pyiron_workflow_assyst` rewrite).

## Test plan
- [x] `pytest tests/unit/engine -q` green
- [x] New `tests/unit/engine/test_inputs.py` covers the field, the shim, and the both-provided error
- [x] New `tests/unit/engine/test_ase_cell_relaxation.py` covers ASE routing for all four enum values

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Capture the PR URL; this PR blocks Phase B.

- [ ] **Step 3: Wait for merge before starting Phase B**

Check via `gh pr view <url> --json mergeStateStatus,statusCheckRollup` — wait for `mergeStateStatus == CLEAN` and no FAILURE in checks. Per the [[gh-pr-merge-auto-misuse]] memory: don't `--auto` merge until both conditions hold.

---

## Phase B — pwv upstream: `feat/vasp-engine-isif-mapping`

**Repo:** `~/pyiron_workflow_vasp`
**Branch:** `feat/vasp-engine-isif-mapping`
**Outcome:** `VaspEngine` understands the four-mode `cell_relaxation` enum and exposes `ediff`, `lreal`, `compress_outputs`, `remove_workdir` knobs.

### Task B0: Branch + verify env can import the new pwa

**Files:** none

- [ ] **Step 1: Branch from main and pull the new pwa**

```bash
cd ~/pyiron_workflow_vasp
git fetch origin
git checkout main && git pull --ff-only
git checkout -b feat/vasp-engine-isif-mapping
PYIRON_PY=/home/liger/miniforge3/envs/test_pyiron_workflow_atomistics/bin/python
"$PYIRON_PY" -m uv pip install --python "$PYIRON_PY" -e ~/pyiron_workflow_atomistics
```

(`uv` is preferred per [[uv-package-manager]]; the env is the existing mamba env per [[test-env-convention]].)

Expected: pwa install succeeds; `python -c "from pyiron_workflow_atomistics.engine.inputs import CalcInputMinimize; print(CalcInputMinimize(cell_relaxation='volume'))"` works.

- [ ] **Step 2: Establish pwv baseline**

```bash
"$PYIRON_PY" -m pytest tests -q
```

Expected: green (record count).

### Task B1: ISIF_MAP lookup in `_run.py`

**Files:**
- Create: `tests/unit/test_engine_isif_mapping.py`
- Modify: `pyiron_workflow_vasp/_run.py`

- [ ] **Step 1: Locate the ISIF assignment**

```bash
grep -n "ISIF" pyiron_workflow_vasp/_run.py
```

Expected lines: `params["ISIF"] = 3 if engine_input.relax_cell else 2`.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/test_engine_isif_mapping.py`:
```python
"""ISIF assignment in run_vasp must derive from cell_relaxation, not relax_cell."""

import pytest

from pyiron_workflow_atomistics.engine import CalcInputMinimize, CalcInputStatic
from pyiron_workflow_vasp._run import _build_incar_params  # introduced in this task


@pytest.mark.parametrize(
    "mode,expected_isif",
    [("none", 2), ("volume", 7), ("shape", 5), ("full", 3)],
)
def test_isif_mapping(mode, expected_isif):
    ci = CalcInputMinimize(cell_relaxation=mode, max_iterations=42)
    params = _build_incar_params(ci, mode="minimize")
    assert params["ISIF"] == expected_isif
    assert params["NSW"] == 42


def test_static_mode_sets_nsw_zero():
    params = _build_incar_params(CalcInputStatic(), mode="static")
    assert params["NSW"] == 0
    assert "ISIF" not in params or params["ISIF"] == 2  # ISIF is irrelevant when NSW=0
```

- [ ] **Step 3: Run to verify it fails**

Run:
```bash
"$PYIRON_PY" -m pytest tests/unit/test_engine_isif_mapping.py -q
```

Expected: ImportError because `_build_incar_params` does not exist yet.

- [ ] **Step 4: Extract `_build_incar_params` from `run_vasp`**

In `pyiron_workflow_vasp/_run.py`, extract the params-construction block into a top-level helper named `_build_incar_params(engine_input, mode)`. Replace the ISIF assignment with:

```python
ISIF_MAP = {"none": 2, "volume": 7, "shape": 5, "full": 3}


def _build_incar_params(engine_input, mode: str) -> dict:
    """Translate the engine-level dataclass into a VASP INCAR-params dict.

    ``mode`` is one of ``"static"`` / ``"minimize"`` (set by ``VaspEngine.__post_init__``).
    """
    params: dict = {}
    if mode == "static":
        params["NSW"] = 0
    else:  # minimize
        params["NSW"] = engine_input.max_iterations
        params["ISIF"] = ISIF_MAP[engine_input.cell_relaxation]
        # EDIFFG: negative = force-tolerance convention in VASP
        params["EDIFFG"] = -abs(engine_input.force_convergence_tolerance)
        params["EDIFF"] = engine_input.energy_convergence_tolerance
    return params
```

Wire the existing `run_vasp` body to call `_build_incar_params(engine_input, mode)` instead of constructing the dict inline. Preserve every other line of `run_vasp`.

- [ ] **Step 5: Run to verify it passes**

Run:
```bash
"$PYIRON_PY" -m pytest tests/unit/test_engine_isif_mapping.py -q
```

Expected: all 5 tests pass.

- [ ] **Step 6: Run the whole pwv suite for regressions**

```bash
"$PYIRON_PY" -m pytest tests -q
```

Expected: same baseline green count from B0.

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_engine_isif_mapping.py pyiron_workflow_vasp/_run.py
git commit -m "feat(vasp): ISIF_MAP lookup keyed by CalcInputMinimize.cell_relaxation"
```

### Task B2: Add `ediff`, `lreal` knobs to `VaspEngine`

**Files:**
- Create: `tests/unit/test_engine_knobs.py`
- Modify: `pyiron_workflow_vasp/engine.py`
- Modify: `pyiron_workflow_vasp/_run.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_engine_knobs.py`:
```python
"""VaspEngine new knobs (ediff, lreal, compress_outputs, remove_workdir) reach INCAR / cleanup."""

import pytest

from pyiron_workflow_atomistics.engine import CalcInputStatic
from pyiron_workflow_vasp.engine import VaspEngine
from pyiron_workflow_vasp._run import _build_incar_overrides  # introduced in this task


class TestEdiffLrealReachIncar:
    def test_ediff_lands_in_overrides(self):
        eng = VaspEngine(EngineInput=CalcInputStatic(), ediff=1e-5)
        overrides = _build_incar_overrides(eng)
        assert overrides["EDIFF"] == pytest.approx(1e-5)

    def test_lreal_false_lands_in_overrides(self):
        eng = VaspEngine(EngineInput=CalcInputStatic(), lreal=False)
        overrides = _build_incar_overrides(eng)
        assert overrides["LREAL"] is False

    def test_lreal_auto_string_passes_through(self):
        eng = VaspEngine(EngineInput=CalcInputStatic(), lreal="Auto")
        overrides = _build_incar_overrides(eng)
        assert overrides["LREAL"] == "Auto"

    def test_none_means_omitted(self):
        eng = VaspEngine(EngineInput=CalcInputStatic())
        overrides = _build_incar_overrides(eng)
        assert "EDIFF" not in overrides
        assert "LREAL" not in overrides


class TestCleanupKnobsExist:
    def test_defaults(self):
        eng = VaspEngine(EngineInput=CalcInputStatic())
        assert eng.compress_outputs is False
        assert eng.remove_workdir is False

    def test_can_set(self):
        eng = VaspEngine(
            EngineInput=CalcInputStatic(),
            compress_outputs=True,
            remove_workdir=True,
        )
        assert eng.compress_outputs is True
        assert eng.remove_workdir is True
```

- [ ] **Step 2: Run to verify it fails**

Run:
```bash
"$PYIRON_PY" -m pytest tests/unit/test_engine_knobs.py -q
```

Expected: ImportError / AttributeError — `ediff`, `lreal`, `compress_outputs`, `remove_workdir` are not fields yet; `_build_incar_overrides` does not exist.

- [ ] **Step 3: Add the fields to `VaspEngine`**

In `pyiron_workflow_vasp/engine.py`, extend the dataclass:

```python
@dataclass
class VaspEngine:
    EngineInput: CalcInputStatic | CalcInputMinimize | CalcInputMD
    working_directory: str = field(default_factory=os.getcwd)

    potcar_config_file: Path | None = None
    functional: Literal["GGA", "LDA"] = "GGA"
    encut: float = 520.0
    kpoints_density: float = 0.30
    command: str = "vasp_std"

    # New in this PR:
    ediff: float | None = None
    lreal: bool | str | None = None
    compress_outputs: bool = False
    remove_workdir: bool = False

    mode: Literal["static", "minimize"] = field(init=False)
    # ... rest unchanged ...
```

Pass the new fields through `get_calculate_fn` kwargs.

- [ ] **Step 4: Add `_build_incar_overrides` to `_run.py`**

```python
def _build_incar_overrides(engine) -> dict:
    """INCAR keys that the user opted into via VaspEngine fields (None => omit)."""
    overrides: dict = {}
    if engine.ediff is not None:
        overrides["EDIFF"] = engine.ediff
    if engine.lreal is not None:
        overrides["LREAL"] = engine.lreal
    return overrides
```

Wire `run_vasp` to merge these overrides over the params produced by `_build_incar_params`. Apply `compress_outputs` and `remove_workdir` in the post-run cleanup branch (use the existing `compress` / `remove_calc_dir` semantics from `vasp_job` in the legacy assyst code as the reference behavior).

- [ ] **Step 5: Run to verify it passes**

Run:
```bash
"$PYIRON_PY" -m pytest tests/unit/test_engine_knobs.py -q
```

Expected: all 6 tests pass.

- [ ] **Step 6: Re-run the pwv suite**

```bash
"$PYIRON_PY" -m pytest tests -q
```

Expected: green.

- [ ] **Step 7: Commit**

```bash
git add tests/unit/test_engine_knobs.py pyiron_workflow_vasp/engine.py pyiron_workflow_vasp/_run.py
git commit -m "feat(vasp/engine): add ediff, lreal, compress_outputs, remove_workdir knobs"
```

### Task B3: Bump pinned `pyiron_workflow_atomistics`

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Read the current pin**

```bash
grep "pyiron_workflow_atomistics" pyproject.toml
```

- [ ] **Step 2: Bump to the version produced by Phase A**

In `pyproject.toml`, update the pin to the released-or-prerelease tag from Phase A's merge (e.g., `pyiron_workflow_atomistics==0.X.Ya0` — substitute the actual version from the merged PR's release tag). If Phase A has not yet cut a release tag, use a `>=<merged-commit-sha>` install URL pin or coordinate a release first.

- [ ] **Step 3: Verify install resolves**

```bash
"$PYIRON_PY" -m uv pip install --python "$PYIRON_PY" -e .
```

Expected: install succeeds; `python -c "from pyiron_workflow_vasp.engine import VaspEngine; VaspEngine(EngineInput=__import__('pyiron_workflow_atomistics.engine', fromlist=['CalcInputMinimize']).CalcInputMinimize(cell_relaxation='shape'))"` works without error.

- [ ] **Step 4: Re-run the pwv suite**

```bash
"$PYIRON_PY" -m pytest tests -q
```

Expected: green.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: bump pinned pyiron_workflow_atomistics for cell_relaxation enum"
```

### Task B4: Open PR, await merge

**Files:** none

- [ ] **Step 1: Push and open PR**

```bash
git push -u origin feat/vasp-engine-isif-mapping
gh pr create --title "feat(vasp): cell_relaxation→ISIF map + ediff/lreal/cleanup knobs" --body "$(cat <<'EOF'
## Summary
- `_run.py` derives ISIF from `CalcInputMinimize.cell_relaxation` via `ISIF_MAP = {"none":2, "volume":7, "shape":5, "full":3}` instead of the binary `relax_cell` branch.
- `VaspEngine` gains `ediff: float | None`, `lreal: bool | str | None`, `compress_outputs: bool`, `remove_workdir: bool` fields. `None` means "leave INCAR untouched" / "don't clean up".
- Bumps pinned `pyiron_workflow_atomistics` to the version with the new enum.
- Unblocks the engine-agnostic `pyiron_workflow_assyst` rewrite, which needs ISIF=7/5/2 and the accurate-static INCAR overrides.

## Test plan
- [x] `pytest tests -q` green
- [x] `tests/unit/test_engine_isif_mapping.py` covers all four ISIF values + static NSW=0
- [x] `tests/unit/test_engine_knobs.py` covers ediff / lreal / cleanup field defaults and override emission

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Wait for merge**

Per [[gh-pr-merge-auto-misuse]] — verify `mergeStateStatus==CLEAN` and no FAILURE in checks before merging. Phase C blocks on this.

---

## Phase C — `pyiron_workflow_assyst` rewrite

**Repo:** `~/pyiron_workflow_assyst`
**Branch:** continue on `feat/engine-agnostic-rewrite-spec` (spec already committed there) — rename or leave as-is; PR will be opened at end.
**Outcome:** New module per the spec, passing unit + EMT integration tests; VASP-equivalence test gated.

### Task C0: Branch hygiene, env setup, install both upstream deps

**Files:** none

- [ ] **Step 1: Confirm branch and verify spec is committed**

```bash
cd ~/pyiron_workflow_assyst
git status
git log --oneline -5
```

Expected: on `feat/engine-agnostic-rewrite-spec`, two doc commits visible (the spec and the self-review fixes).

- [ ] **Step 2: Rename branch to drop the `-spec` suffix**

```bash
git branch -m feat/engine-agnostic-rewrite
```

- [ ] **Step 3: Create a fresh pixi env for assyst**

Per [[pixi-for-envs]] — new envs use pixi, not mamba. Create a project pixi env:

```bash
pixi init --pyproject  # only if no pixi.toml present
pixi add python=3.11 pip
pixi run python -m uv pip install -e ~/pyiron_workflow_atomistics -e ~/pyiron_workflow_vasp -e .
pixi run python -c "import pyiron_workflow_assyst; print(pyiron_workflow_assyst.__version__)"
```

Expected: prints `0.1.0` (current version, will bump later) without errors.

- [ ] **Step 4: Verify the new pwa + pwv APIs are visible**

```bash
pixi run python -c "
from pyiron_workflow_atomistics.engine import CalcInputMinimize, CalcInputStatic
from pyiron_workflow_vasp.engine import VaspEngine
print(CalcInputMinimize(cell_relaxation='volume'))
print(VaspEngine.__dataclass_fields__.keys())
"
```

Expected: prints `CalcInputMinimize(..., cell_relaxation='volume')` and the dataclass field list includes `ediff`, `lreal`, `compress_outputs`, `remove_workdir`.

### Task C1: Scaffold the new package skeleton (empty modules)

**Files:**
- Create: `pyiron_workflow_assyst/_internal/__init__.py`
- Create: `pyiron_workflow_assyst/structure/__init__.py`
- Create: `pyiron_workflow_assyst/physics/__init__.py`
- Create: `pyiron_workflow_assyst/analysis/__init__.py`
- Create: `pyiron_workflow_assyst/testing/__init__.py`
- Create: `pyiron_workflow_assyst/py.typed` (empty)
- Modify: `pyiron_workflow_assyst/__init__.py`

- [ ] **Step 1: Create the subpackage `__init__.py` files**

`pyiron_workflow_assyst/_internal/__init__.py`:
```python
"""Private plumbing for the ASSYST package. NOT part of the public API."""
```

`pyiron_workflow_assyst/structure/__init__.py`:
```python
"""Engine-agnostic structure operations for ASSYST."""

from .deformations import apply_rattle, apply_shear_strain, apply_triaxial_strain
from .filters import RCORE, filter_distance_by_species, get_minimum_distance, is_valid_structure
from .permutations import generate_assyst_permutations

__all__ = [
    "RCORE",
    "apply_rattle",
    "apply_shear_strain",
    "apply_triaxial_strain",
    "filter_distance_by_species",
    "generate_assyst_permutations",
    "get_minimum_distance",
    "is_valid_structure",
]
```

`pyiron_workflow_assyst/physics/__init__.py`:
```python
"""ASSYST physics workflows.

Import per topic; this package intentionally re-exports nothing::

    from pyiron_workflow_assyst.physics.relaxation import multistage_relax
    from pyiron_workflow_assyst.physics.assyst     import run_assyst
"""
```

`pyiron_workflow_assyst/analysis/__init__.py`:
```python
"""Post-processing on lists of EngineOutput."""

from .collect import collect_relaxation_frames
from .export import export_training_set

__all__ = ["collect_relaxation_frames", "export_training_set"]
```

`pyiron_workflow_assyst/testing/__init__.py`:
```python
"""Shared pytest fixtures for ASSYST consumers (optional)."""
```

`pyiron_workflow_assyst/py.typed`: empty file.

- [ ] **Step 2: Rewrite top-level `__init__.py` with lazy `__getattr__`**

Replace `pyiron_workflow_assyst/__init__.py` entirely:

```python
"""pyiron_workflow_assyst — ASSYST training-set generation workflows for pyiron.

The package follows ``pyiron_workflow_atomistics`` design principles: physics
workflows are engine-agnostic and route through the ``Engine`` Protocol;
structure operations are pure ASE-in/ASE-out; post-processing operates on
``EngineOutput``.

Subpackages
-----------
structure : engine-agnostic builders, deformations, filters, permutations
physics   : ``multistage_relax``, ``run_assyst`` macros
analysis  : trajectory frame collection, training-set export
testing   : shared pytest fixtures

Import the workflows per topic::

    from pyiron_workflow_assyst.physics.assyst import run_assyst
"""

from . import _version

__version__ = _version.get_versions()["version"]
__all__ = ["__version__"]


def __getattr__(name):
    """PEP 562 lazy loader — keeps build-time imports cheap.

    Without this, versioneer's ``[tool.setuptools.dynamic.version] attr =
    "pyiron_workflow_assyst.__version__"`` triggers a full package import
    at build time, cascading through ``from . import structure`` and
    pulling ``ase`` / ``pymatgen`` into the build-isolation env. See
    pyiron_workflow_atomistics' [[pwa-install-quirk]] for the historical fix.
    """
    if name in {"structure", "physics", "analysis", "testing"}:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```

- [ ] **Step 3: Verify package imports**

```bash
pixi run python -c "
import pyiron_workflow_assyst
print(pyiron_workflow_assyst.__version__)
print(pyiron_workflow_assyst.structure)  # triggers lazy load
"
```

Expected: prints `0.1.0` then an ImportError because `structure/filters.py` doesn't exist yet — that's fine; we just verify the lazy hook routes correctly. If it raises `ModuleNotFoundError: pyiron_workflow_assyst.structure.deformations` rather than the AttributeError, the scaffold is correct.

- [ ] **Step 4: Commit**

```bash
git add pyiron_workflow_assyst/__init__.py pyiron_workflow_assyst/_internal pyiron_workflow_assyst/structure pyiron_workflow_assyst/physics pyiron_workflow_assyst/analysis pyiron_workflow_assyst/testing pyiron_workflow_assyst/py.typed
git commit -m "scaffold: empty subpackages + PEP 562 lazy __getattr__"
```

### Task C2: `structure/filters.py` — port RCORE table and validators

**Files:**
- Create: `tests/unit/__init__.py` (empty)
- Create: `tests/unit/structure/__init__.py` (empty)
- Create: `tests/unit/structure/test_filters.py`
- Create: `pyiron_workflow_assyst/structure/filters.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/structure/test_filters.py`:
```python
"""Filter functions preserve the legacy ASSYST numerics."""

import importlib.util
import pathlib

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from pyiron_workflow_assyst.structure.filters import (
    RCORE,
    filter_distance_by_species,
    get_minimum_distance,
    is_valid_structure,
)


def _load_legacy_filters():
    """Import the frozen legacy module by path so tests can compare numerics."""
    p = pathlib.Path(__file__).resolve().parents[2] / "_legacy_assyst" / "structure_filter_utils.py"
    spec = importlib.util.spec_from_file_location("_legacy_filters", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRCORE:
    def test_table_bytes_identical_to_legacy(self):
        legacy = _load_legacy_filters()
        assert RCORE == legacy.RCORE


class TestMinimumDistance:
    def test_isolated_dimer(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[10, 10, 10], pbc=True)
        assert get_minimum_distance(atoms) == pytest.approx(0.8, rel=1e-6)

    def test_bulk_copper(self):
        cu = bulk("Cu", "fcc", a=3.6, cubic=True)
        d = get_minimum_distance(cu)
        assert d == pytest.approx(3.6 / np.sqrt(2), rel=1e-6)


class TestIsValidStructure:
    def test_relaxed_bulk_is_valid(self):
        cu = bulk("Cu", "fcc", a=3.6, cubic=True)
        assert is_valid_structure(cu, min_dist=1.0, core_overlap_tolerance=0.2) is True

    def test_too_close_is_invalid(self):
        atoms = Atoms("H2", positions=[[0, 0, 0], [0.1, 0, 0]], cell=[10, 10, 10], pbc=True)
        assert is_valid_structure(atoms, min_dist=1.0) is False

    def test_core_overlap_threshold(self):
        # Place two Cu atoms at exactly 0.79 * 2*RCORE_Cu apart — overlap-tolerance of
        # 0.2 (i.e. allowed_distance = 0.8 * sum) means this should be invalid.
        rcu = RCORE["Cu"]
        atoms = Atoms("Cu2", positions=[[0, 0, 0], [0.79 * 2 * rcu, 0, 0]], cell=[10, 10, 10], pbc=True)
        assert filter_distance_by_species(atoms, core_overlap_tolerance=0.2) is False
```

- [ ] **Step 2: Snapshot the legacy filter file into `tests/_legacy_assyst/`**

```bash
mkdir -p tests/_legacy_assyst
cp pyiron_workflow_assyst/structure_filter_utils.py tests/_legacy_assyst/
touch tests/_legacy_assyst/__init__.py
```

This freezes the reference for both `test_filters.py::TestRCORE` and the eventual VASP-equivalence test in Task C13.

- [ ] **Step 3: Run to verify it fails**

```bash
pixi run pytest tests/unit/structure/test_filters.py -q
```

Expected: ImportError on `from pyiron_workflow_assyst.structure.filters import ...`.

- [ ] **Step 4: Create `structure/filters.py`**

Copy the body of the legacy `structure_filter_utils.py` into `pyiron_workflow_assyst/structure/filters.py`, then change the **public** function signatures to accept `ase.Atoms` and convert to pymatgen Structure internally:

```python
"""Distance-based validity filters for ASSYST structure generation.

Public surface accepts ``ase.Atoms`` for consistency with the rest of the
package. Internally, the existing pymatgen-based ``get_all_neighbors``
algorithm is preserved bit-for-bit to keep numerics stable.
"""

from collections import defaultdict
from itertools import combinations_with_replacement

import numpy as np
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

RCORE = {
    # POTCAR RCORE × Bohr→Å. KEEP IN SYNC with the legacy
    # tests/_legacy_assyst/structure_filter_utils.py table — equivalence
    # test_filters.py::TestRCORE::test_table_bytes_identical_to_legacy
    # asserts byte equality.
    "H": 1.100000 * 0.5291773,
    "He": 1.100000 * 0.5291773,
    # ... (paste the rest verbatim from the legacy file) ...
}


def _to_pymatgen(atoms: Atoms):
    return AseAtomsAdaptor.get_structure(atoms)


def _element_wise_dist(structure):
    pair = defaultdict(lambda: np.inf)
    neighbors = structure.get_all_neighbors(r=5.0, include_index=True)
    for i, neighbor_list in enumerate(neighbors):
        for neighbor in neighbor_list:
            j, d = neighbor.index, neighbor.nn_distance
            ei, ej = sorted((structure[i].specie.symbol, structure[j].specie.symbol))
            pair[ei, ej] = min(d, pair[ei, ej])
    return pair


def get_minimum_distance(atoms: Atoms) -> float:
    """Minimum pair distance (Å) in the cell, excluding self-distance."""
    pmg = _to_pymatgen(atoms)
    dm = pmg.distance_matrix
    np.fill_diagonal(dm, np.inf)
    return float(np.min(dm))


def filter_distance_by_species(
    atoms: Atoms,
    *,
    rcore: dict = RCORE,
    core_overlap_tolerance: float = 0.2,
) -> bool:
    """True iff every species-pair distance ≥ (1 - tol) · (RCORE_i + RCORE_j)."""
    pmg = _to_pymatgen(atoms)
    if len(pmg) == 1:
        pmg = pmg * [2, 2, 2]
    pair = _element_wise_dist(pmg)
    species = sorted({site.specie.symbol for site in pmg})
    for ei, ej in combinations_with_replacement(species, 2):
        allowed = (1 - core_overlap_tolerance) * (rcore[ei] + rcore[ej])
        if pair[ei, ej] < allowed:
            return False
    return True


def is_valid_structure(
    atoms: Atoms,
    *,
    min_dist: float = 1.0,
    core_overlap_tolerance: float = 0.2,
) -> bool:
    """Combine min-dist and RCORE filters."""
    if get_minimum_distance(atoms) < min_dist:
        return False
    return filter_distance_by_species(atoms, core_overlap_tolerance=core_overlap_tolerance)
```

Paste the **entire** RCORE dictionary verbatim from the legacy file — every key and the same `* 0.5291773` factor.

- [ ] **Step 5: Run to verify it passes**

```bash
pixi run pytest tests/unit/structure/test_filters.py -q
```

Expected: all 6 tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/structure/test_filters.py tests/_legacy_assyst pyiron_workflow_assyst/structure/filters.py
git commit -m "feat(structure/filters): port RCORE + validators with ASE-Atoms surface"
```

### Task C3: `structure/deformations.py` — pure ASE deformations, seeded RNG

**Files:**
- Create: `tests/unit/structure/test_deformations.py`
- Create: `pyiron_workflow_assyst/structure/deformations.py`

- [ ] **Step 1: Generate golden values from the legacy code**

Run this one-shot snippet inside the env to produce the reference arrays the new tests will compare against:

```bash
pixi run python - <<'EOF'
import numpy as np
import sys, pathlib
sys.path.insert(0, str(pathlib.Path("tests/_legacy_assyst")))
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import bulk

# legacy code uses module-level numpy random — seed it
np.random.seed(42)
atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
pmg = AseAtomsAdaptor.get_structure(atoms)

# Import the deformation helpers from the legacy workflow.py (snapshot first):
EOF
```

Then snapshot the legacy `workflow.py` so the script can import from it (the equivalence test in Task C13 needs this snapshot too):

```bash
cp pyiron_workflow_assyst/workflow.py tests/_legacy_assyst/workflow.py
```

Now generate the golden values:

```bash
pixi run python - <<'EOF' > /tmp/assyst_golden.py
import numpy as np
import pathlib, sys
sys.path.insert(0, str(pathlib.Path("tests/_legacy_assyst")))
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor

# Imports from the snapshot
from workflow import apply_rattle as legacy_rattle
from workflow import apply_shear_strain as legacy_shear
from workflow import apply_triaxial_strain as legacy_triax

atoms = bulk("Cu", "fcc", a=3.6, cubic=True)
pmg = AseAtomsAdaptor.get_structure(atoms)

np.random.seed(42)
r = legacy_rattle(pmg, displacement=0.1, max_cell_strain=0.05)
np.random.seed(42)
t = legacy_triax(pmg, max_strain=0.05)
np.random.seed(42)
s = legacy_shear(pmg, max_strain=0.05)

print("RATTLE_POSITIONS =", repr(np.array([site.coords for site in r]).tolist()))
print("RATTLE_CELL = ", repr(np.array(r.lattice.matrix).tolist()))
print("TRIAX_CELL = ", repr(np.array(t.lattice.matrix).tolist()))
print("SHEAR_CELL = ", repr(np.array(s.lattice.matrix).tolist()))
EOF
cat /tmp/assyst_golden.py
```

Paste the printed arrays into the test file in Step 2 as Python literals. **This is the bit-for-bit equivalence guarantor — do not skip.**

- [ ] **Step 2: Write the failing test**

Create `tests/unit/structure/test_deformations.py`:
```python
"""Deformations preserve the legacy ASSYST distributions bit-for-bit when the
RNG is seeded identically."""

import numpy as np
import pytest
from ase.build import bulk

from pyiron_workflow_assyst.structure.deformations import (
    apply_rattle,
    apply_shear_strain,
    apply_triaxial_strain,
)


# Paste from /tmp/assyst_golden.py output in Step 1:
RATTLE_POSITIONS = ...   # noqa: REPLACE WITH GOLDEN
RATTLE_CELL = ...        # noqa: REPLACE WITH GOLDEN
TRIAX_CELL = ...         # noqa: REPLACE WITH GOLDEN
SHEAR_CELL = ...         # noqa: REPLACE WITH GOLDEN


@pytest.fixture
def cu():
    return bulk("Cu", "fcc", a=3.6, cubic=True)


class TestRattle:
    def test_matches_legacy_with_seed_42(self, cu):
        rng = np.random.default_rng(42)
        out = apply_rattle(cu, displacement=0.1, max_cell_strain=0.05, rng=rng)
        np.testing.assert_allclose(out.get_positions(), RATTLE_POSITIONS, atol=1e-10)
        np.testing.assert_allclose(out.cell.array, RATTLE_CELL, atol=1e-10)


class TestTriaxialStrain:
    def test_matches_legacy_with_seed_42(self, cu):
        rng = np.random.default_rng(42)
        out = apply_triaxial_strain(cu, max_strain=0.05, rng=rng)
        np.testing.assert_allclose(out.cell.array, TRIAX_CELL, atol=1e-10)


class TestShearStrain:
    def test_matches_legacy_with_seed_42(self, cu):
        rng = np.random.default_rng(42)
        out = apply_shear_strain(cu, max_strain=0.05, rng=rng)
        np.testing.assert_allclose(out.cell.array, SHEAR_CELL, atol=1e-10)


class TestRNGContract:
    def test_no_rng_does_not_crash(self, cu):
        apply_rattle(cu)
        apply_triaxial_strain(cu)
        apply_shear_strain(cu)
```

> Note on the seed contract: the legacy code calls `np.random.normal(...)` / `np.random.uniform(...)` against the **module-level** numpy random state. The new code uses a `numpy.random.Generator` passed in explicitly. The golden values can be reproduced by seeding the legacy module's `np.random` state and the new code's `Generator` with the same seed, **but the underlying algorithms (`Generator.normal` vs `np.random.normal`) produce different sequences**. So we cannot expect byte equality of golden arrays generated from legacy global vs new `Generator`.
>
> Resolution: the test uses `apply_rattle(rng=np.random.default_rng(42))` and the golden arrays are generated **with the same `Generator`-based code path**. Specifically, write the new module first with a clear contract — `rng: np.random.Generator | None` → `rng.normal(...)` / `rng.uniform(...)` — and use it to **regenerate the golden values** by running an in-tree script. The equivalence with the legacy distribution at the distribution level is then asserted separately by a **statistical test** (large-sample mean/std checks), not by bit-for-bit comparison.
>
> Update Step 1: instead of running the legacy code to harvest golden values, write the new code first, then run a one-shot script using the new code (with seed 42) and paste those outputs as golden. The bit-for-bit guarantee is between the new code and itself across releases.
>
> Add to the same test file:
>
> ```python
> class TestDistributionMatchesLegacy:
>     """Statistical-level equivalence: 1e4 samples agree to 3σ with legacy."""
>
>     def test_rattle_displacement_distribution(self, cu):
>         from tests._legacy_assyst.workflow import apply_rattle as legacy_rattle
>         from pymatgen.io.ase import AseAtomsAdaptor
>         import numpy as np
>
>         pmg = AseAtomsAdaptor.get_structure(cu)
>         legacy_disp = []
>         np.random.seed(0)
>         for _ in range(1000):
>             r = legacy_rattle(pmg, displacement=0.1, max_cell_strain=0.0)
>             legacy_disp.append(np.linalg.norm(np.array([s.coords for s in r]) - np.array([s.coords for s in pmg])))
>         legacy_mean = np.mean(legacy_disp)
>
>         rng = np.random.default_rng(0)
>         new_disp = []
>         for _ in range(1000):
>             r = apply_rattle(cu, displacement=0.1, max_cell_strain=0.0, rng=rng)
>             new_disp.append(np.linalg.norm(r.get_positions() - cu.get_positions()))
>         new_mean = np.mean(new_disp)
>
>         # σ ≈ 0.1 * √(3N) where N = 4 ⇒ σ_mean ≈ 0.1·√12 / √1000 ≈ 0.011
>         assert abs(legacy_mean - new_mean) < 3 * 0.011
> ```

- [ ] **Step 3: Run to verify it fails**

```bash
pixi run pytest tests/unit/structure/test_deformations.py -q
```

Expected: ImportError — module doesn't exist.

- [ ] **Step 4: Implement `structure/deformations.py`**

```python
"""Pure-ASE structural deformations for ASSYST permutations.

All functions accept and return ``ase.Atoms``. Distributions match the legacy
ASSYST code at the statistical level (verified by
``tests/unit/structure/test_deformations.py::TestDistributionMatchesLegacy``);
identity within the new code is guaranteed bit-for-bit given the same
``rng`` (verified by the golden-value tests).
"""

from __future__ import annotations

import numpy as np
from ase import Atoms


def _resolve_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def apply_rattle(
    atoms: Atoms,
    *,
    displacement: float = 0.1,
    max_cell_strain: float = 0.05,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Gaussian per-atom displacements + uniform diagonal cell strain."""
    rng = _resolve_rng(rng)
    out = atoms.copy()
    positions = out.get_positions()
    positions += rng.normal(0.0, displacement, size=positions.shape)
    out.set_positions(positions)
    strain_vec = 1.0 + rng.uniform(-max_cell_strain, max_cell_strain, size=3)
    cell = out.cell.array @ np.diag(strain_vec)
    out.set_cell(cell, scale_atoms=True)
    return out


def apply_triaxial_strain(
    atoms: Atoms,
    *,
    max_strain: float = 0.8,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Uniform-random diagonal strain in U(-max_strain, +max_strain)."""
    rng = _resolve_rng(rng)
    out = atoms.copy()
    strain_vec = 1.0 + rng.uniform(-max_strain, max_strain, size=3)
    cell = out.cell.array @ np.diag(strain_vec)
    out.set_cell(cell, scale_atoms=True)
    return out


def apply_shear_strain(
    atoms: Atoms,
    *,
    max_strain: float = 0.8,
    rng: np.random.Generator | None = None,
) -> Atoms:
    """Uniform-random full 3×3 strain with diagonal forced to 1 (shear-only)."""
    rng = _resolve_rng(rng)
    out = atoms.copy()
    shear = np.eye(3) + rng.uniform(-max_strain, max_strain, size=(3, 3))
    np.fill_diagonal(shear, 1.0)
    cell = out.cell.array @ shear
    out.set_cell(cell, scale_atoms=True)
    return out
```

- [ ] **Step 5: Regenerate golden values using the new code**

```bash
pixi run python - <<'EOF'
import numpy as np
from ase.build import bulk
from pyiron_workflow_assyst.structure.deformations import (
    apply_rattle, apply_triaxial_strain, apply_shear_strain,
)

cu = bulk("Cu", "fcc", a=3.6, cubic=True)
rng = np.random.default_rng(42)
r = apply_rattle(cu, displacement=0.1, max_cell_strain=0.05, rng=rng)
print("RATTLE_POSITIONS =", repr(r.get_positions().tolist()))
print("RATTLE_CELL =", repr(r.cell.array.tolist()))

rng = np.random.default_rng(42)
t = apply_triaxial_strain(cu, max_strain=0.05, rng=rng)
print("TRIAX_CELL =", repr(t.cell.array.tolist()))

rng = np.random.default_rng(42)
s = apply_shear_strain(cu, max_strain=0.05, rng=rng)
print("SHEAR_CELL =", repr(s.cell.array.tolist()))
EOF
```

Paste the printed Python literals into `tests/unit/structure/test_deformations.py` at the `# REPLACE WITH GOLDEN` markers.

- [ ] **Step 6: Run to verify all deformation tests pass**

```bash
pixi run pytest tests/unit/structure/test_deformations.py -q
```

Expected: all tests pass (3 golden + 1 RNG-contract + 1 statistical = 5).

- [ ] **Step 7: Commit**

```bash
git add tests/unit/structure/test_deformations.py tests/_legacy_assyst/workflow.py pyiron_workflow_assyst/structure/deformations.py
git commit -m "feat(structure/deformations): ASE-native rattle/triax/shear with seeded RNG"
```

### Task C4: `structure/permutations.py` — generate_assyst_permutations

**Files:**
- Create: `tests/unit/structure/test_permutations.py`
- Create: `pyiron_workflow_assyst/structure/permutations.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/structure/test_permutations.py`:
```python
"""generate_assyst_permutations produces the right shape / names / validity."""

import numpy as np
import pytest
from ase.build import bulk

from pyiron_workflow_assyst.structure.filters import is_valid_structure
from pyiron_workflow_assyst.structure.permutations import generate_assyst_permutations


@pytest.fixture
def two_bases():
    cu = bulk("Cu", "fcc", a=3.6, cubic=True)
    return [cu, cu.copy()], ["pyxtal_0", "pyxtal_1"]


class TestShape:
    def test_count_per_category(self, two_bases):
        bases, names = two_bases
        atoms, perm_names = generate_assyst_permutations.node_function(
            base_structures=bases, base_names=names,
            n_rattle=2, n_triaxial=2, n_shear=2,
            rattle_displacement=0.05, rattle_cell_strain=0.02,
            triaxial_strain=0.05, shear_strain=0.05,
            seed=7,
        )
        # 2 bases × (2 rattle + 2 triax + 2 shear) = 12
        assert len(atoms) == 12
        assert len(perm_names) == 12


class TestNames:
    def test_name_scheme(self, two_bases):
        bases, names = two_bases
        _, perm_names = generate_assyst_permutations.node_function(
            base_structures=bases, base_names=names,
            n_rattle=1, n_triaxial=1, n_shear=1,
            rattle_displacement=0.05, rattle_cell_strain=0.02,
            triaxial_strain=0.05, shear_strain=0.05,
            seed=7,
        )
        # Order: for each base, rattle then triax then shear
        assert perm_names == [
            "pyxtal_0_rattle_1", "pyxtal_0_triax_1", "pyxtal_0_shear_1",
            "pyxtal_1_rattle_1", "pyxtal_1_triax_1", "pyxtal_1_shear_1",
        ]


class TestValidity:
    def test_all_outputs_pass_filter(self, two_bases):
        bases, names = two_bases
        atoms, _ = generate_assyst_permutations.node_function(
            base_structures=bases, base_names=names,
            n_rattle=2, n_triaxial=2, n_shear=2,
            rattle_displacement=0.05, rattle_cell_strain=0.02,
            triaxial_strain=0.05, shear_strain=0.05,
            min_dist=1.0, core_overlap_tolerance=0.2,
            seed=7,
        )
        for a in atoms:
            assert is_valid_structure(a, min_dist=1.0, core_overlap_tolerance=0.2)


class TestDeterminism:
    def test_same_seed_same_output(self, two_bases):
        bases, names = two_bases
        a1, n1 = generate_assyst_permutations.node_function(
            base_structures=bases, base_names=names,
            n_rattle=1, n_triaxial=1, n_shear=1, seed=99,
        )
        a2, n2 = generate_assyst_permutations.node_function(
            base_structures=bases, base_names=names,
            n_rattle=1, n_triaxial=1, n_shear=1, seed=99,
        )
        assert n1 == n2
        for x, y in zip(a1, a2):
            np.testing.assert_array_equal(x.get_positions(), y.get_positions())
            np.testing.assert_array_equal(x.cell.array, y.cell.array)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pixi run pytest tests/unit/structure/test_permutations.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `structure/permutations.py`**

```python
"""ASSYST permutation generator: rattle + triaxial + shear, each validated."""

from __future__ import annotations

import numpy as np
import pyiron_workflow as pwf
from ase import Atoms

from .deformations import apply_rattle, apply_shear_strain, apply_triaxial_strain
from .filters import is_valid_structure


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
    """For each base structure, generate ``n_rattle`` + ``n_triaxial`` + ``n_shear``
    valid permutations. Each candidate is rejected if ``is_valid_structure``
    returns False; up to ``max_attempts_per_perm`` retries per slot before
    giving up.

    The output order is rattle → triax → shear per base, matching the legacy
    ASSYST naming and downstream consumers' expectations.
    """
    rng = np.random.default_rng(seed)
    out_atoms: list[Atoms] = []
    out_names: list[str] = []
    for atoms, base_name in zip(base_structures, base_names):
        out_atoms.extend(_fill(atoms, n_rattle, rng, "rattle", base_name,
                               apply_rattle, dict(displacement=rattle_displacement,
                                                  max_cell_strain=rattle_cell_strain),
                               min_dist, core_overlap_tolerance, max_attempts_per_perm,
                               out_names))
        out_atoms.extend(_fill(atoms, n_triaxial, rng, "triax", base_name,
                               apply_triaxial_strain, dict(max_strain=triaxial_strain),
                               min_dist, core_overlap_tolerance, max_attempts_per_perm,
                               out_names))
        out_atoms.extend(_fill(atoms, n_shear, rng, "shear", base_name,
                               apply_shear_strain, dict(max_strain=shear_strain),
                               min_dist, core_overlap_tolerance, max_attempts_per_perm,
                               out_names))
    return out_atoms, out_names


def _fill(base, n, rng, tag, base_name, fn, kwargs, min_dist, tol, max_attempts, name_acc):
    accepted: list[Atoms] = []
    attempts = 0
    while len(accepted) < n and attempts < max_attempts * n:
        candidate = fn(base, rng=rng, **kwargs)
        if is_valid_structure(candidate, min_dist=min_dist, core_overlap_tolerance=tol):
            accepted.append(candidate)
            name_acc.append(f"{base_name}_{tag}_{len(accepted)}")
        attempts += 1
    return accepted
```

- [ ] **Step 4: Run to verify it passes**

```bash
pixi run pytest tests/unit/structure/test_permutations.py -q
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/structure/test_permutations.py pyiron_workflow_assyst/structure/permutations.py
git commit -m "feat(structure/permutations): generate_assyst_permutations function node"
```

### Task C5: `structure/generate.py` — PyXtal random crystals (optional)

**Files:**
- Create: `tests/unit/structure/test_generate.py`
- Create: `pyiron_workflow_assyst/structure/generate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/structure/test_generate.py`:
```python
import pytest

pyxtal = pytest.importorskip("pyxtal")

from pyiron_workflow_assyst.structure.generate import pyxtal_random_crystals


class TestPyXtal:
    def test_generates_requested_count(self, tmp_path):
        atoms_list, names = pyxtal_random_crystals.node_function(
            composition={"Cu": 4},
            n_structures=2,
            space_groups=[225],  # Fm-3m
            name_prefix="cu",
            seed=1,
        )
        assert len(atoms_list) == 2
        assert names == ["cu_0", "cu_1"]

    def test_missing_pyxtal_raises_friendly(self, monkeypatch):
        from pyiron_workflow_assyst.structure import generate

        monkeypatch.setattr(generate, "_PYXTAL_AVAILABLE", False)
        with pytest.raises(ImportError, match=r"\[pyxtal\]"):
            pyxtal_random_crystals.node_function(composition={"Cu": 4}, n_structures=1)
```

- [ ] **Step 2: Run to verify it fails (or is skipped)**

```bash
pixi run pytest tests/unit/structure/test_generate.py -q
```

Expected: skipped if pyxtal isn't installed; otherwise ImportError on the assyst side.

- [ ] **Step 3: Install pyxtal in the env**

```bash
pixi add pyxtal
# or
pixi run python -m uv pip install pyxtal
```

- [ ] **Step 4: Implement `structure/generate.py`**

```python
"""PyXtal random crystal generation (optional extra)."""

from __future__ import annotations

import pyiron_workflow as pwf
from ase import Atoms

try:
    from pyxtal import pyxtal as _PyXtal  # type: ignore[import-untyped]

    _PYXTAL_AVAILABLE = True
except ImportError:
    _PYXTAL_AVAILABLE = False


@pwf.as_function_node("structures", "names")
def pyxtal_random_crystals(
    composition: dict[str, int],
    n_structures: int = 100,
    space_groups: list[int] | range = range(1, 231),
    volume_factor: float = 1.0,
    name_prefix: str = "pyxtal",
    seed: int | None = None,
) -> tuple[list[Atoms], list[str]]:
    """Sample ``n_structures`` random crystal structures across the requested
    space groups using PyXtal.

    Requires the optional ``[pyxtal]`` extra. ``composition`` maps element
    symbols to atom counts in the unit cell (e.g. ``{"Cu": 4}``).
    """
    if not _PYXTAL_AVAILABLE:
        raise ImportError(
            "PyXtal is required for random crystal generation. "
            "Install with: pip install pyiron_workflow_assyst[pyxtal]"
        )

    import random

    if seed is not None:
        random.seed(seed)

    species = list(composition.keys())
    numIons = list(composition.values())
    atoms_list: list[Atoms] = []
    names: list[str] = []
    sg_list = list(space_groups)

    while len(atoms_list) < n_structures:
        sg = random.choice(sg_list)
        struct = _PyXtal()
        try:
            struct.from_random(
                dim=3, group=sg, species=species, numIons=numIons,
                factor=volume_factor,
            )
        except Exception:
            continue
        if not struct.valid:
            continue
        atoms_list.append(struct.to_ase())
        names.append(f"{name_prefix}_{len(atoms_list) - 1}")

    return atoms_list, names
```

- [ ] **Step 5: Run to verify it passes**

```bash
pixi run pytest tests/unit/structure/test_generate.py -q
```

Expected: 2 tests pass.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/structure/test_generate.py pyiron_workflow_assyst/structure/generate.py
git commit -m "feat(structure/generate): PyXtal random-crystal generator (optional dep)"
```

### Task C6: `analysis/collect.py` — collect_relaxation_frames

**Files:**
- Create: `tests/unit/__init__.py` (if not present)
- Create: `tests/unit/analysis/__init__.py` (empty)
- Create: `tests/unit/analysis/test_collect.py`
- Create: `pyiron_workflow_assyst/analysis/collect.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/analysis/test_collect.py`:
```python
"""collect_relaxation_frames threshold + convergence semantics."""

import numpy as np
import pytest
from ase.build import bulk

from pyiron_workflow_atomistics.engine import EngineOutput

from pyiron_workflow_assyst.analysis.collect import (
    _select_indices_by_threshold,
    collect_relaxation_frames,
)


@pytest.fixture
def trajectory_4_frames():
    """Synthetic EngineOutput with 4 trajectory frames, monotonically descending
    energy by 0.05 eV per frame for a 2-atom cell ⇒ 0.025 eV/atom per frame."""
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True)[:2]
    energies = [-3.50, -3.55, -3.60, -3.65]
    structures = [atoms.copy() for _ in range(4)]
    return EngineOutput(
        final_structure=structures[-1],
        final_energy=energies[-1],
        converged=True,
        energies=energies,
        forces=[np.zeros((2, 3))] * 4,
        structures=structures,
        n_ionic_steps=4,
    )


class TestSelectIndicesByThreshold:
    def test_negative_one_returns_last_only(self):
        assert _select_indices_by_threshold([0.0, 1.0, 2.0, 3.0], threshold=-1.0) == [3]

    def test_positive_threshold_selects_first_last_and_jumps(self):
        # First (0), then |1-0|=1 > 0.5 ⇒ 1, then |2-1|=1 > 0.5 ⇒ 2, last (3) already in
        assert _select_indices_by_threshold([0.0, 1.0, 2.0, 3.0], threshold=0.5) == [0, 1, 2, 3]


class TestCollectFrames:
    def test_last_only_default(self, trajectory_4_frames):
        out = collect_relaxation_frames.node_function(
            engine_output=trajectory_4_frames, base_name="ISIF2",
        )
        # Returns (structures, energies, names, indices)
        atoms, energies, names, indices = out
        assert len(atoms) == 1
        assert energies == [-3.65]
        assert names == ["ISIF2_accur_relaxstep3"]
        assert indices == [3]

    def test_threshold_picks_all_frames(self, trajectory_4_frames):
        atoms, energies, names, indices = collect_relaxation_frames.node_function(
            engine_output=trajectory_4_frames, base_name="ISIF2",
            image_selection_eV_atom_threshold=0.01,
        )
        # 0.025 eV/atom delta per frame > 0.01 ⇒ pick all 4
        assert indices == [0, 1, 2, 3]
        assert names == [f"ISIF2_accur_relaxstep{i}" for i in [0, 1, 2, 3]]


class TestConvergenceFilter:
    def test_drops_unconverged_when_required(self, trajectory_4_frames):
        # Mark the synthetic EngineOutput as not converged
        trajectory_4_frames.converged = False
        atoms, energies, names, indices = collect_relaxation_frames.node_function(
            engine_output=trajectory_4_frames, base_name="ISIF2",
            require_converged=True,
        )
        assert atoms == []
        assert energies == []
        assert names == []
        assert indices == []

    def test_keeps_unconverged_when_not_required(self, trajectory_4_frames):
        trajectory_4_frames.converged = False
        atoms, _, _, _ = collect_relaxation_frames.node_function(
            engine_output=trajectory_4_frames, base_name="ISIF2",
            require_converged=False,
        )
        assert len(atoms) == 1
```

- [ ] **Step 2: Run to verify it fails**

```bash
pixi run pytest tests/unit/analysis/test_collect.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `analysis/collect.py`**

```python
"""Harvest training frames from a relaxation EngineOutput.

Replaces the legacy ``collect_structures`` + ``select_indices_by_threshold``
helpers from ``pyiron_workflow_assyst/workflow.py``. Operates on the canonical
``EngineOutput`` instead of a VASP-parser DataFrame.
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

    ``image_selection_eV_atom_threshold == -1`` returns only the last frame
    (matches legacy default). Positive values select first/last plus any
    intermediate frame whose eV/atom differs from the previously selected
    frame by more than the threshold.

    If ``require_converged`` is True and the engine reports
    ``converged=False``, returns four empty lists (matches legacy SCF-filter
    behavior).
    """
    if require_converged and not engine_output.converged:
        return [], [], [], []

    if not engine_output.structures or not engine_output.energies:
        # Static / no trajectory → final values only.
        return [engine_output.final_structure], [engine_output.final_energy], [
            f"{base_name}_accur_relaxstep0"
        ], [0]

    energies = engine_output.energies
    structures = engine_output.structures
    n_atoms = len(structures[0])
    eV_atom = [e / n_atoms for e in energies]
    indices = _select_indices_by_threshold(eV_atom, image_selection_eV_atom_threshold)

    out_atoms = [structures[i] for i in indices]
    out_energies = [energies[i] for i in indices]
    out_names = [f"{base_name}_accur_relaxstep{i}" for i in indices]
    return out_atoms, out_energies, out_names, indices
```

- [ ] **Step 4: Run to verify it passes**

```bash
pixi run pytest tests/unit/analysis/test_collect.py -q
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/analysis pyiron_workflow_assyst/analysis/collect.py
git commit -m "feat(analysis/collect): collect_relaxation_frames + threshold helper"
```

### Task C7: `analysis/export.py` — export_training_set

**Files:**
- Create: `tests/unit/analysis/test_export.py`
- Create: `pyiron_workflow_assyst/analysis/export.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/analysis/test_export.py`:
```python
"""export_training_set: pickle_df schema + extxyz round-trip."""

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk
from ase.io import read as ase_read

from pyiron_workflow_atomistics.engine import EngineOutput

from pyiron_workflow_assyst.analysis.export import export_training_set


@pytest.fixture
def two_outputs():
    a = bulk("Cu", "fcc", a=3.6, cubic=True)
    out = [
        EngineOutput(
            final_structure=a, final_energy=-3.5, converged=True,
            final_forces=np.zeros((4, 3)),
            final_stress_voigt=np.zeros(6),
            final_volume=a.get_volume(),
        ),
        EngineOutput(
            final_structure=a, final_energy=-3.6, converged=True,
            final_forces=np.zeros((4, 3)),
            final_stress_voigt=np.zeros(6),
            final_volume=a.get_volume(),
        ),
    ]
    return out, ["base_0", "perm_0"]


class TestPickleDf:
    def test_schema(self, two_outputs, tmp_path):
        outs, names = two_outputs
        path = tmp_path / "train.pkl"
        export_training_set.node_function(
            engine_outputs=outs, names=names,
            path=str(path), format="pickle_df",
        )
        df = pd.read_pickle(path)
        assert list(df.columns) == ["name", "energy", "volume", "converged", "structure", "forces", "stress"]
        assert len(df) == 2
        assert df.iloc[0]["name"] == "base_0"
        assert df.iloc[1]["energy"] == pytest.approx(-3.6)


class TestExtxyz:
    def test_round_trip(self, two_outputs, tmp_path):
        outs, names = two_outputs
        path = tmp_path / "train.xyz"
        export_training_set.node_function(
            engine_outputs=outs, names=names,
            path=str(path), format="extxyz",
        )
        atoms_list = ase_read(str(path), index=":")
        assert len(atoms_list) == 2
        assert atoms_list[0].info["energy"] == pytest.approx(-3.5)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pixi run pytest tests/unit/analysis/test_export.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `analysis/export.py`**

```python
"""Training-set export — pickle DataFrame, extended XYZ, or ASE-DB."""

from __future__ import annotations

from typing import Literal

import pandas as pd
import pyiron_workflow as pwf
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

    All formats produce one record per ``EngineOutput`` (one frame per row).
    The ``pickle_df`` schema is a clean rewrite of the legacy
    one-row-per-job list-valued schema — see the spec §10.
    """
    if len(engine_outputs) != len(names):
        raise ValueError(
            f"engine_outputs ({len(engine_outputs)}) and names ({len(names)}) "
            "must have the same length"
        )

    if format == "pickle_df":
        rows = []
        for name, out in zip(names, engine_outputs):
            rows.append({
                "name": name,
                "energy": out.final_energy,
                "volume": out.final_volume,
                "converged": out.converged,
                "structure": out.final_structure,
                "forces": out.final_forces,
                "stress": out.final_stress_voigt,
            })
        pd.DataFrame(rows, columns=[
            "name", "energy", "volume", "converged",
            "structure", "forces", "stress",
        ]).to_pickle(path)
        return path

    if format == "extxyz":
        atoms_list = []
        for name, out in zip(names, engine_outputs):
            a = out.final_structure.copy()
            a.info["energy"] = out.final_energy
            a.info["name"] = name
            if out.final_forces is not None:
                a.set_array("forces", out.final_forces)
            atoms_list.append(a)
        ase_write(path, atoms_list, format="extxyz")
        return path

    if format == "ase_db":
        from ase.db import connect

        with connect(path) as db:
            for name, out in zip(names, engine_outputs):
                db.write(
                    out.final_structure,
                    name=name,
                    energy=out.final_energy,
                    converged=out.converged,
                )
        return path

    raise ValueError(f"Unknown export format: {format!r}")
```

- [ ] **Step 4: Run to verify it passes**

```bash
pixi run pytest tests/unit/analysis/test_export.py -q
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/analysis/test_export.py pyiron_workflow_assyst/analysis/export.py
git commit -m "feat(analysis/export): pickle_df / extxyz / ase_db exporters"
```

### Task C8: `_internal/engine_fanout.py` — sub-engine fan-out helpers

**Files:**
- Create: `tests/unit/_internal/__init__.py`
- Create: `tests/unit/_internal/test_engine_fanout.py`
- Create: `pyiron_workflow_assyst/_internal/engine_fanout.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/_internal/test_engine_fanout.py`:
```python
"""_build_subengines composes per-name working_directories purely."""

import os

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic
from ase.calculators.emt import EMT

from pyiron_workflow_assyst._internal.engine_fanout import _build_subengines, _concat


def test_build_subengines_one_per_name(tmp_path):
    parent = ASEEngine(EngineInput=CalcInputStatic(), calculator=EMT(), working_directory=str(tmp_path))
    subs = _build_subengines.node_function(engine=parent, names=["a", "b", "c"])
    assert [s.working_directory for s in subs] == [
        os.path.join(str(tmp_path), "a"),
        os.path.join(str(tmp_path), "b"),
        os.path.join(str(tmp_path), "c"),
    ]
    # Parent untouched.
    assert parent.working_directory == str(tmp_path)


def test_concat():
    assert _concat.node_function(a=[1, 2], b=[3, 4]) == [1, 2, 3, 4]
```

- [ ] **Step 2: Run to verify it fails**

```bash
pixi run pytest tests/unit/_internal/test_engine_fanout.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement the helpers**

```python
"""Tiny function-node helpers for fanning out engines / concatenating lists
inside a pyiron_workflow macro graph.

These are private because they exist purely to bridge channel-bound values
into ordinary Python lists at graph-execution time — list comprehensions
don't dispatch through the channel ``__call__`` correctly.
"""

from __future__ import annotations

import pyiron_workflow as pwf

from pyiron_workflow_atomistics.engine import Engine


@pwf.as_function_node("subengines")
def _build_subengines(engine: Engine, names: list[str]) -> list[Engine]:
    return [engine.with_working_directory(n) for n in names]


@pwf.as_function_node("concatenated")
def _concat(a: list, b: list) -> list:
    return list(a) + list(b)
```

- [ ] **Step 4: Run to verify it passes**

```bash
pixi run pytest tests/unit/_internal/test_engine_fanout.py -q
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/_internal pyiron_workflow_assyst/_internal/engine_fanout.py
git commit -m "feat(_internal/engine_fanout): _build_subengines and _concat helpers"
```

### Task C9: `physics/relaxation.py` — multistage_relax macro

**Files:**
- Create: `tests/unit/physics/__init__.py`
- Create: `tests/unit/physics/test_relaxation.py`
- Create: `pyiron_workflow_assyst/physics/relaxation.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/physics/test_relaxation.py`:
```python
"""multistage_relax defaults match the ASSYST ISIF=7/5/2 cascade."""

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize

from pyiron_workflow_assyst.physics.relaxation import multistage_relax


def test_default_stages_are_volume_shape_none():
    # Inspect the macro's default via a probing run with a dry-fire engine —
    # easier to assert on a constructed graph than to run EMT on ISIF7/5
    # (which ASE doesn't support). Use the macro's _default_stages helper.
    from pyiron_workflow_assyst.physics.relaxation import _default_stages, _default_stage_names

    stages = _default_stages()
    names = _default_stage_names()
    assert [s.cell_relaxation for s in stages] == ["volume", "shape", "none"]
    assert names == ["ISIF7", "ISIF5", "ISIF2"]
    assert all(isinstance(s, CalcInputMinimize) for s in stages)


def test_runs_one_stage_with_ase_full(tmp_path):
    """ASE only supports 'full' and 'none' — verify the macro composes correctly
    when the caller supplies a single-stage chain."""
    engine = ASEEngine(
        EngineInput=CalcInputMinimize(cell_relaxation="full", max_iterations=3),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    out = multistage_relax(
        structure=bulk("Cu", "fcc", a=3.6, cubic=True),
        engine=engine,
        stages=[CalcInputMinimize(cell_relaxation="full", max_iterations=3)],
        stage_names=["fullrelax"],
    )
    out.run()
    assert len(out.outputs.engine_outputs.value) == 1
    assert out.outputs.final_structure.value is not None
```

- [ ] **Step 2: Run to verify it fails**

```bash
pixi run pytest tests/unit/physics/test_relaxation.py -q
```

Expected: ImportError.

- [ ] **Step 3: Implement `physics/relaxation.py`**

```python
"""Multi-stage structural relaxation macro.

Default chain mirrors the ASSYST methodology — volume relax → shape relax →
ion relax (ISIF=7/5/2 under VaspEngine). Callers may override ``stages`` and
``stage_names`` for any other cascade. The macro feeds each stage's final
structure into the next stage's input automatically and runs each stage
inside its own subdirectory under ``engine.working_directory``.
"""

from __future__ import annotations

from dataclasses import replace

import pyiron_workflow as pwf
from ase import Atoms

from pyiron_workflow_atomistics.engine import (
    CalcInputMinimize,
    Engine,
    EngineOutput,
    calculate,
)


def _default_stages() -> list[CalcInputMinimize]:
    return [
        CalcInputMinimize(cell_relaxation="volume"),
        CalcInputMinimize(cell_relaxation="shape"),
        CalcInputMinimize(cell_relaxation="none"),
    ]


def _default_stage_names() -> list[str]:
    return ["ISIF7", "ISIF5", "ISIF2"]


@pwf.as_function_node("sub_engine")
def _replace_engine_input_and_subdir(
    engine: Engine, stage: CalcInputMinimize, subdir: str
) -> Engine:
    """Return a copy of ``engine`` with ``EngineInput=stage`` and
    ``working_directory`` extended by ``subdir``."""
    return replace(engine, EngineInput=stage).with_working_directory(subdir)


@pwf.as_macro_node("engine_outputs", "final_structure", "converged")
def multistage_relax(
    wf,
    structure: Atoms,
    engine: Engine,
    stages: list[CalcInputMinimize] | None = None,
    stage_names: list[str] | None = None,
):
    if stages is None:
        stages = _default_stages()
    if stage_names is None:
        stage_names = _default_stage_names()
    if len(stages) != len(stage_names):
        raise ValueError(
            f"stages ({len(stages)}) and stage_names ({len(stage_names)}) "
            "must have the same length"
        )

    # We can't unroll a variable-length loop inside a macro graph; restrict
    # to N=3 for the default (and provide a helper for N=1 used by tests).
    if len(stages) == 1:
        wf.engine_0 = _replace_engine_input_and_subdir(engine, stages[0], stage_names[0])
        wf.calc_0 = calculate(structure=structure, engine=wf.engine_0)
        engine_outputs = [wf.calc_0]
        return engine_outputs, wf.calc_0.outputs.engine_output.final_structure, wf.calc_0.outputs.engine_output.converged

    if len(stages) == 3:
        wf.engine_0 = _replace_engine_input_and_subdir(engine, stages[0], stage_names[0])
        wf.calc_0 = calculate(structure=structure, engine=wf.engine_0)

        wf.engine_1 = _replace_engine_input_and_subdir(engine, stages[1], stage_names[1])
        wf.calc_1 = calculate(structure=wf.calc_0.outputs.engine_output.final_structure, engine=wf.engine_1)

        wf.engine_2 = _replace_engine_input_and_subdir(engine, stages[2], stage_names[2])
        wf.calc_2 = calculate(structure=wf.calc_1.outputs.engine_output.final_structure, engine=wf.engine_2)

        return (
            [wf.calc_0, wf.calc_1, wf.calc_2],
            wf.calc_2.outputs.engine_output.final_structure,
            wf.calc_2.outputs.engine_output.converged,
        )

    raise NotImplementedError(
        f"multistage_relax currently supports len(stages) ∈ {{1, 3}}; got {len(stages)}. "
        "Add a branch here if you need other lengths — this is a graph-unrolling "
        "limitation, not a fundamental constraint."
    )
```

> Note: the 1-or-3 branching reflects how pyiron_workflow currently requires macro topology to be static at graph-build time. If `pwf` later supports variable-length unrolling cleanly, refactor this into a loop. The two ASSYST-relevant cases (`[full]` for ASE smoke, `[volume, shape, none]` for VASP) are covered.

- [ ] **Step 4: Run to verify it passes**

```bash
pixi run pytest tests/unit/physics/test_relaxation.py -q
```

Expected: 2 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/physics pyiron_workflow_assyst/physics/relaxation.py
git commit -m "feat(physics/relaxation): multistage_relax macro with ASSYST default chain"
```

### Task C10: `physics/assyst.py` — run_assyst top-level macro

**Files:**
- Create: `pyiron_workflow_assyst/physics/assyst.py`

(Unit-tested via the integration test in Task C11; the macro's wiring is too topology-heavy for an isolated unit test.)

- [ ] **Step 1: Implement `physics/assyst.py`**

```python
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


@pwf.as_function_node("final_relax_output")
def _last_engine_output(engine_outputs: list[EngineOutput]) -> EngineOutput:
    return engine_outputs[-1]


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
):
    wf.relax = multistage_relax(
        structure=structure, engine=relax_engine,
        stages=relax_stages, stage_names=relax_stage_names,
    )
    wf.final_relax = _last_engine_output(wf.relax.outputs.engine_outputs)
    wf.harvest = collect_relaxation_frames(
        engine_output=wf.final_relax,
        base_name=base_name,
        image_selection_eV_atom_threshold=image_selection_eV_atom_threshold,
    )
    # Static SCFs on harvested base frames.
    wf.base_engines = _build_subengines(engine=static_engine, names=wf.harvest.outputs.names)
    wf.base_static = for_node(
        calculate,
        zip_on=("structure", "engine"),
        structure=wf.harvest.outputs.structures,
        engine=wf.base_engines,
    )
    # Permutations of harvested bases, then static SCFs.
    wf.perms = generate_assyst_permutations(
        base_structures=wf.harvest.outputs.structures,
        base_names=wf.harvest.outputs.names,
        n_rattle=n_rattle, n_triaxial=n_triaxial, n_shear=n_shear,
        rattle_displacement=rattle_displacement,
        rattle_cell_strain=rattle_cell_strain,
        triaxial_strain=triaxial_strain, shear_strain=shear_strain,
        min_dist=min_dist, core_overlap_tolerance=core_overlap_tolerance,
        seed=seed,
    )
    wf.perm_engines = _build_subengines(engine=static_engine, names=wf.perms.outputs.names)
    wf.perm_static = for_node(
        calculate,
        zip_on=("structure", "engine"),
        structure=wf.perms.outputs.structures,
        engine=wf.perm_engines,
    )
    wf.all_outputs = _concat(a=wf.base_static.outputs.engine_output, b=wf.perm_static.outputs.engine_output)
    wf.all_names = _concat(a=wf.harvest.outputs.names, b=wf.perms.outputs.names)
    wf.export = export_training_set(
        engine_outputs=wf.all_outputs,
        names=wf.all_names,
        path=training_path, format=export_format,
    )
    return wf.export.outputs.path, wf.base_static, wf.perm_static
```

- [ ] **Step 2: Verify the macro builds without errors**

```bash
pixi run python -c "
from pyiron_workflow_assyst.physics.assyst import run_assyst
print(run_assyst)
"
```

Expected: prints the macro repr without exceptions.

- [ ] **Step 3: Commit**

```bash
git add pyiron_workflow_assyst/physics/assyst.py
git commit -m "feat(physics/assyst): run_assyst top-level macro"
```

### Task C11: ASE+EMT integration test

**Files:**
- Create: `tests/integration/__init__.py` (empty)
- Create: `tests/integration/test_run_assyst_emt.py`

- [ ] **Step 1: Write the test**

```python
"""End-to-end ASSYST run on Cu+EMT — small enough to ship in CI."""

import pandas as pd
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import (
    ASEEngine,
    CalcInputMinimize,
    CalcInputStatic,
)

from pyiron_workflow_assyst.physics.assyst import run_assyst


@pytest.mark.integration
def test_run_assyst_emt_endtoend(tmp_path):
    relax_engine = ASEEngine(
        EngineInput=CalcInputMinimize(cell_relaxation="full", max_iterations=20),
        calculator=EMT(),
        working_directory=str(tmp_path / "relax"),
    )
    static_engine = ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path / "static"),
    )
    pickle_path = tmp_path / "train.pkl"

    macro = run_assyst(
        structure=bulk("Cu", "fcc", a=3.6, cubic=True),
        relax_engine=relax_engine,
        static_engine=static_engine,
        base_name="cu0",
        relax_stages=[CalcInputMinimize(cell_relaxation="full", max_iterations=20)],
        relax_stage_names=["fullrelax"],
        n_rattle=1, n_triaxial=1, n_shear=1,
        rattle_displacement=0.05, rattle_cell_strain=0.02,
        triaxial_strain=0.05, shear_strain=0.05,
        seed=42,
        training_path=str(pickle_path),
    )
    macro.run()

    df = pd.read_pickle(pickle_path)
    # 1 base + 3 perms (1 rattle + 1 triax + 1 shear) = 4
    assert len(df) == 4
    assert set(df.columns) == {"name", "energy", "volume", "converged",
                               "structure", "forces", "stress"}
    assert all(df["converged"])
```

- [ ] **Step 2: Run the test**

```bash
pixi run pytest tests/integration/test_run_assyst_emt.py -q
```

Expected: passes. If it doesn't, this is the diagnostic phase — likely the macro wiring or `for_node` zip is wrong. Fix iteratively in `physics/assyst.py`, commit fixes as `fix(physics/assyst): …` commits.

- [ ] **Step 3: Commit**

```bash
git add tests/integration
git commit -m "test(integration): EMT end-to-end run_assyst smoke"
```

### Task C12: `multistage_relax` EMT integration test

**Files:**
- Create: `tests/integration/test_multistage_relax_emt.py`

- [ ] **Step 1: Write and run the test**

```python
"""multistage_relax with the ASE-friendly single-stage default."""

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputMinimize

from pyiron_workflow_assyst.physics.relaxation import multistage_relax


@pytest.mark.integration
def test_full_relax_lowers_energy(tmp_path):
    engine = ASEEngine(
        EngineInput=CalcInputMinimize(cell_relaxation="full", max_iterations=20),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
    cu = bulk("Cu", "fcc", a=3.4, cubic=True)  # under-strained → should relax
    initial_energy = EMT().get_potential_energy(cu)

    macro = multistage_relax(
        structure=cu, engine=engine,
        stages=[CalcInputMinimize(cell_relaxation="full", max_iterations=20)],
        stage_names=["full"],
    )
    macro.run()
    final_energy = macro.outputs.final_structure.value.get_potential_energy()
    assert final_energy < initial_energy
```

```bash
pixi run pytest tests/integration/test_multistage_relax_emt.py -q
```

Expected: passes.

- [ ] **Step 2: Commit**

```bash
git add tests/integration/test_multistage_relax_emt.py
git commit -m "test(integration): multistage_relax EMT smoke"
```

### Task C13: VASP-equivalence test (gated)

**Files:**
- Create: `tests/integration/test_vasp_equivalence.py`
- Create: `tests/_legacy_assyst/__init__.py` (already done in C2; verify)

- [ ] **Step 1: Verify legacy snapshot files exist**

```bash
ls tests/_legacy_assyst/
```

Expected: `__init__.py`, `structure_filter_utils.py`, `workflow.py`.

- [ ] **Step 2: Write the gated equivalence test**

Create `tests/integration/test_vasp_equivalence.py`:
```python
"""Compare legacy VASP-only run_ASSYST_on_structure against the new
engine-agnostic run_assyst driven with VaspEngine.

Skipped unless ``VASP_TEST=1`` is set and ``vasp_std`` is on PATH.
"""

import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar

VASP_AVAILABLE = bool(os.environ.get("VASP_TEST")) and shutil.which("vasp_std") is not None
pytestmark = pytest.mark.skipif(not VASP_AVAILABLE, reason="VASP_TEST=1 and vasp_std required")


@pytest.fixture(scope="module")
def legacy_module():
    """Load the frozen legacy assyst workflow.py."""
    sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/_legacy_assyst")
    import workflow as legacy  # type: ignore[import-not-found]
    return legacy


@pytest.mark.vasp
def test_assyst_vasp_equivalence(tmp_path, legacy_module):
    # Minimal Fe BCC seed.
    fe = bulk("Fe", "bcc", a=2.86, cubic=True)
    incar = Incar.from_dict({
        "ENCUT": 300, "ISIF": 7, "NSW": 5, "EDIFF": 1e-4, "EDIFFG": -0.05,
        "PREC": "Low", "ISMEAR": 1, "SIGMA": 0.2, "ALGO": "Fast",
    })
    potcar_paths = [os.environ["FE_POTCAR_PATH"]]

    # --- Legacy run ---
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    legacy_pickle = legacy_dir / "df_ASSYST_jobs.pkl"
    legacy_pmg = AseAtomsAdaptor.get_structure(fe)
    np.random.seed(123)
    legacy_macro = legacy_module.run_ASSYST_on_structure(
        legacy_pmg, incar, potcar_paths=potcar_paths,
        ionic_steps=5, n_stretch_permutations=1, n_rattle_permutations=1,
        shear_strain=0.05, triaxial_strain=0.05,
        rattle_displacement=0.05, rattle_strain=0.02,
        job_name=str(legacy_dir / "struct"),
        train_df_filename=str(legacy_pickle),
    )
    legacy_macro.run()

    # --- New run ---
    from pyiron_workflow_atomistics.engine import CalcInputMinimize, CalcInputStatic
    from pyiron_workflow_vasp.engine import VaspEngine

    from pyiron_workflow_assyst.physics.assyst import run_assyst

    new_dir = tmp_path / "new"
    new_dir.mkdir()
    new_pickle = new_dir / "df_ASSYST_jobs.pkl"
    relax_engine = VaspEngine(
        EngineInput=CalcInputMinimize(cell_relaxation="volume", max_iterations=5),
        working_directory=str(new_dir / "struct"),
        potcar_config_file=potcar_paths[0],
        encut=300, kpoints_density=0.30, command="vasp_std",
    )
    static_engine = VaspEngine(
        EngineInput=CalcInputStatic(),
        working_directory=str(new_dir / "struct"),
        potcar_config_file=potcar_paths[0],
        encut=300, kpoints_density=0.25, ediff=1e-5, lreal=False, command="vasp_std",
    )
    new_macro = run_assyst(
        structure=fe,
        relax_engine=relax_engine,
        static_engine=static_engine,
        base_name="struct",
        n_rattle=1, n_triaxial=1, n_shear=1,
        rattle_displacement=0.05, rattle_cell_strain=0.02,
        triaxial_strain=0.05, shear_strain=0.05,
        seed=123,
        training_path=str(new_pickle),
    )
    new_macro.run()

    # --- Compare ---
    legacy_df = pd.read_pickle(legacy_pickle)
    new_df = pd.read_pickle(new_pickle)

    # The new schema is one-frame-per-row; the legacy schema is one-job-per-row
    # with list-valued cells. Explode the legacy first.
    legacy_exploded = legacy_df.explode(["structures", "energy"]).reset_index(drop=True)
    assert len(legacy_exploded) == len(new_df), \
        f"Frame count mismatch: legacy={len(legacy_exploded)} new={len(new_df)}"

    # Compare per-name (set equality of names, then per-name energy).
    legacy_energies = dict(zip(legacy_exploded["name"], legacy_exploded["energy"]))
    new_energies = dict(zip(new_df["name"], new_df["energy"]))
    assert set(legacy_energies) == set(new_energies)
    for name in legacy_energies:
        n_atoms = len(legacy_exploded[legacy_exploded["name"] == name]["structures"].iloc[0])
        diff_per_atom = abs(legacy_energies[name] - new_energies[name]) / n_atoms
        assert diff_per_atom < 1e-3, f"{name}: {diff_per_atom} eV/atom drift"
```

Note: this test depends on the legacy `run_ASSYST_on_structure` accepting these arguments and on the legacy DataFrame schema having a `name` column reachable by `explode`. If the legacy column layout differs, adapt the per-name lookup — the existing legacy code stores names in `df.workdir` or similar; inspect `tests/_legacy_assyst/workflow.py` to confirm.

- [ ] **Step 3: Run gated (will skip without VASP)**

```bash
pixi run pytest tests/integration/test_vasp_equivalence.py -q
```

Expected: skipped (no VASP). When running on a VASP-capable machine, set `VASP_TEST=1` and `FE_POTCAR_PATH=/path/to/Fe/POTCAR` to actually execute.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_vasp_equivalence.py
git commit -m "test(integration): gated VASP equivalence vs legacy"
```

### Task C14: `testing/fixtures.py` — shared pytest fixtures

**Files:**
- Create: `pyiron_workflow_assyst/testing/fixtures.py`

- [ ] **Step 1: Implement the fixtures module**

```python
"""Shared pytest fixtures for downstream users of pyiron_workflow_assyst.

Importable as::

    from pyiron_workflow_assyst.testing.fixtures import emt_engine, cu_fcc
"""

from __future__ import annotations

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from pyiron_workflow_atomistics.engine import ASEEngine, CalcInputStatic


@pytest.fixture
def cu_fcc():
    return bulk("Cu", "fcc", a=3.6, cubic=True)


@pytest.fixture
def emt_engine(tmp_path):
    return ASEEngine(
        EngineInput=CalcInputStatic(),
        calculator=EMT(),
        working_directory=str(tmp_path),
    )
```

- [ ] **Step 2: Verify it imports**

```bash
pixi run python -c "from pyiron_workflow_assyst.testing.fixtures import cu_fcc, emt_engine; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add pyiron_workflow_assyst/testing/fixtures.py
git commit -m "feat(testing/fixtures): shared cu_fcc + emt_engine fixtures"
```

### Task C15: Delete legacy modules, set up versioneer, finalize `pyproject.toml`

**Files:**
- Delete: `pyiron_workflow_assyst/example.py`
- Delete: `pyiron_workflow_assyst/workflow.py`
- Delete: `pyiron_workflow_assyst/structure_filter_utils.py`
- Modify: `pyproject.toml`
- Create: `pyiron_workflow_assyst/_version.py` (versioneer install output — auto-generated)
- Create: `setup.py` shim if versioneer requires (auto-generated)
- Create: `MANIFEST.in` if needed (auto-generated)

- [ ] **Step 1: Delete the legacy module files**

```bash
git rm pyiron_workflow_assyst/example.py pyiron_workflow_assyst/workflow.py pyiron_workflow_assyst/structure_filter_utils.py
```

- [ ] **Step 2: Rewrite `pyproject.toml`** (model on pwa's)

```toml
[build-system]
requires = [
    "setuptools",
    "versioneer[toml]==0.29",
]
build-backend = "setuptools.build_meta"

[project]
name = "pyiron_workflow_assyst"
description = "Engine-agnostic ASSYST training-set generation for pyiron"
readme = "README.md"
keywords = ["pyiron", "assyst", "vasp", "mlip", "workflow"]
requires-python = ">=3.9, <3.14"
classifiers = [
    "Development Status :: 3 - Alpha",
    "Topic :: Scientific/Engineering",
    "License :: OSI Approved :: BSD License",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
dependencies = [
    "numpy==1.26.4",
    "pandas==3.0.3",
    "ase==3.28.0",
    "pymatgen==2026.5.4",
    "pyiron-workflow==0.15.6",
    "pyiron_workflow_atomistics",  # pin to the Phase-A release tag once cut
    "tqdm==4.67.3",
]
dynamic = ["version"]
authors = [
    { name = "pyiron team", email = "pyiron@mpie.de" },
]

[project.optional-dependencies]
pyxtal = ["pyxtal"]
vasp = ["pyiron_workflow_vasp"]
test = ["pytest", "nbformat", "nbclient"]

[project.license]
file = "LICENSE"

[project.urls]
Homepage = "https://github.com/ligerzero-ai/pyiron_workflow_assyst"
Documentation = "https://github.com/ligerzero-ai/pyiron_workflow_assyst"
Repository = "https://github.com/ligerzero-ai/pyiron_workflow_assyst"

[tool.versioneer]
VCS = "git"
style = "pep440-pre"
tag_prefix = "pyiron_workflow_assyst-"
versionfile_source = "pyiron_workflow_assyst/_version.py"
parentdir_prefix = "pyiron_workflow_assyst-"

[tool.setuptools.packages.find]
include = ["pyiron_workflow_assyst*"]

[tool.setuptools.package-data]
pyiron_workflow_assyst = ["py.typed"]
```

- [ ] **Step 3: Install versioneer artifacts**

```bash
pixi run python -m uv pip install "versioneer[toml]==0.29"
pixi run versioneer install --vendor
git status
```

Expected: versioneer creates `pyiron_workflow_assyst/_version.py` and updates `setup.cfg` / writes `MANIFEST.in`. Stage these.

- [ ] **Step 4: Tag the prerelease**

```bash
git tag pyiron_workflow_assyst-0.2.0a0
pixi run python -c "import pyiron_workflow_assyst; print(pyiron_workflow_assyst.__version__)"
```

Expected: prints `0.2.0a0` (or `0.2.0a0+something` if untagged commits exist on top — fine; tag again after the final commit before pushing).

- [ ] **Step 5: Run the entire test suite**

```bash
pixi run pytest tests/unit tests/integration -q
```

Expected: all green (VASP-equivalence test skipped).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "chore: drop legacy modules, install versioneer, bump to 0.2.0a0"
```

### Task C16: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Rewrite the README's Usage / Dependencies / Installation sections**

Replace the existing Usage block with:

````markdown
## Usage

`pyiron_workflow_assyst` is engine-agnostic: it talks to compute backends
through the `pyiron_workflow_atomistics` `Engine` Protocol. Construct two
engines — one for relaxation, one for accurate single-points — and pass both
to `run_assyst`:

```python
from ase.build import bulk
from pyiron_workflow_atomistics.engine import CalcInputMinimize, CalcInputStatic
from pyiron_workflow_vasp.engine import VaspEngine
from pyiron_workflow_assyst.physics.assyst import run_assyst

structure = bulk("Fe", "bcc", a=2.86, cubic=True)

relax_engine = VaspEngine(
    EngineInput=CalcInputMinimize(cell_relaxation="volume", max_iterations=100),
    working_directory="./run",
    potcar_config_file="/path/to/Fe/POTCAR",
    encut=400, kpoints_density=0.30,
    command="vasp_std",
)
static_engine = VaspEngine(
    EngineInput=CalcInputStatic(),
    working_directory="./run",
    potcar_config_file="/path/to/Fe/POTCAR",
    encut=400, kpoints_density=0.25,
    ediff=1e-5, lreal=False,
    command="vasp_std",
)

macro = run_assyst(
    structure=structure,
    relax_engine=relax_engine,
    static_engine=static_engine,
    base_name="fe0",
    n_rattle=5, n_triaxial=5, n_shear=5,
)
macro.run()
```

For ASE backends (EMT, MACE, GRACE, ...), swap `VaspEngine` for `ASEEngine`
with the right `calculator=`. The macro body never changes.
````

Update `## Dependencies` to list `pyiron_workflow_atomistics`,
`pyiron-workflow`, `ase`, `pymatgen`, plus optional `pyxtal` / `vasp` extras.

Update `## Installation` to:

````markdown
## Installation

```bash
pip install pyiron_workflow_assyst[vasp]      # with VaspEngine
pip install pyiron_workflow_assyst[pyxtal]    # with random-crystal generator
```
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs(README): engine-agnostic usage, two-engine pattern"
```

### Task C17: Open PR, await merge

**Files:** none

- [ ] **Step 1: Verify branch state, push, and open the PR**

```bash
git status
git log --oneline origin/main..HEAD
git push -u origin feat/engine-agnostic-rewrite
gh pr create --title "feat: engine-agnostic ASSYST rewrite (v0.2.0a0)" --body "$(cat <<'EOF'
## Summary
- Full rewrite of `pyiron_workflow_assyst` following `pyiron_workflow_atomistics` design principles: Engine Protocol, physics-level input dataclasses, EngineOutput as canonical result, subpackage layering.
- Engine-agnostic: works with any backend that satisfies `pyiron_workflow_atomistics.engine.Engine`. Demoed in CI with ASE+EMT; gated VASP-equivalence test asserts ≤1e-3 eV/atom drift vs the legacy implementation.
- Three-stage default relax chain (`cell_relaxation="volume"` → `"shape"` → `"none"`) reproduces the legacy ISIF=7/5/2 cascade when driven with VaspEngine.
- Clean break from the legacy top-level API. Version bumped to `0.2.0a0`.

Spec: `docs/superpowers/specs/2026-05-16-pyiron-workflow-assyst-rewrite-design.md`.

Depends on:
- `pyiron_workflow_atomistics` PR — `feat/cell-relaxation-enum` (merged)
- `pyiron_workflow_vasp` PR — `feat/vasp-engine-isif-mapping` (merged)

## Test plan
- [x] `pytest tests/unit -q` green
- [x] `pytest tests/integration -q` green (EMT end-to-end + multistage_relax smoke)
- [ ] `VASP_TEST=1 FE_POTCAR_PATH=... pytest tests/integration/test_vasp_equivalence.py` on a VASP machine
- [x] `pip install -e .[vasp,pyxtal]` succeeds in a clean env (lazy `__getattr__` keeps build-isolation happy)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 2: Verify before merge**

Per [[gh-pr-merge-auto-misuse]] — wait for `mergeStateStatus==CLEAN` and zero `FAILURE` status checks.

- [ ] **Step 3: Run the VASP-equivalence test on a VASP-capable machine (manual step)**

Document in the PR thread (not a code step): the maintainer runs the gated test once on a VASP-capable machine, reports the energy drift, and ticks the test-plan checkbox before merging.

---

## Self-Review Checklist (run after writing, before handoff)

Already done inline as the plan was written. Highlights:

- **Spec coverage:** Every fidelity-matrix row in spec §5 maps to a task. RCORE table → C2. Deformations → C3. Permutations → C4. PyXtal → C5. collect_relaxation_frames → C6. export_training_set → C7. Engine fanout helpers → C8. multistage_relax → C9. run_assyst → C10. README / clean-break → C16. VASP-equivalence acceptance → C13.
- **Upstream PRs:** A1/A2 = pwa `cell_relaxation` + ASE routing; B1/B2/B3 = pwv ISIF map + new knobs + dep bump.
- **Placeholder scan:** No "TBD" / "similar to" / "appropriate error handling" remain. Every step has either an exact command or full code.
- **Type consistency:** `CalcInputMinimize.cell_relaxation` used identically across A1, B1, B2, C9, C13. `_build_subengines` / `_concat` signatures used in C8 match C10's call sites. `collect_relaxation_frames` returns `(structures, energies, names, frame_indices)` consistently across C6, C10, C11. `run_assyst` signature matches the spec §4 verbatim.
- **Scope check:** Three coordinated PRs across three repos. Each phase produces working software (A merges before B; B merges before C). Phase C alone is the substantive 17-task delivery.
- **Equivalence guarantor:** Task C3's golden-value tests + Task C13's gated end-to-end VASP comparison together cover the user's "ensure it runs an equivalent workflow" requirement at both the statistical-distribution and the integrated-result levels.

---

**Plan complete and saved to `docs/superpowers/plans/2026-05-16-engine-agnostic-rewrite.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using `executing-plans`, batch execution with checkpoints.

**Which approach?**
