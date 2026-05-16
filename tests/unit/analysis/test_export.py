"""export_training_set: pickle_df schema + extxyz round-trip + ase_db."""

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
            final_structure=a,
            final_energy=-3.5,
            converged=True,
            final_forces=np.zeros((4, 3)),
            final_stress_voigt=np.zeros(6),
            final_volume=a.get_volume(),
        ),
        EngineOutput(
            final_structure=a,
            final_energy=-3.6,
            converged=True,
            final_forces=np.zeros((4, 3)),
            final_stress_voigt=np.zeros(6),
            final_volume=a.get_volume(),
        ),
    ]
    return out, ["base_0", "perm_0"]


class TestPickleDf:
    def test_schema_and_content(self, two_outputs, tmp_path):
        outs, names = two_outputs
        path = tmp_path / "train.pkl"
        export_training_set.node_function(
            engine_outputs=outs,
            names=names,
            path=str(path),
            format="pickle_df",
        )
        df = pd.read_pickle(path)
        assert list(df.columns) == [
            "name",
            "energy",
            "volume",
            "converged",
            "structure",
            "forces",
            "stress",
        ]
        assert len(df) == 2
        assert df.iloc[0]["name"] == "base_0"
        assert df.iloc[1]["energy"] == pytest.approx(-3.6)
        # Per-frame, not list-valued
        assert isinstance(df.iloc[0]["energy"], float)
        assert df.iloc[0]["structure"].get_chemical_formula() == "Cu4"


class TestExtxyz:
    def test_round_trip(self, two_outputs, tmp_path):
        outs, names = two_outputs
        path = tmp_path / "train.xyz"
        export_training_set.node_function(
            engine_outputs=outs,
            names=names,
            path=str(path),
            format="extxyz",
        )
        atoms_list = ase_read(str(path), index=":")
        assert len(atoms_list) == 2
        # ASE absorbs 'energy' from extxyz header into SinglePointCalculator.results
        # (it is in ase.calculators.calculator.all_properties), so check calc.results
        calc = atoms_list[0].calc
        assert calc is not None
        assert calc.results.get("energy") == pytest.approx(-3.5)


class TestAseDb:
    def test_writes_rows(self, two_outputs, tmp_path):
        outs, names = two_outputs
        path = tmp_path / "train.db"
        export_training_set.node_function(
            engine_outputs=outs,
            names=names,
            path=str(path),
            format="ase_db",
        )
        from ase.db import connect

        db = connect(str(path))
        assert db.count() == 2
        rows = list(db.select())
        assert {r.name for r in rows} == {"base_0", "perm_0"}


class TestErrors:
    def test_length_mismatch_raises(self, two_outputs, tmp_path):
        outs, _ = two_outputs
        with pytest.raises(ValueError, match="same length"):
            export_training_set.node_function(
                engine_outputs=outs,
                names=["only_one"],
                path=str(tmp_path / "x.pkl"),
                format="pickle_df",
            )

    def test_unknown_format_raises(self, two_outputs, tmp_path):
        outs, names = two_outputs
        with pytest.raises(ValueError, match="Unknown export format"):
            export_training_set.node_function(
                engine_outputs=outs,
                names=names,
                path=str(tmp_path / "x.pkl"),
                format="parquet",  # not supported
            )
