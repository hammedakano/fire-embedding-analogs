"""Results must not depend on the order in which library files are read.

An earlier working notebook built its baseline distance matrices from a copy
of the library that was loaded in a different file order from the copy used
for the embedding matrix and the observed values. Row i then meant different
fires in different matrices, and every baseline prediction was wrong. These
tests load the same fires in two different file orders and require identical
per-fire predictions from every retrieval method.
"""

import json

import numpy as np
import pandas as pd

from fireanalogs.config import Config
from fireanalogs.library import load_library
from fireanalogs.pipeline import run_event_level


def _write_chunks(frame: pd.DataFrame, folder, names):
    """Split ``frame`` in two and write the halves under the given file names."""
    folder.mkdir(parents=True, exist_ok=True)
    half = len(frame) // 2
    frame.iloc[:half].to_csv(folder / names[0], index=False)
    frame.iloc[half:].to_csv(folder / names[1], index=False)


def _as_export(library: pd.DataFrame) -> pd.DataFrame:
    """Turn the synthetic library into the shape of an Earth Engine export."""
    export = library.drop(
        columns=["doy", "year", "has_fuelslope", "total_px", "obs_high_frac"]
    ).copy()
    export[".geo"] = [
        json.dumps({"type": "Point", "coordinates": [lon, lat]})
        for lon, lat in zip(library["lon"], library["lat"])
    ]
    export = export.drop(columns=["lat", "lon"])
    export["event_date"] = export["event_date"].dt.strftime("%Y-%m-%d")
    return export


def test_predictions_do_not_depend_on_file_order(synthetic_library, tmp_path):
    export = _as_export(synthetic_library)
    shuffled = export.sample(frac=1.0, random_state=7).reset_index(drop=True)

    # Same fires, different split across files and different read order.
    _write_chunks(export, tmp_path / "a", ["fire_lib_1.csv", "fire_lib_2.csv"])
    _write_chunks(shuffled, tmp_path / "b", ["fire_lib_2.csv", "fire_lib_1.csv"])

    cfg = Config.load()
    first = run_event_level(load_library(tmp_path / "a", embedding_only_file=None), cfg)
    second = run_event_level(load_library(tmp_path / "b", embedding_only_file=None), cfg)

    a = first["loo_results"].set_index("fire_id").sort_index()
    b = second["loo_results"].set_index("fire_id").sort_index()
    assert len(a) > 0
    for column in ["pred_emb", "pred_geo", "pred_sea", "pred_gs", "n_pool"]:
        np.testing.assert_allclose(a[column].values, b[column].values, err_msg=column)


def test_library_rows_are_in_canonical_order(synthetic_library, tmp_path):
    export = _as_export(synthetic_library).sample(frac=1.0, random_state=3)
    _write_chunks(export, tmp_path, ["fire_lib_x.csv", "fire_lib_y.csv"])
    library = load_library(tmp_path, embedding_only_file=None)
    keys = list(zip(library["event_date"], library["fire_id"]))
    assert keys == sorted(keys)
