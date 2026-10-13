"""Assembly of the exported fire library.

The Earth Engine exports arrive as many CSV chunks, one per month or week,
plus one embedding-only export covering every fire. This module merges them
into a single table with the derived fields the rest of the pipeline needs.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from .config import (
    embedding_columns,
    fuel_slope_columns,
    severity_histogram_columns,
)

WEATHER_COLUMNS = ["wth_temp_2m_mean", "wth_dew_2m_mean", "wth_precip_30d_sum"]


def extract_centroid(geo_json: str) -> tuple:
    """Longitude and latitude of a feature centroid from the ``.geo`` column.

    Earth Engine writes the exported geometry as a GeoJSON string. Points are
    returned directly. Polygons are reduced to the mean of their outer ring,
    which is what the published analysis used. Unparseable values return NaN
    rather than raising, so one malformed row cannot stop library assembly.
    """
    try:
        geo = json.loads(geo_json)
        if geo["type"] == "Point":
            return geo["coordinates"][0], geo["coordinates"][1]
        ring = np.asarray(geo["coordinates"][0], dtype=float)
        return float(ring[:, 0].mean()), float(ring[:, 1].mean())
    except Exception:
        return np.nan, np.nan


def _read_chunks(paths: Iterable[str]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        try:
            chunk = pd.read_csv(path)
        except Exception:
            continue
        if len(chunk) == 0:
            continue
        chunk["source"] = Path(path).stem
        frames.append(chunk)
    if not frames:
        raise FileNotFoundError("No non-empty fire library CSVs were found.")
    return pd.concat(frames).drop_duplicates(subset="fire_id").reset_index(drop=True)


def _add_derived_fields(frame: pd.DataFrame, has_fuel_slope: bool) -> pd.DataFrame:
    if ".geo" in frame.columns:
        frame[["lon", "lat"]] = frame[".geo"].apply(
            lambda value: pd.Series(extract_centroid(value))
        )
    else:
        frame["lon"] = np.nan
        frame["lat"] = np.nan

    frame["event_date"] = pd.to_datetime(frame["event_date"])
    frame["doy"] = frame["event_date"].dt.dayofyear
    frame["year"] = frame["event_date"].dt.year
    frame["has_fuelslope"] = has_fuel_slope
    return frame


def load_library(
    data_root: str | Path,
    full_glob: str = "fire_lib_*.csv",
    embedding_only_file: Optional[str] = "fire_lib_embeddings_only.csv",
    n_emb_dims: int = 64,
) -> pd.DataFrame:
    """Build the retrieval library from the exported CSV chunks.

    Fires exported with the full summariser carry the fuel by slope
    conditional probabilities and the severity histogram, so they can act both
    as analogs and as leave-one-out targets. Fires exported with the
    embedding-only summariser carry the embedding alone, so they enlarge the
    retrieval pool without becoming validation targets. Where a fire appears in
    both, the full record wins.
    """
    data_root = Path(data_root)
    chunk_paths = sorted(
        p
        for p in glob.glob(str(data_root / full_glob))
        if not p.endswith("fire_lib_embeddings_only.csv")
    )
    full = _add_derived_fields(_read_chunks(chunk_paths), has_fuel_slope=True)

    frames = [full]
    if embedding_only_file:
        emb_path = data_root / embedding_only_file
        if emb_path.exists():
            emb_only = pd.read_csv(emb_path)
            if len(emb_only):
                frames.append(_add_derived_fields(emb_only, has_fuel_slope=False))

    library = (
        pd.concat(frames)
        .drop_duplicates(subset="fire_id", keep="first")
        .reset_index(drop=True)
    )

    for column in WEATHER_COLUMNS + fuel_slope_columns():
        if column not in library.columns:
            library[column] = np.nan

    emb_cols = embedding_columns(n_emb_dims)
    missing_embedding = library[emb_cols].isna().all(axis=1)
    library = library[~missing_embedding]

    # Canonical row order. Every similarity and distance matrix is built from
    # this one table, so row i always means the same fire in every matrix, and
    # results do not depend on the order in which the CSV files were read.
    library = library.sort_values(["event_date", "fire_id"], kind="stable")
    library = library.reset_index(drop=True)

    return add_observed_high_fraction(library)


def add_observed_high_fraction(library: pd.DataFrame) -> pd.DataFrame:
    """Derive the observed high-severity fraction from the severity histogram.

    Classes 2 and 3 of the dNBR classification are treated as high severity.
    Fires with no classified pixels get NaN so they are excluded from
    validation rather than counted as zero.
    """
    hist_cols = severity_histogram_columns()
    for column in hist_cols:
        if column not in library.columns:
            library[column] = 0
        library[column] = pd.to_numeric(library[column], errors="coerce").fillna(0)

    library["total_px"] = library[hist_cols].sum(axis=1)
    library["obs_high_frac"] = np.where(
        library["total_px"] > 0,
        (library["sev_hist_2"] + library["sev_hist_3"]) / library["total_px"],
        np.nan,
    )
    return library


def embedding_matrix(library: pd.DataFrame, n_emb_dims: int = 64) -> np.ndarray:
    """Dense embedding matrix, missing dimensions filled with zero."""
    return library[embedding_columns(n_emb_dims)].fillna(0).values.astype(np.float64)


def fuel_slope_matrix(library: pd.DataFrame) -> np.ndarray:
    """Dense 16-column conditional probability matrix, missing cells zeroed."""
    return library[fuel_slope_columns()].fillna(0).values.astype(np.float64)


def has_usable_coordinates(library: pd.DataFrame, threshold: float = 0.8) -> bool:
    """Whether enough centroids parsed for the geographic baselines to run."""
    return bool(library["lat"].notna().sum() > len(library) * threshold)
