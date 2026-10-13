"""The event-level analysis in one call.

Every similarity and distance matrix is built here from a single library
table, so row ``i`` refers to the same fire in every matrix. Keeping that
construction in one place is the guard against the row-alignment error that
affected an earlier working notebook, in which the baseline distance matrices
were built from a differently ordered copy of the library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import robustness, similarity, statistics, validation
from .config import Config
from .library import embedding_matrix, fuel_slope_matrix, has_usable_coordinates
from .retrieval import fallback_report


@dataclass
class Matrices:
    """Square matrices over the library, all in the library's row order."""

    cosine: np.ndarray
    doy: np.ndarray
    geo: Optional[np.ndarray]
    geo_seasonal: Optional[np.ndarray]


def build_matrices(library: pd.DataFrame, cfg: Config) -> Matrices:
    """Build every retrieval matrix from the same table."""
    cosine = similarity.cosine_similarity_matrix(embedding_matrix(library, cfg.n_emb_dims))
    doy = similarity.circular_doy_distance(library["doy"].values)
    if has_usable_coordinates(library):
        geo = similarity.haversine_km(library["lat"].values, library["lon"].values)
        geo_seasonal = similarity.geo_seasonal_distance(
            geo, doy, cfg.geo_norm_km, cfg.doy_norm_days
        )
    else:
        geo = geo_seasonal = None
    return Matrices(cosine=cosine, doy=doy, geo=geo, geo_seasonal=geo_seasonal)


def run_event_level(library: pd.DataFrame, cfg: Config) -> Dict[str, object]:
    """Leave-one-out validation and every event-level diagnostic.

    Returns a dictionary of tables and summaries keyed by the name of the
    output file each one is written to.
    """
    m = build_matrices(library, cfg)
    k = cfg.k_default

    loo = validation.leave_one_out(
        library, m.cosine, m.doy, m.geo, m.geo_seasonal, k=k, prior_only=cfg.prior_only
    )
    metrics = validation.event_metrics(loo)
    comparison = statistics.compare_methods(loo)
    relationship = validation.similarity_error_relationship(loo)
    k_table = robustness.k_sensitivity(library, m.cosine, cfg.k_list, prior_only=cfg.prior_only)
    transfer = robustness.fuel_slope_transfer(
        library, m.cosine, fuel_slope_matrix(library), k=k, prior_only=cfg.prior_only
    )
    loo_ens = robustness.add_ensembles(loo)
    ensembles = robustness.compare_ensembles(loo_ens)

    eligible = library.index[library["has_fuelslope"] & library["obs_high_frac"].notna()]
    fallback = fallback_report(library["event_date"].values, eligible.to_numpy(), k)

    return {
        "loo_results": loo,
        "loo_metrics": metrics,
        "method_comparison": comparison,
        "similarity_error": relationship,
        "k_sensitivity": k_table,
        "fuel_slope_transfer": transfer,
        "fuel_slope_transfer_summary": robustness.summarise_fuel_slope_transfer(transfer),
        "ensemble_comparison": ensembles,
        "fallback": fallback,
    }
