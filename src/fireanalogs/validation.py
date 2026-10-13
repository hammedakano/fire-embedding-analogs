"""Leave-one-out validation and event-level metrics."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .retrieval import (
    MIN_POOL_SIZE,
    candidate_pool,
    predict_analog_mean,
    top_k_indices,
)

METHOD_LABELS: Dict[str, str] = {
    "pred_emb": "Embedding (novel)",
    "pred_geo": "Geographic",
    "pred_sea": "Seasonal",
    "pred_gs": "Geo + Seasonal",
}


def leave_one_out(
    library: pd.DataFrame,
    cosine: np.ndarray,
    doy_distance: np.ndarray,
    geo_distance: Optional[np.ndarray],
    geo_seasonal_distance: Optional[np.ndarray],
    k: int = 7,
    prior_only: bool = True,
) -> pd.DataFrame:
    """Run leave-one-out validation across every eligible target fire.

    A fire is eligible when it carries fuel by slope data and a valid observed
    high-severity fraction. The retrieval pool is every other fire in the
    library, including embedding-only fires, subject to the prediction-time
    rule. All four retrieval methods share one pool per target.

    Returns one row per evaluated fire with the observed fraction, the four
    predictions, the top cosine similarity and the pool size actually used.
    """
    observed = library["obs_high_frac"].values
    event_dates = library["event_date"].values
    eligible = library.index[library["has_fuelslope"]].tolist()
    use_geo = geo_distance is not None and geo_seasonal_distance is not None

    rows = []
    for index in eligible:
        if np.isnan(observed[index]):
            continue
        pool = candidate_pool(index, event_dates, k, prior_only=prior_only)
        if len(pool) < MIN_POOL_SIZE:
            continue

        emb_idx = top_k_indices(cosine[index], pool, k, higher_better=True)
        sea_idx = top_k_indices(-doy_distance[index], pool, k, higher_better=True)
        if use_geo:
            geo_idx = top_k_indices(-geo_distance[index], pool, k, higher_better=True)
            gs_idx = top_k_indices(
                -geo_seasonal_distance[index], pool, k, higher_better=True
            )
        else:
            geo_idx, gs_idx = emb_idx, sea_idx

        rows.append(
            {
                "fire_id": library.iloc[index]["fire_id"],
                "event_date": library.iloc[index]["event_date"],
                "year": library.iloc[index]["year"],
                "area_ha": library.iloc[index]["area_ha"],
                "obs_high_frac": observed[index],
                "pred_emb": predict_analog_mean(emb_idx, observed),
                "pred_geo": predict_analog_mean(geo_idx, observed),
                "pred_sea": predict_analog_mean(sea_idx, observed),
                "pred_gs": predict_analog_mean(gs_idx, observed),
                "top_cosine": float(cosine[index][emb_idx[0]]),
                "n_pool": len(pool),
            }
        )

    return pd.DataFrame(rows).dropna(
        subset=["pred_emb", "pred_geo", "pred_sea", "pred_gs"]
    )


def event_metrics(
    loo: pd.DataFrame, methods: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Event-level RMSE, MAE, R squared, bias and Pearson r for each method."""
    from scipy import stats

    methods = methods or METHOD_LABELS
    observed = loo["obs_high_frac"].values

    rows = []
    for column, label in methods.items():
        predicted = loo[column].values
        rows.append(
            {
                "Method": label,
                "RMSE": float(np.sqrt(mean_squared_error(observed, predicted))),
                "MAE": float(mean_absolute_error(observed, predicted)),
                "R2": float(r2_score(observed, predicted)),
                "Bias": float((predicted - observed).mean()),
                "Pearson r": float(stats.pearsonr(observed, predicted)[0]),
            }
        )
    return pd.DataFrame(rows).set_index("Method")


def similarity_error_relationship(loo: pd.DataFrame) -> dict:
    """Association between top-1 cosine similarity and absolute error.

    Reported in the manuscript as a weak but significant rank association,
    which is the basis for the claim that similarity is informative but not
    sufficient on its own.
    """
    from scipy import stats

    abs_error = (loo["pred_emb"] - loo["obs_high_frac"]).abs()
    pearson_r, pearson_p = stats.pearsonr(loo["top_cosine"], abs_error)
    spearman_r, spearman_p = stats.spearmanr(loo["top_cosine"], abs_error)

    quartiles = pd.qcut(loo["top_cosine"], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    by_quartile = (
        abs_error.groupby(quartiles, observed=False)
        .agg(["mean", "median", "count"])
        .rename(columns={"mean": "mean_abs_error", "median": "median_abs_error"})
    )

    return {
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_rho": float(spearman_r),
        "spearman_p": float(spearman_p),
        "by_quartile": by_quartile,
    }
