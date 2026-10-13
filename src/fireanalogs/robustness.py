"""Robustness diagnostics: neighbourhood size, fuel by slope transfer, ensembles."""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.stats.multitest import multipletests

from .retrieval import (
    MIN_POOL_SIZE,
    candidate_pool,
    predict_analog_mean,
    predict_fuel_slope_mean,
    top_k_indices,
)


def k_sensitivity(
    library: pd.DataFrame,
    cosine: np.ndarray,
    k_values: Iterable[int],
    prior_only: bool = True,
) -> pd.DataFrame:
    """Repeat embedding-based leave-one-out validation across values of K.

    The manuscript retains the smallest K at which error has substantially
    stabilised, which shows the headline result is not an artefact of a
    narrow neighbourhood or of a single highly similar fire.
    """
    observed = library["obs_high_frac"].values
    event_dates = library["event_date"].values
    eligible = library.index[library["has_fuelslope"]].tolist()

    rows = []
    for k in k_values:
        squared, absolute = [], []
        for index in eligible:
            if np.isnan(observed[index]):
                continue
            pool = candidate_pool(index, event_dates, k, prior_only=prior_only)
            if len(pool) < k:
                continue
            analogs = top_k_indices(cosine[index], pool, k, higher_better=True)
            predicted = predict_analog_mean(analogs, observed)
            if np.isnan(predicted):
                continue
            squared.append((predicted - observed[index]) ** 2)
            absolute.append(abs(predicted - observed[index]))

        rows.append(
            {
                "k": k,
                "rmse": float(np.sqrt(np.mean(squared))) if squared else np.nan,
                "mae": float(np.mean(absolute)) if absolute else np.nan,
                "n_evaluated": len(squared),
            }
        )
    return pd.DataFrame(rows)


def fuel_slope_transfer(
    library: pd.DataFrame,
    cosine: np.ndarray,
    fuel_slope: np.ndarray,
    k: int = 7,
    prior_only: bool = True,
) -> pd.DataFrame:
    """Event-level skill of the fuel by slope probability transfer.

    Retrieval is held at embedding similarity. For each withheld fire the K
    analogs' 16 conditional probability cells are averaged, giving one
    predicted high-severity fraction. Returns one row per evaluated fire.
    """
    observed = library["obs_high_frac"].values
    event_dates = library["event_date"].values
    eligible = library.index[library["has_fuelslope"]].tolist()

    rows = []
    for index in eligible:
        if np.isnan(observed[index]):
            continue
        pool = candidate_pool(index, event_dates, k, prior_only=prior_only)
        if len(pool) < MIN_POOL_SIZE:
            continue
        analogs = top_k_indices(cosine[index], pool, k, higher_better=True)
        rows.append(
            {
                "fire_id": library.iloc[index]["fire_id"],
                "obs_high_frac": observed[index],
                "pred_fuelslope": predict_fuel_slope_mean(analogs, fuel_slope),
            }
        )
    return pd.DataFrame(rows)


def summarise_fuel_slope_transfer(transfer: pd.DataFrame) -> dict:
    """RMSE, MAE, bias and Pearson r of the fuel by slope transfer."""
    observed = transfer["obs_high_frac"].values
    predicted = transfer["pred_fuelslope"].values
    return {
        "n": int(len(transfer)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mae": float(mean_absolute_error(observed, predicted)),
        "bias": float((predicted - observed).mean()),
        "pearson_r": float(stats.pearsonr(observed, predicted)[0]),
    }


ENSEMBLES: Dict[str, tuple] = {
    "pred_emb_geo": ("pred_emb", "pred_geo"),
    "pred_emb_sea": ("pred_emb", "pred_sea"),
    "pred_emb_geo_sea": ("pred_emb", "pred_geo", "pred_sea"),
}


def add_ensembles(
    loo: pd.DataFrame, ensembles: Optional[Dict[str, tuple]] = None
) -> pd.DataFrame:
    """Add simple unweighted mean ensembles of the retrieval methods."""
    ensembles = ensembles or ENSEMBLES
    out = loo.copy()
    for name, members in ensembles.items():
        out[name] = out[list(members)].mean(axis=1)
    return out


def compare_ensembles(
    loo: pd.DataFrame,
    reference: str = "pred_emb",
    ensembles: Optional[Dict[str, tuple]] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Test whether any ensemble beats embedding-only retrieval.

    The manuscript reports mixed results here, which supports the reading that
    the learned representation encodes information the baselines do not, and
    that averaging it with weaker criteria dilutes rather than improves it.
    """
    ensembles = ensembles or ENSEMBLES
    observed = loo["obs_high_frac"]
    reference_error = (loo[reference] - observed).abs()
    r_reference = float(stats.pearsonr(observed, loo[reference])[0])

    rows, p_values = [], []
    for column in ensembles:
        ensemble_error = (loo[column] - observed).abs()
        w_stat, p_value = stats.wilcoxon(reference_error, ensemble_error)
        rows.append(
            {
                "Comparison": f"{column} vs {reference}",
                "RMSE": float(np.sqrt(mean_squared_error(observed, loo[column]))),
                "MAE": float(mean_absolute_error(observed, loo[column])),
                "Bias": float((loo[column] - observed).mean()),
                "Pearson r": float(stats.pearsonr(observed, loo[column])[0]),
                "Pearson r (reference)": r_reference,
                "Wilcoxon W": round(float(w_stat), 0),
                "Wilcoxon p": p_value,
            }
        )
        p_values.append(p_value)

    if p_values:
        _, adjusted, _, _ = multipletests(p_values, alpha=alpha, method="holm")
        for i, row in enumerate(rows):
            row["Holm-adjusted p"] = round(float(adjusted[i]), 4)

    return pd.DataFrame(rows)
