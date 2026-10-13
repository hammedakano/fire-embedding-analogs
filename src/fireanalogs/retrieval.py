"""Analog retrieval and probability transfer.

The framework has two separable parts. Retrieval chooses which historical
fires inform a prediction. Transfer converts the retrieved analogs into a
predicted high-severity quantity. Every retrieval space in the manuscript is
paired with the same transfer function, which is what makes the comparison a
test of the retrieval space rather than of the model as a whole.
"""

from __future__ import annotations

import numpy as np

MIN_POOL_SIZE = 3


def candidate_pool(
    index: int,
    event_dates: np.ndarray,
    k: int,
    prior_only: bool = True,
) -> np.ndarray:
    """Indices eligible to act as analogs for the fire at ``index``.

    Under the prediction-time information rule the pool is restricted to fires
    that occurred strictly before the target event, so that retrieval uses only
    information that would have been available when the prediction was made.
    When fewer than ``k`` prior fires exist, which happens for events early in
    the season, the pool falls back to all other fires. Use
    :func:`fallback_report` to count how often that happens.

    The same pool is returned regardless of retrieval method, so embedding and
    baseline retrieval always draw from identical candidates.
    """
    others = np.where(np.arange(len(event_dates)) != index)[0]
    if not prior_only:
        return others
    prior = others[event_dates[others] < event_dates[index]]
    if len(prior) >= k:
        return prior
    return others


def fallback_report(
    event_dates: np.ndarray,
    eligible: np.ndarray,
    k: int,
) -> dict:
    """Count targets whose candidate pool fell back to all other fires.

    Returns the number and percentage of evaluated targets that had fewer than
    ``k`` prior fires available. The manuscript reports these values when
    describing the prediction-time information rule.
    """
    n_fallback = 0
    for index in eligible:
        others = np.where(np.arange(len(event_dates)) != index)[0]
        prior = others[event_dates[others] < event_dates[index]]
        if len(prior) < k:
            n_fallback += 1
    n_total = len(eligible)
    return {
        "n_targets": int(n_total),
        "n_fallback": int(n_fallback),
        "pct_fallback": float(100.0 * n_fallback / n_total) if n_total else 0.0,
    }


def top_k_indices(
    scores: np.ndarray,
    pool: np.ndarray,
    k: int,
    higher_better: bool = True,
) -> np.ndarray:
    """The ``k`` best candidates in ``pool`` ranked by ``scores``.

    A stable sort is used, so candidates with exactly equal scores keep their
    order in ``pool``. Because :func:`fireanalogs.library.load_library` sorts
    the library by event date and fire ID, ties are broken the same way on
    every machine, every NumPy version and every file order. This matters for
    the seasonal baseline, where many fires share a day of year. Cosine
    similarities on 64-dimensional float embeddings do not tie in practice.
    """
    subset = np.asarray(scores)[pool]
    key = -subset if higher_better else subset
    order = np.argsort(key, kind="stable")
    return pool[order[:k]]


def predict_analog_mean(analog_indices: np.ndarray, observed: np.ndarray) -> float:
    """Predicted high-severity fraction as the mean over retrieved analogs.

    Analogs with a missing observed fraction are dropped rather than treated as
    zero. Returns NaN when no analog has a valid observation.
    """
    values = observed[analog_indices]
    valid = values[~np.isnan(values)]
    return float(valid.mean()) if len(valid) else float("nan")


def predict_fuel_slope_mean(
    analog_indices: np.ndarray, fuel_slope_matrix: np.ndarray
) -> float:
    """Mean conditional probability across all 16 fuel by slope cells.

    This is an unweighted mean over cells, not weighted by the pixel count in
    each cell. It gives the event-level summary of the fuel by slope transfer
    reported in the manuscript.

    A slope-only variant (average the fuel groups out, then the slope bins) is
    deliberately not provided. With equal weights it is algebraically the same
    number as this function, so comparing the two tests nothing.
    """
    return float(fuel_slope_matrix[analog_indices].mean())


def cell_probabilities(
    analog_indices: np.ndarray,
    library,
    fuel_slope_columns,
) -> np.ndarray:
    """Mean conditional probability per fuel by slope cell across analogs.

    Returns a length-16 array in the order given by ``fuel_slope_columns``.
    Cells where no analog has a value are returned as 0.0, matching the
    remapping applied when the prediction raster is built in Earth Engine.
    """
    out = []
    for column in fuel_slope_columns:
        if column in library.columns:
            values = library.iloc[analog_indices][column].values.astype(float)
            valid = values[~np.isnan(values)]
            out.append(float(valid.mean()) if len(valid) else 0.0)
        else:
            out.append(0.0)
    return np.asarray(out, dtype=float)


def analog_disagreement(
    analog_indices: np.ndarray,
    library,
    fuel_slope_columns,
) -> np.ndarray:
    """Standard deviation across analogs for each fuel by slope cell.

    This is the uncertainty surface reported alongside the probability surface.
    High values mark cells where the retrieved fires disagree about severity
    behaviour, so the prediction there is less well supported.
    """
    out = []
    for column in fuel_slope_columns:
        if column in library.columns:
            values = library.iloc[analog_indices][column].values.astype(float)
            valid = values[~np.isnan(values)]
            out.append(float(valid.std(ddof=0)) if len(valid) > 1 else 0.0)
        else:
            out.append(0.0)
    return np.asarray(out, dtype=float)
