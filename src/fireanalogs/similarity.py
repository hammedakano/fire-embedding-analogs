"""Pairwise similarity and distance matrices over the fire library.

Three retrieval spaces are compared in the manuscript. Each is reduced to a
single square matrix so that top-K selection is identical across methods and
any difference in skill is attributable to the retrieval space alone.
"""

from __future__ import annotations

import numpy as np

EARTH_RADIUS_KM = 6371.0
_ZERO_NORM_EPS = 1e-10


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Cosine similarity between every pair of embedding vectors.

    Parameters
    ----------
    embeddings
        Array of shape ``(n_fires, n_dims)``.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(n_fires, n_fires)`` matrix with values in ``[-1, 1]``.
        Rows with zero norm are given a small epsilon norm so the result is
        finite rather than undefined.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, _ZERO_NORM_EPS, norms)
    normalised = embeddings / norms
    return normalised @ normalised.T


def haversine_km(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Great-circle distance in kilometres between every pair of centroids."""
    lat_r = np.radians(np.asarray(lats, dtype=np.float64))
    lon_r = np.radians(np.asarray(lons, dtype=np.float64))
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2
    )
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def circular_doy_distance(doys: np.ndarray) -> np.ndarray:
    """Day-of-year separation, wrapped so that day 1 and day 365 are close."""
    doys = np.asarray(doys, dtype=np.float64)
    diff = np.abs(doys[:, None] - doys[None, :])
    return np.minimum(diff, 365 - diff)


def geo_seasonal_distance(
    geo_km: np.ndarray,
    doy_days: np.ndarray,
    geo_norm_km: float = 2000.0,
    doy_norm_days: float = 182.0,
) -> np.ndarray:
    """Scaled sum of geographic and seasonal distance.

    Each component is divided by its own scale so that the two contribute on a
    comparable range before they are added.
    """
    return np.asarray(geo_km) / geo_norm_km + np.asarray(doy_days) / doy_norm_days
