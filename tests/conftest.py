"""Shared fixtures: a small synthetic fire library with known structure."""

import numpy as np
import pandas as pd
import pytest

from fireanalogs.config import embedding_columns, fuel_slope_columns


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20260910)


@pytest.fixture
def synthetic_library(rng):
    """A 40-fire library in which embedding similarity carries real signal.

    Fires are drawn from three latent clusters. Within a cluster the embedding
    vectors are close and the observed high-severity fractions are similar, so
    embedding retrieval should beat retrieval by geography or season, which are
    assigned independently of cluster. This gives the regression tests a
    ground truth without shipping the real library.
    """
    n_fires, n_clusters, n_dims = 40, 3, 64
    cluster = rng.integers(0, n_clusters, n_fires)
    centres = rng.normal(size=(n_clusters, n_dims))

    embeddings = centres[cluster] + rng.normal(scale=0.05, size=(n_fires, n_dims))
    cluster_severity = np.array([0.15, 0.55, 0.90])
    observed = np.clip(
        cluster_severity[cluster] + rng.normal(scale=0.04, size=n_fires), 0, 1
    )

    frame = pd.DataFrame(embeddings, columns=embedding_columns(n_dims))
    frame["fire_id"] = [f"fire_{i:03d}" for i in range(n_fires)]
    frame["fire_name"] = [f"Fire {i}" for i in range(n_fires)]
    frame["event_date"] = pd.to_datetime("2025-10-01") + pd.to_timedelta(
        np.sort(rng.integers(0, 120, n_fires)), unit="D"
    )
    frame["doy"] = frame["event_date"].dt.dayofyear
    frame["year"] = frame["event_date"].dt.year
    frame["area_ha"] = rng.uniform(1, 5000, n_fires)
    frame["lat"] = rng.uniform(-39.0, -34.0, n_fires)
    frame["lon"] = rng.uniform(141.0, 150.0, n_fires)
    frame["has_fuelslope"] = True

    total = 1000
    high = (observed * total).round().astype(int)
    frame["sev_hist_0"] = total - high
    frame["sev_hist_1"] = 0
    frame["sev_hist_2"] = high
    frame["sev_hist_3"] = 0
    frame["total_px"] = total
    frame["obs_high_frac"] = observed

    for i, column in enumerate(fuel_slope_columns()):
        frame[column] = np.clip(
            observed + rng.normal(scale=0.05, size=n_fires) + (i - 8) * 0.005, 0, 1
        )

    return frame.reset_index(drop=True)
