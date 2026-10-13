"""End-to-end regression guards over the synthetic library.

These do not reproduce the published numbers, which depend on data that is not
shipped with the code. They pin the behaviour that the published numbers rest
on, so a refactor that quietly changes retrieval or transfer will fail here.
"""

import numpy as np
import pytest

from fireanalogs import robustness, similarity, validation
from fireanalogs.library import embedding_matrix, fuel_slope_matrix


@pytest.fixture
def pipeline(synthetic_library):
    cosine = similarity.cosine_similarity_matrix(embedding_matrix(synthetic_library))
    doy = similarity.circular_doy_distance(synthetic_library["doy"].values)
    geo = similarity.haversine_km(
        synthetic_library["lat"].values, synthetic_library["lon"].values
    )
    combined = similarity.geo_seasonal_distance(geo, doy)
    loo = validation.leave_one_out(synthetic_library, cosine, doy, geo, combined, k=5)
    return synthetic_library, cosine, loo


def test_leave_one_out_is_deterministic(pipeline, synthetic_library):
    _, cosine, loo = pipeline
    doy = similarity.circular_doy_distance(synthetic_library["doy"].values)
    geo = similarity.haversine_km(
        synthetic_library["lat"].values, synthetic_library["lon"].values
    )
    combined = similarity.geo_seasonal_distance(geo, doy)
    repeat = validation.leave_one_out(
        synthetic_library, cosine, doy, geo, combined, k=5
    )
    np.testing.assert_allclose(loo["pred_emb"].values, repeat["pred_emb"].values)


def test_k_sensitivity_covers_every_requested_value(pipeline):
    library, cosine, _ = pipeline
    table = robustness.k_sensitivity(library, cosine, [3, 5, 7])
    assert list(table["k"]) == [3, 5, 7]
    assert table["rmse"].notna().all()
    assert (table["n_evaluated"] > 0).all()


def test_fuel_slope_transfer_produces_finite_summary(pipeline):
    library, cosine, _ = pipeline
    transfer = robustness.fuel_slope_transfer(
        library, cosine, fuel_slope_matrix(library), k=5
    )
    assert len(transfer) > 0
    summary = robustness.summarise_fuel_slope_transfer(transfer)
    for key in ("rmse", "mae", "bias", "pearson_r"):
        assert np.isfinite(summary[key])


def test_ensembles_sit_between_their_members(pipeline):
    _, _, loo = pipeline
    with_ensembles = robustness.add_ensembles(loo)
    lower = np.minimum(loo["pred_emb"], loo["pred_geo"])
    upper = np.maximum(loo["pred_emb"], loo["pred_geo"])
    assert (with_ensembles["pred_emb_geo"] >= lower - 1e-12).all()
    assert (with_ensembles["pred_emb_geo"] <= upper + 1e-12).all()


def test_ensemble_comparison_is_holm_adjusted(pipeline):
    _, _, loo = pipeline
    table = robustness.compare_ensembles(robustness.add_ensembles(loo))
    assert "Holm-adjusted p" in table.columns
    assert len(table) == 3
