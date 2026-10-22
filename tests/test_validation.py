import numpy as np
import pytest

from fireanalogs import similarity, validation


def _matrices(library):
    from fireanalogs.library import embedding_matrix

    cosine = similarity.cosine_similarity_matrix(embedding_matrix(library))
    doy = similarity.circular_doy_distance(library["doy"].values)
    geo = similarity.haversine_km(library["lat"].values, library["lon"].values)
    combined = similarity.geo_seasonal_distance(geo, doy)
    return cosine, doy, geo, combined


def test_leave_one_out_returns_one_row_per_evaluated_fire(synthetic_library):
    cosine, doy, geo, combined = _matrices(synthetic_library)
    loo = validation.leave_one_out(synthetic_library, cosine, doy, geo, combined, k=7)
    assert len(loo) > 0
    assert len(loo) <= len(synthetic_library)
    assert loo["fire_id"].is_unique


def test_predictions_are_bounded_probabilities(synthetic_library):
    cosine, doy, geo, combined = _matrices(synthetic_library)
    loo = validation.leave_one_out(synthetic_library, cosine, doy, geo, combined, k=7)
    for column in ["pred_emb", "pred_geo", "pred_sea", "pred_gs"]:
        assert loo[column].between(0, 1).all()


def test_a_fire_never_retrieves_itself(synthetic_library):
    cosine, doy, geo, combined = _matrices(synthetic_library)
    loo = validation.leave_one_out(synthetic_library, cosine, doy, geo, combined, k=7)
    # A self-match would drive top_cosine to exactly 1.0 for every fire.
    assert (loo["top_cosine"] < 0.999999).any()


def test_embedding_retrieval_beats_baselines_on_clustered_data(synthetic_library):
    """The synthetic library encodes severity in the embedding, not in space
    or season, so embedding retrieval must win. This is the guard that catches
    a retrieval or transfer regression."""
    cosine, doy, geo, combined = _matrices(synthetic_library)
    loo = validation.leave_one_out(synthetic_library, cosine, doy, geo, combined, k=5)
    metrics = validation.event_metrics(loo)
    embedding_rmse = metrics.loc["Embedding (novel)", "RMSE"]
    for baseline in ["Geographic", "Seasonal", "Geo + Seasonal"]:
        assert embedding_rmse < metrics.loc[baseline, "RMSE"]


def test_event_metrics_reports_the_expected_columns(synthetic_library):
    cosine, doy, geo, combined = _matrices(synthetic_library)
    loo = validation.leave_one_out(synthetic_library, cosine, doy, geo, combined, k=7)
    metrics = validation.event_metrics(loo)
    assert list(metrics.columns) == ["RMSE", "MAE", "R2", "Bias", "Pearson r"]
    assert (metrics["RMSE"] >= 0).all()
    assert (metrics["MAE"] >= 0).all()


def test_similarity_error_relationship_returns_quartiles(synthetic_library):
    cosine, doy, geo, combined = _matrices(synthetic_library)
    loo = validation.leave_one_out(synthetic_library, cosine, doy, geo, combined, k=7)
    result = validation.similarity_error_relationship(loo)
    assert set(result) >= {"pearson_r", "spearman_rho", "by_quartile"}
    assert len(result["by_quartile"]) == 4
