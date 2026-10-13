import numpy as np
import pytest

from fireanalogs import similarity, statistics, validation


def test_steiger_is_zero_when_correlations_are_identical():
    z, p = statistics.steiger_test(0.6, 0.6, 0.5, 100)
    assert z == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0, abs=1e-6)


def test_steiger_is_antisymmetric_in_its_first_two_arguments():
    z1, _ = statistics.steiger_test(0.7, 0.4, 0.5, 80)
    z2, _ = statistics.steiger_test(0.4, 0.7, 0.5, 80)
    assert z1 == pytest.approx(-z2)


def test_steiger_detects_a_large_difference():
    z, p = statistics.steiger_test(0.75, 0.25, 0.3, 200)
    assert z > 0
    assert p < 0.01


def test_compare_methods_returns_one_row_per_baseline(synthetic_library):
    from fireanalogs.library import embedding_matrix

    cosine = similarity.cosine_similarity_matrix(embedding_matrix(synthetic_library))
    doy = similarity.circular_doy_distance(synthetic_library["doy"].values)
    geo = similarity.haversine_km(
        synthetic_library["lat"].values, synthetic_library["lon"].values
    )
    combined = similarity.geo_seasonal_distance(geo, doy)
    loo = validation.leave_one_out(synthetic_library, cosine, doy, geo, combined, k=5)

    table = statistics.compare_methods(loo)
    assert len(table) == 3
    assert "Steiger Holm-adj p" in table.columns
    assert "Wilcoxon Holm-adj p" in table.columns
    # Adjusted p-values can never be smaller than the raw ones.
    assert (table["Steiger Holm-adj p"] >= table["Steiger p"].round(4) - 1e-9).all()
