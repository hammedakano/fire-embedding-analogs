import numpy as np
import pytest

from fireanalogs import similarity


def test_cosine_matrix_is_symmetric_with_unit_diagonal():
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    matrix = similarity.cosine_similarity_matrix(x)
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-12)
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)


def test_cosine_matrix_known_values():
    x = np.array([[1.0, 0.0], [0.0, 1.0]])
    matrix = similarity.cosine_similarity_matrix(x)
    assert matrix[0, 1] == pytest.approx(0.0, abs=1e-12)


def test_cosine_matrix_tolerates_zero_vectors():
    x = np.array([[0.0, 0.0], [1.0, 1.0]])
    matrix = similarity.cosine_similarity_matrix(x)
    assert np.isfinite(matrix).all()


def test_haversine_zero_on_diagonal_and_known_distance():
    lats = np.array([0.0, 0.0])
    lons = np.array([0.0, 1.0])
    d = similarity.haversine_km(lats, lons)
    assert d[0, 0] == pytest.approx(0.0, abs=1e-9)
    # One degree of longitude at the equator is about 111.19 km.
    assert d[0, 1] == pytest.approx(111.19, rel=0.01)


def test_circular_doy_wraps_across_new_year():
    doys = np.array([1.0, 365.0, 180.0])
    d = similarity.circular_doy_distance(doys)
    assert d[0, 1] == pytest.approx(1.0)
    assert d[0, 2] <= 183.0


def test_geo_seasonal_scales_each_component():
    geo = np.array([[0.0, 2000.0]])
    doy = np.array([[0.0, 182.0]])
    combined = similarity.geo_seasonal_distance(geo, doy, 2000.0, 182.0)
    assert combined[0, 1] == pytest.approx(2.0)
