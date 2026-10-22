import numpy as np
import pytest

from fireanalogs import exposure
from fireanalogs.pixel import FireRasters, brier_score


def _rasters(p_emb, p_geo, observed, built=None):
    valid = (~np.isnan(observed)) & (~np.isnan(p_emb))
    zeros = np.zeros_like(observed)
    return FireRasters(
        severity=observed * 2,
        p_high_emb=p_emb,
        p_high_geo=p_geo,
        brier_emb=(p_emb - observed) ** 2,
        brier_geo=(p_geo - observed) ** 2,
        brier_diff=(p_emb - observed) ** 2 - (p_geo - observed) ** 2,
        fuel_groups=zeros,
        slope_bins=zeros,
        built_mask=built,
        obs_binary=observed,
        valid_mask=valid,
    )


def test_brier_score_is_zero_for_perfect_prediction():
    observed = np.array([[1.0, 0.0], [1.0, 0.0]])
    assert brier_score(observed.copy(), observed) == pytest.approx(0.0)


def test_brier_score_is_one_for_confidently_wrong_prediction():
    observed = np.array([[1.0, 0.0]])
    predicted = np.array([[0.0, 1.0]])
    assert brier_score(predicted, observed) == pytest.approx(1.0)


def test_built_exposure_counts_only_built_pixels():
    observed = np.array([[1.0, 1.0], [0.0, 0.0]])
    built = np.array([[1.0, 0.0], [1.0, 0.0]])
    p_emb = np.full((2, 2), 0.8)
    p_geo = np.full((2, 2), 0.2)
    result = exposure.built_exposure(
        _rasters(p_emb, p_geo, observed, built), pixel_area_ha=0.01, threshold=0.5
    )
    row = result.iloc[0]
    assert row["total_built_ha"] == pytest.approx(0.02)
    # Only the top-left pixel is both built and observed high severity.
    assert row["true_exposure_ha"] == pytest.approx(0.01)


def test_probabilistic_exposure_sums_probabilities_not_counts():
    observed = np.zeros((1, 2))
    built = np.ones((1, 2))
    p_emb = np.array([[0.25, 0.75]])
    result = exposure.built_exposure(
        _rasters(p_emb, p_emb, observed, built), pixel_area_ha=0.01
    )
    assert result.iloc[0]["emb_predicted_ha"] == pytest.approx(0.01)


def test_recall_on_built_pixels_rewards_catching_the_at_risk_area():
    observed = np.array([[1.0, 1.0]])
    built = np.ones((1, 2))
    p_emb = np.array([[0.9, 0.9]])
    p_geo = np.array([[0.1, 0.1]])
    row = exposure.built_exposure(
        _rasters(p_emb, p_geo, observed, built), threshold=0.5
    ).iloc[0]
    assert row["emb_recall_built"] == pytest.approx(1.0)
    assert row["geo_recall_built"] == pytest.approx(0.0)


def test_built_exposure_requires_a_built_raster():
    observed = np.zeros((1, 2))
    with pytest.raises(ValueError):
        exposure.built_exposure(_rasters(observed, observed, observed, None))


def test_outcome_analysis_partitions_the_footprint():
    observed = np.array([[1.0, 1.0], [0.0, 0.0]])
    p_emb = np.array([[0.9, 0.1], [0.9, 0.1]])
    p_geo = np.array([[0.1, 0.9], [0.1, 0.9]])
    row = exposure.outcome_analysis(
        _rasters(p_emb, p_geo, observed), pixel_area_ha=0.01, threshold=0.5
    ).iloc[0]
    total_high = (
        row["emb_wins_ha"] + row["geo_wins_ha"] + row["both_correct_ha"] + row["both_fail_ha"]
    )
    assert total_high == pytest.approx(row["high_sev_ha"])
    assert row["high_sev_ha"] + row["low_sev_ha"] == pytest.approx(row["total_ha"])
