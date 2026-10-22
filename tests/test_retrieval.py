import numpy as np
import pandas as pd
import pytest

from fireanalogs import retrieval
from fireanalogs.config import fuel_slope_columns


def test_pool_excludes_the_target_itself(synthetic_library):
    dates = synthetic_library["event_date"].values
    pool = retrieval.candidate_pool(10, dates, k=7)
    assert 10 not in pool


def test_prediction_time_rule_keeps_only_prior_fires(synthetic_library):
    dates = synthetic_library["event_date"].values
    index = 30
    pool = retrieval.candidate_pool(index, dates, k=7, prior_only=True)
    assert (dates[pool] < dates[index]).all()


def test_pool_falls_back_when_too_few_prior_fires(synthetic_library):
    dates = synthetic_library["event_date"].values
    pool = retrieval.candidate_pool(0, dates, k=7, prior_only=True)
    # The earliest fire has no prior events, so the pool must widen.
    assert len(pool) == len(dates) - 1


def test_disabling_the_rule_returns_every_other_fire(synthetic_library):
    dates = synthetic_library["event_date"].values
    pool = retrieval.candidate_pool(20, dates, k=7, prior_only=False)
    assert len(pool) == len(dates) - 1


def test_fallback_report_counts_early_season_targets(synthetic_library):
    dates = synthetic_library["event_date"].values
    eligible = np.arange(len(dates))
    report = retrieval.fallback_report(dates, eligible, k=7)
    assert report["n_targets"] == len(dates)
    assert 0 < report["n_fallback"] < len(dates)
    assert 0 < report["pct_fallback"] < 100


def test_top_k_returns_the_highest_scores():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    pool = np.array([0, 1, 2, 3])
    top = retrieval.top_k_indices(scores, pool, k=2, higher_better=True)
    assert list(top) == [1, 3]


def test_top_k_can_rank_ascending_for_distances():
    distances = np.array([10.0, 1.0, 5.0])
    pool = np.array([0, 1, 2])
    top = retrieval.top_k_indices(-distances, pool, k=2, higher_better=True)
    assert list(top) == [1, 2]


def test_analog_mean_ignores_missing_observations():
    observed = np.array([0.2, np.nan, 0.6])
    assert retrieval.predict_analog_mean(np.array([0, 1, 2]), observed) == pytest.approx(0.4)


def test_analog_mean_is_nan_when_nothing_is_observed():
    observed = np.array([np.nan, np.nan])
    assert np.isnan(retrieval.predict_analog_mean(np.array([0, 1]), observed))


def test_slope_only_transfer_averages_fuel_groups_out():
    # 16 cells whose value depends only on the slope bin.
    single = np.tile(np.array([0.1, 0.2, 0.3, 0.4]), 4).reshape(1, 16)
    full = retrieval.predict_fuel_slope_mean(np.array([0]), single)
    slope_only = retrieval.predict_slope_only(np.array([0]), single)
    assert full == pytest.approx(slope_only)


def test_cell_probabilities_and_disagreement_align(synthetic_library):
    columns = fuel_slope_columns()
    analogs = np.array([0, 1, 2])
    means = retrieval.cell_probabilities(analogs, synthetic_library, columns)
    spread = retrieval.analog_disagreement(analogs, synthetic_library, columns)
    assert means.shape == (16,)
    assert spread.shape == (16,)
    assert (spread >= 0).all()


def test_cell_probabilities_default_to_zero_for_absent_columns(synthetic_library):
    trimmed = synthetic_library.drop(columns=["p_high_f0_s0"])
    values = retrieval.cell_probabilities(
        np.array([0, 1]), trimmed, ["p_high_f0_s0"]
    )
    assert values[0] == 0.0
