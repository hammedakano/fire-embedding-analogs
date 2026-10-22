"""Built-surface exposure screening.

Exposure here means the coincidence of built surface with predicted
high-severity probability. It is not a damage, vulnerability or loss estimate,
and the outputs must not be read as one.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .pixel import FireRasters


def built_exposure(
    rasters: FireRasters,
    pixel_area_ha: float = 0.01,
    threshold: float = 0.5,
    fire_id: Optional[str] = None,
    fire_name: Optional[str] = None,
) -> pd.DataFrame:
    """Observed and predicted built-surface exposure for one fire.

    Predicted exposure is probabilistic. Each built pixel contributes its
    predicted probability rather than being forced into an exposed or
    unexposed class, which keeps intermediate likelihoods in the total.
    Recall on built pixels is computed after thresholding, because a screening
    user cares about whether an at-risk area was flagged at all.
    """
    if rasters.built_mask is None:
        raise ValueError("No built-surface raster was loaded for this fire.")

    mask = (
        (~np.isnan(rasters.obs_binary))
        & (~np.isnan(rasters.p_high_emb))
        & (~np.isnan(rasters.built_mask))
    )
    built = (rasters.built_mask > 0) & mask

    observed_exposure = (built & (rasters.obs_binary > 0.5)).sum() * pixel_area_ha

    emb_weighted = np.where(built, rasters.p_high_emb, 0.0)
    geo_weighted = np.where(built, rasters.p_high_geo, 0.0)
    emb_predicted = float(np.nansum(emb_weighted) * pixel_area_ha)
    geo_predicted = float(np.nansum(geo_weighted) * pixel_area_ha)

    observed_high = rasters.obs_binary > 0.5
    emb_tp = built & (rasters.p_high_emb > threshold) & observed_high
    geo_tp = built & (rasters.p_high_geo > threshold) & observed_high
    emb_fn = built & (rasters.p_high_emb <= threshold) & observed_high
    geo_fn = built & (rasters.p_high_geo <= threshold) & observed_high

    def recall(tp: np.ndarray, fn: np.ndarray) -> float:
        denominator = tp.sum() + fn.sum()
        return float(tp.sum() / denominator) if denominator else 0.0

    return pd.DataFrame(
        [
            {
                "fire_id": fire_id,
                "fire_name": fire_name,
                "threshold": threshold,
                "total_built_ha": float(built.sum() * pixel_area_ha),
                "true_exposure_ha": float(observed_exposure),
                "emb_predicted_ha": emb_predicted,
                "geo_predicted_ha": geo_predicted,
                "emb_pred_error_ha": emb_predicted - float(observed_exposure),
                "geo_pred_error_ha": geo_predicted - float(observed_exposure),
                "emb_caught_ha": float(emb_tp.sum() * pixel_area_ha),
                "geo_caught_ha": float(geo_tp.sum() * pixel_area_ha),
                "emb_missed_ha": float(emb_fn.sum() * pixel_area_ha),
                "geo_missed_ha": float(geo_fn.sum() * pixel_area_ha),
                "emb_recall_built": recall(emb_tp, emb_fn),
                "geo_recall_built": recall(geo_tp, geo_fn),
            }
        ]
    )


def outcome_analysis(
    rasters: FireRasters,
    pixel_area_ha: float = 0.01,
    threshold: float = 0.3,
    fire_id: Optional[str] = None,
    fire_name: Optional[str] = None,
) -> pd.DataFrame:
    """Head-to-head confusion analysis across the whole fire footprint.

    The default threshold differs from the one used for built-surface recall.
    Both are carried in the output so any published number can be traced to the
    threshold that produced it.
    """
    mask = rasters.valid_mask
    observed_high = (rasters.obs_binary > 0.5) & mask
    observed_low = (rasters.obs_binary < 0.5) & mask

    emb_high = (rasters.p_high_emb > threshold) & mask
    geo_high = (rasters.p_high_geo > threshold) & mask

    def to_ha(array: np.ndarray) -> float:
        return float(array.sum() * pixel_area_ha)

    def metrics(predicted_high: np.ndarray) -> dict:
        tp = to_ha(observed_high & predicted_high)
        fn = to_ha(observed_high & ~predicted_high)
        fp = to_ha(observed_low & predicted_high)
        tn = to_ha(observed_low & ~predicted_high)
        total = tp + fn + fp + tn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall_value = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall_value / (precision + recall_value)
            if (precision + recall_value)
            else 0.0
        )
        return {
            "TP_ha": tp,
            "FN_ha": fn,
            "FP_ha": fp,
            "TN_ha": tn,
            "precision": precision,
            "recall": recall_value,
            "f1": f1,
            "accuracy": (tp + tn) / total if total else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        }

    emb = metrics(emb_high)
    geo = metrics(geo_high)

    row = {
        "fire_id": fire_id,
        "fire_name": fire_name,
        "threshold": threshold,
        "total_ha": to_ha(mask),
        "high_sev_ha": to_ha(observed_high),
        "low_sev_ha": to_ha(observed_low),
        "emb_wins_ha": to_ha(observed_high & emb_high & ~geo_high),
        "geo_wins_ha": to_ha(observed_high & ~emb_high & geo_high),
        "both_correct_ha": to_ha(observed_high & emb_high & geo_high),
        "both_fail_ha": to_ha(observed_high & ~emb_high & ~geo_high),
    }
    row.update({f"emb_{key}": value for key, value in emb.items()})
    row.update({f"geo_{key}": value for key, value in geo.items()})
    return pd.DataFrame([row])
