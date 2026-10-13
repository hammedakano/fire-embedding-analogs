"""Pixel-level validation of a single target fire.

The rasters consumed here are the Earth Engine exports produced by
:mod:`fireanalogs.gee.export`. Every array is read at the same 10 metre grid
and the same footprint, so they can be compared cell by cell without
resampling.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

RASTER_SUFFIXES = (
    "obs_high_binary",
    "target_dnbr",
    "target_severity",
    "pHigh_emb",
    "pHigh_geo",
    "brier_emb",
    "brier_geo",
    "brier_diff",
    "fuel_groups",
    "slope_bins",
    "built_mask",
    "expected_built_exposure",
)


def load_raster(path: str | Path) -> tuple:
    """Read a single-band GeoTIFF as a float array with nodata as NaN."""
    import rasterio

    with rasterio.open(path) as source:
        data = source.read(1).astype(np.float32)
        nodata = source.nodata
        profile = source.profile
        bounds = source.bounds
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    return data, profile, bounds


def find_raster(data_root: str | Path, fire_prefix: str, suffix: str) -> str:
    """Locate one exported raster for a fire.

    Earth Engine sometimes appends a numeric suffix when a task is retried, so
    an exact match is tried first and a strict numeric variant second. Anything
    looser would let ``pHigh_emb`` match ``pHigh_emb_something_else``.
    """
    data_root = Path(data_root)
    exact = list(glob.glob(str(data_root / f"{fire_prefix}_{suffix}.tif")))
    if exact:
        return exact[0]

    pattern = re.compile(rf"^{re.escape(fire_prefix)}_{re.escape(suffix)}(\d*)\.tif$")
    loose = [
        path
        for path in glob.glob(str(data_root / f"{fire_prefix}_{suffix}*.tif"))
        if pattern.match(os.path.basename(path))
    ]
    if loose:
        return sorted(loose)[0]

    raise FileNotFoundError(f"No raster found for {fire_prefix}_{suffix} in {data_root}")


@dataclass
class FireRasters:
    """The aligned rasters for one target fire."""

    severity: np.ndarray
    p_high_emb: np.ndarray
    p_high_geo: np.ndarray
    brier_emb: np.ndarray
    brier_geo: np.ndarray
    brier_diff: np.ndarray
    fuel_groups: np.ndarray
    slope_bins: np.ndarray
    built_mask: Optional[np.ndarray]
    obs_binary: np.ndarray
    valid_mask: np.ndarray


def load_fire(
    data_root: str | Path, fire_prefix: str, high_class_min: int = 2
) -> FireRasters:
    """Load every raster for one fire and derive the observed binary map.

    The observed high-severity map is derived from the exported severity
    classes rather than read from its own file, so that the binary definition
    used in validation cannot drift from the class breaks used to make it.
    """
    severity, _, _ = load_raster(find_raster(data_root, fire_prefix, "target_severity"))
    p_high_emb, _, _ = load_raster(find_raster(data_root, fire_prefix, "pHigh_emb"))
    p_high_geo, _, _ = load_raster(find_raster(data_root, fire_prefix, "pHigh_geo"))
    brier_emb, _, _ = load_raster(find_raster(data_root, fire_prefix, "brier_emb"))
    brier_geo, _, _ = load_raster(find_raster(data_root, fire_prefix, "brier_geo"))
    brier_diff, _, _ = load_raster(find_raster(data_root, fire_prefix, "brier_diff"))
    fuel_groups, _, _ = load_raster(find_raster(data_root, fire_prefix, "fuel_groups"))

    try:
        slope_bins, _, _ = load_raster(find_raster(data_root, fire_prefix, "slope_bins"))
    except FileNotFoundError:
        slope_bins = np.zeros_like(severity)

    try:
        built_mask, _, _ = load_raster(find_raster(data_root, fire_prefix, "built_mask"))
    except FileNotFoundError:
        built_mask = None

    obs_binary = np.where(
        np.isnan(severity), np.nan, (severity >= high_class_min).astype(np.float32)
    )
    valid_mask = (~np.isnan(obs_binary)) & (~np.isnan(p_high_emb))

    return FireRasters(
        severity=severity,
        p_high_emb=p_high_emb,
        p_high_geo=p_high_geo,
        brier_emb=brier_emb,
        brier_geo=brier_geo,
        brier_diff=brier_diff,
        fuel_groups=fuel_groups,
        slope_bins=slope_bins,
        built_mask=built_mask,
        obs_binary=obs_binary,
        valid_mask=valid_mask,
    )


def brier_score(predicted: np.ndarray, observed: np.ndarray) -> float:
    """Mean Brier score over the pixels where both arrays are finite."""
    finite = (~np.isnan(predicted)) & (~np.isnan(observed))
    if not finite.any():
        return float("nan")
    return float(np.mean((predicted[finite] - observed[finite]) ** 2))


def brier_summary(rasters: FireRasters) -> Dict[str, float]:
    """Brier scores for both methods and the share of pixels each wins."""
    mean_emb = float(np.nanmean(rasters.brier_emb))
    mean_geo = float(np.nanmean(rasters.brier_geo))
    improvement = mean_geo - mean_emb
    pct_emb_better = float(
        (rasters.brier_diff[rasters.valid_mask] < 0).mean() * 100
    )
    return {
        "brier_emb": mean_emb,
        "brier_geo": mean_geo,
        "brier_improvement": improvement,
        "brier_improvement_pct": (
            float(improvement / mean_geo * 100) if mean_geo else float("nan")
        ),
        "pct_pixels_emb_better": pct_emb_better,
        "valid_pixels": int(rasters.valid_mask.sum()),
    }


def classification_metrics(
    rasters: FireRasters, threshold: float = 0.5
) -> pd.DataFrame:
    """Recall, precision, F1 and accuracy for both methods at one threshold."""
    observed_high = (rasters.obs_binary > 0.5) & rasters.valid_mask
    rows = []
    for label, predicted in (
        ("Embedding", rasters.p_high_emb),
        ("Geographic", rasters.p_high_geo),
    ):
        predicted_high = (predicted > threshold) & rasters.valid_mask
        tp = int((predicted_high & observed_high).sum())
        fp = int((predicted_high & ~observed_high & rasters.valid_mask).sum())
        fn = int((~predicted_high & observed_high).sum())
        tn = int((~predicted_high & ~observed_high & rasters.valid_mask).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        total = tp + fp + fn + tn
        rows.append(
            {
                "method": label,
                "threshold": threshold,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": (tp + tn) / total if total else 0.0,
                "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def threshold_sweep(
    rasters: FireRasters, thresholds: Iterable[float]
) -> pd.DataFrame:
    """Recall and precision for both methods across a range of thresholds.

    The published decision threshold is a reporting convention, not an optimum.
    This sweep is what lets a reader see how the choice moves the trade-off.
    """
    observed_high = (rasters.obs_binary > 0.5) & rasters.valid_mask
    rows = []
    for threshold in thresholds:
        emb_high = (rasters.p_high_emb > threshold) & rasters.valid_mask
        geo_high = (rasters.p_high_geo > threshold) & rasters.valid_mask
        n_obs = observed_high.sum()
        rows.append(
            {
                "threshold": threshold,
                "emb_recall": float((emb_high & observed_high).sum() / n_obs)
                if n_obs
                else 0.0,
                "geo_recall": float((geo_high & observed_high).sum() / n_obs)
                if n_obs
                else 0.0,
                "emb_precision": float(
                    (emb_high & observed_high).sum() / emb_high.sum()
                )
                if emb_high.sum()
                else 0.0,
                "geo_precision": float(
                    (geo_high & observed_high).sum() / geo_high.sum()
                )
                if geo_high.sum()
                else 0.0,
            }
        )
    return pd.DataFrame(rows)


def calibration_by_fuel_slope(
    rasters: FireRasters, min_pixels: int = 10
) -> pd.DataFrame:
    """Mean predicted and observed values within each fuel by slope cell.

    Cells holding fewer than ``min_pixels`` valid pixels are dropped, because
    their means are too noisy to interpret as calibration.
    """
    fs_code = rasters.fuel_groups * 10 + rasters.slope_bins
    rows = []
    for fuel in range(4):
        for slope in range(4):
            code = fuel * 10 + slope
            cell = (fs_code == code) & rasters.valid_mask
            if cell.sum() < min_pixels:
                continue
            mean_emb = float(np.nanmean(rasters.p_high_emb[cell]))
            mean_geo = float(np.nanmean(rasters.p_high_geo[cell]))
            mean_obs = float(np.nanmean(rasters.obs_binary[cell]))
            rows.append(
                {
                    "fuel_group": fuel,
                    "slope_bin": slope,
                    "fs_code": code,
                    "mean_pred_emb": mean_emb,
                    "mean_pred_geo": mean_geo,
                    "mean_obs": mean_obs,
                    "n_pixels": int(cell.sum()),
                    "bias_emb": mean_emb - mean_obs,
                    "bias_geo": mean_geo - mean_obs,
                }
            )
    return pd.DataFrame(rows)


def reliability(
    rasters: FireRasters,
    predicted: np.ndarray,
    n_bins: int = 10,
    min_pixels_per_bin: int = 50,
) -> tuple:
    """Reliability curve and expected calibration error.

    Returns a table of binned mean predicted and mean observed values plus the
    pixel-count-weighted expected calibration error.
    """
    observed_values = rasters.obs_binary[rasters.valid_mask].ravel()
    predicted_values = predicted[rasters.valid_mask].ravel()

    edges = np.linspace(0, 1, n_bins + 1)
    bin_index = np.clip(np.digitize(predicted_values, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        selected = bin_index == b
        if selected.sum() <= min_pixels_per_bin:
            continue
        rows.append(
            {
                "bin": b,
                "mean_predicted": float(predicted_values[selected].mean()),
                "mean_observed": float(observed_values[selected].mean()),
                "n_pixels": int(selected.sum()),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table, float("nan")

    weights = table["n_pixels"].values
    gap = np.abs(table["mean_predicted"].values - table["mean_observed"].values)
    ece = float(np.sum(weights * gap) / weights.sum())
    return table, ece
