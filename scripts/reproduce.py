"""Reproduce every table, number and figure in the manuscript from the archived data.

Usage
-----
    python scripts/reproduce.py                 # data in ./data, results to ./results
    python scripts/reproduce.py --data PATH     # data somewhere else
    python scripts/reproduce.py --skip-pixel    # event-level analysis only

No Earth Engine account is needed. The script reads the fire library CSVs and
the per-fire rasters from the Zenodo archive (see scripts/download_data.py),
runs the analysis, writes every output to the results folder, and finally
compares the headline numbers with config/expected_results.csv. It exits with
a non-zero status if any number differs, so it can also be used as a check.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fireanalogs import exposure, figures, pixel  # noqa: E402
from fireanalogs.config import Config  # noqa: E402
from fireanalogs.library import load_library  # noqa: E402
from fireanalogs.pipeline import run_event_level  # noqa: E402

METHOD_KEYS = {
    "Embedding (novel)": "embedding",
    "Geographic": "geographic",
    "Seasonal": "seasonal",
    "Geo + Seasonal": "geo_seasonal",
}


def say(message: str) -> None:
    print(message, flush=True)


def event_level(cfg: Config, data: Path, out: Path) -> list:
    """Stages 1 to 3: library, leave-one-out validation, tests and diagnostics."""
    library = load_library(data, n_emb_dims=cfg.n_emb_dims)
    say(f"  library: {len(library)} fires, "
        f"{int(library['obs_high_frac'].notna().sum())} with a valid observed fraction")

    r = run_event_level(library, cfg)
    loo = r["loo_results"]

    library.drop(columns=[".geo"], errors="ignore").to_csv(out / "fire_library_merged.csv", index=False)
    loo.to_csv(out / "loo_results.csv", index=False)
    r["loo_metrics"].to_csv(out / "loo_metrics.csv")
    r["method_comparison"].to_csv(out / "method_comparison_wilcoxon_steiger_holm.csv", index=False)
    r["k_sensitivity"].to_csv(out / "k_sensitivity.csv", index=False)
    r["fuel_slope_transfer"].to_csv(out / "fuel_slope_transfer.csv", index=False)
    r["ensemble_comparison"].to_csv(out / "ensemble_comparison_holm.csv", index=False)
    r["similarity_error"]["by_quartile"].to_csv(out / "similarity_error_quartiles.csv")

    # Figures from the event-level tables.
    figures.loo_metrics_bars(r["loo_metrics"], out / "fig_loo_metrics.png")
    figures.predicted_vs_observed(loo, out / "fig_pred_vs_obs.png")
    figures.similarity_vs_error(loo, r["similarity_error"]["by_quartile"], out / "fig_similarity_vs_error.png")
    figures.k_sensitivity(r["k_sensitivity"], cfg.k_default, out / "fig_k_sensitivity.png")
    figures.manuscript_figure4(loo, out / "Figure4_similarity_error.png")
    plt.close("all")

    # Headline numbers, named so they can be checked one by one.
    rows = [
        ("n_library_fires", len(library)),
        ("n_validation_fires", len(loo)),
        ("n_fallback_targets", r["fallback"]["n_fallback"]),
        ("top_cosine_mean", loo["top_cosine"].mean()),
    ]
    for label, row in r["loo_metrics"].iterrows():
        key = METHOD_KEYS[label]
        for metric in ["RMSE", "MAE", "Bias", "Pearson r"]:
            rows.append((f"{key}_{metric.lower().replace(' ', '_')}", row[metric]))
    for _, row in r["method_comparison"].iterrows():
        key = row["Comparison"].replace("Embedding vs ", "").lower().replace(" + ", "_").replace(" ", "_")
        key = {"geo_seasonal": "geo_seasonal"}.get(key, key)
        rows += [
            (f"wilcoxon_W_vs_{key}", row["Wilcoxon W"]),
            (f"wilcoxon_holm_p_vs_{key}", row["Wilcoxon Holm-adj p"]),
            (f"steiger_z_vs_{key}", row["Steiger z"]),
            (f"steiger_holm_p_vs_{key}", row["Steiger Holm-adj p"]),
        ]
    s = r["similarity_error"]
    rows += [
        ("similarity_error_pearson_r", s["pearson_r"]),
        ("similarity_error_pearson_p", s["pearson_p"]),
        ("similarity_error_spearman_rho", s["spearman_rho"]),
        ("similarity_error_spearman_p", s["spearman_p"]),
    ]
    for _, row in r["k_sensitivity"].iterrows():
        rows += [(f"k{int(row['k'])}_rmse", row["rmse"]), (f"k{int(row['k'])}_mae", row["mae"])]
    t = r["fuel_slope_transfer_summary"]
    rows += [("fuel_slope_transfer_rmse", t["rmse"]), ("fuel_slope_transfer_mae", t["mae"]),
             ("fuel_slope_transfer_pearson_r", t["pearson_r"])]
    for _, row in r["ensemble_comparison"].iterrows():
        key = row["Comparison"].split(" vs ")[0].replace("pred_", "ensemble_")
        rows += [(f"{key}_rmse", row["RMSE"]), (f"{key}_mae", row["MAE"]),
                 (f"{key}_bias", row["Bias"]), (f"{key}_pearson_r", row["Pearson r"])]
    return rows


def pixel_level(cfg: Config, data: Path, out: Path) -> list:
    """Stage 4: per-fire Brier scores, classification, calibration and exposure."""
    targets = pd.read_csv(REPO / "data" / "manifests" / "target_fires.csv")
    rasters_dir = data / "rasters"
    if not rasters_dir.exists():
        say(f"  no rasters folder at {rasters_dir}, skipping the pixel-level stage")
        return []

    area = cfg.pixel_area_ha
    decision = cfg["pixel_validation"]["decision_threshold"]
    summaries, classes, built, outcomes, rows = [], [], [], [], []

    for _, fire in targets.iterrows():
        prefix, label = fire["file_prefix"], fire["fire_label"]
        key = label.lower().replace(" ", "_").replace("-", "_")
        say(f"  {label}")
        r = pixel.load_fire(rasters_dir, prefix, high_class_min=cfg.high_class_min)

        summary = pixel.brier_summary(r)
        summaries.append({"fire": label, **summary})

        for threshold in sorted({decision, cfg["exposure"]["outcome_threshold"]}):
            c = pixel.classification_metrics(r, threshold=threshold)
            c.insert(0, "fire", label)
            classes.append(c)

        b = exposure.built_exposure(r, pixel_area_ha=area,
                                    threshold=cfg["exposure"]["built_threshold"], fire_id=prefix)
        b.insert(0, "fire", label)
        built.append(b)

        o = exposure.outcome_analysis(r, pixel_area_ha=area,
                                      threshold=cfg["exposure"]["outcome_threshold"], fire_id=prefix)
        o.insert(0, "fire", label)
        outcomes.append(o)

        pixel.threshold_sweep(r, cfg["pixel_validation"]["threshold_sweep"]).to_csv(
            out / f"threshold_sweep_{key}.csv", index=False)
        calibration = pixel.calibration_by_fuel_slope(
            r, min_pixels=cfg["pixel_validation"]["min_pixels_per_cell"])
        calibration.to_csv(out / f"calibration_fuelslope_{key}.csv", index=False)

        figures.spatial_validation(r, summary, out / f"fig_spatial_validation_{key}.png")
        figures.reliability_diagram(
            {"Embedding": pixel.reliability(r, r.p_high_emb),
             "Geographic": pixel.reliability(r, r.p_high_geo)},
            out / f"fig_reliability_{key}.png")
        figures.calibration_heatmap(calibration, out / f"fig_calibration_fuelslope_{key}.png")
        plt.close("all")

        at_decision = pixel.classification_metrics(r, threshold=decision).set_index("method")
        rows += [
            (f"{key}_brier_embedding", summary["brier_emb"]),
            (f"{key}_brier_geographic", summary["brier_geo"]),
            (f"{key}_pct_pixels_embedding_better", summary["pct_pixels_emb_better"]),
            (f"{key}_recall_embedding_t{decision}", at_decision.loc["Embedding", "recall"]),
            (f"{key}_recall_geographic_t{decision}", at_decision.loc["Geographic", "recall"]),
            (f"{key}_built_observed_ha", b["true_exposure_ha"].iloc[0]),
            (f"{key}_built_embedding_ha", b["emb_predicted_ha"].iloc[0]),
            (f"{key}_built_geographic_ha", b["geo_predicted_ha"].iloc[0]),
            (f"{key}_built_recall_embedding", b["emb_recall_built"].iloc[0]),
            (f"{key}_built_recall_geographic", b["geo_recall_built"].iloc[0]),
        ]

    summary_table = pd.DataFrame(summaries)
    class_table = pd.concat(classes)
    summary_table.to_csv(out / "pixel_validation_summary.csv", index=False)
    class_table.to_csv(out / "pixel_classification_metrics.csv", index=False)
    labels = ["Berrys Lane\nLongwood", "Streatham", "Mount\nMercer", "Walmer-Fogartys\nGap"]
    figures.manuscript_figure6(summary_table, class_table[class_table["threshold"] == decision],
                               list(targets["fire_label"]), labels, out / "Figure6_pixel_metrics.png")
    plt.close("all")
    pd.concat(built).to_csv(out / "built_exposure.csv", index=False)
    pd.concat(outcomes).to_csv(out / "outcome_analysis.csv", index=False)
    return rows


def check(numbers: pd.DataFrame, expected_path: Path) -> bool:
    """Compare reproduced numbers with the expected values, within tolerance."""
    if not expected_path.exists():
        say(f"\nNo {expected_path.name} found, so nothing to check against.")
        return True
    expected = pd.read_csv(expected_path)
    merged = expected.merge(numbers, on="name", how="left", suffixes=("_expected", ""))
    merged["difference"] = (merged["value"] - merged["value_expected"]).abs()
    merged["ok"] = merged["difference"] <= merged["tolerance"]
    failed = merged[~merged["ok"]]
    say(f"\nChecked {len(merged)} numbers against {expected_path.name}: "
        f"{int(merged['ok'].sum())} match, {len(failed)} differ.")
    if len(failed):
        say(failed[["name", "value_expected", "value", "difference"]].to_string(index=False))
    return failed.empty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=REPO / "data", help="folder holding fire_lib_*.csv and rasters/")
    parser.add_argument("--out", type=Path, default=REPO / "results", help="where outputs are written")
    parser.add_argument("--config", type=Path, default=REPO / "config" / "default.yaml")
    parser.add_argument("--skip-pixel", action="store_true", help="skip the per-fire raster stage")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    start = time.time()

    say("Event-level analysis (library, leave-one-out, tests, robustness)")
    rows = event_level(cfg, args.data, args.out)
    if not args.skip_pixel:
        say("Pixel-level validation and built-surface exposure")
        rows += pixel_level(cfg, args.data, args.out)

    numbers = pd.DataFrame(rows, columns=["name", "value"])
    numbers["value"] = numbers["value"].astype(float)
    numbers.to_csv(args.out / "key_numbers.csv", index=False)
    say(f"\nWrote {len(list(args.out.iterdir()))} files to {args.out} in {time.time() - start:.0f} s")

    expected = REPO / "config" / "expected_results.csv"
    if args.skip_pixel and expected.exists():
        exp = pd.read_csv(expected)
        exp = exp[exp["name"].isin(numbers["name"])]
        tmp = args.out / "_expected_event_level.csv"
        exp.to_csv(tmp, index=False)
        ok = check(numbers, tmp)
        tmp.unlink()
    else:
        ok = check(numbers, expected)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
