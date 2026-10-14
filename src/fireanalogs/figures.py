"""Manuscript figures.

Every figure is written from the tables produced upstream, so a figure can be
regenerated without rerunning retrieval or validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PALETTE = ["#e63946", "#457b9d", "#2a9d8f", "#f4a261"]
DPI_SCREEN = 150
DPI_PRINT = 300


def _save(fig, path: Optional[str | Path], dpi: int) -> None:
    if path is None:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def loo_metrics_bars(metrics: pd.DataFrame, path=None):
    """Event-level RMSE and MAE by retrieval method."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric in zip(axes, ["RMSE", "MAE"]):
        bars = ax.bar(
            metrics.index, metrics[metric], color=PALETTE, width=0.5, edgecolor="white"
        )
        ax.set_title(f"Leave-one-out {metric}", fontweight="bold")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=20)
        for bar, value in zip(bars, metrics[metric]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )
    fig.suptitle("Retrieval method comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, path, DPI_SCREEN)
    return fig


def predicted_vs_observed(loo: pd.DataFrame, path=None):
    """Predicted against observed high-severity fraction for each method."""
    methods = [
        ("pred_emb", "Embedding"),
        ("pred_geo", "Geographic"),
        ("pred_sea", "Seasonal"),
        ("pred_gs", "Geo + seasonal"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for ax, (column, label), colour in zip(axes.ravel(), methods, PALETTE):
        ax.scatter(
            loo["obs_high_frac"], loo[column], s=18, alpha=0.6, color=colour,
            edgecolors="white", linewidths=0.4,
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1.2)
        r = float(np.corrcoef(loo["obs_high_frac"], loo[column])[0, 1])
        ax.set_title(f"{label}  (r = {r:.3f})", fontweight="bold", fontsize=11)
        ax.set_xlabel("Observed high-severity fraction")
        ax.set_ylabel("Predicted high-severity fraction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, path, DPI_SCREEN)
    return fig


def similarity_vs_error(loo: pd.DataFrame, quartiles: pd.DataFrame, path=None):
    """Top-1 cosine similarity against absolute error, with quartile summary."""
    abs_error = (loo["pred_emb"] - loo["obs_high_frac"]).abs()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(loo["top_cosine"], abs_error, s=16, alpha=0.5, color=PALETTE[0])
    slope, intercept = np.polyfit(loo["top_cosine"], abs_error, 1)
    xs = np.linspace(loo["top_cosine"].min(), loo["top_cosine"].max(), 100)
    axes[0].plot(xs, slope * xs + intercept, color="black", lw=1.5)
    axes[0].set_xlabel("Top-1 cosine similarity")
    axes[0].set_ylabel("Absolute prediction error")
    axes[0].set_title("a) Similarity and error", fontweight="bold")

    axes[1].bar(
        quartiles.index.astype(str),
        quartiles["mean_abs_error"],
        color=PALETTE[1],
        width=0.55,
        edgecolor="white",
    )
    axes[1].plot(
        quartiles.index.astype(str),
        quartiles["median_abs_error"],
        "o--",
        color="black",
        label="median",
    )
    axes[1].set_xlabel("Top-1 similarity quartile")
    axes[1].set_ylabel("Absolute error")
    axes[1].set_title("b) Error by similarity quartile", fontweight="bold")
    axes[1].legend()

    fig.tight_layout()
    _save(fig, path, DPI_SCREEN)
    return fig


def k_sensitivity(k_table: pd.DataFrame, k_default: int = 7, path=None):
    """Error against analog neighbourhood size."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (column, label) in zip(axes, [("rmse", "RMSE"), ("mae", "MAE")]):
        ax.plot(k_table["k"], k_table[column], "o-", color=PALETTE[0], lw=2, ms=8)
        ax.axvline(k_default, color="grey", ls=":", lw=1.2)
        ax.set_xlabel("K (number of retrieved analogs)")
        ax.set_ylabel(label)
        ax.set_title(f"{label} against K", fontweight="bold")
        ax.set_xticks(k_table["k"])
    fig.suptitle("Sensitivity to analog neighbourhood size", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, path, DPI_SCREEN)
    return fig


def manuscript_figure5(k_table: pd.DataFrame, k_default: int = 7, path=None):
    """Figure 5 of the paper: RMSE and MAE against the number of analogs K."""
    fig, ax = plt.subplots(figsize=(7.84, 4.64))
    ax.plot(k_table["k"], k_table["rmse"], "-o", color="#1f77b4", lw=2, ms=7, label="RMSE")
    ax.plot(k_table["k"], k_table["mae"], "--o", color="#ff7f0e", lw=1.6, ms=7, label="MAE")
    ax.axvline(k_default, ls=":", color="#1f77b4", lw=1.3)
    ax.text(k_default + 0.15, 0.41, f"Default (K={k_default})", fontsize=10, va="center")
    ax.set_xticks(list(k_table["k"]))
    ax.set_ylim(0.25, 0.42)
    ax.set_xlabel("Number of analogs (K)", fontsize=13)
    ax.set_ylabel("Error metric", fontsize=13)
    ax.legend(frameon=False, fontsize=11)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=DPI_PRINT)
    return fig


def spatial_validation(rasters, summary: dict, path=None):
    """Six-panel pixel-level validation figure for one target fire."""
    import matplotlib.pyplot as plt

    valid = ~np.isnan(rasters.p_high_emb)
    observed = np.where(valid, rasters.obs_binary, np.nan)

    panels = [
        (observed, "a) Observed high severity", "Reds", 0, 1, "High severity"),
        (rasters.p_high_emb, "b) Predicted P(high)\nEmbedding retrieval", "YlOrRd", 0, 1, "Probability"),
        (rasters.p_high_geo, "c) Predicted P(high)\nGeographic baseline", "YlOrRd", 0, 1, "Probability"),
        (rasters.brier_emb, "d) Brier score, embedding", "hot_r", 0, 0.5, "Brier score"),
        (rasters.brier_geo, "e) Brier score, geographic", "hot_r", 0, 0.5, "Brier score"),
        (rasters.brier_diff, "f) Brier difference\n(green favours embedding)", "RdYlGn_r", -0.3, 0.3, "Emb minus geo"),
    ]

    fig = plt.figure(figsize=(18, 12))
    grid = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.15)
    axes = [fig.add_subplot(grid[r, c]) for r in range(2) for c in range(3)]

    for ax, (data, title, cmap, vmin, vmax, label) in zip(axes, panels):
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad(color="none")
        image = ax.imshow(
            data, cmap=cmap_obj, vmin=vmin, vmax=vmax,
            interpolation="nearest", aspect="auto",
        )
        ax.set_title(title, fontsize=12, fontweight="bold", pad=6)
        ax.axis("off")
        bar = plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        bar.set_label(label, fontsize=14)
        bar.ax.tick_params(labelsize=11)

    annotations = [
        (3, f"Mean Brier = {summary['brier_emb']:.4f}", "white", "black"),
        (4, f"Mean Brier = {summary['brier_geo']:.4f}", "white", "black"),
        (5, f"{summary['pct_pixels_emb_better']:.1f}% pixels favour embedding", "black", "white"),
    ]
    for index, text, colour, box in annotations:
        axes[index].text(
            0.05, 0.05, text, transform=axes[index].transAxes, fontsize=13,
            color=colour, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc=box, alpha=0.6),
        )

    _save(fig, path, DPI_PRINT)
    return fig


def reliability_diagram(tables: dict, path=None):
    """Reliability curves for each method, with expected calibration error."""
    fig, axes = plt.subplots(1, len(tables), figsize=(6 * len(tables), 5))
    if len(tables) == 1:
        axes = [axes]
    for ax, (label, (table, ece)), colour in zip(axes, tables.items(), PALETTE):
        sizes = table["n_pixels"] / table["n_pixels"].max() * 400 + 40
        ax.scatter(
            table["mean_predicted"], table["mean_observed"], s=sizes,
            color=colour, alpha=0.85, edgecolors="white", zorder=3,
        )
        ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Mean observed fraction")
        ax.set_title(f"Reliability, {label}", fontweight="bold")
        ax.text(
            0.05, 0.92, f"ECE = {ece:.4f}", transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )
        ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save(fig, path, DPI_PRINT)
    return fig


def calibration_heatmap(calibration: pd.DataFrame, path=None):
    """Predicted and observed probabilities across the fuel by slope grid."""
    import matplotlib.colors as mcolors

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    specs = [
        ("mean_pred_emb", "Predicted P(high)\nEmbedding", "YlOrRd", None),
        ("mean_pred_geo", "Predicted P(high)\nGeographic", "YlOrRd", None),
        ("bias_emb", "Bias (predicted minus observed)\nEmbedding", "RdBu_r", 0),
    ]
    for ax, (column, title, cmap, centre) in zip(axes, specs):
        pivot = calibration.pivot(index="fuel_group", columns="slope_bin", values=column)
        if centre is not None:
            vmax = calibration[column].abs().max()
            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
            image = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")
        else:
            image = ax.imshow(pivot.values, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Slope bin")
        ax.set_ylabel("Fuel group")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_xticklabels(["<10", "10-20", "20-30", ">30"][: pivot.shape[1]])
        ax.set_yticklabels(["Sparse", "Low", "Medium", "Dense"][: pivot.shape[0]])

        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                value = pivot.values[r, c]
                if not np.isnan(value):
                    ax.text(
                        c, r, f"{value:.2f}", ha="center", va="center",
                        fontsize=9, fontweight="bold",
                        color="white" if abs(value) > 0.5 else "black",
                    )
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Calibration by fuel group and slope bin", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, path, DPI_PRINT)
    return fig


def manuscript_figure4(loo: pd.DataFrame, path=None):
    """Manuscript Figure 4: top-1 cosine similarity against absolute error.

    Panel (a) is the scatter with a fitted linear trend and the Pearson and
    Spearman statistics. Panel (b) shows absolute error by similarity quartile.
    """
    from scipy import stats

    abs_error = (loo["pred_emb"] - loo["obs_high_frac"]).abs()
    similarity = loo["top_cosine"]
    pearson_r, pearson_p = stats.pearsonr(similarity, abs_error)
    spearman_r, spearman_p = stats.spearmanr(similarity, abs_error)

    def fmt(p):
        return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"

    fig, axes = plt.subplots(1, 2, figsize=(11, 11 / 2.168))
    axes[0].scatter(similarity, abs_error, s=10, alpha=0.6, color="#2f7bb6")
    slope, intercept = np.polyfit(similarity, abs_error, 1)
    xs = np.linspace(similarity.min(), 1, 50)
    axes[0].plot(xs, slope * xs + intercept, "--", color="#0b3d6e", lw=1.5)
    axes[0].set_xlabel("Top-1 cosine similarity", fontsize=13)
    axes[0].set_ylabel("Absolute prediction error", fontsize=13)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_xlim(similarity.min() - 0.005, 1.005)
    axes[0].text(
        0.02, 0.97,
        f"Pearson r = {pearson_r:.3f}, {fmt(pearson_p)}\nSpearman ρ = {spearman_r:.3f}, {fmt(spearman_p)}",
        transform=axes[0].transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.6"),
    )

    quartile = pd.qcut(similarity, 4, labels=["Q1\n(lowest)", "Q2", "Q3", "Q4\n(highest)"])
    groups = [abs_error[quartile == label].values for label in quartile.cat.categories]
    boxes = axes[1].boxplot(
        groups, patch_artist=True, showfliers=False, widths=0.55,
        medianprops=dict(color="#e8711a", lw=2),
    )
    for box in boxes["boxes"]:
        box.set(facecolor="#b3d1ea", edgecolor="0.35")
    axes[1].plot(range(1, 5), [g.mean() for g in groups], "--o", color="#0b5a9c", ms=7, label="Mean")
    axes[1].plot([], [], color="black", label="Median (boxplot line)")
    axes[1].set_xticks(range(1, 5))
    axes[1].set_xticklabels(quartile.cat.categories)
    axes[1].set_xlabel("Cosine similarity quartile", fontsize=13)
    axes[1].set_ylabel("Absolute prediction error", fontsize=13)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(frameon=False, loc="upper right")
    for ax, label in zip(axes, ["(a)", "(b)"]):
        ax.text(-0.1, 1.04, label, transform=ax.transAxes, fontsize=14, fontweight="bold")
    fig.tight_layout()
    if path is not None:
        # fixed canvas, no tight cropping, so the aspect ratio matches the manuscript frame
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI_PRINT)
    return fig


def manuscript_figure6(summary: pd.DataFrame, classification: pd.DataFrame, fires, labels=None, path=None):
    """Manuscript Figure 6: Brier scores, and recall and F1 at one threshold.

    ``summary`` is pixel_validation_summary.csv and ``classification`` is
    pixel_classification_metrics.csv filtered to a single threshold.
    """
    labels = labels or fires
    summary = summary.set_index("fire").loc[fires]
    x = np.arange(len(fires))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(11, 11 / 2.470))
    axes[0].bar(x - width, summary["brier_emb"], width, label="Brier, embedding", color="#1f77b4")
    axes[0].bar(x, summary["brier_geo"], width, label="Brier, geographic", color="#ff7f0e")
    axes[0].bar(x + width, summary["brier_improvement"], width, label="Brier improvement", color="#2ca02c")
    axes[0].legend(loc="upper left", fontsize=9)

    def metric(method, column):
        return classification[classification["method"] == method].set_index("fire").loc[fires, column].values

    observed = []
    for fire in fires:
        row = classification[(classification["fire"] == fire) & (classification["method"] == "Embedding")].iloc[0]
        observed.append((row.TP + row.FN) / (row.TP + row.FN + row.FP + row.TN))
    axes[1].plot(x, metric("Embedding", "recall"), "-o", color="#1f77b4", label="Recall, embedding")
    axes[1].plot(x, metric("Geographic", "recall"), "--o", color="#ff7f0e", label="Recall, geographic")
    axes[1].plot(x, metric("Embedding", "f1"), "-s", color="#2ca02c", label="F1, embedding")
    axes[1].plot(x, metric("Geographic", "f1"), "--s", color="#d62728", label="F1, geographic")
    axes[1].plot(x, observed, "D", color="#1f77b4", ms=7, ls="none", label="Observed high-severity fraction")
    axes[1].legend(loc="center", bbox_to_anchor=(0.52, 0.45), fontsize=8.5, framealpha=0.95)
    for ax, label in zip(axes, ["(a)", "(b)"]):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight="bold", fontsize=9)
        ax.set_ylabel("Metric value", fontweight="bold")
        ax.text(-0.12, 1.03, label, transform=ax.transAxes, fontsize=14, fontweight="bold")
    axes[0].set_ylim(0, 1.05)
    axes[1].set_ylim(-0.02, 1.05)
    fig.tight_layout()
    if path is not None:
        # fixed canvas, no tight cropping, so the aspect ratio matches the manuscript frame
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=DPI_PRINT)
    return fig
