"""Retrieval-augmented analog prediction of wildfire burn severity.

Reference implementation for the analysis reported in the manuscript
"Geospatial foundation embeddings for analog-based screening of wildfire
burn severity and built-surface exposure".

The package is organised by pipeline stage:

    config      configuration loading and column naming
    gee         Earth Engine export code (fire library and per-fire rasters)
    library     assembly of the exported CSV chunks into one fire library
    similarity  cosine, haversine and circular day-of-year distance matrices
    retrieval   candidate pools, top-K selection and probability transfer
    validation  leave-one-out validation and event-level metrics
    statistics  Wilcoxon, Steiger and Holm-adjusted comparisons
    robustness  neighbourhood-size sweep, ensembles, transfer-function test
    pixel       per-fire raster validation, Brier scores and calibration
    exposure    built-surface exposure screening
    figures     manuscript figures
"""

__version__ = "1.0.0"

from . import (  # noqa: F401
    config,
    library,
    similarity,
    retrieval,
    validation,
    statistics,
    robustness,
    pixel,
    exposure,
)
