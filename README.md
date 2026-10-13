# fireanalogs

Retrieval-augmented analog prediction of wildfire burn severity and
built-surface exposure from geospatial foundation embeddings.

This repository is the reference implementation for the manuscript
*Geospatial foundation embeddings for analog-based screening of wildfire burn
severity and built-surface exposure*, covering the 2025-2026 Victorian fire
season in south-eastern Australia.

## What the method does

For a fire that has not yet been mapped, the framework retrieves the most
similar historical fires and transfers their observed severity behaviour to it.

1. Each fire is summarised by the mean of its pre-fire Google Satellite
   Embedding V1 vector, 64 dimensions, taken over the fire footprint.
2. Similarity between a target fire and each candidate is cosine similarity in
   that embedding space. The top K candidates form the analog set.
3. Each analog carries a table of observed high-severity probabilities across a
   16-cell fuel by slope grid. Those tables are averaged over the analog set.
4. The averaged table is mapped back onto the target footprint, giving a
   probability surface, an analog-disagreement surface, and the list of which
   analogs drove the prediction.
5. The probability surface is intersected with built-surface data to screen
   where severe fire is likely to coincide with built assets.

Three non-embedding baselines use the same transfer step but choose analogs by
geographic distance, by day-of-year distance, or by both. Because only the
retrieval step differs, any difference in skill is attributable to the
retrieval space.

## The prediction-time information rule

Analog candidates are restricted to fires that occurred before the target
event, so retrieval uses only information that would have been available when
the prediction was made. This separates forward analog prediction from
retrospective reconstruction, in which later fires can inform earlier ones and
inflate apparent skill.

The same restriction is applied to every retrieval method, so the comparison is
not affected by differences in candidate availability. Where fewer than K prior
fires exist, which happens for events early in the season, the pool widens to
all other fires. `fireanalogs.retrieval.fallback_report` counts how often that
happens and is what the manuscript reports.

## Repository layout

    config/default.yaml       every parameter used in the published run
    src/fireanalogs/          the importable package
      gee/                    Earth Engine export code, needs authentication
      library.py              merge the exported CSV chunks into one library
      similarity.py           cosine, haversine and circular day-of-year
      retrieval.py            pools, top-K selection, probability transfer
      validation.py           leave-one-out validation and event metrics
      statistics.py           Wilcoxon, Steiger and Holm-adjusted comparisons
      robustness.py           K sweep, transfer ablation, ensembles
      pixel.py                per-fire raster validation and calibration
      exposure.py             built-surface exposure screening
      figures.py              manuscript figures
    notebooks/                one thin driver notebook per stage
    tests/                    unit and regression tests
    data/manifests/           file manifests and data source records
    docs/reproducibility.md   known traps and version sensitivities

## Installation

    pip install -e ".[dev]"
    make test

Earth Engine stages need the optional dependency and an authenticated session.

    pip install -e ".[gee]"
    earthengine authenticate

## Running the pipeline

The stages are numbered and each writes the inputs the next one reads.

| Notebook | Stage | Needs Earth Engine |
| --- | --- | --- |
| `01_build_fire_library.ipynb` | Export and merge the fire library | Yes for export, no for merge |
| `02_retrieval_and_loo.ipynb` | Similarity matrices and leave-one-out validation | No |
| `03_statistics_and_robustness.ipynb` | Paired tests, K sweep, ablation, ensembles | No |
| `04_pixel_validation.ipynb` | Per-fire rasters, Brier, calibration, exposure | Yes for export, no for analysis |
| `05_figures.ipynb` | Manuscript figures | No |

Stages 2, 3 and 5 run from the archived tables alone, so a reviewer can
reproduce every reported number without an Earth Engine account.

## Data

Nothing large is stored in git. The archived data products are on Zenodo and
described in `data/README.md`. Source datasets and their terms are recorded in
`LICENSE-data.md`.

## Licensing

Code is MIT, in `LICENSE`. Derived data products are CC BY 4.0, in
`LICENSE-data.md`, subject to the third-party terms recorded there.

## Citation

See `CITATION.cff`. Please cite both the software record and the paper.
