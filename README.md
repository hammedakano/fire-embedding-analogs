# fireanalogs

Code for the paper *Geospatial foundation embeddings enable
retrieval-augmented analog prediction of wildfire burn severity and built
exposure*.

For a fire that has not been mapped yet, the method finds the most similar
past fires and uses their observed burn severity to predict where severe fire
is likely. "Similar" is measured with Google Satellite Embedding V1, a
64-number description of the pre-fire landscape. The study covers 334 fires
from the 2025 to 2026 fire season in Victoria, Australia.

## Reproduce the paper in four commands

You need Python 3.10 or newer and about 100 MB of disk space. No Google Earth
Engine account is needed.

    git clone https://github.com/hammedakano/fire-embedding-analogs
    cd fire-embedding-analogs
    pip install -e .
    python scripts/download_data.py
    python scripts/reproduce.py

The last command takes under a minute. It writes every table and figure to
`results/` and then checks 101 reported numbers against
`config/expected_results.csv`. It ends with a line such as

    Checked 101 numbers against expected_results.csv: 101 match, 0 differ.

If you prefer `make`, the same steps are `make install`, `make data RECORD=ID`
and `make reproduce`.

To use exactly the package versions of the published run, install from
`requirements.txt` (Python 3.11) or `environment.yml` (conda) first.

## Where each result in the paper comes from

| In the paper | File in `results/` |
| --- | --- |
| Table 2, event-level validation | `loo_metrics.csv`, per fire in `loo_results.csv` |
| Table S5, paired tests | `method_comparison_wilcoxon_steiger_holm.csv` |
| Section 3.3 and Figure 4, similarity and error | `similarity_error_quartiles.csv`, `Figure4_similarity_error.png` |
| Figure 5, choice of K | `k_sensitivity.csv`, `Figure5_k_sensitivity.png` |
| Fuel by slope transfer skill | `fuel_slope_transfer.csv` |
| Table 3, ensembles | `ensemble_comparison_holm.csv` |
| Figure 6, pixel-level metrics (threshold 0.5) | `pixel_validation_summary.csv`, `pixel_classification_metrics.csv`, `Figure6_pixel_metrics.png` |
| Figure 7 and Figure S2, spatial validation maps | `fig_spatial_validation_<fire>.png` |
| Table S6, built-surface exposure | `built_exposure.csv` |
| Every headline number in one place | `key_numbers.csv` |

## Step by step instead

The five notebooks in `notebooks/` run the same analysis one stage at a time
and are easier to read.

| Notebook | What it does | Earth Engine needed |
| --- | --- | --- |
| `01_build_fire_library` | Merge the exported fire library files | Only to re-export |
| `02_retrieval_and_loo` | Leave-one-out validation of the four retrieval methods | No |
| `03_statistics_and_robustness` | Paired tests, choice of K, transfer skill, ensembles | No |
| `04_pixel_validation` | Brier scores, calibration and exposure for one fire | Only to re-export |
| `05_figures` | The paper figures | No |

## Rerunning from raw satellite data

The archived data are Earth Engine exports. To regenerate them you need an
Earth Engine account and your own copy of the VicEmergency burn-area
perimeters.

    pip install -e ".[gee]"
    earthengine authenticate

Then set `earth_engine.fires_asset` in `config/default.yaml` to your asset and
run the export cells in notebooks 01 and 04. Source collections are sometimes
revised, so a fresh export can differ slightly from the archive.
`docs/reproducibility.md` lists this and other known pitfalls.

## How the method works

1. Each fire is described by the mean of its pre-fire Satellite Embedding
   vectors over the fire footprint.
2. For a target fire, the K most similar earlier fires are found by cosine
   similarity (K = 7). Only fires that happened before the target are used,
   unless fewer than K exist.
3. The target's predicted high-severity fraction is the mean observed fraction
   of those analogs. For maps, each analog's high-severity probability in 16
   fuel by slope classes is averaged and painted back onto the target.
4. Three baselines pick analogs by distance, by day of year, or by both. Only
   the way analogs are picked changes, so differences in skill come from the
   retrieval method alone.

## Repository layout

    config/default.yaml           every setting used in the paper
    config/expected_results.csv   the numbers reproduce.py checks against
    scripts/download_data.py      fetch and verify the Zenodo data
    scripts/reproduce.py          run the whole analysis
    src/fireanalogs/              the Python package
    notebooks/                    the same analysis, one stage at a time
    tests/                        automated tests (run with pytest)
    data/manifests/               the four target fires and the data sources
    docs/reproducibility.md       known pitfalls

## Data and licences

The data are archived on Zenodo, not in this repository, at
https://doi.org/10.5281/zenodo.23384287 (six zip files, which
`scripts/download_data.py` downloads, checks and combines). This code is
archived on Zenodo at https://doi.org/10.5281/zenodo.23385052. The code is under
the MIT licence (`LICENSE`). The data are under CC BY 4.0 with the source
attributions listed in `LICENSE-data.md`.

## Citation

Please cite the paper, the software and the data. `CITATION.cff` has the
details, and GitHub shows a "Cite this repository" button built from it.
