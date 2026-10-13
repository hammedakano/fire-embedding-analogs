# Reproducibility notes

Things that can go wrong when rerunning this analysis, written down so nobody
has to rediscover them.

## Guards built into the code

**One table for every matrix.** The cosine similarity, geographic distance
and day-of-year distance matrices must all be built from the same library
table, in the same row order. If one of them comes from a copy of the library
that was loaded in a different file order, row i means different fires in
different matrices and the baseline predictions are silently wrong. An early
working notebook had exactly this problem. `fireanalogs.pipeline.build_matrices`
builds every matrix from one table, and `tests/test_row_order.py` fails if
results change when the input files are read in a different order.

**Canonical row order and tie-breaking.** `load_library` sorts fires by event
date and then fire ID. Top-K selection uses a stable sort, so candidates with
exactly equal scores are taken in that order. This matters for the seasonal
baseline, where many fires share a day of year. Different tie-breaking rules
move seasonal RMSE by about plus or minus 0.006 but never change the paired
test result.

**Checked numbers.** `scripts/reproduce.py` compares 101 reported numbers with
`config/expected_results.csv` and exits with an error if any differ.

## Earth Engine exports can change

**Catalogue revisions.** Sentinel-2, GHSL and the Satellite Embedding
collection have all been revised in place at some point. Re-exporting the
library later can give slightly different values from the same code. Record
the export date with any regenerated library.

**Resampling.** Slope comes from a 30 m DEM and is resampled to the 10 m grid
by Earth Engine's default method. Another method changes which pixels fall in
which slope bin.

**Large fires.** Footprint means use `bestEffort=True`, so for very large
fires Earth Engine may work at a coarser scale than 10 m.

## Choices that are easy to get wrong

**Embedding year.** The embedding layer is an annual composite. The year before
the event is used, limited to the years available. It describes the landscape,
not fuel drying in the weeks before ignition.

**Fuel groups are a proxy.** Groups come from fixed NDVI and NDMI rules. They
are a greenness and moisture gradient, not a vegetation map.

**Two thresholds.** Pixel recall, F1 and built-surface recall are reported at
0.5. The whole-footprint outcome table (`outcome_analysis.csv`) uses 0.3. Both
are set in `config/default.yaml` and written into the output tables. Check
which one a number used before quoting it.

**Event-level prediction.** In leave-one-out validation the predicted
high-severity fraction of a fire is the mean observed fraction of its K
analogs. The fuel by slope transfer, which averages the analogs' 16
conditional probability cells, is reported separately as a transfer-skill
diagnostic and is what the probability maps use.

**Unweighted cell means.** The fuel by slope summary averages the 16 cells
without weighting by pixel count. A slope-only version that first averages
out fuel groups gives exactly the same number under equal weights, so it is
not offered as a comparison.

## Before releasing a new version

- Clear notebook outputs (they can contain local paths).
- Check that no Earth Engine key or token is committed.
- Rerun `python scripts/reproduce.py` and confirm all checks pass.
