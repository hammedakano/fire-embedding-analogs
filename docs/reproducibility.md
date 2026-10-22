# Reproducibility notes

Things that will bite anyone rerunning this, recorded so they do not have to be
rediscovered.

## Exact reproduction is not guaranteed across versions

**Nearest-neighbour tie-breaking.** `top_k_indices` ranks candidates with
`numpy.argsort`. When two candidates have exactly equal scores the retained
order depends on the sort implementation, so analog sets can differ between
NumPy versions. Cosine similarities on 64-dimensional float embeddings tie only
in degenerate cases, but the possibility is real. A FAISS-based nearest
neighbour search would break ties differently again.

**Earth Engine catalogue drift.** Collections can be revised in place. The
Sentinel-2 harmonised collection, the GHSL built-surface product and the
Satellite Embedding collection have all been revised at some point. Rerunning
the export stage at a later date can therefore produce a slightly different
library from the same code. Record the run date alongside any regenerated
library.

**Resampling at the 10 metre alignment step.** Slope is derived from a 30 metre
DEM and reprojected to the 10 metre analysis grid by Earth Engine's default
resampling. Changing the resampling method changes which pixels fall in which
slope bin, and therefore changes the conditional probability tables.

## Choices that are easy to get wrong

**Embedding year.** The embedding layer is an annual composite. The year before
the event is used, clamped to the availability window in
`config/default.yaml`. It is a pre-fire landscape descriptor, not a real-time
condition layer, and it does not capture drying or curing in the weeks before
ignition.

**Mean pooling footprint.** The embedding vector for a fire is the mean over the
whole fire polygon at 10 metre scale, with `bestEffort=True`. For very large
fires Earth Engine may reduce the working scale to stay within limits, so the
mean is not always taken at exactly 10 metres. Band order follows the
collection's own order and is preserved by the `emb_A00` to `emb_A63` naming.

**Fuel grouping is a proxy, not a fuel-type map.** Groups come from fixed NDVI
and NDMI thresholds evaluated in order. Pixels matching none of rules 1 to 3
fall into group 0. This is a broad greenness and moisture gradient, not a
vegetation classification.

**Two decision thresholds are in play.** Built-surface recall uses 0.5. The
whole-footprint outcome confusion matrices in the original working notebook used
0.3. Both are recorded in `config/default.yaml` and carried in the output
tables, so any published number can be traced to the threshold that produced
it. Check which one a given figure or table used before quoting it.

**Unweighted cell means.** The fuel by slope transfer averages the 16 cells
without weighting by pixel count. This is deliberate and matches the published
diagnostic, but it is not the only defensible choice, and a pixel-weighted mean
would give different numbers.

## Before releasing

- Scrub notebook outputs that contain absolute Drive paths.
- Confirm no Earth Engine service-account key or token is committed. `.gitignore`
  covers the usual paths but is not a substitute for checking.
- The fire asset ID in `config/default.yaml` points at the original author's
  Earth Engine project. Anyone else must substitute their own copy.
- Regenerate `requirements.txt` from the environment that produced the final
  results, not from a fresh install.
