# Licence for the derived data products

The code in this repository is released under the MIT Licence, in `LICENSE`.

The derived data products archived with this repository on Zenodo, namely the
leave-one-out results tables, the event-level and pixel-level metrics, the
per-fire conditional probability tables and the validation rasters for the four
target fires, are released under Creative Commons Attribution 4.0
International (CC BY 4.0).

https://creativecommons.org/licenses/by/4.0/

## Third-party terms that still apply

These products are derived from datasets with their own terms. Redistribution
of the derived products does not grant rights over the source datasets.

- Sentinel-2 surface reflectance. Copernicus Sentinel data, free and open under
  the Copernicus licence.
- Copernicus DEM GLO-30. Distributed by ESA under its own licence terms.
- Global Human Settlement Layer built-surface product. European Commission
  Joint Research Centre, CC BY 4.0.
- Google Satellite Embedding V1. Subject to Google Earth Engine terms of
  service. Confirm the current terms before redistributing embedding values or
  products derived directly from them.
- VicEmergency incident feed. Fire perimeter data from Emergency Management
  Victoria. Confirm attribution and redistribution conditions before
  republishing the perimeters.

## Open item

Whether the mean embedding vectors themselves may be redistributed, as opposed
to results computed from them, has not yet been confirmed against the Google
Earth Engine terms. Until that is settled, the archive should ship the analysis
outputs and withhold `fire_lib_embeddings_only.csv`.
