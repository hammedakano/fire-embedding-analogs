# Data

The data are not stored in git. Get them with

    python scripts/download_data.py --record ZENODO-RECORD-ID

which places them here:

    data/
      fire_lib_*.csv        the fire library, 334 fires in ten export files
      rasters/              12 GeoTIFF layers for each of the four target fires
      results/              the outputs of reproduce.py, for reference
      README.md             full description of every column and layer
      LICENSE.md            licence and source attributions
      CHECKSUMS.sha256      used to verify the download

Only this `manifests/` folder is kept in git.

| File | What it records |
| --- | --- |
| `manifests/target_fires.csv` | The four pixel-validation fires and their file prefixes |
| `manifests/data_sources.csv` | Each source dataset, its Earth Engine ID, provider and licence |
