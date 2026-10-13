# Data

No large files are stored in git. This directory holds manifests that describe
where everything lives and what role it plays.

## Layout expected by the code

    data/
      fire_lib_*.csv                    exported library chunks, full summariser
      fire_lib_embeddings_only.csv      embedding-only export, all fires
      rasters/<fire_prefix>_*.tif       per-fire validation rasters
      manifests/                        the files described below

`fireanalogs.library.load_library` merges the chunks. Empty chunks, which are
months with no qualifying fires, are skipped automatically.

## Manifests

| File | What it records |
| --- | --- |
| `manifests/drive_file_manifest.csv` | Every file in the working Google Drive folders, its role, whether it is current or superseded, and whether it belongs in the release |
| `manifests/data_sources.csv` | Each source dataset, its collection ID, provider and licence |
| `manifests/target_fires.csv` | The four pixel-validation fires and their file prefixes |

## Release set

The Zenodo archive carries the code, the derived result tables, and the
validation rasters for the four target fires. It does not carry the superseded
region-of-interest experiments from the earlier development phase, the empty
export chunks, or the large leave-one-fire-out sample files.

Whether the mean embedding vectors may be redistributed is unresolved. See the
open item in `LICENSE-data.md`. Until it is settled,
`fire_lib_embeddings_only.csv` is withheld from the archive.
