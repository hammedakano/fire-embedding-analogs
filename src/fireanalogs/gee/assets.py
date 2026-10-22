"""Dataset handles and the fire inventory, with every asset ID pinned.

Collection IDs are recorded here rather than inline so that a future catalogue
change is a single visible edit. Google Earth Engine collections can be
revised in place, so the version actually used for the published run is noted
in ``docs/reproducibility.md``.
"""

from __future__ import annotations

from typing import Any, Dict


def datasets(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the dataset handles used by the summarisers and exporters."""
    import ee

    ee_cfg = config["earth_engine"]

    dem = (
        ee.ImageCollection(ee_cfg["dem_collection"])
        .mosaic()
        .select("DEM")
        .setDefaultProjection("EPSG:3857", None, ee_cfg["dem_scale_m"])
        .rename("elev")
    )
    terrain = ee.Terrain.products(dem)

    return {
        "embedding": ee.ImageCollection(ee_cfg["embedding_collection"]),
        "s2": ee.ImageCollection(ee_cfg["s2_collection"]),
        "era5": ee.ImageCollection(ee_cfg["era5_collection"]),
        "ghsl": ee.ImageCollection(ee_cfg["ghsl_collection"]).mosaic(),
        "dem": dem,
        "slope": terrain.select("slope"),
        "aspect": terrain.select("aspect"),
    }


def fire_collection(config: Dict[str, Any]):
    """Cleaned burn-area polygons from the VicEmergency feed asset.

    Only burn-area features with both a created and an updated timestamp are
    kept. ``created`` supplies the event anchor date used for all compositing
    windows.
    """
    import ee

    raw = ee.FeatureCollection(config["earth_engine"]["fires_asset"])
    return (
        raw.filter(ee.Filter.eq("feedType", "burn-area"))
        .filter(ee.Filter.neq("created", ""))
        .filter(ee.Filter.neq("updated", ""))
        .map(
            lambda f: ee.Feature(f.geometry(), f.toDictionary()).set(
                {
                    "fire_id": ee.String(f.get("id")),
                    "fire_name": ee.String(f.get("sourceTitl")),
                    "start_date": ee.String(f.get("created")).slice(0, 10),
                    "end_date": ee.String(f.get("updated")).slice(0, 10),
                }
            )
        )
    )
