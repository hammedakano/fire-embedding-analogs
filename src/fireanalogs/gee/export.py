"""Earth Engine summarisers and export tasks.

Two summarisers are used. The full one produces everything a fire needs to act
as a leave-one-out target, namely the embedding vector, the severity histogram
and the 16 fuel by slope conditional probabilities. The embedding-only one
produces the vector alone, which is enough for a fire to serve as an analog and
is far cheaper, so it can be run over every event in the inventory.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..config import (
    BASE_COLUMNS,
    embedding_columns,
    fuel_slope_columns,
    severity_histogram_columns,
)


def _image_helpers(config: Dict[str, Any], ds: Dict[str, Any]):
    import ee

    comp = config["compositing"]
    sev = config["severity"]
    fuel = config["fuel"]
    slope_breaks = config["slope_bins_deg"]
    emb_cfg = config["embedding"]

    def parse_date(value):
        return ee.Date.parse("YYYY-MM-dd", ee.String(value))

    def mask_s2(image):
        # Sentinel-2 scene classification: 3 cloud shadow, 8 to 10 cloud,
        # 11 snow. All are removed before compositing.
        scl = image.select("SCL")
        keep = (
            scl.neq(3)
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )
        return image.updateMask(keep).copyProperties(image, image.propertyNames())

    def s2_composite(start, end, geometry):
        return (
            ds["s2"]
            .filterBounds(geometry)
            .filterDate(start, end)
            .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", 60))
            .map(mask_s2)
            .median()
            .clip(geometry)
        )

    def nbr(image):
        return image.normalizedDifference(["B8", "B12"]).rename("nbr")

    def dnbr(pre, post):
        return nbr(pre).subtract(nbr(post)).rename("dnbr")

    def severity_class(dnbr_image):
        expression = (
            f"(d<={sev['t1']})?0:(d<={sev['t2']})?1:(d<={sev['t3']})?2:3"
        )
        return dnbr_image.expression(
            expression, {"d": dnbr_image.select("dnbr")}
        ).rename("severity")

    def fuel_group_image(pre):
        ndvi = pre.normalizedDifference(["B8", "B4"]).rename("ndvi")
        ndmi = pre.normalizedDifference(["B8", "B11"]).rename("ndmi")
        return (
            ee.Image(0)
            .where(ndvi.lt(fuel["ndvi_low"]), 0)
            .where(
                ndvi.gte(fuel["ndvi_low"])
                .And(ndvi.lt(fuel["ndvi_high"]))
                .And(ndmi.lt(fuel["ndmi_dry"])),
                1,
            )
            .where(ndvi.gte(fuel["ndvi_high"]).And(ndmi.lt(fuel["ndmi_moderate"])), 2)
            .where(ndvi.gte(fuel["ndvi_high"]).And(ndmi.gte(fuel["ndmi_moderate"])), 3)
            .rename("fuel_group")
        )

    def slope_bins_image(geometry):
        s = ds["slope"].clip(geometry)
        a, b, c = slope_breaks
        return s.expression(
            f"(s<{a})?0:(s<{b})?1:(s<{c})?2:3", {"s": s}
        ).rename("slope_bin")

    def prefire_embedding(event_date, geometry):
        # The embedding layer is an annual composite, so the year before the
        # event is used, clamped to the availability window.
        year = (
            ee.Number(event_date.get("year"))
            .subtract(1)
            .max(emb_cfg["min_year"])
            .min(emb_cfg["max_year"])
        )
        start = ee.Date.fromYMD(year, 1, 1)
        end = start.advance(1, "year")
        return (
            ee.Image(ds["embedding"].filterBounds(geometry).filterDate(start, end).mosaic())
            .clip(geometry)
            .set("emb_year", year)
        )

    def prefix_dict(dictionary, prefix):
        dictionary = ee.Dictionary(dictionary)
        keys = dictionary.keys()
        return ee.Dictionary.fromLists(
            keys.map(lambda k: ee.String(prefix).cat(ee.String(k))),
            keys.map(lambda k: dictionary.get(k)),
        )

    def severity_histogram(severity_image, geometry):
        hist = severity_image.reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=geometry,
            scale=config["earth_engine"]["export_scale_m"],
            bestEffort=True,
            tileScale=4,
            maxPixels=int(1e13),
        ).get("severity")
        return ee.Dictionary(
            ee.Algorithms.If(
                ee.Algorithms.IsEqual(hist, None), ee.Dictionary({}), hist
            )
        )

    def conditional_probabilities(high_image, fuel_image, slope_image, geometry):
        """Mean high-severity indicator within each fuel by slope cell.

        Empty cells return 0 rather than null so that every fire exports the
        same 16 columns and the library stays rectangular.
        """
        fuel_ids = ee.List.sequence(0, 3)
        slope_ids = ee.List.sequence(0, 3)

        keys = fuel_ids.map(
            lambda f: slope_ids.map(
                lambda s: ee.String("p_high_f")
                .cat(ee.Number(f).format("%.0f"))
                .cat("_s")
                .cat(ee.Number(s).format("%.0f"))
            )
        ).flatten()

        def cell_mean(f, s):
            masked = high_image.updateMask(
                fuel_image.eq(ee.Number(f)).And(slope_image.eq(ee.Number(s)))
            )
            return masked.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=config["earth_engine"]["export_scale_m"],
                bestEffort=True,
                tileScale=4,
                maxPixels=int(1e13),
            ).get("high_sev")

        values = fuel_ids.map(
            lambda f: slope_ids.map(
                lambda s: ee.Algorithms.If(
                    ee.Algorithms.IsEqual(cell_mean(f, s), None), 0, cell_mean(f, s)
                )
            )
        ).flatten()

        return ee.Dictionary.fromLists(keys, values)

    return {
        "parse_date": parse_date,
        "s2_composite": s2_composite,
        "dnbr": dnbr,
        "severity_class": severity_class,
        "fuel_group_image": fuel_group_image,
        "slope_bins_image": slope_bins_image,
        "prefire_embedding": prefire_embedding,
        "prefix_dict": prefix_dict,
        "severity_histogram": severity_histogram,
        "conditional_probabilities": conditional_probabilities,
        "comp": comp,
        "high_class_min": sev["high_class_min"],
    }


def make_summarisers(config: Dict[str, Any], ds: Dict[str, Any]):
    """Return the full and embedding-only per-fire summariser functions."""
    import ee

    h = _image_helpers(config, ds)
    comp = h["comp"]
    scale = config["earth_engine"]["export_scale_m"]

    def summarise_fire(feature):
        geometry = feature.geometry()
        event_date = h["parse_date"](feature.get("start_date"))

        pre = h["s2_composite"](
            event_date.advance(-comp["pre_days"], "day"),
            event_date.advance(-comp["pre_buffer"], "day"),
            geometry,
        )
        post = h["s2_composite"](
            event_date.advance(comp["post_buffer"], "day"),
            event_date.advance(comp["post_days"], "day"),
            geometry,
        )

        severity = h["severity_class"](h["dnbr"](pre, post))
        high = severity.gte(h["high_class_min"]).rename("high_sev")
        embedding = h["prefire_embedding"](event_date, geometry)

        embedding_mean = embedding.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            tileScale=4,
            maxPixels=int(1e13),
        )
        terrain = (
            ds["dem"]
            .addBands(ds["slope"].rename("slope"))
            .addBands(ds["aspect"].rename("aspect"))
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=geometry,
                scale=config["earth_engine"]["dem_scale_m"],
                bestEffort=True,
                tileScale=4,
                maxPixels=int(1e13),
            )
        )

        fuel = h["fuel_group_image"](pre)
        slope_bins = h["slope_bins_image"](geometry)

        return (
            feature.setMulti(h["prefix_dict"](embedding_mean, "emb_"))
            .setMulti(h["prefix_dict"](terrain, "trn_"))
            .setMulti(h["conditional_probabilities"](high, fuel, slope_bins, geometry))
            .setMulti(h["prefix_dict"](h["severity_histogram"](severity, geometry), "sev_hist_"))
            .set(
                {
                    "event_date": event_date.format("YYYY-MM-dd"),
                    "emb_year": embedding.get("emb_year"),
                    "area_ha": geometry.area(1).divide(10000),
                }
            )
        )

    def summarise_fire_embedding_only(feature):
        geometry = feature.geometry()
        event_date = h["parse_date"](feature.get("start_date"))
        embedding = h["prefire_embedding"](event_date, geometry)
        embedding_mean = embedding.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=scale,
            bestEffort=True,
            tileScale=4,
            maxPixels=int(1e13),
        )
        return feature.setMulti(h["prefix_dict"](embedding_mean, "emb_")).set(
            {
                "event_date": event_date.format("YYYY-MM-dd"),
                "emb_year": embedding.get("emb_year"),
                "area_ha": geometry.area(1).divide(10000),
            }
        )

    return summarise_fire, summarise_fire_embedding_only


def full_export_columns(n_emb_dims: int = 64) -> List[str]:
    """Column order for the full library export."""
    return (
        BASE_COLUMNS
        + embedding_columns(n_emb_dims)
        + fuel_slope_columns()
        + severity_histogram_columns()
    )


def embedding_only_export_columns(n_emb_dims: int = 64) -> List[str]:
    """Column order for the embedding-only library export."""
    return [
        "fire_id",
        "fire_name",
        "event_date",
        "emb_year",
        "area_ha",
    ] + embedding_columns(n_emb_dims)


def export_library_chunk(collection, summariser, columns, description, folder):
    """Submit one library export task.

    The inventory is exported in monthly or weekly chunks because a single task
    over the full season exceeds the Earth Engine memory limit. Each chunk is
    written as its own CSV and merged later by
    :func:`fireanalogs.library.load_library`.
    """
    import ee

    compact = collection.map(summariser).map(
        lambda f: ee.Feature(f.geometry().centroid(100), f.toDictionary(columns))
    )
    task = ee.batch.Export.table.toDrive(
        collection=compact,
        description=description,
        folder=folder,
        fileFormat="CSV",
    )
    task.start()
    return task


def export_target_fire_rasters(
    config: Dict[str, Any],
    ds: Dict[str, Any],
    target_geometry,
    event_date,
    cell_probabilities_emb,
    cell_probabilities_geo,
    fire_prefix: str,
    folder: str,
):
    """Export the twelve validation rasters for one target fire.

    The predicted probability rasters are built by remapping the fuel by slope
    code image through the analog-derived cell probabilities, so the raster is
    exactly the transfer function applied spatially.
    """
    import ee

    from ..config import fuel_slope_codes

    h = _image_helpers(config, ds)
    comp = h["comp"]
    scale = config["earth_engine"]["export_scale_m"]

    pre = h["s2_composite"](
        event_date.advance(-comp["pre_days"], "day"),
        event_date.advance(-comp["pre_buffer"], "day"),
        target_geometry,
    )
    post = h["s2_composite"](
        event_date.advance(comp["post_buffer"], "day"),
        event_date.advance(comp["post_days"], "day"),
        target_geometry,
    )

    dnbr_image = h["dnbr"](pre, post)
    severity = h["severity_class"](dnbr_image)
    observed_high = severity.gte(h["high_class_min"]).rename("obs_high").toFloat()

    fuel = h["fuel_group_image"](pre).clip(target_geometry)
    slope_bins = h["slope_bins_image"](target_geometry)
    fs_code = fuel.multiply(10).add(slope_bins).rename("fs_code").clip(target_geometry)

    codes = fuel_slope_codes()
    p_high_emb = (
        fs_code.remap(codes, list(map(float, cell_probabilities_emb)))
        .rename("p_high_emb")
        .clip(target_geometry)
    )
    p_high_geo = (
        fs_code.remap(codes, list(map(float, cell_probabilities_geo)))
        .rename("p_high_geo")
        .clip(target_geometry)
    )

    brier_emb = p_high_emb.subtract(observed_high).pow(2).rename("brier_emb")
    brier_geo = p_high_geo.subtract(observed_high).pow(2).rename("brier_geo")
    brier_diff = brier_emb.subtract(brier_geo).rename("brier_diff")

    built_mask = (
        ds["ghsl"].select("built_surface").gt(0).clip(target_geometry).rename("built")
    )
    expected_built = (
        built_mask.multiply(p_high_emb)
        .multiply(ee.Image.pixelArea())
        .rename("exp_built_m2")
    )

    images = [
        (observed_high, "obs_high_binary"),
        (dnbr_image, "target_dnbr"),
        (severity, "target_severity"),
        (p_high_emb, "pHigh_emb"),
        (p_high_geo, "pHigh_geo"),
        (brier_emb, "brier_emb"),
        (brier_geo, "brier_geo"),
        (brier_diff, "brier_diff"),
        (fuel, "fuel_groups"),
        (slope_bins, "slope_bins"),
        (built_mask, "built_mask"),
        (expected_built, "expected_built_exposure"),
    ]

    tasks = {}
    for image, suffix in images:
        description = f"{fire_prefix}_{suffix}"
        task = ee.batch.Export.image.toDrive(
            image=image.clip(target_geometry),
            description=description,
            folder=folder,
            region=target_geometry,
            scale=scale,
            maxPixels=int(1e13),
            fileFormat="GeoTIFF",
        )
        task.start()
        tasks[description] = task
    return tasks


def wait_for_tasks(tasks: Dict[str, Any], poll_seconds: int = 60) -> Dict[str, str]:
    """Block until every task reaches a terminal state, then report states."""
    import time

    terminal = {"COMPLETED", "FAILED", "CANCELLED", "UNSUBMITTED"}
    while True:
        states = {name: task.status()["state"] for name, task in tasks.items()}
        if all(state in terminal for state in states.values()):
            return states
        time.sleep(poll_seconds)
