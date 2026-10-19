"""Reusable geographic tile features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString, box


WGS84 = "EPSG:4326"
TARGET_CRS = "EPSG:26918"
FT_PER_M = 3.28084


def make_lines(df, cols):
    """Create projected lines between each pair."""

    left = gpd.GeoSeries(
        gpd.points_from_xy(
            df[cols["left_lon"]],
            df[cols["left_lat"]],
        ),
        crs=WGS84,
    ).to_crs(TARGET_CRS)

    right = gpd.GeoSeries(
        gpd.points_from_xy(
            df[cols["right_lon"]],
            df[cols["right_lat"]],
        ),
        crs=WGS84,
    ).to_crs(TARGET_CRS)

    out = df.copy()

    out["geometry"] = [
        LineString([a, b])
        for a, b in zip(left, right)
    ]

    out = gpd.GeoDataFrame(
        out,
        geometry="geometry",
        crs=TARGET_CRS,
    )

    out["straight_line_length_ft"] = (
        out.geometry.length * FT_PER_M
    )

    return out


def clip_layer(layer, aoi):
    """Return only OSM features touching the AOI."""

    if layer is None or layer.empty:
        return layer

    idx = list(
        layer.sindex.intersection(aoi.bounds)
    )

    if not idx:
        return layer.iloc[0:0].copy()

    layer = layer.iloc[idx].copy()

    return layer[
        layer.intersects(aoi)
    ]


def add_crossings(lines, layer, name, key):
    """Count objects crossed by each pair."""

    count = f"{name}_crossing_count"
    flag = f"{name}_crossing_flag"

    if layer is None or layer.empty:
        lines[count] = 0
        lines[flag] = 0
        return lines

    hits = gpd.sjoin(
        lines[key + ["geometry"]],
        layer[["geometry"]],
        how="left",
        predicate="intersects",
    )

    counts = (
        hits
        .dropna(subset=["index_right"])
        .groupby(key)
        .size()
        .rename(count)
        .reset_index()
    )

    lines = lines.merge(
        counts,
        on=key,
        how="left",
    )

    lines[count] = (
        lines[count]
        .fillna(0)
        .astype(int)
    )

    lines[flag] = (
        lines[count] > 0
    ).astype(int)

    return lines


def add_buildings(lines, buildings, key):
    """Calculate building obstruction features."""

    defaults = {
        "building_count_between": 0,
        "building_between_flag": 0,
        "blocked_length_ft": 0.0,
        "pct_line_blocked_by_building": 0.0,
    }

    if buildings is None or buildings.empty:
        return lines.assign(**defaults)

    buildings = buildings[
        buildings.geometry.geom_type.isin(
            ["Polygon", "MultiPolygon"]
        )
    ].copy()

    if buildings.empty:
        return lines.assign(**defaults)

    buildings = buildings.reset_index(drop=True)
    buildings["building_id"] = buildings.index

    hits = gpd.sjoin(
        lines[key + ["geometry"]],
        buildings[["building_id", "geometry"]],
        how="left",
        predicate="intersects",
    ).dropna(subset=["building_id"])

    if hits.empty:
        return lines.assign(**defaults)

    bgeom = buildings[
        ["building_id", "geometry"]
    ].rename(
        columns={"geometry": "building_geom"}
    )

    hits = hits.merge(
        bgeom,
        on="building_id",
    )

    hits["blocked_length_ft"] = [
        line.intersection(building).length * FT_PER_M
        for line, building in zip(
            hits.geometry,
            hits.building_geom,
        )
    ]

    stats = (
        hits.groupby(key)
        .agg(
            building_count_between=(
                "building_id",
                "size",
            ),
            blocked_length_ft=(
                "blocked_length_ft",
                "sum",
            ),
        )
        .reset_index()
    )

    lines = lines.merge(
        stats,
        on=key,
        how="left",
    )

    lines["building_count_between"] = (
        lines["building_count_between"]
        .fillna(0)
        .astype(int)
    )

    lines["blocked_length_ft"] = (
        lines["blocked_length_ft"]
        .fillna(0)
    )

    lines["building_between_flag"] = (
        lines["building_count_between"] > 0
    ).astype(int)

    lines["pct_line_blocked_by_building"] = (
        lines["blocked_length_ft"]
        / lines["straight_line_length_ft"]
        .replace(0, np.nan)
    ).fillna(0)

    return lines


def split_roads(roads):
    """Split roads into major and local roads."""

    if (
        roads is None
        or roads.empty
        or "fclass" not in roads
    ):
        empty = (
            roads.iloc[0:0].copy()
            if roads is not None
            else gpd.GeoDataFrame()
        )

        return empty, empty

    major = {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
    }

    local = {
        "residential",
        "service",
        "unclassified",
        "living_street",
    }

    return (
        roads[roads.fclass.isin(major)],
        roads[roads.fclass.isin(local)],
    )


def build_tile_features(
    df,
    osm_layers,
    cols,
    aoi_buffer_m=50,
):
    """Build all spatial features for a pair dataset."""

    key = [
        cols["left_id"],
        cols["right_id"],
    ]

    lines = make_lines(df, cols)

    aoi = box(
        *lines.total_bounds
    ).buffer(aoi_buffer_m)

    layers = {
        name: clip_layer(
            osm_layers[name],
            aoi,
        )
        for name in [
            "buildings",
            "roads",
            "waterways",
            "railways",
            "barriers",
            "landuse",
        ]
    }

    major, local = split_roads(
        layers["roads"]
    )

    lines = add_buildings(
        lines,
        layers["buildings"],
        key,
    )

    crossings = {
        "street": layers["roads"],
        "major_road": major,
        "local_road": local,
        "waterway": layers["waterways"],
        "railway": layers["railways"],
        "barrier": layers["barriers"],
        "landuse": layers["landuse"],
    }

    for name, layer in crossings.items():
        lines = add_crossings(
            lines,
            layer,
            name,
            key,
        )

    other_roads = (
        lines["street_crossing_count"]
        - lines["major_road_crossing_count"]
        - lines["local_road_crossing_count"]
    ).clip(lower=0)

    lines["spatial_obstruction_score"] = (
        3 * lines["pct_line_blocked_by_building"]
        + 3 * lines["building_count_between"]
        + 5 * lines["waterway_crossing_count"]
        + 5 * lines["railway_crossing_count"]
        + lines["major_road_crossing_count"]
        + 0.5 * lines["local_road_crossing_count"]
        + 0.1 * other_roads
        + lines["barrier_crossing_count"]
    )

    return pd.DataFrame(
        lines.drop(
            columns="geometry",
            errors="ignore",
        )
    )