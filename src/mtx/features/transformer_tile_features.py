from __future__ import annotations

import pickle
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import (
    LineString,
    Point,
    box,
)
TARGET_CRS = "EPSG:26918"
WGS84 = "EPSG:4326"
GRID_SIZE_M = 1000
GRID_PADDING_M = 500
TILE_AOI_BUFFER_M = 50
MAX_ROWS_PER_TILE = 1000
CHUNK_SIZE = 500
KEY = [
    "meter_number",
    "candidate_transformer",
]
def load_osm_layers(
    pickle_path,
) -> dict:
    with open(
        pickle_path,
        "rb",
    ) as file:
        layers = pickle.load(
            file
        )
    for layer in layers.values():
        if (
            layer is not None
            and
            not layer.empty
        ):
            _ = layer.sindex
    return layers
def dedupe_pairs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    sort_columns = [
        column
        for column
        in (
            KEY
            + [
                "distance_ft",
            ]
        )
        if column
        in df.columns
    ]
    return (
        df
        .sort_values(
            sort_columns
        )
        .drop_duplicates(
            KEY,
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )
def build_grid(
    bounds,
) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bounds
    minx -= GRID_PADDING_M
    miny -= GRID_PADDING_M
    maxx += GRID_PADDING_M
    maxy += GRID_PADDING_M
    cell_ids = []
    geometries = []
    cell_id = 0
    xs = np.arange(
        minx,
        maxx + GRID_SIZE_M,
        GRID_SIZE_M,
    )
    ys = np.arange(
        miny,
        maxy + GRID_SIZE_M,
        GRID_SIZE_M,
    )
    for x in xs[:-1]:
        for y in ys[:-1]:
            cell_ids.append(
                cell_id
            )
            geometries.append(
                box(
                    x,
                    y,
                    x + GRID_SIZE_M,
                    y + GRID_SIZE_M,
                )
            )
            cell_id += 1
    return gpd.GeoDataFrame(
        {
            "cell_id":
                cell_ids,
        },
        geometry=geometries,
        crs=TARGET_CRS,
    )
def clip_layer(
    layer,
    aoi,
):
    if (
        layer is None
        or
        layer.empty
    ):
        return layer
    indexes = list(
        layer
        .sindex
        .intersection(
            aoi.bounds
        )
    )
    if not indexes:
        return (
            layer
            .iloc[
                0:0
            ]
            .copy()
        )
    subset = (
        layer
        .iloc[
            indexes
        ]
        .copy()
    )
    return subset[
        subset.intersects(
            aoi
        )
    ]
def make_meter_transformer_lines(
    df: pd.DataFrame,
) -> gpd.GeoDataFrame:
    df = dedupe_pairs(
        df
    )
    required = {
        "LONGITUDE",
        "LATITUDE",
        "Location_X",
        "Location_Y",
    }
    missing = (
        required
        - set(df.columns)
    )
    if missing:
        raise ValueError(
            "Tile stage missing coordinate "
            f"columns: {sorted(missing)}"
        )
    meter_points = (
        gpd.GeoSeries(
            gpd.points_from_xy(
                df[
                    "LONGITUDE"
                ],
                df[
                    "LATITUDE"
                ],
            ),
            crs=WGS84,
        )
        .to_crs(
            TARGET_CRS
        )
    )
    transformer_points = (
        gpd.GeoSeries(
            gpd.points_from_xy(
                df[
                    "Location_X"
                ],
                df[
                    "Location_Y"
                ],
            ),
            crs=WGS84,
        )
        .to_crs(
            TARGET_CRS
        )
    )
    output = (
        df.copy()
    )
    output[
        "geometry"
    ] = [
        LineString(
            [
                meter,
                transformer,
            ]
        )
        for (
            meter,
            transformer,
        )
        in zip(
            meter_points.values,
            transformer_points.values,
        )
    ]
    lines = (
        gpd.GeoDataFrame(
            output,
            geometry="geometry",
            crs=TARGET_CRS,
        )
    )
    lines[
        "straight_line_length_ft"
    ] = (
        lines
        .geometry
        .length
        * 3.28084
    )
    return lines
def add_crossings(
    lines,
    layer,
    name,
):
    count_column = (
        f"{name}_crossing_count"
    )
    flag_column = (
        f"{name}_crossing_flag"
    )
    if (
        layer is None
        or
        layer.empty
    ):
        lines[
            count_column
        ] = 0
        lines[
            flag_column
        ] = 0
        return lines
    hits = gpd.sjoin(
        lines,
        layer[
            [
                "geometry",
            ]
        ],
        how="left",
        predicate="intersects",
    )
    counts = (
        hits
        .dropna(
            subset=[
                "index_right"
            ]
        )
        .groupby(
            KEY
        )
        .size()
        .rename(
            count_column
        )
        .reset_index()
    )
    lines = lines.merge(
        counts,
        on=KEY,
        how="left",
    )
    lines[
        count_column
    ] = (
        lines[
            count_column
        ]
        .fillna(0)
        .astype(int)
    )
    lines[
        flag_column
    ] = (
        lines[
            count_column
        ]
        > 0
    ).astype(int)
    return dedupe_pairs(
        lines
    )
def add_building_features(
    lines,
    buildings,
):
    default_columns = {
        "building_count_between": 0,
        "building_between_flag": 0,
        "blocked_length_ft": 0.0,
        "pct_line_blocked_by_building": 0.0,
    }
    if (
        buildings is None
        or
        buildings.empty
    ):
        for (
            column,
            value,
        ) in default_columns.items():
            lines[
                column
            ] = value
        return lines
    buildings = (
        buildings[
            buildings
            .geometry
            .type
            .isin(
                [
                    "Polygon",
                    "MultiPolygon",
                ]
            )
        ]
        .copy()
    )
    if buildings.empty:
        for (
            column,
            value,
        ) in default_columns.items():
            lines[
                column
            ] = value
        return lines
    buildings[
        "building_id"
    ] = (
        buildings
        .index
        .astype(str)
    )
    hits = (
        gpd.sjoin(
            lines,
            buildings[
                [
                    "building_id",
                    "geometry",
                ]
            ],
            how="left",
            predicate="intersects",
        )
        .dropna(
            subset=[
                "building_id"
            ]
        )
    )
    if hits.empty:
        for (
            column,
            value,
        ) in default_columns.items():
            lines[
                column
            ] = value
        return lines
    building_geometry = (
        buildings[
            [
                "building_id",
                "geometry",
            ]
        ]
        .rename(
            columns={
                "geometry":
                    "building_geom",
            }
        )
    )
    hits = hits.merge(
        building_geometry,
        on="building_id",
        how="left",
    )
    hits[
        "blocked_length_ft"
    ] = [
        (
            line
            .intersection(
                building
            )
            .length
            * 3.28084
        )
        if (
            line is not None
            and
            building is not None
        )
        else 0.0
        for (
            line,
            building,
        )
        in zip(
            hits.geometry,
            hits[
                "building_geom"
            ],
        )
    ]
    counts = (
        hits
        .groupby(
            KEY
        )
        .size()
        .rename(
            "building_count_between"
        )
        .reset_index()
    )
    blocked = (
        hits
        .groupby(
            KEY
        )[
            "blocked_length_ft"
        ]
        .sum()
        .reset_index()
    )
    lines = lines.merge(
        counts,
        on=KEY,
        how="left",
    )
    lines = lines.merge(
        blocked,
        on=KEY,
        how="left",
    )
    lines[
        "building_count_between"
    ] = (
        lines[
            "building_count_between"
        ]
        .fillna(0)
        .astype(int)
    )
    lines[
        "blocked_length_ft"
    ] = (
        lines[
            "blocked_length_ft"
        ]
        .fillna(0.0)
    )
    lines[
        "building_between_flag"
    ] = (
        lines[
            "building_count_between"
        ]
        > 0
    ).astype(int)
    lines[
        "pct_line_blocked_by_building"
    ] = (
        lines[
            "blocked_length_ft"
        ]
        /
        lines[
            "straight_line_length_ft"
        ].replace(
            0,
            np.nan,
        )
    ).fillna(0.0)
    return dedupe_pairs(
        lines
    )
def split_roads(
    roads,
):
    if (
        roads is None
        or
        roads.empty
        or
        "fclass"
        not in roads.columns
    ):
        if roads is None:
            empty = (
                gpd.GeoDataFrame(
                    geometry=[],
                    crs=TARGET_CRS,
                )
            )
        else:
            empty = (
                roads
                .iloc[
                    0:0
                ]
                .copy()
            )
        return (
            empty,
            empty,
        )
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
        roads[
            roads[
                "fclass"
            ].isin(
                major
            )
        ].copy(),
        roads[
            roads[
                "fclass"
            ].isin(
                local
            )
        ].copy(),
    )
def process_tile(
    tile_df: pd.DataFrame,
    osm_layers: dict,
) -> pd.DataFrame:
    tile_df = dedupe_pairs(
        tile_df
    )
    lines = (
        make_meter_transformer_lines(
            tile_df
        )
    )
    aoi = box(
        *lines.total_bounds
    ).buffer(
        TILE_AOI_BUFFER_M
    )
    buildings = clip_layer(
        osm_layers.get(
            "buildings"
        ),
        aoi,
    )
    roads = clip_layer(
        osm_layers.get(
            "roads"
        ),
        aoi,
    )
    waterways = clip_layer(
        osm_layers.get(
            "waterways"
        ),
        aoi,
    )
    railways = clip_layer(
        osm_layers.get(
            "railways"
        ),
        aoi,
    )
    barriers = clip_layer(
        osm_layers.get(
            "barriers"
        ),
        aoi,
    )
    landuse = clip_layer(
        osm_layers.get(
            "landuse"
        ),
        aoi,
    )
    (
        roads_major,
        roads_local,
    ) = split_roads(
        roads
    )
    lines = add_building_features(
        lines,
        buildings,
    )
    for (
        layer,
        name,
    ) in [
        (
            roads,
            "street",
        ),
        (
            roads_major,
            "major_road",
        ),
        (
            roads_local,
            "local_road",
        ),
        (
            waterways,
            "waterway",
        ),
        (
            railways,
            "railway",
        ),
        (
            barriers,
            "barrier",
        ),
        (
            landuse,
            "landuse",
        ),
    ]:
        lines = add_crossings(
            lines,
            layer,
            name,
        )
    lines[
        "spatial_obstruction_score"
    ] = (
        lines[
            "pct_line_blocked_by_building"
        ]
        * 3
        +
        lines[
            "building_count_between"
        ]
        * 3
        +
        lines[
            "waterway_crossing_count"
        ]
        * 5
        +
        lines[
            "railway_crossing_count"
        ]
        * 5
        +
        lines[
            "major_road_crossing_count"
        ]
        +
        lines[
            "local_road_crossing_count"
        ]
        * 0.5
        +
        (
            lines[
                "street_crossing_count"
            ]
            -
            lines[
                "major_road_crossing_count"
            ]
            -
            lines[
                "local_road_crossing_count"
            ]
        )
        .clip(
            lower=0
        )
        * 0.1
        +
        lines[
            "barrier_crossing_count"
        ]
    )
    output = (
        lines
        .drop(
            columns=[
                "geometry",
            ],
            errors="ignore",
        )
    )
    return dedupe_pairs(
        pd.DataFrame(
            output
        )
    )
def process_tile_chunked(
    tile_df: pd.DataFrame,
    osm_layers: dict,
) -> pd.DataFrame:
    tile_df = dedupe_pairs(
        tile_df
    )
    if (
        len(tile_df)
        <= MAX_ROWS_PER_TILE
    ):
        return process_tile(
            tile_df,
            osm_layers,
        )
    chunks = []
    for start in range(
        0,
        len(tile_df),
        CHUNK_SIZE,
    ):
        chunk = (
            tile_df
            .iloc[
                start:
                start + CHUNK_SIZE
            ]
            .copy()
        )
        chunks.append(
            process_tile(
                chunk,
                osm_layers,
            )
        )
    output = pd.concat(
        chunks,
        ignore_index=True,
    )
    return dedupe_pairs(
        output
    )
def assign_tiles(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = dedupe_pairs(
        df
    )
    meter_points = (
        gpd.GeoSeries(
            gpd.points_from_xy(
                df[
                    "LONGITUDE"
                ],
                df[
                    "LATITUDE"
                ],
            ),
            crs=WGS84,
        )
        .to_crs(
            TARGET_CRS
        )
    )
    transformer_points = (
        gpd.GeoSeries(
            gpd.points_from_xy(
                df[
                    "Location_X"
                ],
                df[
                    "Location_Y"
                ],
            ),
            crs=WGS84,
        )
        .to_crs(
            TARGET_CRS
        )
    )
    midpoint = (
        gpd.GeoSeries(
            [
                Point(
                    (
                        meter.x
                        + transformer.x
                    )
                    / 2,
                    (
                        meter.y
                        + transformer.y
                    )
                    / 2,
                )
                for (
                    meter,
                    transformer,
                )
                in zip(
                    meter_points,
                    transformer_points,
                )
            ],
            crs=TARGET_CRS,
        )
    )
    assignment = (
        gpd.GeoDataFrame(
            df.copy(),
            geometry=midpoint,
            crs=TARGET_CRS,
        )
    )
    grid = build_grid(
        assignment.total_bounds
    )
    assigned = gpd.sjoin(
        assignment,
        grid,
        how="left",
        predicate="intersects",
    )
    assigned = (
        assigned
        .drop(
            columns=[
                "geometry",
                "index_right",
            ],
            errors="ignore",
        )
        .sort_values(
            KEY
            + [
                "cell_id",
            ]
        )
        .drop_duplicates(
            KEY,
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )
    return assigned