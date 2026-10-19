# scripts/meter_to_transformer/stages/run_tile_features.py

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl

from pyproj import Transformer
from shapely.geometry import LineString, box
import time

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TILE_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "tile_features"
)

CURRENT_DIR = (
    TILE_CACHE_DIR
    / "current"
)

WORK_DIR = Path(
    os.getenv(
        "SMARTLINK_TILE_WORK_DIR",
        str(
            Path(
                os.getenv(
                    "LOCALAPPDATA",
                    str(Path.home()),
                )
            )
            / "SmartLink"
            / "tile_work"
        ),
    )
)

STATE_FILE = (
    CURRENT_DIR
    / "_state.json"
)

DEFAULT_OSM_LAYER_FILE = Path(
    os.getenv(
        "OSM_LAYER_FILE",
        str(
            PROJECT_ROOT
            / "data"
            / "reference"
            / "osm"
            / "osm_layers.pkl"
        ),
    )
)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass(frozen=True)
class TileFeatureConfig:
    """Configuration for spatial tile-feature processing."""

    # Coordinate systems
    target_crs: str = "EPSG:26918"
    wgs84_crs: str = "EPSG:4326"

    # Used only to geographically group candidate pairs.
    # This does not limit candidate distance.
    grid_size_m: float = 1000.0

    # Added around the meter-transformer lines when selecting
    # relevant OSM geometries.
    aoi_buffer_m: float = 50.0

    # Maximum number of candidate pairs processed by
    # GeoPandas/Shapely in one batch.
    max_rows_per_batch: int = 500

    # Batch files are always checkpoints. This controls how
    # frequently progress.json is refreshed.
    checkpoint_every_batches: int = 1

    # Delete checkpoint batches after the division has been
    # successfully assembled and validated.
    delete_work_files_on_success: bool = False

    # Parquet compression.
    compression: str = "zstd"

    # Prevent an unexpectedly large building spatial join
    # from exhausting memory.
    max_building_hits_per_batch: int = 50_000

    # Recompute spatial features when the OSM source or
    # spatial configuration changes.
    invalidate_on_osm_change: bool = True

    # Increment manually when feature equations change.
    feature_schema_version: str = "1"


DEFAULT_TILE_CONFIG = TileFeatureConfig()


# ============================================================
# COLUMNS
# ============================================================

METER = "meter_number"
CANDIDATE = "candidate_transformer"

KEY = [
    METER,
    CANDIDATE,
]

COORD_COLUMNS = [
    "meter_latitude",
    "meter_longitude",
    "transformer_latitude",
    "transformer_longitude",
]

TILE_FEATURE_COLUMNS = [
    "straight_line_length_ft",
    "building_count_between",
    "blocked_length_ft",
    "building_between_flag",
    "pct_line_blocked_by_building",
    "street_crossing_count",
    "street_crossing_flag",
    "major_road_crossing_count",
    "major_road_crossing_flag",
    "local_road_crossing_count",
    "local_road_crossing_flag",
    "waterway_crossing_count",
    "waterway_crossing_flag",
    "railway_crossing_count",
    "railway_crossing_flag",
    "barrier_crossing_count",
    "barrier_crossing_flag",
    "landuse_crossing_count",
    "landuse_crossing_flag",
    "spatial_obstruction_score",
]

OUTPUT_COLUMNS = [
    METER,
    CANDIDATE,
    "division",
    *TILE_FEATURE_COLUMNS,
]

FLOAT_COLUMNS = [
    "straight_line_length_ft",
    "blocked_length_ft",
    "pct_line_blocked_by_building",
    "spatial_obstruction_score",
]

COUNT_COLUMNS = [
    "building_count_between",
    "street_crossing_count",
    "major_road_crossing_count",
    "local_road_crossing_count",
    "waterway_crossing_count",
    "railway_crossing_count",
    "barrier_crossing_count",
    "landuse_crossing_count",
]

FLAG_COLUMNS = [
    "building_between_flag",
    "street_crossing_flag",
    "major_road_crossing_flag",
    "local_road_crossing_flag",
    "waterway_crossing_flag",
    "railway_crossing_flag",
    "barrier_crossing_flag",
    "landuse_crossing_flag",
]


# ============================================================
# JSON HELPERS
# ============================================================

def read_json(
    path: Path,
) -> dict:
    """Read a JSON file, returning an empty dictionary if absent."""

    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def write_json_atomic(
    data: dict,
    path: Path,
) -> None:
    """Write JSON through a temporary file before replacement."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        ".tmp.json"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                default=str,
            )

# Retry temporary Windows file locks for up to ~5 seconds.
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.5)



    finally:
        temporary.unlink(
            missing_ok=True
        )


# ============================================================
# OSM SOURCE SIGNATURE
# ============================================================

def osm_signature(
    osm_file: Path,
) -> dict:
    """Return identifying information for the OSM source file."""

    if not osm_file.exists():
        raise FileNotFoundError(
            f"OSM layer file not found: {osm_file}"
        )

    file_statistics = osm_file.stat()

    return {
        "path": str(
            osm_file.resolve()
        ),
        "size": file_statistics.st_size,
        "mtime_ns": file_statistics.st_mtime_ns,
    }


def spatial_signature(
    osm_file: Path,
    config: TileFeatureConfig,
) -> str:
    """Build a hash representing the spatial dependencies."""

    payload = {
        "osm": osm_signature(
            osm_file
        ),
        "target_crs": config.target_crs,
        "aoi_buffer_m": config.aoi_buffer_m,
        "feature_schema_version": (
            config.feature_schema_version
        ),
    }

    encoded = json.dumps(
        payload,
        sort_keys=True,
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


# ============================================================
# CANDIDATE INPUT
# ============================================================

def candidate_input(
    path: Path,
    division: str,
) -> pl.DataFrame:
    """
    Read candidate-pair keys and coordinates required for
    spatial processing.
    """

    schema = (
        pl.scan_parquet(path)
        .collect_schema()
        .names()
    )

    aliases = {
        "meter_latitude": [
            "meter_latitude",
            "LATITUDE",
        ],
        "meter_longitude": [
            "meter_longitude",
            "LONGITUDE",
        ],
        "transformer_latitude": [
            "transformer_latitude",
            "Location_Y",
        ],
        "transformer_longitude": [
            "transformer_longitude",
            "Location_X",
        ],
    }

    expressions = [
        pl.col(METER).cast(
            pl.String
        ),
        pl.col(CANDIDATE).cast(
            pl.String
        ),
    ]

    if "division" in schema:
        expressions.append(
            pl.col("division").cast(
                pl.String
            )
        )

    elif "Division" in schema:
        expressions.append(
            pl.col("Division")
            .cast(pl.String)
            .alias("division")
        )

    else:
        expressions.append(
            pl.lit(division)
            .cast(pl.String)
            .alias("division")
        )

    for canonical, possible_names in aliases.items():
        actual_name = next(
            (
                name
                for name in possible_names
                if name in schema
            ),
            None,
        )

        if actual_name is None:
            raise ValueError(
                f"{path.name} is missing a coordinate "
                f"column for {canonical}."
            )

        expressions.append(
            pl.col(actual_name)
            .cast(pl.Float64)
            .alias(canonical)
        )

    dataframe = (
        pl.scan_parquet(path)
        .select(expressions)
        .collect()
    )

    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    unique_count = (
        dataframe
        .unique(
            subset=KEY
        )
        .height
    )

    if unique_count != dataframe.height:
        raise ValueError(
            f"[{division}] duplicate meter-transformer "
            "pairs found in candidate features."
        )

    # --------------------------------------------------------
    # COORDINATE CHECK
    # --------------------------------------------------------

    invalid_coordinates = (
        dataframe
        .filter(
            pl.any_horizontal(
                [
                    pl.col(column).is_null()
                    for column in COORD_COLUMNS
                ]
            )
        )
        .height
    )

    if invalid_coordinates:
        raise ValueError(
            f"[{division}] {invalid_coordinates:,} candidate "
            "rows have missing spatial coordinates."
        )

    return dataframe


# ============================================================
# HISTORY INPUT
# ============================================================

def normalize_changes(
    changes,
) -> pl.DataFrame:
    """
    Convert candidate-feature history changes into a standard
    Polars DataFrame.
    """

    empty_schema = {
        METER: pl.String,
        CANDIDATE: pl.String,
        "division": pl.String,
        "tile_action": pl.String,
    }

    if changes is None:
        return pl.DataFrame(
            schema=empty_schema
        )

    if isinstance(
        changes,
        pd.DataFrame,
    ):
        if changes.empty:
            return pl.DataFrame(
                schema=empty_schema
            )

        changes = pl.from_pandas(
            changes
        )

    elif not isinstance(
        changes,
        pl.DataFrame,
    ):
        raise TypeError(
            "candidate_feature_changes must be "
            "a Pandas or Polars DataFrame."
        )

    required = {
        METER,
        CANDIDATE,
        "tile_action",
    }

    missing = (
        required
        - set(changes.columns)
    )

    if missing:
        return pl.DataFrame(
            schema=empty_schema
        )

    expressions = [
        pl.col(METER).cast(
            pl.String
        ),
        pl.col(CANDIDATE).cast(
            pl.String
        ),
        pl.col("tile_action").cast(
            pl.String
        ),
    ]

    if "division" in changes.columns:
        expressions.append(
            pl.col("division").cast(
                pl.String
            )
        )

    else:
        expressions.append(
            pl.lit(None)
            .cast(pl.String)
            .alias("division")
        )

    return (
        changes
        .select(expressions)
        .unique()
    )


# ============================================================
# OSM LOADING
# ============================================================

def load_osm_layers(
    osm_file: Path,
    config: TileFeatureConfig,
) -> dict[str, gpd.GeoDataFrame]:
    """Load and project the OSM reference layers."""

    print(
        f"\nLoading OSM layers from:\n{osm_file}"
    )

    with osm_file.open(
        "rb"
    ) as file:
        layers = pickle.load(file)

    expected_layers = [
        "buildings",
        "roads",
        "waterways",
        "railways",
        "barriers",
        "landuse",
    ]

    output = {}

    for layer_name in expected_layers:
        layer = layers.get(
            layer_name
        )

        if (
            layer is None
            or layer.empty
        ):
            output[layer_name] = (
                gpd.GeoDataFrame(
                    geometry=[],
                    crs=config.target_crs,
                )
            )

            continue

        if layer.crs is None:
            raise ValueError(
                f"OSM layer {layer_name!r} has no CRS."
            )

        if str(layer.crs) != config.target_crs:
            layer = layer.to_crs(
                config.target_crs
            )

        # Build the spatial index once.
        _ = layer.sindex

        output[layer_name] = layer

        print(
            f"  {layer_name}: {len(layer):,}"
        )

    return output


# ============================================================
# GEOSPATIAL HELPERS
# ============================================================

def dedupe_pandas(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure one row per meter-transformer pair."""

    return (
        dataframe
        .drop_duplicates(
            subset=KEY,
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )


def clip_layer(
    layer: gpd.GeoDataFrame,
    area_of_interest,
) -> gpd.GeoDataFrame:
    """
    Use a spatial index to select features intersecting the
    processing area.
    """

    if (
        layer is None
        or layer.empty
    ):
        return layer

    indexes = list(
        layer.sindex.intersection(
            area_of_interest.bounds
        )
    )

    if not indexes:
        return layer.iloc[
            0:0
        ].copy()

    subset = layer.iloc[
        indexes
    ].copy()

    return subset.loc[
        subset.intersects(
            area_of_interest
        )
    ].copy()


def make_lines(
    dataframe: pd.DataFrame,
    config: TileFeatureConfig,
) -> gpd.GeoDataFrame:
    """Create projected meter-to-transformer line geometries."""

    meter_points = (
        gpd.GeoSeries(
            gpd.points_from_xy(
                dataframe["meter_longitude"],
                dataframe["meter_latitude"],
            ),
            crs=config.wgs84_crs,
        )
        .to_crs(
            config.target_crs
        )
    )

    transformer_points = (
        gpd.GeoSeries(
            gpd.points_from_xy(
                dataframe[
                    "transformer_longitude"
                ],
                dataframe[
                    "transformer_latitude"
                ],
            ),
            crs=config.wgs84_crs,
        )
        .to_crs(
            config.target_crs
        )
    )

    output = dataframe.copy()

    output["geometry"] = [
        LineString(
            [
                meter_point,
                transformer_point,
            ]
        )
        for (
            meter_point,
            transformer_point,
        ) in zip(
            meter_points.array,
            transformer_points.array,
        )
    ]

    lines = gpd.GeoDataFrame(
        output,
        geometry="geometry",
        crs=config.target_crs,
    )

    lines["straight_line_length_ft"] = (
        lines.geometry.length
        * 3.28084
    )

    return lines


# ============================================================
# CROSSING FEATURES
# ============================================================

def add_crossings(
    lines: gpd.GeoDataFrame,
    layer: gpd.GeoDataFrame,
    name: str,
) -> gpd.GeoDataFrame:
    """Add crossing counts and flags for one OSM layer."""

    count_column = (
        f"{name}_crossing_count"
    )

    flag_column = (
        f"{name}_crossing_flag"
    )

    if (
        layer is None
        or layer.empty
    ):
        lines[count_column] = 0
        lines[flag_column] = 0

        return lines

    hits = gpd.sjoin(
        lines,
        layer[
            ["geometry"]
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
        .groupby(KEY)
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
        validate="one_to_one",
    )

    lines[count_column] = (
        lines[count_column]
        .fillna(0)
        .astype(np.int32)
    )

    lines[flag_column] = (
        lines[count_column] > 0
    ).astype(np.int8)

    return lines


# ============================================================
# BUILDING FEATURES
# ============================================================

def add_building_features(
    lines: gpd.GeoDataFrame,
    buildings: gpd.GeoDataFrame | None,
    config: TileFeatureConfig,
) -> gpd.GeoDataFrame:
    """Add building intersection counts and overlap lengths."""

    if (
        buildings is None
        or buildings.empty
    ):
        lines["building_count_between"] = 0
        lines["building_between_flag"] = 0
        lines["blocked_length_ft"] = 0.0
        lines["pct_line_blocked_by_building"] = 0.0

        return lines

    buildings = buildings.loc[
        buildings.geometry.geom_type.isin(
            [
                "Polygon",
                "MultiPolygon",
            ]
        )
    ].copy()

    if buildings.empty:
        return add_building_features(
            lines,
            None,
            config,
        )

    buildings["_building_id"] = (
        buildings.index.astype(str)
    )

    hits = gpd.sjoin(
        lines,
        buildings[
            [
                "_building_id",
                "geometry",
            ]
        ],
        how="left",
        predicate="intersects",
    )

    hits = hits.dropna(
        subset=[
            "_building_id"
        ]
    ).copy()

    if hits.empty:
        return add_building_features(
            lines,
            None,
            config,
        )

    if (
        len(hits)
        > config.max_building_hits_per_batch
    ):
        raise RuntimeError(
            "Building intersection count exceeded the "
            "safety threshold: "
            f"{len(hits):,} > "
            f"{config.max_building_hits_per_batch:,}"
        )

    building_geometry_lookup = (
        buildings[
            [
                "_building_id",
                "geometry",
            ]
        ]
        .rename(
            columns={
                "geometry": "_building_geometry",
            }
        )
    )

    hits = hits.merge(
        building_geometry_lookup,
        on="_building_id",
        how="left",
        validate="many_to_one",
    )

    blocked_lengths = []

    for (
        line_geometry,
        building_geometry,
    ) in zip(
        hits.geometry,
        hits["_building_geometry"],
    ):
        try:
            if (
                line_geometry is None
                or building_geometry is None
            ):
                blocked_lengths.append(
                    0.0
                )

                continue

            intersection = (
                line_geometry.intersection(
                    building_geometry
                )
            )

            blocked_lengths.append(
                intersection.length
                * 3.28084
            )

        except Exception:
            blocked_lengths.append(
                0.0
            )

    hits["blocked_length_ft"] = (
        blocked_lengths
    )

    building_count = (
        hits
        .groupby(KEY)
        .size()
        .rename(
            "building_count_between"
        )
        .reset_index()
    )

    blocked_length = (
        hits
        .groupby(KEY)[
            "blocked_length_ft"
        ]
        .sum()
        .reset_index()
    )

    lines = lines.merge(
        building_count,
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    lines = lines.merge(
        blocked_length,
        on=KEY,
        how="left",
        validate="one_to_one",
    )

    lines["building_count_between"] = (
        lines["building_count_between"]
        .fillna(0)
        .astype(np.int32)
    )

    lines["blocked_length_ft"] = (
        lines["blocked_length_ft"]
        .fillna(0.0)
    )

    lines["building_between_flag"] = (
        lines["building_count_between"] > 0
    ).astype(np.int8)

    lines["pct_line_blocked_by_building"] = (
        lines["blocked_length_ft"]
        / lines["straight_line_length_ft"].replace(
            0,
            np.nan,
        )
    ).fillna(0.0)

    return lines

# ============================================================
# ROAD SPLIT
# ============================================================

def split_roads(
    roads: gpd.GeoDataFrame,
) -> tuple[
    gpd.GeoDataFrame,
    gpd.GeoDataFrame,
]:
    """Split roads into major-road and local-road subsets."""

    if (
        roads is None
        or roads.empty
    ):
        empty = gpd.GeoDataFrame(
            geometry=[],
            crs=(
                roads.crs
                if roads is not None
                else None
            ),
        )

        return (
            empty,
            empty,
        )

    classification_column = next(
        (
            column
            for column in [
                "fclass",
                "highway",
                "class",
            ]
            if column in roads.columns
        ),
        None,
    )

    if classification_column is None:
        empty = roads.iloc[
            0:0
        ].copy()

        return (
            empty,
            empty,
        )

    road_class = (
        roads[classification_column]
        .astype(str)
        .str.lower()
    )

    major_road_classes = {
        "motorway",
        "trunk",
        "primary",
        "secondary",
        "tertiary",
    }

    local_road_classes = {
        "residential",
        "service",
        "unclassified",
        "living_street",
    }

    major_roads = roads.loc[
        road_class.isin(
            major_road_classes
        )
    ].copy()

    local_roads = roads.loc[
        road_class.isin(
            local_road_classes
        )
    ].copy()

    return (
        major_roads,
        local_roads,
    )


# ============================================================
# PROCESS ONE SPATIAL BATCH
# ============================================================

def process_spatial_batch(
    batch: pd.DataFrame,
    osm_layers: dict[str, gpd.GeoDataFrame],
    config: TileFeatureConfig,
) -> pd.DataFrame:
    """Calculate spatial features for one candidate-pair batch."""

    batch = dedupe_pandas(
        batch
    )

    if batch.empty:
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    lines = make_lines(
        batch,
        config,
    )

    # --------------------------------------------------------
    # AREA OF INTEREST
    # --------------------------------------------------------

    min_x, min_y, max_x, max_y = (
        lines.total_bounds
    )

    area_of_interest = box(
        min_x,
        min_y,
        max_x,
        max_y,
    ).buffer(
        config.aoi_buffer_m
    )

    # --------------------------------------------------------
    # CLIP STATIC OSM DATA
    # --------------------------------------------------------

    buildings = clip_layer(
        osm_layers["buildings"],
        area_of_interest,
    )

    roads = clip_layer(
        osm_layers["roads"],
        area_of_interest,
    )

    waterways = clip_layer(
        osm_layers["waterways"],
        area_of_interest,
    )

    railways = clip_layer(
        osm_layers["railways"],
        area_of_interest,
    )

    barriers = clip_layer(
        osm_layers["barriers"],
        area_of_interest,
    )

    landuse = clip_layer(
        osm_layers["landuse"],
        area_of_interest,
    )

    major_roads, local_roads = split_roads(
        roads
    )

    # --------------------------------------------------------
    # BUILDING FEATURES
    # --------------------------------------------------------

    lines = add_building_features(
        lines,
        buildings,
        config,
    )

    # --------------------------------------------------------
    # CROSSING FEATURES
    # --------------------------------------------------------

    crossing_layers = [
        (
            roads,
            "street",
        ),
        (
            major_roads,
            "major_road",
        ),
        (
            local_roads,
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
    ]

    for layer, feature_name in crossing_layers:
        lines = add_crossings(
            lines,
            layer,
            feature_name,
        )

    # --------------------------------------------------------
    # OBSTRUCTION SCORE
    # --------------------------------------------------------

    other_street_crossings = (
        lines["street_crossing_count"]
        - lines["major_road_crossing_count"]
        - lines["local_road_crossing_count"]
    ).clip(
        lower=0
    )

    lines["spatial_obstruction_score"] = (
        lines[
            "pct_line_blocked_by_building"
        ] * 3
        + lines[
            "building_count_between"
        ] * 3
        + lines[
            "waterway_crossing_count"
        ] * 5
        + lines[
            "railway_crossing_count"
        ] * 5
        + lines[
            "major_road_crossing_count"
        ]
        + lines[
            "local_road_crossing_count"
        ] * 0.5
        + other_street_crossings * 0.1
        + lines[
            "barrier_crossing_count"
        ]
    )

    result = pd.DataFrame(
        lines.drop(
            columns=[
                "geometry",
            ],
            errors="ignore",
        )
    )

    return (
        result[
            OUTPUT_COLUMNS
        ]
        .drop_duplicates(
            subset=KEY,
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# DTYPE OPTIMIZATION
# ============================================================

def optimize_output(
    dataframe: pl.DataFrame,
) -> pl.DataFrame:
    """Apply compact output types to spatial feature columns."""

    expressions = [
        pl.col(METER).cast(
            pl.String
        ),
        pl.col(CANDIDATE).cast(
            pl.String
        ),
        pl.col("division").cast(
            pl.String
        ),
    ]

    expressions.extend(
        [
            pl.col(column).cast(
                pl.Float32
            )
            for column in FLOAT_COLUMNS
        ]
    )

    expressions.extend(
        [
            pl.col(column).cast(
                pl.Int32
            )
            for column in COUNT_COLUMNS
        ]
    )

    expressions.extend(
        [
            pl.col(column).cast(
                pl.Int8
            )
            for column in FLAG_COLUMNS
        ]
    )

    return dataframe.select(
        expressions
    )


# ============================================================
# IN-MEMORY TILE ASSIGNMENT
# ============================================================

def assign_tile_keys(
    dataframe: pl.DataFrame,
    config: TileFeatureConfig,
) -> pl.DataFrame:
    """Assign processing tiles using candidate-pair midpoints."""

    projection = Transformer.from_crs(
        config.wgs84_crs,
        config.target_crs,
        always_xy=True,
    )

    meter_x, meter_y = projection.transform(
        dataframe[
            "meter_longitude"
        ].to_numpy(),
        dataframe[
            "meter_latitude"
        ].to_numpy(),
    )

    transformer_x, transformer_y = (
        projection.transform(
            dataframe[
                "transformer_longitude"
            ].to_numpy(),
            dataframe[
                "transformer_latitude"
            ].to_numpy(),
        )
    )

    midpoint_x = (
        meter_x
        + transformer_x
    ) / 2

    midpoint_y = (
        meter_y
        + transformer_y
    ) / 2

    tile_x = np.floor(
        midpoint_x
        / config.grid_size_m
    ).astype(
        np.int32
    )

    tile_y = np.floor(
        midpoint_y
        / config.grid_size_m
    ).astype(
        np.int32
    )

    return (
        dataframe
        .with_columns(
            [
                pl.Series(
                    "_tile_x",
                    tile_x,
                ),
                pl.Series(
                    "_tile_y",
                    tile_y,
                ),
            ]
        )
        .sort(
            [
                "_tile_x",
                "_tile_y",
                METER,
                CANDIDATE,
            ]
        )
    )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def candidate_file_signature(
    path: Path,
) -> dict:
    """Return identifying information for a candidate-feature file."""

    file_statistics = path.stat()

    return {
        "path": str(
            path.resolve()
        ),
        "size": file_statistics.st_size,
        "mtime_ns": file_statistics.st_mtime_ns,
    }


def prepare_work_directory(
    division: str,
    candidate_file: Path,
    work_count: int,
    spatial_signature_value: str,
    config: TileFeatureConfig,
) -> Path:
    """Create or validate the resumable work directory."""

    division_directory = (
        WORK_DIR
        / division
    )

    manifest_file = (
        division_directory
        / "_manifest.json"
    )

    manifest = {
        "division": division,
        "candidate_file": (
            candidate_file_signature(
                candidate_file
            )
        ),
        "work_rows": work_count,
        "spatial_signature": (
            spatial_signature_value
        ),
        "grid_size_m": config.grid_size_m,
        "aoi_buffer_m": config.aoi_buffer_m,
        "max_rows_per_batch": (
            config.max_rows_per_batch
        ),
    }

    existing_manifest = read_json(
        manifest_file
    )

    if (
        division_directory.exists()
        and existing_manifest
        and existing_manifest != manifest
    ):
        print(
            f"[{division}] workload changed. "
            "Discarding old incomplete checkpoints."
        )

        shutil.rmtree(
            division_directory
        )

    division_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json_atomic(
        manifest,
        manifest_file,
    )

    return division_directory


def checkpoint_valid(
    path: Path,
) -> bool:
    """Return True if a checkpoint contains all expected columns."""

    if not path.exists():
        return False

    try:
        schema = (
            pl.scan_parquet(path)
            .collect_schema()
            .names()
        )

        return all(
            column in schema
            for column in OUTPUT_COLUMNS
        )

    except Exception:
        return False


def write_batch(
    dataframe: pl.DataFrame,
    path: Path,
    compression: str,
) -> None:
    """Write and validate one compressed checkpoint batch."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        ".tmp.parquet"
    )

    try:
        dataframe.write_parquet(
            temporary,
            compression=compression,
        )

        # Basic integrity check.
        pl.scan_parquet(
            temporary
        ).select(
            pl.len()
        ).collect()

        for attempt in range(10):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.5)


    finally:
        temporary.unlink(
            missing_ok=True
        )


# ============================================================
# PROCESS REQUIRED PAIRS
# ============================================================

def process_work_set(
    work: pl.DataFrame,
    division: str,
    candidate_file: Path,
    osm_layers: dict[str, gpd.GeoDataFrame],
    spatial_signature_value: str,
    config: TileFeatureConfig,
) -> Path:
    """Process or resume spatial checkpoints for one division."""

    work_directory = prepare_work_directory(
        division=division,
        candidate_file=candidate_file,
        work_count=work.height,
        spatial_signature_value=(
            spatial_signature_value
        ),
        config=config,
    )

    if work.is_empty():
        return work_directory

    tiled = assign_tile_keys(
        work,
        config,
    )

    tile_counts = (
        tiled
        .group_by(
            [
                "_tile_x",
                "_tile_y",
            ],
            maintain_order=True,
        )
        .len()
    )

    offset = 0
    completed_batches = 0

    progress_file = (
        work_directory
        / "progress.json"
    )

    for tile_information in tile_counts.iter_rows(
        named=True
    ):
        tile_x = int(
            tile_information[
                "_tile_x"
            ]
        )

        tile_y = int(
            tile_information[
                "_tile_y"
            ]
        )

        tile_length = int(
            tile_information[
                "len"
            ]
        )

        tile_dataframe = tiled.slice(
            offset,
            tile_length,
        )

        offset += tile_length
        batch_number = 0

        for batch in tile_dataframe.iter_slices(
            config.max_rows_per_batch
        ):
            batch_number += 1

            checkpoint_file = (
                work_directory
                / (
                    f"batch_"
                    f"{tile_x}_"
                    f"{tile_y}_"
                    f"{batch_number:04d}"
                    f".parquet"
                )
            )

            # ------------------------------------------------
            # CRASH RECOVERY
            # ------------------------------------------------

            if checkpoint_valid(
                checkpoint_file
            ):
                completed_batches += 1

                continue

            checkpoint_file.unlink(
                missing_ok=True
            )

            batch_pandas = (
                batch
                .drop(
                    [
                        "_tile_x",
                        "_tile_y",
                    ]
                )
                .to_pandas()
            )

            spatial_features = (
                process_spatial_batch(
                    batch=batch_pandas,
                    osm_layers=osm_layers,
                    config=config,
                )
            )

            spatial_features_polars = (
                optimize_output(
                    pl.from_pandas(
                        spatial_features
                    )
                )
            )

            write_batch(
                dataframe=spatial_features_polars,
                path=checkpoint_file,
                compression=config.compression,
            )

            completed_batches += 1

            print(
                f"[{division}] "
                f"tile=({tile_x},{tile_y}) "
                f"batch={batch_number} "
                f"rows={len(spatial_features):,}"
            )

            if (
                completed_batches
                % config.checkpoint_every_batches
                == 0
            ):
                write_json_atomic(
                    {
                        "division": division,
                        "completed_batches": (
                            completed_batches
                        ),
                        "updated_at": datetime.now(
                            timezone.utc
                        ).isoformat(),
                    },
                    progress_file,
                )

    return work_directory


# ============================================================
# FINAL CACHE VALIDATION
# ============================================================

def validate_final_cache(
    path: Path,
    expected_keys: pl.DataFrame,
    division: str,
) -> None:
    """Validate uniqueness and candidate-pair coverage."""

    final = pl.scan_parquet(
        path
    )

    row_count = (
        final
        .select(
            pl.len()
        )
        .collect()
        .item()
    )

    unique_count = (
        final
        .select(
            pl.struct(KEY)
            .n_unique()
            .alias("n")
        )
        .collect()
        .item()
    )

    if row_count != unique_count:
        raise ValueError(
            f"[{division}] duplicate pairs found "
            "in the final tile cache."
        )

    final_keys = final.select(
        KEY
    )

    missing_count = (
        expected_keys
        .lazy()
        .join(
            final_keys,
            on=KEY,
            how="anti",
        )
        .select(
            pl.len()
        )
        .collect()
        .item()
    )

    extra_count = (
        final_keys
        .join(
            expected_keys.lazy(),
            on=KEY,
            how="anti",
        )
        .select(
            pl.len()
        )
        .collect()
        .item()
    )

    if (
        missing_count
        or extra_count
    ):
        raise ValueError(
            f"[{division}] final tile cache does not "
            "reconcile. "
            f"Missing={missing_count:,}, "
            f"Extra={extra_count:,}."
        )


# ============================================================
# BUILD FINAL CURRENT CACHE
# ============================================================

def assemble_current_cache(
    division: str,
    expected_keys: pl.DataFrame,
    compute_keys: pl.DataFrame,
    current_file: Path,
    work_directory: Path,
    config: TileFeatureConfig,
) -> None:
    """Combine reusable rows with new or recomputed checkpoints."""

    lazy_frames = []

    # --------------------------------------------------------
    # REUSE UNCHANGED EXISTING ROWS
    # --------------------------------------------------------

    if current_file.exists():
        existing = (
            pl.scan_parquet(
                current_file
            )
            .join(
                expected_keys.lazy(),
                on=KEY,
                how="semi",
            )
        )

        if not compute_keys.is_empty():
            existing = existing.join(
                compute_keys.lazy(),
                on=KEY,
                how="anti",
            )

        lazy_frames.append(
            existing.select(OUTPUT_COLUMNS)
        )

    # --------------------------------------------------------
    # NEW OR RECOMPUTED CHECKPOINTS
    # --------------------------------------------------------

    # Read all checkpoints through one scan and filter once.

    if not compute_keys.is_empty():

        checkpoint_files = sorted(
            work_directory.glob("batch_*.parquet")
        )

        if not checkpoint_files:
            raise RuntimeError(
                f"[{division}] no checkpoint batches were found."
            )

        computed = (
            pl.scan_parquet(
                [str(path) for path in checkpoint_files]
            )
            .select(OUTPUT_COLUMNS)
            .join(
                compute_keys.lazy(),
                on=KEY,
                how="semi",
            )
        )

        lazy_frames.append(computed)
    if not lazy_frames:
        raise RuntimeError(
            f"[{division}] no rows are available "
            "to build the current tile cache."
        )

    final = (
        pl.concat(
            lazy_frames,
            how="vertical_relaxed",
        )
        .unique(
            subset=KEY,
            keep="last",
        )
        .select(
            OUTPUT_COLUMNS
        )
    )

    current_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = current_file.with_suffix(
        ".tmp.parquet"
    )

    temporary.unlink(
        missing_ok=True
    )

    try:
        final.sink_parquet(
            temporary,
            compression=config.compression,
        )

        validate_final_cache(
            path=temporary,
            expected_keys=expected_keys,
            division=division,
        )

        os.replace(
            temporary,
            current_file,
        )

    finally:
        temporary.unlink(
            missing_ok=True
        )


# ============================================================
# STATE
# ============================================================

def update_state(
    state: dict,
    division: str,
    current_file: Path,
    expected_count: int,
    spatial_signature_value: str,
    candidate_file: Path,
) -> None:
    """Update the saved tile-cache state for one division."""

    state.setdefault(
        "divisions",
        {},
    )

    state["divisions"][division] = {
        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "rows": expected_count,
        "candidate_snapshot": (
            candidate_file.parent.name
        ),
        "candidate_file": str(
            candidate_file
        ),
        "tile_file": str(
            current_file
        ),
        "spatial_signature": (
            spatial_signature_value
        ),
    }

    write_json_atomic(
        state,
        STATE_FILE,
    )


# ============================================================
# MAIN STAGE
# ============================================================

def run_tile_feature_stage(
    config,
    candidate_feature_files: dict[str, Path],
    candidate_feature_changes=None,
    *,
    tile_config: TileFeatureConfig | None = None,
    osm_layer_file: Path | None = None,
) -> dict[str, Path]:
    """
    Maintain the current SmartLink spatial/tile feature cache.

    First run:
        Computes every candidate pair.

    Later runs:
        Reuses unchanged cached tile features.
        Computes new or missing pairs.
        Recomputes pairs flagged RECOMPUTE.
        Removes stale pairs.

    Returns
    -------
    dict[str, Path]
        Division -> current tile-feature parquet.
    """

    del config

    tile_config = (
        tile_config
        or DEFAULT_TILE_CONFIG
    )

    osm_file = (
        Path(osm_layer_file)
        if osm_layer_file is not None
        else DEFAULT_OSM_LAYER_FILE
    )

    CURRENT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    WORK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    changes = normalize_changes(
        candidate_feature_changes
    )

    current_spatial_signature = (
        spatial_signature(
            osm_file,
            tile_config,
        )
    )

    state = read_json(
        STATE_FILE
    )

    outputs: dict[str, Path] = {}

    osm_layers = None

    print(
        "\n" + "=" * 80
    )
    print(
        "SMARTLINK TILE / SPATIAL FEATURES"
    )
    print(
        "=" * 80
    )

    print(
        f"Grid size       : "
        f"{tile_config.grid_size_m:,.0f} m"
    )

    print(
        f"AOI buffer      : "
        f"{tile_config.aoi_buffer_m:,.0f} m"
    )

    print(
        f"Rows per batch  : "
        f"{tile_config.max_rows_per_batch:,}"
    )

    print(
        f"Compression     : "
        f"{tile_config.compression}"
    )

    for division in sorted(
        candidate_feature_files
    ):
        print(
            "\n" + "-" * 80
        )

        print(
            f"[{division}]"
        )

        candidate_file = Path(
            candidate_feature_files[
                division
            ]
        )

        current_file = (
            CURRENT_DIR
            / f"{division}.parquet"
        )

        expected = candidate_input(
            candidate_file,
            division,
        )

        expected_keys = (
            expected
            .select(KEY)
            .unique()
        )

        if current_file.exists():
            existing_keys = (
                pl.scan_parquet(
                    current_file
                )
                .select(KEY)
                .collect()
                .unique()
            )
        else:
            existing_keys = pl.DataFrame(
                schema={
                    METER: pl.String,
                    CANDIDATE: pl.String,
                }
            )

        missing_keys = (
            expected_keys.join(
                existing_keys,
                on=KEY,
                how="anti",
            )
        )

        stale_keys = (
            existing_keys.join(
                expected_keys,
                on=KEY,
                how="anti",
            )
        )

        division_changes = (
            changes.filter(
                (
                    pl.col("division")
                    == division
                )
                |
                pl.col("division").is_null()
            )
        )

        requested_compute = (
            division_changes
            .filter(
                pl.col(
                    "tile_action"
                ).is_in(
                    [
                        "COMPUTE",
                        "RECOMPUTE",
                    ]
                )
            )
            .select(KEY)
            .unique()
            .join(
                expected_keys,
                on=KEY,
                how="semi",
            )
        )

        division_state = (
            state
            .get(
                "divisions",
                {}
            )
            .get(
                division
            )
        )

        source_changed = False

        if (
            current_file.exists()
            and tile_config.invalidate_on_osm_change
        ):
            if division_state is None:
                source_changed = True

            elif (
                division_state.get(
                    "spatial_signature"
                )
                != current_spatial_signature
            ):
                source_changed = True

        if source_changed:
            print(
                "Spatial source/configuration changed. "
                "Recomputing this division."
            )

            compute_keys = expected_keys

        else:
            compute_keys = (
                pl.concat(
                    [
                        missing_keys,
                        requested_compute,
                    ],
                    how="vertical",
                )
                .unique()
            )

        if (
            compute_keys.is_empty()
            and stale_keys.is_empty()
            and current_file.exists()
        ):
            print(
                f"Current pairs   : "
                f"{expected_keys.height:,}"
            )

            print(
                "Compute         : 0"
            )

            print(
                "Recompute       : 0"
            )

            print(
                "Drop            : 0"
            )

            print(
                "Reusing current tile cache."
            )

            outputs[
                division
            ] = current_file

            continue

        work = (
            expected.join(
                compute_keys,
                on=KEY,
                how="semi",
            )
        )

        print(
            f"Current pairs   : "
            f"{expected_keys.height:,}"
        )

        print(
            f"Missing/new     : "
            f"{missing_keys.height:,}"
        )

        print(
            f"Requested update: "
            f"{requested_compute.height:,}"
        )

        print(
            f"Drop            : "
            f"{stale_keys.height:,}"
        )

        print(
            f"Spatial compute : "
            f"{work.height:,}"
        )

        if (
            not work.is_empty()
            and osm_layers is None
        ):
            osm_layers = (
                load_osm_layers(
                    osm_file,
                    tile_config,
                )
            )

        if not work.is_empty():
            work_directory = (
                process_work_set(
                    work=work,
                    division=division,
                    candidate_file=candidate_file,
                    osm_layers=osm_layers,
                    spatial_signature_value=current_spatial_signature,
                    config=tile_config,
                )
            )
        else:
            work_directory = (
                WORK_DIR
                / division
            )

            work_directory.mkdir(
                parents=True,
                exist_ok=True,
            )
            
        print(
        f"[{division}] Spatial batches finished. "
        "Assembling and validating the current cache...",
        flush=True,
        ) 

        assemble_current_cache(
            division=division,
            expected_keys=expected_keys,
            compute_keys=compute_keys,
            current_file=current_file,
            work_directory=work_directory,
            config=tile_config,
        )

        update_state(
            state=state,
            division=division,
            current_file=current_file,
            expected_count=expected_keys.height,
            spatial_signature_value=current_spatial_signature,
            candidate_file=candidate_file,
        )

        outputs[
            division
        ] = current_file

        print(
            f"[{division}] saved:"
        )

        print(
            current_file
        )

        if (
            tile_config.delete_work_files_on_success
            and work_directory.exists()
        ):
            shutil.rmtree(
                work_directory
            )

            print(
                f"[{division}] temporary "
                "checkpoint files removed."
            )

    print(
        "\nTile feature stage complete."
    )

    return outputs