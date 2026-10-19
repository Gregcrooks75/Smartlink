# meter_to_transformer/stages/run_tile_features.py

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

WORK_DIR = (
    TILE_CACHE_DIR
    / "_work"
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
    # GeoPandas and Shapely in one batch.
    max_rows_per_batch: int = 500

    # Batch result files are always checkpoints. This controls
    # how frequently progress.json is refreshed.
    checkpoint_every_batches: int = 1

    # Delete checkpoint batches after a division has been
    # successfully assembled and validated.
    delete_work_files_on_success: bool = True

    # Parquet compression.
    compression: str = "zstd"

    # Prevent an unexpectedly large building spatial join from
    # exhausting available memory.
    max_building_hits_per_batch: int = 50_000

    # Recompute spatial features if the OSM reference source
    # or tile configuration changes.
    invalidate_on_osm_change: bool = True

    # Increment manually if feature equations change.
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

    temp = path.with_suffix(
        ".tmp.json"
    )

    try:
        with temp.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                default=str,
            )

        os.replace(
            temp,
            path,
        )

    finally:
        temp.unlink(
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

    stat = osm_file.stat()

    return {
        "path": str(
            osm_file.resolve()
        ),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def spatial_signature(
    osm_file: Path,
    config: TileFeatureConfig,
) -> str:
    """Build a hash representing the spatial feature dependencies."""

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
    ).encode("utf-8")

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
    """Read the coordinates required for spatial processing."""

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

    for canonical, possibilities in aliases.items():

        actual = next(
            (
                name
                for name in possibilities
                if name in schema
            ),
            None,
        )

        if actual is None:
            raise ValueError(
                f"{path.name} is missing a coordinate "
                f"column for {canonical}."
            )

        expressions.append(
            pl.col(actual)
            .cast(pl.Float64)
            .alias(canonical)
        )

    df = (
        pl.scan_parquet(path)
        .select(expressions)
        .collect()
    )

    # --------------------------------------------------------
    # DUPLICATE CHECK
    # --------------------------------------------------------

    unique_count = (
        df.unique(
            subset=KEY
        )
        .height
    )

    if unique_count != df.height:
        raise ValueError(
            f"[{division}] duplicate meter-transformer "
            "pairs found in candidate features."
        )

    # --------------------------------------------------------
    # COORDINATE CHECK
    # --------------------------------------------------------

    invalid_coordinates = (
        df.filter(
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

    return df


# ============================================================
# HISTORY INPUT
# ============================================================

def normalize_changes(
    changes,
) -> pl.DataFrame:
    """Convert candidate-feature changes to a standard Polars frame."""

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

    missing = required - set(
        changes.columns
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

    expected = [
        "buildings",
        "roads",
        "waterways",
        "railways",
        "barriers",
        "landuse",
    ]

    output = {}

    for name in expected:

        layer = layers.get(name)

        if (
            layer is None
            or layer.empty
        ):
            output[name] = (
                gpd.GeoDataFrame(
                    geometry=[],
                    crs=config.target_crs,
                )
            )

            continue

        if layer.crs is None:
            raise ValueError(
                f"OSM layer {name!r} has no CRS."
            )

        if str(layer.crs) != config.target_crs:
            layer = layer.to_crs(
                config.target_crs
            )

        # Build the spatial index once.
        _ = layer.sindex

        output[name] = layer

        print(
            f"  {name}: {len(layer):,}"
        )

    return output


# ============================================================
# GEOSPATIAL HELPERS
# ============================================================

def dedupe_pandas(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Ensure one row per meter-transformer pair."""

    return (
        df
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
    aoi,
) -> gpd.GeoDataFrame:
    """Use the spatial index to select features intersecting the AOI."""

    if (
        layer is None
        or layer.empty
    ):
        return layer

    indexes = list(
        layer.sindex.intersection(
            aoi.bounds
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
        subset.intersects(aoi)
    ].copy()


def make_lines(
    df: pd.DataFrame,
    config: TileFeatureConfig,
) -> gpd.GeoDataFrame:
    """Create projected meter-to-transformer line geometries."""

    meter_points = (
        gpd.GeoSeries(
            gpd.points_from_xy(
                df["meter_longitude"],
                df["meter_latitude"],
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
                df["transformer_longitude"],
                df["transformer_latitude"],
            ),
            crs=config.wgs84_crs,
        )
        .to_crs(
            config.target_crs
        )
    )

    out = df.copy()

    out["geometry"] = [
        LineString(
            [
                meter,
                transformer,
            ]
        )
        for meter, transformer in zip(
            meter_points.array,
            transformer_points.array,
        )
    ]

    lines = gpd.GeoDataFrame(
        out,
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
        .rename(count_column)
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

    building_geometry = (
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
        building_geometry,
        on="_building_id",
        how="left",
        validate="many_to_one",
    )

    blocked_lengths = []

    for (
        line_geometry,
        building_geometry_value,
    ) in zip(
        hits.geometry,
        hits["_building_geometry"],
    ):
        try:
            if (
                line_geometry is None
                or building_geometry_value is None
            ):
                blocked_lengths.append(
                    0.0
                )

                continue

            intersection = (
                line_geometry.intersection(
                    building_geometry_value
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
        roads.loc[
            road_class.isin(major)
        ].copy(),
        roads.loc[
            road_class.isin(local)
        ].copy(),
    )


# ============================================================
# PROCESS ONE SPATIAL BATCH
# ============================================================

def process_spatial_batch(
    batch: pd.DataFrame,
    osm_layers: dict,
    config: TileFeatureConfig,
) -> pd.DataFrame:
    """Calculate spatial features for one candidate-pair batch."""

    batch = dedupe_pandas(
        batch
    )

    lines = make_lines(
        batch,
        config,
    )

    # --------------------------------------------------------
    # AOI
    # --------------------------------------------------------

    min_x, min_y, max_x, max_y = (
        lines.total_bounds
    )

    aoi = box(
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
        aoi,
    )

    roads = clip_layer(
        osm_layers["roads"],
        aoi,
    )

    waterways = clip_layer(
        osm_layers["waterways"],
        aoi,
    )

    railways = clip_layer(
        osm_layers["railways"],
        aoi,
    )

    barriers = clip_layer(
        osm_layers["barriers"],
        aoi,
    )

    landuse = clip_layer(
        osm_layers["landuse"],
        aoi,
    )

    major_roads, local_roads = split_roads(
        roads
    )

    # --------------------------------------------------------
    # BUILDINGS
    # --------------------------------------------------------

    lines = add_building_features(
        lines,
        buildings,
        config,
    )

    # --------------------------------------------------------
    # CROSSINGS
    # --------------------------------------------------------

    for layer, feature_name in [
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
    ]:
        lines = add_crossings(
            lines,
            layer,
            feature_name,
        )

    # --------------------------------------------------------
    # OBSTRUCTION SCORE
    # --------------------------------------------------------

    other_streets = (
        lines["street_crossing_count"]
        - lines["major_road_crossing_count"]
        - lines["local_road_crossing_count"]
    ).clip(
        lower=0
    )

    lines["spatial_obstruction_score"] = (
        lines["pct_line_blocked_by_building"] * 3
        + lines["building_count_between"] * 3
        + lines["waterway_crossing_count"] * 5
        + lines["railway_crossing_count"] * 5
        + lines["major_road_crossing_count"]
        + lines["local_road_crossing_count"] * 0.5
        + other_streets * 0.1
        + lines["barrier_crossing_count"]
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
    df: pl.DataFrame,
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

    return df.select(
        expressions
    )


# ============================================================
# IN-MEMORY TILE ASSIGNMENT
# ============================================================

def assign_tile_keys(
    df: pl.DataFrame,
    config: TileFeatureConfig,
) -> pl.DataFrame:
    """Assign processing tiles using candidate-pair midpoints."""

    projection = Transformer.from_crs(
        config.wgs84_crs,
        config.target_crs,
        always_xy=True,
    )

    meter_x, meter_y = projection.transform(
        df["meter_longitude"].to_numpy(),
        df["meter_latitude"].to_numpy(),
    )

    transformer_x, transformer_y = (
        projection.transform(
            df[
                "transformer_longitude"
            ].to_numpy(),
            df[
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
        df
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
    """Return identifying information for a candidate feature file."""

    stat = path.stat()

    return {
        "path": str(
            path.resolve()
        ),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }

def prepare_work_directory(
    division: str,
    candidate_file: Path,
    work_count: int,
    spatial_signature_value: str,
    config: TileFeatureConfig,
) -> Path:
    """Create or validate the resumable work directory."""

    division_dir = (
        WORK_DIR
        / division
    )

    manifest_file = (
        division_dir
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
        division_dir.exists()
        and existing_manifest
        and existing_manifest != manifest
    ):
        print(
            f"[{division}] workload changed. "
            "Discarding old incomplete checkpoints."
        )

        shutil.rmtree(
            division_dir
        )

    division_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_json_atomic(
        manifest,
        manifest_file,
    )

    return division_dir