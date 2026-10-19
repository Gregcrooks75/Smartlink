"""Run reusable tile features over feature partitions."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import polars as pl

from mtx.config import (
    METER_PAIR_FEATURE_DIRECTORY,
    SPATIAL_FEATURE_DIRECTORY,
    OSM_LAYER_FILE,
    METER_PAIR_TILE_COLUMNS,
)
from mtx.features.tile_features import (
    build_tile_features,
)


# ============================================================
# EXPECTED OUTPUT FEATURES
# ============================================================

TILE_FEATURE_COLUMNS = {
    "straight_line_length_ft",
    "building_count_between",
    "building_between_flag",
    "blocked_length_ft",
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
}


# ============================================================
# OUTPUT PATH
# ============================================================

def output_path(
    feature_file: Path,
) -> Path:
    """Mirror Stage-3 structure inside Stage 4."""

    feature_file = Path(
        feature_file
    ).resolve()

    relative = feature_file.relative_to(
        METER_PAIR_FEATURE_DIRECTORY.resolve()
    )

    output = (
        SPATIAL_FEATURE_DIRECTORY
        / relative
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output


# ============================================================
# PARQUET HELPERS
# ============================================================

def row_count(
    path: Path,
) -> int:
    """Return Parquet row count."""

    return (
        pl.scan_parquet(path)
        .select(pl.len())
        .collect()
        .item()
    )


def valid_output(
    input_file: Path,
    output_file: Path,
) -> bool:
    """
    Return True when an existing Stage-4
    partition is safe to reuse.
    """

    if (
        not output_file.is_file()
        or output_file.stat().st_size == 0
    ):
        return False

    try:
        schema = set(
            pl.read_parquet_schema(
                output_file
            )
        )

        if not TILE_FEATURE_COLUMNS.issubset(
            schema
        ):
            return False

        return (
            row_count(input_file)
            == row_count(output_file)
        )

    except Exception:
        return False


# ============================================================
# OSM
# ============================================================

def load_osm_layers() -> dict:
    """Load OSM layers once for the stage."""

    if not OSM_LAYER_FILE.is_file():
        raise FileNotFoundError(
            "OSM layer file not found: "
            f"{OSM_LAYER_FILE}"
        )

    with open(
        OSM_LAYER_FILE,
        "rb",
    ) as file:
        layers = pickle.load(file)

    required_layers = {
        "buildings",
        "roads",
        "waterways",
        "railways",
        "barriers",
        "landuse",
    }

    missing = (
        required_layers
        - set(layers)
    )

    if missing:
        raise ValueError(
            "OSM file is missing layers: "
            f"{sorted(missing)}"
        )

    # Build spatial indexes once.
    for layer in layers.values():
        if (
            layer is not None
            and not layer.empty
        ):
            _ = layer.sindex

    return layers


# ============================================================
# INPUT VALIDATION
# ============================================================

def validate_input(
    df: pd.DataFrame,
    feature_file: Path,
) -> None:
    """Validate columns required by tile features."""

    required = {
        METER_PAIR_TILE_COLUMNS["left_id"],
        METER_PAIR_TILE_COLUMNS["right_id"],
        METER_PAIR_TILE_COLUMNS["left_lon"],
        METER_PAIR_TILE_COLUMNS["left_lat"],
        METER_PAIR_TILE_COLUMNS["right_lon"],
        METER_PAIR_TILE_COLUMNS["right_lat"],
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{feature_file.name} is missing "
            f"required columns: {sorted(missing)}"
        )

    coordinate_columns = [
        METER_PAIR_TILE_COLUMNS["left_lon"],
        METER_PAIR_TILE_COLUMNS["left_lat"],
        METER_PAIR_TILE_COLUMNS["right_lon"],
        METER_PAIR_TILE_COLUMNS["right_lat"],
    ]

    missing_coordinates = (
        df[coordinate_columns]
        .isna()
        .any(axis=1)
    )

    if missing_coordinates.any():
        raise ValueError(
            f"{feature_file.name} has "
            f"{missing_coordinates.sum():,} rows "
            "with missing coordinates."
        )


# ============================================================
# STAGE
# ============================================================

def run_tile_feature_stage(
    feature_files: dict[str, list[Path]],
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """
    Add spatial/tile features to every Stage-3
    feature partition.

    One input partition produces one output partition.
    """

    osm_layers = load_osm_layers()
    spatial_files = {}

    for division, files in feature_files.items():
        spatial_files[division] = []

        print(
            f"\n[{division}] "
            f"{len(files):,} partitions",
            flush=True,
        )

        for feature_file in files:
            feature_file = Path(
                feature_file
            )

            if not feature_file.is_file():
                raise FileNotFoundError(
                    "Feature partition not found: "
                    f"{feature_file}"
                )

            target = output_path(
                feature_file
            )

            spatial_files[division].append(
                target
            )

            # =================================================
            # REUSE
            # =================================================

            if (
                not overwrite
                and valid_output(
                    feature_file,
                    target,
                )
            ):
                print(
                    f"[{division}] Reusing "
                    f"{target.name}",
                    flush=True,
                )

                continue

            # =================================================
            # READ
            # =================================================

            print(
                f"[{division}] Processing "
                f"{feature_file.name}",
                flush=True,
            )

            df = pd.read_parquet(
                feature_file
            )

            expected_rows = len(df)

            validate_input(
                df=df,
                feature_file=feature_file,
            )

            # =================================================
            # FEATURES
            # =================================================

            result = build_tile_features(
                df=df,
                osm_layers=osm_layers,
                cols=METER_PAIR_TILE_COLUMNS,
            )

            # =================================================
            # VALIDATE RESULT
            # =================================================

            if len(result) != expected_rows:
                raise ValueError(
                    f"{feature_file.name}: "
                    f"expected {expected_rows:,} rows "
                    f"but got {len(result):,}."
                )

            missing_features = (
                TILE_FEATURE_COLUMNS
                - set(result.columns)
            )

            if missing_features:
                raise ValueError(
                    f"{feature_file.name} is missing "
                    "tile features: "
                    f"{sorted(missing_features)}"
                )

            # Ensure Stage-3 columns survived.
            missing_original_columns = (
                set(df.columns)
                - set(result.columns)
            )

            if missing_original_columns:
                raise ValueError(
                    f"{feature_file.name} lost "
                    "original columns: "
                    f"{sorted(missing_original_columns)}"
                )

            # =================================================
            # SAVE
            # =================================================

            target.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            result.to_parquet(
                target,
                index=False,
            )

            # Verify what was actually written.
            if not valid_output(
                feature_file,
                target,
            ):
                raise RuntimeError(
                    "Saved spatial partition failed "
                    f"validation: {target}"
                )

            print(
                f"[{division}] Saved "
                f"{len(result):,} rows -> "
                f"{target.name}",
                flush=True,
            )

    return spatial_files