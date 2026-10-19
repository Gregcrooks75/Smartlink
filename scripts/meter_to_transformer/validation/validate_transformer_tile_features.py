from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
KEY = [
    "meter_number",
    "candidate_transformer",
]
def pairset(
    df,
):
    return set(
        map(
            tuple,
            df[
                KEY
            ]
            .astype(str)
            .values,
        )
    )
def validate_transformer_tile_features(
    tile_feature_files: dict[str, Path],
    feature_files: dict[str, Path],
) -> None:
    required_tile_features = [
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
    ]
    for division in tile_feature_files:
        print(
            "\n"
            + "=" * 80
        )
        print(
            f"VALIDATING TILE FEATURES: "
            f"{division}"
        )
        before = pd.read_parquet(
            feature_files[
                division
            ]
        )
        after = pd.read_parquet(
            tile_feature_files[
                division
            ]
        )
        missing = [
            column
            for column
            in required_tile_features
            if column
            not in after.columns
        ]
        if missing:
            raise AssertionError(
                f"{division}: missing tile "
                f"features {missing}"
            )
        before_valid = (
            before
            .dropna(
                subset=[
                    "LATITUDE",
                    "LONGITUDE",
                    "Location_X",
                    "Location_Y",
                ]
            )
        )
        if (
            pairset(
                before_valid
            )
            !=
            pairset(
                after
            )
        ):
            raise AssertionError(
                f"{division}: candidate rows "
                "changed during tile processing."
            )
        if (
            after
            .duplicated(
                KEY
            )
            .any()
        ):
            raise AssertionError(
                f"{division}: duplicates "
                "after tile processing."
            )
        if (
            after[
                "straight_line_length_ft"
            ]
            < 0
        ).any():
            raise AssertionError(
                f"{division}: negative "
                "line length."
            )
        if (
            after[
                "spatial_obstruction_score"
            ]
            < 0
        ).any():
            raise AssertionError(
                f"{division}: negative "
                "obstruction score."
            )
        numeric = (
            after
            .select_dtypes(
                include="number"
            )
        )
        if (
            np.isinf(
                numeric
            )
            .any()
            .any()
        ):
            raise AssertionError(
                f"{division}: infinity "
                "in tile features."
            )
        print(
            f"Rows: {len(after):,}"
        )
    print(
        "\nTILE FEATURE VALIDATION PASSED"
    )