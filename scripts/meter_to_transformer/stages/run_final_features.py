from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


# ======================================================
# PROJECT PATHS
# ======================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

FINAL_FEATURE_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "final_features"
)


# ======================================================
# UNIQUE PAIR KEY
# ======================================================

PAIR_COLUMNS = [
    "meter_number",
    "candidate_transformer",
]


# ======================================================
# VALIDATION
# ======================================================

def validate_feature_file(
    df: pd.DataFrame,
    division: str,
    source_name: str,
) -> None:
    """
    Validate that the feature dataset contains the
    required meter-to-transformer pair columns and
    exactly one row per pair.
    """

    # --------------------------------------------------
    # Required columns
    # --------------------------------------------------

    missing = (
        set(PAIR_COLUMNS)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"[{division}] "
            f"{source_name} is missing required columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------
    # Empty dataset
    # --------------------------------------------------

    if df.empty:
        raise ValueError(
            f"[{division}] "
            f"{source_name} is empty."
        )

    # --------------------------------------------------
    # Null pair keys
    # --------------------------------------------------

    null_keys = (
        df[PAIR_COLUMNS]
        .isna()
        .any(axis=1)
    )

    if null_keys.any():
        raise ValueError(
            f"[{division}] "
            f"{source_name} contains "
            f"{int(null_keys.sum()):,} rows "
            f"with null meter/candidate keys."
        )

    # --------------------------------------------------
    # Duplicate meter-transformer pairs
    # --------------------------------------------------

    duplicates = (
        df.duplicated(
            subset=PAIR_COLUMNS,
            keep=False,
        )
    )

    if duplicates.any():
        duplicate_count = int(
            duplicates.sum()
        )

        examples = (
            df.loc[
                duplicates,
                PAIR_COLUMNS,
            ]
            .head(5)
            .to_dict("records")
        )

        raise ValueError(
            f"[{division}] "
            f"{source_name} contains "
            f"{duplicate_count:,} rows belonging "
            f"to duplicate meter-transformer pairs. "
            f"Examples: {examples}"
        )


# ======================================================
# FINAL FEATURE ASSEMBLY
# ======================================================

def run_final_feature_stage(
    candidate_feature_files: dict[str, Path],
    tile_feature_files: dict[str, Path],
    snapshot_date: date | None = None,
) -> dict[str, Path]:
    """
    Combine regular candidate features and tile/spatial
    features into one final modeling feature set per
    division.

    The stage requires both inputs to contain exactly
    the same unique:

        meter_number + candidate_transformer

    pairs.

    Final files are written to:

        data/cache/final_features/YYYY-MM-DD/

    If a final feature file already exists for a division,
    the existing cached file is reused.
    """

    if snapshot_date is None:
        snapshot_date = date.today()

    snapshot_string = (
        snapshot_date.isoformat()
    )

    # ==================================================
    # OUTPUT DIRECTORY
    # ==================================================

    output_dir = (
        FINAL_FEATURE_CACHE_DIR
        / snapshot_string
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ==================================================
    # VALIDATE DIVISION POPULATION
    # ==================================================

    candidate_divisions = set(
        candidate_feature_files.keys()
    )

    tile_divisions = set(
        tile_feature_files.keys()
    )

    missing_tile_divisions = (
        candidate_divisions
        - tile_divisions
    )

    missing_candidate_divisions = (
        tile_divisions
        - candidate_divisions
    )

    if missing_tile_divisions:
        raise ValueError(
            "Tile features are missing divisions: "
            f"{sorted(missing_tile_divisions)}"
        )

    if missing_candidate_divisions:
        raise ValueError(
            "Candidate features are missing divisions: "
            f"{sorted(missing_candidate_divisions)}"
        )

    # ==================================================
    # ASSEMBLE EACH DIVISION
    # ==================================================

    outputs: dict[str, Path] = {}

    for division in sorted(
        candidate_divisions
    ):

        print(
            "\n"
            f"[{division}] "
            "Checking final features..."
        )

        # --------------------------------------------------
        # Define output path before processing
        # --------------------------------------------------

        output_path = (
            output_dir
            / f"{division}.parquet"
        )

        # --------------------------------------------------
        # Reuse existing final feature file
        # --------------------------------------------------

        if output_path.exists():
            print(
                f"[{division}] "
                "Final features already exist. "
                "Reusing cached file."
            )

            print(
                f"[{division}] "
                f"Cached file: {output_path}"
            )

            outputs[division] = output_path

            continue

        print(
            f"[{division}] "
            "No cached final feature file found. "
            "Assembling final features..."
        )

        candidate_path = Path(
            candidate_feature_files[
                division
            ]
        )

        tile_path = Path(
            tile_feature_files[
                division
            ]
        )

        # --------------------------------------------------
        # Confirm source files exist
        # --------------------------------------------------

        if not candidate_path.exists():
            raise FileNotFoundError(
                f"[{division}] "
                f"Candidate feature file not found: "
                f"{candidate_path}"
            )

        if not tile_path.exists():
            raise FileNotFoundError(
                f"[{division}] "
                f"Tile feature file not found: "
                f"{tile_path}"
            )

        # --------------------------------------------------
        # Read feature files
        # --------------------------------------------------

        candidate_df = pd.read_parquet(
            candidate_path
        )

        tile_df = pd.read_parquet(
            tile_path
        )

        print(
            f"[{division}] "
            f"Candidate feature rows: "
            f"{len(candidate_df):,}"
        )

        print(
            f"[{division}] "
            f"Tile feature rows: "
            f"{len(tile_df):,}"
        )

        # --------------------------------------------------
        # Validate both files independently
        # --------------------------------------------------

        validate_feature_file(
            df=candidate_df,
            division=division,
            source_name="candidate features",
        )

        validate_feature_file(
            df=tile_df,
            division=division,
            source_name="tile features",
        )

        # ==================================================
        # REMOVE OVERLAPPING NON-KEY COLUMNS
        # ==================================================
        #
        # The candidate feature file is treated as the
        # primary dataset.
        #
        # If the tile file contains a column that already
        # exists in candidate features, we do not duplicate
        # it.
        # ==================================================

        tile_feature_columns = [
            column
            for column in tile_df.columns
            if (
                column in PAIR_COLUMNS
                or column
                not in candidate_df.columns
            )
        ]

        tile_df = tile_df[
            tile_feature_columns
        ]

        # ==================================================
        # MERGE
        # ==================================================
        #
        # OUTER is intentional.
        #
        # It allows us to identify any pair that exists in
        # one file but not the other before accepting the
        # final dataset.
        # ==================================================

        final_df = candidate_df.merge(
            tile_df,
            on=PAIR_COLUMNS,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )

        # ==================================================
        # VERIFY EXACT SAME METER-TRANSFORMER PAIRS
        # ==================================================

        candidate_only = (
            final_df["_merge"]
            == "left_only"
        )

        tile_only = (
            final_df["_merge"]
            == "right_only"
        )

        candidate_only_count = int(
            candidate_only.sum()
        )

        tile_only_count = int(
            tile_only.sum()
        )

        if (
            candidate_only_count > 0
            or tile_only_count > 0
        ):
            print(
                f"[{division}] "
                "PAIR VALIDATION FAILED"
            )

            print(
                f"[{division}] "
                f"Pairs only in candidate features: "
                f"{candidate_only_count:,}"
            )

            print(
                f"[{division}] "
                f"Pairs only in tile features: "
                f"{tile_only_count:,}"
            )

            mismatch_examples = (
                final_df.loc[
                    final_df["_merge"] != "both",
                    PAIR_COLUMNS + ["_merge"],
                ]
                .head(10)
            )

            print(
                mismatch_examples.to_string(
                    index=False
                )
            )

            raise ValueError(
                f"[{division}] "
                "Candidate features and tile features "
                "do not contain the exact same "
                "meter-to-transformer candidate pairs."
            )

        # ==================================================
        # PAIR VALIDATION PASSED
        # ==================================================

        final_df = final_df.drop(
            columns=["_merge"]
        )

        print(
            f"[{division}] "
            "Pair validation passed: "
            f"{len(final_df):,} "
            "meter-transformer pairs."
        )

        # ==================================================
        # FINAL SANITY CHECK
        # ==================================================

        if len(final_df) != len(
            candidate_df
        ):
            raise ValueError(
                f"[{division}] "
                "Final feature row count does not "
                "match candidate feature row count."
            )

        # ==================================================
        # SAVE FINAL FEATURE SET
        # ==================================================

        final_df.to_parquet(
            output_path,
            index=False,
            compression="zstd",
        )

        outputs[
            division
        ] = output_path

        print(
            f"[{division}] "
            f"Final feature set saved: "
            f"{output_path}"
        )

        print(
            f"[{division}] "
            f"Rows: {len(final_df):,} | "
            f"Columns: "
            f"{len(final_df.columns):,}"
        )

    # ==================================================
    # COMPLETE
    # ==================================================

    total_rows = 0

    for path in outputs.values():
        total_rows += len(
            pd.read_parquet(
                path,
                columns=PAIR_COLUMNS,
            )
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL FEATURE ASSEMBLY COMPLETE"
    )

    print(
        f"Divisions: "
        f"{len(outputs):,}"
    )

    print(
        f"Total meter-transformer pairs: "
        f"{total_rows:,}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )

    print(
        "=" * 80
    )

    return outputs