"""Create or reuse meter-pair feature partitions."""

from pathlib import Path

import pandas as pd
import polars as pl

from mtx.config import (
    METER_PAIR_DIRECTORY,
    METER_PAIR_FEATURE_DIRECTORY,
)
from mtx.features.meter_pair_features import build_meter_pair_features
from mtx.run_config import RunConfig


REQUIRED_FEATURE_COLUMNS = {
    "division",
    "meter_1",
    "meter_2",
    "distance_ft",
    "correlation",
}


def feature_output_path(pair_file: Path) -> Path:
    """Return the feature path matching a pair partition."""

    relative_path = pair_file.resolve().relative_to(
        METER_PAIR_DIRECTORY.resolve()
    )

    return METER_PAIR_FEATURE_DIRECTORY / relative_path


def parquet_row_count(path: Path) -> int:
    """Return the number of rows in a Parquet file."""

    return (
        pl.scan_parquet(path)
        .select(pl.len())
        .collect()
        .item()
    )


def valid_feature_file(
    feature_file: Path,
    pair_file: Path,
) -> bool:
    """Return True when an existing feature file is valid."""

    if not feature_file.is_file():
        return False

    try:
        columns = set(pl.read_parquet_schema(feature_file))

        return (
            REQUIRED_FEATURE_COLUMNS.issubset(columns)
            and parquet_row_count(feature_file)
            == parquet_row_count(pair_file)
        )

    except Exception:
        return False


def save_feature_file(
    feature_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save a feature partition."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_df.to_parquet(
        output_path,
        index=False,
    )


def run_meter_pair_feature_stage(
    config: RunConfig,
    pair_files: dict[str, Path | list[Path]],
    ami_df: pd.DataFrame,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Build or reuse meter-pair feature partitions."""

    if ami_df is None or ami_df.empty:
        raise ValueError("AMI data is empty.")

    feature_files = {}

    for division, files in pair_files.items():

        if isinstance(files, Path):
            files = [files]

        files = [Path(file) for file in files]
        feature_files[division] = []

        for pair_file in files:

            if not pair_file.is_file():
                raise FileNotFoundError(
                    f"Pair file not found: {pair_file}"
                )

            output_path = feature_output_path(pair_file)
            feature_files[division].append(output_path)

            # Reuse valid existing output
            if (
                not overwrite
                and valid_feature_file(
                    output_path,
                    pair_file,
                )
            ):
                print(
                    f"[{division}] Reusing {output_path.name}"
                )
                continue

            print(
                f"[{division}] Processing {pair_file.name}"
            )

            pair_df = pd.read_parquet(pair_file)

            feature_df = build_meter_pair_features(
                pair_df=pair_df,
                ami_df=ami_df,
                selected_ami_columns=config.ami_columns,
                similarity_columns=config.similarity_columns,
                exact_match_columns=config.exact_match_columns,
            )

            save_feature_file(
                feature_df,
                output_path,
            )

            print(
                f"[{division}] Saved "
                f"{len(feature_df):,} rows"
            )

    return feature_files