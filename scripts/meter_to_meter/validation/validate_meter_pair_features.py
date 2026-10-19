"""Validate meter-pair feature partitions."""

from pathlib import Path

import polars as pl

from mtx.config import (
    METER_PAIR_DIRECTORY,
    METER_PAIR_FEATURE_DIRECTORY,
)
from mtx.run_config import RunConfig


REQUIRED_COLUMNS = {
    "division",
    "meter_1",
    "meter_2",
    "distance_ft",
    "correlation",
}


def source_pair_path(feature_file: Path) -> Path:
    """Return the pair partition behind a feature partition."""

    relative_path = feature_file.resolve().relative_to(
        METER_PAIR_FEATURE_DIRECTORY.resolve()
    )

    return METER_PAIR_DIRECTORY / relative_path


def flatten_files(
    files: dict[str, Path | list[Path]],
) -> list:

    paths: list[Path] = []

    for values in files.values():
        if isinstance(values, Path):
            paths.append(values)
        else:
            paths.extend(
                Path(value)
                for value in values
            )

    return sorted(paths)


def validate_meter_pair_features(
    config: RunConfig,
    feature_files: dict[str, Path | list[Path]],
) -> None:
    """Validate feature files from one configured run."""

    files = flatten_files(feature_files)

    if not files:
        raise FileNotFoundError(
            "No feature partitions were provided."
        )

    for feature_file in files:
        print(
            f"Validating: {feature_file.name}",
            flush=True,
        )

        source_file = source_pair_path(feature_file)

        if not source_file.is_file():
            raise FileNotFoundError(
                f"Source pair file not found: {source_file}"
            )

        feature_schema = pl.read_parquet_schema(
            feature_file
        )

        missing = REQUIRED_COLUMNS - set(feature_schema)

        if missing:
            raise ValueError(
                f"{feature_file.name} is missing: "
                f"{sorted(missing)}"
            )

        feature_df = pl.scan_parquet(feature_file)
        pair_df = pl.scan_parquet(source_file)

        feature_rows = (
            feature_df
            .select(pl.len())
            .collect()
            .item()
        )

        pair_rows = (
            pair_df
            .select(pl.len())
            .collect()
            .item()
        )

        if feature_rows != pair_rows:
            raise ValueError(
                f"{feature_file.name} has "
                f"{feature_rows:,} rows; "
                f"its source has {pair_rows:,}."
            )

        invalid_rows = (
            feature_df
            .filter(
                pl.col("meter_1").is_null()
                | pl.col("meter_2").is_null()
                | (
                    pl.col("meter_1")
                    == pl.col("meter_2")
                )
                | pl.col("distance_ft").is_null()
                | (pl.col("distance_ft") < 0)
                | (
                    pl.col("distance_ft")
                    > config.meter_correlation_radius + 0.01
                )
                | (
                    pl.col("correlation").is_not_null()
                    & (
                        (pl.col("correlation") < -1)
                        | (pl.col("correlation") > 1)
                    )
                )
            )
            .select(pl.len())
            .collect()
            .item()
        )

        if invalid_rows:
            raise ValueError(
                f"{feature_file.name} contains "
                f"{invalid_rows:,} invalid rows."
            )

        similarity_columns = [
            column
            for column in feature_schema
            if column.endswith("_similarity_pct")
        ]

        for column in similarity_columns:
            invalid_similarity = (
                feature_df
                .filter(
                    pl.col(column).is_not_null()
                    & (
                        (pl.col(column) < 0)
                        | (pl.col(column) > 100)
                    )
                )
                .select(pl.len())
                .collect()
                .item()
            )

            if invalid_similarity:
                raise ValueError(
                    f"{feature_file.name} contains invalid "
                    f"values in {column}."
                )

        print(
            f"PASS: {feature_rows:,} rows",
            flush=True,
        )

    print(
        "\nAll feature partitions passed validation.",
        flush=True,
    )