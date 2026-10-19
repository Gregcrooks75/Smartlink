"""Repair and validate completed meter-pair partitions."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import polars as pl

from mtx.run_config import RunConfig


REQUIRED_COLUMNS = {
    "division",
    "meter_1",
    "meter_2",
    "distance_ft",
    "correlation",
}


def normalize_pair_files(
    pair_files: dict[str, Path | list[Path]],
) -> dict[str, list[Path]]:
    """Ensure every division maps to a list of paths."""
    return {
        division: (
            [Path(paths)]
            if isinstance(paths, Path)
            else sorted(Path(path) for path in paths)
        )
        for division, paths in pair_files.items()
    }


def sql_path(path: Path) -> str:
    """Escape a path for DuckDB SQL."""
    return str(path.resolve()).replace("'", "''")


def parquet_source(paths: list[Path]) -> str:
    """Create a DuckDB Parquet source expression."""
    sources = ", ".join(
        f"'{sql_path(path)}'"
        for path in paths
    )

    if len(paths) == 1:
        return f"read_parquet({sources})"

    return f"read_parquet([{sources}])"


def find_duplicate_divisions(
    pair_files: dict[str, Path | list[Path]],
) -> dict[str, int]:
    """
    Quickly count duplicate and reversed pairs by division.

    Only meter_1 and meter_2 are read from the Parquet files.
    """
    normalized = normalize_pair_files(pair_files)
    duplicate_divisions: dict[str, int] = {}

    print(
        "\nRunning fast duplicate check...",
        flush=True,
    )

    with TemporaryDirectory(
        prefix="smartlink_duplicate_check_"
    ) as temporary_directory:
        database_path = (
            Path(temporary_directory)
            / "check.duckdb"
        )

        connection = duckdb.connect(
            str(database_path)
        )

        try:
            connection.execute(
                "SET preserve_insertion_order = false"
            )

            for division, paths in sorted(
                normalized.items()
            ):
                if not paths:
                    continue

                missing_files = [
                    path
                    for path in paths
                    if not path.is_file()
                ]

                if missing_files:
                    raise FileNotFoundError(
                        f"[{division}] Missing pair files:\n"
                        + "\n".join(
                            str(path)
                            for path in missing_files
                        )
                    )

                source = parquet_source(paths)

                duplicate_count = connection.execute(
                    f"""
                    WITH canonical AS (
                        SELECT
                            LEAST(
                                CAST(meter_1 AS VARCHAR),
                                CAST(meter_2 AS VARCHAR)
                            ) AS pair_first,
                            GREATEST(
                                CAST(meter_1 AS VARCHAR),
                                CAST(meter_2 AS VARCHAR)
                            ) AS pair_second
                        FROM {source}
                    ),
                    duplicate_groups AS (
                        SELECT
                            pair_first,
                            pair_second,
                            COUNT(*) AS occurrences
                        FROM canonical
                        GROUP BY
                            pair_first,
                            pair_second
                        HAVING COUNT(*) > 1
                    )
                    SELECT
                        COALESCE(
                            SUM(occurrences - 1),
                            0
                        )
                    FROM duplicate_groups
                    """
                ).fetchone()[0]

                duplicate_count = int(
                    duplicate_count
                )

                if duplicate_count:
                    duplicate_divisions[division] = (
                        duplicate_count
                    )

                    print(
                        f"[{division}] "
                        f"{duplicate_count:,} duplicate or "
                        "reversed pairs found.",
                        flush=True,
                    )
                else:
                    print(
                        f"[{division}] No duplicates.",
                        flush=True,
                    )

        finally:
            connection.close()

    if not duplicate_divisions:
        print(
            "\nNo duplicate or reversed pairs found. "
            "Repair skipped.",
            flush=True,
        )

    return duplicate_divisions


def remove_duplicate_pairs(
    pair_files: dict[str, Path | list[Path]],
) -> int:
    """
    Remove exact and reversed pairs from affected divisions.

    A-B and B-A are treated as the same pair. Existing partitions
    are preserved and only affected files are rewritten.
    """
    normalized_files = normalize_pair_files(
        pair_files
    )

    duplicate_divisions = find_duplicate_divisions(
        normalized_files
    )

    if not duplicate_divisions:
        return 0

    # Avoid processing divisions that passed the fast check.
    files_to_repair = {
        division: paths
        for division, paths in normalized_files.items()
        if division in duplicate_divisions
    }

    removed_total = 0

    print(
        "\nRepairing affected divisions...",
        flush=True,
    )

    with TemporaryDirectory(
        prefix="smartlink_pair_repair_"
    ) as temporary_directory:
        database_path = (
            Path(temporary_directory)
            / "repair.duckdb"
        )

        connection = duckdb.connect(
            str(database_path)
        )

        try:
            connection.execute(
                "SET preserve_insertion_order = false"
            )

            connection.execute(
                """
                CREATE TABLE seen_pairs (
                    division VARCHAR,
                    pair_first VARCHAR,
                    pair_second VARCHAR
                )
                """
            )

            for division, paths in sorted(
                files_to_repair.items()
            ):
                division_removed = 0

                print(
                    f"\n[{division}] Repairing "
                    f"{len(paths)} partition(s)...",
                    flush=True,
                )

                for file_path in paths:
                    if not file_path.is_file():
                        raise FileNotFoundError(
                            f"Pair file not found: "
                            f"{file_path}"
                        )

                    schema = pl.read_parquet_schema(
                        file_path
                    )

                    missing = (
                        REQUIRED_COLUMNS
                        - set(schema)
                    )

                    if missing:
                        raise ValueError(
                            f"{file_path.name} is missing "
                            f"columns: {sorted(missing)}"
                        )

                    source = sql_path(file_path)

                    connection.execute(
                        f"""
                        CREATE OR REPLACE TEMP TABLE retained AS

                        WITH canonical AS (
                            SELECT
                                *,
                                LEAST(
                                    CAST(meter_1 AS VARCHAR),
                                    CAST(meter_2 AS VARCHAR)
                                ) AS _pair_first,
                                GREATEST(
                                    CAST(meter_1 AS VARCHAR),
                                    CAST(meter_2 AS VARCHAR)
                                ) AS _pair_second
                            FROM read_parquet('{source}')
                        ),
                        ranked AS (
                            SELECT
                                *,
                                ROW_NUMBER() OVER (
                                    PARTITION BY
                                        division,
                                        _pair_first,
                                        _pair_second
                                    ORDER BY
                                        correlation IS NULL,
                                        meter_1,
                                        meter_2
                                ) AS _occurrence
                            FROM canonical
                        )
                        SELECT
                            ranked.*
                        FROM ranked
                        ANTI JOIN seen_pairs
                            ON (
                                seen_pairs.division
                                = ranked.division
                            )
                            AND (
                                seen_pairs.pair_first
                                = ranked._pair_first
                            )
                            AND (
                                seen_pairs.pair_second
                                = ranked._pair_second
                            )
                        WHERE _occurrence = 1
                        """
                    )

                    original_rows = (
                        connection.execute(
                            f"""
                            SELECT COUNT(*)
                            FROM read_parquet('{source}')
                            """
                        ).fetchone()[0]
                    )

                    retained_rows = (
                        connection.execute(
                            """
                            SELECT COUNT(*)
                            FROM retained
                            """
                        ).fetchone()[0]
                    )

                    removed_rows = (
                        original_rows
                        - retained_rows
                    )

                    connection.execute(
                        """
                        INSERT INTO seen_pairs
                        SELECT
                            division,
                            _pair_first,
                            _pair_second
                        FROM retained
                        """
                    )

                    if removed_rows:
                        temporary_output = (
                            file_path.parent
                            / "__deduplicated__.tmp"
                        )

                        temporary_output.unlink(
                            missing_ok=True
                        )

                        destination = sql_path(
                            temporary_output
                        )

                        connection.execute(
                            f"""
                            COPY (
                                SELECT * EXCLUDE (
                                    _pair_first,
                                    _pair_second,
                                    _occurrence
                                )
                                FROM retained
                            )
                            TO '{destination}'
                            (
                                FORMAT PARQUET,
                                COMPRESSION ZSTD
                            )
                            """
                        )

                        cleaned_schema = (
                            pl.read_parquet_schema(
                                temporary_output
                            )
                        )

                        if not REQUIRED_COLUMNS.issubset(
                            set(cleaned_schema)
                        ):
                            temporary_output.unlink(
                                missing_ok=True
                            )

                            raise ValueError(
                                "Invalid cleaned file: "
                                f"{temporary_output}"
                            )

                        cleaned_rows = (
                            pl.scan_parquet(
                                temporary_output
                            )
                            .select(pl.len())
                            .collect()
                            .item()
                        )

                        if cleaned_rows != retained_rows:
                            temporary_output.unlink(
                                missing_ok=True
                            )

                            raise ValueError(
                                f"{file_path.name}: expected "
                                f"{retained_rows:,} cleaned rows, "
                                f"found {cleaned_rows:,}."
                            )

                        # Release the table before replacing the
                        # source file on Windows.
                        connection.execute(
                            "DROP TABLE retained"
                        )

                        os.replace(
                            temporary_output,
                            file_path,
                        )

                        division_removed += removed_rows

                        print(
                            f"REPAIRED: {file_path.name} | "
                            f"removed {removed_rows:,}",
                            flush=True,
                        )
                    else:
                        connection.execute(
                            "DROP TABLE retained"
                        )

                removed_total += division_removed

                print(
                    f"[{division}] Total removed: "
                    f"{division_removed:,}",
                    flush=True,
                )

        finally:
            connection.close()

    print(
        f"\nDuplicate repair complete: "
        f"{removed_total:,} rows removed.",
        flush=True,
    )

    return removed_total


def validate_pair_file(
    file_path: Path,
    maximum_radius_ft: float,
) -> int:
    """Validate one meter-pair partition."""
    if not file_path.is_file():
        raise FileNotFoundError(
            f"Pair file not found: {file_path}"
        )

    schema = pl.read_parquet_schema(file_path)
    missing = REQUIRED_COLUMNS - set(schema)

    if missing:
        raise ValueError(
            f"{file_path.name} is missing columns: "
            f"{sorted(missing)}"
        )

    pair_scan = pl.scan_parquet(file_path)

    meter_1 = pl.col("meter_1")
    meter_2 = pl.col("meter_2")
    division = pl.col("division")
    distance = pl.col("distance_ft")
    correlation = pl.col("correlation")

    statistics = (
        pair_scan.select(
            pl.len().alias("rows"),
            division.is_null()
            .sum()
            .alias("null_divisions"),
            (
                meter_1.is_null()
                | meter_2.is_null()
            )
            .sum()
            .alias("null_meters"),
            (
                meter_1 == meter_2
            )
            .sum()
            .alias("self_pairs"),
            correlation.is_null()
            .sum()
            .alias("null_correlations"),
            correlation.is_nan()
            .sum()
            .alias("nan_correlations"),
            correlation.is_infinite()
            .sum()
            .alias("infinite_correlations"),
            (
                correlation.is_finite()
                & (
                    (correlation < -1)
                    | (correlation > 1)
                )
            )
            .sum()
            .alias("outside_range"),
            pl.when(correlation.is_finite())
            .then(correlation)
            .otherwise(None)
            .min()
            .alias("minimum_correlation"),
            pl.when(correlation.is_finite())
            .then(correlation)
            .otherwise(None)
            .max()
            .alias("maximum_correlation"),
            (
                distance.is_null()
                | distance.is_nan()
                | distance.is_infinite()
                | (distance < 0)
                | (
                    distance
                    > maximum_radius_ft + 0.01
                )
            )
            .sum()
            .alias("invalid_distances"),
            distance.max().alias(
                "maximum_distance"
            ),
        )
        .collect()
        .row(
            0,
            named=True,
        )
    )

    if statistics["null_divisions"]:
        raise ValueError(
            f"{file_path.name} contains null divisions."
        )

    if statistics["null_meters"]:
        raise ValueError(
            f"{file_path.name} contains "
            f"{statistics['null_meters']:,} "
            "null meter IDs."
        )

    if statistics["self_pairs"]:
        raise ValueError(
            f"{file_path.name} contains "
            f"{statistics['self_pairs']:,} self-pairs."
        )

    if statistics["nan_correlations"]:
        raise ValueError(
            f"{file_path.name} contains "
            f"{statistics['nan_correlations']:,} "
            "NaN correlations."
        )

    if statistics["infinite_correlations"]:
        raise ValueError(
            f"{file_path.name} contains "
            "infinite correlations."
        )

    if statistics["outside_range"]:
        raise ValueError(
            f"{file_path.name} contains correlations "
            "outside [-1, 1]."
        )

    if statistics["invalid_distances"]:
        raise ValueError(
            f"{file_path.name} contains "
            f"{statistics['invalid_distances']:,} "
            "invalid distances."
        )

    minimum = statistics[
        "minimum_correlation"
    ]

    maximum = statistics[
        "maximum_correlation"
    ]

    maximum_distance = (
        statistics["maximum_distance"]
        or 0
    )

    print(
        f"PASS: {file_path.name} | "
        f"{statistics['rows']:,} rows | "
        f"{statistics['null_correlations']:,} "
        f"null correlations | "
        f"correlation range: "
        f"{minimum} to {maximum} | "
        f"maximum distance: "
        f"{maximum_distance:,.2f} ft",
        flush=True,
    )

    return int(statistics["rows"])


def validate_meter_pairs(
    config: RunConfig,
    pair_files: dict[str, Path | list[Path]],
    remove_duplicates: bool = True,
) -> None:
    """Repair and validate meter-pair outputs."""
    normalized_files = normalize_pair_files(
        pair_files
    )

    if not normalized_files:
        raise FileNotFoundError(
            "No meter-pair files were provided."
        )

    if remove_duplicates:
        remove_duplicate_pairs(
            normalized_files
        )

    maximum_radius = getattr(
        config,
        "meter_correlation_radius",
        None,
    )

    if maximum_radius is None:
        maximum_radius = getattr(
            config,
            "radius_ft",
            None,
        )

    if maximum_radius is None:
        raise AttributeError(
            "RunConfig does not define a "
            "correlation radius."
        )

    print(
        "\nRunning meter-pair value validation...",
        flush=True,
    )

    total_files = 0
    total_rows = 0

    for division, paths in sorted(
        normalized_files.items()
    ):
        print(
            f"\nValidating division: {division}",
            flush=True,
        )

        for file_path in paths:
            total_rows += validate_pair_file(
                file_path=file_path,
                maximum_radius_ft=float(
                    maximum_radius
                ),
            )

            total_files += 1

    print(
        "\nAll meter-pair files passed validation.\n"
        f"Files validated: {total_files:,}\n"
        f"Pairs validated: {total_rows:,}",
        flush=True,
    )


def main() -> None:
    """Validate the configured meter-pair run."""
    start_date = "2026-02-02"
    end_date = "2026-02-08"

    config = RunConfig(
        week_start=start_date,
        week_end=end_date,
        correlation_start=start_date,
        correlation_end=end_date,
        meter_correlation_radius=550,
        correlation_method="spearman",
        weekly_partition_size=25_000,
        seed_batch_size=5_000,
    )

    pair_files: dict[str, list[Path]] = {}

    for file_path in sorted(
        config.pair_directory.rglob(
            "part_*.parquet"
        )
    ):
        division = file_path.parent.name

        pair_files.setdefault(
            division,
            [],
        ).append(file_path)

    if not pair_files:
        raise FileNotFoundError(
            "No meter-pair partitions found under "
            f"{config.pair_directory}"
        )

    validate_meter_pairs(
        config=config,
        pair_files=pair_files,
        remove_duplicates=True,
    )


if __name__ == "__main__":
    main()