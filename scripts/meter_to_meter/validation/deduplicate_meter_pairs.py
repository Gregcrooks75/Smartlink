"""Remove duplicate and reversed meter pairs across partitions."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import polars as pl


REQUIRED_COLUMNS = {
    "division",
    "meter_1",
    "meter_2",
    "distance_ft",
    "correlation",
}


def _paths_as_list(
    paths: Path | list[Path],
) -> list:
    if isinstance(paths, Path):
        return [paths]

    return sorted(Path(path) for path in paths)


def _sql_path(path: Path) -> str:
    """Escape a filesystem path for DuckDB SQL."""
    return str(path.resolve()).replace("'", "''")


def deduplicate_meter_pairs(
    pair_files: dict[str, Path | list[Path]],
) -> dict[str, list[Path]]:
    """
    Remove exact and reversed duplicates across pair partitions.

    The files remain partitioned. For every unordered pair, the first
    occurrence is retained and later occurrences are removed.
    """
    cleaned_files: dict[str, list[Path]] = {}
    total_removed = 0

    for division, division_paths in sorted(pair_files.items()):
        paths = _paths_as_list(division_paths)

        if not paths:
            cleaned_files[division] = []
            continue

        print(
            f"\n[{division}] Checking {len(paths)} pair partition(s)...",
            flush=True,
        )

        work_database = (
            paths[0].parent / ".meter_pair_deduplication.duckdb"
        )
        work_database.unlink(missing_ok=True)

        connection = duckdb.connect(str(work_database))

        try:
            # The seen-pair table is stored on disk rather than holding
            # millions of identifiers entirely in memory.
            connection.execute(
                """
                CREATE TABLE seen_pairs (
                    division VARCHAR,
                    pair_first VARCHAR,
                    pair_second VARCHAR
                )
                """
            )

            division_removed = 0

            for part_number, file_path in enumerate(paths, start=1):
                if not file_path.is_file():
                    raise FileNotFoundError(file_path)

                schema = pl.read_parquet_schema(file_path)
                missing = REQUIRED_COLUMNS - set(schema)

                if missing:
                    raise ValueError(
                        f"{file_path.name} is missing "
                        f"{sorted(missing)}"
                    )

                source_path = _sql_path(file_path)

                connection.execute(
                    f"""
                    CREATE TEMP TABLE current_pairs AS
                    SELECT
                        *,
                        LEAST(
                            CAST(meter_1 AS VARCHAR),
                            CAST(meter_2 AS VARCHAR)
                        ) AS _pair_first,
                        GREATEST(
                            CAST(meter_1 AS VARCHAR),
                            CAST(meter_2 AS VARCHAR)
                        ) AS _pair_second,
                        ROW_NUMBER() OVER (
                            PARTITION BY
                                division,
                                LEAST(
                                    CAST(meter_1 AS VARCHAR),
                                    CAST(meter_2 AS VARCHAR)
                                ),
                                GREATEST(
                                    CAST(meter_1 AS VARCHAR),
                                    CAST(meter_2 AS VARCHAR)
                                )
                            ORDER BY
                                meter_1,
                                meter_2
                        ) AS _occurrence
                    FROM read_parquet('{source_path}')
                    """
                )

                original_rows = connection.execute(
                    "SELECT COUNT(*) FROM current_pairs"
                ).fetchone()[0]

                # Keep one occurrence inside the current partition and
                # exclude pairs already encountered in earlier parts.
                connection.execute(
                    """
                    CREATE TEMP TABLE retained_pairs AS
                    SELECT
                        current_pairs.*
                    FROM current_pairs
                    WHERE
                        _occurrence = 1
                        AND NOT EXISTS (
                            SELECT 1
                            FROM seen_pairs
                            WHERE
                                seen_pairs.division = current_pairs.division
                                AND seen_pairs.pair_first =
                                    current_pairs._pair_first
                                AND seen_pairs.pair_second =
                                    current_pairs._pair_second
                        )
                    """
                )

                retained_rows = connection.execute(
                    "SELECT COUNT(*) FROM retained_pairs"
                ).fetchone()[0]

                removed_rows = original_rows - retained_rows
                division_removed += removed_rows

                # Register retained pairs before checking the next part.
                connection.execute(
                    """
                    INSERT INTO seen_pairs
                    SELECT DISTINCT
                        division,
                        _pair_first,
                        _pair_second
                    FROM retained_pairs
                    """
                )

                if removed_rows:
                    temporary_path = file_path.with_suffix(
                        ".deduplicated.tmp"
                    )
                    temporary_path.unlink(missing_ok=True)

                    destination = _sql_path(temporary_path)

                    connection.execute(
                        f"""
                        COPY (
                            SELECT * EXCLUDE (
                                _pair_first,
                                _pair_second,
                                _occurrence
                            )
                            FROM retained_pairs
                        )
                        TO '{destination}'
                        (
                            FORMAT PARQUET,
                            COMPRESSION ZSTD
                        )
                        """
                    )

                    temporary_schema = pl.read_parquet_schema(
                        temporary_path
                    )

                    if not REQUIRED_COLUMNS.issubset(temporary_schema):
                        temporary_path.unlink(missing_ok=True)
                        raise ValueError(
                            "Invalid cleaned file: "
                            f"{temporary_path}"
                        )

                    # Release the source before replacing it on Windows.
                    connection.execute(
                        "DROP TABLE retained_pairs"
                    )
                    connection.execute(
                        "DROP TABLE current_pairs"
                    )

                    os.replace(temporary_path, file_path)

                    print(
                        f"[{division}] Part "
                        f"{part_number}/{len(paths)}: "
                        f"removed {removed_rows:,} duplicates",
                        flush=True,
                    )
                else:
                    connection.execute(
                        "DROP TABLE retained_pairs"
                    )
                    connection.execute(
                        "DROP TABLE current_pairs"
                    )

            total_removed += division_removed
            cleaned_files[division] = paths

            print(
                f"[{division}] Complete: "
                f"{division_removed:,} duplicate pairs removed.",
                flush=True,
            )

        finally:
            connection.close()
            work_database.unlink(missing_ok=True)

    print(
        f"\nDeduplication complete: "
        f"{total_removed:,} total rows removed.",
        flush=True,
    )

    return cleaned_files