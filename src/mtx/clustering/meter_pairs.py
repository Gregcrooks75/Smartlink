"""
Compute spatially constrained meter correlations.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from scipy.stats import rankdata
from sklearn.neighbors import BallTree


EARTH_RADIUS_FT = 20_902_231
CORRELATION_TOLERANCE = 1e-5


PAIR_SCHEMA = {
    "division": pl.String,
    "meter_1": pl.String,
    "meter_2": pl.String,
    "distance_ft": pl.Float32,
    "correlation": pl.Float32,
}


def valid_pair_file(path: Path) -> bool:
    """
    Return whether an existing pair partition is usable.
    """
    if not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        schema = pl.read_parquet_schema(path)
    except Exception:
        return False

    if not set(PAIR_SCHEMA).issubset(schema):
        return False

    try:
        invalid_count = (
            pl.scan_parquet(path)
            .filter(
                pl.col("correlation").is_not_null()
                & (
                    ~pl.col("correlation").is_finite()
                    | (pl.col("correlation") < -1)
                    | (pl.col("correlation") > 1)
                )
            )
            .select(pl.len())
            .collect()
            .item()
        )
    except Exception:
        return False

    return invalid_count == 0


def _voltage_columns(
    paths: list[Path],
    start_date: str,
    end_date: str,
) -> list:
    if not paths:
        raise ValueError(
            "No weekly-voltage partitions were provided."
        )

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)

    columns: list[str] = []

    first_schema = pl.read_parquet_schema(paths[0])

    for column in first_schema:
        if column == "meter_number":
            continue

        try:
            timestamp = pd.Timestamp(column)
        except (TypeError, ValueError):
            continue

        if start <= timestamp < end:
            columns.append(column)

    if not columns:
        raise ValueError(
            f"No voltage columns found from "
            f"{start_date} to {end_date}."
        )

    required = {
        "meter_number",
        *columns,
    }

    # Ensure every weekly partition has the same required columns.
    for path in paths:
        schema = pl.read_parquet_schema(path)
        missing = required - set(schema)

        if missing:
            raise ValueError(
                f"{path.name} is missing voltage columns: "
                f"{sorted(missing)}"
            )

    return columns


def _normalize_voltage(
    matrix: np.ndarray,
    method: str,
    batch_size: int = 5_000,
) -> np.ndarray:
    """
    Normalize voltage rows for vectorized correlation.
    """
    if method not in {
        "pearson",
        "spearman",
    }:
        raise ValueError(
            "Correlation method must be pearson or spearman."
        )

    normalized = np.full(
        matrix.shape,
        np.nan,
        dtype=np.float32,
    )

    for start in range(
        0,
        len(matrix),
        batch_size,
    ):
        stop = min(
            start + batch_size,
            len(matrix),
        )

        block = matrix[start:stop]

        if method == "spearman":
            block = rankdata(
                block,
                axis=1,
                method="average",
                nan_policy="propagate",
            ).astype(np.float32)

        block = block - block.mean(
            axis=1,
            keepdims=True,
        )

        norms = np.linalg.norm(
            block,
            axis=1,
        )

        valid_rows = (
            np.isfinite(norms)
            & (norms > 0)
            & np.isfinite(block).all(axis=1)
        )

        target = normalized[start:stop]

        target[valid_rows] = (
            block[valid_rows]
            / norms[valid_rows, None]
        )

    return normalized


def _pair_correlations(
    normalized: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    batch_size: int = 100_000,
) -> np.ndarray:
    """
    Calculate pair correlations in memory-safe batches.
    """
    correlations = np.empty(
        len(left),
        dtype=np.float32,
    )

    for start in range(
        0,
        len(left),
        batch_size,
    ):
        stop = min(
            start + batch_size,
            len(left),
        )

        correlations[start:stop] = np.einsum(
            "ij,ij->i",
            normalized[left[start:stop]],
            normalized[right[start:stop]],
        )

    return correlations


def _clean_correlations(
    correlations: np.ndarray,
    division: str,
    part_number: int,
) -> tuple[np.ndarray, int]:
    """
    Validate correlations and convert minor overflow safely.
    """
    if np.isinf(correlations).any():
        infinity_count = int(
            np.isinf(correlations).sum()
        )

        raise ValueError(
            f"[{division}] Part {part_number} produced "
            f"{infinity_count:,} infinite correlations."
        )

    finite_mask = np.isfinite(correlations)

    substantially_invalid = (
        finite_mask
        & (
            (correlations < -1 - CORRELATION_TOLERANCE)
            | (correlations > 1 + CORRELATION_TOLERANCE)
        )
    )

    if substantially_invalid.any():
        invalid_values = correlations[
            substantially_invalid
        ]

        raise ValueError(
            f"[{division}] Part {part_number} produced "
            f"{len(invalid_values):,} correlations substantially "
            "outside [-1, 1]. "
            f"Minimum: {invalid_values.min()}, "
            f"maximum: {invalid_values.max()}."
        )

    cleaned = correlations.copy()

    # Correct only insignificant floating-point overflow.
    cleaned[finite_mask] = np.clip(
        cleaned[finite_mask],
        -1.0,
        1.0,
    )

    undefined_count = int(
        np.isnan(cleaned).sum()
    )

    return cleaned, undefined_count


def _save_partition(
    pair_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """
    Write a pair partition atomically.
    """
    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp.parquet"
    )

    pair_df.write_parquet(
        temporary_path,
        compression="zstd",
    )

    if not valid_pair_file(temporary_path):
        temporary_path.unlink(missing_ok=True)

        raise ValueError(
            f"Invalid temporary output: {temporary_path}"
        )

    os.replace(
        temporary_path,
        output_path,
    )


def compute_meter_pair_partitions(
    ami_df: pd.DataFrame,
    weekly_voltage_paths: list[Path],
    division: str,
    output_directory: Path,
    start_date: str,
    end_date: str,
    radius_ft: float = 550,
    correlation_method: str = "spearman",
    seed_batch_size: int = 5_000,
    overwrite: bool = False,
) -> list:

    if radius_ft <= 0:
        raise ValueError("radius_ft must be positive.")

    if seed_batch_size <= 0:
        raise ValueError(
            "seed_batch_size must be positive."
        )

    weekly_voltage_paths = [
        Path(path)
        for path in weekly_voltage_paths
    ]

    if not weekly_voltage_paths:
        raise ValueError(
            f"No weekly-voltage files provided for {division}."
        )

    missing_files = [
        path
        for path in weekly_voltage_paths
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing weekly-voltage files for {division}: "
            f"{missing_files}"
        )

    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Read every weekly partition for the complete division.
    weekly_ids = (
        pl.read_parquet(
            weekly_voltage_paths,
            columns=["meter_number"],
        )
        .with_columns(
            pl.col("meter_number")
            .cast(pl.String)
            .str.strip_chars()
        )
    )

    if weekly_ids.is_empty():
        raise ValueError(
            f"No weekly meters found for {division}."
        )

    if (
        weekly_ids["meter_number"].n_unique()
        != weekly_ids.height
    ):
        raise ValueError(
            f"{division} weekly files contain duplicate meters."
        )

    original_meter_ids = (
        weekly_ids["meter_number"].to_list()
    )

    # Prepare AMI information.
    ami = ami_df.rename(
        columns={"MeterID": "meter_number"}
    ).copy()

    required_ami_columns = {
        "meter_number",
        "Division",
        "LATITUDE",
        "LONGITUDE",
    }

    missing_ami_columns = (
        required_ami_columns
        - set(ami.columns)
    )

    if missing_ami_columns:
        raise ValueError(
            "AMI data is missing columns: "
            f"{sorted(missing_ami_columns)}"
        )

    ami["meter_number"] = (
        ami["meter_number"]
        .astype("string")
        .str.strip()
    )

    ami["Division"] = (
        ami["Division"]
        .astype("string")
        .str.strip()
    )

    ami = ami[
        ami["Division"] == division
    ].copy()

    if ami["meter_number"].duplicated().any():
        duplicate_count = int(
            ami["meter_number"]
            .duplicated()
            .sum()
        )

        raise ValueError(
            f"AMI data contains {duplicate_count:,} duplicate "
            f"meters for {division}."
        )

    ami = (
        ami.set_index("meter_number")
        .reindex(original_meter_ids)
    )

    latitude = pd.to_numeric(
        ami["LATITUDE"],
        errors="coerce",
    ).to_numpy()

    longitude = pd.to_numeric(
        ami["LONGITUDE"],
        errors="coerce",
    ).to_numpy()

    valid_coordinates = (
        np.isfinite(latitude)
        & np.isfinite(longitude)
        & (latitude >= -90)
        & (latitude <= 90)
        & (longitude >= -180)
        & (longitude <= 180)
    )

    meter_ids = np.asarray(
        original_meter_ids,
        dtype=object,
    )[valid_coordinates]

    coordinates = np.radians(
        np.column_stack(
            [
                latitude[valid_coordinates],
                longitude[valid_coordinates],
            ]
        )
    )

    if not len(meter_ids):
        raise ValueError(
            f"No valid coordinates found for {division}."
        )

    invalid_coordinate_count = (
        len(original_meter_ids)
        - len(meter_ids)
    )

    if invalid_coordinate_count:
        print(
            f"[{division}] Excluding "
            f"{invalid_coordinate_count:,} meters with invalid "
            "coordinates.",
            flush=True,
        )

    partition_count = int(
        np.ceil(
            len(meter_ids)
            / seed_batch_size
        )
    )

    output_paths = [
        output_directory
        / f"part_{part_number:04d}.parquet"
        for part_number in range(
            1,
            partition_count + 1,
        )
    ]

    # Skip the expensive work if all partitions are valid.
    if (
        not overwrite
        and all(
            valid_pair_file(path)
            for path in output_paths
        )
    ):
        print(
            f"[{division}] Reusing "
            f"{partition_count} valid parts.",
            flush=True,
        )

        return output_paths

    voltage_columns = _voltage_columns(
        paths=weekly_voltage_paths,
        start_date=start_date,
        end_date=end_date,
    )

    voltage_df = (
        pl.read_parquet(
            weekly_voltage_paths,
            columns=[
                "meter_number",
                *voltage_columns,
            ],
        )
        .with_columns(
            pl.col("meter_number")
            .cast(pl.String)
            .str.strip_chars()
        )
    )

    if (
        voltage_df["meter_number"].n_unique()
        != voltage_df.height
    ):
        raise ValueError(
            f"{division} voltage files contain duplicate meters."
        )

    # Join explicitly to preserve alignment between meter IDs,
    # coordinates and voltage rows.
    valid_meter_order = pl.DataFrame(
        {
            "meter_number": meter_ids.tolist(),
            "_row_order": np.arange(
                len(meter_ids),
                dtype=np.int64,
            ),
        }
    )

    voltage_df = (
        valid_meter_order.join(
            voltage_df,
            on="meter_number",
            how="left",
            validate="1:1",
        )
        .sort("_row_order")
    )

    missing_voltage_rows = (
        voltage_df
        .select(
            pl.any_horizontal(
                pl.col(voltage_columns).is_null()
            ).sum()
        )
        .item()
    )

    if missing_voltage_rows:
        print(
            f"[{division}] {missing_voltage_rows:,} meters contain "
            "null voltage values; their correlations may be null.",
            flush=True,
        )

    matrix = (
        voltage_df
        .select(voltage_columns)
        .to_numpy()
        .astype(
            np.float32,
            copy=False,
        )
    )

    normalized = _normalize_voltage(
        matrix=matrix,
        method=correlation_method,
    )

    undefined_meter_count = int(
        np.isnan(normalized).all(axis=1).sum()
    )

    if undefined_meter_count:
        print(
            f"[{division}] {undefined_meter_count:,} meters have "
            "undefined normalized voltage profiles.",
            flush=True,
        )

    del matrix, voltage_df

    tree = BallTree(
        coordinates,
        metric="haversine",
    )

    radius_radians = (
        radius_ft / EARTH_RADIUS_FT
    )

    for part_number, seed_start in enumerate(
        range(
            0,
            len(meter_ids),
            seed_batch_size,
        ),
        start=1,
    ):
        output_path = output_paths[
            part_number - 1
        ]

        if (
            not overwrite
            and valid_pair_file(output_path)
        ):
            print(
                f"[{division}] Reusing part "
                f"{part_number}/{partition_count}.",
                flush=True,
            )
            continue

        seed_stop = min(
            seed_start + seed_batch_size,
            len(meter_ids),
        )

        neighbors, distances = tree.query_radius(
            coordinates[seed_start:seed_stop],
            r=radius_radians,
            return_distance=True,
            sort_results=True,
        )

        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        distance_parts: list[np.ndarray] = []

        for offset, (
            neighbor_indices,
            neighbor_distances,
        ) in enumerate(
            zip(neighbors, distances)
        ):
            seed_index = seed_start + offset

            # Keep every unordered pair exactly once.
            keep = neighbor_indices > seed_index

            if not keep.any():
                continue

            selected = neighbor_indices[keep]

            left_parts.append(
                np.full(
                    len(selected),
                    seed_index,
                    dtype=np.int64,
                )
            )

            right_parts.append(
                selected.astype(np.int64)
            )

            distance_parts.append(
                neighbor_distances[keep]
                * EARTH_RADIUS_FT
            )

        if left_parts:
            left = np.concatenate(left_parts)

            right = np.concatenate(right_parts)

            distance = np.concatenate(
                distance_parts
            ).astype(np.float32)

            correlations = _pair_correlations(
                normalized=normalized,
                left=left,
                right=right,
            )

            correlations, undefined_count = (
                _clean_correlations(
                    correlations=correlations,
                    division=division,
                    part_number=part_number,
                )
            )

            pair_df = pl.DataFrame(
                {
                    "division": (
                        [division] * len(left)
                    ),
                    "meter_1": (
                        meter_ids[left].tolist()
                    ),
                    "meter_2": (
                        meter_ids[right].tolist()
                    ),
                    "distance_ft": distance,
                    "correlation": correlations,
                },
                schema=PAIR_SCHEMA,
            )

            # NaN means mathematically undefined.
            # Null preserves this distinction from
            # zero correlation.
            pair_df = pair_df.with_columns(
                pl.col("correlation")
                .fill_nan(None)
            )

            if undefined_count:
                print(
                    f"[{division}] Part {part_number}: "
                    f"{undefined_count:,} undefined correlations "
                    "saved as null.",
                    flush=True,
                )

        else:
            pair_df = pl.DataFrame(
                schema=PAIR_SCHEMA
            )

        _save_partition(
            pair_df=pair_df,
            output_path=output_path,
        )

        print(
            f"[{division}] Saved part "
            f"{part_number}/{partition_count}: "
            f"{pair_df.height:,} pairs.",
            flush=True,
        )

    return output_paths