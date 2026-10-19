"""Build weekly voltage Parquet files by division."""

from __future__ import annotations

import gc
import os
import re
from pathlib import Path
from typing import Iterator

import pandas as pd
import polars as pl


def _valid_parquet(path: Path) -> bool:
    """Return whether a weekly-voltage Parquet is usable."""

    if not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        schema = pl.read_parquet_schema(path)
    except Exception:
        return False

    return "meter_number" in schema


def _chunks(
    values: list[str],
    size: int | None,
) -> Iterator[list[str]]:
    """Split meter identifiers into partitions."""

    if size is None:
        yield values
        return

    if size <= 0:
        raise ValueError("Partition size must be positive.")

    for start in range(0, len(values), size):
        yield values[start : start + size]


def _build_part(
    division: str,
    meters: set[str],
    dates: pd.DatetimeIndex,
    input_directory: Path,
    output_path: Path,
) -> None:
    """Build one weekly-voltage partition."""

    weekly: pl.DataFrame | None = None

    for voltage_date in dates:
        daily_path = (
            input_directory
            / (
                "interpolated_voltage_data_ny_"
                f"{voltage_date:%m-%d-%Y}.pkl"
            )
        )

        if not daily_path.is_file():
            raise FileNotFoundError(
                f"Missing voltage file: {daily_path}"
            )

        print(
            f"[{division}] Reading {daily_path.name}",
            flush=True,
        )

        daily = pd.read_pickle(daily_path).drop(
            columns=["index"],
            errors="ignore",
        )

        if "meter_number" not in daily.columns:
            raise ValueError(
                f"{daily_path.name} has no meter_number."
            )

        daily["meter_number"] = (
            daily["meter_number"]
            .astype("string")
            .str.strip()
        )

        daily = daily[
            daily["meter_number"].isin(meters)
        ].copy()

        if daily["meter_number"].duplicated().any():
            duplicate_count = int(
                daily["meter_number"].duplicated().sum()
            )

            raise ValueError(
                f"{daily_path.name} contains "
                f"{duplicate_count:,} duplicate meters."
            )

        frame = (
            pl.from_pandas(daily)
            .with_columns(
                pl.col("meter_number").cast(pl.String),
                pl.exclude("meter_number").cast(
                    pl.Float32,
                    strict=False,
                ),
            )
        )

        if weekly is None:
            weekly = frame
        else:
            weekly = weekly.join(
                frame,
                on="meter_number",
                how="inner",
                validate="1:1",
            )

        del daily, frame
        gc.collect()

    if weekly is None or weekly.is_empty():
        raise ValueError(
            f"No weekly voltage data created for {division}."
        )

    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp.parquet"
    )

    weekly.write_parquet(
        temporary_path,
        compression="zstd",
    )

    if not _valid_parquet(temporary_path):
        temporary_path.unlink(missing_ok=True)

        raise ValueError(
            f"Invalid temporary output: {temporary_path}"
        )

    os.replace(
        temporary_path,
        output_path,
    )

    print(
        f"[{division}] Saved {output_path.name}: "
        f"{weekly.height:,} meters",
        flush=True,
    )


def build_weekly_voltages(
    ami_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    input_directory: str | Path,
    output_directory: str | Path,
    excluded_divisions: tuple[str, ...] = (),
    partition_sizes: dict[str, int] | None = None,
    overwrite: bool = False,
) -> dict[str, list[Path]]:
    """Create missing weekly files and reuse valid files."""

    input_directory = Path(input_directory)
    output_directory = Path(output_directory)

    if not input_directory.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_directory}"
        )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    partition_sizes = partition_sizes or {}

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    )

    if dates.empty:
        raise ValueError(
            "The weekly-voltage date range is empty."
        )

    ami = ami_df.rename(
        columns={"MeterID": "meter_number"}
    ).copy()

    required_columns = {
        "meter_number",
        "Division",
        "LATITUDE",
    }

    missing_columns = required_columns - set(ami.columns)

    if missing_columns:
        raise ValueError(
            "AMI data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    ami = ami.dropna(
        subset=[
            "meter_number",
            "Division",
            "LATITUDE",
        ]
    ).copy()

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
        ~ami["Division"].isin(excluded_divisions)
    ].copy()

    ami = ami.drop_duplicates(
        subset=["meter_number"]
    )

    date_label = (
        f"{dates[0]:%Y%m%d}_"
        f"{dates[-1]:%Y%m%d}"
    )

    outputs: dict[str, list[Path]] = {}

    for division in sorted(ami["Division"].unique()):
        meters = sorted(
            ami.loc[
                ami["Division"] == division,
                "meter_number",
            ].tolist()
        )

        if not meters:
            continue

        part_size = partition_sizes.get(division)

        meter_parts = list(
            _chunks(
                meters,
                part_size,
            )
        )

        outputs[division] = []

        safe_division = re.sub(
            r"[^A-Za-z0-9_-]+",
            "_",
            division,
        )

        for part_number, part_meters in enumerate(
            meter_parts,
            start=1,
        ):
            division_directory = (
                output_directory
                / safe_division
            )

            division_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                division_directory
                / f"part_{part_number:04d}.parquet"
            )

            outputs[division].append(output_path)

            if (
                not overwrite
                and _valid_parquet(output_path)
            ):
                print(
                    f"[{division}] Using existing: "
                    f"{output_path.name}",
                    flush=True,
                )

                continue

            print(
                f"\n[{division}] Building part "
                f"{part_number}/{len(meter_parts)} "
                f"({len(part_meters):,} AMI meters)",
                flush=True,
            )

            _build_part(
                division=division,
                meters=set(part_meters),
                dates=dates,
                input_directory=input_directory,
                output_path=output_path,
            )

    return outputs