"""Create or reuse weekly-voltage partitions."""

from pathlib import Path

from mtx.config import INTERPOLATED_VOLTAGE_DIRECTORY
from mtx.processing.weekly_voltages import (
    build_weekly_voltages,
)
from mtx.run_config import RunConfig
import pandas as pd

def normalize_weekly_files(
    weekly_files: dict[str, Path | list[Path]],
) -> dict[str, list[Path]]:
    """Map every division to a list of Parquet files."""

    normalized: dict[str, list[Path]] = {}

    for division, paths in weekly_files.items():
        if isinstance(paths, Path):
            normalized[division] = [paths]
        else:
            normalized[division] = [
                Path(path)
                for path in paths
            ]

    return normalized


def run_weekly_voltage_stage(
    config: RunConfig,
    ami_df: pd.DataFrame,
    overwrite: bool = False,
    
) -> dict[str, list[Path]]:
    """Create or reuse weekly voltage partitions."""

    print(
        "Extracting AMI data...",
        flush=True,
    )

    print(
        f"AMI extraction complete: "
        f"{len(ami_df):,} records.",
        flush=True,
    )

    divisions = (
        ami_df["Division"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
    )

    partition_sizes = {
        division: config.weekly_partition_size
        for division in divisions
    }

    weekly_files = build_weekly_voltages(
        ami_df=ami_df,
        start_date=config.week_start,
        end_date=config.week_end,
        input_directory=INTERPOLATED_VOLTAGE_DIRECTORY,
        output_directory=config.weekly_directory,
        excluded_divisions=(),
        partition_sizes=partition_sizes,
        overwrite=overwrite,
    )

    weekly_files = normalize_weekly_files(
        weekly_files
    )

    print(
        "\nWeekly-voltage files ready:",
        flush=True,
    )

    for division, paths in weekly_files.items():
        print(
            f"  {division}: "
            f"{len(paths)} partition(s)",
            flush=True,
        )

    return weekly_files