"""Create or reuse meter-correlation partitions."""

from pathlib import Path

import polars as pl

from mtx.clustering.meter_pairs import (
    compute_meter_pair_partitions,
)
from mtx.run_config import RunConfig
import pandas as pd

def run_meter_correlations(
    config: RunConfig,
    weekly_files: dict[str, list[Path]],
    ami_df: pd.DataFrame,
    overwrite: bool = False,
) -> tuple[
    dict[str, list[Path]],
    pl.LazyFrame,
]:
    if not weekly_files:
        raise ValueError(
            "No weekly-voltage files were provided."
        )

    print(
        "Extracting AMI data...",
        flush=True,
    )


    print(
        f"AMI extraction complete: {len(ami_df):,} records.",
        flush=True,
    )

    pair_files: dict[str, list[Path]] = {}
    failures: dict[str, str] = {}

    for division, voltage_paths in sorted(
        weekly_files.items()
    ):
        output_directory = (
            config.pair_directory / division
        )

        try:
            pair_files[division] = (
                compute_meter_pair_partitions(
                    ami_df=ami_df,
                    weekly_voltage_paths=voltage_paths,
                    division=division,
                    output_directory=output_directory,
                    start_date=config.correlation_start,
                    end_date=config.correlation_end,
                    radius_ft=config.meter_correlation_radius,
                    correlation_method=config.correlation_method,
                    seed_batch_size=config.seed_batch_size,
                    overwrite=overwrite,
                )
            )

        except Exception as exc:
            failures[division] = str(exc)

            print(
                f"[{division}] FAILED: {exc}",
                flush=True,
            )

    if failures:
        failed = ", ".join(failures)

        raise RuntimeError(
            f"Correlation processing failed for: {failed}. "
            "Completed partitions were preserved."
        )

    all_paths = [
        path
        for division_paths in pair_files.values()
        for path in division_paths
    ]

    if not all_paths:
        raise FileNotFoundError(
            "No meter-pair partitions were created."
        )

    all_pairs = pl.scan_parquet(all_paths)

    return pair_files, all_pairs