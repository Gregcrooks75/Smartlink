"""Create or reuse the daily NY AMI snapshot."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl

from mtx.config import AMI_CACHE_DIRECTORY
from mtx.db.db_AMI import get_ny_ami_data


AMI_FILE = AMI_CACHE_DIRECTORY / "ny_ami.parquet"

AMI_METADATA_FILE = (
    AMI_CACHE_DIRECTORY / "ny_ami.metadata.json"
)

REQUIRED_AMI_COLUMNS = {
    "MeterID",
    "Division",
    "LATITUDE",
    "LONGITUDE",
}


def _today() -> str:
    """Return today's local date."""
    return datetime.now().astimezone().date().isoformat()


def _valid_ami_file(file_path: Path) -> bool:
    """Return whether the AMI Parquet is readable."""
    if (
        not file_path.is_file()
        or file_path.stat().st_size == 0
    ):
        return False

    try:
        schema = pl.read_parquet_schema(file_path)
    except Exception:
        return False

    return REQUIRED_AMI_COLUMNS.issubset(schema)


def _cache_is_current() -> bool:
    """Return whether today's valid AMI snapshot exists."""
    if not _valid_ami_file(AMI_FILE):
        return False

    if not AMI_METADATA_FILE.is_file():
        return False

    try:
        metadata = json.loads(
            AMI_METADATA_FILE.read_text(
                encoding="utf-8",
            )
        )
    except (OSError, json.JSONDecodeError):
        return False

    return metadata.get("snapshot_date") == _today()


def _validate_ami_dataframe(
    ami_df: pd.DataFrame,
) -> None:
    """Validate AMI data before saving it."""
    missing_columns = (
        REQUIRED_AMI_COLUMNS - set(ami_df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Extracted AMI data is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if ami_df.empty:
        raise ValueError(
            "The AMI database extraction returned no records."
        )

    if ami_df["MeterID"].isna().any():
        raise ValueError(
            "The AMI extraction contains null MeterID values."
        )

    duplicate_meters = int(
        ami_df["MeterID"].duplicated().sum()
    )

    if duplicate_meters:
        raise ValueError(
            "The AMI extraction contains "
            f"{duplicate_meters:,} duplicate MeterID values."
        )


def _save_ami_cache(
    ami_df: pd.DataFrame,
) -> None:
    """Save the AMI snapshot and metadata atomically."""
    AMI_CACHE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_ami = AMI_FILE.with_name(
        "ny_ami.temporary.parquet"
    )

    temporary_metadata = AMI_METADATA_FILE.with_name(
        "ny_ami.metadata.temporary.json"
    )

    ami_df.to_parquet(
        temporary_ami,
        index=False,
    )

    if not _valid_ami_file(temporary_ami):
        temporary_ami.unlink(missing_ok=True)

        raise ValueError(
            f"Invalid temporary AMI file: {temporary_ami}"
        )

    metadata = {
        "snapshot_date": _today(),
        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),
        "row_count": len(ami_df),
        "column_count": len(ami_df.columns),
    }

    temporary_metadata.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    os.replace(
        temporary_ami,
        AMI_FILE,
    )

    os.replace(
        temporary_metadata,
        AMI_METADATA_FILE,
    )


def load_or_refresh_ny_ami(
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load today's AMI snapshot or refresh it from the database."""
    if not force_refresh and _cache_is_current():
        print(
            f"Using today's AMI snapshot: {AMI_FILE}",
            flush=True,
        )

        ami_df = pd.read_parquet(AMI_FILE)

        print(
            f"AMI snapshot loaded: {len(ami_df):,} records.",
            flush=True,
        )

        return ami_df

    print(
        "AMI snapshot is missing, invalid, or outdated.",
        flush=True,
    )

    print(
        "Extracting current AMI data from the database...",
        flush=True,
    )

    ami_df = get_ny_ami_data()

    _validate_ami_dataframe(ami_df)
    _save_ami_cache(ami_df)

    print(
        f"AMI snapshot refreshed: {len(ami_df):,} records.",
        flush=True,
    )

    print(
        f"Saved to: {AMI_FILE}",
        flush=True,
    )

    return ami_df