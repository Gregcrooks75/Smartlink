from pathlib import Path
from datetime import date

import json
import os
import uuid

import pandas as pd
import polars as pl

from mtx.db.db_GIS import get_devices_with_structures


# ======================================================
# PROJECT PATHS
# ======================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "transformers"
)

CACHE_FILE = (
    CACHE_DIR
    / "ny_transformers.parquet"
)

METADATA_FILE = (
    CACHE_DIR
    / "ny_transformers_metadata.json"
)


# ======================================================
# CACHE VALIDATION
# ======================================================

def cache_is_current() -> bool:
    """
    Return True when the transformer cache
    was refreshed today.
    """

    if not CACHE_FILE.exists():
        return False

    if not METADATA_FILE.exists():
        return False

    try:
        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

    except (
        OSError,
        json.JSONDecodeError,
    ):
        return False

    return (
        metadata.get("cache_date")
        == date.today().isoformat()
    )


# ======================================================
# TRANSFORMER VALIDATION
# ======================================================

def validate_transformers(
    df: pd.DataFrame,
) -> None:
    """
    Basic validation before allowing transformer
    data to become the pipeline cache.
    """

    if df.empty:
        raise ValueError(
            "Transformer query returned no rows."
        )

    required_columns = {
        "Location_Y",
        "Location_X",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Transformer data missing required columns: "
            f"{missing}"
        )


# ======================================================
# ATOMIC SAVE CACHE
# ======================================================

def save_transformer_cache(
    df: pd.DataFrame,
) -> None:
    """
    Atomically save transformer data so that a
    failed write cannot destroy the existing cache.
    """

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_file = CACHE_DIR / (
        f"ny_transformers_"
        f"{uuid.uuid4().hex}"
        f".tmp.parquet"
    )

    try:

        # ------------------------------------------
        # Write temporary parquet
        # ------------------------------------------

        df.to_parquet(
            temp_file,
            index=False,
            compression="zstd",
        )

        # ------------------------------------------
        # Validate temporary parquet
        # ------------------------------------------

        check_df = pl.read_parquet(
            temp_file
        )

        if check_df.height != len(df):
            raise ValueError(
                "Transformer cache validation failed: "
                f"expected {len(df):,} rows, "
                f"found {check_df.height:,}."
            )

        # ------------------------------------------
        # Replace existing cache atomically
        # ------------------------------------------

        os.replace(
            temp_file,
            CACHE_FILE,
        )

        # ------------------------------------------
        # Save metadata
        # ------------------------------------------

        metadata = {
            "cache_date": (
                date.today().isoformat()
            ),
            "row_count": len(df),
            "column_count": len(
                df.columns
            ),
            "columns": (
                df.columns.tolist()
            ),
        }

        with open(
            METADATA_FILE,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                metadata,
                f,
                indent=4,
            )

        print(
            "[Transformers] Cache saved to:"
        )

        print(
            f" {CACHE_FILE}"
        )

    finally:

        if temp_file.exists():
            temp_file.unlink()


# ======================================================
# LOAD OR REFRESH
# ======================================================

def load_or_refresh_ny_transformers(
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load today's cached NY transformer data when
    possible.

    Otherwise query NYGIS and replace the transformer
    cache.
    """

    # --------------------------------------------------
    # USE TODAY'S CACHE
    # --------------------------------------------------

    if (
        not force_refresh
        and cache_is_current()
    ):

        print(
            "[Transformers] Reusing cached data:"
        )

        print(
            f" {CACHE_FILE}"
        )

        return pd.read_parquet(
            CACHE_FILE
        )

    # --------------------------------------------------
    # REFRESH FROM GIS
    # --------------------------------------------------

    print(
        "[Transformers] Refreshing transformer "
        "data from NYGIS..."
    )

    transformer_df = (
        get_devices_with_structures(
            server="nygisprd19",
            database="NYGIS",
            device="transformer",
        )
    )

    # Protect against GIS returning Polars.
    if isinstance(
        transformer_df,
        pl.DataFrame,
    ):
        transformer_df = (
            transformer_df.to_pandas()
        )

    # --------------------------------------------------
    # VALIDATE
    # --------------------------------------------------

    validate_transformers(
        transformer_df
    )

    # --------------------------------------------------
    # SAVE
    # --------------------------------------------------

    save_transformer_cache(
        transformer_df
    )

    print(
        f"[Transformers] Cached "
        f"{len(transformer_df):,} transformers."
    )

    return transformer_df