from __future__ import annotations

import json

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from mtx.data.extract_ami_data import (
    get_adam_validated_data,
)

# ======================================================
# PATHS
# ======================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "adam"
)

CACHE_FILE = (
    CACHE_DIR
    / "adam_validated.parquet"
)

METADATA_FILE = (
    CACHE_DIR
    / "adam_validated_metadata.json"
)

# ======================================================
# CACHE DATE
# ======================================================

def cache_date():

    # ----------------------------------------------
    # Metadata exists
    # ----------------------------------------------

    if METADATA_FILE.exists():

        try:

            metadata = json.loads(
                METADATA_FILE.read_text(
                    encoding="utf-8"
                )
            )

            return datetime.fromisoformat(
                metadata["created_at"]
            ).date()

        except Exception:
            pass

    # ----------------------------------------------
    # Cache exists but metadata does not
    # ----------------------------------------------

    if CACHE_FILE.exists():

        return datetime.fromtimestamp(
            CACHE_FILE.stat().st_mtime
        ).date()

    return None


# ======================================================
# WRITE METADATA
# ======================================================

def write_metadata(
    row_count: int,
):
    metadata = {
        "created_at": datetime.now().isoformat(),
        "row_count": row_count,
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )


# ======================================================
# LOAD OR REFRESH
# ======================================================

def load_or_refresh_adam_data(
    force_refresh: bool = False,
) -> pd.DataFrame:

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    today = date.today()

    # ==================================================
    # 1. CACHE EXISTS AND IS FROM TODAY
    # ==================================================

    if (
        not force_refresh
        and CACHE_FILE.exists()
        and cache_date() == today
    ):

        print(
            "Using today's Adam validated-data cache."
        )

        adam_df = pd.read_parquet(
            CACHE_FILE
        )

        # Metadata may not exist yet.
        # Create it without refreshing data.

        if not METADATA_FILE.exists():

            write_metadata(
                row_count=len(adam_df)
            )

            print(
                "Adam metadata created."
            )

        return adam_df

    # ==================================================
    # 2. CACHE EXISTS BUT IS OLD
    # ==================================================

    elif (
        not force_refresh
        and CACHE_FILE.exists()
    ):

        print(
            "Adam cache is from a previous day."
        )

        print(
            "Refreshing Adam validated data..."
        )

    # ==================================================
    # 3. NO CACHE
    # ==================================================

    else:

        print(
            "No current Adam cache found."
        )

        print(
            "Loading Adam validated data..."
        )

    # ==================================================
    # QUERY CURRENT DATA
    # ==================================================

    adam_df = get_adam_validated_data()

    # ==================================================
    # SAVE CACHE
    # ==================================================

    adam_df.to_parquet(
        CACHE_FILE,
        index=False,
    )

    write_metadata(
        row_count=len(adam_df)
    )

    print(
        f"Cached "
        f"{len(adam_df):,} "
        f"Adam validated rows."
    )

    return adam_df