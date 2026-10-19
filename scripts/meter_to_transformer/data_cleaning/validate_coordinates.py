from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import geopandas as gpd
import pandas as pd
from datetime import date
from pathlib import Path

import pandas as pd





PROJECT_ROOT = Path(__file__).resolve().parents[3]

REJECTED_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "rejected"
)

REJECTED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
HISTORY_FILE = (
    REJECTED_DIR
    / "rejection_history.parquet"
)

import geopandas as gpd


def validate_coordinates(
    df,
    division_gdf,
):
    """
    Check whether each meter coordinate falls inside
    the polygon of its assigned Division.

    Invalid meters are returned separately.
    """

    # ==================================================
    # CREATE METER POINTS
    # ==================================================

    df = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(
            df["longitude"],
            df["latitude"],
        ),
        crs="EPSG:4326",
    )

    # ==================================================
    # DIVISION POLYGONS
    # ==================================================

    divisions = division_gdf[
        [
            "Division",
            "geometry",
        ]
    ].copy()

    divisions = divisions.to_crs(
        epsg=4326
    )

    # ==================================================
    # MATCH EACH METER TO ITS ASSIGNED DIVISION
    # ==================================================

    df = df.merge(
        divisions,
        on="Division",
        how="left",
        suffixes=(
            "_meter",
            "_division",
        ),
    )

    # ==================================================
    # CHECK WHETHER POINT FALLS INSIDE DIVISION POLYGON
    # ==================================================

    df["coordinate_valid"] = (
        df["geometry_meter"].within(
            df["geometry_division"]
        )
    )

    # ==================================================
    # SPLIT VALID / INVALID
    # ==================================================

    valid_data = (
        df[
            df["coordinate_valid"]
        ]
        .copy()
    )

    invalid_data = (
        df[
            ~df["coordinate_valid"]
        ]
        .copy()
    )

    # ==================================================
    # SUMMARY
    # ==================================================

    print(
        f"Valid meters: "
        f"{len(valid_data):,}"
    )

    print(
        f"Invalid meters: "
        f"{len(invalid_data):,}"
    )

    return (
        valid_data.reset_index(drop=True),
        invalid_data.reset_index(drop=True),
    )
    




def save_rejection_log(
    invalid_meters,
    invalid_transformers,
):
    """
    Save:
        1. Historical rejection IDs (Parquet)
        2. Daily review file (Excel)

    Skip if today's files already exist.
    """

    today = date.today().isoformat()

    excel_file = (
        REJECTED_DIR
        / f"rejections_{today}.xlsx"
    )

    # ==================================================
    # SKIP IF ALREADY RUN TODAY
    # ==================================================

    if (
        HISTORY_FILE.exists()
        and excel_file.exists()
    ):
        print(
            f"Rejection log already exists for {today}."
        )
        print(
            "Skipping rejection log creation."
        )
        return

    # ==================================================
    # TODAY'S HISTORY RECORDS
    # ==================================================

    meters = pd.DataFrame(
        {
            "rejection_date": today,
            "asset_type": "meter",
            "asset_id": invalid_meters[
                "meter_number"
            ].astype("string"),
        }
    )

    transformers = pd.DataFrame(
        {
            "rejection_date": today,
            "asset_type": "transformer",
            "asset_id": invalid_transformers[
                "transformer_floc"
            ].astype("string"),
        }
    )

    today_history = pd.concat(
        [
            meters,
            transformers,
        ],
        ignore_index=True,
    )

    today_history = today_history.drop_duplicates(
        subset=[
            "rejection_date",
            "asset_type",
            "asset_id",
        ]
    )

    # ==================================================
    # UPDATE HISTORY
    # ==================================================

    if HISTORY_FILE.exists():

        history = pd.read_parquet(
            HISTORY_FILE
        )

        history = history[
            history["rejection_date"]
            != today
        ]

        history = pd.concat(
            [
                history,
                today_history,
            ],
            ignore_index=True,
        )

    else:

        history = today_history

    history.to_parquet(
        HISTORY_FILE,
        index=False,
        compression="zstd",
    )

    # ==================================================
    # WRITE EXCEL
    # ==================================================

    with pd.ExcelWriter(
        excel_file,
        engine="openpyxl",
    ) as writer:

        invalid_meters.to_excel(
            writer,
            sheet_name="Meters",
            index=False,
        )

        invalid_transformers.to_excel(
            writer,
            sheet_name="Transformers",
            index=False,
        )

    # ==================================================
    # SUMMARY
    # ==================================================

    print(
        f"{today}: "
        f"{len(meters):,} meters rejected | "
        f"{len(transformers):,} transformers rejected"
    )

    print(f"History: {HISTORY_FILE}")
    print(f"Today's Excel: {excel_file}")