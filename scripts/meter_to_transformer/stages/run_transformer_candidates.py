# src/mtx/processing/transformer_candidates.py

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from mtx.run_config import RunConfig


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CANDIDATE_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "candidates"
)

EARTH_RADIUS_FT = 20_902_231


# ============================================================
# SCHEMAS
# ============================================================

PAIR_COLUMNS = [
    "division",
    "meter_number",
    "candidate_transformer",
    "distance_ft",
]

METER_STATUS_COLUMNS = [
    "division",
    "meter_number",
    "latitude",
    "longitude",
    "candidate_count",
    "has_candidates",
    "status",
    "search_radius_ft",
]

TRANSFORMER_STATUS_COLUMNS = [
    "division",
    "transformer_floc",
    "latitude",
    "longitude",
    "candidate_meter_count",
    "has_candidate_meters",
    "status",
    "search_radius_ft",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_ami(
    ami_df: pd.DataFrame,
) -> pd.DataFrame:
    """Standardize AMI columns required for candidate generation."""

    df = ami_df.copy()

    rename_map = {}

    if (
        "MeterID" in df.columns
        and "meter_number" not in df.columns
    ):
        rename_map["MeterID"] = "meter_number"

    if (
        "Division" in df.columns
        and "division" not in df.columns
    ):
        rename_map["Division"] = "division"

    if (
        "LATITUDE" in df.columns
        and "latitude" not in df.columns
    ):
        rename_map["LATITUDE"] = "latitude"

    if (
        "LONGITUDE" in df.columns
        and "longitude" not in df.columns
    ):
        rename_map["LONGITUDE"] = "longitude"

    df = df.rename(
        columns=rename_map
    )

    required = {
        "meter_number",
        "division",
        "latitude",
        "longitude",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            "AMI missing required candidate columns: "
            f"{sorted(missing)}"
        )

    df = df[
        [
            "meter_number",
            "division",
            "latitude",
            "longitude",
        ]
    ].copy()

    df["meter_number"] = (
        df["meter_number"]
        .astype("string")
        .str.strip()
    )

    df["division"] = (
        df["division"]
        .astype("string")
        .str.strip()
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    if df["meter_number"].duplicated().any():
        raise ValueError(
            "AMI contains duplicate meter_number values."
        )

    return df


def normalize_transformers(
    transformer_df: pd.DataFrame,
    transformer_division_col: str = "Division",
) -> pd.DataFrame:
    """Standardize transformer columns required for BallTree."""

    df = transformer_df.copy()

    rename_map = {}

    if (
        "FuncLoc" in df.columns
        and "transformer_floc" not in df.columns
    ):
        rename_map["FuncLoc"] = "transformer_floc"

    if (
        transformer_division_col in df.columns
        and "division" not in df.columns
    ):
        rename_map[transformer_division_col] = "division"

    if (
        "Location_Y" in df.columns
        and "latitude" not in df.columns
    ):
        rename_map["Location_Y"] = "latitude"

    if (
        "Location_X" in df.columns
        and "longitude" not in df.columns
    ):
        rename_map["Location_X"] = "longitude"

    df = df.rename(
        columns=rename_map
    )

    required = {
        "transformer_floc",
        "division",
        "latitude",
        "longitude",
    }

    missing = required - set(df.columns)

    if missing:
        raise KeyError(
            "Transformer data missing required candidate columns: "
            f"{sorted(missing)}"
        )

    df = df[
        [
            "transformer_floc",
            "division",
            "latitude",
            "longitude",
        ]
    ].copy()

    df["transformer_floc"] = (
        df["transformer_floc"]
        .astype("string")
        .str.strip()
    )

    df["division"] = (
        df["division"]
        .astype("string")
        .str.strip()
    )

    df["latitude"] = pd.to_numeric(
        df["latitude"],
        errors="coerce",
    )

    df["longitude"] = pd.to_numeric(
        df["longitude"],
        errors="coerce",
    )

    # Candidate matching is by transformer FLOC.
    df = df.drop_duplicates(
        subset=["transformer_floc"],
        keep="first",
    )

    return df


def valid_coordinates(
    df: pd.DataFrame,
) -> pd.Series:
    """Return True for rows containing valid latitude and longitude values."""

    return (
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    )


def save_parquet(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """Safely write Parquet through a temporary file."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_name(
        f".{path.stem}.{uuid4().hex}.tmp.parquet"
    )

    try:
        df.to_parquet(
            temp,
            index=False,
            compression="zstd",
        )

        # Confirm that the temporary Parquet file can be read.
        check = pd.read_parquet(
            temp
        )

        if list(check.columns) != list(df.columns):
            raise OSError(
                f"Parquet validation failed: {temp}"
            )

        if len(check) != len(df):
            raise OSError(
                "Parquet row-count validation failed: "
                f"expected={len(df):,}, "
                f"actual={len(check):,}, "
                f"path={temp}"
            )

        temp.replace(
            path
        )

    finally:
        temp.unlink(
            missing_ok=True
        )


# ============================================================
# BALLTREE CANDIDATE SEARCH
# ============================================================

def find_transformer_candidates(
    meters: pd.DataFrame,
    transformers: pd.DataFrame,
    *,
    division: str,
    radius_ft: float,
) -> pd.DataFrame:
    """
    Find every transformer within radius_ft of every meter.

    Important
    ---------
    This function has no dependency on ITRON or meter grouping.

    Candidate membership depends only on:

    - Meter coordinates
    - Transformer coordinates
    - Division
    - Search radius
    """

    valid_meters = meters.loc[
        valid_coordinates(meters)
    ].copy()

    valid_transformers = transformers.loc[
        valid_coordinates(transformers)
    ].copy()

    if (
        valid_meters.empty
        or valid_transformers.empty
    ):
        return pd.DataFrame(
            columns=PAIR_COLUMNS
        )

    transformer_coordinates = np.radians(
        valid_transformers[
            ["latitude", "longitude"]
        ].to_numpy()
    )

    meter_coordinates = np.radians(
        valid_meters[
            ["latitude", "longitude"]
        ].to_numpy()
    )

    tree = BallTree(
        transformer_coordinates,
        metric="haversine",
    )

    (
        candidate_indices,
        candidate_distances,
    ) = tree.query_radius(
        meter_coordinates,
        r=radius_ft / EARTH_RADIUS_FT,
        return_distance=True,
        sort_results=True,
    )

    transformer_ids = (
        valid_transformers[
            "transformer_floc"
        ].to_numpy()
    )

    meter_ids = (
        valid_meters[
            "meter_number"
        ].to_numpy()
    )

    rows = []

    for meter_idx, (
        tx_indices,
        distances,
    ) in enumerate(
        zip(
            candidate_indices,
            candidate_distances,
        )
    ):
        meter_number = meter_ids[
            meter_idx
        ]

        for tx_idx, distance_rad in zip(
            tx_indices,
            distances,
        ):
            rows.append(
                {
                    "division": division,
                    "meter_number": meter_number,
                    "candidate_transformer": (
                        transformer_ids[
                            int(tx_idx)
                        ]
                    ),
                    "distance_ft": float(
                        distance_rad
                        * EARTH_RADIUS_FT
                    ),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=PAIR_COLUMNS
        )

    result = pd.DataFrame(
        rows
    )

    result = (
        result
        .sort_values(
            [
                "meter_number",
                "distance_ft",
                "candidate_transformer",
            ]
        )
        .drop_duplicates(
            [
                "meter_number",
                "candidate_transformer",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# ONE DIVISION
# ============================================================

def build_division(
    division: str,
    ami: pd.DataFrame,
    transformers: pd.DataFrame,
    radius_ft: float,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build candidate and status outputs for one division."""

    meters = (
        ami.loc[
            ami["division"] == division
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    tx = (
        transformers.loc[
            transformers["division"] == division
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Spatial search
    # --------------------------------------------------------

    pairs = find_transformer_candidates(
        meters,
        tx,
        division=division,
        radius_ft=radius_ft,
    )

    # --------------------------------------------------------
    # Meter candidate status
    # --------------------------------------------------------

    candidate_counts = (
        pairs
        .groupby(
            "meter_number"
        )["candidate_transformer"]
        .nunique()
    )

    meters["candidate_count"] = (
        meters["meter_number"]
        .map(candidate_counts)
        .fillna(0)
        .astype("int64")
    )

    meters["has_candidates"] = (
        meters["candidate_count"] > 0
    )

    meters["status"] = (
        "NO_TRANSFORMER_WITHIN_RADIUS"
    )

    meters.loc[
        ~valid_coordinates(meters),
        "status",
    ] = "INVALID_COORDINATES"

    valid_transformer_rows = tx.loc[
        valid_coordinates(tx)
    ]

    if valid_transformer_rows.empty:
        meters.loc[
            valid_coordinates(meters),
            "status",
        ] = "NO_VALID_TRANSFORMERS"

    meters.loc[
        meters["has_candidates"],
        "status",
    ] = "HAS_CANDIDATES"

    meters["search_radius_ft"] = (
        radius_ft
    )

    meter_status = meters[
        METER_STATUS_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # Transformer candidate status
    # --------------------------------------------------------

    transformer_counts = (
        pairs
        .groupby(
            "candidate_transformer"
        )["meter_number"]
        .nunique()
    )

    tx["candidate_meter_count"] = (
        tx["transformer_floc"]
        .map(transformer_counts)
        .fillna(0)
        .astype("int64")
    )

    tx["has_candidate_meters"] = (
        tx["candidate_meter_count"] > 0
    )

    tx["status"] = (
        "NO_METERS_WITHIN_RADIUS"
    )

    tx.loc[
        ~valid_coordinates(tx),
        "status",
    ] = "INVALID_COORDINATES"

    valid_meter_rows = meters.loc[
        valid_coordinates(meters)
    ]

    if valid_meter_rows.empty:
        tx.loc[
            valid_coordinates(tx),
            "status",
        ] = "NO_VALID_METERS"

    tx.loc[
        tx["has_candidate_meters"],
        "status",
    ] = "HAS_CANDIDATE_METERS"

    tx["search_radius_ft"] = (
        radius_ft
    )

    transformer_status = tx[
        TRANSFORMER_STATUS_COLUMNS
    ].copy()

    return (
        pairs,
        meter_status,
        transformer_status,
    )


# ============================================================
# DAILY CANDIDATE CACHE
# ============================================================

def run_transformer_candidate_stage(
    config: RunConfig,
    ami_df: pd.DataFrame,
    transformer_df: pd.DataFrame,
    *,
    snapshot_date=None,
    cache_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Generate spatial transformer candidates for every AMI meter.

    Grouping is deliberately not an input.

    Cache structure
    ---------------

    data/cache/candidates/
        YYYY-MM-DD/
            pairs/
            meters/
            transformers/
            no_candidates/
            _complete.json

    If the requested date's completed cache already exists,
    it is reused.
    """

    day = date.fromisoformat(
        str(
            snapshot_date
            or date.today()
        )[:10]
    ).isoformat()

    root = Path(
        cache_dir
        or CANDIDATE_CACHE_DIR
    )

    day_dir = (
        root
        / day
    )

    radius_ft = float(
        config.transformer_candidate_radius_ft
    )

    if (
        not np.isfinite(radius_ft)
        or radius_ft <= 0
    ):
        raise ValueError(
            "transformer_candidate_radius_ft "
            "must be positive."
        )

    complete_file = (
        day_dir
        / "_complete.json"
    )

    # --------------------------------------------------------
    # COMPLETED CACHE ALREADY EXISTS
    # --------------------------------------------------------

    if complete_file.exists():
        print(
            "Candidate cache already exists "
            f"for {day}."
        )

        print(
            "Reusing the completed candidate cache. "
            "No BallTree search is required."
        )

        with complete_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            manifest = json.load(
                file
            )

        output_paths = {
            division: (
                day_dir
                / "pairs"
                / f"{division}.parquet"
            )
            for division
            in manifest["meter_divisions"]
        }

        missing_outputs = [
            path
            for path in output_paths.values()
            if not path.is_file()
        ]

        if missing_outputs:
            raise FileNotFoundError(
                "Candidate cache manifest exists, but candidate "
                f"files are missing: {missing_outputs}"
            )

        return output_paths

    # --------------------------------------------------------
    # NORMALIZE INPUTS
    # --------------------------------------------------------

    ami = normalize_ami(
        ami_df
    )

    transformers = normalize_transformers(
        transformer_df,
        transformer_division_col=(
            config.transformer_division_col
        ),
    )

    meter_divisions = sorted(
        ami["division"]
        .dropna()
        .unique()
        .tolist()
    )

    transformer_divisions = set(
        transformers["division"]
        .dropna()
        .unique()
        .tolist()
    )

    all_divisions = sorted(
        set(meter_divisions)
        | transformer_divisions
    )

    manifest = {
        "version": 1,
        "date": day,
        "snapshot_id": uuid4().hex,
        "created_at": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "radius_ft": radius_ft,
        "meter_divisions": meter_divisions,
        "divisions": all_divisions,
        "counts": {},
    }

    output_paths: dict[str, Path] = {}

    total_meters = 0
    total_pairs = 0
    total_no_candidates = 0

    # --------------------------------------------------------
    # PROCESS BY DIVISION
    # --------------------------------------------------------

    for division in meter_divisions:
        print(
            f"\nCandidate search: {division}"
        )

        (
            pairs,
            meter_status,
            transformer_status,
        ) = build_division(
            division,
            ami,
            transformers,
            radius_ft,
        )

        pairs_path = (
            day_dir
            / "pairs"
            / f"{division}.parquet"
        )

        meters_path = (
            day_dir
            / "meters"
            / f"{division}.parquet"
        )

        transformers_path = (
            day_dir
            / "transformers"
            / f"{division}.parquet"
        )

        no_candidates_path = (
            day_dir
            / "no_candidates"
            / f"{division}.parquet"
        )

        no_candidates = (
            meter_status.loc[
                ~meter_status["has_candidates"]
            ]
            .copy()
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        if not pairs_path.exists():
            save_parquet(
                pairs,
                pairs_path,
            )
        

        if not meters_path.exists():
            save_parquet(
                meter_status,
                meters_path,
            )

        if not transformers_path.exists():
            save_parquet(
                transformer_status,
                transformers_path,
            )

        if not no_candidates_path.exists():
            save_parquet(
                no_candidates,
                no_candidates_path,
            )

        output_paths[
            division
        ] = pairs_path

        meters_with_candidates = int(
            meter_status[
                "has_candidates"
            ].sum()
        )

        manifest["counts"][division] = {
            "meters": len(
                meter_status
            ),
            "meters_with_candidates": (
                meters_with_candidates
            ),
            "meters_without_candidates": len(
                no_candidates
            ),
            "transformers": len(
                transformer_status
            ),
            "candidate_pairs": len(
                pairs
            ),
        }

        total_meters += len(
            meter_status
        )

        total_pairs += len(
            pairs
        )

        total_no_candidates += len(
            no_candidates
        )

        print(
            f"  Meters: {len(meter_status):,}"
        )

        print(
            f"  Candidate pairs: {len(pairs):,}"
        )

        print(
            "  Meters with candidates: "
            f"{meters_with_candidates:,}"
        )

        print(
            "  Meters without candidates: "
            f"{len(no_candidates):,}"
        )

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if total_meters != len(ami):
        raise ValueError(
            "Candidate meter population does not reconcile "
            "with AMI: "
            f"candidate meters={total_meters:,}, "
            f"AMI meters={len(ami):,}."
        )

    manifest["totals"] = {
        "meters": total_meters,
        "candidate_pairs": total_pairs,
        "meters_without_candidates": (
            total_no_candidates
        ),
    }

    # --------------------------------------------------------
    # MARK DAILY SNAPSHOT COMPLETE
    # --------------------------------------------------------

    day_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_manifest = (
        day_dir
        / f"._complete.{uuid4().hex}.tmp"
    )

    try:
        with temp_manifest.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                manifest,
                file,
                indent=2,
            )

        temp_manifest.replace(
            complete_file
        )

    finally:
        temp_manifest.unlink(
            missing_ok=True
        )

    print(
        "\nCandidate cache complete."
    )

    print(
        f"Date: {day}"
    )

    print(
        f"Meters searched: {total_meters:,}"
    )

    print(
        f"Candidate pairs: {total_pairs:,}"
    )

    print(
        "Meters with no candidates: "
        f"{total_no_candidates:,}"
    )

    print(
        f"Saved under:\n{day_dir}"
    )

    return output_paths