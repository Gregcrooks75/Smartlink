# src/mtx/history/candidate_feature_history.py

"""
Track daily changes in candidate-feature snapshots.

This history determines which meter-transformer pairs require spatial
tile-feature processing.

Tile actions
------------
COMPUTE:
    A new candidate pair was added.

DROP:
    A candidate pair was removed.

RECOMPUTE:
    A spatial dependency changed, such as meter or transformer
    coordinates.

REUSE:
    Candidate features changed, but the spatial dependencies did not.
    Existing tile features can normally be reused.
"""

from __future__ import annotations

import json
import re

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from pandas.api.types import (
    is_float_dtype,
    is_numeric_dtype,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

FEATURE_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "candidate_features"
)

HISTORY_DIR = (
    FEATURE_CACHE_DIR
    / "history"
)

ADDED_DIR = (
    HISTORY_DIR
    / "added_pairs"
)

REMOVED_DIR = (
    HISTORY_DIR
    / "removed_pairs"
)

CHANGED_DIR = (
    HISTORY_DIR
    / "changed_rows"
)

SUMMARY_FILE = (
    HISTORY_DIR
    / "candidate_feature_daily_summary.parquet"
)


# ============================================================
# CONFIGURATION
# ============================================================

KEY_COLUMNS = [
    "meter_number",
    "candidate_transformer",
]

# These candidate-feature columns influence the later
# tile/spatial feature calculation.
#
# If these columns do not change, existing tile features can
# normally be reused.
TILE_DEPENDENCY_COLUMNS = [
    "meter_latitude",
    "meter_longitude",
    "transformer_latitude",
    "transformer_longitude",
]

EXCLUDE_FROM_FEATURE_COMPARISON = {
    "division",
    *KEY_COLUMNS,
}

DATE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}$"
)

FLOAT_HASH_DECIMALS = 8


# ============================================================
# FILE HELPERS
# ============================================================

def ensure_directories() -> None:
    """Create the candidate-feature history directories."""

    for path in [
        ADDED_DIR,
        REMOVED_DIR,
        CHANGED_DIR,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_parquet(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """
    Write a temporary Parquet file, validate it, and then replace
    the destination file.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        ".tmp.parquet"
    )

    try:
        df.to_parquet(
            temp,
            index=False,
            compression="zstd",
        )

        validation_columns = [
            column
            for column in KEY_COLUMNS
            if column in df.columns
        ]

        if validation_columns:
            checked = pd.read_parquet(
                temp,
                columns=validation_columns,
            )
        else:
            checked = pd.read_parquet(
                temp
            )

        if len(checked) != len(df):
            raise OSError(
                "Temporary candidate-feature history file "
                "failed row-count validation: "
                f"expected={len(df):,}, "
                f"actual={len(checked):,}, "
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
# SNAPSHOT HELPERS
# ============================================================

def snapshot_dir(
    snapshot_date: date,
) -> Path:
    """Return the feature-cache directory for one snapshot date."""

    return (
        FEATURE_CACHE_DIR
        / snapshot_date.isoformat()
    )


def completed_dates() -> list:
    """
    Return dates with completed candidate-feature snapshots.

    A feature snapshot is complete only when its directory contains
    _complete.json.
    """

    if not FEATURE_CACHE_DIR.exists():
        return []

    dates: list[date] = []

    for folder in FEATURE_CACHE_DIR.iterdir():

        if not folder.is_dir():
            continue

        if not DATE_PATTERN.match(
            folder.name
        ):
            continue

        if not (
            folder
            / "_complete.json"
        ).exists():
            continue

        try:
            dates.append(
                date.fromisoformat(
                    folder.name
                )
            )

        except ValueError:
            continue

    return sorted(
        dates
    )


def previous_snapshot(
    current_date: date,
) -> date | None:
    """
    Return the most recent completed snapshot before current_date.
    """

    previous_dates = [
        snapshot_date
        for snapshot_date in completed_dates()
        if snapshot_date < current_date
    ]

    if not previous_dates:
        return None

    return max(
        previous_dates
    )


def snapshot_divisions(
    snapshot_date: date,
) -> set:
    """Return the divisions saved in one feature snapshot."""

    root = snapshot_dir(
        snapshot_date
    )

    if not root.exists():
        return set()

    return {
        file.stem
        for file in root.glob(
            "*.parquet"
        )
    }


def read_division(
    snapshot_date: date,
    division: str,
) -> pd.DataFrame:
    """Read and standardize one division's feature snapshot."""

    path = (
        snapshot_dir(snapshot_date)
        / f"{division}.parquet"
    )

    if not path.exists():
        return pd.DataFrame(
            columns=KEY_COLUMNS
        )

    df = pd.read_parquet(
        path
    )

    for column in KEY_COLUMNS:

        if column not in df.columns:
            raise ValueError(
                f"{path} is missing required column "
                f"{column!r}."
            )

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    duplicate_count = int(
        df.duplicated(
            KEY_COLUMNS
        ).sum()
    )

    if duplicate_count:
        raise ValueError(
            f"{path} contains {duplicate_count:,} duplicate "
            "meter-transformer pairs."
        )

    return (
        df
        .drop_duplicates(
            subset=KEY_COLUMNS,
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# HASHING
# ============================================================

def prepare_hash_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Normalize columns before hashing.

    Floating-point values are rounded so extremely small numerical
    differences do not appear as business changes.
    """

    out = df[
        columns
    ].copy()

    for column in columns:

        if is_float_dtype(
            out[column]
        ):
            out[column] = (
                out[column]
                .round(
                    FLOAT_HASH_DECIMALS
                )
            )

        elif not is_numeric_dtype(
            out[column]
        ):
            out[column] = (
                out[column]
                .astype("string")
                .fillna("<NULL>")
            )

    return out


def row_hash(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.Series:
    """Create a deterministic hash for selected columns in each row."""

    if not columns:
        return pd.Series(
            0,
            index=df.index,
            dtype="UInt64",
        )

    prepared = prepare_hash_columns(
        df,
        columns,
    )

    values = pd.util.hash_pandas_object(
        prepared,
        index=False,
    )

    return pd.Series(
        values,
        index=df.index,
        dtype="UInt64",
    )


# ============================================================
# COMPARISON HELPERS
# ============================================================

def columns_equal(
    left: pd.Series,
    right: pd.Series,
) -> np.ndarray:
    """Compare two columns while handling null and numeric values."""

    if (
        is_numeric_dtype(left)
        and is_numeric_dtype(right)
    ):
        left_numeric = pd.to_numeric(
            left,
            errors="coerce",
        )

        right_numeric = pd.to_numeric(
            right,
            errors="coerce",
        )

        return np.isclose(
            left_numeric,
            right_numeric,
            equal_nan=True,
            rtol=1e-7,
            atol=1e-8,
        )

    left_text = (
        left
        .astype("string")
        .fillna("<NULL>")
    )

    right_text = (
        right
        .astype("string")
        .fillna("<NULL>")
    )

    return (
        left_text
        .eq(right_text)
        .to_numpy()
    )


def identify_changed_columns(
    current_df: pd.DataFrame,
    previous_df: pd.DataFrame,
    changed_keys: pd.DataFrame,
    compare_columns: list[str],
) -> pd.DataFrame:
    """
    Identify the individual feature columns that changed for each
    modified candidate pair.
    """

    result_columns = [
        *KEY_COLUMNS,
        "changed_columns",
        "changed_column_count",
    ]

    if changed_keys.empty:
        return pd.DataFrame(
            columns=result_columns
        )

    current_changed = changed_keys.merge(
        current_df[
            KEY_COLUMNS
            + compare_columns
        ],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    previous_changed = changed_keys.merge(
        previous_df[
            KEY_COLUMNS
            + compare_columns
        ],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )

    changed_lists = [
        []
        for _ in range(
            len(changed_keys)
        )
    ]

    for column in compare_columns:

        equal = columns_equal(
            current_changed[column],
            previous_changed[column],
        )

        changed_positions = np.flatnonzero(
            ~equal
        )

        # This loop runs only over rows whose full feature hash
        # has already been identified as changed.
        for position in changed_positions:
            changed_lists[
                position
            ].append(
                column
            )

    result = changed_keys.copy()

    result["changed_columns"] = [
        "|".join(columns)
        for columns in changed_lists
    ]

    result["changed_column_count"] = [
        len(columns)
        for columns in changed_lists
    ]

    return result[
        result_columns
    ]


# ============================================================
# DIVISION COMPARISON
# ============================================================

def compare_division(
    division: str,
    current_date: date,
    previous_date: date,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    """
    Compare one division between two completed feature snapshots.
    """

    current = read_division(
        current_date,
        division,
    )

    previous = read_division(
        previous_date,
        division,
    )

    current_columns = set(
        current.columns
    )

    previous_columns = set(
        previous.columns
    )

    common_columns = (
        current_columns
        & previous_columns
    )

    compare_columns = sorted(
        common_columns
        - EXCLUDE_FROM_FEATURE_COMPARISON
    )

    schema_added = sorted(
        current_columns
        - previous_columns
    )

    schema_removed = sorted(
        previous_columns
        - current_columns
    )

    # --------------------------------------------------------
    # FULL FEATURE SIGNATURE
    # --------------------------------------------------------

    current_signature = current[
        KEY_COLUMNS
    ].copy()

    previous_signature = previous[
        KEY_COLUMNS
    ].copy()

    current_signature[
        "_feature_hash"
    ] = row_hash(
        current,
        compare_columns,
    )

    previous_signature[
        "_feature_hash"
    ] = row_hash(
        previous,
        compare_columns,
    )

    # --------------------------------------------------------
    # TILE DEPENDENCY SIGNATURE
    # --------------------------------------------------------

    tile_columns = [
        column
        for column in TILE_DEPENDENCY_COLUMNS
        if (
            column in current.columns
            and column in previous.columns
        )
    ]

    current_signature[
        "_tile_hash"
    ] = row_hash(
        current,
        tile_columns,
    )

    previous_signature[
        "_tile_hash"
    ] = row_hash(
        previous,
        tile_columns,
    )

    # --------------------------------------------------------
    # OUTER JOIN
    # --------------------------------------------------------

    comparison = current_signature.merge(
        previous_signature,
        on=KEY_COLUMNS,
        how="outer",
        suffixes=(
            "_current",
            "_previous",
        ),
        indicator=True,
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # ADDED PAIRS
    # --------------------------------------------------------

    added = (
        comparison.loc[
            comparison["_merge"].eq(
                "left_only"
            ),
            KEY_COLUMNS,
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    added.insert(
        0,
        "date",
        current_date,
    )

    added.insert(
        1,
        "division",
        division,
    )

    added["change_type"] = (
        "PAIR_ADDED"
    )

    added["tile_action"] = (
        "COMPUTE"
    )

    # --------------------------------------------------------
    # REMOVED PAIRS
    # --------------------------------------------------------

    removed = (
        comparison.loc[
            comparison["_merge"].eq(
                "right_only"
            ),
            KEY_COLUMNS,
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    removed.insert(
        0,
        "date",
        current_date,
    )

    removed.insert(
        1,
        "division",
        division,
    )

    removed["change_type"] = (
        "PAIR_REMOVED"
    )

    removed["tile_action"] = (
        "DROP"
    )

    # --------------------------------------------------------
    # MODIFIED FEATURE ROWS
    # --------------------------------------------------------

    both = comparison.loc[
        comparison["_merge"].eq(
            "both"
        )
    ].copy()

    changed_mask = (
        both["_feature_hash_current"]
        != both["_feature_hash_previous"]
    )

    changed_base = both.loc[
        changed_mask
    ].copy()

    changed_keys = (
        changed_base[
            KEY_COLUMNS
        ]
        .reset_index(
            drop=True
        )
    )

    changed = identify_changed_columns(
        current_df=current,
        previous_df=previous,
        changed_keys=changed_keys,
        compare_columns=compare_columns,
    )

    if not changed.empty:

        tile_lookup = changed_base[
            KEY_COLUMNS
            + [
                "_tile_hash_current",
                "_tile_hash_previous",
            ]
        ].copy()

        changed = changed.merge(
            tile_lookup,
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )

        changed[
            "tile_dependency_changed"
        ] = (
            changed["_tile_hash_current"]
            != changed["_tile_hash_previous"]
        )

        changed["tile_action"] = np.where(
            changed[
                "tile_dependency_changed"
            ],
            "RECOMPUTE",
            "REUSE",
        )

        changed = changed.drop(
            columns=[
                "_tile_hash_current",
                "_tile_hash_previous",
            ]
        )

    else:
        changed[
            "tile_dependency_changed"
        ] = pd.Series(
            dtype=bool
        )

        changed["tile_action"] = pd.Series(
            dtype="string"
        )

    changed.insert(
        0,
        "date",
        current_date,
    )

    changed.insert(
        1,
        "division",
        division,
    )

    changed["change_type"] = (
        "FEATURES_CHANGED"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    if changed.empty:
        tile_rows_to_recompute = 0
        tile_rows_to_reuse = 0

    else:
        tile_rows_to_recompute = int(
            changed[
                "tile_action"
            ]
            .eq("RECOMPUTE")
            .sum()
        )

        tile_rows_to_reuse = int(
            changed[
                "tile_action"
            ]
            .eq("REUSE")
            .sum()
        )

    stats = {
        "date": current_date,
        "previous_snapshot_date": previous_date,
        "division": division,
        "candidate_pairs_current": len(current),
        "candidate_pairs_previous": len(previous),
        "pairs_added": len(added),
        "pairs_removed": len(removed),
        "feature_rows_changed": len(changed),
        "tile_rows_to_compute": len(added),
        "tile_rows_to_recompute": tile_rows_to_recompute,
        "tile_rows_to_reuse": tile_rows_to_reuse,
        "tile_rows_to_drop": len(removed),
        "schema_columns_added": "|".join(
            schema_added
        ),
        "schema_columns_removed": "|".join(
            schema_removed
        ),
    }

    return (
        added,
        removed,
        changed,
        stats,
    )


# ============================================================
# SUMMARY
# ============================================================

def update_summary(
    new_rows: list[dict],
    current_date: date,
) -> None:
    """Append or replace summary rows for current_date."""

    new = pd.DataFrame(
        new_rows
    )

    if SUMMARY_FILE.exists():
        existing = pd.read_parquet(
            SUMMARY_FILE
        )

        existing["date"] = (
            pd.to_datetime(
                existing["date"]
            )
            .dt.date
        )

        existing = existing.loc[
            existing["date"]
            != current_date
        ].copy()

        summary = pd.concat(
            [
                existing,
                new,
            ],
            ignore_index=True,
        )

    else:
        summary = new

    if not summary.empty:
        summary = (
            summary
            .sort_values(
                [
                    "date",
                    "division",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    save_parquet(
        summary,
        SUMMARY_FILE,
    )


# ============================================================
# EXISTING DAILY RESULTS
# ============================================================

def load_existing_changes(
    current_date: date,
) -> pd.DataFrame:
    """Load all saved feature-history changes for one date."""

    day = current_date.isoformat()

    frames = []

    for folder in [
        ADDED_DIR,
        REMOVED_DIR,
        CHANGED_DIR,
    ]:
        current_folder = (
            folder
            / day
        )

        if not current_folder.exists():
            continue

        for file in sorted(
            current_folder.glob(
                "*.parquet"
            )
        ):
            df = pd.read_parquet(
                file
            )

            if not df.empty:
                frames.append(
                    df
                )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


# ============================================================
# MAIN
# ============================================================

def track_candidate_feature_history(
    snapshot_date: date | None = None,
) -> pd.DataFrame:
    """
    Track changes in one completed candidate-feature snapshot.

    The first completed snapshot establishes a baseline and records no
    changes. Later snapshots are compared with the most recent earlier
    completed snapshot.
    """

    ensure_directories()

    current_date = (
        snapshot_date
        if snapshot_date is not None
        else date.today()
    )

    current_snapshot = snapshot_dir(
        current_date
    )

    completion_file = (
        current_snapshot
        / "_complete.json"
    )

    if not completion_file.exists():
        raise FileNotFoundError(
            "Candidate-feature snapshot is not complete "
            f"for {current_date}. Expected:\n"
            f"{completion_file}"
        )

    processed_marker = (
        HISTORY_DIR
        / (
            f"_processed_"
            f"{current_date.isoformat()}.json"
        )
    )

    # --------------------------------------------------------
    # SAME-DAY IDEMPOTENCY
    # --------------------------------------------------------

    if processed_marker.exists():
        print(
            "Candidate feature history already "
            f"processed for {current_date}; "
            "returning saved changes."
        )

        return load_existing_changes(
            current_date
        )

    previous_date = previous_snapshot(
        current_date
    )

    print(
        "\n"
        "============================================\n"
        "CANDIDATE FEATURE HISTORY\n"
        "============================================"
    )

    print(
        f"Current snapshot : {current_date}"
    )

    # --------------------------------------------------------
    # INITIAL BASELINE
    # --------------------------------------------------------

    if previous_date is None:
        print(
            "Previous snapshot: None"
        )

        print(
            "Establishing initial feature baseline."
        )

        print(
            "No feature changes recorded."
        )

        marker_data = {
            "snapshot_date": current_date.isoformat(),
            "previous_snapshot_date": None,
            "baseline": True,
        }

        temp_marker = processed_marker.with_suffix(
            ".tmp.json"
        )

        try:
            with temp_marker.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    marker_data,
                    file,
                    indent=2,
                )

            temp_marker.replace(
                processed_marker
            )

        finally:
            temp_marker.unlink(
                missing_ok=True
            )

        return pd.DataFrame()

    print(
        f"Previous snapshot: {previous_date}"
    )

    current_divisions = snapshot_divisions(
        current_date
    )

    previous_divisions = snapshot_divisions(
        previous_date
    )

    divisions = sorted(
        current_divisions
        | previous_divisions
    )

    if not divisions:
        raise ValueError(
            "No candidate-feature divisions were found "
            f"for {current_date} or {previous_date}."
        )

    all_changes = []
    summary_rows = []

    day = current_date.isoformat()

    # --------------------------------------------------------
    # PROCESS EACH DIVISION
    # --------------------------------------------------------

    for division in divisions:
        print(
            f"\n[{division}]"
        )

        (
            added,
            removed,
            changed,
            stats,
        ) = compare_division(
            division=division,
            current_date=current_date,
            previous_date=previous_date,
        )

        save_parquet(
            added,
            ADDED_DIR
            / day
            / f"{division}.parquet",
        )

        save_parquet(
            removed,
            REMOVED_DIR
            / day
            / f"{division}.parquet",
        )

        save_parquet(
            changed,
            CHANGED_DIR
            / day
            / f"{division}.parquet",
        )

        summary_rows.append(
            stats
        )

        for frame in [
            added,
            removed,
            changed,
        ]:
            if not frame.empty:
                all_changes.append(
                    frame
                )

        if changed.empty:
            recompute = 0
            reuse = 0

        else:
            recompute = int(
                changed[
                    "tile_action"
                ]
                .eq("RECOMPUTE")
                .sum()
            )

            reuse = int(
                changed[
                    "tile_action"
                ]
                .eq("REUSE")
                .sum()
            )

        print(
            f"  Pairs added: {len(added):,}"
        )

        print(
            f"  Pairs removed: {len(removed):,}"
        )

        print(
            "  Feature rows changed: "
            f"{len(changed):,}"
        )

        print(
            f"  Tile recompute: {recompute:,}"
        )

        print(
            f"  Tile reuse: {reuse:,}"
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    update_summary(
        summary_rows,
        current_date,
    )

    # --------------------------------------------------------
    # PROCESSED MARKER
    # --------------------------------------------------------

    marker_data = {
        "snapshot_date": current_date.isoformat(),
        "previous_snapshot_date": previous_date.isoformat(),
        "baseline": False,
        "divisions": divisions,
    }

    temp_marker = processed_marker.with_suffix(
        ".tmp.json"
    )

    try:
        with temp_marker.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                marker_data,
                file,
                indent=2,
            )

        temp_marker.replace(
            processed_marker
        )

    finally:
        temp_marker.unlink(
            missing_ok=True
        )

    print(
        "\nCandidate feature history complete."
    )

    if not all_changes:
        return pd.DataFrame()

    return pd.concat(
        all_changes,
        ignore_index=True,
        sort=False,
    )