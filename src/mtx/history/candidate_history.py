"""
Daily transformer-candidate history for SmartLink.

Candidate generation itself is handled separately and cached under:

    data/cache/candidates/YYYY-MM-DD/

This module compares today's completed candidate snapshot with the most
recent previous completed snapshot.

History is tracked from both directions:

1. Meter perspective
   - candidate transformer added
   - candidate transformer removed
   - candidate count changed
   - gained first candidate
   - lost all candidates

2. Transformer perspective
   - candidate meter added
   - candidate meter removed
   - candidate meter count changed
   - gained first candidate meter
   - lost all candidate meters

Candidate history is intentionally independent of ITRON/grouping.
"""

from __future__ import annotations

import json
import re

from datetime import date
from pathlib import Path

import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CANDIDATE_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "candidates"
)

HISTORY_DIR = (
    CANDIDATE_CACHE_DIR
    / "history"
)


# ============================================================
# HISTORY PATHS
# ============================================================

METER_HISTORY_DIR = (
    HISTORY_DIR
    / "meters"
)

TRANSFORMER_HISTORY_DIR = (
    HISTORY_DIR
    / "transformers"
)

METER_ADDED_DIR = (
    METER_HISTORY_DIR
    / "added_candidates"
)

METER_REMOVED_DIR = (
    METER_HISTORY_DIR
    / "removed_candidates"
)

METER_CHANGED_DIR = (
    METER_HISTORY_DIR
    / "changed_rows"
)

TRANSFORMER_ADDED_DIR = (
    TRANSFORMER_HISTORY_DIR
    / "added_meters"
)

TRANSFORMER_REMOVED_DIR = (
    TRANSFORMER_HISTORY_DIR
    / "removed_meters"
)

TRANSFORMER_CHANGED_DIR = (
    TRANSFORMER_HISTORY_DIR
    / "changed_rows"
)

SUMMARY_FILE = (
    HISTORY_DIR
    / "candidate_daily_summary.parquet"
)


# ============================================================
# HELPERS
# ============================================================

DATE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}$"
)


def ensure_history_directories() -> None:
    """Create the candidate-history directory structure."""

    directories = [
        METER_ADDED_DIR,
        METER_REMOVED_DIR,
        METER_CHANGED_DIR,
        TRANSFORMER_ADDED_DIR,
        TRANSFORMER_REMOVED_DIR,
        TRANSFORMER_CHANGED_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_parquet(
    df: pd.DataFrame,
    path: Path,
) -> None:
    """
    Save parquet atomically.

    A temporary file is written first and validated before
    replacing the final file.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        ".tmp.parquet"
    )

    df.to_parquet(
        temp_path,
        index=False,
        compression="zstd",
    )

    # Validate before replacing final file.
    pd.read_parquet(
        temp_path
    )

    temp_path.replace(path)


def completed_candidate_dates() -> list: 
    """
    Return all dates with a completed candidate snapshot.

    A snapshot is considered valid only when _complete.json exists.
    """

    if not CANDIDATE_CACHE_DIR.exists():
        return []

    dates: list[date] = []

    for folder in CANDIDATE_CACHE_DIR.iterdir():

        if not folder.is_dir():
            continue

        if not DATE_PATTERN.match(folder.name):
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

    return sorted(dates)


def find_previous_snapshot(
    current_date: date,
) -> date | None:
    """
    Find the most recent completed snapshot before current_date.

    This deliberately does NOT assume yesterday exists.
    """

    previous_dates = [
        d
        for d in completed_candidate_dates()
        if d < current_date
    ]

    if not previous_dates:
        return None

    return max(previous_dates)


def snapshot_dir(
    snapshot_date: date,
) -> Path:
    return (
        CANDIDATE_CACHE_DIR
        / snapshot_date.isoformat()
    )


def read_division_file(
    snapshot_date: date,
    folder: str,
    division: str,
) -> pd.DataFrame:
    """
    Read a division file from a candidate snapshot.
    """

    path = (
        snapshot_dir(snapshot_date)
        / folder
        / f"{division}.parquet"
    )

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def snapshot_divisions(
    snapshot_date: date,
) -> set:
    """
    Determine divisions present in a completed snapshot.
    """

    pairs_dir = (
        snapshot_dir(snapshot_date)
        / "pairs"
    )

    meters_dir = (
        snapshot_dir(snapshot_date)
        / "meters"
    )

    divisions: set[str] = set()

    for directory in [
        pairs_dir,
        meters_dir,
    ]:

        if not directory.exists():
            continue

        divisions.update(
            file.stem
            for file in directory.glob(
                "*.parquet"
            )
        )

    return divisions


# ============================================================
# STANDARDIZE PAIRS
# ============================================================

def standardize_pairs(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Standardize candidate-pair data.

    Candidate identity is ONLY:

    - meter_number
    - candidate_transformer

    distance_ft is descriptive and may change independently.
    """

    columns = [
        "meter_number",
        "candidate_transformer",
        "distance_ft",
    ]

    if df.empty:
        return pd.DataFrame(
            columns=columns
        )

    out = df.copy()

    required = {
        "meter_number",
        "candidate_transformer",
    }

    missing = (
        required
        - set(out.columns)
    )

    if missing:
        raise ValueError(
            "Candidate pair file missing columns: "
            f"{sorted(missing)}"
        )

    if "distance_ft" not in out.columns:
        out["distance_ft"] = pd.NA

    out["meter_number"] = (
        out["meter_number"]
        .astype("string")
    )

    out["candidate_transformer"] = (
        out["candidate_transformer"]
        .astype("string")
    )

    out = (
        out[
            [
                "meter_number",
                "candidate_transformer",
                "distance_ft",
            ]
        ]
        .drop_duplicates(
            subset=[
                "meter_number",
                "candidate_transformer",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return out


# ============================================================
# METER HISTORY
# ============================================================

def meter_history(
    division: str,
    current_date: date,
    previous_date: date | None,
) -> dict[str, int]:
    """
    Compare candidate relationships from the meter perspective.
    """

    current_pairs = standardize_pairs(
        read_division_file(
            current_date,
            "pairs",
            division,
        )
    )

    if previous_date is None:

        previous_pairs = pd.DataFrame(
            columns=current_pairs.columns
        )

    else:

        previous_pairs = standardize_pairs(
            read_division_file(
                previous_date,
                "pairs",
                division,
            )
        )

    # --------------------------------------------------------
    # Candidate relationship identity
    # --------------------------------------------------------

    current_keys = current_pairs[
        [
            "meter_number",
            "candidate_transformer",
        ]
    ].copy()

    previous_keys = previous_pairs[
        [
            "meter_number",
            "candidate_transformer",
        ]
    ].copy()

    current_keys["_current"] = True
    previous_keys["_previous"] = True

    comparison = current_keys.merge(
        previous_keys,
        on=[
            "meter_number",
            "candidate_transformer",
        ],
        how="outer",
    )

    comparison["_current"] = (
        comparison["_current"]
        .fillna(False)
        .astype(bool)
    )

    comparison["_previous"] = (
        comparison["_previous"]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # Added candidate transformers
    # --------------------------------------------------------

    added = comparison[
        comparison["_current"]
        & ~comparison["_previous"]
    ][
        [
            "meter_number",
            "candidate_transformer",
        ]
    ].copy()

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
        "CANDIDATE_ADDED"
    )

    # --------------------------------------------------------
    # Removed candidate transformers
    # --------------------------------------------------------

    removed = comparison[
        ~comparison["_current"]
        & comparison["_previous"]
    ][
        [
            "meter_number",
            "candidate_transformer",
        ]
    ].copy()

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
        "CANDIDATE_REMOVED"
    )
        # --------------------------------------------------------
    # Candidate counts
    # --------------------------------------------------------

    current_counts = (
        current_pairs
        .groupby("meter_number")
        .size()
        .rename(
            "current_candidate_count"
        )
    )

    previous_counts = (
        previous_pairs
        .groupby("meter_number")
        .size()
        .rename(
            "previous_candidate_count"
        )
    )

    # IMPORTANT:
    # Use meter status files so meters with ZERO candidates
    # are still included.

    current_meter_state = (
        read_division_file(
            current_date,
            "meters",
            division,
        )
    )

    if previous_date is None:

        previous_meter_state = pd.DataFrame()

    else:

        previous_meter_state = (
            read_division_file(
                previous_date,
                "meters",
                division,
            )
        )

    current_meters = set()

    if (
        not current_meter_state.empty
        and "meter_number"
        in current_meter_state.columns
    ):
        current_meters = set(
            current_meter_state[
                "meter_number"
            ].astype("string")
        )

    previous_meters = set()

    if (
        not previous_meter_state.empty
        and "meter_number"
        in previous_meter_state.columns
    ):
        previous_meters = set(
            previous_meter_state[
                "meter_number"
            ].astype("string")
        )

    all_meters = sorted(
        current_meters
        | previous_meters
        | set(current_counts.index)
        | set(previous_counts.index)
    )

    changed_rows = pd.DataFrame(
        {
            "meter_number": all_meters,
        }
    )

    changed_rows[
        "previous_candidate_count"
    ] = (
        changed_rows["meter_number"]
        .map(previous_counts)
        .fillna(0)
        .astype(int)
    )

    changed_rows[
        "current_candidate_count"
    ] = (
        changed_rows["meter_number"]
        .map(current_counts)
        .fillna(0)
        .astype(int)
    )

    changed_rows[
        "candidate_count_change"
    ] = (
        changed_rows[
            "current_candidate_count"
        ]
        - changed_rows[
            "previous_candidate_count"
        ]
    )

    changed_rows[
        "previous_has_candidates"
    ] = (
        changed_rows[
            "previous_candidate_count"
        ] > 0
    )

    changed_rows[
        "current_has_candidates"
    ] = (
        changed_rows[
            "current_candidate_count"
        ] > 0
    )

    changed_rows["meter_added"] = (
        changed_rows["meter_number"]
        .isin(
            current_meters
            - previous_meters
        )
    )

    changed_rows["meter_removed"] = (
        changed_rows["meter_number"]
        .isin(
            previous_meters
            - current_meters
        )
    )

    changed_rows[
        "gained_first_candidate"
    ] = (
        (
            changed_rows[
                "previous_candidate_count"
            ] == 0
        )
        & (
            changed_rows[
                "current_candidate_count"
            ] > 0
        )
    )

    changed_rows[
        "lost_all_candidates"
    ] = (
        (
            changed_rows[
                "previous_candidate_count"
            ] > 0
        )
        & (
            changed_rows[
                "current_candidate_count"
            ] == 0
        )
    )

    # Keep only actual changes.

    changed_rows = changed_rows[
        (
            changed_rows[
                "candidate_count_change"
            ] != 0
        )
        | changed_rows["meter_added"]
        | changed_rows["meter_removed"]
        | changed_rows[
            "gained_first_candidate"
        ]
        | changed_rows[
            "lost_all_candidates"
        ]
    ].copy()

    changed_rows.insert(
        0,
        "date",
        current_date,
    )

    changed_rows.insert(
        1,
        "division",
        division,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    date_string = (
        current_date.isoformat()
    )

    save_parquet(
        added,
        METER_ADDED_DIR
        / date_string
        / f"{division}.parquet",
    )

    save_parquet(
        removed,
        METER_REMOVED_DIR
        / date_string
        / f"{division}.parquet",
    )

    save_parquet(
        changed_rows,
        METER_CHANGED_DIR
        / date_string
        / f"{division}.parquet",
    )

    return {
        "meter_candidate_pairs_added": len(added),
        "meter_candidate_pairs_removed": len(removed),
        "meters_changed": len(changed_rows),
        "meters_gained_candidates": (
            int(
                changed_rows[
                    "gained_first_candidate"
                ].sum()
            )
            if not changed_rows.empty
            else 0
        ),
        "meters_lost_all_candidates": (
            int(
                changed_rows[
                    "lost_all_candidates"
                ].sum()
            )
            if not changed_rows.empty
            else 0
        ),
    }


# ============================================================
# TRANSFORMER HISTORY
# ============================================================

def transformer_history(
    division: str,
    current_date: date,
    previous_date: date | None,
) -> dict[str, int]:
    """
    Compare candidate relationships from the transformer perspective.
    """

    current_pairs = standardize_pairs(
        read_division_file(
            current_date,
            "pairs",
            division,
        )
    )

    if previous_date is None:

        previous_pairs = pd.DataFrame(
            columns=current_pairs.columns
        )

    else:

        previous_pairs = standardize_pairs(
            read_division_file(
                previous_date,
                "pairs",
                division,
            )
        )

    # --------------------------------------------------------
    # Candidate meter relationships
    # --------------------------------------------------------

    current_keys = current_pairs[
        [
            "candidate_transformer",
            "meter_number",
        ]
    ].copy()

    previous_keys = previous_pairs[
        [
            "candidate_transformer",
            "meter_number",
        ]
    ].copy()

    current_keys["_current"] = True
    previous_keys["_previous"] = True

    comparison = current_keys.merge(
        previous_keys,
        on=[
            "candidate_transformer",
            "meter_number",
        ],
        how="outer",
    )

    comparison["_current"] = (
        comparison["_current"]
        .fillna(False)
        .astype(bool)
    )

    comparison["_previous"] = (
        comparison["_previous"]
        .fillna(False)
        .astype(bool)
    )

    # --------------------------------------------------------
    # Added candidate meters
    # --------------------------------------------------------

    added = comparison[
        comparison["_current"]
        & ~comparison["_previous"]
    ][
        [
            "candidate_transformer",
            "meter_number",
        ]
    ].copy()

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
        "METER_ADDED"
    )

    # --------------------------------------------------------
    # Removed candidate meters
    # --------------------------------------------------------

    removed = comparison[
        ~comparison["_current"]
        & comparison["_previous"]
    ][
        [
            "candidate_transformer",
            "meter_number",
        ]
    ].copy()

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
        "METER_REMOVED"
    )
        # --------------------------------------------------------
    # Candidate meter counts
    # --------------------------------------------------------

    current_counts = (
        current_pairs
        .groupby("candidate_transformer")
        .size()
        .rename(
            "current_candidate_meter_count"
        )
    )

    previous_counts = (
        previous_pairs
        .groupby("candidate_transformer")
        .size()
        .rename(
            "previous_candidate_meter_count"
        )
    )

    current_state = read_division_file(
        current_date,
        "transformers",
        division,
    )

    if previous_date is None:

        previous_state = pd.DataFrame()

    else:

        previous_state = read_division_file(
            previous_date,
            "transformers",
            division,
        )

    current_transformers = set()

    if (
        not current_state.empty
        and "transformer_floc"
        in current_state.columns
    ):
        current_transformers = set(
            current_state[
                "transformer_floc"
            ].astype("string")
        )

    previous_transformers = set()

    if (
        not previous_state.empty
        and "transformer_floc"
        in previous_state.columns
    ):
        previous_transformers = set(
            previous_state[
                "transformer_floc"
            ].astype("string")
        )

    all_transformers = sorted(
        current_transformers
        | previous_transformers
        | set(current_counts.index)
        | set(previous_counts.index)
    )

    changed_rows = pd.DataFrame(
        {
            "transformer_floc": all_transformers,
        }
    )

    changed_rows[
        "previous_candidate_meter_count"
    ] = (
        changed_rows["transformer_floc"]
        .map(previous_counts)
        .fillna(0)
        .astype(int)
    )

    changed_rows[
        "current_candidate_meter_count"
    ] = (
        changed_rows["transformer_floc"]
        .map(current_counts)
        .fillna(0)
        .astype(int)
    )

    changed_rows[
        "candidate_meter_count_change"
    ] = (
        changed_rows[
            "current_candidate_meter_count"
        ]
        - changed_rows[
            "previous_candidate_meter_count"
        ]
    )

    changed_rows[
        "previous_has_candidate_meters"
    ] = (
        changed_rows[
            "previous_candidate_meter_count"
        ] > 0
    )

    changed_rows[
        "current_has_candidate_meters"
    ] = (
        changed_rows[
            "current_candidate_meter_count"
        ] > 0
    )

    changed_rows["transformer_added"] = (
        changed_rows["transformer_floc"]
        .isin(
            current_transformers
            - previous_transformers
        )
    )

    changed_rows["transformer_removed"] = (
        changed_rows["transformer_floc"]
        .isin(
            previous_transformers
            - current_transformers
        )
    )

    changed_rows[
        "gained_first_candidate_meter"
    ] = (
        (
            changed_rows[
                "previous_candidate_meter_count"
            ] == 0
        )
        & (
            changed_rows[
                "current_candidate_meter_count"
            ] > 0
        )
    )

    changed_rows[
        "lost_all_candidate_meters"
    ] = (
        (
            changed_rows[
                "previous_candidate_meter_count"
            ] > 0
        )
        & (
            changed_rows[
                "current_candidate_meter_count"
            ] == 0
        )
    )

    changed_rows = changed_rows[
        (
            changed_rows[
                "candidate_meter_count_change"
            ] != 0
        )
        | changed_rows[
            "transformer_added"
        ]
        | changed_rows[
            "transformer_removed"
        ]
        | changed_rows[
            "gained_first_candidate_meter"
        ]
        | changed_rows[
            "lost_all_candidate_meters"
        ]
    ].copy()

    changed_rows.insert(
        0,
        "date",
        current_date,
    )

    changed_rows.insert(
        1,
        "division",
        division,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    date_string = current_date.isoformat()

    save_parquet(
        added,
        TRANSFORMER_ADDED_DIR
        / date_string
        / f"{division}.parquet",
    )

    save_parquet(
        removed,
        TRANSFORMER_REMOVED_DIR
        / date_string
        / f"{division}.parquet",
    )

    save_parquet(
        changed_rows,
        TRANSFORMER_CHANGED_DIR
        / date_string
        / f"{division}.parquet",
    )

    return {
        "transformer_candidate_meters_added": len(added),
        "transformer_candidate_meters_removed": len(removed),
        "transformers_changed": len(changed_rows),
        "transformers_gained_candidate_meters": (
            int(
                changed_rows[
                    "gained_first_candidate_meter"
                ].sum()
            )
            if not changed_rows.empty
            else 0
        ),
        "transformers_lost_all_candidate_meters": (
            int(
                changed_rows[
                    "lost_all_candidate_meters"
                ].sum()
            )
            if not changed_rows.empty
            else 0
        ),
    }


# ============================================================
# DAILY SUMMARY
# ============================================================

def update_summary(
    rows: list[dict],
    current_date: date,
) -> Path:
    """
    Append/replace today's division summaries.
    """

    new_summary = pd.DataFrame(
        rows
    )

    if SUMMARY_FILE.exists():

        existing = pd.read_parquet(
            SUMMARY_FILE
        )

        if "date" in existing.columns:

            existing["date"] = (
                pd.to_datetime(
                    existing["date"]
                ).dt.date
            )

            existing = existing[
                existing["date"]
                != current_date
            ]

        summary = pd.concat(
            [
                existing,
                new_summary,
            ],
            ignore_index=True,
        )

    else:

        summary = new_summary

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

    return SUMMARY_FILE


# ============================================================
# MAIN HISTORY FUNCTION
# ============================================================

def track_candidate_history(
    snapshot_date: date | None = None,
) -> Path:
    """
    Track candidate history for a completed daily snapshot.

    Parameters
    ----------
    snapshot_date:
        Candidate snapshot date to process.
        Defaults to today.

    Returns
    -------
    Path
        Path to candidate_daily_summary.parquet.
    """

    ensure_history_directories()

    current_date = (
        snapshot_date
        if snapshot_date is not None
        else date.today()
    )

    current_dir = snapshot_dir(
        current_date
    )

    completion_file = (
        current_dir
        / "_complete.json"
    )

    if not completion_file.exists():
        raise FileNotFoundError(
            "Candidate snapshot is not complete for "
            f"{current_date}. Expected:\n"
            f"{completion_file}"
        )

    previous_date = find_previous_snapshot(current_date)

    print(
        "\n"
        "============================================\n"
        "CANDIDATE HISTORY\n"
        "============================================"
    )

    print(
        f"Current snapshot : {current_date}"
    )

    if previous_date is None:

        print(
            "Previous snapshot: None (initial baseline)"
        )

        print(
            "No prior completed candidate snapshot found."
        )

    else:

        print(
            f"Previous snapshot: {previous_date}"
        )
    

    print(
        "\n"
        "============================================\n"
        "CANDIDATE HISTORY\n"
        "============================================"
    )

    print(
        f"Current snapshot : {current_date}"
    )


    current_divisions = (
        snapshot_divisions(
            current_date
        )
    )

    if previous_date is None:

        previous_divisions = set()

    else:

        previous_divisions = (
            snapshot_divisions(
                previous_date
            )
        )

    divisions = sorted(
        current_divisions
        | previous_divisions
    )

    if not divisions:
        raise ValueError(
            "No candidate divisions found "
            f"for {current_date}."
        )

    summary_rows: list[dict] = []

    for division in divisions:

        print(
            f"\n[{division}]"
        )

        meter_stats = meter_history(
            division=division,
            current_date=current_date,
            previous_date=previous_date,
        )

        transformer_stats = (
            transformer_history(
                division=division,
                current_date=current_date,
                previous_date=previous_date,
            )
        )

        row = {
            "date": current_date,
            "previous_snapshot_date": previous_date,
            "division": division,
            **meter_stats,
            **transformer_stats,
        }

        summary_rows.append(
            row
        )

        print(
            " Meter candidate pairs: "
            f"+{meter_stats['meter_candidate_pairs_added']} "
            f"-{meter_stats['meter_candidate_pairs_removed']}"
        )

        print(
            " Meters changed: "
            f"{meter_stats['meters_changed']}"
        )

        print(
            " Transformers changed: "
            f"{transformer_stats['transformers_changed']}"
        )

    summary_path = update_summary(
        rows=summary_rows,
        current_date=current_date,
    )

    # --------------------------------------------------------
    # Daily completion marker
    # --------------------------------------------------------

    marker = (
        HISTORY_DIR
        / f"_processed_{current_date.isoformat()}.json"
    )

    marker_data = {
        "snapshot_date": (
            current_date.isoformat()
        ),
        "previous_snapshot_date": (
            previous_date.isoformat()
            if previous_date
            else None
        ),
        "divisions": divisions,
    }

    temp_marker = marker.with_suffix(
        ".tmp.json"
    )

    with open(
        temp_marker,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            marker_data,
            file,
            indent=2,
        )

    temp_marker.replace(
        marker
    )

    print(
        "\nCandidate history complete."
    )

    print(
        f"History: {HISTORY_DIR}"
    )

    print(
        f"Summary: {summary_path}"
    )

    return summary_path