from pathlib import Path
from datetime import date, datetime, timezone
from uuid import uuid4

import polars as pl


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

HISTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "ami"
    / "history"
)


# ============================================================
# CONFIGURATION
# ============================================================

KEY = "meter_number"

SUMMARY_SCHEMA = {
    "date": pl.Date,
    "meter_count": pl.Int64,
    "added": pl.Int64,
    "removed": pl.Int64,
    "modified": pl.Int64,
}

CHANGE_SCHEMA = {
    "change_date": pl.String,
    KEY: pl.String,
    "change_type": pl.String,
    "field": pl.String,
    "previous_value": pl.String,
    "current_value": pl.String,
    "run_id": pl.String,
    "changed_at": pl.String,
}


# ============================================================
# LEGACY COLUMN NAMES
# ============================================================

# This mapping is used only when reading older history files.
# current_df must already be standardized before it is passed
# to track_ami_history().

OLD_NAMES = {
    "MeterID": KEY,
    "MeterId": KEY,
    "MeterNumber": KEY,
    "Meter_Number": KEY,
    "LATITUDE": "latitude",
    "Latitude": "latitude",
    "LONGITUDE": "longitude",
    "Longitude": "longitude",
    "SuperiorFLOC": "current_meter_floc",
    "division": "Division",
    "ServiceCenterName": "Division",
    "OPCO": "opco",
}


# ============================================================
# LEGACY COLUMN HELPER
# ============================================================

def oldnames(
    frame: pl.DataFrame | pl.LazyFrame,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Rename legacy AMI columns when reading older history files.

    This function does not standardize the current AMI input.
    The current AMI DataFrame must already be standardized.
    """

    if isinstance(frame, pl.LazyFrame):
        names = frame.collect_schema().names()
    else:
        names = frame.columns

    rename_map = {
        old_name: new_name
        for old_name, new_name in OLD_NAMES.items()
        if old_name in names
        and new_name not in names
    }

    return frame.rename(rename_map)


# ============================================================
# SAFE PARQUET SAVE
# ============================================================

def _save(
    frame: pl.DataFrame,
    path: Path,
) -> None:
    """
    Write a complete temporary Parquet file before replacing
    the destination file.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.write_parquet(
        temporary_path,
        compression="zstd",
    )

    temporary_path.replace(path)


# ============================================================
# TRACK AMI HISTORY
# ============================================================

def track_ami_history(
    current_df,
    snapshot_date=None,
    *,
    history_dir=HISTORY_DIR,
):
    """
    Record one complete standardized AMI extract per date.

    Parameters
    ----------
    current_df:
        Complete standardized AMI DataFrame.

        Accepts either a Pandas DataFrame or a Polars
        DataFrame.

        The DataFrame must already contain the standardized
        meter_number column.

    snapshot_date:
        Snapshot date in YYYY-MM-DD format.

        Defaults to today's date.

    history_dir:
        Directory where AMI history files are stored.

        Use a separate history directory when testing with
        sample data.

    Returns
    -------
    pl.DataFrame
        Field-level AMI changes for the snapshot date.

    Behavior
    --------
    First run:
        Saves the complete current AMI DataFrame as the initial
        baseline.

        Saves all current rows to changed_rows as INITIAL.

        Added, removed, and modified counts are zero.

    Subsequent runs:
        Compares current_df with the latest successfully saved
        previous snapshot.

        Records added, removed, and modified meters.

    Same date:
        Returns the previously saved change log without
        rewriting history.

    Requirements
    ------------
    - Run one AMI history process at a time.
    - Snapshot dates must advance.
    - The current AMI must not be empty.
    - meter_number must be non-null, nonblank, and unique.
    - Input columns and types must remain stable.
    - current_df must already be standardized.
    """

    # ========================================================
    # 1. DETERMINE SNAPSHOT DATE
    # ========================================================

    day = date.fromisoformat(
        str(
            snapshot_date
            or date.today()
        )[:10]
    )

    # ========================================================
    # 2. BUILD HISTORY PATHS
    # ========================================================

    root = Path(
        history_dir
    )

    previous_file = (
        root
        / "previous_ami_snapshot.parquet"
    )

    summary_file = (
        root
        / "ami_daily_summary.parquet"
    )

    changes_file = (
        root
        / "changes"
        / f"{day}.parquet"
    )

    rows_file = (
        root
        / "changed_rows"
        / f"{day}.parquet"
    )

    pending_file = (
        root
        / f"pending_ami_{day}.parquet"
    )

    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    changes_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 3. LOAD DAILY SUMMARY
    # ========================================================

    if summary_file.exists():
        summary = pl.read_parquet(
            summary_file
        )

    else:
        summary = pl.DataFrame(
            schema=SUMMARY_SCHEMA
        )

    # Support older summary files that used snapshot_date
    # instead of date.

    if (
        "date" not in summary.columns
        and "snapshot_date" in summary.columns
    ):
        summary = summary.rename(
            {
                "snapshot_date": "date",
            }
        )

    if "date" not in summary.columns:
        raise KeyError(
            "AMI daily summary must contain 'date'."
        )

    summary = summary.with_columns(
        pl.col("date")
        .cast(pl.String)
        .str.slice(0, 10)
        .str.to_date()
    )

    # ========================================================
    # 4. RECOVER AN INTERRUPTED FINAL SAVE
    # ========================================================

    for pending in root.glob(
        "pending_ami_*.parquet"
    ):
        pending_date_text = (
            pending.stem.removeprefix(
                "pending_ami_"
            )
        )

        pending_day = date.fromisoformat(
            pending_date_text
        )

        pending_is_in_summary = (
            summary.height > 0
            and pending_day in summary["date"]
        )

        if pending_is_in_summary:
            latest_summary_day = (
                summary["date"].max()
            )

            if pending_day != latest_summary_day:
                raise ValueError(
                    "Unexpected stale pending file: "
                    f"{pending}"
                )

            pending.replace(
                previous_file
            )

        elif pending_day != day:
            raise ValueError(
                f"Finish the interrupted {pending_day} "
                "history run before processing another date. "
                f"Pending file: {pending}"
            )

    # ========================================================
    # 5. RETURN EXISTING HISTORY FOR THE SAME DATE
    # ========================================================

    date_already_processed = (
        summary.height > 0
        and day in summary["date"]
    )

    if date_already_processed:
        print(
            f"AMI history already processed for {day}; "
            "returning saved changes."
        )

        if changes_file.exists():
            return pl.read_parquet(
                changes_file
            )

        return pl.DataFrame(
            schema=CHANGE_SCHEMA
        )

    # ========================================================
    # 6. PREVENT HISTORY FROM MOVING BACKWARD
    # ========================================================

    if summary.height > 0:
        last_recorded_date = (
            summary["date"].max()
        )

        if day <= last_recorded_date:
            raise ValueError(
                "Use a snapshot date after the last "
                f"recorded date: {last_recorded_date}."
            )

    # ========================================================
    # 7. IDENTIFY FIRST RUN
    # ========================================================

    first_run = (
        not previous_file.exists()
    )

    if first_run and summary.height > 0:
        raise FileNotFoundError(
            "The AMI summary exists, but the comparison "
            f"snapshot is missing: {previous_file}"
        )

    history_files_exist = (
        changes_file.exists()
        or rows_file.exists()
    )

    if (
        history_files_exist
        and not pending_file.exists()
    ):
        raise ValueError(
            f"History files already exist for {day}, "
            "but the summary does not contain that date."
        )

    # ========================================================
    # 8. CONVERT CURRENT AMI TO POLARS
    # ========================================================

    if isinstance(
        current_df,
        pl.DataFrame,
    ):
        current = current_df.clone()
        current = oldnames(current)

    else:
        current = pl.from_pandas(
            current_df
        )
        current = oldnames(current)

    if current.is_empty():
        raise ValueError(
            "The current AMI DataFrame is empty. "
            "History was not changed."
        )

    # ========================================================
    # 9. VALIDATE METER NUMBER COLUMN
    # ========================================================

    if KEY not in current.columns:
        raise KeyError(
            f"Current AMI must contain '{KEY}'. "
            f"Available columns: {current.columns}"
        )

    current = current.with_columns(
        pl.col(KEY)
        .cast(pl.String)
        .str.strip_chars()
    )

    # ========================================================
    # 10. LOAD PREVIOUS SNAPSHOT
    # ========================================================

    if first_run:
        old = current.head(
            0
        )

    else:
        print(previous_file)
        old = oldnames(
            pl.read_parquet(
                previous_file
            )
        )

        if KEY not in old.columns:
            raise KeyError(
                "Previous AMI snapshot is missing "
                f"'{KEY}'."
            )

        old = old.with_columns(
            pl.col(KEY)
            .cast(pl.String)
            .str.strip_chars()
        )

    # ========================================================
    # 11. VALIDATE METER NUMBERS
    # ========================================================

    for label, frame in [
        (
            "Current AMI",
            current,
        ),
        (
            "Previous AMI",
            old,
        ),
    ]:
        null_count = (
            frame[KEY].null_count()
        )

        unique_count = (
            frame[KEY].n_unique()
        )

        blank_count = (
            frame
            .select(
                pl.col(KEY)
                .str.strip_chars()
                .eq("")
                .sum()
                .alias("blank_count")
            )
            .item()
        )

        if (
            null_count > 0
            or unique_count != frame.height
            or blank_count > 0
        ):
            raise ValueError(
                f"{label}: '{KEY}' must be unique, "
                "nonblank, and non-null."
            )

    # ========================================================
    # 12. VALIDATE STANDARDIZED COLUMNS
    # ========================================================

    old_columns = set(
        old.columns
    )

    current_columns = set(
        current.columns
    )

    if old_columns != current_columns:
        only_in_old = (
            old_columns
            - current_columns
        )

        only_in_current = (
            current_columns
            - old_columns
        )

        raise ValueError(
            "Align standardized columns before tracking. "
            f"Only in previous snapshot: {only_in_old}; "
            f"only in current AMI: {only_in_current}"
        )

    # Ensure both DataFrames have the same column order.

    old = old.select(
        current.columns
    )

    # ========================================================
    # 13. ALIGN COLUMNS THAT WERE PREVIOUSLY ENTIRELY NULL
    # ========================================================

    if old.schema != current.schema:
        old_null_casts = {
            column: current.schema[column]
            for column in old.columns
            if (
                old.schema[column] == pl.Null
                and current.schema[column] != pl.Null
            )
        }

        current_null_casts = {
            column: old.schema[column]
            for column in current.columns
            if (
                current.schema[column] == pl.Null
                and old.schema[column] != pl.Null
            )
        }

        if old_null_casts:
            old = old.cast(
                old_null_casts
            )

        if current_null_casts:
            current = current.cast(
                current_null_casts
            )

    if old.schema != current.schema:
        old_schema = {
            column: str(data_type)
            for column, data_type in old.schema.items()
        }

        current_schema = {
            column: str(data_type)
            for column, data_type in current.schema.items()
        }

        raise ValueError(
            "Column types changed. Align input types "
            "before tracking. "
            f"Previous schema: {old_schema}. "
            f"Current schema: {current_schema}."
        )

    # ========================================================
    # 14. VALIDATE RESERVED HISTORY COLUMNS
    # ========================================================

    reserved_columns = (
        set(CHANGE_SCHEMA)
        - {
            KEY,
            "field",
            "previous_value",
            "current_value",
        }
    )

    reserved_found = (
        reserved_columns
        .intersection(
            current.columns
        )
    )

    has_row_version = (
        "row_version" in current.columns
    )

    old_suffix_columns = [
        column
        for column in current.columns
        if column.endswith(
            "__old"
        )
    ]

    if (
        reserved_found
        or has_row_version
        or old_suffix_columns
    ):
        raise ValueError(
            "The current AMI contains columns reserved "
            "for history metadata. "
            f"Reserved columns found: {reserved_found}. "
            f"Columns ending in '__old': {old_suffix_columns}."
        )

    # ========================================================
    # 15. FIND ADDED METERS
    # ========================================================

    added = current.join(
        old.select(
            KEY
        ),
        on=KEY,
        how="anti",
    )

    # ========================================================
    # 16. FIND REMOVED METERS
    # ========================================================

    removed = old.join(
        current.select(
            KEY
        ),
        on=KEY,
        how="anti",
    )

    # ========================================================
    # 17. DEFINE FIELDS TO COMPARE
    # ========================================================

    fields = [
        column
        for column in current.columns
        if column != KEY
    ]

    # ========================================================
    # 18. CREATE FIELD COMPARISON EXPRESSIONS
    # ========================================================

    different = {}

    for column in fields:
        new_value = pl.col(
            column
        )

        old_value = pl.col(
            f"{column}__old"
        )

        # Treat NaN as null when comparing floating-point
        # columns.

        if current.schema[column].is_float():
            new_value = new_value.fill_nan(
                None
            )

            old_value = old_value.fill_nan(
                None
            )

        different[column] = (
            ~new_value.eq_missing(
                old_value
            )
        )

    # ========================================================
    # 19. JOIN COMMON METERS
    # ========================================================

    joined = current.join(
        old,
        on=KEY,
        how="inner",
        suffix="__old",
    )

    # ========================================================
    # 20. FIND MODIFIED METERS
    # ========================================================

    if fields:
        modified_condition = (
            pl.any_horizontal(
                list(
                    different.values()
                )
            )
        )

    else:
        modified_condition = pl.lit(
            False
        )

    modified = joined.filter(
        modified_condition
    )

    modified_ids = modified.select(
        KEY
    )

    # ========================================================
    # 21. CREATE RUN METADATA
    # ========================================================

    stamp = {
        "change_date": str(day),
        "run_id": uuid4().hex[:12],
        "changed_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    # ========================================================
    # 22. BUILD FIELD-LEVEL CHANGE LOG
    # ========================================================

    logs = [
        pl.DataFrame(
            schema=CHANGE_SCHEMA
        )
    ]

    # Do not classify every meter as ADDED during the first
    # baseline run.

    if not first_run:
        for frame, change_type in [
            (
                added,
                "ADDED",
            ),
            (
                removed,
                "REMOVED",
            ),
        ]:
            row_events = (
                frame
                .select(
                    KEY
                )
                .with_columns(
                    change_type=pl.lit(
                        change_type
                    ),
                    field=pl.lit(
                        None,
                        dtype=pl.String,
                    ),
                    previous_value=pl.lit(
                        None,
                        dtype=pl.String,
                    ),
                    current_value=pl.lit(
                        None,
                        dtype=pl.String,
                    ),
                )
            )

            logs.append(
                row_events
            )

    # Create one change-log record for every modified field.

    for field in fields:
        field_changes = (
            modified
            .filter(
                different[field]
            )
            .select(
                KEY,
                pl.lit(
                    "MODIFIED"
                ).alias(
                    "change_type"
                ),
                pl.lit(
                    field
                ).alias(
                    "field"
                ),
                pl.col(
                    f"{field}__old"
                )
                .cast(
                    pl.String
                )
                .alias(
                    "previous_value"
                ),
                pl.col(
                    field
                )
                .cast(
                    pl.String
                )
                .alias(
                    "current_value"
                ),
            )
        )

        logs.append(
            field_changes
        )

    changes = (
        pl.concat(
            logs,
            how="diagonal_relaxed",
        )
        .with_columns(
            change_date=pl.lit(
                stamp["change_date"]
            ),
            run_id=pl.lit(
                stamp["run_id"]
            ),
            changed_at=pl.lit(
                stamp["changed_at"]
            ),
        )
        .select(
            list(
                CHANGE_SCHEMA.keys()
            )
        )
    )

    # ========================================================
    # 23. BUILD COMPLETE CHANGED ROWS
    # ========================================================

    if first_run:
        versions = [
            (
                current,
                "INITIAL",
                "CURRENT",
            ),
        ]

    else:
        versions = [
            (
                added,
                "ADDED",
                "CURRENT",
            ),
            (
                removed,
                "REMOVED",
                "PREVIOUS",
            ),
            (
                old.join(
                    modified_ids,
                    on=KEY,
                    how="semi",
                ),
                "MODIFIED_BEFORE",
                "PREVIOUS",
            ),
            (
                current.join(
                    modified_ids,
                    on=KEY,
                    how="semi",
                ),
                "MODIFIED_AFTER",
                "CURRENT",
            ),
        ]

    changed_row_frames = []

    for (
        frame,
        change_type,
        row_version,
    ) in versions:
        changed_frame = frame.with_columns(
            change_date=pl.lit(
                stamp["change_date"]
            ),
            run_id=pl.lit(
                stamp["run_id"]
            ),
            changed_at=pl.lit(
                stamp["changed_at"]
            ),
            change_type=pl.lit(
                change_type
            ),
            row_version=pl.lit(
                row_version
            ),
        )

        changed_row_frames.append(
            changed_frame
        )

    rows = pl.concat(
        changed_row_frames,
        how="diagonal_relaxed",
    )

    # ========================================================
    # 24. BUILD DAILY SUMMARY
    # ========================================================

    if first_run:
        added_count = 0
        removed_count = 0
        modified_count = 0

    else:
        added_count = (
            added.height
        )

        removed_count = (
            removed.height
        )

        modified_count = (
            modified.height
        )

    daily = pl.DataFrame(
        [
            {
                "date": day,
                "meter_count": current.height,
                "added": added_count,
                "removed": removed_count,
                "modified": modified_count,
            }
        ],
        schema=SUMMARY_SCHEMA,
    )

    updated_summary = pl.concat(
        [
            summary,
            daily,
        ],
        how="diagonal_relaxed",
    )

    # ========================================================
    # 25. SAVE THE CURRENT AMI AS A PENDING SNAPSHOT
    # ========================================================

    # The existing previous snapshot remains unchanged until
    # the changes, changed rows, and summary are all saved.

    _save(
        current,
        pending_file,
    )

    # ========================================================
    # 26. SAVE FIELD-LEVEL CHANGES
    # ========================================================

    _save(
        changes,
        changes_file,
    )

    # ========================================================
    # 27. SAVE COMPLETE CHANGED ROWS
    # ========================================================

    _save(
        rows,
        rows_file,
    )

    # ========================================================
    # 28. SAVE UPDATED DAILY SUMMARY
    # ========================================================

    _save(
        updated_summary,
        summary_file,
    )

    # ========================================================
    # 29. UPDATE THE PREVIOUS SNAPSHOT LAST
    # ========================================================

    pending_file.replace(
        previous_file
    )

    # ========================================================
    # 30. PRINT RUN RESULTS
    # ========================================================

    print(
        f"\nAMI history completed for {day}"
    )

    print(
        f"Meter count: {current.height:,}"
    )

    print(
        f"Added:      {added_count:,}"
    )

    print(
        f"Removed:    {removed_count:,}"
    )

    print(
        f"Modified:   {modified_count:,}"
    )

    print(
        f"Changes:    {changes_file}"
    )

    print(
        f"Rows:       {rows_file}"
    )

    print(
        f"Summary:    {summary_file}"
    )

    print(
        f"Baseline:   {previous_file}"
    )

    return changes


# ============================================================
# GET HISTORY FOR ONE METER
# ============================================================

def get_meter_history(
    meter_number,
    *,
    full_rows=False,
    history_dir=HISTORY_DIR,
):
    """
    Retrieve all saved history for one meter.

    Parameters
    ----------
    meter_number:
        Meter number to retrieve.

    full_rows:
        False returns field-level changes.

        True returns complete INITIAL, ADDED, REMOVED,
        MODIFIED_BEFORE, and MODIFIED_AFTER rows.

    history_dir:
        Directory containing the AMI history.

    Returns
    -------
    pl.DataFrame
        Saved history for the specified meter.
    """

    if full_rows:
        folder_name = (
            "changed_rows"
        )

    else:
        folder_name = (
            "changes"
        )

    folder = (
        Path(
            history_dir
        )
        / folder_name
    )

    if not folder.exists():
        return pl.DataFrame()

    queries = []

    for path in sorted(
        folder.glob(
            "*.parquet"
        )
    ):
        query = oldnames(
            pl.scan_parquet(
                path
            )
        )

        query_columns = (
            query
            .collect_schema()
            .names()
        )

        if KEY not in query_columns:
            continue

        query = (
            query
            .with_columns(
                pl.col(KEY)
                .cast(pl.String)
            )
            .filter(
                pl.col(KEY)
                == str(meter_number)
            )
        )

        queries.append(
            query
        )

    if not queries:
        return pl.DataFrame()

    result = (
        pl.concat(
            queries,
            how="diagonal_relaxed",
        )
        .collect()
    )

    if (
        result.is_empty()
        or "change_date" not in result.columns
    ):
        return result

    return result.sort(
        "change_date",
        maintain_order=True,
    )
