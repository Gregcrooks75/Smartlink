"""
Daily ITRON grouping history using the same layout as AMI history.

Paste into a notebook cell or import from this file. All ITRON grouping
constants and private helpers have their own names so they can coexist
with the AMI code.
"""

from collections.abc import Mapping
from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import polars as pl


# ============================================================
# PROJECT PATH AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ITRON_HISTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "itron"
    / "history"
)

ITRON_KEY = "meter_number"
ITRON_GROUP = "tx_group_id"


ITRON_SUMMARY_SCHEMA = {
    "date": pl.Date,
    "meter_count": pl.Int64,
    "added": pl.Int64,
    "removed": pl.Int64,
    "modified": pl.Int64,
    "assigned_meter_count": pl.Int64,
    "group_count": pl.Int64,
    "group_size_changed": pl.Int64,
    "new_assignments": pl.Int64,
    "reassigned": pl.Int64,
    "lost_assignments": pl.Int64,
    "metadata_only_changes": pl.Int64,
    "assignments_changed": pl.Boolean,
}


ITRON_CHANGE_SCHEMA = {
    "change_date": pl.String,
    ITRON_KEY: pl.String,
    "change_type": pl.String,
    "field": pl.String,
    "previous_value": pl.String,
    "current_value": pl.String,
    "run_id": pl.String,
    "changed_at": pl.String,
}


ITRON_ASSIGNMENT_SCHEMA = {
    "change_date": pl.String,
    ITRON_KEY: pl.String,
    "event": pl.String,
    "previous_group": pl.String,
    "current_group": pl.String,
    "run_id": pl.String,
    "changed_at": pl.String,
}


# Used only when reading older history.
# Never used to standardize current_df.
ITRON_OLD_NAMES = {
    "MeterID": ITRON_KEY,
    "MeterId": ITRON_KEY,
    "MeterNumber": ITRON_KEY,
    "Meter_Number": ITRON_KEY,
    "division": "Division",
    "SuperiorFLOC": "current_meter_floc",
    "OPCO": "opco",
}


# ============================================================
# PRIVATE HELPERS
# ============================================================

def itronoldnames(frame):
    """Rename legacy ITRON grouping columns when reading older history."""

    names = (
        frame.collect_schema().names()
        if isinstance(frame, pl.LazyFrame)
        else frame.columns
    )

    renames = {
        old: new
        for old, new in ITRON_OLD_NAMES.items()
        if old in names and new not in names
    }

    if len(set(renames.values())) != len(renames):
        raise ValueError(
            "Multiple legacy ITRON grouping columns map to the same name."
        )

    return frame.rename(renames)


def itronsave(frame, path):
    """Finish a temporary Parquet file before replacing its destination."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(path.suffix + ".tmp")

    frame.write_parquet(
        temporary,
        compression="zstd",
    )

    temporary.replace(path)


def itrongroup(column):
    """
    Return a nonblank, stripped tx_group_id expression.

    This expression is used only for assignment comparisons. The original
    group text is retained in full rows and field-level changes.

    A nonblank ID counts as assigned, including explicit standalone group
    IDs.
    """

    value = (
        pl.col(column)
        .cast(pl.String)
        .str.strip_chars()
    )

    return (
        pl.when(
            value.is_not_null()
            & value.ne("")
        )
        .then(value)
        .otherwise(None)
    )


def itronprepare(frame, description):
    """Validate and prepare an ITRON grouping DataFrame."""

    missing = {
        ITRON_KEY,
        ITRON_GROUP,
    } - set(frame.columns)

    if missing:
        raise KeyError(
            f"{description} is missing required columns: "
            f"{sorted(missing)}"
        )

    reserved = {
        "change_date",
        "change_type",
        "row_version",
        "run_id",
        "changed_at",
    }

    conflicts = reserved.intersection(frame.columns)

    conflicts.update(
        column
        for column in frame.columns
        if column.endswith("__old")
    )

    if conflicts:
        raise ValueError(
            f"{description} contains reserved history columns: "
            f"{sorted(conflicts)}"
        )

    # Avoid converting floating-point missing meter or group IDs
    # to the literal string "NaN".
    frame = (
        frame.with_columns(
            [
                (
                    pl.col(column).fill_nan(None)
                    if frame.schema[column].is_float()
                    else pl.col(column)
                )
                .cast(pl.String)
                .alias(column)
                for column in (
                    ITRON_KEY,
                    ITRON_GROUP,
                )
            ]
        )
        .with_columns(
            pl.col(ITRON_KEY).str.strip_chars()
        )
    )

    key = frame[ITRON_KEY]

    if (
        key.null_count()
        or key.eq("").any()
        or key.n_unique() != frame.height
    ):
        raise ValueError(
            f"{description}: '{ITRON_KEY}' must be unique, "
            "nonblank, and non-null. Resolve duplicate meters "
            "or assignments before tracking."
        )

    for column in (
        "group_source",
        "tx_group_source",
    ):
        if column in frame.columns:
            source = (
                frame[column]
                .cast(pl.String)
                .str.strip_chars()
                .str.to_lowercase()
            )

            if (
                source.null_count()
                or not source.eq("itron").all()
            ):
                raise ValueError(
                    f"{description}: every '{column}' must be 'itron'."
                )

    return frame


def itronread_summary(root):
    """Read and validate the committed ITRON grouping daily summary."""

    path = Path(root) / "grouping_daily_summary.parquet"

    if not path.exists():
        return pl.DataFrame(
            schema=ITRON_SUMMARY_SCHEMA
        )

    summary = pl.read_parquet(path)

    if (
        "date" not in summary.columns
        and "snapshot_date" in summary.columns
    ):
        summary = summary.rename(
            {"snapshot_date": "date"}
        )

    required = {
        "date",
        "meter_count",
        "added",
        "removed",
        "modified",
    }

    if not required.issubset(summary.columns):
        missing = required - set(summary.columns)

        raise ValueError(
            "ITRON grouping summary is missing columns: "
            f"{sorted(missing)}"
        )

    summary = summary.with_columns(
        pl.col("date")
        .cast(pl.String)
        .str.slice(0, 10)
        .str.to_date()
    )

    if (
        summary["date"].null_count()
        or summary["date"].n_unique() != summary.height
    ):
        raise ValueError(
            "ITRON grouping summary dates must be non-null and unique."
        )

    # Older summaries may lack assignment counters.
    # Unknown historical counts remain null.
    summary = summary.with_columns(
        [
            pl.lit(
                None,
                dtype=dtype,
            ).alias(column)
            for column, dtype in ITRON_SUMMARY_SCHEMA.items()
            if column not in summary.columns
        ]
    )

    return (
        summary
        .select(list(ITRON_SUMMARY_SCHEMA))
        .cast(ITRON_SUMMARY_SCHEMA)
        .sort("date")
    )


# ============================================================
# TRACK ITRON GROUPING HISTORY
# ============================================================

def track_grouping_history(
    current_df,
    snapshot_date=None,
    *,
    history_dir=ITRON_HISTORY_DIR,
    group_source="itron",
):
    """
    Record one complete, standardized ITRON grouping extract per date.

    Parameters
    ----------
    current_df
        Pandas or Polars DataFrame containing the complete standardized
        ITRON grouping data.

    snapshot_date
        Date assigned to the snapshot. Defaults to today's date.

    history_dir
        Directory used to store the daily ITRON grouping history.

    group_source
        Must be "itron". Other grouping sources must use separate history.

    Required columns
    ----------------
    meter_number
    tx_group_id

    Notes
    -----
    All supplied columns, including Tx_Subgroup_ID, tx_group_size, Division,
    group_type, meter_type, and current_meter_floc, are compared and retained.

    Every meter must occur once. Use all divisions together and consistently
    use either the ITRON cache or final grouping output. Do not switch between
    schemas. Prefer the final output if standalone meters should be included.

    First run:
        Saves INITIAL rows. Delta counters are zero.

    Later dates:
        Compares with previous_grouping_snapshot.parquet.

    Same date:
        Returns saved field changes. Newer same-day edits are not processed.

    Returns
    -------
    pl.DataFrame
        Field-level changes using the same general layout as AMI history.

    Important
    ---------
    Run one writer at a time. Dates must advance.

    Empty extracts, invalid keys, column changes, and incompatible types are
    rejected before new data is saved.

    Use a separate history_dir for samples because a subset would imply that
    the excluded meters were removed.

    Assignment events compare stripped, nonblank tx_group_id values. Group
    IDs must be stable and division-qualified, such as
    "itron::Ithaca::9075".

    Group renumbering is recorded as reassignment. Splits and merges are not
    inferred.

    group_count counts distinct nonblank IDs, including standalone IDs.

    group_size_changed counts common meters with a changed tx_group_size,
    not the number of groups. It is null when that column is absent.

    assignments_changed covers meter and group-ID pairs, not every other
    field.

    changed_at is the UTC observation time, not a source-system event time.

    This file is self-contained. No shared snapshot-history module is
    required.

    Existing incompatible history requires migration or a separate directory.
    """

    if str(group_source).strip().lower() != "itron":
        raise ValueError(
            "This tracker is for ITRON only; keep other sources separate."
        )

    day = date.fromisoformat(
        str(
            snapshot_date
            or date.today()
        )[:10]
    )

    root = Path(history_dir)

    previous_file = (
        root
        / "previous_grouping_snapshot.parquet"
    )

    summary_file = (
        root
        / "grouping_daily_summary.parquet"
    )

    pending_file = (
        root
        / f"pending_grouping_{day}.parquet"
    )

    daily_files = {
        folder: (
            root
            / folder
            / f"{day}.parquet"
        )
        for folder in (
            "changes",
            "changed_rows",
            "assignment_changes",
        )
    }

    # Refuse to mix the old per-run layout into this daily
    # history directory.
    if (root / "runs").exists():
        raise ValueError(
            "This directory contains older per-run ITRON grouping history. "
            "Use a separate directory for the daily layout, or migrate it "
            "first."
        )

    summary = itronread_summary(root)

    # Recover a run that committed its summary but did not promote
    # its snapshot.
    for pending in sorted(
        root.glob("pending_grouping_*.parquet")
    ):
        pending_day = date.fromisoformat(
            pending.stem.removeprefix(
                "pending_grouping_"
            )
        )

        if pending_day in summary["date"]:
            if pending_day != summary["date"].max():
                raise ValueError(
                    f"Unexpected stale pending file: {pending}"
                )

            for folder in daily_files:
                path = (
                    root
                    / folder
                    / f"{pending_day}.parquet"
                )

                if not path.exists():
                    raise FileNotFoundError(
                        "Committed ITRON grouping history is incomplete: "
                        f"{path}"
                    )

            pending.replace(previous_file)

        elif pending_day != day:
            raise ValueError(
                f"Finish the interrupted {pending_day} ITRON grouping "
                f"run first: {pending}"
            )

    first_run = not previous_file.exists()

    if (
        first_run
        and summary.is_empty()
        and not pending_file.exists()
    ):
        if any(root.rglob("*.parquet")):
            raise ValueError(
                "Existing history files do not match this daily layout. "
                "Migrate the old history or use a separate history_dir."
            )

    if first_run and summary.height:
        raise FileNotFoundError(
            "ITRON grouping summary exists but comparison snapshot "
            f"is missing: {previous_file}"
        )

    if (
        not first_run
        and summary.is_empty()
    ):
        raise ValueError(
            "ITRON grouping comparison snapshot exists but its daily "
            "summary is missing or empty."
        )

    # Return the already committed result for the same date.
    if day in summary["date"]:
        for path in daily_files.values():
            if not path.exists():
                raise FileNotFoundError(
                    "Saved ITRON grouping history is incomplete: "
                    f"{path}"
                )

        print(
            f"ITRON grouping history already processed for {day}; "
            "returning saved changes."
        )

        return pl.read_parquet(
            daily_files["changes"]
        )

    if (
        summary.height
        and day <= summary["date"].max()
    ):
        raise ValueError(
            "Use a date after the last recorded date: "
            f"{summary['date'].max()}."
        )

    if (
        any(
            path.exists()
            for path in daily_files.values()
        )
        and not pending_file.exists()
    ):
        raise ValueError(
            f"ITRON grouping history files exist for {day}, but its "
            "summary entry is missing."
        )

    # Convert Pandas input to Polars, or clone an existing
    # Polars DataFrame.
    current = (
        current_df.clone()
        if isinstance(current_df, pl.DataFrame)
        else pl.from_pandas(current_df)
    )

    if current.is_empty():
        raise ValueError(
            "The current ITRON grouping DataFrame is empty. "
            "History was not changed."
        )

    current = itronprepare(
        current,
        "Current standardized ITRON grouping data",
    )

    if first_run:
        old = current.head(0)

    else:
        old = itronprepare(
            itronoldnames(
                pl.read_parquet(previous_file)
            ),
            "Previous ITRON grouping snapshot",
        )

    # Compare every source column.
    # Permit an all-null column to acquire a type.
    if set(old.columns) != set(current.columns):
        raise ValueError(
            "Align standardized columns before tracking. "
            f"Only in previous: "
            f"{sorted(set(old.columns) - set(current.columns))}; "
            f"only in current: "
            f"{sorted(set(current.columns) - set(old.columns))}."
        )

    old = old.select(current.columns)

    old = old.cast(
        {
            column: current.schema[column]
            for column in old.columns
            if old.schema[column] == pl.Null
        }
    )

    current = current.cast(
        {
            column: old.schema[column]
            for column in current.columns
            if current.schema[column] == pl.Null
        }
    )

    if old.schema != current.schema:
        raise ValueError(
            "Column types changed; align input types before tracking. "
            f"Previous: {old.schema}. "
            f"Current: {current.schema}."
        )

    # --------------------------------------------------------
    # IDENTIFY ADDED, REMOVED, AND MODIFIED METERS
    # --------------------------------------------------------

    added = current.join(
        old.select(ITRON_KEY),
        on=ITRON_KEY,
        how="anti",
    )

    removed = old.join(
        current.select(ITRON_KEY),
        on=ITRON_KEY,
        how="anti",
    )

    joined = current.join(
        old,
        on=ITRON_KEY,
        how="inner",
        suffix="__old",
    )

    fields = [
        column
        for column in current.columns
        if column != ITRON_KEY
    ]

    different = {}

    for column in fields:
        new = pl.col(column)
        before = pl.col(
            column + "__old"
        )

        if current.schema[column].is_float():
            new = new.fill_nan(None)
            before = before.fill_nan(None)

        different[column] = (
            ~new.eq_missing(before)
        )

    modified = joined.filter(
        pl.any_horizontal(
            list(different.values())
        )
    )

    modified_ids = modified.select(
        ITRON_KEY
    )

    stamp = {
        "change_date": str(day),
        "run_id": uuid4().hex[:12],
        "changed_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    metadata = {
        key: pl.lit(value)
        for key, value in stamp.items()
    }

    null = pl.lit(
        None,
        dtype=pl.String,
    )

    # --------------------------------------------------------
    # FIELD-LEVEL CHANGE LOG
    # --------------------------------------------------------

    logs = [
        pl.DataFrame(
            schema=ITRON_CHANGE_SCHEMA
        )
    ]

    if not first_run:
        for frame, kind in (
            (added, "ADDED"),
            (removed, "REMOVED"),
        ):
            logs.append(
                frame.select(
                    ITRON_KEY,
                    pl.lit(kind).alias(
                        "change_type"
                    ),
                    null.alias("field"),
                    null.alias(
                        "previous_value"
                    ),
                    null.alias(
                        "current_value"
                    ),
                )
            )

        for field in fields:
            logs.append(
                modified.filter(
                    different[field]
                ).select(
                    ITRON_KEY,
                    pl.lit(
                        "MODIFIED"
                    ).alias(
                        "change_type"
                    ),
                    pl.lit(field).alias(
                        "field"
                    ),
                    pl.col(
                        field + "__old"
                    )
                    .cast(pl.String)
                    .alias(
                        "previous_value"
                    ),
                    pl.col(field)
                    .cast(pl.String)
                    .alias(
                        "current_value"
                    ),
                )
            )

    changes = (
        pl.concat(
            logs,
            how="diagonal_relaxed",
        )
        .with_columns(**metadata)
        .select(
            list(ITRON_CHANGE_SCHEMA)
        )
        .cast(ITRON_CHANGE_SCHEMA)
    )

    # --------------------------------------------------------
    # GROUP-ASSIGNMENT EVENTS
    # --------------------------------------------------------

    assignment_parts = [
        pl.DataFrame(
            schema=ITRON_ASSIGNMENT_SCHEMA
        )
    ]

    if not first_run:
        assignment_parts.append(
            added.filter(
                itrongroup(
                    ITRON_GROUP
                ).is_not_null()
            ).select(
                ITRON_KEY,
                pl.lit(
                    "ADDED_ASSIGNED"
                ).alias("event"),
                null.alias(
                    "previous_group"
                ),
                itrongroup(
                    ITRON_GROUP
                ).alias(
                    "current_group"
                ),
            )
        )

        assignment_parts.append(
            removed.filter(
                itrongroup(
                    ITRON_GROUP
                ).is_not_null()
            ).select(
                ITRON_KEY,
                pl.lit(
                    "ASSIGNED_METER_REMOVED"
                ).alias("event"),
                itrongroup(
                    ITRON_GROUP
                ).alias(
                    "previous_group"
                ),
                null.alias(
                    "current_group"
                ),
            )
        )

        before = itrongroup(
            ITRON_GROUP + "__old"
        )

        after = itrongroup(
            ITRON_GROUP
        )

        event = (
            pl.when(
                before.is_null()
                & after.is_not_null()
            )
            .then(
                pl.lit(
                    "GROUP_ASSIGNED"
                )
            )
            .when(
                before.is_not_null()
                & after.is_null()
            )
            .then(
                pl.lit(
                    "GROUP_CLEARED"
                )
            )
            .otherwise(
                pl.lit(
                    "GROUP_REASSIGNED"
                )
            )
        )

        assignment_parts.append(
            joined.filter(
                ~before.eq_missing(after)
            ).select(
                ITRON_KEY,
                event.alias("event"),
                before.alias(
                    "previous_group"
                ),
                after.alias(
                    "current_group"
                ),
            )
        )

    assignment_changes = (
        pl.concat(
            assignment_parts,
            how="diagonal_relaxed",
        )
        .with_columns(**metadata)
        .select(
            list(ITRON_ASSIGNMENT_SCHEMA)
        )
        .cast(ITRON_ASSIGNMENT_SCHEMA)
    )

    # --------------------------------------------------------
    # SAVE COMPLETE CHANGED-ROW VERSIONS
    # --------------------------------------------------------

    if first_run:
        versions = [
            (
                current,
                "INITIAL",
                "CURRENT",
            )
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
                    on=ITRON_KEY,
                    how="semi",
                ),
                "MODIFIED_BEFORE",
                "PREVIOUS",
            ),
            (
                current.join(
                    modified_ids,
                    on=ITRON_KEY,
                    how="semi",
                ),
                "MODIFIED_AFTER",
                "CURRENT",
            ),
        ]

    rows = pl.concat(
        [
            frame.with_columns(
                **metadata,
                change_type=pl.lit(kind),
                row_version=pl.lit(version),
            )
            for frame, kind, version in versions
        ],
        how="diagonal_relaxed",
    )

    # --------------------------------------------------------
    # DAILY SUMMARY COUNTERS
    # --------------------------------------------------------

    def count_events(*events):
        return assignment_changes.filter(
            pl.col("event").is_in(events)
        ).height

    new_assignments = count_events(
        "ADDED_ASSIGNED",
        "GROUP_ASSIGNED",
    )

    reassigned = count_events(
        "GROUP_REASSIGNED"
    )

    lost_assignments = count_events(
        "GROUP_CLEARED",
        "ASSIGNED_METER_REMOVED",
    )

    # Includes cosmetic group-ID changes, such as changes
    # involving surrounding whitespace only.
    metadata_only = modified_ids.join(
        assignment_changes.select(
            ITRON_KEY
        ),
        on=ITRON_KEY,
        how="anti",
    ).height

    group_count = current.select(
        itrongroup(
            ITRON_GROUP
        )
        .drop_nulls()
        .n_unique()
    ).item()

    size_changed = (
        modified.filter(
            different["tx_group_size"]
        ).height
        if "tx_group_size" in different
        else None
    )

    daily = pl.DataFrame(
        [
            {
                "date": day,
                "meter_count": current.height,
                "added": (
                    0
                    if first_run
                    else added.height
                ),
                "removed": removed.height,
                "modified": modified.height,
                "assigned_meter_count": (
                    current.filter(
                        itrongroup(
                            ITRON_GROUP
                        ).is_not_null()
                    ).height
                ),
                "group_count": group_count,
                "group_size_changed": size_changed,
                "new_assignments": new_assignments,
                "reassigned": reassigned,
                "lost_assignments": lost_assignments,
                "metadata_only_changes": metadata_only,
                "assignments_changed": bool(
                    new_assignments
                    or reassigned
                    or lost_assignments
                ),
            }
        ],
        schema=ITRON_SUMMARY_SCHEMA,
    )

    updated_summary = pl.concat(
        [
            summary,
            daily,
        ]
    )

    # The summary is the commit marker.
    # Update the comparison snapshot last.
    #
    # An interrupted, uncommitted same-date run is recomputed
    # using current_df.
    itronsave(
        current,
        pending_file,
    )

    itronsave(
        changes,
        daily_files["changes"],
    )

    itronsave(
        rows,
        daily_files["changed_rows"],
    )

    itronsave(
        assignment_changes,
        daily_files["assignment_changes"],
    )

    itronsave(
        updated_summary,
        summary_file,
    )

    pending_file.replace(
        previous_file
    )

    result = daily.row(
        0,
        named=True,
    )

    initial_message = (
        " - initial baseline"
        if first_run
        else ""
    )

    print(
        f"\nITRON grouping history completed for {day}"
        f"{initial_message}"
    )

    print(
        f"Meters: {current.height:,} | "
        f"Assigned: {result['assigned_meter_count']:,}"
    )

    print(
        f"Added: {result['added']:,} | "
        f"Removed: {result['removed']:,} | "
        f"Modified: {result['modified']:,}"
    )

    print(
        f"Groups: {group_count:,} | "
        f"Meters with changed group size: {size_changed}"
    )

    print(
        f"New assignments: {new_assignments:,} | "
        f"Reassigned: {reassigned:,} | "
        f"Lost assignments: {lost_assignments:,}"
    )

    print(
        "Metadata/formatting-only changes: "
        f"{metadata_only:,}"
    )

    print(
        "Group assignments changed: "
        f"{result['assignments_changed']}"
    )

    print(
        f"History: {root}"
    )

    return changes


# ============================================================
# RETRIEVE SAVED HISTORY
# ============================================================

def get_meter_grouping_history(
    meter_number,
    *,
    full_rows=False,
    assignments_only=False,
    history_dir=ITRON_HISTORY_DIR,
):
    """
    Return the saved ITRON grouping history for one meter.

    Field-level changes are returned by default.

    Parameters
    ----------
    meter_number
        Meter number to retrieve.

    full_rows
        Return complete changed-row versions.

    assignments_only
        Return only group-assignment events.

    history_dir
        ITRON grouping history directory.

    Notes
    -----
    Only dates committed in the summary are included. Incomplete runs
    remain hidden.

    INITIAL rows are present only when full_rows=True.
    """

    if full_rows and assignments_only:
        raise ValueError(
            "Choose either full_rows=True or assignments_only=True."
        )

    if full_rows:
        folder = "changed_rows"

    elif assignments_only:
        folder = "assignment_changes"

    else:
        folder = "changes"

    root = Path(history_dir)
    queries = []

    for day in itronread_summary(root)["date"]:
        path = (
            root
            / folder
            / f"{day}.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                "Saved ITRON grouping history is incomplete: "
                f"{path}"
            )

        query = itronoldnames(
            pl.scan_parquet(path)
        )

        if (
            ITRON_KEY
            not in query.collect_schema().names()
        ):
            raise KeyError(
                f"Saved history is missing "
                f"'{ITRON_KEY}': {path}"
            )

        query = (
            query
            .with_columns(
                pl.col(ITRON_KEY)
                .cast(pl.String)
                .str.strip_chars()
            )
            .filter(
                pl.col(ITRON_KEY)
                == str(
                    meter_number
                ).strip()
            )
        )

        queries.append(query)

    if not queries:
        if full_rows:
            return pl.DataFrame()

        schema = (
            ITRON_ASSIGNMENT_SCHEMA
            if assignments_only
            else ITRON_CHANGE_SCHEMA
        )

        return pl.DataFrame(
            schema=schema
        )

    return (
        pl.concat(
            queries,
            how="diagonal_relaxed",
        )
        .collect()
        .sort(
            "change_date",
            maintain_order=True,
        )
    )


def get_grouping_assignment_changes(
    snapshot_date=None,
    *,
    history_dir=ITRON_HISTORY_DIR,
):
    """
    Return group-assignment events for a committed date.

    The latest committed date is used by default.
    """

    root = Path(history_dir)
    summary = itronread_summary(root)

    if summary.is_empty():
        return pl.DataFrame(
            schema=ITRON_ASSIGNMENT_SCHEMA
        )

    day = (
        summary["date"].max()
        if snapshot_date is None
        else date.fromisoformat(
            str(snapshot_date)[:10]
        )
    )

    if day not in summary["date"]:
        raise ValueError(
            f"No committed ITRON grouping history for {day}."
        )

    path = (
        root
        / "assignment_changes"
        / f"{day}.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Saved ITRON grouping assignment history is missing: "
            f"{path}"
        )

    return pl.read_parquet(path)


# ============================================================
# TRACK COMPLETE GROUPING OUTPUT SPLIT ACROSS PARQUET FILES
# ============================================================

def track_grouping_files_history(
    group_files,
    *,
    expected_divisions=None,
    **kwargs,
):
    """
    Combine standardized grouping files and record one full history run.

    Parameters
    ----------
    group_files
        A path, list, tuple, or nested mapping of divisions to Parquet paths.

    expected_divisions
        Optional complete collection of expected Division values.

    **kwargs
        Additional arguments passed to track_grouping_history().

    Notes
    -----
    Pass the complete set of files, not one division at a time.

    Missing files, duplicate paths, and duplicate meters raise errors.

    No source-column renaming or deduplication is performed.
    """

    paths = []

    def collect(value):
        if isinstance(
            value,
            (
                str,
                Path,
            ),
        ):
            paths.append(
                Path(value)
            )

        elif isinstance(
            value,
            Mapping,
        ):
            for nested in value.values():
                collect(nested)

        elif isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            for nested in value:
                collect(nested)

        else:
            raise TypeError(
                "Expected grouping file paths, got "
                f"{type(value).__name__}."
            )

    collect(group_files)

    if not paths:
        raise ValueError(
            "No grouping files supplied; history was not changed."
        )

    resolved_paths = {
        path.resolve()
        for path in paths
    }

    if len(resolved_paths) != len(paths):
        raise ValueError(
            "The same grouping file was supplied more than once."
        )

    frames = []

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(
                f"Grouping file is missing: {path}"
            )

        frames.append(
            pl.read_parquet(path)
        )

    # Enforce matching columns across the partitions.
    columns = frames[0].columns

    if any(
        set(frame.columns) != set(columns)
        for frame in frames
    ):
        raise ValueError(
            "Grouping files have different columns; "
            "standardize them first."
        )

    current = pl.concat(
        [
            frame.select(columns)
            for frame in frames
        ]
    )

    if expected_divisions is not None:
        if "Division" not in current.columns:
            raise ValueError(
                "Cannot validate divisions: no Division column "
                "exists in the grouping output."
            )

        if current["Division"].null_count():
            raise ValueError(
                "Grouping output contains null divisions."
            )

        expected = set(
            expected_divisions
        )

        actual = set(
            current["Division"].to_list()
        )

        if expected != actual:
            raise ValueError(
                "Grouping divisions differ: "
                f"missing={expected - actual}, "
                f"unexpected={actual - expected}"
            )

    return track_grouping_history(
        current_df=current,
        **kwargs,
    )

