"""
Daily transformer history.

Same standalone layout/behavior as the AMI and ITRON history trackers.

Tracks:
- added transformers
- removed transformers
- modified transformers
- every changed field
- complete BEFORE / AFTER rows
- coordinate changes
- FLOC changes
- kVA changes
- Division changes

current_df must ALREADY be standardized.
"""

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import polars as pl


# ============================================================
# PROJECT PATH AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

TRANSFORMER_HISTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "transformers"
    / "history"
)


# Preferred transformer identity.
#
# SAP ID is preferred because the FLOC itself could theoretically change.
# If SAP ID is unavailable/non-unique, GIS ID is tried next.
# FLOC is used if it is itself unique.

TRANSFORMER_KEY_CANDIDATES = (
    ("sap_id",),
    ("gis_id",),
    ("transformer_floc",),
    ("transformer_floc", "sap_id"),
    ("transformer_floc", "gis_id"),
)


TRANSFORMER_SUMMARY_SCHEMA = {
    "date": pl.Date,
    "transformer_count": pl.Int64,
    "added": pl.Int64,
    "removed": pl.Int64,
    "modified": pl.Int64,
    "field_changes": pl.Int64,
    "floc_changed": pl.Int64,
    "coordinates_changed": pl.Int64,
    "kva_changed": pl.Int64,
    "division_changed": pl.Int64,
    "changes_detected": pl.Boolean,
    "key_columns": pl.String,
}


# ============================================================
# OLD COLUMN NAMES
# ============================================================
#
# ONLY used when reading OLD snapshots/history.
#
# current_df is expected to already have gone through your
# standardization stage.
# ============================================================

TRANSFORMER_OLD_NAMES = {
    "FuncLoc": "transformer_floc",
    "FUNCLOC": "transformer_floc",
    "Location_Y": "latitude",
    "Location_X": "longitude",
    "Transformer_kVA": "transformer_kva",
    "SAPID": "sap_id",
    "GISID": "gis_id",
    "DivName": "Division",
    "division": "Division",
    "opco": "OPCO",

}


# ============================================================
# HELPERS
# ============================================================

def transformerold_names(frame):
    """
    Standardize OLD history only.

    Does NOT modify the current standardized DataFrame.
    """
    names = (
        frame.collect_schema().names()
        if isinstance(frame, pl.LazyFrame)
        else frame.columns
    )

    renames = {
        old: new
        for old, new in TRANSFORMER_OLD_NAMES.items()
        if old in names and new not in names
    }

    if len(set(renames.values())) != len(renames):
        raise ValueError(
            "Multiple old transformer columns map to the same "
            "standardized column."
        )

    return frame.rename(renames)


def transformersave(frame, path):
    """
    Safely write a parquet file.

    The destination is replaced only after the complete temporary
    parquet has successfully been written.
    """
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    frame.write_parquet(
        temporary,
        compression="zstd",
    )

    temporary.replace(path)


def transformerread_summary(root):
    """
    Read transformer_daily_summary.parquet.
    """
    path = (
        Path(root)
        / "transformer_daily_summary.parquet"
    )

    if not path.exists():
        return pl.DataFrame(
            schema=TRANSFORMER_SUMMARY_SCHEMA
        )

    summary = pl.read_parquet(path)

    # --------------------------------------------------------
    # SMALL COMPATIBILITY WITH OLDER HISTORY NAMES
    # --------------------------------------------------------

    renames = {}

    if (
        "date" not in summary.columns
        and "snapshot_date" in summary.columns
    ):
        renames["snapshot_date"] = "date"

    if "transformer_count" not in summary.columns:
        for old_name in (
            "row_count",
            "entity_count",
            "current_count",
        ):
            if old_name in summary.columns:
                renames[old_name] = "transformer_count"
                break

    for old_name, new_name in (
        ("added_count", "added"),
        ("removed_count", "removed"),
        ("modified_count", "modified"),
    ):
        if (
            new_name not in summary.columns
            and old_name in summary.columns
        ):
            renames[old_name] = new_name

    if renames:
        summary = summary.rename(renames)

    required = {
        "date",
        "transformer_count",
        "added",
        "removed",
        "modified",
    }

    missing = required - set(summary.columns)

    if missing:
        raise ValueError(
            "Transformer summary is missing columns: "
            f"{sorted(missing)}"
        )

    summary = summary.with_columns(
        pl.col("date")
        .cast(pl.String)
        .str.slice(0, 10)
        .str.to_date()
    )

    if summary["date"].null_count():
        raise ValueError(
            "Transformer summary contains null dates."
        )

    if summary["date"].n_unique() != summary.height:
        raise ValueError(
            "Transformer summary dates must be unique."
        )

    # Add newer counters if older history did not contain them.

    summary = summary.with_columns(
        [
            pl.lit(
                None,
                dtype=dtype,
            ).alias(column)
            for column, dtype in TRANSFORMER_SUMMARY_SCHEMA.items()
            if column not in summary.columns
        ]
    )

    return (
        summary
        .select(
            list(TRANSFORMER_SUMMARY_SCHEMA)
        )
        .cast(
            TRANSFORMER_SUMMARY_SCHEMA
        )
        .sort("date")
    )


def transformerkey_text(key_columns):
    return "|".join(key_columns)


def transformersaved_key(summary):
    """
    Retrieve the key selected on the first successful run.
    """
    if summary.is_empty():
        return None

    values = (
        summary["key_columns"]
        .drop_nulls()
        .unique()
        .to_list()
    )

    values = [
        value
        for value in values
        if str(value).strip()
    ]

    if not values:
        return None

    if len(values) != 1:
        raise ValueError(
            "Transformer history contains multiple "
            f"entity-key definitions: {values}"
        )

    return tuple(
        str(values[0]).split("|")
    )


def normalizerequested_key(key_columns):
    """
    Normalize manually supplied key_columns.
    """
    if key_columns is None:
        return None

    if isinstance(key_columns, str):
        key_columns = (key_columns,)
    else:
        key_columns = tuple(key_columns)

    if not key_columns:
        raise ValueError(
            "key_columns cannot be empty."
        )

    if len(set(key_columns)) != len(key_columns):
        raise ValueError(
            "key_columns contains duplicates."
        )

    return key_columns


def transformerkey_is_valid(
    frame,
    key_columns,
):
    """
    Check that a proposed transformer key is:

    - present
    - non-null
    - nonblank
    - unique
    """
    if any(
        column not in frame.columns
        for column in key_columns
    ):
        return False

    normalized = frame.select(
        [
            (
                pl.col(column).fill_nan(None)
                if frame.schema[column].is_float()
                else pl.col(column)
            )
            .cast(pl.String)
            .str.strip_chars()
            .alias(column)
            for column in key_columns
        ]
    )

    for column in key_columns:
        values = normalized[column]

        if values.null_count():
            return False

        if values.eq("").any():
            return False

    unique_count = normalized.select(
        pl.struct(
            list(key_columns)
        ).n_unique()
    ).item()

    return unique_count == frame.height


def choosetransformer_key(
    current,
    old,
    requested_key=None,
    saved_key=None,
):
    """
    Select the transformer entity key.

    Once history exists, the key cannot change.
    """
    requested_key = normalizerequested_key(
        requested_key
    )

    # --------------------------------------------------------
    # EXISTING HISTORY: USE SAVED KEY
    # --------------------------------------------------------

    if saved_key is not None:
        if (
            requested_key is not None
            and tuple(requested_key) != tuple(saved_key)
        ):
            raise ValueError(
                "Transformer history key cannot change.\n"
                f"Saved key: {saved_key}\n"
                f"Requested key: {requested_key}"
            )

        chosen = tuple(saved_key)

    # --------------------------------------------------------
    # USER EXPLICITLY PROVIDED KEY
    # --------------------------------------------------------

    elif requested_key is not None:
        chosen = tuple(requested_key)

    # --------------------------------------------------------
    # FIRST RUN: FIND BEST UNIQUE KEY
    # --------------------------------------------------------

    else:
        chosen = None

        for candidate in TRANSFORMER_KEY_CANDIDATES:
            if not transformerkey_is_valid(
                current,
                candidate,
            ):
                continue

            if (
                old.height
                and not transformerkey_is_valid(
                    old,
                    candidate,
                )
            ):
                continue

            chosen = candidate
            break

        if chosen is None:
            raise ValueError(
                "Could not automatically find a unique "
                "transformer identifier.\n\n"
                "Try specifying one manually, for example:\n\n"
                "key_columns=('sap_id',)\n\n"
                "or:\n\n"
                "key_columns=('transformer_floc', 'sap_id')"
            )

    # --------------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------------

    if not transformerkey_is_valid(
        current,
        chosen,
    ):
        raise ValueError(
            f"Current transformer key {chosen} "
            "must be unique, non-null and nonblank."
        )

    if (
        old.height
        and not transformerkey_is_valid(
            old,
            chosen,
        )
    ):
        raise ValueError(
            f"Previous transformer key {chosen} "
            "must be unique, non-null and nonblank."
        )

    return tuple(chosen)


def preparetransformers(
    frame,
    description,
    key_columns,
):
    """
    Validate a standardized transformer DataFrame.
    """
    missing = (
        set(key_columns)
        - set(frame.columns)
    )

    if missing:
        raise KeyError(
            f"{description} is missing key columns: "
            f"{sorted(missing)}"
        )

    reserved = {
        "change_date",
        "change_type",
        "row_version",
        "run_id",
        "changed_at",
    }

    conflicts = reserved.intersection(
        frame.columns
    )

    conflicts.update(
        column
        for column in frame.columns
        if column.endswith("__old")
    )

    if conflicts:
        raise ValueError(
            f"{description} contains reserved "
            f"history columns: {sorted(conflicts)}"
        )

    # History key values are stored consistently as strings.

    frame = frame.with_columns(
        [
            (
                pl.col(column).fill_nan(None)
                if frame.schema[column].is_float()
                else pl.col(column)
            )
            .cast(pl.String)
            .str.strip_chars()
            .alias(column)
            for column in key_columns
        ]
    )

    for column in key_columns:
        values = frame[column]

        if (
            values.null_count()
            or values.eq("").any()
        ):
            raise ValueError(
                f"{description}: '{column}' "
                "contains null/blank values."
            )

    unique_count = frame.select(
        pl.struct(
            list(key_columns)
        ).n_unique()
    ).item()

    if unique_count != frame.height:
        duplicates = (
            frame
            .group_by(
                list(key_columns)
            )
            .len()
            .filter(
                pl.col("len") > 1
            )
            .sort(
                "len",
                descending=True,
            )
            .head(10)
        )

        raise ValueError(
            f"{description}: transformer key "
            f"{key_columns} is not unique.\n\n"
            f"Example duplicates:\n{duplicates}"
        )

    return frame


def transformerchange_schema(
    key_columns,
):
    """
    Field-level history schema.
    """
    return {
        "change_date": pl.String,
        **{
            column: pl.String
            for column in key_columns
        },
        "change_type": pl.String,
        "field": pl.String,
        "previous_value": pl.String,
        "current_value": pl.String,
        "run_id": pl.String,
        "changed_at": pl.String,
    }


# ============================================================
# TRACK TRANSFORMER HISTORY
# ============================================================

def track_transformer_history(
    current_df,
    snapshot_date=None,
    *,
    history_dir=TRANSFORMER_HISTORY_DIR,
    key_columns=None,
):
    """
    Record one COMPLETE standardized transformer extract per day.

    Parameters
    ----------
    current_df:
        COMPLETE standardized transformer DataFrame.

    snapshot_date:
        Optional date. Defaults to today.

    history_dir:
        Transformer history directory.

    key_columns:
        Optional manual transformer identity.
        Normally leave this as None.

        The tracker tries:

        sap_id
        gis_id
        transformer_floc
        transformer_floc + sap_id
        transformer_floc + gis_id

    FIRST RUN
    ---------
    Saves every transformer as INITIAL.

    No additions/removals/modifications are reported because
    there is no earlier comparison snapshot.

    LATER RUNS
    ----------
    Tracks:

    ADDED
    REMOVED
    MODIFIED

    Every supplied non-key field is compared automatically.

    FULL ROW HISTORY
    ----------------
    changed_rows contains:

    INITIAL
    ADDED
    REMOVED
    MODIFIED_BEFORE
    MODIFIED_AFTER

    IMPORTANT
    ---------
    current_df must already have gone through your existing
    transformer standardization.
    """
    day = date.fromisoformat(
        str(
            snapshot_date
            or date.today()
        )[:10]
    )

    root = Path(history_dir)

    previous_file = (
        root
        / "previous_transformer_snapshot.parquet"
    )

    summary_file = (
        root
        / "transformer_daily_summary.parquet"
    )

    pending_file = (
        root
        / f"pending_transformer_{day}.parquet"
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

    # ========================================================
    # READ EXISTING SUMMARY
    # ========================================================

    summary = transformerread_summary(
        root
    )

    saved_key = transformersaved_key(
        summary
    )

    # ========================================================
    # RECOVER INTERRUPTED FINAL SAVE
    # ========================================================

    for pending in sorted(
        root.glob(
            "pending_transformer_*.parquet"
        )
    ):
        pending_day = date.fromisoformat(
            pending.stem.removeprefix(
                "pending_transformer_"
            )
        )

        if pending_day in summary["date"]:
            if pending_day != summary["date"].max():
                raise ValueError(
                    "Unexpected stale pending "
                    f"transformer file: {pending}"
                )

            saved_changes = (
                root
                / "changes"
                / f"{pending_day}.parquet"
            )

            saved_rows = (
                root
                / "changed_rows"
                / f"{pending_day}.parquet"
            )

            if not saved_changes.exists():
                raise FileNotFoundError(
                    "Committed transformer history "
                    f"is incomplete: {saved_changes}"
                )

            if not saved_rows.exists():
                raise FileNotFoundError(
                    "Committed transformer history "
                    f"is incomplete: {saved_rows}"
                )

            pending.replace(
                previous_file
            )

        elif pending_day != day:
            raise ValueError(
                "Finish the interrupted "
                f"{pending_day} transformer "
                f"history run first: {pending}"
            )

    # ========================================================
    # SAME DATE: RETURN SAVED RESULT
    # ========================================================

    if day in summary["date"]:
        if not changes_file.exists():
            raise FileNotFoundError(
                f"Missing saved changes: {changes_file}"
            )

        if not rows_file.exists():
            raise FileNotFoundError(
                f"Missing saved changed rows: {rows_file}"
            )

        print(
            f"Transformer history already processed for {day}; "
            "returning saved changes."
        )

        return pl.read_parquet(
            changes_file
        )

    # ========================================================
    # PREVENT BACKDATED RUNS
    # ========================================================

    if (
        summary.height
        and day <= summary["date"].max()
    ):
        raise ValueError(
            "Use a date after the last "
            "recorded transformer history date: "
            f"{summary['date'].max()}."
        )

    # ========================================================
    # DETECT PARTIAL HISTORY
    # ========================================================

    if (
        (
            changes_file.exists()
            or rows_file.exists()
        )
        and not pending_file.exists()
    ):
        raise ValueError(
            f"Transformer history files already exist for {day}, "
            "but the summary entry is missing."
        )

    # ========================================================
    # CURRENT DATA
    # ========================================================

    current = (
        current_df.clone()
        if isinstance(
            current_df,
            pl.DataFrame,
        )
        else pl.from_pandas(
            current_df
        )
    )

    if current.is_empty():
        raise ValueError(
            "Current transformer DataFrame is empty. "
            "History was not changed."
        )

    # ========================================================
    # PREVIOUS DATA
    # ========================================================

    first_run = not previous_file.exists()

    if (
        first_run
        and summary.height
    ):
        raise FileNotFoundError(
            "Transformer summary exists but "
            "the comparison snapshot is missing: "
            f"{previous_file}"
        )

    if (
        not first_run
        and summary.is_empty()
    ):
        raise ValueError(
            "Transformer comparison snapshot exists "
            "but its daily summary is missing."
        )

    old = (
        current.head(0)
        if first_run
        else transformerold_names(
            pl.read_parquet(
                previous_file
            )
        )
    )

    # ========================================================
    # CHOOSE ENTITY KEY
    # ========================================================

    chosen_key = choosetransformer_key(
        current=current,
        old=old,
        requested_key=key_columns,
        saved_key=saved_key,
    )

    current = preparetransformers(
        current,
        "Current standardized transformer data",
        chosen_key,
    )

    old = preparetransformers(
        old,
        "Previous transformer snapshot",
        chosen_key,
    )

    # ========================================================
    # CHECK COLUMNS
    # ========================================================
        # ========================================================
    # CHECK COLUMNS
    # ========================================================

    previous_only = sorted(
        set(old.columns) - set(current.columns)
    )

    current_only = sorted(
        set(current.columns) - set(old.columns)
    )

    if previous_only or current_only:
        print("\nTransformer history column mismatch:")
        print(f"Previous-only columns: {previous_only}")
        print(f"Current-only columns: {current_only}")

        raise ValueError(
            "Align standardized transformer columns before tracking.\n\n"
            f"Only previous:\n{previous_only}\n\n"
            f"Only current:\n{current_only}"
        )

    old = old.select(current.columns)
    if set(old.columns) != set(current.columns):
        raise ValueError(
            "Align standardized transformer columns before tracking.\n\n"
            f"Only previous:\n"
            f"{sorted(set(old.columns) - set(current.columns))}\n\n"
            f"Only current:\n"
            f"{sorted(set(current.columns) - set(old.columns))}"
        )

    old = old.select(
        current.columns
    )

    # ========================================================
    # HANDLE ALL-NULL COLUMNS
    # ========================================================

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
            "Transformer column types changed. "
            "Align input types before tracking.\n\n"
            f"Previous schema:\n{old.schema}\n\n"
            f"Current schema:\n{current.schema}"
        )

    # ========================================================
    # COMPARE SNAPSHOTS
    # ========================================================

    key_list = list(
        chosen_key
    )

    # --------------------------------------------------------
    # ADDED
    # --------------------------------------------------------

    added = current.join(
        old.select(
            key_list
        ),
        on=key_list,
        how="anti",
    )

    # --------------------------------------------------------
    # REMOVED
    # --------------------------------------------------------

    removed = old.join(
        current.select(
            key_list
        ),
        on=key_list,
        how="anti",
    )

    # --------------------------------------------------------
    # COMMON TRANSFORMERS
    # --------------------------------------------------------

    joined = current.join(
        old,
        on=key_list,
        how="inner",
        suffix="__old",
    )

    fields = [
        column
        for column in current.columns
        if column not in chosen_key
    ]

    # --------------------------------------------------------
    # FIELD COMPARISONS
    # --------------------------------------------------------

    different = {}

    for column in fields:
        new_value = pl.col(
            column
        )

        old_value = pl.col(
            column + "__old"
        )

        # Treat NaN as missing.

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

    if different:
        modified = joined.filter(
            pl.any_horizontal(
                list(
                    different.values()
                )
            )
        )
    else:
        modified = joined.head(0)

    modified_keys = modified.select(
        key_list
    )

    # ========================================================
    # RUN METADATA
    # ========================================================

    stamp = {
        "change_date": str(day),
        "run_id": uuid4().hex[:12],
        "changed_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    metadata = {
        name: pl.lit(value)
        for name, value in stamp.items()
    }

    null = pl.lit(
        None,
        dtype=pl.String,
    )

    change_schema = transformerchange_schema(
        chosen_key
    )

    # ========================================================
    # FIELD-LEVEL CHANGE LOG
    # ========================================================

    logs = [
        pl.DataFrame(
            schema=change_schema
        )
    ]

    # --------------------------------------------------------
    # ADDED / REMOVED
    # --------------------------------------------------------

    if not first_run:
        for frame, kind in (
            (added, "ADDED"),
            (removed, "REMOVED"),
        ):
            logs.append(
                frame
                .select(
                    key_list
                )
                .with_columns(
                    change_type=pl.lit(
                        kind
                    ),
                    field=null,
                    previous_value=null,
                    current_value=null,
                )
            )

    # --------------------------------------------------------
    # MODIFIED FIELD EVENTS
    # --------------------------------------------------------

    for field in fields:
        logs.append(
            modified
            .filter(
                different[field]
            )
            .select(
                *key_list,
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
                    field + "__old"
                )
                .cast(pl.String)
                .alias(
                    "previous_value"
                ),
                pl.col(
                    field
                )
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
        .with_columns(
            **metadata
        )
        .select(
            list(
                change_schema
            )
        )
        .cast(
            change_schema
        )
    )

    # ========================================================
    # COMPLETE SOURCE RECORDS
    # ========================================================

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
                    modified_keys,
                    on=key_list,
                    how="semi",
                ),
                "MODIFIED_BEFORE",
                "PREVIOUS",
            ),
            (
                current.join(
                    modified_keys,
                    on=key_list,
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
                change_type=pl.lit(
                    change_type
                ),
                row_version=pl.lit(
                    row_version
                ),
            )
            for (
                frame,
                change_type,
                row_version,
            ) in versions
        ]
    )

    # ========================================================
    # TRANSFORMER-SPECIFIC COUNTERS
    # ========================================================

    def changed_transformer_count(
        columns,
    ):
        expressions = [
            different[column]
            for column in columns
            if column in different
        ]

        if not expressions:
            return None

        return modified.filter(
            pl.any_horizontal(
                expressions
            )
        ).height

    floc_changed = changed_transformer_count(
        ["transformer_floc"]
    )

    coordinates_changed = changed_transformer_count(
        [
            "latitude",
            "longitude",
        ]
    )

    kva_changed = changed_transformer_count(
        ["transformer_kva"]
    )

    division_changed = changed_transformer_count(
        ["Division"]
    )

    field_change_count = (
        changes
        .filter(
            pl.col("change_type")
            == "MODIFIED"
        )
        .height
    )

    # ========================================================
    # DAILY SUMMARY
    # ========================================================

    daily = pl.DataFrame(
        [
            {
                "date": day,
                "transformer_count": current.height,
                "added": (
                    0
                    if first_run
                    else added.height
                ),
                "removed": (
                    0
                    if first_run
                    else removed.height
                ),
                "modified": (
                    0
                    if first_run
                    else modified.height
                ),
                "field_changes": (
                    0
                    if first_run
                    else field_change_count
                ),
                "floc_changed": (
                    0
                    if first_run
                    else floc_changed
                ),
                "coordinates_changed": (
                    0
                    if first_run
                    else coordinates_changed
                ),
                "kva_changed": (
                    0
                    if first_run
                    else kva_changed
                ),
                "division_changed": (
                    0
                    if first_run
                    else division_changed
                ),
                "changes_detected": (
                    False
                    if first_run
                    else bool(
                        added.height
                        or removed.height
                        or modified.height
                    )
                ),
                "key_columns": transformerkey_text(
                    chosen_key
                ),
            }
        ],
        schema=TRANSFORMER_SUMMARY_SCHEMA,
    )

    updated_summary = pl.concat(
        [
            summary,
            daily,
        ],
        how="diagonal_relaxed",
    )

    # ========================================================
    # SAFE SAVE ORDER
    # ========================================================
    #
    # IMPORTANT:
    #
    # previous_transformer_snapshot.parquet is NOT replaced
    # until ALL history outputs have successfully been saved.
    #
    # ========================================================

    transformersave(
        current,
        pending_file,
    )

    transformersave(
        changes,
        changes_file,
    )

    transformersave(
        rows,
        rows_file,
    )

    transformersave(
        updated_summary,
        summary_file,
    )

    # Only now advance the comparison baseline.

    pending_file.replace(
        previous_file
    )

    # ========================================================
    # CONSOLE SUMMARY
    # ========================================================

    result = daily.row(
        0,
        named=True,
    )

    print(
        f"\nTransformer history completed for {day}"
        + (
            " - initial baseline"
            if first_run
            else ""
        )
    )

    print(
        f"Entity key: {chosen_key}"
    )

    print(
        f"Transformers: {current.height:,}"
    )

    print(
        f"Added: {result['added']:,} | "
        f"Removed: {result['removed']:,} | "
        f"Modified: {result['modified']:,}"
    )

    print(
        f"Field changes: "
        f"{result['field_changes']:,}"
    )

    print(
        f"FLOC changed: "
        f"{result['floc_changed']} | "
        f"Coordinates changed: "
        f"{result['coordinates_changed']} | "
        f"kVA changed: "
        f"{result['kva_changed']} | "
        f"Division changed: "
        f"{result['division_changed']}"
    )

    print(
        f"Changes detected: "
        f"{result['changes_detected']}"
    )

    print(
        f"History: {root}"
    )

    return changes


# ============================================================
# GET ONE TRANSFORMER'S HISTORY
# ============================================================

def get_transformer_history(
    transformer_floc=None,
    *,
    sap_id=None,
    gis_id=None,
    full_rows=True,
    history_dir=TRANSFORMER_HISTORY_DIR,
):
    """
    Retrieve the complete saved history of a transformer.

    Supply exactly ONE:

    transformer_floc=
    sap_id=
    gis_id=

    full_rows=True:

    INITIAL
    ADDED
    REMOVED
    MODIFIED_BEFORE
    MODIFIED_AFTER

    full_rows=False:

    AMI-style field-level changes.

    The function identifies the transformer's stable history key first,
    then follows that entity across all dates.
    """
    supplied = [
        (column, value)
        for column, value in (
            (
                "transformer_floc",
                transformer_floc,
            ),
            (
                "sap_id",
                sap_id,
            ),
            (
                "gis_id",
                gis_id,
            ),
        )
        if value is not None
    ]

    if len(supplied) != 1:
        raise ValueError(
            "Provide exactly one of "
            "transformer_floc, sap_id or gis_id."
        )

    query_column, query_value = supplied[0]

    query_value = str(
        query_value
    ).strip()

    root = Path(
        history_dir
    )

    summary = transformerread_summary(
        root
    )

    if summary.is_empty():
        return pl.DataFrame()

    chosen_key = transformersaved_key(
        summary
    )

    if chosen_key is None:
        raise ValueError(
            "Transformer history does not "
            "contain its saved entity key."
        )

    key_list = list(
        chosen_key
    )

    # ========================================================
    # FIND STABLE KEY(S) ASSOCIATED WITH REQUESTED IDENTIFIER
    # ========================================================

    matching_key_queries = []

    for day in summary["date"]:
        path = (
            root
            / "changed_rows"
            / f"{day}.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                "Saved transformer history "
                f"is incomplete: {path}"
            )

        query = transformerold_names(
            pl.scan_parquet(
                path
            )
        )

        columns = (
            query
            .collect_schema()
            .names()
        )

        if query_column not in columns:
            continue

        matching_key_queries.append(
            query
            .filter(
                pl.col(
                    query_column
                )
                .cast(pl.String)
                .str.strip_chars()
                == query_value
            )
            .select(
                key_list
            )
        )

    if not matching_key_queries:
        return pl.DataFrame()

    matching_keys = (
        pl.concat(
            matching_key_queries,
            how="diagonal_relaxed",
        )
        .unique()
        .collect()
    )

    if matching_keys.is_empty():
        return pl.DataFrame()

    # ========================================================
    # READ COMPLETE HISTORY FOR THOSE STABLE KEY(S)
    # ========================================================

    folder = (
        "changed_rows"
        if full_rows
        else "changes"
    )

    history_queries = []

    for day in summary["date"]:
        path = (
            root
            / folder
            / f"{day}.parquet"
        )

        if not path.exists():
            raise FileNotFoundError(
                "Saved transformer history "
                f"is incomplete: {path}"
            )

        query = transformerold_names(
            pl.scan_parquet(
                path
            )
        )

        columns = (
            query
            .collect_schema()
            .names()
        )

        missing = (
            set(key_list)
            - set(columns)
        )

        if missing:
            raise KeyError(
                "Saved transformer history "
                "is missing key columns "
                f"{sorted(missing)}: {path}"
            )

        query = query.with_columns(
            [
                pl.col(column)
                .cast(pl.String)
                .str.strip_chars()
                .alias(column)
                for column in key_list
            ]
        )

        history_queries.append(
            query.join(
                matching_keys.lazy(),
                on=key_list,
                how="semi",
            )
        )

    result = (
        pl.concat(
            history_queries,
            how="diagonal_relaxed",
        )
        .collect()
    )

    if result.is_empty():
        return result

    return result.sort(
        "change_date",
        maintain_order=True,
    )


# ============================================================
# GET ONE DAY'S TRANSFORMER CHANGES
# ============================================================

def get_transformer_daily_changes(
    snapshot_date=None,
    *,
    full_rows=False,
    history_dir=TRANSFORMER_HISTORY_DIR,
):
    """
    Return a committed day's transformer changes.

    If snapshot_date is omitted, returns the latest date.
    """
    root = Path(
        history_dir
    )

    summary = transformerread_summary(
        root
    )

    if summary.is_empty():
        return pl.DataFrame()

    day = (
        summary["date"].max()
        if snapshot_date is None
        else date.fromisoformat(
            str(snapshot_date)[:10]
        )
    )

    if day not in summary["date"]:
        raise ValueError(
            "No committed transformer "
            f"history exists for {day}."
        )

    folder = (
        "changed_rows"
        if full_rows
        else "changes"
    )

    path = (
        root
        / folder
        / f"{day}.parquet"
    )

    if not path.exists():
        raise FileNotFoundError(
            "Saved transformer history "
            f"is incomplete: {path}"
        )

    return transformerold_names(
        pl.read_parquet(
            path
        )
    )
