"""Division-aware history for the AMI-to-ITRON grouping result."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

HISTORY_DIR = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "grouping"
    / "itron"
    / "history"
)

KEY = "meter_number"
GROUP = "tx_group_id"

GROUP_COLUMNS = [
    KEY,
    "division",
    GROUP,
    "tx_group_size",
    "tx_group_source",
    "meter_type",
]

CHANGE_COLUMNS = [
    "change_date",
    KEY,
    "change_type",
    "field",
    "previous_value",
    "current_value",
    "run_id",
    "changed_at",
]

ASSIGNMENT_COLUMNS = [
    "change_date",
    KEY,
    "division",
    "event",
    "previous_group",
    "current_group",
    "run_id",
    "changed_at",
]

SUMMARY_COLUMNS = [
    "date",
    "division",
    "meter_count",
    "assigned_meter_count",
    "unassigned_meter_count",
    "added",
    "removed",
    "modified",
    "assignment_changes",
    "assignments_changed",
]


def _atomic_write(
    frame: pd.DataFrame,
    path: Path,
) -> None:
    """Write and read-check a temporary Parquet before replacing its target."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        f".{path.stem}.{uuid4().hex}.tmp.parquet"
    )

    try:
        frame.to_parquet(
            temporary,
            index=False,
        )

        checked = pd.read_parquet(
            temporary
        )

        if (
            len(checked) != len(frame)
            or list(checked.columns) != list(frame.columns)
        ):
            raise OSError(
                f"Temporary history file failed validation: {temporary}"
            )

        temporary.replace(path)

    finally:
        if temporary.exists():
            temporary.unlink()


def _prepare(
    frame: pd.DataFrame,
    description: str,
) -> pd.DataFrame:
    """Standardize and validate a grouping DataFrame."""

    current = frame.rename(
        columns={
            "Division": "division",
        }
    ).copy()

    missing = set(GROUP_COLUMNS) - set(current.columns)

    if missing:
        raise ValueError(
            f"{description} missing columns: {sorted(missing)}"
        )

    current = current[GROUP_COLUMNS].copy()

    current[KEY] = (
        current[KEY]
        .astype("string")
        .str.strip()
    )

    current["division"] = (
        current["division"]
        .astype("string")
        .str.strip()
    )

    current[GROUP] = (
        current[GROUP]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    current["tx_group_source"] = (
        current["tx_group_source"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    current["tx_group_size"] = pd.to_numeric(
        current["tx_group_size"],
        errors="raise",
    ).astype("Int64")

    if (
        current[KEY].isna().any()
        or current[KEY].eq("").any()
    ):
        raise ValueError(
            f"{description}: meter_number must be non-null and nonblank."
        )

    if not current[KEY].is_unique:
        raise ValueError(
            f"{description}: meter_number must be unique."
        )

    if (
        current["division"].isna().any()
        or current["division"].eq("").any()
    ):
        raise ValueError(
            f"{description}: division must be non-null and nonblank."
        )

    if not current["tx_group_source"].eq("itron").all():
        raise ValueError(
            f"{description}: tx_group_source must always be 'itron'."
        )

    return (
        current
        .sort_values(KEY)
        .reset_index(drop=True)
    )


def _different(
    current: pd.Series,
    previous: pd.Series,
) -> pd.Series:
    """Return True where two nullable Series differ."""

    return ~(
        current.eq(previous).fillna(False)
        | (
            current.isna()
            & previous.isna()
        )
    )


def _text(value):
    """Convert a value to text while preserving missing values as None."""

    return None if pd.isna(value) else str(value)


def _read_summary(
    root: Path,
) -> pd.DataFrame:
    """Read and validate the grouping daily summary."""

    path = (
        root
        / "grouping_daily_summary.parquet"
    )

    if not path.exists():
        return pd.DataFrame(
            columns=SUMMARY_COLUMNS
        )

    summary = pd.read_parquet(
        path
    )

    missing = (
        set(SUMMARY_COLUMNS)
        - set(summary.columns)
    )

    if missing:
        raise ValueError(
            "Grouping history summary missing columns: "
            f"{sorted(missing)}"
        )

    summary = summary[
        SUMMARY_COLUMNS
    ].copy()

    summary["date"] = (
        summary["date"]
        .astype(str)
        .str[:10]
    )

    return summary


def _daily_paths(
    root: Path,
    division: str,
    day: str,
) -> dict[str, Path]:
    """Build the daily history paths for one division."""

    return {
        name: (
            root
            / name
            / division
            / f"{day}.parquet"
        )
        for name in (
            "changes",
            "changed_rows",
            "assignment_changes",
        )
    }


def _stamp_rows(
    frame: pd.DataFrame,
    stamp: dict[str, str],
    change_type: str,
    row_version: str,
) -> pd.DataFrame:
    """Add history metadata to complete grouping rows."""

    result = frame.copy()

    for column, value in stamp.items():
        result[column] = value

    result["change_type"] = change_type
    result["row_version"] = row_version

    return result


def _track_division(
    current_df: pd.DataFrame,
    division: str,
    day: str,
    root: Path,
) -> pd.DataFrame:
    """Compare and commit one division without loading another division."""

    current = _prepare(
        current_df,
        f"Current {division} grouping",
    )

    previous_file = (
        root
        / "previous_state"
        / f"{division}.parquet"
    )

    pending_file = (
        root
        / "pending"
        / f"{division}_{day}.parquet"
    )

    daily = _daily_paths(
        root,
        division,
        day,
    )

    summary_file = (
        root
        / "grouping_daily_summary.parquet"
    )

    summary = _read_summary(
        root
    )

    pending_directory = (
        root
        / "pending"
    )

    pending_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Recover interrupted runs for this division.
    for pending in sorted(
        pending_directory.glob(
            f"{division}_*.parquet"
        )
    ):
        pending_day = pending.stem.removeprefix(
            f"{division}_"
        )

        if not summary.empty:
            committed = (
                summary["division"].eq(division)
                & summary["date"].eq(pending_day)
            )
        else:
            committed = pd.Series(
                False,
                index=summary.index,
                dtype=bool,
            )

        if bool(committed.any()):
            committed_files = _daily_paths(
                root,
                division,
                pending_day,
            )

            if any(
                not path.exists()
                for path in committed_files.values()
            ):
                raise FileNotFoundError(
                    f"Committed {division} grouping history is incomplete "
                    f"for {pending_day}."
                )

            previous_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            pending.replace(
                previous_file
            )

        elif pending_day != day:
            raise ValueError(
                f"Finish the interrupted {division} grouping history run "
                f"for {pending_day} first."
            )

    if not summary.empty:
        processed = (
            summary["division"].eq(division)
            & summary["date"].eq(day)
        )
    else:
        processed = pd.Series(
            False,
            index=summary.index,
            dtype=bool,
        )

    if pending_file.exists():
        pending_file.unlink()

    if bool(processed.any()):
        return pd.read_parquet(
            daily["changes"]
        )

    division_summary = summary.loc[
        summary["division"].eq(division)
    ]

    first_run = not previous_file.exists()

    if first_run and not division_summary.empty:
        raise FileNotFoundError(
            f"{division} history summary exists without {previous_file}."
        )

    if not first_run and division_summary.empty:
        raise ValueError(
            f"{division} previous grouping state exists "
            "without summary history."
        )

    if (
        not division_summary.empty
        and day <= division_summary["date"].max()
    ):
        raise ValueError(
            f"{division}: use a date after "
            f"{division_summary['date'].max()}."
        )

    if first_run:
        previous = current.head(
            0
        )

    else:
        previous = _prepare(
            pd.read_parquet(
                previous_file
            ),
            f"Previous {division} grouping",
        )

    if set(previous.columns) != set(current.columns):
        raise ValueError(
            f"{division}: current and previous grouping columns differ."
        )

    current_ids = set(
        current[KEY]
    )

    previous_ids = set(
        previous[KEY]
    )

    added = current.loc[
        current[KEY].isin(
            current_ids - previous_ids
        )
    ].copy()

    removed = previous.loc[
        previous[KEY].isin(
            previous_ids - current_ids
        )
    ].copy()

    joined = current.merge(
        previous,
        on=KEY,
        how="inner",
        suffixes=("", "__old"),
    )

    fields = [
        column
        for column in GROUP_COLUMNS
        if column != KEY
    ]

    differences = {
        field: _different(
            joined[field],
            joined[f"{field}__old"],
        )
        for field in fields
    }

    modified_mask = pd.Series(
        False,
        index=joined.index,
        dtype=bool,
    )

    for field_difference in differences.values():
        modified_mask |= field_difference

    modified = joined.loc[
        modified_mask
    ].copy()

    run_id = uuid4().hex[:12]

    changed_at = datetime.now(
        timezone.utc
    ).isoformat()

    stamp = {
        "change_date": day,
        "run_id": run_id,
        "changed_at": changed_at,
    }

    # ========================================================
    # BUILD FIELD-LEVEL CHANGES
    # ========================================================

    change_records = []

    if not first_run:
        for meter_number in added:
            change_records.append(
                {
                    KEY: meter_number,
                    "change_type": "ADDED",
                    "field": None,
                    "previous_value": None,
                    "current_value": None,
                }
            )

        for meter_number in removed:
            change_records.append(
                {
                    KEY: meter_number,
                    "change_type": "REMOVED",
                    "field": None,
                    "previous_value": None,
                    "current_value": None,
                }
            )

        for field, field_difference in differences.items():
            changed_field_rows = joined.loc[
                field_difference
            ]

            for _, row in changed_field_rows.iterrows():
                change_records.append(
                    {
                        KEY: row[KEY],
                        "change_type": "MODIFIED",
                        "field": field,
                        "previous_value": _text(
                            row[f"{field}__old"]
                        ),
                        "current_value": _text(
                            row[field]
                        ),
                    }
                )

    changes = pd.DataFrame(
        change_records
    )

    if changes.empty:
        changes = pd.DataFrame(
            columns=CHANGE_COLUMNS
        )

    else:
        for column, value in stamp.items():
            changes[column] = value

        changes = changes[
            CHANGE_COLUMNS
        ]

    # ========================================================
    # BUILD ASSIGNMENT EVENTS
    # ========================================================

    assignment_records = []

    if not first_run:
        for _, row in added.iterrows():
            assignment_records.append(
                {
                    KEY: row[KEY],
                    "division": division,
                    "event": (
                        "ADDED_ASSIGNED"
                        if pd.notna(row[GROUP])
                        else "ADDED_UNASSIGNED"
                    ),
                    "previous_group": None,
                    "current_group": _text(
                        row[GROUP]
                    ),
                }
            )

        for _, row in removed.iterrows():
            assignment_records.append(
                {
                    KEY: row[KEY],
                    "division": division,
                    "event": (
                        "METER_REMOVED_ASSIGNED"
                        if pd.notna(row[GROUP])
                        else "METER_REMOVED_UNASSIGNED"
                    ),
                    "previous_group": _text(
                        row[GROUP]
                    ),
                    "current_group": None,
                }
            )

        group_changed = differences[
            GROUP
        ]

        for _, row in joined.loc[
            group_changed
        ].iterrows():
            before = _text(
                row[f"{GROUP}__old"]
            )

            after = _text(
                row[GROUP]
            )

            if before is None:
                event = "ASSIGNMENT_ADDED"

            elif after is None:
                event = "ASSIGNMENT_REMOVED"

            else:
                event = "GROUP_CHANGED"

            assignment_records.append(
                {
                    KEY: row[KEY],
                    "division": division,
                    "event": event,
                    "previous_group": before,
                    "current_group": after,
                }
            )

    assignment_changes = pd.DataFrame(
        assignment_records
    )

    if assignment_changes.empty:
        assignment_changes = pd.DataFrame(
            columns=ASSIGNMENT_COLUMNS
        )

    else:
        for column, value in stamp.items():
            assignment_changes[column] = value

        assignment_changes = assignment_changes[
            ASSIGNMENT_COLUMNS
        ]

    expected_group_changes = set(
        joined.loc[
            differences[GROUP],
            KEY,
        ]
    )

    captured_group_changes = set(
        assignment_changes.loc[
            assignment_changes["event"].isin(
                [
                    "ASSIGNMENT_ADDED",
                    "ASSIGNMENT_REMOVED",
                    "GROUP_CHANGED",
                ]
            ),
            KEY,
        ]
    )

    if expected_group_changes != captured_group_changes:
        raise ValueError(
            f"{division}: tx_group_id changes were not fully captured."
        )

    # ========================================================
    # BUILD COMPLETE CHANGED ROWS
    # ========================================================

    if first_run:
        row_parts = [
            _stamp_rows(
                current,
                stamp,
                "INITIAL",
                "CURRENT",
            )
        ]

    else:
        modified_ids = set(
            modified[KEY]
        )

        row_parts = [
            _stamp_rows(
                added,
                stamp,
                "ADDED",
                "CURRENT",
            ),
            _stamp_rows(
                removed,
                stamp,
                "REMOVED",
                "PREVIOUS",
            ),
            _stamp_rows(
                previous.loc[
                    previous[KEY].isin(
                        modified_ids
                    )
                ],
                stamp,
                "MODIFIED_BEFORE",
                "PREVIOUS",
            ),
            _stamp_rows(
                current.loc[
                    current[KEY].isin(
                        modified_ids
                    )
                ],
                stamp,
                "MODIFIED_AFTER",
                "CURRENT",
            ),
        ]

    changed_rows = pd.concat(
        row_parts,
        ignore_index=True,
    )

    # ========================================================
    # BUILD SUMMARY
    # ========================================================

    assigned = int(
        current[GROUP]
        .notna()
        .sum()
    )

    unassigned = int(
        current[GROUP]
        .isna()
        .sum()
    )

    if assigned + unassigned != len(current):
        raise ValueError(
            f"{division}: assignment counts do not reconcile."
        )

    summary_row = pd.DataFrame(
        [
            {
                "date": day,
                "division": division,
                "meter_count": len(current),
                "assigned_meter_count": assigned,
                "unassigned_meter_count": unassigned,
                "added": (
                    0
                    if first_run
                    else len(added)
                ),
                "removed": (
                    0
                    if first_run
                    else len(removed)
                ),
                "modified": (
                    0
                    if first_run
                    else len(modified)
                ),
                "assignment_changes": (
                    0
                    if first_run
                    else len(assignment_changes)
                ),
                "assignments_changed": (
                    False
                    if first_run
                    else not assignment_changes.empty
                ),
            }
        ]
    )[SUMMARY_COLUMNS]

    # ========================================================
    # SAVE HISTORY
    # ========================================================

    _atomic_write(
        current,
        pending_file,
    )

    _atomic_write(
        changes,
        daily["changes"],
    )

    _atomic_write(
        changed_rows,
        daily["changed_rows"],
    )

    _atomic_write(
        assignment_changes,
        daily["assignment_changes"],
    )

    updated_summary = pd.concat(
        [
            summary,
            summary_row,
        ],
        ignore_index=True,
    )

    _atomic_write(
        updated_summary,
        summary_file,
    )

    previous_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    pending_file.replace(
        previous_file
    )

    print(
        f"[{division}] grouping history {day}: "
        f"meters={len(current):,}, "
        f"assigned={assigned:,}, "
        f"unassigned={unassigned:,}, "
        f"assignment events={len(assignment_changes):,}"
        + (
            " (initial baseline)"
            if first_run
            else ""
        )
    )

    return changes


def track_meter_grouping_history(
    group_files: dict[str, Path],
    ami_df: pd.DataFrame,
    snapshot_date=None,
    *,
    history_dir: Path = HISTORY_DIR,
) -> pd.DataFrame:
    """Validate and track all division files without combining populations."""

    day = str(
        snapshot_date
        or date.today()
    )[:10]

    root = Path(
        history_dir
    )

    ami = (
        ami_df.rename(
            columns={
                "MeterID": KEY,
                "Division": "division",
            }
        )[
            [KEY, "division"]
        ]
        .dropna()
        .copy()
    )

    ami[KEY] = (
        ami[KEY]
        .astype(str)
        .str.strip()
    )

    ami["division"] = (
        ami["division"]
        .astype(str)
        .str.strip()
    )

    if (
        ami[KEY].eq("").any()
        or ami["division"].eq("").any()
    ):
        raise ValueError(
            "Filtered AMI contains blank meter_number "
            "or division values."
        )

    if not ami[KEY].is_unique:
        raise ValueError(
            "Filtered AMI must contain one row per meter_number."
        )

    expected_divisions = set(
        ami["division"]
    )

    if set(group_files) != expected_divisions:
        raise ValueError(
            "Grouping divisions differ from filtered AMI: "
            f"missing={expected_divisions - set(group_files)}, "
            f"unexpected={set(group_files) - expected_divisions}."
        )

    all_changes = []

    total_population = 0
    total_assigned = 0
    total_unassigned = 0

    for division in sorted(
        group_files
    ):
        current = _prepare(
            pd.read_parquet(
                group_files[division]
            ),
            f"Current {division} grouping file",
        )

        if set(current["division"]) != {division}:
            raise ValueError(
                f"{division}: grouping file contains another division."
            )

        expected_ids = set(
            ami.loc[
                ami["division"].eq(division),
                KEY,
            ]
        )

        actual_ids = set(
            current[KEY]
        )

        if (
            expected_ids != actual_ids
            or len(current) != len(expected_ids)
        ):
            raise ValueError(
                f"{division}: AMI/grouping population mismatch; "
                f"missing={len(expected_ids - actual_ids):,}, "
                f"unexpected={len(actual_ids - expected_ids):,}."
            )

        total_population += len(
            current
        )

        total_assigned += int(
            current[GROUP]
            .notna()
            .sum()
        )

        total_unassigned += int(
            current[GROUP]
            .isna()
            .sum()
        )

        all_changes.append(
            _track_division(
                current,
                division,
                day,
                root,
            )
        )

    if total_population != len(ami):
        raise ValueError(
            "Sum of division grouping populations does not equal "
            "filtered AMI: "
            f"grouping={total_population:,}, "
            f"AMI={len(ami):,}."
        )

    if (
        total_assigned
        + total_unassigned
        != total_population
    ):
        raise ValueError(
            "Enterprise assigned/unassigned counts do not reconcile."
        )

    print(
        "Grouping history validation passed: "
        f"{total_population:,} meters = "
        f"{total_assigned:,} assigned + "
        f"{total_unassigned:,} unassigned."
    )

    if not all_changes:
        return pd.DataFrame(
            columns=CHANGE_COLUMNS
        )

    return pd.concat(
        all_changes,
        ignore_index=True,
    )[CHANGE_COLUMNS]


def get_grouping_assignment_changes(
    snapshot_date=None,
    *,
    division: str | None = None,
    history_dir: Path = HISTORY_DIR,
) -> pd.DataFrame:
    """Read assignment events across all divisions or one division."""

    root = (
        Path(history_dir)
        / "assignment_changes"
    )

    if division:
        divisions = [
            division
        ]

    else:
        divisions = [
            path.name
            for path in root.glob("*")
            if path.is_dir()
        ]

    frames = []

    for name in sorted(
        divisions
    ):
        division_directory = (
            root
            / name
        )

        if snapshot_date:
            paths = [
                division_directory
                / f"{str(snapshot_date)[:10]}.parquet"
            ]

        else:
            paths = sorted(
                division_directory.glob(
                    "*.parquet"
                )
            )

        frames.extend(
            pd.read_parquet(path)
            for path in paths
            if path.exists()
        )

    if not frames:
        return pd.DataFrame(
            columns=ASSIGNMENT_COLUMNS
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )[ASSIGNMENT_COLUMNS]


def get_meter_grouping_history(
    meter_number,
    *,
    full_rows: bool = False,
    assignments_only: bool = False,
    history_dir: Path = HISTORY_DIR,
) -> pd.DataFrame:
    """Read one meter's field, row, or assignment history across divisions."""

    if full_rows and assignments_only:
        raise ValueError(
            "Choose full_rows or assignments_only, not both."
        )

    if full_rows:
        folder = "changed_rows"

    elif assignments_only:
        folder = "assignment_changes"

    else:
        folder = "changes"

    paths = sorted(
        (
            Path(history_dir)
            / folder
        ).glob(
            "*/*.parquet"
        )
    )

    frames = []

    target = str(
        meter_number
    ).strip()

    for path in paths:
        frame = pd.read_parquet(
            path
        )

        matched = frame.loc[
            frame[KEY]
            .astype(str)
            .eq(target)
        ]

        if not matched.empty:
            frames.append(
                matched
            )

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
    )