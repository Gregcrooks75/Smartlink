"""Step 18: score new meters, preserve history, and export review recommendations.

Place in:
    scripts/meter_to_transformer/stages/run_recommendations.py

Requires:
    Existing SmartLink training code, pandas, numpy, pyarrow, xgboost,
    xlsxwriter, and openpyxl if importing an edited Excel Tracker sheet.

Pipeline additions
------------------
1. Immediately AFTER loading AMI, BEFORE standardization or rejection:

    full_ami_df = ami_df.copy()

2. Immediately AFTER standardize_all_inputs(), BEFORE coordinate rejection:

    full_transformer_df = transformer_df.copy()

3. After step 17, call run_recommendations_stage() with today's
   final_feature_outputs:

    from meter_to_transformer.stages.run_recommendations import (
        run_recommendations_stage,
    )

    recommendation_directory = run_recommendations_stage(
        project_root=PROJECT_ROOT,
        model_date="2026-10-20",
        min_score_gap=0.004216045,
        ami_df=ami_df,
        full_ami_df=full_ami_df,
        transformer_df=full_transformer_df,
        final_feature_files=final_feature_outputs,
    )

Refresh AMI and transformers in step 1 and run today's input/feature stages.
The GitHub pipeline copy comments out the feature-building block; enable it.

An older MODEL is supported; old prediction FEATURES are refused.

Model folders:
    data/processed/models/transformer_ranker/YYYY-MM-DD/

Daily outputs:
    data/processed/recommendations/YYYY-MM-DD/

    predictions/Division.parquet
        Permanent, unfiltered score checkpoints.

    recommendations/Division.parquet
        Recommendations with flags and status.

    all_recommendations.parquet
        All divisions, including review cases.

    final_recommendations.parquet
        Eligible, unsent changes only, <= 550 ft.

    recommendations.xlsx
        Summary, Recommendations, Review, and Tracker.

    empty_transformers.xlsx
        Departures held to prevent empty transformers.

    division_summary.parquet
        Counts and empirical expected error counts.

    review_map.html / review_updates.csv
        Manual review. The map uses internet tiles.

Status workflow:
    Edit the Excel Tracker sheet or review_updates.csv, or export CSV from
    the map. Then call:

        import_review_updates(PROJECT_ROOT, edited_file_path)

    Use:
        PENDING
        APPROVED
        REJECTED
        SENT
        INTEGRATED

    SENT/INTEGRATED requires batch_id and sent_date and appends the meter to
    the EXISTING All_Batches_Sent.csv source.

Only mark SENT after actually submitting a batch. No updates or messages are
sent.

Import edits BEFORE rerunning the stage. Workbooks and the review map are
regenerated exports.

The tracker and status-change history are stored in:

    recommendations/history/

ANY meter ever scored stays excluded on later days, even if unchanged, held,
below threshold, or using another model. Same-day reruns reuse checkpoints.

Daily input snapshots and the model are frozen once scoring starts. Changed
inputs on a restart are refused instead of silently reusing stale predictions.

Prior train/validation/test prediction files are also excluded from scoring.

Customer counts represent unique meters in the FULL current AMI snapshot.
Empty-transformer checks simulate this run's eligible moves together,
including incoming customers, and repeat after removing unsafe moves.

Historical sent meters whose unique target does not yet match AMI are reserved
as pending departures. Their arrivals are not credited. Conflicting or unknown
historical targets are reserved conservatively too.

Precision estimates are empirical validation rates at the selected score gap:
division sample >= 30, otherwise global. They are not probabilities or
guarantees for individual records.

Review flags cannot be overridden by APPROVED.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from meter_to_transformer.data_cleaning.standardize_inputs import (
    DIVISION_MAPPING,
)
from meter_to_transformer.training.run_model_training import (
    predict_transformers,
)


STATUSES = {
    "PENDING",
    "APPROVED",
    "REJECTED",
    "SENT",
    "INTEGRATED",
}

ID = "recommendation_id"

PAIR = [
    "meter_number",
    "candidate_transformer",
]

EMPTY = "WOULD_EMPTY_TRANSFORMER"


def _id(values):
    """Normalize meter, transformer, and related identifier columns."""
    return (
        values.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"^(\d+)\.0+$", r"\1", regex=True)
        .replace(
            [
                "",
                "NONE",
                "NULL",
                "NAN",
                "N/A",
                "<NA>",
            ],
            pd.NA,
        )
    )


def _columns(frame, names):
    """Accept the raw and standardized names already used by SmartLink."""
    frame = frame.copy()

    for target, aliases in names.items():
        if target not in frame:
            found = next(
                (name for name in aliases if name in frame),
                None,
            )

            if found is None:
                raise ValueError(
                    f"Missing {target}; expected one of {aliases}"
                )

            frame = frame.rename(columns={found: target})

    return frame


def _atomic(frame, path):
    """Write a DataFrame to a temporary file before replacing the target."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_name(
        path.stem + ".tmp" + path.suffix
    )

    if path.suffix.lower() == ".csv":
        frame.to_csv(
            temporary,
            index=False,
            encoding="utf-8-sig",
        )
    else:
        frame.to_parquet(
            temporary,
            index=False,
            compression="zstd",
        )

    temporary.replace(path)


@contextmanager
def _lock(folder):
    """Prevent multiple recommendation runs from writing simultaneously."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / "_running.lock"

    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(
            f"Another run holds {path}. "
            "After a crashed process exits, delete this lock."
        ) from exc

    try:
        with handle:
            handle.write(datetime.now().isoformat())

        yield

    finally:
        path.unlink(missing_ok=True)


def _hash_file(path):
    """Calculate the SHA-256 hash for a file."""
    with Path(path).open("rb") as source:
        return hashlib.file_digest(
            source,
            "sha256",
        ).hexdigest()


def _distance(lat1, lon1, lat2, lon2):
    """Calculate haversine distance in feet between coordinate arrays."""
    lat1, lon1, lat2, lon2 = [
        np.radians(
            pd.to_numeric(x, errors="coerce").to_numpy(
                dtype=float,
                na_value=np.nan,
            )
        )
        for x in (
            lat1,
            lon1,
            lat2,
            lon2,
        )
    ]

    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin((lon2 - lon1) / 2) ** 2
    )

    return (
        2
        * 20_902_231
        * np.arcsin(
            np.sqrt(
                np.clip(a, 0, 1)
            )
        )
    )


def _inputs(ami_df, transformer_df):
    """Standardize the AMI and transformer inputs used for recommendations."""
    ami = _columns(
        ami_df,
        {
            "meter_number": (
                "MeterID",
            ),
            "current_transformer": (
                "current_meter_floc",
                "SuperiorFLOC",
            ),
            "division": (
                "Division",
            ),
            "latitude": (
                "LATITUDE",
            ),
            "longitude": (
                "LONGITUDE",
            ),
            "opco": (
                "OPCO",
            ),
            "connection_object": (
                "ConnectionObject",
            ),
            "premise_id": (
                "PremiseID",
            ),
        },
    )

    tx = _columns(
        transformer_df,
        {
            "transformer_floc": (
                "FuncLoc",
            ),
            "division": (
                "Division",
                "DivName",
            ),
            "latitude": (
                "Location_Y",
            ),
            "longitude": (
                "Location_X",
            ),
            "opco": (
                "OPCO",
            ),
        },
    )

    identifier_columns = (
        (
            ami,
            [
                "meter_number",
                "current_transformer",
                "connection_object",
                "premise_id",
                "opco",
            ],
        ),
        (
            tx,
            [
                "transformer_floc",
                "opco",
            ],
        ),
    )

    for frame, columns in identifier_columns:
        for column in columns:
            frame[column] = _id(frame[column])

        frame["division"] = (
            frame["division"]
            .astype("string")
            .str.strip()
            .str.title()
            .replace(DIVISION_MAPPING)
        )

        for column in (
            "latitude",
            "longitude",
        ):
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    meter_columns = [
        "meter_number",
        "current_transformer",
        "division",
        "latitude",
        "longitude",
        "opco",
        "connection_object",
        "premise_id",
    ]

    for column in (
        "Address",
        "City",
        "ZipCode",
        "Circuit",
        "sap_id",
        "gis_id",
    ):
        if column in ami:
            meter_columns.append(column)

    ami = ami[meter_columns].drop_duplicates()

    if (
        ami["meter_number"].isna().any()
        or ami["meter_number"].duplicated().any()
    ):
        raise ValueError(
            "Full AMI must contain one consistent row per meter. "
            "Resolve duplicate or conflicting records."
        )

    transformer_columns = [
        "transformer_floc",
        "division",
        "latitude",
        "longitude",
        "opco",
    ]

    # Several bank units may share one FLOC, but the location, division,
    # and OPCO associated with the FLOC must agree.
    # Division and OPCO must agree across units sharing a FLOC.
    conflicts = (
        tx.groupby("transformer_floc")[["division", "opco"]]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )

    if tx["transformer_floc"].isna().any() or conflicts.any():
        raise ValueError(
            "Transformer FLOCs are missing or have conflicting divisions/OPCOs."
        )

    # Match candidate generation: keep the first location per FLOC.
    unique_tx = tx[transformer_columns].drop_duplicates(
        subset=["transformer_floc"],
        keep="first",
    )

    extra_columns = [
        column
        for column in (
            "transformer_kva",
            "Transformer_kVA",
            "sap_id",
            "gis_id",
            "SAPID",
            "GISID",
        )
        if column in tx
    ]

    if extra_columns:
        unique_tx = unique_tx.merge(
            tx[
                ["transformer_floc"] + extra_columns
            ].drop_duplicates("transformer_floc"),
            on="transformer_floc",
            validate="one_to_one",
        )

    transformer_unit_counts = (
        tx.groupby("transformer_floc")
        .size()
    )

    unique_tx["unit_count"] = (
        unique_tx["transformer_floc"]
        .map(transformer_unit_counts)
    )

    ami = ami.set_index(
        "meter_number",
        drop=False,
    )

    unique_tx = unique_tx.set_index(
        "transformer_floc",
        drop=False,
    )

    return ami, unique_tx


def _sent(root):
    """Load the existing historical sent-batch exclusion file."""
    path = (
        Path(root)
        / "data"
        / "reference"
        / "Historical Corrections"
        / "All_Batches_Sent.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Required batch-exclusion file: {path}"
        )

    frame = pd.read_csv(
        path,
        dtype="string",
        encoding="utf-8-sig",
    )

    frame.columns = (
        frame.columns
        .str.strip()
        .str.lower()
    )

    for column in (
        "meter_number",
        "current_floc",
        "final_recommended_floc",
    ):
        if column not in frame:
            raise ValueError(
                f"{path.name} is missing {column}"
            )

        frame[column] = _id(frame[column])

    return path, frame


def _tracker(folder):
    """Load the recommendation status tracker or create an empty tracker."""
    path = (
        Path(folder)
        / "history"
        / "recommendation_status.parquet"
    )

    columns = [
        ID,
        "meter_number",
        "current_transformer",
        "recommended_transformer",
        "prediction_date",
        "model_date",
        "status",
        "notes",
        "batch_id",
        "sent_date",
        "updated_at",
    ]

    if path.exists():
        return pd.read_parquet(path)

    return pd.DataFrame(columns=columns)


def _pending_departures(
    batches,
    tracker,
    population,
):
    """Count sent departures that have not yet appeared in current AMI."""
    records = batches[
        [
            "meter_number",
            "final_recommended_floc",
        ]
    ].rename(
        columns={
            "final_recommended_floc": "recommended_transformer",
        }
    )

    extra = tracker.loc[
        tracker["status"].isin(
            [
                "SENT",
                "INTEGRATED",
            ]
        ),
        [
            "meter_number",
            "recommended_transformer",
        ],
    ]

    if len(extra):
        records = pd.concat(
            [
                records,
                extra,
            ],
            ignore_index=True,
        )

    targets = (
        records.groupby("meter_number")["recommended_transformer"]
        .agg(
            [
                "first",
                "nunique",
            ]
        )
    )

    current = population["current_transformer"].reindex(
        targets.index
    )

    confirmed = targets[
        targets["nunique"].eq(1)
        & targets["first"].eq(current)
    ]

    pending = (
        set(records["meter_number"].dropna())
        - set(confirmed.index)
    )

    return (
        population.loc[
            population.index.isin(pending)
        ]
        .groupby("current_transformer")
        .size()
    )


def _save_tracker(folder, before, after):
    """Save the current tracker and append changed rows to its history."""
    if after[ID].duplicated().any():
        raise ValueError(
            "Duplicate recommendation IDs in tracker"
        )

    old = (
        before.set_index(ID)
        .reindex(after[ID])
        .reset_index()
    )

    changed = pd.Series(
        False,
        index=after.index,
    )

    for column in (
        "status",
        "notes",
        "batch_id",
        "sent_date",
    ):
        changed |= (
            after[column]
            .fillna("")
            .reset_index(drop=True)
            .ne(
                old[column]
                .fillna("")
                .reset_index(drop=True)
            )
        )

    changes = after.loc[changed].copy()

    if len(changes):
        previous_status = (
            before.set_index(ID)["status"]
            if not before.empty
            else pd.Series(dtype="string")
        )

        changes["previous_status"] = (
            changes[ID]
            .map(previous_status)
        )

        stamp = datetime.now().strftime(
            "%Y-%m-%d_%H%M%S_%f"
        )

        _atomic(
            changes,
            (
                Path(folder)
                / "history"
                / "status_changes"
                / f"{stamp}.parquet"
            ),
        )

    _atomic(
        after,
        (
            Path(folder)
            / "history"
            / "recommendation_status.parquet"
        ),
    )


def import_review_updates(
    project_root,
    edited_file_path,
):
    """Import an edited Excel Tracker sheet or review CSV.

    This function updates the status history. It does not submit or send
    a batch.
    """
    root = Path(project_root)

    folder = (
        root
        / "data"
        / "processed"
        / "recommendations"
    )

    with _lock(folder):
        before = _tracker(folder)

        edited_file_path = Path(edited_file_path)

        if edited_file_path.suffix.lower() == ".xlsx":
            with pd.ExcelFile(edited_file_path) as workbook:
                sheets = [
                    name
                    for name in workbook.sheet_names
                    if (
                        name == "Tracker"
                        or name.startswith("Tracker_")
                    )
                ]

                if not sheets:
                    raise ValueError(
                        "Edited workbook needs its Tracker sheet(s)"
                    )

                updates = pd.concat(
                    [
                        pd.read_excel(
                            workbook,
                            sheet_name=name,
                            dtype="string",
                            keep_default_na=False,
                        )
                        for name in sheets
                    ],
                    ignore_index=True,
                )

        else:
            updates = pd.read_csv(
                edited_file_path,
                dtype="string",
                keep_default_na=False,
                encoding="utf-8-sig",
            )

        required = {
            ID,
            "status",
            "notes",
            "batch_id",
            "sent_date",
        }

        missing = required - set(updates.columns)

        if missing:
            raise ValueError(
                f"Review file needs {sorted(required)}. "
                f"Missing columns: {sorted(missing)}"
            )

        if (
            updates[ID].duplicated().any()
            or not updates[ID].isin(before[ID]).all()
        ):
            raise ValueError(
                "Review file contains duplicate or unknown "
                "recommendation IDs"
            )

        updates["status"] = (
            updates["status"]
            .str.strip()
            .str.upper()
        )

        if not updates["status"].isin(STATUSES).all():
            raise ValueError(
                f"Allowed statuses: {sorted(STATUSES)}"
            )

        submitted = updates["status"].isin(
            [
                "SENT",
                "INTEGRATED",
            ]
        )

        missing_submission_fields = (
            updates["batch_id"].str.strip().eq("")
            | updates["sent_date"].str.strip().eq("")
        )

        if (
            submitted
            & missing_submission_fields
        ).any():
            raise ValueError(
                "SENT/INTEGRATED requires batch_id and "
                "sent_date in YYYY-MM-DD format"
            )

        updates.loc[submitted, "sent_date"] = (
            pd.to_datetime(
                updates.loc[submitted, "sent_date"],
                errors="raise",
            )
            .dt.strftime("%Y-%m-%d")
        )

        future_sent_dates = (
            updates.loc[submitted, "sent_date"]
            .gt(date.today().isoformat())
        )

        if future_sent_dates.any():
            raise ValueError(
                "sent_date cannot be in the future"
            )

        old_status = updates[ID].map(
            before.set_index(ID)["status"]
        )

        returning_to_unsent = (
            old_status.isin(
                [
                    "SENT",
                    "INTEGRATED",
                ]
            )
            & ~submitted
        )

        if returning_to_unsent.any():
            raise ValueError(
                "A sent recommendation cannot be changed "
                "back to an unsent status"
            )

        after = before.set_index(ID)
        updates = updates.set_index(ID)

        for column in (
            "status",
            "notes",
            "batch_id",
            "sent_date",
        ):
            after.loc[
                updates.index,
                column,
            ] = updates[column]

        after.loc[
            updates.index,
            "updated_at",
        ] = datetime.now().isoformat(
            timespec="seconds"
        )

        after = after.reset_index()

        batch_path, batches = _sent(root)

        additions = after.loc[
            after["status"].isin(
                [
                    "SENT",
                    "INTEGRATED",
                ]
            )
        ].rename(
            columns={
                "current_transformer": "current_floc",
                "recommended_transformer": (
                    "final_recommended_floc"
                ),
            }
        )

        batch_columns = [
            "meter_number",
            "current_floc",
            "final_recommended_floc",
            "batch_id",
            "sent_date",
        ]

        # Record the status in the existing historical source too.
        # Rerunning the same import is safe because duplicate records
        # are removed below.
        batches = (
            pd.concat(
                [
                    batches,
                    additions[batch_columns],
                ],
                ignore_index=True,
            )
            .drop_duplicates(
                [
                    "meter_number",
                    "current_floc",
                    "final_recommended_floc",
                ],
                keep="last",
            )
        )

        _atomic(
            batches,
            batch_path,
        )

        _save_tracker(
            folder,
            before,
            after,
        )

    return (
        folder
        / "history"
        / "recommendation_status.parquet"
    )


def _fresh_features(
    frame,
    division,
    ami,
    tx,
    metadata,
):
    """Validate that final features match the current daily inputs."""
    frame = frame.copy()

    required_identifier_columns = (
        PAIR
        + [
            "current_meter_floc",
            "tx_group_id",
        ]
    )

    for column in required_identifier_columns:
        if column not in frame:
            raise ValueError(
                f"{division}: missing {column} in final features"
            )

        frame[column] = _id(
            frame[column]
        )

    if (
        frame[PAIR].isna().any().any()
        or frame.duplicated(PAIR).any()
    ):
        raise ValueError(
            f"{division}: null or duplicate candidate pairs"
        )

    meters_exist = (
        frame["meter_number"]
        .isin(ami.index)
        .all()
    )

    transformers_exist = (
        frame["candidate_transformer"]
        .isin(tx.index)
        .all()
    )

    if not meters_exist or not transformers_exist:
        raise ValueError(
            f"{division}: feature meters or transformers no longer "
            "exist. Rebuild today's features."
        )

    meter = frame["meter_number"]
    candidate = frame["candidate_transformer"]

    feature_divisions = meter.map(
        ami["division"]
    )

    if not (
        feature_divisions
        .eq(division)
        .fillna(False)
        .all()
    ):
        raise ValueError(
            f"{division}: feature meters have moved divisions. "
            "Rebuild features."
        )

    feature_current_transformer = (
        frame["current_meter_floc"]
        .fillna("")
    )

    current_transformer = (
        meter.map(
            ami["current_transformer"]
        )
        .fillna("")
    )

    if not feature_current_transformer.eq(
        current_transformer
    ).all():
        raise ValueError(
            f"{division}: AMI assignments changed. "
            "Rebuild candidate and final features."
        )

    coordinate_sources = (
        (
            "meter",
            ami,
            meter,
        ),
        (
            "transformer",
            tx,
            candidate,
        ),
    )

    for source, lookup, key in coordinate_sources:
        for coordinate in (
            "latitude",
            "longitude",
        ):
            column = f"{source}_{coordinate}"

            if column not in frame:
                raise ValueError(
                    f"{division}: missing {column}. "
                    "Rebuild today's features."
                )

            feature_coordinates = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

            current_coordinates = key.map(
                lookup[coordinate]
            )

            valid_coordinates = np.isclose(
                feature_coordinates,
                current_coordinates,
                rtol=0,
                atol=1e-7,
                equal_nan=False,
            )

            if not valid_coordinates.all():
                raise ValueError(
                    f"{division}: stale or invalid {column}. "
                    "Rebuild today's features."
                )

    customer_counts = (
        ami.groupby("current_transformer")
        .size()
    )

    if "customer_count" in frame:
        feature_customer_counts = pd.to_numeric(
            frame["customer_count"],
            errors="coerce",
        )

        current_customer_counts = (
            candidate.map(customer_counts)
            .fillna(0)
        )

        if not np.allclose(
            feature_customer_counts,
            current_customer_counts,
            rtol=0,
            atol=0,
        ):
            raise ValueError(
                f"{division}: customer counts changed. "
                "Rebuild candidate features."
            )

    frame["division"] = division

    frame["current_transformer"] = meter.map(
        ami["current_transformer"]
    )

    frame["subgroup_id"] = frame["tx_group_id"]

    meters = frame[
        [
            "meter_number",
            "subgroup_id",
            "current_transformer",
        ]
    ].drop_duplicates()

    if meters["meter_number"].duplicated().any():
        raise ValueError(
            f"{division}: conflicting subgroup assignments"
        )

    subgroup_quality = (
        meters.groupby("subgroup_id")[
            "current_transformer"
        ]
        .nunique()
    )

    frame["tx_current_transformer_count"] = (
        frame["subgroup_id"]
        .map(subgroup_quality)
        .where(
            frame["subgroup_id"].notna(),
            1,
        )
    )

    frame["candidate_matches_current"] = (
        frame["candidate_transformer"]
        .eq(frame["current_transformer"])
        .astype("float32")
    )

    candidate_parent = (
        frame["candidate_transformer"]
        .str.replace(
            r"-[^-]+$",
            "",
            regex=True,
        )
    )

    current_parent = (
        frame["current_transformer"]
        .str.replace(
            r"-[^-]+$",
            "",
            regex=True,
        )
    )

    frame["candidate_same_current_parent"] = (
        candidate_parent
        .eq(current_parent)
        .fillna(False)
        .astype("float32")
    )

    model_features = set(
        metadata["feature_schema"]["source_features"]
    )

    missing_features = (
        model_features
        - set(frame.columns)
    )

    if missing_features:
        raise ValueError(
            f"{division}: model features missing: "
            f"{sorted(missing_features)}"
        )

    return frame


def _enrich(
    predictions,
    ami,
    tx,
    model_date,
    day,
):
    """Add current meter and transformer information to model predictions."""
    out = predictions.copy()

    out["prediction_date"] = day
    out["model_date"] = model_date

    meter_information = (
        ami.reset_index(drop=True)
        .rename(
            columns={
                column: f"meter_{column}"
                for column in (
                    "division",
                    "opco",
                    "latitude",
                    "longitude",
                )
            }
        )
    )

    out = (
        out.drop(
            columns=[
                "division",
                "current_transformer",
            ],
            errors="ignore",
        )
        .merge(
            meter_information,
            on="meter_number",
            how="left",
            validate="one_to_one",
        )
    )

    out["division"] = out["meter_division"]

    transformer_lookups = (
        (
            "current",
            "current_transformer",
        ),
        (
            "recommended",
            "recommended_transformer",
        ),
    )

    for prefix, key in transformer_lookups:
        lookup = (
            tx.drop(
                columns="transformer_floc"
            )
            .add_prefix(f"{prefix}_")
        )

        out = out.merge(
            lookup,
            left_on=key,
            right_index=True,
            how="left",
            validate="many_to_one",
        )

    out["current_distance_ft"] = _distance(
        out["meter_latitude"],
        out["meter_longitude"],
        out["current_latitude"],
        out["current_longitude"],
    )

    out["recommended_distance_ft"] = _distance(
        out["meter_latitude"],
        out["meter_longitude"],
        out["recommended_latitude"],
        out["recommended_longitude"],
    )

    out["division_mismatch"] = (
        out["meter_division"]
        .ne(out["recommended_division"])
        .astype("boolean")
    )

    out["opco_mismatch"] = (
        out["meter_opco"]
        .ne(out["recommended_opco"])
        .astype("boolean")
    )

    out["current_division_mismatch"] = (
        out["meter_division"]
        .ne(out["current_division"])
        .astype("boolean")
    )

    out["current_opco_mismatch"] = (
        out["meter_opco"]
        .ne(out["current_opco"])
        .astype("boolean")
    )

    grouped = (
        out.loc[out["subgroup_id"].notna()]
        .groupby(
            [
                "division",
                "subgroup_id",
            ]
        )
    )

    statistics = grouped.agg(
        group_scored_meters=(
            "meter_number",
            "size",
        ),
        group_recommended_transformers=(
            "recommended_transformer",
            "nunique",
        ),
    )

    out = out.merge(
        statistics,
        on=[
            "division",
            "subgroup_id",
        ],
        how="left",
        validate="many_to_one",
    )

    out["group_predictions_partial"] = (
        out["group_scored_meters"]
        .lt(out["tx_group_size"])
    )

    payload = (
        out[
            [
                "meter_number",
                "current_transformer",
                "recommended_transformer",
            ]
        ]
        .fillna("")
        .astype(str)
    )

    recommendation_ids = [
        hashlib.sha256(
            (
                f"{day}|{model_date}|"
                f"{meter_number}|"
                f"{current_transformer}|"
                f"{recommended_transformer}"
            ).encode()
        ).hexdigest()[:24]
        for (
            meter_number,
            current_transformer,
            recommended_transformer,
        ) in payload.itertuples(
            index=False,
            name=None,
        )
    ]

    out[ID] = pd.Series(
        recommendation_ids,
        index=out.index,
        dtype="string",
    )

    return out


def _links(out):
    """Add map URLs for proposed meter-to-transformer changes."""
    out = out.copy()

    map_sources = (
        (
            "meter",
            "meter_latitude",
            "meter_longitude",
        ),
        (
            "current_transformer",
            "current_latitude",
            "current_longitude",
        ),
        (
            "recommended_transformer",
            "recommended_latitude",
            "recommended_longitude",
        ),
    )

    for prefix, latitude, longitude in map_sources:
        coordinates = (
            out[latitude].round(7).astype("string")
            + ","
            + out[longitude].round(7).astype("string")
        )

        out[f"{prefix}_map"] = (
            "https://www.google.com/maps/search/"
            "?api=1&query="
            + coordinates
        )

    out["street_view"] = (
        "https://www.google.com/maps/@?"
        "api=1&map_action=pano&viewpoint="
        + out["meter_latitude"].round(7).astype("string")
        + ","
        + out["meter_longitude"].round(7).astype("string")
    )

    out["satellite_map"] = (
        "https://www.google.com/maps/@?"
        "api=1&map_action=map&zoom=20"
        "&basemap=satellite&center="
        + out["meter_latitude"].round(7).astype("string")
        + ","
        + out["meter_longitude"].round(7).astype("string")
    )

    return out


def _protect_empty(
    out,
    ami,
    pending_departures=None,
):
    """Remove departures that would leave a transformer empty.

    The empty-transformer check is repeated after unsafe departures are
    removed because losing an incoming meter can expose another empty
    transformer.
    """
    out = out.copy()

    customers_before = (
        ami.groupby("current_transformer")
        .size()
    )

    if pending_departures is None:
        reserved_departures = pd.Series(
            dtype="int64"
        )
    else:
        reserved_departures = pending_departures

    available_customers = customers_before.subtract(
        reserved_departures,
        fill_value=0,
    )

    out["current_customers_before"] = (
        out["current_transformer"]
        .map(customers_before)
        .fillna(0)
        .astype(int)
    )

    out["current_pending_sent_departures"] = (
        out["current_transformer"]
        .map(reserved_departures)
        .fillna(0)
        .astype(int)
    )

    out["recommended_customers_before"] = (
        out["recommended_transformer"]
        .map(customers_before)
        .fillna(0)
        .astype(int)
    )

    out["would_empty_current_transformer"] = False
    out["customers_after_if_applied"] = np.nan

    out["empty_check_iteration"] = pd.Series(
        pd.NA,
        index=out.index,
        dtype="Int64",
    )

    iteration = 0

    while True:
        ready = out.loc[
            out["review_flags"].eq("")
            & out["status"].isin(
                [
                    "PENDING",
                    "APPROVED",
                ]
            )
        ]

        outgoing = (
            ready.groupby("current_transformer")
            .size()
        )

        incoming = (
            ready.groupby("recommended_transformer")
            .size()
        )

        customers_after = (
            available_customers
            .subtract(
                outgoing,
                fill_value=0,
            )
            .add(
                incoming,
                fill_value=0,
            )
        )

        emptied_transformers = customers_before.index[
            customers_before.gt(0)
            & customers_after.reindex(
                customers_before.index
            ).le(0)
        ]

        affected = out.index[
            out.index.isin(ready.index)
            & out["current_transformer"].isin(
                emptied_transformers
            )
        ]

        if not len(affected):
            break

        iteration += 1

        out.loc[
            affected,
            "review_flags",
        ] = EMPTY

        out.loc[
            affected,
            "would_empty_current_transformer",
        ] = True

        out.loc[
            affected,
            "empty_check_iteration",
        ] = iteration

        out.loc[
            affected,
            "customers_after_if_applied",
        ] = (
            out.loc[
                affected,
                "current_transformer",
            ]
            .map(customers_after)
        )

    out["current_customers_after_final"] = (
        out["current_transformer"]
        .map(customers_after)
        .fillna(0)
        .astype(int)
    )

    out["recommended_customers_after_final"] = (
        out["recommended_transformer"]
        .map(customers_after)
        .fillna(0)
        .astype(int)
    )

    out["ready_for_export"] = (
        out["review_flags"].eq("")
        & out["status"].isin(
            [
                "PENDING",
                "APPROVED",
            ]
        )
    )

    assert not (
        out.loc[
            out["ready_for_export"],
            "current_customers_after_final",
        ]
        <= 0
    ).any()

    return out


def _estimates(
    out,
    validation,
    gap,
    min_division_sample,
):
    """Add empirical precision estimates from validation predictions."""
    required = {
        "proposed_change",
        "correct",
        "score_gap",
        "division",
        "recommended_distance_ft",
    }

    missing = required - set(validation.columns)

    if missing:
        raise ValueError(
            "Validation predictions missing "
            f"{sorted(missing)}"
        )

    selected = validation.loc[
        validation["proposed_change"].eq(True)
        & validation["score_gap"].ge(gap)
        & validation["recommended_distance_ft"].between(
            0,
            550,
        )
    ].copy()

    if not selected["correct"].isin(
        [
            True,
            False,
            0,
            1,
        ]
    ).all():
        raise ValueError(
            "Validation correct must contain Boolean outcomes"
        )

    by_division = (
        selected.groupby("division")["correct"]
        .agg(
            [
                "mean",
                "size",
            ]
        )
    )

    if len(selected):
        global_precision = float(
            selected["correct"].mean()
        )
    else:
        global_precision = np.nan

    out["validation_sample_size"] = (
        out["division"]
        .map(by_division["size"])
        .fillna(0)
        .astype(int)
    )

    enough_samples = (
        out["validation_sample_size"]
        .ge(min_division_sample)
    )

    out["estimated_precision"] = (
        out["division"]
        .map(by_division["mean"])
        .where(
            enough_samples,
            global_precision,
        )
    )

    out["estimate_basis"] = np.where(
        enough_samples,
        "DIVISION_VALIDATION",
        "GLOBAL_VALIDATION",
    )

    out.loc[
        out["estimated_precision"].isna(),
        "estimate_basis",
    ] = "NO_VALIDATION_SAMPLE"

    out["estimate_sample_size"] = (
        out["validation_sample_size"]
        .where(
            enough_samples,
            len(selected),
        )
    )

    return out


def _excel(path, sheets):
    """Write sortable Excel review tables.

    Long tables are split into multiple sheets to avoid Excel hyperlink and
    worksheet-size issues.
    """
    path = Path(path)

    temporary = path.with_name(
        path.stem + ".tmp.xlsx"
    )

    writer_options = {
        "strings_to_formulas": False,
        "strings_to_urls": False,
    }

    with pd.ExcelWriter(
        temporary,
        engine="xlsxwriter",
        engine_kwargs={
            "options": writer_options,
        },
    ) as writer:
        for title, frame in sheets.items():
            row_starts = range(
                0,
                max(1, len(frame)),
                10_000,
            )

            for part, start in enumerate(
                row_starts,
                start=1,
            ):
                if len(frame) <= 10_000:
                    sheet_name = title
                else:
                    sheet_name = f"{title}_{part}"

                block = frame.iloc[
                    start : start + 10_000
                ]

                block.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                )

                worksheet = writer.sheets[
                    sheet_name
                ]

                worksheet.freeze_panes(
                    1,
                    2,
                )

                worksheet.autofilter(
                    0,
                    0,
                    len(block),
                    max(
                        0,
                        len(block.columns) - 1,
                    ),
                )

                header_format = writer.book.add_format(
                    {
                        "bold": True,
                        "bg_color": "#16354D",
                        "font_color": "white",
                        "text_wrap": True,
                    }
                )

                for column_index, label in enumerate(
                    block.columns
                ):
                    worksheet.write(
                        0,
                        column_index,
                        label,
                        header_format,
                    )

                    width = min(
                        36,
                        max(
                            16,
                            len(label) + 2,
                        ),
                    )

                    worksheet.set_column(
                        column_index,
                        column_index,
                        width,
                    )

                    if (
                        label.endswith("_map")
                        or label == "street_view"
                    ):
                        for row_index, url in enumerate(
                            block[label],
                            start=1,
                        ):
                            if pd.notna(url):
                                worksheet.write_url(
                                    row_index,
                                    column_index,
                                    str(url),
                                    string="Open map",
                                )

                    elif "precision" in label:
                        percentage_format = (
                            writer.book.add_format(
                                {
                                    "num_format": "0.0%",
                                }
                            )
                        )

                        worksheet.set_column(
                            column_index,
                            column_index,
                            20,
                            percentage_format,
                        )

                    elif "distance_ft" in label:
                        distance_format = (
                            writer.book.add_format(
                                {
                                    "num_format": "0.0",
                                }
                            )
                        )

                        worksheet.set_column(
                            column_index,
                            column_index,
                            20,
                            distance_format,
                        )

                worksheet.set_row(
                    0,
                    44,
                )

    temporary.replace(path)


def _map(path, frame):
    """Generate the interactive HTML review map."""
    columns = [
        ID,
        "meter_number",
        "current_transformer",
        "recommended_transformer",
        "division",
        "meter_latitude",
        "meter_longitude",
        "current_latitude",
        "current_longitude",
        "recommended_latitude",
        "recommended_longitude",
        "score_gap",
        "recommended_distance_ft",
        "connection_object",
        "premise_id",
        "review_flags",
        "status",
        "notes",
        "batch_id",
        "sent_date",
    ]

    data = (
        frame[columns]
        .to_json(orient="records")
        .replace("<", "\\u003c")
    )

    Path(path).write_text(
        MAP_HTML.replace(
            "__DATA__",
            data,
        ),
        encoding="utf-8",
    )


def run_recommendations_stage(
    *,
    project_root,
    model_date,
    ami_df,
    transformer_df,
    full_ami_df,
    final_feature_files,
    min_score_gap=0.0,
    min_division_sample=30,
):
    """Score new meters and export recommendation review files.

    Uses today's fresh features. Any meter ever scored is excluded from future
    scoring.

    ``full_ami_df`` must be the full AMI snapshot captured before coordinate
    or division rejection.

    ``ami_df`` is the validated AMI used to build today's candidate features.

    ``transformer_df`` must be the full current transformer reference captured
    before coordinate rejection.

    Repeating today's run resumes its saved checkpoints without rescoring them.
    """
    root = Path(project_root)

    day = date.today().isoformat()

    date.fromisoformat(model_date)

    if (
        not np.isfinite(min_score_gap)
        or min_score_gap < 0
        or min_division_sample < 1
    ):
        raise ValueError(
            "Use a finite nonnegative score gap and "
            "a positive sample minimum"
        )

    model = (
        root
        / "data"
        / "processed"
        / "models"
        / "transformer_ranker"
        / model_date
    )

    metadata_path = model / "metadata.json"

    metadata = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        metadata.get(
            "training_date",
            model_date,
        )
        != model_date
    ):
        raise ValueError(
            "Selected folder and model training date disagree"
        )

    folder = (
        root
        / "data"
        / "processed"
        / "recommendations"
    )

    today = folder / day

    today.mkdir(
        parents=True,
        exist_ok=True,
    )

    with _lock(folder):
        batch_path, batches = _sent(root)
        tracker = _tracker(folder)

        ami, tx = _inputs(
            ami_df,
            transformer_df,
        )

        population, _ = _inputs(
            full_ami_df,
            transformer_df,
        )

        if not ami.index.isin(
            population.index
        ).all():
            raise ValueError(
                "Full AMI population does not include all "
                "validated meters"
            )

        assignments_match = (
            ami["current_transformer"]
            .fillna("")
            .eq(
                population.loc[
                    ami.index,
                    "current_transformer",
                ].fillna("")
            )
        )

        if not assignments_match.all():
            raise ValueError(
                "Validated AMI and full AMI must come from "
                "the same current snapshot"
            )

        files = {
            division: Path(path)
            for division, path
            in final_feature_files.items()
        }

        invalid_feature_files = (
            not files
            or any(
                path.parent.name != day
                or not path.is_file()
                for path in files.values()
            )
        )

        if invalid_feature_files:
            raise ValueError(
                "Pass today's complete final_feature_outputs "
                "from step 12. Older feature folders are refused."
            )

        missing_divisions = (
            set(
                ami["division"]
                .dropna()
                .unique()
            )
            - set(files)
        )

        if missing_divisions:
            raise ValueError(
                "Today's final features are missing divisions: "
                f"{sorted(missing_divisions)}"
            )

        manifest = {
            "model_date": model_date,
            "model_sha256": _hash_file(
                model / "model.ubj"
            ),
            "metadata_sha256": _hash_file(
                metadata_path
            ),
            "ami_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(
                    ami.sort_index(),
                    index=True,
                ).values.tobytes()
            ).hexdigest(),
            "full_ami_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(
                    population.sort_index(),
                    index=True,
                ).values.tobytes()
            ).hexdigest(),
            "tx_sha256": hashlib.sha256(
                pd.util.hash_pandas_object(
                    tx.sort_index(),
                    index=True,
                ).values.tobytes()
            ).hexdigest(),
            "features": {
                division: _hash_file(path)
                for division, path
                in sorted(files.items())
            },
        }

        manifest_path = (
            today
            / "_inputs.json"
        )

        if manifest_path.exists():
            saved_manifest = json.loads(
                manifest_path.read_text(
                    encoding="utf-8"
                )
            )

            if saved_manifest != manifest:
                raise ValueError(
                    "Today's saved inputs or model changed. "
                    "Existing predictions cannot be relabeled or "
                    "rescored. Use the next day's fresh run."
                )

        manifest_path.write_text(
            json.dumps(
                manifest,
                indent=2,
            ),
            encoding="utf-8",
        )

        (
            today
            / "_complete.json"
        ).unlink(missing_ok=True)

        excluded_sent = (
            set(
                batches["meter_number"]
                .dropna()
            )
            | set(
                tracker.loc[
                    tracker["status"].isin(
                        [
                            "SENT",
                            "INTEGRATED",
                        ]
                    ),
                    "meter_number",
                ]
            )
        )

        checkpoints = list(
            folder.glob(
                "????-??-??/predictions/*.parquet"
            )
        )

        predicted = set()

        for checkpoint_path in checkpoints:
            checkpoint_meters = pd.read_parquet(
                checkpoint_path,
                columns=[
                    "meter_number",
                ],
            )

            predicted.update(
                _id(
                    checkpoint_meters[
                        "meter_number"
                    ]
                ).dropna()
            )

        # Model evaluation outputs are predictions too. Never rescore these
        # meters for production recommendations.
        model_root = (
            root
            / "data"
            / "processed"
            / "models"
            / "transformer_ranker"
        )

        for split in (
            "train",
            "validation",
            "test",
        ):
            for prediction_path in model_root.glob(
                f"*/{split}_predictions.parquet"
            ):
                model_prediction_meters = (
                    pd.read_parquet(
                        prediction_path,
                        columns=[
                            "meter_number",
                        ],
                    )
                )

                predicted.update(
                    _id(
                        model_prediction_meters[
                            "meter_number"
                        ]
                    ).dropna()
                )

        parts = []
        audit = []

        for division, feature_path in sorted(
            files.items()
        ):
            # The historical batch file can change while a long scoring run
            # is in progress, so reload it before scoring each division.
            _, batches = _sent(root)

            excluded_sent.update(
                batches["meter_number"]
                .dropna()
            )

            checkpoint = (
                today
                / "predictions"
                / f"{division}.parquet"
            )

            if checkpoint.exists():
                cached = pd.read_parquet(
                    checkpoint
                )

                parts.append(
                    cached
                )

                audit.append(
                    {
                        "division": division,
                        "new_predictions": 0,
                        "reused_predictions": len(
                            cached
                        ),
                    }
                )

                continue

            raw_features = pd.read_parquet(feature_path)

            if "customer_count" in raw_features:
                candidate = _id(
                    raw_features["candidate_transformer"]
                )

                filtered_counts = (
                    ami.groupby("current_transformer").size()
                )

                full_counts = (
                    population.groupby("current_transformer").size()
                )

                debug = pd.DataFrame(
                    {
                        "transformer": candidate,
                        "saved_count": pd.to_numeric(
                            raw_features["customer_count"],
                            errors="coerce",
                        ),
                        "filtered_ami_count": candidate.map(
                            filtered_counts
                        ).fillna(0),
                        "full_ami_count": candidate.map(
                            full_counts
                        ).fillna(0),
                    }
                )

                debug["matches_filtered"] = np.isclose(
                    debug["saved_count"],
                    debug["filtered_ami_count"],
                    rtol=0,
                    atol=0,
                )

                debug["matches_full"] = np.isclose(
                    debug["saved_count"],
                    debug["full_ami_count"],
                    rtol=0,
                    atol=0,
                )

                examples = debug.loc[
                    ~debug["matches_filtered"]
                ].drop_duplicates()

                print(f"\n[{division}] CUSTOMER COUNT DEBUG")
                print(f"Filtered AMI meters: {len(ami):,}")
                print(f"Full AMI meters: {len(population):,}")

                print(
                    "All saved counts match filtered AMI:",
                    bool(debug["matches_filtered"].all()),
                )

                print(
                    "All saved counts match full AMI:",
                    bool(debug["matches_full"].all()),
                )

                print("\nExamples failing the current check:")
                print(
                    examples.head(20).to_string(index=False)
                )

            features = _fresh_features(
                raw_features,
                division,
                ami,
                tx,
                metadata,
            )

            feature_meter_count = (
                features["meter_number"]
                .nunique()
            )

            # Exclude before calling the model. This includes every
            # historical, conflicting, or previously sent row.
            excluded_meters = (
                predicted
                | excluded_sent
            )

            features = features.loc[
                ~features["meter_number"].isin(
                    excluded_meters
                )
            ].copy()

            allowed = (
                features[
                    "tx_current_transformer_count"
                ].le(
                    metadata[
                        "max_subgroup_transformers"
                    ]
                )
                & features.groupby(
                    "meter_number"
                )[
                    "candidate_transformer"
                ]
                .transform("size")
                .ge(2)
            )

            features = features.loc[
                allowed
            ].copy()

            if features.empty:
                audit.append(
                    {
                        "division": division,
                        "feature_meters": (
                            feature_meter_count
                        ),
                        "new_predictions": 0,
                    }
                )

                continue

            predictions = predict_transformers(
                features,
                model,
            )

            group_sizes = (
                features.groupby(
                    "meter_number"
                )["tx_group_size"]
                .first()
            )

            predictions["tx_group_size"] = (
                predictions["meter_number"]
                .map(group_sizes)
            )

            predictions = _enrich(
                predictions,
                ami,
                tx,
                model_date,
                day,
            )

            # A committed per-division prediction checkpoint is also the
            # permanent exclusion ledger.
            _atomic(
                predictions,
                checkpoint,
            )

            draft = predictions.loc[
                predictions[
                    "current_transformer"
                ].notna()
                & predictions[
                    "recommended_transformer"
                ].ne(
                    predictions[
                        "current_transformer"
                    ]
                )
                & predictions[
                    "score_gap"
                ].ge(min_score_gap)
            ].copy()

            draft["ready_for_export"] = False
            draft["review_flags"] = (
                "PENDING_GLOBAL_CHECKS"
            )

            _atomic(
                draft,
                (
                    today
                    / "recommendations"
                    / f"{division}.parquet"
                ),
            )

            predicted.update(
                predictions["meter_number"]
            )

            parts.append(
                predictions
            )

            audit.append(
                {
                    "division": division,
                    "feature_meters": (
                        feature_meter_count
                    ),
                    "new_predictions": len(
                        predictions
                    ),
                }
            )

            print(
                f"[{division}] saved "
                f"{len(predictions):,} new predictions"
            )

        if parts:
            scored = pd.concat(
                parts,
                ignore_index=True,
            )

        else:
            empty_prediction_columns = [
                "meter_number",
                "current_transformer",
                "recommended_transformer",
                "subgroup_id",
                "tx_group_size",
                "score_gap",
                "rank_score",
                "second_score",
                "candidate_count",
            ]

            scored = _enrich(
                pd.DataFrame(
                    columns=empty_prediction_columns
                ),
                ami,
                tx,
                model_date,
                day,
            )

        if scored["meter_number"].duplicated().any():
            raise ValueError(
                "A meter was predicted in more than one division"
            )

        _, batches = _sent(root)

        excluded_sent.update(
            batches["meter_number"]
            .dropna()
        )

        proposed = (
            scored["current_transformer"].notna()
            & scored["recommended_transformer"].ne(
                scored["current_transformer"]
            )
        )

        out = scored.loc[
            proposed
            & scored["score_gap"].ge(
                min_score_gap
            )
            & ~scored["meter_number"].isin(
                excluded_sent
            )
        ].copy()

        out["min_score_gap"] = float(
            min_score_gap
        )

        out["review_flags"] = ""

        def flag(mask, reason):
            """Append a review reason to rows matching the supplied mask."""
            out.loc[
                mask.fillna(True),
                "review_flags",
            ] += f"{reason};"

        flag(
            ~out[
                "recommended_distance_ft"
            ].between(
                0,
                550,
            ),
            "OUTSIDE_550FT",
        )

        flag(
            out["score_gap"].le(0),
            "TOP_SCORE_TIE",
        )

        flag(
            out["division_mismatch"].fillna(
                False
            ),
            "DIVISION_MISMATCH",
        )

        flag(
            out["opco_mismatch"].fillna(
                False
            ),
            "OPCO_MISMATCH",
        )

        flag(
            out[
                [
                    "meter_division",
                    "recommended_division",
                    "meter_opco",
                    "recommended_opco",
                ]
            ].isna().any(axis=1),
            "MISSING_DIVISION_OR_OPCO",
        )

        flag(
            out["connection_object"].isna()
            | out["premise_id"].isna(),
            "MISSING_SAP_IDENTIFIERS",
        )

        coordinate_columns = [
            f"{prefix}_{coordinate}"
            for prefix in (
                "meter",
                "current",
                "recommended",
            )
            for coordinate in (
                "latitude",
                "longitude",
            )
        ]

        invalid_coordinates = pd.Series(
            False,
            index=out.index,
        )

        for column in coordinate_columns:
            if column.endswith("latitude"):
                limit = 90
            else:
                limit = 180

            invalid_coordinates |= (
                ~out[column].between(
                    -limit,
                    limit,
                )
            )

        flag(
            invalid_coordinates,
            "MISSING_OR_INVALID_COORDINATES",
        )

        out["review_flags"] = (
            out["review_flags"]
            .str.rstrip(";")
        )

        validation_predictions = pd.read_parquet(
            model
            / "validation_predictions.parquet"
        )

        out = _estimates(
            out,
            validation_predictions,
            min_score_gap,
            min_division_sample,
        )

        missing_estimate = (
            out["estimated_precision"]
            .isna()
        )

        out.loc[
            missing_estimate,
            "review_flags",
        ] += ";NO_VALIDATION_AT_THRESHOLD"

        out["review_flags"] = (
            out["review_flags"]
            .str.strip(";")
        )

        before = tracker.copy()

        new_recommendations = out.loc[
            ~out[ID].isin(
                tracker[ID]
            ),
            [
                ID,
                "meter_number",
                "current_transformer",
                "recommended_transformer",
                "prediction_date",
                "model_date",
            ],
        ].copy()

        new_recommendations["status"] = "PENDING"
        new_recommendations["notes"] = ""
        new_recommendations["batch_id"] = ""
        new_recommendations["sent_date"] = ""

        new_recommendations["updated_at"] = (
            datetime.now().isoformat(
                timespec="seconds"
            )
        )

        if len(new_recommendations):
            if tracker.empty:
                tracker = (
                    new_recommendations
                    .reset_index(drop=True)
                )
            else:
                tracker = pd.concat(
                    [
                        tracker,
                        new_recommendations,
                    ],
                    ignore_index=True,
                )

        _save_tracker(
            folder,
            before,
            tracker,
        )

        out = out.merge(
            tracker[
                [
                    ID,
                    "status",
                    "notes",
                    "batch_id",
                    "sent_date",
                ]
            ],
            on=ID,
            validate="one_to_one",
        )

        pending_departures = _pending_departures(
            batches,
            tracker,
            population,
        )

        out = _protect_empty(
            out,
            population,
            pending_departures,
        )

        out = _links(
            out
        )

        grouped_sources = (
            (
                "group_proposed_transformers",
                out,
            ),
            (
                "group_ready_transformers",
                out.loc[
                    out["ready_for_export"]
                ],
            ),
        )

        for label, selected in grouped_sources:
            counts = (
                selected.loc[
                    selected["subgroup_id"].notna()
                ]
                .groupby(
                    [
                        "division",
                        "subgroup_id",
                    ]
                )[
                    "recommended_transformer"
                ]
                .nunique()
            )

            group_index = pd.MultiIndex.from_frame(
                out[
                    [
                        "division",
                        "subgroup_id",
                    ]
                ]
            )

            out[label] = (
                group_index
                .map(counts)
                .astype(float)
            )

        lead_columns = [
            ID,
            "meter_number",
            "division",
            "connection_object",
            "premise_id",
            "current_transformer",
            "recommended_transformer",
            "prediction_date",
            "model_date",
            "score_gap",
            "recommended_distance_ft",
            "current_distance_ft",
            "ready_for_export",
            "review_flags",
            "status",
            "notes",
            "batch_id",
            "sent_date",
        ]

        remaining_columns = [
            column
            for column in out.columns
            if column not in lead_columns
        ]

        out = out[
            lead_columns
            + remaining_columns
        ]

        ready = out.loc[
            out["ready_for_export"]
        ].copy()

        held = out.loc[
            ~out["ready_for_export"]
        ].copy()

        summary = (
            out.groupby("division")
            .agg(
                proposed_recommendations=(
                    ID,
                    "size",
                ),
                ready_for_export=(
                    "ready_for_export",
                    "sum",
                ),
                held_for_review=(
                    "ready_for_export",
                    lambda values: (
                        ~values
                    ).sum(),
                ),
                division_mismatches=(
                    "division_mismatch",
                    "sum",
                ),
                opco_mismatches=(
                    "opco_mismatch",
                    "sum",
                ),
                empty_transformer_cases=(
                    "would_empty_current_transformer",
                    "sum",
                ),
                estimated_precision=(
                    "estimated_precision",
                    "first",
                ),
                validation_sample_size=(
                    "validation_sample_size",
                    "first",
                ),
                estimate_basis=(
                    "estimate_basis",
                    "first",
                ),
                estimate_sample_size=(
                    "estimate_sample_size",
                    "first",
                ),
            )
            .reset_index()
        )

        summary["expected_correct_ready"] = (
            summary["ready_for_export"]
            * summary["estimated_precision"]
        )

        summary["expected_incorrect_ready"] = (
            summary["ready_for_export"]
            * (
                1
                - summary["estimated_precision"]
            )
        )

        summary.insert(
            0,
            "run_date",
            day,
        )

        summary.insert(
            1,
            "model_date",
            model_date,
        )

        summary["min_score_gap"] = (
            min_score_gap
        )

        # Include divisions that produced zero recommendations in the
        # daily summary.
        summary = pd.DataFrame(
            {
                "division": sorted(files),
            }
        ).merge(
            summary,
            on="division",
            how="left",
        )

        count_columns = [
            "proposed_recommendations",
            "ready_for_export",
            "held_for_review",
            "division_mismatches",
            "opco_mismatches",
            "empty_transformer_cases",
            "validation_sample_size",
            "estimate_sample_size",
        ]

        summary[count_columns] = (
            summary[count_columns]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
            .fillna(0)
            .astype(int)
        )

        summary["run_date"] = day
        summary["model_date"] = model_date
        summary["min_score_gap"] = min_score_gap

        no_ready_recommendations = (
            summary["ready_for_export"]
            .eq(0)
        )

        summary.loc[
            no_ready_recommendations,
            [
                "expected_correct_ready",
                "expected_incorrect_ready",
            ],
        ] = 0.0

        _atomic(
            out,
            today
            / "all_recommendations.parquet",
        )

        _atomic(
            ready,
            today
            / "final_recommendations.parquet",
        )

        _atomic(
            summary,
            today
            / "division_summary.parquet",
        )

        for division in files:
            division_recommendations = out.loc[
                out["division"].eq(
                    division
                )
            ]

            _atomic(
                division_recommendations,
                (
                    today
                    / "recommendations"
                    / f"{division}.parquet"
                ),
            )

        review_columns = [
            ID,
            "meter_number",
            "status",
            "notes",
            "batch_id",
            "sent_date",
        ]

        review_updates_path = (
            today
            / "review_updates.csv"
        )

        if not review_updates_path.exists():
            _atomic(
                out[review_columns],
                review_updates_path,
            )

        _excel(
            today
            / "recommendations.xlsx",
            {
                "Summary": summary,
                "Recommendations": ready,
                "Review": held,
                "Tracker": out[
                    review_columns
                ],
            },
        )

        empty_transformer_cases = out.loc[
            out[
                "would_empty_current_transformer"
            ]
        ]

        _excel(
            today
            / "empty_transformers.xlsx",
            {
                "Cases": empty_transformer_cases,
            },
        )

        _map(
            today
            / "review_map.html",
            out,
        )

        _atomic(
            pd.DataFrame(audit),
            today
            / "scoring_summary.parquet",
        )

        completion_information = {
            "run_date": day,
            "model_date": model_date,
            "min_score_gap": min_score_gap,
            "recommendations": len(out),
            "ready": len(ready),
            "held": len(held),
        }

        (
            today
            / "_complete.json"
        ).write_text(
            json.dumps(
                completion_information,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            f"{len(ready):,} recommendations ready; "
            f"{len(held):,} held. Saved: {today}"
        )

        return today