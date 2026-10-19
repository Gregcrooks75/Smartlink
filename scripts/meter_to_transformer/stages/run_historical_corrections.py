from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

REFERENCE_FILE = (
    PROJECT_ROOT
    / "data"
    / "reference"
    / "Historical Corrections"
    / "All_Batches_Sent.csv"
)

CACHE_ROOT = (
    PROJECT_ROOT
    / "data"
    / "cache"
    / "historical_corrections"
)

HISTORY_ROOT = CACHE_ROOT / "history"

RUN_DATE = date.today().isoformat()

HISTORICAL_COLUMNS = [
    "meter_number",
    "current_floc",
    "final_recommended_floc",
]


def normalize(series):
    return (
        series.astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


def read_historical():

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(REFERENCE_FILE)

    df = pd.read_csv(
        REFERENCE_FILE,
        dtype="string",
        encoding="utf-8-sig",
    )

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    missing = (
        set(HISTORICAL_COLUMNS)
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Historical columns missing: {sorted(missing)}"
        )

    df = df[HISTORICAL_COLUMNS].copy()

    for col in HISTORICAL_COLUMNS:
        df[col] = normalize(df[col])

    df = df.dropna(
        subset=["meter_number"]
    )

    df = df.drop_duplicates()

    df = df.rename(
        columns={
            "current_floc": "historical_floc"
        }
    )

    df["recommendation_sent"] = True

    return df


def prepare_ami(
    ami_df,
    meter_col="meter_number",
    division_col="Division",
    floc_col="current_meter_floc",
):

    required = [
        meter_col,
        division_col,
        floc_col,
    ]

    missing = set(required) - set(ami_df.columns)

    if missing:
        raise ValueError(
            f"AMI columns missing: {sorted(missing)}"
        )

    ami = ami_df[required].copy()

    ami = ami.rename(
        columns={
            meter_col: "meter_number",
            division_col: "division",
            floc_col: "latest_floc",
        }
    )

    for col in [
        "meter_number",
        "division",
        "latest_floc",
    ]:
        ami[col] = normalize(
            ami[col]
        )

    ami = ami.dropna(
        subset=["meter_number"]
    )

    ami = ami.drop_duplicates()

    conflicts = (
        ami.groupby("meter_number")
        .agg(
            division_count=(
                "division",
                "nunique"
            ),
            floc_count=(
                "latest_floc",
                "nunique"
            ),
        )
    )

    conflicts = conflicts[
        (conflicts["division_count"] > 1)
        | (conflicts["floc_count"] > 1)
    ]

    if not conflicts.empty:
        raise ValueError(
            f"{len(conflicts):,} AMI meters have "
            "conflicting division or FLOC values."
        )

    return ami.drop_duplicates(
        subset=["meter_number"]
    )

def prepare_recommendations(historical):

    counts = (
        historical.groupby("meter_number")
        .agg(
            original_count=(
                "historical_floc",
                "nunique"
            ),
            recommendation_count=(
                "final_recommended_floc",
                "nunique"
            ),
        )
        .reset_index()
    )

    counts["historical_conflict"] = (
        (counts["original_count"] > 1)
        | (counts["recommendation_count"] > 1)
    )

    historical = historical.merge(
        counts[
            [
                "meter_number",
                "historical_conflict"
            ]
        ],
        on="meter_number",
        how="left",
        validate="many_to_one",
    )

    conflicts = historical[
        historical["historical_conflict"]
    ].copy()

    valid = historical[
        ~historical["historical_conflict"]
    ].copy()

    valid = valid.drop_duplicates(
        subset=["meter_number"]
    )

    return valid, conflicts


def reconcile(historical, ami):

    result = historical.merge(
        ami,
        on="meter_number",
        how="left",
        indicator=True,
        validate="one_to_one",
    )

    missing_meter = (
        result["_merge"].eq("left_only")
    )

    same_original = (
        result["latest_floc"]
        .eq(result["historical_floc"])
        .fillna(False)
    )

    same_recommended = (
        result["latest_floc"]
        .eq(result["final_recommended_floc"])
        .fillna(False)
    )

    result["historical_status"] = (
        "DIFFERENT_ASSOCIATION"
    )

    result.loc[
        missing_meter,
        "historical_status",
    ] = "METER_NOT_IN_CURRENT_AMI"

    result.loc[
        ~missing_meter
        & result["latest_floc"].isna(),
        "historical_status",
    ] = "CURRENT_FLOC_MISSING"

    result.loc[
        same_original & ~missing_meter,
        "historical_status",
    ] = "ORIGINAL_ASSOCIATION"

    result.loc[
        same_recommended & ~missing_meter,
        "historical_status",
    ] = "MATCHES_PREVIOUS_RECOMMENDATION"

    result["implementation_status"] = "UNKNOWN"

    result["run_date"] = RUN_DATE

    return result.drop(
        columns=["_merge"]
    )


def get_previous_snapshot():

    if not CACHE_ROOT.exists():
        return None

    previous_dates = sorted(
        path
        for path in CACHE_ROOT.iterdir()
        if (
            path.is_dir()
            and path.name < RUN_DATE
            and (
                path
                / "all_historical_corrections.parquet"
            ).exists()
        )
    )

    if not previous_dates:
        return None

    return pd.read_parquet(
        previous_dates[-1]
        / "all_historical_corrections.parquet"
    )


def write_history(current, previous):

    if previous is None:
        print(
            "No previous snapshot: baseline established."
        )
        return

    compare_cols = [
        "historical_floc",
        "final_recommended_floc",
        "division",
        "latest_floc",
        "historical_status",
    ]

    old = previous.set_index(
        "meter_number"
    )

    new = current.set_index(
        "meter_number"
    )

    all_meters = old.index.union(
        new.index
    )

    old = old.reindex(
        all_meters
    )

    new = new.reindex(
        all_meters
    )

    old_exists = pd.Series(
        all_meters.isin(
            previous["meter_number"]
        ),
        index=all_meters,
    )

    new_exists = pd.Series(
        all_meters.isin(
            current["meter_number"]
        ),
        index=all_meters,
    )

    changes = pd.DataFrame({
        "meter_number": all_meters,
    })

    changes["change_type"] = "UNCHANGED"

    changes.loc[
        ~old_exists.to_numpy(),
        "change_type",
    ] = "ADDED"

    changes.loc[
        ~new_exists.to_numpy(),
        "change_type",
    ] = "REMOVED"

    changed = pd.Series(
        False,
        index=all_meters,
    )

    for col in compare_cols:

        before = old[col].fillna(
            "<NULL>"
        )

        after = new[col].fillna(
            "<NULL>"
        )

        changed |= before.ne(after)

    common = (
        old_exists & new_exists
    )

    changes.loc[
        (changed & common).to_numpy(),
        "change_type",
    ] = "UPDATED"

    changes = changes[
        changes["change_type"] != "UNCHANGED"
    ].copy()

    if changes.empty:
        print(
            "No historical reconciliation changes."
        )
        return

    changes["run_date"] = RUN_DATE

    changed_rows = changes.merge(
        current,
        on="meter_number",
        how="left",
    )

    assignments = pd.DataFrame({
        "meter_number": all_meters,
        "previous_floc": old[
            "latest_floc"
        ].to_numpy(),
        "current_floc": new[
            "latest_floc"
        ].to_numpy(),
    })

    before = old["latest_floc"].fillna(
        "<NULL>"
    )

    after = new["latest_floc"].fillna(
        "<NULL>"
    )

    assignments = assignments[
        (
            before.ne(after)
            & common
        ).to_numpy()
    ].copy()

    assignments["run_date"] = RUN_DATE

    folders = {
        "changes": changes,
        "changed_rows": changed_rows,
        "assignment_changes": assignments,
    }

    for name, frame in folders.items():

        folder = HISTORY_ROOT / name

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        frame.to_parquet(
            folder / f"{RUN_DATE}.parquet",
            index=False,
        )

    print(
        f"History changes: {len(changes):,}"
    )

    print(
        f"Assignment changes: "
        f"{len(assignments):,}"
    )


def save_outputs(result, conflicts):

    output = CACHE_ROOT / RUN_DATE

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_parquet(
        output
        / "all_historical_corrections.parquet",
        index=False,
    )

    conflicts.to_parquet(
        output / "conflicts.parquet",
        index=False,
    )

    for division, frame in (
        result
        .dropna(subset=["division"])
        .groupby("division")
    ):

        frame.to_parquet(
            output / f"{division}.parquet",
            index=False,
        )

        print(
            f"[{division}] {len(frame):,} meters"
        )

    missing = result[
        result["historical_status"]
        == "METER_NOT_IN_CURRENT_AMI"
    ]

    missing.to_parquet(
        output
        / "meters_not_in_current_ami.parquet",
        index=False,
    )

    summary = (
        result
        .groupby("historical_status")
        .agg(
            meters=(
                "meter_number",
                "nunique"
            )
        )
        .reset_index()
    )

    summary.to_csv(
        output / "summary.csv",
        index=False,
    )

    print(
        summary.to_string(index=False)
    )

    print(
        f"Conflicting historical meters: "
        f"{conflicts['meter_number'].nunique():,}"
    )

    print(
        f"Saved to {output}"
    )

def run_historical_corrections(ami_df):

    historical = read_historical()

    valid, conflicts = prepare_recommendations(
        historical
    )

    ami = prepare_ami(
        ami_df=ami_df
    )

    current = reconcile(
        valid,
        ami,
    )

    previous = get_previous_snapshot()

    write_history(
        current,
        previous,
    )

    save_outputs(
        current,
        conflicts,
    )

    return current

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ami-path",
        required=True,
    )

    parser.add_argument(
        "--meter-col",
        default="meter_number",
    )

    parser.add_argument(
        "--division-col",
        default="division",
    )

    parser.add_argument(
        "--floc-col",
        default="current_floc",
    )

    args = parser.parse_args()

    run_historical_corrections(
        ami_path=args.ami_path,
        meter_col=args.meter_col,
        division_col=args.division_col,
        floc_col=args.floc_col,
    )


if __name__ == "__main__":
    main()