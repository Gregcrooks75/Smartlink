from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

HISTORICAL_INPUT_DIR = (
    PROJECT_ROOT
    / "pickles"
    / "Pipeline"
    / "20. Sending Final Recommendations"
)

REFERENCE_DIR = (
    DATA_DIR
    / "reference"
    / "historical_recommendations"
)

AMI_CACHE_DIR = DATA_DIR / "cache" / "ami"

OUTPUT_ROOT = (
    DATA_DIR
    / "cache"
    / "historical_recommendation_matches"
)

RUN_DATE = date.today().isoformat()

OUTPUT_DIR = OUTPUT_ROOT / RUN_DATE

HISTORICAL_FILE = (
    REFERENCE_DIR
    / "previous_model_recommendations.parquet"
)

AMI_FILE = AMI_CACHE_DIR / "ami.parquet"


# ============================================================
# Columns
# ============================================================

METER_COL = "meter_number"
DIVISION_COL = "division"
CURRENT_FLOC_COL = "current_floc"

HISTORICAL_COLUMNS = [
    "meter_number",
    "current_floc",
    "final_recommended_floc",
]


# ============================================================
# Helper Functions
# ============================================================

def normalize_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


# ============================================================
# Load Historical Recommendations
# ============================================================

def load_historical_recommendations() -> pd.DataFrame:

    REFERENCE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    files = sorted(
        HISTORICAL_INPUT_DIR.glob("Batch_*.xlsx")
    )

    if not files:

*       if HISTORICAL_FILE.exists()*
            print("Reusing consol*dated historical recommendations."*

            return pd.read_parqu*t(
                HISTORICAL_FILE*            )

        raise FileN*tFoundError(
            f"No hist*rical Excel files found in {HISTOR*CAL_INPUT_DIR}"
        )

    fra*es = []

    for file in files:

 *      print(f"Reading {file.name}"*

        batch = pd.read_excel(
 *          file,
            dtype=*string"
        )

        batch.c*lumns = (
            batch.column*
            .str.strip()
        *   .str.lower()
        )

       *missing = set(HISTORICAL_COLUMNS) * set(batch.columns)

        if mi*sing:
            raise ValueError*
                f"{file.name} mis*ing columns: {sorted(missing)}"
  *         )

        batch = batch[HISTORICAL_COLUMNS].copy()

       *batch["source_file"] = file.name

*       frames.append(batch)

    h*storical = pd.concat(
        fram*s,
        ignore_index=True
    )*
    for column in HISTORICAL_COLU*NS:
        historical[column] = n*rmalize_text(
            historic*l[column]
        )

    historica* = historical.dropna(
        subs*t=["meter_number"]
    )

    hist*rical = historical.rename(
        columns={
            "current_floc": "historical_floc"
        }
    )

    historical = historical.drop_duplicates()

    historical.to_parquet(
        HISTORICAL_FILE,
        index=False
    )

    print(
        f"Historical recommendation rows: {len(historical):,}"
    )

    print(
        f"Unique historical meters: "
        f"{historical['meter_number'].nunique():,}"
    )

    return historical


# ============================================================
# Load Current AMI
# ============================================================

def load_current_ami() -> pd.DataFrame:

    if not AMI_FILE.exists():
        raise FileNotFoundError(
            f"AMI cache not found: {AMI_FILE}\n"
            "Update AMI_FILE to your actual current AMI parquet path."
        )

    ami = pd.read_parquet(
        AMI_FILE
    )

    required = [
        METER_COL,
        DIVISION_COL,
        CURRENT_FLOC_COL
    ]

    missing = set(required) - set(ami.columns)

    if missing:
        raise ValueError(
            f"AMI cache missing columns: {sorted(missing)}"
        )

    ami = ami[required].copy()

    for column in required:
        ami[column] = normalize_text(
            ami[column]
        )

    ami = ami.dropna(
        subset=[METER_COL]
    )

    conflicts = (
        ami.groupby(METER_COL)
        .agg(
            division_count=(DIVISION_COL, "nunique"),
            floc_count=(CURRENT_FLOC_COL, "nunique")
        )
    )

    conflicts = conflicts[
        (conflicts["division_count"] > 1)
        | (conflicts["floc_count"] > 1)
    ]

    if not conflicts.empty:
        raise ValueError(
            f"AMI contains {len(conflicts):,} meters "
            "with conflicting current assignments or divisions."
        )

    ami = ami.drop_duplicates(
        subset=[METER_COL]
    )

    print(
        f"Current AMI meters: {len(ami):,}"
    )

    return ami


# ============================================================
# Reconcile Historical Recommendations
# ============================================================

def reconcile_recommendations(
    historical: pd.DataFrame,
    ami: pd.DataFrame
) -> pd.DataFrame:

    historical = historical.copy()

    historical["previously_flagged"] = True

    current = ami.rename(
        columns={
            CURRENT_FLOC_COL: "latest_floc",
            DIVISION_COL: "latest_division"
        }
    )

    matched = historical.merge(
        current,
        on=METER_COL,
        how="left",
        indicator=True,
        validate="many_to_one"
    )

    matched["historical_status"] = "DIFFERENT_ASSOCIATION"

    missing = matched["_merge"].eq("left_only")

    same_as_original = (
        matched["latest_floc"]
        .eq(matched["historical_floc"])
        .fillna(False)
    )

    same_as_recommended = (
        matched["latest_floc"]
        .eq(matched["final_recommended_floc"])
        .fillna(False)
    )

    matched.loc[
        missing,
        "historical_status"
    ] = "METER_NOT_IN_CURRENT_AMI"

    matched.loc[
        ~missing & matched["latest_floc"].isna(),
        "historical_status"
    ] = "CURRENT_FLOC_MISSING"

    matched.loc[
        same_as_original & ~missing,
        "historical_status"
    ] = "ORIGINAL_ASSOCIATION"

    matched.loc[
        same_as_recommended & ~missing,
        "historical_status"
    ] = "MATCHES_PREVIOUS_RECOMMENDATION"

    matched["division"] = matched["latest_division"]

    matched["run_date"] = RUN_DATE

    matched = matched.drop(
        columns=["_merge"]
    )

    return matched


# ============================================================
# Save Outputs
# ============================================================

def save_outputs(matched: pd.DataFrame) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    matched.to_parquet(
        OUTPUT_DIR / "all_historical_recommendations.parquet",
        index=False
    )

    for division, group in (
        matched
        .dropna(subset=["division"])
        .groupby("division")
    ):

        group.to_parquet(
            OUTPUT_DIR / f"{division}.parquet",
            index=False
        )

        print(
            f"[{division}] {len(group):,} historical rows"
        )

    missing = matched[
        matched["historical_status"]
        == "METER_NOT_IN_CURRENT_AMI"
    ]

    missing.to_parquet(
        OUTPUT_DIR / "meters_not_in_current_ami.parquet",
        index=False
    )

    summary = (
        matched.groupby(
            "historical_status",
            dropna=False
        )
        .agg(
            recommendation_rows=("meter_number", "size"),
            unique_meters=("meter_number", "nunique")
        )
        .reset_index()
    )

    summary.to_csv(
        OUTPUT_DIR / "summary.csv",
        index=False
    )

    print("\nHistorical reconciliation summary:")

    print(
        summary.to_string(index=False)
    )

    print(
        f"\nSaved to: {OUTPUT_DIR}"
    )


# ============================================================
# Main
# ============================================================

def main():

    historical = load_historical_recommendations()

    ami = load_current_ami()

    matched = reconcile_recommendations(
        historical,
        ami
    )

    save_outputs(matched)


if __name__ == "__main__":
    main()