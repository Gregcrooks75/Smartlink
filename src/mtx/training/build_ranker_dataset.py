from __future__ import annotations
import pandas as pd
def prepare_verified_labels(
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and normalize MAP-verified labels.
    """
    df = labels.copy()
    df = df.rename(
        columns={
            "MeterNumber":
                "meter_number",
            "Truth":
                "truth_transformer",
        }
    )
    required = {
        "meter_number",
        "truth_transformer",
    }
    missing = (
        required
        - set(df.columns)
    )
    if missing:
        raise ValueError(
            "Label data missing columns: "
            f"{sorted(missing)}"
        )
    df = df.dropna(
        subset=[
            "meter_number",
            "truth_transformer",
        ]
    )
    # --------------------------------------------------
    # ONE TRUTH PER METER
    # --------------------------------------------------
    truth_counts = (
        df
        .groupby(
            "meter_number"
        )[
            "truth_transformer"
        ]
        .nunique()
    )
    conflicts = (
        truth_counts[
            truth_counts > 1
        ]
    )
    if not conflicts.empty:
        raise ValueError(
            "Some meters contain multiple "
            "verified transformer Truth values. "
            f"Count={len(conflicts):,}"
        )
    df = (
        df
        .sort_values(
            "meter_number"
        )
        .drop_duplicates(
            "meter_number",
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )
    return df
def build_ranker_training_dataset(
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Join labels to candidate rows.
    Returns:
        usable_training_rows
        meters_whose_truth_is_not_in_candidate_set
    """
    labels = (
        prepare_verified_labels(
            labels
        )
    )
    label_columns = [
        "meter_number",
        "truth_transformer",
    ]
    for optional in [
        "Division",
        "MeterFLOC",
        "ValidationSource",
    ]:
        if optional in labels.columns:
            label_columns.append(
                optional
            )
    joined = (
        features
        .merge(
            labels[
                label_columns
            ],
            on="meter_number",
            how="inner",
            validate="many_to_one",
        )
    )
    joined[
        "label"
    ] = (
        joined[
            "candidate_transformer"
        ]
        .astype(str)
        ==
        joined[
            "truth_transformer"
        ]
        .astype(str)
    ).astype(int)
    positive_counts = (
        joined
        .groupby(
            "meter_number"
        )[
            "label"
        ]
        .sum()
    )
    usable_meters = (
        positive_counts[
            positive_counts
            == 1
        ]
        .index
    )
    missing_truth_meters = (
        positive_counts[
            positive_counts
            == 0
        ]
        .index
    )
    usable = (
        joined[
            joined[
                "meter_number"
            ]
            .isin(
                usable_meters
            )
        ]
        .copy()
    )
    missing_truth = (
        joined[
            joined[
                "meter_number"
            ]
            .isin(
                missing_truth_meters
            )
        ][
            [
                "meter_number",
                "truth_transformer",
            ]
        ]
        .drop_duplicates(
            "meter_number"
        )
        .reset_index(
            drop=True
        )
    )
    return (
        usable,
        missing_truth,
    )